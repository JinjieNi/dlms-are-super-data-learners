from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.geometric import Geometric

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor


def depthwise_routing(
        self, 
        hidden_states: Tensor, 
        router_layer: nn.Parameter,
        mask: Tensor = None
        ) -> Tuple[Tensor, Tensor]:
        """
        Depthwise routing for the transformer block.
        """
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states * mask
        router_layer = router_layer.to(torch.float32)
        router_logits = router_layer(hidden_states.to(torch.float32))
        router_logits = F.softmax(router_logits, dim=-1).to(original_dtype) # It's important to keep the fp32 for softmax input. See the Figure 15 of the switch transformer paper for details.
        router_decisions = torch.argmax(router_logits, dim=-1)
        router_decisions[mask == 0] = -100 # set the decisions for the masked tokens to -100
        
        return router_logits, router_decisions

def adacom_layer_forward(
        self,
        layer: TransformerLayer,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        
        # adacom specific args
        token_states_last: Tensor = None,
        router_logits_last: Tensor = None,
        l_no: int = None,
        num_layers: int = None,
        max_recurrent_steps: int = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform the adacom forward pass through the transformer layer.
        
        Args:
            layer: TransformerLayer: The transformer layer to be executed.
            hidden_states: Tensor: The input tensor of shape [s, b, h] where s is the sequence length, b is the batch size, and h is the hidden size.
            attention_mask: Tensor: Boolean tensor of shape [1, 1, s, s] for masking self-attention.
            context: Tensor: Context tensor for cross-attention.
            context_mask: Tensor: Mask for cross-attention context
            rotary_pos_emb: Tensor: Rotary positional embeddings.
            rotary_pos_cos: Tensor: Cosine part of the rotary positional embeddings.
            rotary_pos_sin: Tensor: Sine part of the rotary positional embeddings.
            attention_bias: Tensor: Bias tensor for Q * K.T of shape in shape broadcastable to [b, num_head, sq, skv], e.g. [1, 1, sq, skv]. Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_params: InferenceParams: Parameters for inference-time optimizations.
            packed_seq_params: PackedSeqParams: Parameters for packed sequence processing.
            
            token_states_last: Tensor: The states from the previous layer. Integer tensor of shape [s, b]. The states can be -2 (global recurrent), -1 (recurrent), 0 (forward), 1 (skip), 2 (model halted).
            router_logits_last: Tensor: The router logits from the previous layer. The logits are used to decide the tokens to forward or skip.
            l_no: int: The layer number.
            num_layers: int: The total number of layers.
            max_recurrent_steps: int: The maximum number of recurrent steps.
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]: The output hidden states tensor of shape [s, b, h], the updated context tensor if cross-attention is used, and the token states for the next layer.
        """
        
        # TODO big changes: now the last router is the global router; also token skip is brought forward to before the layer router
        
        layer_properties = ['normal layer', 'initial layer', 'last layer', 'global recurrented first layer']
        current_layer_property = layer_properties[0]
        l, b, h = hidden_states.size()
        max_recurrent_steps_ori = max_recurrent_steps
        
        
        # init router layer is only used in the first layer (including recurrence from the last layer)
        if self.router_layer_i is not None and l_no == 0 and token_states_last is None:
            current_layer_property = layer_properties[1]
            assert router_logits_last is None, "router_logits_last must be None when using the initial router layer"
            router_logits_i, router_decisions_i = self.depthwise_routing(hidden_states, self.router_layer_i)
            token_states_last = router_decisions_i # set the token states
            
            skip_mask_i = torch.ones(l, b, device=hidden_states.device)
            forward_mask_i = torch.ones(l, b, device=hidden_states.device)
            
            skip_mask_i[router_decisions_i != 1] = 0
            forward_mask_i[router_decisions_i != 0] = 0
            
        # If the token_states_last does not exist, forward all tokens into the layer
        if token_states_last is None:
            current_layer_property = layer_properties[1]
            token_states_last = torch.zeros(l, b, device=hidden_states.device)
        
        # Process the token masks and states based on the token states from the last layer
        if -2 in token_states_last: # if the last router is from the global router
            current_layer_property = layer_properties[3]
            assert l_no == 0, "The global router logits can only be applied in the first layer"
            assert 0 not in token_states_last and 2 not in token_states_last, "The tokens routed from the global router must be either global recurrent, model halted, or skipped to the next layer"
            recurrent_mask_g = torch.ones(l, b, device=hidden_states.device)
            recurrent_mask_g[token_states_last != -2] = 0
            skip_mask_g = torch.ones(l, b, device=hidden_states.device)
            skip_mask_g[token_states_last != 1] = 0
            # set token state from global recurrence to forward
            token_states_last[token_states_last == -2] = 0
        else: # if the last router is from the last layer router
            forward_mask = torch.ones(l, b, device=hidden_states.device)
            forward_mask[token_states_last != 0] = 0
            skip_mask = torch.ones(l, b, device=hidden_states.device)
            skip_mask[token_states_last != 1] = 0
            # if some tokens were still recurrenting, but the max_recurrent_steps is reached, set them to forward
            token_states_last[token_states_last == -1] = 0
        
        # Make sure we only have 0, 1, 2 in the token_states_last before sending it to the loop
        assert -2 not in token_states_last, "The token_states_last must not contain global recurrent states"
        assert -1 not in token_states_last, "The token_states_last must not contain recurrent states" # this state only happens in the loop
        
        while max_recurrent_steps > -1:
            
            # TODO only pass the non-skip and non-model-halted states to the layer
            # TODO manage the kv cache for inference
        
            hidden_states, context = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
            )
            
            if self.router_layers[l_no] is None:
                break
            
            # TODO last layer router is the global router, having four dimensions. Implement the last layer part.
            
            # apply the router logits of the last loop to the layer output to make the gradient flow
            if max_recurrent_steps_ori > max_recurrent_steps:
                hidden_states = hidden_states * (1-recurrent_mask) + hidden_states * (router_logits[:, :, 0].squeeze(-1) * recurrent_mask)
                
            # block router, the first loop only route forward tokens, the other loops only route recurrent tokens
            router_mask = torch.ones(l, b, device=hidden_states.device)
            if max_recurrent_steps_ori == max_recurrent_steps:
                router_mask[token_states_last == 0] = 0
                router_logits, router_decisions = self.depthwise_routing(hidden_states, self.router_layers[l_no], router_mask)
            else:
                router_mask[token_states_last == -1] = 0
                router_logits, router_decisions = self.depthwise_routing(hidden_states, self.router_layers[l_no], router_mask)
            # set the token states
            token_states_last[router_decisions == 0] = -1
            token_states_last[router_decisions == 1] = 0
            token_states_last[router_decisions == 2] = 1
            # set the masks
            recurrent_mask = torch.ones(l, b, device=hidden_states.device)
            recurrent_mask[router_decisions != 0] = 0  # here the router_decisions != 0 tokens is a subset of the router mask
            
            # if all tokens are layer halted, break the loop
            if torch.all(token_states_last != -1): 
                break
            
            max_recurrent_steps -= 1
            
        # Apply the logits for tokens decided to forward (by the previous layer) to the layer output to make the gradient flow
        
        
        # Apply the logits for tokens decided to skip (by the previous previous layer) to the layer output to make the gradient flow. Note that it's not the directly previous layer
        
        
        return hidden_states, context, token_states_last, router_logits

def transformer_block_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_params (InferenceParams, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        
        # from megatron.training.global_vars import get_timers
        # timers = get_timers()
        
        # timers('overall', log_level=0).start(barrier=True)
        

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=self.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=self.tp_only_amax_red
                )
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        with rng_context, fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
            else:
                # timers('vanilla transformer forward [forward through layers]', log_level=0).start()
                for l_no, layer in enumerate(self.layers):
                    with self.offload_context:
                        layer.use_cudagraph = True
                        if (len(self.cuda_graphs) == 0) or (not self.training):
                            hidden_states, context = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                context=context,
                                context_mask=context_mask,
                                rotary_pos_emb=rotary_pos_emb,
                                rotary_pos_cos=rotary_pos_cos,
                                rotary_pos_sin=rotary_pos_sin,
                                attention_bias=attention_bias,
                                inference_params=inference_params,
                                packed_seq_params=packed_seq_params,
                        )
                        else:
                            # TODO - Implement this method
                            raise NotImplementedError
                            # CUDA graph replay for layer `l_no` and microbatch
                            # `self.current_microbatch`. TransformerEngine versions>=1.10
                            # allow keyword arguments with CUDA graph. However, CUDA graph
                            # acccepts only Tensor inputs and Tensor outputs. Hence,
                            # `inference_params` and `packed_seq_params` are excluded from
                            # input list while output is limited to `hidden_states`.
                            cg_index = self.current_microbatch % len(self.cuda_graphs[l_no])
                            assert not any(
                                [inference_params, packed_seq_params]
                            ), "CUDA graph accepts only Tensor inputs."
                            optional_inputs = self.get_cuda_graph_optional_args(
                                attention_mask,
                                context,
                                context_mask,
                                rotary_pos_emb,
                                attention_bias,
                                inference_params,
                                packed_seq_params,
                            )
                            hidden_states = self.cuda_graphs[l_no][cg_index](
                                hidden_states, **optional_inputs
                            )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # timers('vanilla transformer forward [forward through layers]').stop()     
        
        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )
            
        # timers('overall').stop()
        # all_time_elapsed = timers('overall').active_time()
        # print(
        #     f"overall time: {timers('overall').active_time()}, ratio: {timers('overall').active_time() / all_time_elapsed * 100:.2f}%\n"
        #     f"vanilla transformer forward [forward through layers]: {timers('vanilla transformer forward [forward through layers]').active_time()}, ratio: {timers('vanilla transformer forward [forward through layers]').active_time() / all_time_elapsed * 100:.2f}%\n"
        #     )

        return hidden_states

#######################################
# universal transformer - act original
#######################################

def act_halt_function(self, state):
    original_type = state.dtype
    ut_router_layer_norm = self.ut_router_layer_norm
    ut_router_layer = self.ut_router_layer
    ut_router_activation = self.ut_router_activation
    
    if not self.router_precision_converted:
        ut_router_layer_norm.to(torch.float32)
        ut_router_layer.to(torch.float32)
        ut_router_activation.to(torch.float32)
        self.router_precision_converted = True
    
    p = ut_router_activation(
        ut_router_layer(
            ut_router_layer_norm(state.to(torch.float32))
            )[0] # using ColumnParallelLinear, so we need to take the first element
        ).squeeze(-1)
    return p.to(original_type)
    
def should_continue(halting_probability, n_updates, threshold, act_max_steps):
    condition = torch.logical_and(
        halting_probability < threshold,
        n_updates < act_max_steps
    )
    # Return whether any of the condition is True
    return torch.any(condition)

# def retrieve_states(source_states: Tensor, halted_states: List, mask: Tensor) -> Tensor:
#     """
#     A function to update states based on the mask from source_states and halted_states.
#     """
#     s, b, _ = source_states.shape
#     assert len(halted_states) == s
#     assert len(halted_states[0]) == b
#     for i in range(s):
#         for j in range(b):
#             if not mask[i, j, 0]:
#                 source_states[i, j, :] = halted_states[i][j]
#     return source_states

# def update_halted_states(source_states: Tensor, halted_states: List, mask: Tensor) -> List:
#     """
#     A function to update the halted_states based on the mask from source_states.
#     """
#     s, b, _ = source_states.shape
#     assert len(halted_states) == s
#     assert len(halted_states[0]) == b
#     for i in range(s):
#         for j in range(b):
#             if mask[i, j, 0]:
#                 assert halted_states[i][j] is None, "The halted_states should be None before updating."
#                 halted_states[i][j] = source_states[i, j, :]
#     return halted_states

def _ut_act_transformer_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        n_updates: Tensor = None,
        step: Tensor = None,
        halted_k_cache: List[Tensor] = None,
        halted_v_cache: List[Tensor] = None,
        halt_mask: Tensor = None,
        ):
    
    s, b, _ = hidden_states.shape
    
    if self.config.act_fake_halting:
        position_ids_2 = step.unsqueeze(0).unsqueeze(0).expand(s, b) # [s, b]
    else:
        position_ids_2 = n_updates - 1 # [s, b]; -1 to make it 0-indexed
    
    # Compute the 2D RoPE cos and sin to be passed to each attention layer
    if self.config.position_embedding_type == 'rope_2d_position_step':
        
        if inference_params is not None:
            inf_max_seq_length = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            
            position_ids_1 = torch.arange(inf_max_seq_length, device=hidden_states.device).unsqueeze(0).expand(b, -1) # [b, inf_max_seq_length]
            
            # pre-allocate the memory for the 2nd axis of the RoPE. We use a cache mechanism to trace the n_updates for the past kv cache at the inference time.
            if not hasattr(inference_params, 'rope2d_nupdates_memory_dict'):
                inference_params.rope2d_nupdates_memory_dict = {}
                rope2d_nupdates_memory = torch.empty(
                    inf_max_seq_length, 
                    inf_max_batch_size, 
                    dtype=torch.int32, 
                    device=hidden_states.device
                    ) # [inf_max_seq_length, inf_max_batch_size]
                inference_params.rope2d_nupdates_memory_dict['rope2d_nupdates_memory_alllayers'] = rope2d_nupdates_memory
            else:
                rope2d_nupdates_memory = inference_params.rope2d_nupdates_memory_dict['rope2d_nupdates_memory_alllayers']
                
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + s
            assert sequence_end <= rope2d_nupdates_memory.size(0)
            assert inf_max_batch_size == b
            
            rope2d_nupdates_memory[sequence_start:sequence_end,:] = position_ids_2
            
            # extend the position_ids_2 to inf_max_seq_length
            position_ids_2 = rope2d_nupdates_memory[:inf_max_seq_length, :].transpose(0, 1) # [b, inf_max_seq_length]
        else:
            position_ids_1 = torch.arange(s, device=hidden_states.device).unsqueeze(0).expand(b, -1) # [b, s]
            position_ids_2 = position_ids_2.transpose(0, 1) # [b, s]
        
        position_ids = torch.stack([position_ids_1, position_ids_2], dim=0) # [2, b, s] / [2, b, inf_max_seq_length]
        rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb(hidden_states, position_ids) # [2, s, b, h] / [2, inf_max_seq_length, b, h]
    elif self.config.position_embedding_type == 'rope_position_learned_step':
        assert rotary_pos_emb is not None
        learned_abs_pos_emb = self.learned_absolute_step_embedding(position_ids_2) # [s, b, h]
        hidden_states = hidden_states + learned_abs_pos_emb # [s, b, h]
    elif self.config.position_embedding_type == 'rope_position_sinusoid_step':
        hidden_states = self.sinusoid_step_embedding(hidden_states, position_ids_2) # [s, b, h]
        
    # Forward pass through the transformer layers
    if self.config.recompute_granularity == 'full' and self.training:
                raise NotImplementedError
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
    else:
        for l_no, layer in enumerate(self.layers):
            with self.offload_context:
                layer.use_cudagraph = True
                if (len(self.cuda_graphs) == 0) or (not self.training):
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                        halted_k_cache=halted_k_cache,
                        halted_v_cache=halted_v_cache,
                        halt_mask=halt_mask,
                        step=step,
                )
                else:
                    raise NotImplementedError
                    # CUDA graph replay for layer `l_no` and microbatch
                    # `self.current_microbatch`. TransformerEngine versions>=1.10
                    # allow keyword arguments with CUDA graph. However, CUDA graph
                    # acccepts only Tensor inputs and Tensor outputs. Hence,
                    # `inference_params` and `packed_seq_params` are excluded from
                    # input list while output is limited to `hidden_states`.
                    cg_index = self.current_microbatch % len(self.cuda_graphs[l_no])
                    assert not any(
                        [inference_params, packed_seq_params]
                    ), "CUDA graph accepts only Tensor inputs."
                    optional_inputs = self.get_cuda_graph_optional_args(
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        attention_bias,
                        inference_params,
                        packed_seq_params,
                    )
                    hidden_states = self.cuda_graphs[l_no][cg_index](
                        hidden_states, **optional_inputs
                    )

            if (
                torch.is_grad_enabled()
                and self.config.cpu_offloading
                and self.group_prefetch_offload_commit_async is not None
            ):
                hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

    return hidden_states, context
    
def ut_act_forward_original(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    
        """
        Perform the forward pass through the universal transformer block.

        ACT was adapted from the original tensorflow implementation https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/models/research/universal_transformer_util.py#L1023
        """
        # TODO there might be vanishing gradient problem, deal with it. ('Nan')
        
        # initialize the act states
        state = hidden_states
        halted_states = None
        halted_k_cache = None
        halted_v_cache = None
        act_max_steps = self.config.act_max_steps
        threshold = 1.0 - self.config.act_epsilon
        
        halting_probability = torch.zeros(state.size(0), state.size(1), device=state.device, dtype=state.dtype) # [s, b]
        remainders = torch.zeros(state.size(0), state.size(1), device=state.device, dtype=state.dtype) # [s, b]
        n_updates = torch.zeros(state.size(0), state.size(1), device=state.device, dtype=torch.int32) # [s, b]
        previous_state = torch.zeros_like(state, device=state.device, dtype=state.dtype) # [s, b, h]
        step = torch.tensor(0, device=state.device, dtype=torch.int32) # it only works for global input step encoding
            
        
        # In the original UT implementation, the halted states are actually attended in the later forward passes, so-called fake halting.
        # In addition, the original UT implementation does not cache kv, meaning that the auto-regresive decoding will be different from the vanilla transformer: each time it will parallelly forward the decoded tokens and new tokens. The halting decision and the forwarding pass of the later position in decoding will be dependent on a pre-mature kv of the previous states instead of the final ones. 
        
        def act_function(state, step, halting_probability, remainders, n_updates,
                previous_state, halted_states=None, halted_k_cache=None, halted_v_cache=None):
            """act (position-wise halting) with cache and true halting.
            Every time the act halt function predicts the halt

            Args:
            state: 3-D Tensor: [batch_size, length, channel]
            step: indicates number of steps taken so far
            halting_probability: halting probability
            remainders: act remainders
            n_updates: act n_updates
            previous_state: previous state

            Returns:
            transformed_state: transformed state
            step: step+1
            halting_probability: halting probability
            remainders: act remainders
            n_updates: act n_updates
            new_state: new state
            """
            
            p = act_halt_function(self, state) # the halt function prediction should be done in float32 to increase stability
            
            # Mask for inputs which have not halted yet
            still_running = halting_probability < 1.0

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > threshold) * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= threshold) * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability += p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders += new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability += new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates += still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = (p * still_running + remainders * new_halted).unsqueeze(-1) # shape: [s, b, 1]
            
            # apply transformation on the state
            
            if not self.config.act_fake_halting:
                # Initialize halted states, and halted kv cache to reuse the activations of the halted positions
                # if halted_states is None:
                #     halted_states = [[None for _ in range(state.size(1))] for _ in range(state.size(0))]
                if halted_k_cache is None:
                    halted_k_cache = [None] * len(self.layers)
                if halted_v_cache is None:
                    halted_v_cache = [None] * len(self.layers)
                
                # Only transform non-halted states, this include the newly halted states
                newly_halted = new_halted.unsqueeze(-1) # [s, b, 1]
                active_mask = still_running.unsqueeze(-1) + newly_halted # [s, b, 1]
                halt_mask = ~active_mask # [s, b, 1]
                
                halted_states = state
                transformed_state = state * active_mask # [s, b, h]
            else: 
                transformed_state = state
                newly_halted = None
                halt_mask = None
                
            
            transformed_state, _ = _ut_act_transformer_forward( # TODO Current implementation is not saving compuation/memory. Implement the compuation/memory or memory efficient version, though it could be a bit tricky due to the fact that the active tokens in each sequence are not the same.
                self,
                hidden_states=transformed_state,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                n_updates=n_updates,
                step=step,
                halted_k_cache=halted_k_cache,
                halted_v_cache=halted_v_cache,
                halt_mask=halt_mask,
            )
            
            if not self.config.act_fake_halting:
                transformed_state = torch.where(active_mask, transformed_state, halted_states)

            # update running part in the weighted state and keep the rest
            if self.config.act_type == "basic":
                new_state = ((transformed_state * update_weights) +
                            (previous_state * (1 - update_weights))) # Every step the state is normalize to 1. However, it will decay the previous states (give preference to the latest ones, which might be better). The halted states being intact. 
            elif self.config.act_type == "accumulated":
                new_state = (transformed_state * update_weights) + previous_state # it won't decay the previous states. the author actually used the basic one as he claims it performed better in his experiments.
            elif self.config.act_type == "latest":
                new_state = transformed_state * update_weights # apply the predicted logits to make the halting function differentiable
            else:
                raise ValueError(f"Invalid act type: {self.config.act_type}")
            
            
            return (transformed_state, step+1, halting_probability, 
                    remainders, n_updates, new_state, 
                    halted_states, halted_k_cache, halted_v_cache)
        
        
        
        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=self.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=self.tp_only_amax_red
                )
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        with rng_context, fp8_context:
            # Forward pass.
            while should_continue(halting_probability, n_updates, threshold, act_max_steps):
                state, step, halting_probability, remainders, n_updates, previous_state, halted_states, halted_k_cache, halted_v_cache = act_function(
                    state, step, halting_probability, remainders, n_updates, previous_state, halted_states, halted_k_cache, halted_v_cache)
        
        hidden_states = previous_state
        
        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        ponder_times = n_updates
        remainders = remainders
        
        return hidden_states, (ponder_times, remainders)


#######################################
# universal transformer - act SUT
#######################################
def act_sut_halt_function(self, state):
    original_type = state.dtype
    ut_router_layer_norm = self.ut_router_layer_norm
    ut_router_layer = self.ut_router_layer
    ut_router_activation = self.ut_router_activation
    
    if not self.router_precision_converted:
        ut_router_layer_norm.to(torch.float32)
        ut_router_layer.to(torch.float32)
        ut_router_activation.to(torch.float32)
        self.router_precision_converted = True
    
    p = ut_router_activation(
        ut_router_layer(
            ut_router_layer_norm(state.to(torch.float32))
            )[0] # using ColumnParallelLinear, so we need to take the first element
        ).squeeze(-1)
    return p.to(original_type)

# def compute_act_loss_sut(history_logits, act_loss_type, act_loss_factor):
#     k = act_loss_factor
#     act_loss = torch.tensor(0.0, device=history_logits[0][0][0].device)
#     s, b = len(history_logits), len(history_logits[0])
#     if act_loss_type == "linear":
#         for i in range(s):
#             for j in range(b):
#                 # todo: write a new loop to reduce memory usage
#                 act_loss += sum([logit * (step+1) for step, logit in enumerate(history_logits[i][j])])
#         act_loss /= s * b
#     elif act_loss_type == "kl_geo":
#         for i in range(s):
#             for j in range(b):
#                 # normalize the logits to make them sum to 1
#                 logits = torch.stack(history_logits[i][j])
#                 logits = (logits / logits.sum()).log()
#                 # todo: write a new loop to reduce memory usage
#                 prior = torch.tensor([k*((1-k)**(m-1)) for m in range(1, len(logits)+1)], device=logits.device, dtype=logits.dtype)
#                 prior = prior / prior.sum()
#                 act_loss += F.kl_div(logits, prior, reduction='batchmean')
#         act_loss /= s * b
#     else:
#         raise ValueError(f"Invalid act loss type: {act_loss_type}")
#     return act_loss

def _sut_act_transformer_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        key_value_states: Tensor = None,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        n_updates: Tensor = None,
        step: Tensor = None,
        halted_k_cache: List[Tensor] = None,
        halted_v_cache: List[Tensor] = None,
        halt_mask: Tensor = None,
        ):

    s, b, _ = hidden_states.shape
    
    if self.config.act_fake_halting:
        position_ids_2 = step.unsqueeze(0).unsqueeze(0).expand(s, b) # [s, b]
    else:
        position_ids_2 = n_updates - 1 # [s, b]; -1 to make it 0-indexed
    
    # compute the horizontal and vertical position embedding
    if self.config.position_embedding_type == 'rope_2d_position_step':
        # Compute the 2D RoPE cos and sin to be passed to each attention layer
        if inference_params is not None:
            inf_max_seq_length = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            
            position_ids_1 = torch.arange(inf_max_seq_length, device=hidden_states.device).unsqueeze(0).expand(b, -1) # [b, inf_max_seq_length]
            
            # pre-allocate the memory for the 2nd axis of the RoPE. We use a cache mechanism to trace the n_updates for the past kv cache at the inference time.
            if not hasattr(inference_params, 'rope2d_nupdates_memory_dict'):
                inference_params.rope2d_nupdates_memory_dict = {}
                rope2d_nupdates_memory = torch.empty(
                    inf_max_seq_length, 
                    inf_max_batch_size, 
                    dtype=torch.int32, 
                    device=hidden_states.device
                    ) # [inf_max_seq_length, inf_max_batch_size]
                inference_params.rope2d_nupdates_memory_dict['rope2d_nupdates_memory_alllayers'] = rope2d_nupdates_memory
            else:
                rope2d_nupdates_memory = inference_params.rope2d_nupdates_memory_dict['rope2d_nupdates_memory_alllayers']
                
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + s
            assert sequence_end <= rope2d_nupdates_memory.size(0)
            assert inf_max_batch_size == b
            
            rope2d_nupdates_memory[sequence_start:sequence_end,:] = position_ids_2
            
            # extend the position_ids_2 to inf_max_seq_length
            position_ids_2 = rope2d_nupdates_memory[:inf_max_seq_length, :].transpose(0, 1) # [b, inf_max_seq_length]
        else:
            position_ids_1 = torch.arange(s, device=hidden_states.device).unsqueeze(0).expand(b, -1) # [b, s]
            position_ids_2 = position_ids_2.transpose(0, 1) # [b, s]
        
        position_ids = torch.stack([position_ids_1, position_ids_2], dim=0) # [2, b, s] / [2, b, inf_max_seq_length]
        rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb(hidden_states, position_ids) # [2, s, b, h] / [2, inf_max_seq_length, b, h]
    elif self.config.position_embedding_type == 'rope_position_learned_step':
        assert rotary_pos_emb is not None
        learned_abs_pos_emb = self.learned_absolute_step_embedding(position_ids_2) # [s, b, h]
        hidden_states = hidden_states + learned_abs_pos_emb # [s, b, h]
    elif self.config.position_embedding_type == 'rope_position_sinusoid_step':
        hidden_states = self.sinusoid_step_embedding(hidden_states, position_ids_2) # [s, b, h]
        
    # Forward pass through the transformer layers
    if self.config.recompute_granularity == 'full' and self.training:
                raise NotImplementedError
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
    else:
        for l_no, layer in enumerate(self.layers):
            with self.offload_context:
                layer.use_cudagraph = True
                if (len(self.cuda_graphs) == 0) or (not self.training):
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        key_value_states=key_value_states,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                        halted_k_cache=halted_k_cache,
                        halted_v_cache=halted_v_cache,
                        halt_mask=halt_mask,
                        step=step,
                )
                else:
                    raise NotImplementedError
                    # CUDA graph replay for layer `l_no` and microbatch
                    # `self.current_microbatch`. TransformerEngine versions>=1.10
                    # allow keyword arguments with CUDA graph. However, CUDA graph
                    # acccepts only Tensor inputs and Tensor outputs. Hence,
                    # `inference_params` and `packed_seq_params` are excluded from
                    # input list while output is limited to `hidden_states`.
                    cg_index = self.current_microbatch % len(self.cuda_graphs[l_no])
                    assert not any(
                        [inference_params, packed_seq_params]
                    ), "CUDA graph accepts only Tensor inputs."
                    optional_inputs = self.get_cuda_graph_optional_args(
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        attention_bias,
                        inference_params,
                        packed_seq_params,
                    )
                    hidden_states = self.cuda_graphs[l_no][cg_index](
                        hidden_states, **optional_inputs
                    )

            if (
                torch.is_grad_enabled()
                and self.config.cpu_offloading
                and self.group_prefetch_offload_commit_async is not None
            ):
                hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

    return hidden_states, context

def sut_act_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    
        """
        Perform the forward pass through the universal transformer block (SUT halting: https://arxiv.org/abs/2310.07096).
        """
        
        # initialize the act states
        state = hidden_states
        halted_k_cache = None
        halted_v_cache = None
        
        act_max_steps = self.config.act_max_steps
        threshold = 1.0 - self.config.act_epsilon
        act_loss = torch.tensor(0.0, dtype=state.dtype, device=state.device)
        
        halting_probability = torch.zeros(state.size(0), state.size(1), device=state.device, dtype=state.dtype) # [s, b]
        accumulated_nonhalting_probability = torch.ones(state.size(0), state.size(1), device=state.device, dtype=state.dtype) # [s, b]
        accumulated_states = torch.zeros_like(state, device=state.device, dtype=state.dtype) # [s, b, h]
        
        n_updates = torch.zeros(state.size(0), state.size(1), device=state.device, dtype=torch.int32) # [s, b]
        previous_state = state # [s, b, h]
        step = torch.tensor(0, device=state.device, dtype=torch.int32) # it only works for global input step encoding
        
        act_loss = torch.tensor(0.0, device=state.device, dtype=state.dtype)
        if self.config.act_loss_type == "kl_geo":
            history_logits = torch.zeros(state.size(0), state.size(1), act_max_steps, device=state.device, dtype=state.dtype)
            geom = Geometric(probs=torch.tensor([self.config.act_loss_factor]))
            k_value = torch.arange(1, act_max_steps+1, dtype=state.dtype)
            pmf = geom.log_prob(k_value)
            pmf /= pmf.sum()
            prior = pmf[None, None, :].expand(state.size(0), state.size(1), -1)
            
        else:
            history_logits = None
        
        # history_logits = [[[] for _ in range(state.size(1))] for _ in range(state.size(0))] # [s, b]
        
        def act_function(
            state, 
            step, 
            halting_probability, 
            accumulated_nonhalting_probability,
            accumulated_states,
            n_updates,
            previous_state, 
            halted_k_cache=None, 
            halted_v_cache=None,
            act_loss=None,
            history_logits=None
            ):
            """act (position-wise halting) with cache and true halting.
            Every time the act halt function predicts the halt

            Args:
            state: 3-D Tensor: [batch_size, length, channel]
            step: indicates number of steps taken so far
            halting_probability: halting probability
            accumulated_nonhalting_probability: continuous multiplication of the 1-p0
            n_updates: act n_updates
            previous_state: previous state

            Returns:
            transformed_state: transformed state
            step: step+1
            halting_probability: halting probability
            n_updates: act n_updates
            new_state: new state
            """
            
            p0 = act_sut_halt_function(self, state) # the halt function prediction should be done in float32 to increase stability
            
            # compute the real probability of halting at the current step, conditioned on the fact that the previous steps have not halted
            p = accumulated_nonhalting_probability * p0
            
            # update the accumulated_nonhalting_probability
            accumulated_nonhalting_probability = accumulated_nonhalting_probability * (1 - p0)
            
            with torch.no_grad():
                # Mask for inputs which have not halted yet
                still_running = halting_probability <= threshold

                # Mask of inputs which halted at this step
                new_halted = (halting_probability + p * still_running > threshold) * still_running

                # Mask of inputs which haven't halted, and didn't halt this step
                still_running = (halting_probability + p * still_running <= threshold) * still_running
                
                _active_mask = still_running + new_halted
                
                active_mask = _active_mask.unsqueeze(-1)
                
                # Increment n_updates for all inputs which are still running
                n_updates += _active_mask
            
            p *= _active_mask
            
            # Add the halting probability for active positions at present
            halting_probability += p

            # update the act loss
            if self.config.act_loss_type == "linear":
                act_loss += p.mean() * (step + 1)
            elif self.config.act_loss_type == "kl_geo":
                history_logits[:, :, step] = p
            else:
                raise ValueError(f"Invalid act loss type: {self.config.act_loss_type}")
            
            # update the accumulated states
            state_to_add = p.unsqueeze(-1) * state
            accumulated_states += state_to_add

            if not self.config.act_fake_halting:
                # Initialize halted states, and halted kv cache to reuse the activations of the halted positions
                if halted_k_cache is None:
                    halted_k_cache = [None] * len(self.layers)
                if halted_v_cache is None:
                    halted_v_cache = [None] * len(self.layers)
                
                # Only transform non-halted states, this include the newly halted states
                halt_mask = ~active_mask # [s, b, 1]
                halted_states = state
                transformed_state = state * active_mask # [s, b, h]
            else:
                transformed_state = state
                halt_mask = None
                
            # apply transformation on the state
            transformed_state, _ = _sut_act_transformer_forward( # TODO Current implementation is not saving compuation/memory. Implement the compuation/memory or memory efficient version, though it could be a bit tricky due to the fact that the active tokens for each sequence of the batchare not the same.
                self,
                hidden_states=transformed_state,
                attention_mask=attention_mask,
                key_value_states=previous_state,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                n_updates=n_updates,
                step=step,
                halted_k_cache=halted_k_cache,
                halted_v_cache=halted_v_cache,
                halt_mask=halt_mask,
            )
            
            if not self.config.act_fake_halting:
                transformed_state = torch.where(active_mask, transformed_state, halted_states)

            # update the expected halted states
            # if a position is already halted last step, its previous state will be updated to the new state directly
            # for a newly halted position, the transformed state will only constitute a very small part of the new state
            neg_halt = (1-halting_probability).unsqueeze(-1)
            state_to_update = neg_halt * transformed_state
            state_to_update += accumulated_states
            new_state = torch.where(
                (_active_mask).unsqueeze(-1), 
                state_to_update, 
                previous_state
                ) # [s, b, h]
            
            
            return (
                transformed_state, 
                step+1, 
                halting_probability, 
                accumulated_nonhalting_probability,
                accumulated_states,
                n_updates, 
                new_state, 
                halted_k_cache, 
                halted_v_cache,
                act_loss,
                history_logits
                )
        
        
        
        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=self.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=self.tp_only_amax_red
                )
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        with rng_context, fp8_context:
            # Forward pass.
            while should_continue(
                halting_probability, 
                n_updates, 
                threshold, 
                act_max_steps
                ):
                (
                    state, 
                    step, 
                    halting_probability, 
                    accumulated_nonhalting_probability, 
                    accumulated_states,
                    n_updates,
                    previous_state, 
                    halted_k_cache, 
                    halted_v_cache,
                    act_loss,
                    history_logits
                ) = act_function(
                    state, 
                    step, 
                    halting_probability, 
                    accumulated_nonhalting_probability, 
                    accumulated_states,
                    n_updates,
                    previous_state, 
                    halted_k_cache, 
                    halted_v_cache,
                    act_loss,
                    history_logits
                )
        
        hidden_states = state # different from the original ut, sut uses the final transformed state instead of the act state as the output, since it has already attended to the act states in the attention. See the sut code: https://github.com/shawntan/SUT/blob/main/halting.py
        
        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )
            
        # act_loss = compute_act_loss_sut(history_logits, self.config.act_loss_type, self.config.act_loss_factor)
        if self.config.act_loss_type == "kl_geo":
            history_logits /= torch.sum(history_logits, dim=-1, keepdim=True)
            history_logits = torch.log(history_logits)
            act_loss = F.kl_div(history_logits, prior, log_target=True)
        elif self.config.act_loss_type != "linear":
            raise ValueError(f"Invalid act loss type: {self.config.act_loss_type}")

        return hidden_states, act_loss, n_updates


#######################################
# universal transformer - router (post-routing)
#######################################
def should_continue_router(still_running, n_updates, act_max_steps):
    condition = torch.logical_and(
        still_running,
        n_updates < act_max_steps
    )
    # Return whether any of the condition is True
    return torch.any(condition)
    
def router_function(self, state):
    original_type = state.dtype
    ut_router_layer_norm = self.ut_router_layer_norm
    ut_router_layer = self.ut_router_layer
    ut_router_activation = self.ut_router_activation
    
    if not self.router_precision_converted:
        ut_router_layer_norm.to(torch.float32)
        ut_router_layer.to(torch.float32)
        ut_router_activation.to(torch.float32)
        self.router_precision_converted = True
    
    p = ut_router_activation(
        ut_router_layer(
            ut_router_layer_norm(state.to(torch.float32))
            )[0] # using ColumnParallelLinear, so we need to take the first element
        ).squeeze(-1)
    return p.to(original_type)

def aggregate_states(history_states, history_logits, softmax=False):
    # history_states: list, [s, b, inconsistent, h]
    # history_logits: list, [s, b, inconsistent]
    
    for i in range(len(history_states)):
        for j in range(len(history_states[i])):
            if softmax:
                softmax_logits = F.softmax(torch.stack(history_logits[i][j]), dim=0)
                history_states[i][j] = torch.einsum('bs, bsh -> bsh', softmax_logits, torch.stack(history_states[i][j]))
            else:
                history_states[i][j] = torch.einsum('bs, bsh -> bsh', torch.stack(history_logits[i][j]), torch.stack(history_states[i][j]))
    
    return history_states

def compute_act_loss_router(history_logits, act_loss_type, act_loss_factor):
    # k = act_loss_factor
    device = history_logits[0][0][0].device
    dtype = history_logits[0][0][0].dtype

    act_loss = torch.tensor(0.0, device=device, dtype=dtype)
    
    # The logits predicted by each step is independent, so just simply sum them up to add a penalty for further pondering.
    if act_loss_type == "sum":
        for blogits in history_logits:
            block_sum = torch.zeros((), device=device, dtype=dtype)
            for logits in blogits:
                block_sum += sum(logits)/len(logits)
            block_sum /= float(len(blogits))
            act_loss += block_sum
        act_loss /= len(history_logits)
        act_loss = -act_loss
    else:
        raise ValueError(f"Invalid act loss type: {act_loss_type} for router-like UT")
    
    return act_loss

def _ut_router_transformer_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        n_updates: Tensor = None,
        step: Tensor = None,
        halted_k_cache: List[Tensor] = None,
        halted_v_cache: List[Tensor] = None,
        halt_mask: Tensor = None,
        ):

    s, b, _ = hidden_states.shape
    
    if self.config.act_fake_halting:
        position_ids_2 = step.unsqueeze(0).unsqueeze(0).expand(s, b) # [s, b]
    else:
        position_ids_2 = n_updates - 1 # [s, b]; -1 to make it 0-indexed
    
    # Compute the 2D RoPE cos and sin to be passed to each attention layer
    if self.config.position_embedding_type == 'rope_2d_position_step':
        # the n_update will change every forward pass, so we need to re-compute the 2D RoPE cos and sin every forward pass.
        # the inference logic will need all past kv cache, so we need to cache the n_update for the past kv cache at the inference time and return all the rope cos and sin.
        if inference_params is not None:
            inf_max_seq_length = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            
            position_ids_1 = torch.arange(inf_max_seq_length, device=hidden_states.device).unsqueeze(0).expand(b, -1) # [b, inf_max_seq_length]
            
            # pre-allocate the memory for the 2nd axis of the RoPE. We use a cache mechanism to trace the n_updates for the past kv cache at the inference time.
            if not hasattr(inference_params, 'rope2d_nupdates_memory_dict'):
                inference_params.rope2d_nupdates_memory_dict = {}
                rope2d_nupdates_memory = torch.empty(
                    inf_max_seq_length, 
                    inf_max_batch_size, 
                    dtype=torch.int32, 
                    device=hidden_states.device
                    ) # [inf_max_seq_length, inf_max_batch_size]
                inference_params.rope2d_nupdates_memory_dict['rope2d_nupdates_memory_alllayers'] = rope2d_nupdates_memory
            else:
                rope2d_nupdates_memory = inference_params.rope2d_nupdates_memory_dict['rope2d_nupdates_memory_alllayers']
                
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + s
            assert sequence_end <= rope2d_nupdates_memory.size(0)
            assert inf_max_batch_size == b
            
            rope2d_nupdates_memory[sequence_start:sequence_end,:] = position_ids_2
            
            # extend the position_ids_2 to inf_max_seq_length
            position_ids_2 = rope2d_nupdates_memory[:inf_max_seq_length, :].transpose(0, 1) # [b, inf_max_seq_length]
        else:
            position_ids_1 = torch.arange(s, device=hidden_states.device).unsqueeze(0).expand(b, -1) # [b, s]
            position_ids_2 = position_ids_2.transpose(0, 1) # [b, s]
        
        position_ids = torch.stack([position_ids_1, position_ids_2], dim=0) # [2, b, s] / [2, b, inf_max_seq_length]
        rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb(hidden_states, position_ids) # [2, s, b, h] / [2, inf_max_seq_length, b, h]
    elif self.config.position_embedding_type == 'rope_position_learned_step':
        assert rotary_pos_emb is not None
        learned_abs_pos_emb = self.learned_absolute_step_embedding(position_ids_2) # [s, b, h]
        hidden_states = hidden_states + learned_abs_pos_emb # [s, b, h]
    elif self.config.position_embedding_type == 'rope_position_sinusoid_step':
        hidden_states = self.sinusoid_step_embedding(hidden_states, position_ids_2) # [s, b, h]
        
    # Forward pass through the transformer layers
    if self.config.recompute_granularity == 'full' and self.training:
                raise NotImplementedError
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
    else:
        for l_no, layer in enumerate(self.layers):
            with self.offload_context:
                layer.use_cudagraph = True
                if (len(self.cuda_graphs) == 0) or (not self.training):
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                        halted_k_cache=halted_k_cache,
                        halted_v_cache=halted_v_cache,
                        halt_mask=halt_mask,
                        step=step,
                )
                else:
                    raise NotImplementedError
                    # CUDA graph replay for layer `l_no` and microbatch
                    # `self.current_microbatch`. TransformerEngine versions>=1.10
                    # allow keyword arguments with CUDA graph. However, CUDA graph
                    # acccepts only Tensor inputs and Tensor outputs. Hence,
                    # `inference_params` and `packed_seq_params` are excluded from
                    # input list while output is limited to `hidden_states`.
                    cg_index = self.current_microbatch % len(self.cuda_graphs[l_no])
                    assert not any(
                        [inference_params, packed_seq_params]
                    ), "CUDA graph accepts only Tensor inputs."
                    optional_inputs = self.get_cuda_graph_optional_args(
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        attention_bias,
                        inference_params,
                        packed_seq_params,
                    )
                    hidden_states = self.cuda_graphs[l_no][cg_index](
                        hidden_states, **optional_inputs
                    )

            if (
                torch.is_grad_enabled()
                and self.config.cpu_offloading
                and self.group_prefetch_offload_commit_async is not None
            ):
                hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

    return hidden_states, context
    
def ut_router_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    
        """
        Perform the forward pass through the universal transformer block (router as the act function). 
        """
        # TODO there might be vanishing gradient problem, deal with it. ('Nan')
        # from megatron.training.global_vars import get_timers
        # timers = get_timers()
        
        # timers('overall', log_level=0).start(barrier=True)
        # timers('ut forward [initialize the act states and forward states]', log_level=0).start()
        
        # initialize the act states
        state = hidden_states
        halted_states = None
        halted_k_cache = None
        halted_v_cache = None
        act_max_steps = self.config.act_max_steps
        
        p = torch.zeros(state.size(0), state.size(1), device=state.device, dtype=state.dtype) # [s, b]
        still_running = torch.ones(state.size(0), state.size(1), device=state.device, dtype=torch.bool) # [s, b]
        new_halted = torch.zeros(state.size(0), state.size(1), device=state.device, dtype=torch.bool) # [s, b]
        n_updates = torch.zeros(state.size(0), state.size(1), device=state.device, dtype=torch.int32) # [s, b]
        step = torch.tensor(0, device=state.device, dtype=torch.int32) # it only works for global input step encoding
        
        # history_logits = [[[] for _ in range(state.size(1))] for _ in range(state.size(0))] # [s, b]
        # if not self.config.act_type == "latest":
        #     history_states = [[[] for _ in range(state.size(1))] for _ in range(state.size(0))] # [s, b]
        # else:
        #     history_states = None
        
        act_loss = torch.tensor(0.0, device=state.device, dtype=state.dtype)
        aggregated_states = torch.zeros_like(state)

        # In the original UT implementation, the halted states are actually attended in the later forward passes, so-called fake halting.
        # In addition, the original UT implementation does not cache kv, meaning that the auto-regresive decoding will be different from the vanilla transformer: each time it will parallelly forward the decoded tokens and new tokens. The halting decision and the forwarding pass of the later position in decoding will be dependent on a pre-mature kv of the previous states instead of the final ones. 
        
        def act_function(
            state, 
            step, 
            p0,
            new_halted0,
            still_running0, 
            n_updates,
            halted_states, 
            halted_k_cache, 
            halted_v_cache,
            act_loss,
            aggregated_states
            ):
            """act (position-wise halting) with cache and true halting.
            Every time the act halt function predicts the halt

            Args:
            state: 3-D Tensor: [batch_size, length, channel]
            step: indicates number of steps taken so far
            halting_probability: halting probability
            remainders: act remainders
            n_updates: act n_updates
            previous_state: previous state

            Returns:
            transformed_state: transformed state
            step: step+1
            halting_probability: halting probability
            remainders: act remainders
            n_updates: act n_updates
            new_state: new state
            """
            
            # Increment n_updates for all inputs which are still running
            # Put in the beginning to keep it consistent with the other two act types
            
            # print("Allocated (act_function begin):", torch.cuda.memory_allocated() / 1024**2, "MB")
            # print("Reserved (act_function begin): ", torch.cuda.memory_reserved() / 1024**2, "MB")
            # timers('ut forward [initialize and upda the halted kv cache/states]', log_level=0).start()
        
            n_updates += still_running0 + new_halted0
            
            if not self.config.act_fake_halting:
                # Initialize halted states, and halted kv cache to reuse the activations of the halted positions
                # if halted_states is None:
                #     halted_states = [[None for _ in range(state.size(1))] for _ in range(state.size(0))] # [s, b]
                if halted_k_cache is None:
                    halted_k_cache = [None] * len(self.layers)
                if halted_v_cache is None:
                    halted_v_cache = [None] * len(self.layers)
                
                # Only transform non-halted states, this include the newly halted states
                # newly_halted = new_halted0.unsqueeze(-1) # [s, b, 1]
                active_mask = still_running0.unsqueeze(-1) # [s, b, 1], the newly halted states will not be transformed
                halt_mask = ~active_mask # [s, b, 1]
                
                # Store transformed states of the newly halted positions
                # halted_states = update_halted_states(source_states=state, halted_states=halted_states, mask=newly_halted)
                halted_states = state
                state = state * active_mask # [s, b, h]
                
            else: 
                # newly_halted = None
                halt_mask = None
            
            # timers('ut forward [initialize and upda the halted kv cache/states]').stop()
            
            # timers('ut forward [initialize rope and transformer forward]', log_level=0).start()
            # print("Allocated (act_function before transformer):", torch.cuda.memory_allocated() / 1024**2, "MB")
            # print("Reserved (act_function before transformer): ", torch.cuda.memory_reserved() / 1024**2, "MB")
            # apply transformation on the state
            transformed_state, _ = _ut_router_transformer_forward( # TODO Current implementation is not saving compuation/memory. Implement the compuation/memory or memory efficient version, though it could be a bit tricky due to the fact that the active tokens in each sequence are not the same.
                self,
                hidden_states=state,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                n_updates=n_updates,
                step=step,
                halted_k_cache=halted_k_cache,
                halted_v_cache=halted_v_cache,
                halt_mask=halt_mask,
            )
            
            # timers('ut forward [initialize rope and transformer forward]').stop()
            
            # timers('ut forward [router prediction and post-processing]', log_level=0).start()
            # print("Allocated (act_function right after transformer):", torch.cuda.memory_allocated() / 1024**2, "MB")
            # print("Reserved (act_function right after transformer): ", torch.cuda.memory_reserved() / 1024**2, "MB")
            
            if not self.config.act_fake_halting:
                # Combine transformed active states with halted states
                # transformed_state = retrieve_states(source_states=transformed_state, halted_states=halted_states, mask=active_mask)
                transformed_state = torch.where(active_mask, transformed_state, halted_states)
            
            # the halt function prediction should be done in float32 to increase stability
            # todo is this the most reasonable way? i.e., the layer router decision is not aware of the previous layer's router logits.
            p = router_function(self, transformed_state)
            p *= still_running0
            
            # Store the logits and states to be used finally if we aggregate the states of different steps as the final output
            # The predicted logit of a state can be seen as its matureness, the higher the logit, the more dominant the state will be in the final output, and the more likely it will halt.
            
            # todo: do the logits and state aggregation in one step.
            # for i in range(transformed_state.size(0)):
            #     for j in range(transformed_state.size(1)):
            #         if still_running0[i, j]:
            #             history_logits[i][j].append(p[i, j])
            #             if not self.config.act_type == "latest":
            #                 history_states[i][j].append(transformed_state[i, j])
            
            if self.config.act_loss_type == "sum":
                act_loss -= p.mean()
            else:
                raise NotImplementedError
            
            if self.config.act_type == "aggregate":
                aggregated_states += transformed_state * p
            elif self.config.act_type != "latest":
                raise NotImplementedError
            
            # Mask of inputs which halted at this step
            new_halted = (p > 0.5) * still_running0

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (p <= 0.5) * still_running0
            
            # print("Allocated (act_function right after router):", torch.cuda.memory_allocated() / 1024**2, "MB")
            # print("Reserved (act_function right after router): ", torch.cuda.memory_reserved() / 1024**2, "MB")
            
            # Apply the predicted logits of the last round to the routed states to make the router differentiable. This is similar to the MoE and MoD routing.
            running_part = still_running0.unsqueeze(-1) * transformed_state
            running_part *= (1 - p0).unsqueeze(-1)
            halted_part = new_halted0.unsqueeze(-1) * transformed_state
            halted_part *= p0.unsqueeze(-1) # todo: applying logits to the halted states is not necessary, as did in MoD.
            finish_part = (~(still_running0 + new_halted0)).unsqueeze(-1) * transformed_state
            transformed_state = running_part + halted_part + finish_part
            
            # print("Allocated (act_function right after applying logits):", torch.cuda.memory_allocated() / 1024**2, "MB")
            # print("Reserved (act_function right after applying logits): ", torch.cuda.memory_reserved() / 1024**2, "MB")
            
            if self.config.skip_connection:
                transformed_state = transformed_state + state # only the non-halted states need a skip connection. May not need as the original transformer block already has a skip connection.
            
            # timers('ut forward [router prediction and post-processing]').stop()
            
            return (
                transformed_state, 
                step+1, 
                p,
                new_halted,
                still_running, 
                n_updates, 
                halted_states, 
                halted_k_cache, 
                halted_v_cache,
                act_loss,
                aggregated_states
                )

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=self.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=self.tp_only_amax_red
                )
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        # timers('ut forward [initialize the act states and forward states]').stop()
        
        with rng_context, fp8_context:
            # Forward pass.
            while should_continue_router(
                still_running, 
                n_updates, 
                act_max_steps
                ):
                (
                    state, 
                    step, 
                    p,
                    new_halted,
                    still_running, 
                    n_updates, 
                    halted_states, 
                    halted_k_cache, 
                    halted_v_cache,
                    act_loss,
                    aggregated_states
                    ) = act_function(
                    state, 
                    step, 
                    p,
                    new_halted,
                    still_running, 
                    n_updates, 
                    halted_states, 
                    halted_k_cache, 
                    halted_v_cache,
                    act_loss,
                    aggregated_states
                    )

        
        # timers('ut forward [aggregate the states, final layer norm, and compute act loss]', log_level=0).start()
        
        # if self.config.act_type == "latest":
        #     hidden_states = state
        # elif self.config.act_type == "aggregate":
        #     hidden_states = aggregate_states(history_states, history_logits, softmax=False)
        # elif self.config.act_type == "aggregate_softmax":
        #     hidden_states = aggregate_states(history_states, history_logits, softmax=True)
        # else:
        #     raise ValueError(f"Invalid act type: {self.config.act_type}")
        
        if self.config.act_type == "latest":
            hidden_states = state
        elif self.config.act_type == "aggregate":
            hidden_states = aggregated_states
        else:
            raise NotImplementedError
        
        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # act_loss = compute_act_loss_router(history_logits, self.config.act_loss_type, self.config.act_loss_factor)

        # timers('overall').stop()
        # timers('ut forward [aggregate the states, final layer norm, and compute act loss]').stop()
        # all_time_elapsed = timers('overall').active_time()
        # print(
        #     f"overall time: {timers('overall').active_time()}, ratio: {timers('overall').active_time() / all_time_elapsed * 100:.2f}%\n"
        #     f"ut forward [initialize the act states and forward states]: {timers('ut forward [initialize the act states and forward states]').active_time()}, ratio: {timers('ut forward [initialize the act states and forward states]').active_time() / all_time_elapsed * 100:.2f}%\n"
        #     f"ut forward [initialize and upda the halted kv cache/states]: {timers('ut forward [initialize and upda the halted kv cache/states]').active_time()}, ratio: {timers('ut forward [initialize and upda the halted kv cache/states]').active_time() / all_time_elapsed * 100:.2f}%\n"
        #     f"ut forward [initialize rope and transformer forward]: {timers('ut forward [initialize rope and transformer forward]').active_time()}, ratio: {timers('ut forward [initialize rope and transformer forward]').active_time() / all_time_elapsed * 100:.2f}%\n"
        #     f"ut forward [router prediction and post-processing]: {timers('ut forward [router prediction and post-processing]').active_time()}, ratio: {timers('ut forward [router prediction and post-processing]').active_time() / all_time_elapsed * 100:.2f}%\n"
        #     f"ut forward [aggregate the states, final layer norm, and compute act loss]: {timers('ut forward [aggregate the states, final layer norm, and compute act loss]').active_time()}, ratio: {timers('ut forward [aggregate the states, final layer norm, and compute act loss]').active_time() / all_time_elapsed * 100:.2f}%\n"
        #     )
        
        return hidden_states, act_loss, n_updates



#######################################
# universal transformer - router (pre-routing)
#######################################
def ut_router_prerouting_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    
        """
        Perform the forward pass through the universal transformer block (router as the act function). 
        """
        # TODO there might be vanishing gradient problem, deal with it. ('Nan')
        
        # initialize the act states
        state = hidden_states
        halted_states = None
        halted_k_cache = None
        halted_v_cache = None
        act_max_steps = self.config.act_max_steps
        
        still_running = torch.ones(state.size(0), state.size(1), device=state.device) # [s, b]
        n_updates = torch.zeros(state.size(0), state.size(1), device=state.device) # [s, b]
        step = torch.tensor(0, dtype=n_updates.dtype, device=state.device) # it only works for global input step encoding
        
        if not self.config.act_type == "latest":
            history_logits = [[[] for _ in range(state.size(1))] for _ in range(state.size(0))] # [s, b]
            history_states = [[None for _ in range(state.size(1))] for _ in range(state.size(0))] # [s, b]
        else:
            history_logits = None
            history_states = None

        # In the original UT implementation, the halted states are actually attended in the later forward passes, so-called fake halting.
        # In addition, the original UT implementation does not cache kv, meaning that the auto-regresive decoding will be different from the vanilla transformer: each time it will parallelly forward the decoded tokens and new tokens. The halting decision and the forwarding pass of the later position in decoding will be dependent on a pre-mature kv of the previous states instead of the final ones. 
        
        def act_function(
            state, 
            step, 
            still_running, 
            n_updates,
            halted_states, 
            halted_k_cache, 
            halted_v_cache,
            history_logits=None,
            history_states=None
            ):
            """act (position-wise halting) with cache and true halting.
            Every time the act halt function predicts the halt

            Args:
            state: 3-D Tensor: [batch_size, length, channel]
            step: indicates number of steps taken so far
            halting_probability: halting probability
            remainders: act remainders
            n_updates: act n_updates
            previous_state: previous state

            Returns:
            transformed_state: transformed state
            step: step+1
            halting_probability: halting probability
            remainders: act remainders
            n_updates: act n_updates
            new_state: new state
            """
            
            p = router_function(self.ut_router_activation, self.ut_router_layer, state) # the halt function prediction should be done in float32 to increase stability
            
            # Store the logits and states to be used finally if we aggregate the states of different steps as the final output
            # The predicted logit of a state can be seen as its matureness, the higher the logit, the more dominant the state will be in the final output, and the more likely it will halt.
            # TODO such pre-routing will always integrate the embedding represetation into the final output state (if the act_type is not latest), which might be a good thing or bad thing. post-routing can be considered later. On the positive side, it will directly forward the embedding to the output layer, which offers a chance for completely skipping the whole trasnformer block.
            if not self.config.act_type == "latest":
                for i in range(state.size(0)):
                    for j in range(state.size(1)):
                        if still_running[i, j]:
                            history_logits[i][j].append(p[i, j])
                            history_states[i][j].append(state[i, j])
            
            # Mask of inputs which halted at this step
            new_halted = (p > 0.5) * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (p <= 0.5) * still_running
            
            # Increment n_updates for all inputs which are still running
            n_updates += still_running + new_halted
            
            # apply transformation on the state
            
            if not self.config.act_fake_halting:
                # Initialize halted states, and halted kv cache to reuse the activations of the halted positions
                if halted_states is None:
                    halted_states = [[[] for _ in range(state.size(1))] for _ in range(state.size(0))] # [s, b]
                if halted_k_cache is None:
                    halted_k_cache = [None] * len(self.layers)
                if halted_v_cache is None:
                    halted_v_cache = [None] * len(self.layers)
                
                # Only transform non-halted states, this include the newly halted states
                newly_halted = new_halted.unsqueeze(-1) # [s, b, 1]
                active_mask = still_running.unsqueeze(-1) # [s, b, 1], the newly halted states will not be transformed
                halt_mask = ~active_mask # [s, b, 1]
                
                # Store transformed states of the newly halted positions
                halted_states = update_halted_states(source_states=state, halted_states=halted_states, mask=newly_halted)
                
                transformed_state = state * active_mask # [s, b, h]
                
            else: 
                transformed_state = state
                newly_halted = None
                halt_mask = None
                
            
            transformed_state, _ = _ut_router_transformer_forward( # TODO Current implementation is not saving compuation/memory. Implement the compuation/memory or memory efficient version, though it could be a bit tricky due to the fact that the active tokens in each sequence are not the same.
                self,
                hidden_states=transformed_state,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                n_updates=n_updates,
                halted_k_cache=halted_k_cache,
                halted_v_cache=halted_v_cache,
                halt_mask=halt_mask,
            )
            
            if not self.config.act_fake_halting:
                # Combine transformed active states with halted states
                transformed_state = retrieve_states(source_states=transformed_state, halted_states=halted_states, mask=active_mask)
            
            transformed_state = active_mask * transformed_state * (1 - p) + new_halted.unsqueeze(-1) * transformed_state * p
            
            return (
                transformed_state, 
                step+1, 
                still_running, 
                n_updates, 
                halted_states, 
                halted_k_cache, 
                halted_v_cache,
                history_logits,
                history_states
                )
        
        
        
        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=self.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=self.tp_only_amax_red
                )
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        with rng_context, fp8_context:
            # Forward pass.
            while should_continue_router(
                still_running, 
                n_updates, 
                act_max_steps
                ):
                (
                    state, 
                    step, 
                    still_running, 
                    n_updates, 
                    halted_states, 
                    halted_k_cache, 
                    halted_v_cache,
                    history_logits,
                    history_states
                    ) = act_function(
                    self, 
                    state, 
                    step, 
                    still_running, 
                    n_updates, 
                    halted_states, 
                    halted_k_cache, 
                    halted_v_cache,
                    history_logits,
                    history_states
                    )

        if self.config.act_type == "latest":
            hidden_states = state
        elif self.config.act_type == "aggregate":
            hidden_states = aggregate_states(history_states, history_logits, softmax=False)
        elif self.config.act_type == "aggregate_softmax":
            hidden_states = aggregate_states(history_states, history_logits, softmax=True)
        else:
            raise ValueError(f"Invalid act type: {self.config.act_type}")
        
        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        act_loss = compute_act_loss_router(history_logits, self.config.act_loss_type, self.config.act_loss_factor)
        
        return hidden_states, act_loss
