from collections import OrderedDict
from typing import Dict, Literal, Optional

import numpy as np
from torch import Tensor
import torch

from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import DiffLMTransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType

from megatron.core.parallel_state import (
    get_data_parallel_rank
)

from ....training.global_vars import get_args

class GPTModel(LanguageModule):
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)
        
        self.args = get_args()
        
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 4 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
            )

        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

        # DLM Transformer.
        self.decoder = DiffLMTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt'
            )
            
        self.global_training_step_cnt = None
        self.accumulation_step_cnt = None

        # Separate RNG generator for diffusion process
        self.difflm_rng_generator = None
        self._difflm_rng_generator_seed = self.args.seed + get_data_parallel_rank()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def vanilla_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_params: InferenceParams,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        runtime_gather_output: Optional[bool] = None,
        extra_block_kwargs: Optional[dict] = None,
    ) -> Tensor:
        """
        Forward pass for the latent next word prediction.
        The LM decoder will perform teacher-forcing next word prediction in multiple depths of the latent space.
        """
        
        # for the vanilla forward, we use the causal_bottom_right attention mask and revert it back. 
        # we do this inside the vanilla forward function to reduce overhead for other settings as normal NWP is 
        # is a less likely setting in difflm.
        for i in range(len(self.decoder.layers)):
            self.decoder.layers[i].self_attention.attn_mask_type = AttnMaskType.causal_bottom_right
        
        hidden_states = self.decoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                **(extra_block_kwargs or {}),
            )
        
        # revert back the attention mask type based on the original setting
        if self.args.attention_mask_type == 'causal_bottom_right':
            attn_mask_type = AttnMaskType.causal_bottom_right
        elif self.args.attention_mask_type == 'no_mask':
            attn_mask_type = AttnMaskType.no_mask
        elif self.args.attention_mask_type == 'block_causal':
            attn_mask_type = AttnMaskType.block_causal
        else:
            raise ValueError(f"Invalid attention mask type: {self.args.attention_mask_type}")
        
        for i in range(len(self.decoder.layers)):
            self.decoder.layers[i].self_attention.attn_mask_type = attn_mask_type
        
        return hidden_states
    
    def difflm_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_params: InferenceParams,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        runtime_gather_output: Optional[bool] = None,
        extra_block_kwargs: Optional[dict] = None,
    ) -> Tensor:
        """
        Forward pass for the difflm.
        """     
        hidden_states = self.decoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                **(extra_block_kwargs or {}),
            )
        
        return hidden_states

    def test_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_params: InferenceParams,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        runtime_gather_output: Optional[bool] = None,
        extra_block_kwargs: Optional[dict] = None,
    ) -> Tensor:
        """
        Forward pass for the latent next word prediction.
        The LM decoder will perform teacher-forcing next word prediction in multiple depths of the latent space.
        """     
        hidden_states = self.decoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                **(extra_block_kwargs or {}),
            )
        
        return hidden_states
    
    def use_varilen_data(self, input_ids, varilen_prob=0.01):
        if not self.training:
            return False, input_ids.shape[1]
        
        # we ensure forward strategy is the same for all devices of a single accumulation step to minimize bubble
        # record the current global step and accumulation step
        if self.global_training_step_cnt is None:
            self.global_training_step_cnt = self.args.curr_iteration
            assert self.accumulation_step_cnt is None
            self.accumulation_step_cnt = 0
        elif self.global_training_step_cnt != self.args.curr_iteration:
            self.global_training_step_cnt = self.args.curr_iteration
            self.accumulation_step_cnt = 0
        else:
            self.accumulation_step_cnt += 1
        
        # number of accumulation steps in a global training step
        num_accum = self.args.global_batch_size // (self.args.micro_batch_size * self.args.data_parallel_size)

        # we need to store the current rng state to restore it at the end of this function
        rng_state = np.random.get_state()
        _seed = self.global_training_step_cnt * num_accum + self.accumulation_step_cnt
        np.random.seed(_seed)
        random_number = np.random.uniform(0, 1)
        random_length = np.random.randint(low=2, high=input_ids.shape[1] + 1)
        
        # restore the rng state
        np.random.set_state(rng_state)
        return random_number < varilen_prob, random_length
    
    def _get_device_from_generator(self, gen: torch.Generator):
        # PyTorch >= 2.0 exposes a .device attribute; fall back to CPU if unavailable
        return gen.device

    def difflm_forward_process(self, input_ids):
        if self.difflm_rng_generator is None or self.difflm_rng_generator.device != input_ids.device:
            self.difflm_rng_generator = torch.Generator(device=input_ids.device).manual_seed(self._difflm_rng_generator_seed)
            
        # we use different rng states for the difflm forward process for different data parallel ranks, while being the same for the same data parallel rank
        b, l = input_ids.shape
        t = torch.rand(b, device=input_ids.device, generator=self.difflm_rng_generator)
        p_mask = t
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids.device, generator=self.difflm_rng_generator) < p_mask
        
        # choose a random position to mask for those rows with no masked indices
        zero_masked_tokens_row = masked_indices.sum(dim=1) == 0
        rand_indices = torch.randint(0, l, (zero_masked_tokens_row.sum(),), device=input_ids.device, generator=self.difflm_rng_generator)
        masked_indices[zero_masked_tokens_row, rand_indices] = True
        
        # get the exact p_mask
        p_mask = masked_indices.sum(dim=1) / l
        p_mask = p_mask[:, None].repeat(1, l)
        
        noisy_batch = torch.where(masked_indices, self.args.mask_token, input_ids)
        return noisy_batch, masked_indices, p_mask
    
    def reshape_inputs(self, input_ids, position_ids, attention_mask, labels, target_l):
        b, l = input_ids.shape
        assert l % target_l == 0
        if attention_mask is not None:
            raise NotImplementedError("Attention mask reshaping is not supported.")
        input_ids = input_ids.reshape(b, -1, target_l).reshape(-1, target_l)
        labels = labels.reshape(b, -1, target_l).reshape(-1, target_l)
        assert (position_ids[:, 0] == 0).all() and (position_ids[:, -1] == l - 1).all(), f"position_ids are not expected: {position_ids}"
        position_ids = torch.ones_like(input_ids).cumsum(dim=1)-1
        
        return input_ids, position_ids, attention_mask, labels
    
    def update_packing_info_random_shrink(self, packed_seq_params, random_length):
        """
        Upate the packing params due to the random shrink in difflm training.
        """
        if packed_seq_params is None:
            return None
        assert packed_seq_params.cu_seqlens_q is packed_seq_params.cu_seqlens_kv
        cu_lens = packed_seq_params.cu_seqlens_q
        # we only take the :random_length part of the cu_lens
        cu_lens_mask = cu_lens < random_length
        cu_lens = cu_lens[cu_lens_mask]
        cu_lens = torch.cat((cu_lens, torch.tensor([random_length], device=cu_lens.device, dtype=cu_lens.dtype)))
        
        seqlens = cu_lens[1:] - cu_lens[:-1]
        max_seqlen = seqlens.max()
        
        packed_seq_params.cu_seqlens_q = cu_lens
        packed_seq_params.cu_seqlens_kv = cu_lens
        packed_seq_params.max_seqlen_q = max_seqlen
        packed_seq_params.max_seqlen_kv = max_seqlen
        
        return packed_seq_params
        
    def update_packing_info_shift_by_one(self, packed_seq_params):
        """
        Upate the packing params due to the shift by one in difflm training 
        (for convenience, we didn't modify the gpt dataset which shifts the label by one, here we shift it back).
        """
        if packed_seq_params is None:
            return None
        assert packed_seq_params.cu_seqlens_q is packed_seq_params.cu_seqlens_kv
        cu_lens = packed_seq_params.cu_seqlens_q
        cu_lens = torch.unique(torch.cat((cu_lens[0:1], cu_lens[1:] - 1)))
        seqlens = cu_lens[1:] - cu_lens[:-1]
        max_seqlen = seqlens.max()
        
        packed_seq_params.cu_seqlens_q = cu_lens
        packed_seq_params.cu_seqlens_kv = cu_lens
        packed_seq_params.max_seqlen_q = max_seqlen
        packed_seq_params.max_seqlen_kv = max_seqlen
        
        return packed_seq_params
    
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        if "doing_on_the_fly_eval" in self.args and self.args.doing_on_the_fly_eval:
            return self.mdm_compute_loglikelihood(**input_ids)
        
        self.args.model_running_mode_curr = self.args.model_running_mode # todo: remove the legacy impl
            
        difflm_mask = None
        if self.args.model_running_mode_curr == "difflm-noshift":
            if self.args.attention_mask_type == 'no_mask':
                attention_mask = None
                # randomly truncate the data with a 1% probability when use bidirectional attention
                use_varilen_data, random_length = self.use_varilen_data(
                    input_ids, 
                    self.args.difflm_varilen_prob
                    )
                if use_varilen_data:
                    input_ids = input_ids[:, :random_length]
                    position_ids = position_ids[:, :random_length]
                    labels = labels[:, :random_length]
                    packed_seq_params = self.update_packing_info_random_shrink(
                        packed_seq_params, 
                        random_length
                        )
            
            # unshift the input_ids and labels by 1
            input_ids = input_ids[:, 1:]
            position_ids = position_ids[:, 1:]
            labels = labels[:, :-1]
            packed_seq_params = self.update_packing_info_shift_by_one(packed_seq_params)
            self.real_input_length = input_ids.shape[1]
            input_ids, difflm_mask, p_mask = self.difflm_forward_process(input_ids)
        elif self.args.model_running_mode_curr == "vanilla":
            pass
        elif self.args.model_running_mode_curr == "test-forward":
            pass
        else:
            raise NotImplementedError(f"Running mode {self.args.model_running_mode_curr} not implemented")
        
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            raise NotImplementedError("Decoder input is not supported")
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            if not self.training and self.config.flash_decode:
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cos_sin(
                    inference_params.max_sequence_length
                )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_params, self.decoder, decoder_input, self.config, packed_seq_params
                )
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                )
                
        # Run decoder in different modes.

        if self.args.model_running_mode_curr == "difflm-noshift":
            decoder_forward_func = self.difflm_forward
        elif self.args.model_running_mode_curr == "vanilla":
            decoder_forward_func = self.vanilla_forward
        elif self.args.model_running_mode_curr == "test-forward":
            decoder_forward_func = self.test_forward
        else:
            raise NotImplementedError(f"Running mode {self.args.model_running_mode_curr} not implemented")
        
        hidden_states = decoder_forward_func(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                runtime_gather_output=runtime_gather_output,
                **(extra_block_kwargs or {}),
            )
        
        
        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )
        
        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'decoder_input': decoder_input,
                    'logits': logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)
        
        if self.args.model_running_mode_curr == "difflm-noshift":
                loss = loss * difflm_mask / p_mask # [b, l]
        
        if self.config.gpt_block_return_loss_and_logits:
            return loss, logits, difflm_mask
        
        return loss, difflm_mask

    def compute_hybrid_entropy_loss(self, logits: Tensor, difflm_mask: Tensor) -> Tensor:
        """
        Compute the hybrid entropy loss for the remaining tokens in the sequence when using hybrid sampling.
        """
        
        raise NotImplementedError("Entropy loss is not implemented")

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility
        (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the
        # _extra_state key but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}'

        return sharded_state_dict