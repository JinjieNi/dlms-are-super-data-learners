"""Pre-training Diffusion Language Models."""

import os
import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.difflm import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.difflm.gpt_layer_specs import (
    get_difflm_decoder_block_spec,
    get_difflm_layer_with_transformer_engine_spec,
    get_difflm_layer_local_spec
)
from custom_args.difflm import extra_args_provider
from megatron.core.packed_seq_params import PackedSeqParams

stimer = StragglerDetector()

def get_base_model(model):
    """
    Unwraps any nested .module attributes until you reach the actual
    GPTModel (or whichever base class you are after).
    """
    while hasattr(model, "module"):
        model = model.module
    return model

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        raise NotImplementedError("Legacy models are not supported yet.")
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            # raise NotImplementedError("args.spec is not supported yet.")
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_difflm_decoder_block_spec(
                    config, 
                    use_transformer_engine=use_te, 
                    attn_mask_type=args.attention_mask_type, 
                    core_attn_implementation=args.core_attn_implementation, 
                    moe_router_type=args.moe_router_type
                    )
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_difflm_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts, 
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm, 
                        multi_latent_attention=args.multi_latent_attention, 
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                        attn_mask_type=args.attention_mask_type,
                        core_attn_implementation=args.core_attn_implementation
                        )
                else:
                    transformer_layer_spec = get_difflm_layer_local_spec(
                        num_experts=args.num_experts, 
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm, 
                        multi_latent_attention=args.multi_latent_attention, 
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm
                        )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            if args.base_model == "vanilla":
                model = GPTModel(
                    config=config,
                    transformer_layer_spec=transformer_layer_spec,
                    vocab_size=args.padded_vocab_size,
                    max_sequence_length=args.max_position_embeddings,
                    pre_process=pre_process,
                    post_process=post_process,
                    fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                    parallel_output=True,
                    share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                    position_embedding_type=args.position_embedding_type,
                    rotary_percent=args.rotary_percent,
                    rotary_base=args.rotary_base,
                    rope_scaling=args.use_rope_scaling
                )
            else:
                raise NotImplementedError("Models other than vanilla GPT models are not supported yet.")

    print_rank_0(str(model))
    
    return model


def _get_packed_seq_params(tokens, labels, loss_mask, attention_mask, position_ids):
    """
    Get the packed sequence parameters for the given tokens.
    """

    args = get_args()
    
    if not (args.reset_position_ids and args.reset_attention_mask): # we must turn on both to use packed sequence
        return tokens, labels, loss_mask, attention_mask, position_ids, None
    
    tokenizer = get_tokenizer()
    eod_id = tokenizer.eod

    tokens = tokens.view(-1)
    eod_indices = torch.where(tokens == eod_id)[0]
        
    seq_ends = (eod_indices + 1).to(torch.int32)
    
    if args.cut_off_varlen_to_seqlen:
        seqlen_multiples = torch.arange(
            args.seq_length, 
            tokens.shape[0] + args.seq_length, args.seq_length, 
            device=tokens.device, dtype=torch.int32
            )
        seq_ends = torch.cat((seq_ends, seqlen_multiples))
    else:
        raise NotImplementedError("Packing with cross-sequence attention is not yet supported as the dataset blending & shuffling will break the consecutive sequence assumption.")
    
    zero_start = torch.tensor([0], dtype=torch.int32, device=tokens.device)
    cu_end = torch.tensor([tokens.shape[0]], dtype=torch.int32, device=tokens.device)
    cu_seqlens = torch.unique(torch.cat((zero_start, seq_ends, cu_end))) # zero_start and the cu_end can happen to be the start/end of a document
        
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = seqlens.max()
    
    packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format='thd',
        )
    
    # when packing, the batch dimension is a dummy dimension
    tokens = tokens.unsqueeze(0)
    labels = labels.view(-1).unsqueeze(0)
    loss_mask = loss_mask.view(-1).unsqueeze(0)
    attention_mask = attention_mask.view(-1).unsqueeze(0) if attention_mask is not None else None
    position_ids = position_ids.view(-1).unsqueeze(0)
    
    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    
    tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
    
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = _get_packed_seq_params(
        tokens, 
        labels, 
        loss_mask, 
        attention_mask, 
        position_ids
        )

    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params


# define spiky loss as a variation of 20% or more
SPIKY_LOSS_PERC = 0.2


def loss_func(loss_mask: torch.Tensor, model: GPTModel, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(rerun_state_machine.is_spiky_loss, threshold=SPIKY_LOSS_PERC),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    
    gpt_model_instance = get_base_model(model)
    if args.model_running_mode == "difflm-noshift":
        logging_dict = {
            'lm loss': (reporting_loss[0], reporting_loss[1]),
            'real input length': gpt_model_instance.real_input_length,
            }
    elif args.model_running_mode == "vanilla":
        logging_dict = {
            'lm loss': (reporting_loss[0], reporting_loss[1]),
            }
    elif args.model_running_mode == "test-forward":
        logging_dict = {
            'lm loss': (reporting_loss[0], reporting_loss[1]),
            }
    else:
        raise NotImplementedError("not supported yet, need implementation.")
    
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        logging_dict,
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor, difflm_mask = model(tokens, position_ids, attention_mask,
                              labels=labels, packed_seq_params=packed_seq_params)
        
        if args.model_running_mode_curr == "difflm-noshift":
            if args.attention_mask_type == 'no_mask':
                loss_mask = loss_mask[:,:difflm_mask.shape[1]].contiguous()
        elif args.model_running_mode_curr == "vanilla":
            assert difflm_mask is None
        elif args.model_running_mode_curr == "test-forward":
            assert difflm_mask is None
        else:
            raise NotImplementedError("not supported yet, need implementation.")

    return output_tensor, partial(loss_func, loss_mask, model)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        renormalize_blend_weights=args.renormalize_blend_weights,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds
    

if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'HuggingFaceTokenizer'},
        extra_args_provider=extra_args_provider,
    )
