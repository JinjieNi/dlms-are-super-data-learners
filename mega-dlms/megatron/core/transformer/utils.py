# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Utilities for transformer layers."""
from functools import lru_cache
from operator import itemgetter
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import math
import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedStateDict, StateDict
from megatron.core.jit import jit_fuser
from megatron.core.utils import (
    make_sharded_tensor_for_checkpoint,
    make_tp_sharded_tensor_for_checkpoint,
)


def get_linear_layer(rows, columns, init_method, perform_initialization=True):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if perform_initialization:  # Take from modelparallel config
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer


@lru_cache(maxsize=32)
def get_default_causal_mask(sq: int) -> torch.Tensor:
    """Return the causal upper triangular mask for softmax input."""
    return torch.triu(torch.ones(sq, sq, device="cuda"), diagonal=1).bool()


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


@jit_fuser
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def openai_gelu(x):
    return gelu_impl(x)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@jit_fuser
def erf_gelu(x):
    return (
        x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))
    )


def make_sharded_tensors_for_checkpoint(
    state_dict: StateDict,
    prefix: str,
    tensor_parallel_layers_axis_map: Optional[Dict[str, int]] = None,
    sharded_offsets: Iterable[Tuple[int, int, int]] = (),
    extra_state_suffix: str = '_extra_state',
):
    """Wraps tensors from transformer layers with ShardedTensor or ShardedObject.

    For a given `state_dict`, wraps:
    - all _extra_states with ShardedObject
    - all tensors specified in tensor_parallel_layers_axis_map with TP and DP sharded ShardedTensor
    - other values with DP sharded ShardedTensor

    Args:
        state_dict (StateDict): state_dict to convert
        prefix (str): prefix appended to keys in final state dict
        tensor_parallel_layers_axis_map (Dict[str, int], optional): dict mapping layer
            names to the axis for TP sharding
        sharded_offsets (Iterable[Tuple[int, int, int]], optional): sharding already
            applied (e.g. PP related), passed along to ShardedTensor
        extra_state_suffix (str, default = '_extra_state'): layers with this
            suffix will be wrapped with ShardedObject instead of ShardedTensor.

    """

    if tensor_parallel_layers_axis_map is None:
        tensor_parallel_layers_axis_map = {}

    sharded_state_dict = {}
    for layer_name in state_dict.keys():
        tensor = state_dict[layer_name]
        layer_key = f'{prefix}{layer_name}'

        if layer_name.endswith(extra_state_suffix):
            sharded_state_dict[layer_key] = make_sharded_object_for_checkpoint(
                tensor, layer_key, sharded_offsets
            )

        elif layer_name in tensor_parallel_layers_axis_map:
            tp_axis = tensor_parallel_layers_axis_map[layer_name]
            sharded_state_dict[layer_key] = make_tp_sharded_tensor_for_checkpoint(
                tensor, layer_key, tp_axis, prepend_offsets=sharded_offsets
            )

        else:
            sharded_state_dict[layer_key] = make_sharded_tensor_for_checkpoint(
                tensor, layer_key, prepend_offsets=sharded_offsets
            )

    return sharded_state_dict


def make_sharded_object_for_checkpoint(
    obj: Any,
    key: str,
    sharded_offsets: Iterable[Tuple[int, int, int]] = (),
    replica_id: Union[None, int, Tuple[int, ...]] = None,
    **kwargs,
):
    """Helper for instantiating a non-sharded ShardedObject (replicated across TP and DP group).

    Args:
        obj (object): any object to be sharded
        key (str): unique identifier of the object
        sharded_offsets (Iterable[Tuple[int, int, int]]): offsets normally
            prepended to ShardedTensors, will be used as global offsets for
            ShardedObject
        replica_id (Union[None, int, Tuple[int, ...]]): replica id
    """
    if replica_id is None:
        replica_id = (
            0,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

    return ShardedObject(key, obj, *_get_extra_state_offsets(sharded_offsets), replica_id, **kwargs)


def _get_extra_state_offsets(
    sharded_offsets: Iterable[Tuple[int, int, int]]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Turns ShardedTensor offsets into offsets suitable for ShardedObject."""
    if sharded_offsets:
        sharded_offsets = sorted(sharded_offsets, key=itemgetter(0))  # sort by axis
        axis, extra_state_offset, extra_state_shape = zip(*sharded_offsets)
        assert list(axis) == list(
            range(len(axis))
        ), f'Expected contiguous axis for offsets: {sharded_offsets}'
    else:
        extra_state_shape = (1,)
        extra_state_offset = (0,)
    return extra_state_shape, extra_state_offset


def sharded_state_dict_default(
    module: torch.nn.Module,
    prefix: str = '',
    sharded_offsets: Tuple[Tuple[int, int, int]] = (),
    metadata: Optional[dict] = None,
) -> ShardedStateDict:
    """Provides implementation for sharded_state_dict method for non-MegatronModules.

    Tries to call `module.sharded_state_dict` when possible,
    otherwise uses regular state dict and assumes tensors are replicated across TP and DP.

    `keep_vars=True` is passed to module.state_dict so that optimizer states
    can be sharded later on.

    Args:
        module (torch.nn.Module): module which sharded state dict we want to obtain
        prefix (str): prefix for the state dict keys
        sharded_offsets (Tuple[Tuple[int, int, int]], optional): sharding already
            applied (e.g. PP related) by sup-modules. Passed along to ShardedTensor
        metadata (dict, optional): metadata passed to module sharded_state_dict method

    Returns:
        dict: dictionary of state dict keys mapped to ShardedTensors
    """

    if hasattr(module, 'sharded_state_dict'):
        module_sharded_sd = module.sharded_state_dict(
            prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )
    else:
        module_sd = module.state_dict(prefix='', keep_vars=True)
        module_sharded_sd = make_sharded_tensors_for_checkpoint(
            module_sd, prefix, {}, sharded_offsets
        )
    return module_sharded_sd




class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        """
        Args:
            max_seq_len (int): The maximum sequence length for which we store encodings.
            embed_model_dim (int): The dimension of the embedding (d_model).
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        # Create a (max_seq_len x embed_dim) table of positional encodings
        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                if i + 1 < self.embed_dim:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / self.embed_dim)))
        pe = pe.unsqueeze(1)  # shape: [max_seq_len, 1, embed_dim]

        # register_buffer => pe is not a parameter, but still part of the model state
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None):
        """
        Args:
            x (Tensor): An input embedding of shape [seq_len, batch_size, embed_dim]
            position_ids (Optional[Tensor]): 1D or 2D tensor specifying positions.
                                             If None, default to [0, 1, 2, ..., seq_len-1].
        Returns:
            Tensor: The input tensor x plus the corresponding positional encodings.
        """
        # Scale the input embeddings
        x = x * math.sqrt(self.embed_dim)

        # If no position_ids, use default 0..seq_len-1 for each item in the batch
        if position_ids is None:
            seq_len = x.size(0)
            # Just take the first seq_len positions from the precomputed table
            position_encoding = self.pe[:seq_len, :]  # shape [seq_len, 1, embed_dim]
        else:
            # If position_ids is provided, use them directly
            # shape can be [seq_len] or [seq_len, batch_size]
            if position_ids.dim() == 1:
                # Expand dims to match [seq_len, batch_size]
                position_ids = position_ids.unsqueeze(1)
            position_encoding = self.pe[position_ids, 0, :]  # shape [seq_len, batch_size, embed_dim]

        return x + position_encoding
