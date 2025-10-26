from torch import Tensor
import torch
from torch.nn.attention.flex_attention import flex_attention
flex_attention = torch.compile(flex_attention)

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType



class TorchFlexAttention(torch.nn.Module):
    """
    Wrapper for the Torch's flex_attention.
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        k_channels: int = None,
        v_channels: int = None,
        cp_comm_type: str = "p2p",
    ):
        super().__init__()

    def get_extra_state(self):
        return None
    
    def set_extra_state(self, extra_state):
        pass
        
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward."""
        query = query.transpose(0, 1).contiguous().transpose(1,2).contiguous()
        key = key.transpose(0, 1).contiguous().transpose(1,2).contiguous()
        value = value.transpose(0, 1).contiguous().transpose(1,2).contiguous()
        core_attn_out = flex_attention(query, key, value, block_mask=attention_mask, enable_gqa=True)
        core_attn_out = core_attn_out.transpose(1,2).contiguous().transpose(0, 1).contiguous()
        s, b, p, h = core_attn_out.shape
        core_attn_out = core_attn_out.reshape(s, b, p * h)
        
        return core_attn_out
    