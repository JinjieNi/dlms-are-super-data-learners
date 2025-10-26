from torch import nn
import torch
from megatron.core.transformer.qwen2_norm import Qwen2RMSNorm

class Qwen2LayerNormTorchLinear(nn.Linear):
    def __init__(self, in_features, out_features, config, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias)
        self.norm = Qwen2RMSNorm(config, in_features, eps=config.layernorm_epsilon)

        
    def get_extra_state(self):
        return None
    
    def set_extra_state(self, extra_state):
        pass
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(self.norm(input)), None