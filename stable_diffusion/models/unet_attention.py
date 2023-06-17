from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6. affine=True)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(channels, n_heads, channels // n_heads, d_cond=d_cond) for _ in range(n_layers)])
        self.proj_out = nn.Cond2d(channels, channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x: [batch_size, channels, height, width]
        cond: [batch_size, n_cond, d_cond]
        """
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).view(b, h*w, c)
        for block in self.transformer_blocks:
            x = block(x, cond)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        return x + x_in
        
class BasicTransformerBlock(nn.Module):
    pass 

class CrossAttention(nn.Module):
    pass 

class GeGLU(nn.Module):
    pass 
