from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
        super().__init__()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x: [batch_size, channels, height, width]
        cond: [batch_size, n_cond, d_cond]
        """
        b, c, h, w = x.shape
        x_in = x
        

class BasicTransformerBlock(nn.Module):
    pass 

class CrossAttention(nn.Module):
    pass 

class GeGLU(nn.Module):
    pass 

