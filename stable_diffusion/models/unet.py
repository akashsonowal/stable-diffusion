import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_attention import SpatialTransformer

class UNetModel(nn.Module):
    def __init__(self, in_channels, out_channels, channels, n_res_blocks, attention_levels, channel_multipliers, n_heads, tf_layers: int = 1, d_cond: int = 768):
        super().__init__()
        self,channels = channels
        levels = len(channel_multipliers)
        d_time_emb = channels * 4
        self.time_embed = nn.Sequential()
        

class UpSample(nn.Module):
    pass  

class DownSample(nn.Module):
    pass 

class ResBlock(nn.Module):
    pass 

class GroupNorm32():
    pass 

def normalization():
    pass 

def _test_time_embeddings():
    pass 

if __name__ == "__main__":
    _test_time_embeddings()
