import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_attention import SpatialTransformer

class UNetModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels: int, n_res_blocks: int, attention_levels: List[int], channel_multipliers: List[int], n_heads: int, tf_layers: int = 1, d_cond: int = 768):
        super().__init__()
        self.channels = channels
        levels = len(channel_multipliers)
        d_time_emb = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLu,
            nn.Linear(d_time_emb, d_time_emb)
        )

        self.input_blocks = nn.ModuleList()

class TimeStepEmbedSequential(nn.Sequential):
    def forward(self):
        pass

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
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    m = UNetModel(in_channels=1, out_channels=1, channels=320, n_res_blocks=1, attention_levels=[], channel_multipliers=[], n_heads=1, tf_layers=1, d_cond=1)
    te = m.time_step_embedding(torch.arange(0, 1000))
    plt.plot(np.arange(1000), te[:, [50, 100, 190, 260]].numpy())
    plt.legend(["dim %d" %p for p in [50, 100, 190, 260]])
    plt.title("Time embeddings")
    plt.show()

if __name__ == "__main__":
    _test_time_embeddings()
