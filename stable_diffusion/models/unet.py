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
            nn.SiLu(),
            nn.Linear(d_time_emb, d_time_emb)
        )

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(TimeStepEmbedSequential(nn.Conv2d(in_channels, channels, 3, padding=1)))

        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]

        for i in range(levels):
            for _ in range(n_res_blocks):
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
            
            if i in attention_levels:
                layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
            
            self.input_blocks.append(TimeStepEmbedSequential(*layers))
            input_block_channels.append(channels)

            if i != levels - 1:
                self.input_blocks.append(TimeStepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        self.middle_block = TimeStepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, n_heads. tf_layers, d_cond),
            ResBlock(channels, d_time_emb)
        )
        self.output_blocks = nn.ModuleList([]) 

        for i in reversed(range(levels)):
            for j in range(n_res_blocks + 1):
                layers = [ResBlock(channels + input_block_channels.pop(), d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))
                self.output_blocks.append(TimeStepEmbedSequential(*layers))
                
        self.out = nn.Sequential(normalization(channels), nn.SiLU(), nn.Conv2d(channels, out_channels, 3, padding=1))       

    
    def time_step_embedding(self, time_steps: torch.Tensor, max_steps: int = 10000):
        """
        time_steps: (bs, )
        max_steps for min freq
        """
        half = self.channels // 2 # half the channels are sine and other half is cosine
        frequencies = torch.exp(
            math.log(max_steps) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        args = time_steps[:, None].float() * frequencies[None]  
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor):
        """
        x: (bs, c, h, w)
        time_steps: (bs,)
        cond: (bs, n_cond, d_cond)
        """
        x_input_block = [] # store input half for skip connection

        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_-embed(t_emb)

        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)
        
        x = self.middle_block(x, t_emb, cond)

        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)
        
        return self.out(x)


class TimeStepEmbedSequential(nn.Sequential):
    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x

class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor):
        return self.op(x)

class ResBlock(nn.Module):
    def __init__(self, channels: int, d_t_emb: int, *, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = channels 
        
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), nn.Conv2d(channels, out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(d_t_emb, out_channels))
        self.out_layers = nn.Sequential(normalization(out_channels), nn.SiLU(), nn.Dropout(0.), nn.Conv2d(out_channels, out_channels, 3, padding=1))
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor): # x is of shape (bs, c, h, w); t_emb is of shape (bs, d_t_emb)
        h = self.in_layers(x) # initial convolutions
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        h = h + t_emb[:, :, None, None]
        h = self.out_layers(h )# final convolutions
        return self.skip_connection(x) + h

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    return GroupNorm32(32, channels) # 32 groups

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
