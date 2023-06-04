from typing import List

import torch
import torch.nn.functional as F
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, emb_channels: int, z_channels: int):
        # embedding_space (z) to quantized_embedding_space (emb)
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 

        self.quant_conv = nn.Conv2d(2*z_channels, 2*emb_channels, 1) # both mean and variance
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)
    
    def encode(self, img: torch.Tensor) -> "GaussianDistribution":
        z = self.encoder(img) # (batch_size, z_channels * 2, height, weight)
        moments = self.quant_conv(z)
        return GaussianDistribution(moments)
    
    def decode():
        pass
    
class Encoder(nn.Module):
    def __init__(self, *, channels: int, channels_multipliers: List[int], n_resnet_blocks: int, in_channels: int, z_channels: int):
        super().__init__()
        n_resolutions = len(channels_multipliers)
        self.conv_in = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)
        channels_list = [m * channels for m in [1] + channels_multipliers]

        self.down = nn.ModuleList()

        for i in range(n_resolutions):
            resnet_blocks = nn.ModuleList()

            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResNetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]

        self.conv_out = nn.Conv2d(channels, 2*z_channels, 3, stride=1, padding=1)
    
    def forward(self, img: torch.Tensor):
        x = self.conv_in(img)

        for down in self.down:
            for block in down.block:
                x = block(x)

            x = down.downsample(x)
            
        x = self.conv_out(x)
        return x

class Decoder(nn.Module):
    pass 

class GaussianDistribution(nn.Module):
    pass 

class AttnBlock(nn.Module):
    pass 

class UpSample(nn.Module):
    pass 

class DownSample(nn.Module):
    pass 

class ResNetBlock(nn.Module):
    pass 

def swish(x: torch.Tensor):
    return x * torch.sigmoid(x) 

def normalization():
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)



