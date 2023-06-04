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
    
    def encode():
        pass
    
    def decode():
        pass
    
class Encoder(nn.Module):
    pass 

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



