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
    
    def decode(self, z: torch.Tensor):
        z = self.post_quant_conv(z)
        return self.decoder(z)
    
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
            
            down = nn.Module()
            down.block = resnet_blocks

            if i != n_resolutions - 1:
                down.downsample = DownSample(channels)
            else:
                down.downsample = nn.Identity()
            
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResNetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResNetBlock(channels, channels)
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, 2*z_channels, 3, stride=1, padding=1)
    
    def forward(self, img: torch.Tensor):
        x = self.conv_in(img)

        for down in self.down:
            for block in down.block:
                x = block(x)

            x = down.downsample(x)

        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        x = self.norm_out(x) 
        x = swish(x)
        x = self.conv_out(x)
        return x

class Decoder(nn.Module):

    def __init__(self, *, channels: int, channel_multipliers: List[int], n_resnet_blocks: int, out_channels: int, z_channels: int):
        super().__init__()
        num_resolutions = len(channel_multipliers)
        channel_list = [m * channels for m in channel_multipliers]

        channels = channel_list[-1]

        self.conv_in = nn.Conv2d(z_channels, channels, 3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResNetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResNetBlock(channels, channels)

        self.up = nn.ModuleList()

        for i in reversed(range(num_resolutions)):
            resnet_blocks = nn.ModuleList()

            for _  in range(n_resnet_blocks + 1):
                resnet_blocks.append(ResNetBlock(channels, channel_list[i]))
                channels = channel_list[i]

            up = nn.Module()
            up.block = resnet_blocks

            if i!=0:
                up.upsample = UpSample(channels)
            else:
                up.upsample = nn.Identity()
            
            self.up.insert(0, up)
        
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)

    def forward(self, z: torch.Tensor):
        h = self.conv_in(z)
        
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        img = self.conv_out(h)
        return img


class GaussianDistribution(nn.Module):
    """
    parameters are the means and log of variances of the embedding of shape [batch_size, z_channels * 2, z_height, z_height]
    """
    def __init__(self, parameters: torch.Tensor):
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.log_var)
    
    def sample(self):
        return self.mean + self.std * torch.randn_like(self.std)

class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = normalization(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        b, c, h, w = q.shape
        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)

        attn = torch.einsum("bci, bcj -> bij", q, k) * self.scale
        attn = F.softmax(attn, dim=2)
        out = torch.einsum("bij, bcj -> bci", attn, v)
        out = out.view(b, c, h, w)
        out = self.proj_out(out)
        return x + out

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



