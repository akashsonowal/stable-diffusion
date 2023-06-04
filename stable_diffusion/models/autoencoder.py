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
        


