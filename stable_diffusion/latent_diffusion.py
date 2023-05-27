from typing import List

import torch
import torch.nn as nn

from stable_diffusion.models.autoencoder import Autoencoder
from stable_diffusion.models.clip_embedder import CLIPTextEmbedder
from stable_diffusion.models.unet import UNetModel

class DiffusionWrapper(nn.Module):
  def __init__(self, diffusion_model: UNetModel):
    super().__init__()
    self.diffusion_model = diffusion_model
  
  def forward(self):
    pass
