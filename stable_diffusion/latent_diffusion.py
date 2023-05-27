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
  
  def forward(self, x: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor):
    return self.diffusion_model(x, time_steps, context)
 
class LatentDiffusion(nn.Module):
  model: DiffusionWrapper
  first_stage_model: Autoencoder
  cond_stage_model: CLIPTextEmbedder
  
  def __init__(self, unet_model: UNetModel):
    self.model = DiffusionWrapper(unet_model)
  
  def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
    """Predict noise given the latent representations x, time step t and conditioning context c"""
    return self.model(x, t, context)
