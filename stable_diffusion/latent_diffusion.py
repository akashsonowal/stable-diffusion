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
  
  def __init__(self, unet_model: UNetModel, autoencoder: Autoencoder, clip_embedder: CLIPTextEmbedder, latent_scaling_factor: float, n_steps: int, linear_start: float, linear_end: float):
    super().__init__()
    self.first_stage_model = autoencoder
    self.model = DiffusionWrapper(unet_model)
  
  def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
    """Predict noise given the latent representations x, time step t and conditioning context c"""
    return self.model(x, t, context)
