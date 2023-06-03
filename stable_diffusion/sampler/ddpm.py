from typing import Tuple, Optional

import numpy as np
import torch

from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.sampler import DiffusionSampler

class DDPMSampler(DiffusionSampler):
  model: LatentDiffusion

  def __init__(self, model: LatentDiffusion):
    super().__init__(model)
    self.time_steps = np.asarray(list(range(self.n_steps)))
    with torch.no_grad():
      alpha_bar = self.model.alpha_bar
      beta = self.model.beta
      alpha_bar_prev = torch.cat([alpha_bar.new_tensor([1.], alpha_bar[:-1])])
      self.sqrt_alpha_bar = alpha_bar ** .5
      self.sqrt_1m_alpha_bar = (1. - alpha_bar) ** .5
      self.recip_alpha_bar = alpha_bar ** -.5
  
  @torch.no_grad()
  def p_sample(self):
    pass 
  
  @torch.no_grad()
  def q_sample(self):
    pass
