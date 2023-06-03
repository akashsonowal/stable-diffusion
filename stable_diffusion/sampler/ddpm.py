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
      alpha_bar_prev = torch.cat()


