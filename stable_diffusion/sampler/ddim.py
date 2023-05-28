from typing import Optional, List

import numpy as np
import torch

from stable_diffusion.latent_diffusion import LatentDiffusion
import DiffusionSampler

class DDIMSampler(DiffusionSampler):
  model: LatentDiffusion
  
  def __init__(self, model: LatentDiffusion. n_steps: int, ddim_discretize: str = "uniform", ddim_eta: float = 0.):
    super().__init__(model)
    self.n_steps = model.n_steps
    
    if ddim_discretize == "uniform":
      c = self.n_steps // n_steps
      self.time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1
    elif ddim_discretize == "quad":
      self.time_steps = ((np.linspace(0, np.sqrt(self.n_steps * .8), n_steps))**2).astype(int) + 1
    else:
      raise NotImplementedError(ddim_discretize)
