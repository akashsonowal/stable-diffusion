from typing import Optional, List

import numpy as np
import torch

from stable_diffusion.latent_diffusion import LatentDiffusion
import DiffusionSampler

class DDIMSampler(DiffusionSampler):
  model: LatentDiffusion
  
  def __init__(self, model: LatentDiffusion. n_steps: int, ddim_discretize: str = "uniform", ddim_eta: float = 0.):
    super().__init__(model)
  pass
