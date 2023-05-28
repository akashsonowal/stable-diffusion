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
    
    with torch.no_grad():
      alpha_bar = self.model.alpha_bar
      self.ddim_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
      self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
      self.ddim_alpha_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]))
      self.ddim_sigma = (ddim_eta * ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) * (1 - self.ddim_alpha / self.ddim_alpha_prev))**.5)
      self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5
      
    @torch.no_grad()
    def sample(self, shape: List[int], cond: torch.Tensor, repeat_noise: bool = False, temperature: float = 1., x_last: Optional[torch.Tensor] = None, uncond_scale: float = 1., uncond_cond: Optional[torch.Tensor] = None, skip_steps: int = 0):
      """shape is (bs, c, h, w)""""
      device = self.model.device
      bs = shape[0]
      x = x_last if x_last is not None alse torch.randn(shape, device=device)
                                        
      time_steps = np.flip(self.time_steps)[skip_steps:]
      for i, step in enumerate(time_steps):
        index = len(time_steps) - i - 1 # time: 1, 2, 3, ..., S
        ts = x.new_full((bs,), step, dtype=torch.long) # the size is bs and is filled with the step value 
        x, pred_x0. e_t = self.p_sample(x, cond, ts, step, index=index, repeat_noise=repeat_noise, temperature=temperature, uncond_scale=uncond_scale, uncond_cond=uncond_cond)
      return x
