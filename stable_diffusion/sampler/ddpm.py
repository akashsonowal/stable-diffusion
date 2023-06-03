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
      self.recip_m1_alpha_bar = (1 / alpha_bar - 1) ** .5
      variance = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
      self.log_var = torch.log(torch.clamp(variance, min=1e-20)) 
      self.mean_x0_coeff = beta * (alpha_bar_prev ** .5) / (1. - alpha_bar)
      self.mean_xt_coeff = (1. - alpha_bar_prev) * (1 - beta) ** 0.5 / (1. - alpha_bar)
  
  @torch.no_grad()
  def sample(self, shape: List[int], cond: torch.Tensor, repeat_noise: bool = False, temperature: float = 1., x_last: Optional[torch.Tensor] = None, uncond_scale: float = 1., uncond_cond: Optional[torch.Tensor] = None, skip_steps: int = 0):
    device = self.model.device
    bs = shape[0]
    x = x_last if x_last is not None else torch.randn(shape, device=device)
    time_steps = np.flip(self.time_steps)[skip_steps:]
    
    for step in time_steps:
      ts = x.new_full((bs,), step, dtype=torch.long)
      x, pred_x0, e_t = self.p_sample(x, cond, ts, step, repeat_noise=repeat_noise, temperature=temperature, uncond_scale=uncond_scale, uncond_cond=uncond_cond)
      return x
  
  @torch.no_grad()
  def p_sample(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, step: int, repeat_noise: bool = False, temperature: float = 1., uncond_scale: float = 1., uncond_cond: Optional[torch.Tensor] = None):
    e_t = self.get_eps(x, t, c, uncond_scale, uncond_cond)
    bs = x.shape[0]
    sqrt_recip_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recip_alpha_bar[step])
    sqrt_recip_m1_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recip_m1_alpha_bar[step])

    x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t 

    mean_x0_coef = x.new_fill((bs, 1, 1, 1), self.mean_x0_coeff[step])
     
    pass 
  
  @torch.no_grad()
  def q_sample(self):
    pass
