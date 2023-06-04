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
    x = x_last if x_last is not None else torch.randn(shape, device=device)

    time_steps = np.flip(self.time_steps)[skip_steps:] # S-i, S-i-1, ....1
    for i, step in enumerate(time_steps):
      index = len(time_steps) - i - 1 
      ts = x.new_full((bs,), step, dtype=torch.long) # the size is bs and is filled with the step value 
      x, pred_x0. e_t = self.p_sample(x, cond, ts, step, index=index, repeat_noise=repeat_noise, temperature=temperature, uncond_scale=uncond_scale, uncond_cond=uncond_cond)
    return x
  
  @torch.no_grad()
  def p_sample(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int, index: int, repeat_noise: bool = False, temperature: float = 1., uncond_scale: float = 1., uncond_cond: Optional[torch.Tensor] = None):
     """
     x is of shape (bs, c, h, w)
     c is of shape (bs, emb_dim)
     t is of shape (bs)
     step is int
     index is index in the list
     repeat noise specifies whether the noise should be same for all samples in the batch
     temperature: random noise gets multiplied by this
     """
     e_t = self.get_eps(x, t, c, uncond_scale=uncond_scale, uncond_cond=uncond_cond)
     x_prev, x_0 = self.get_x_prev_and_pred_x_0(e_t, index, x, temperature=temperature, repeat_noise=repeat_noise)
     return x_prev, x_0, e_t
  
   def get_x_prev_and_pred_x_0(self, e_t: torch.Tensor, index: int , x: torch.Tensor, temperature: float, repeat_noise: bool):
     alpha = self.ddim_alpha[index]
     alpha_prev = self.ddim_alpha_prev[index]
     sigma = self.ddim_sigma[index]
     sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
     
     pred_x0 =  (x - sqrt_one_minus_alpha * e_t) / (alpha ** .5) # current prediction
     dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t # direction pointing to xt
     if sigma = 0:
      noise = 0
     elif repeat_noise:
      noise = torch.randn((1, *x.shape[1:]), device=x.device) # (1, bs, c, h, w)
     else:
      noise = torch.randn(x.shape, device=x.device)
     noise = noise * temperature
     x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
     return x_prev, pred_x0
   
   @torch.no_grad()
   def q_sample(self, x: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
      if noise is None:
        noise = torch.randn_like(x0)
      return self.ddim_alpha_sqrt(index) * x0 + self.ddim_sqrt_one_minus_alpha(index) * noise 
   
   @torch.no_grad()
   def paint(self, x: torch.Tensor, cond: torch.Tensor, t_start: int, *, orig: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None, uncond_scale: float = 1., uncond_cond: Optional[torch.Tensor] = None):
      bs = x.shape[0]
      time_steps = np.flip(self.time_steps[:t_start])
      
      for i, step in enumerate(time_steps):
        index = len(time_steps) - i - 1
        ts = x.new_full((bs,), step, dtype=torch.long)
        x, _, _ = self.p_sample(x, cond, ts, step, index=index, uncond_scale=uncond_scale, uncond_cond=uncond_cond)
        if orig is not None:
          orig_t = self.q_sample(orig, index, noise=orig_noise)
          x = orig_t * mask + x * ( 1 - mask)          
                                        
      return x                                  
