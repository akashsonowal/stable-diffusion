from typing import Optional, List
import torch
from stable_diffusion.latent_diffusion import LatentDiffusion

class DiffusionSampler:
  model: LatentDiffusion
  def __init__(self, model: LatentDiffusion):
    super().__init__()
    self.model = model
    self.n_steps = model.n_steps
    
  def get_eps(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, *, uncond_scale: float, uncond_cond:Optional[torch.Tensor]):
    if uncond_cond is None or uncond_scale == 1:
      return model(x, t, c)
    
    # duplicate x_t and t
    x_in = torch.cat([x] * 2)
    t_in = torch.cat([t] * 2)
    c_in = torch.cat([uncond_cond, c])
    
    e_t_uncond,  e_t_cond = self.model(x_in, t_in, c_in).chunk(2)
    e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)
    return e_t
  
  def sample(self, shape: List[int], cond: torch.Tensor, repeat_noise: bool = False, temperature: float = 1., x_last: Optional[torch.Tensor] = None, uncond_scale: float = 1., uncond_cond: Optional[torch.Tensor] = None, skips_steps: int = 0):
    raise NotImplementedError()
  
  def paint(self, x: torch.Tensor, cond: torch.Tensor, t_start: int, *, orig: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None, uncond_scale: float = 1., uncond_cond: Optional[torch.Tensor] = None):
    raise NotImplementedError()
  
  def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
    raise NotImplementedError()
