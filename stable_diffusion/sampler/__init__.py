from typing import Optional, List
import torch
from stable_diffusion.latent_diffusion import LatentDiffusion

class DiffusionSampler:
  model: LatentDiffusion
  def __init__(self, model: LatentDiffusion):
    super().__init__()
    self.model = model
    self.n_steps = model.n_steps
    
  def get_eps():
    pass
  
  def sample():
    pass
  
  def paint():
    pass
  
  def q_sample():
    pass
