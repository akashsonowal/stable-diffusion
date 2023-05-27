import argparse
import os
from pathlib import Path

import torch

from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.sampler.ddim import DDIMSampler
from stable_diffusion.sampler.ddpm import DDPMSampler
from stable_diffusion.util import load_model, save_images, set_seed

class Text2Img:
  model: LatentDiffusion
  
  def __init__(self, checkpoint_path: Path, sampler_name: str, n_steps: int = 50, ddim_eta: float = 0.0):
    self.model = load_model(checkpoint_path)
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.model.to(self.device)
    if sampler_name == "ddim":
      self.sampler = DDIMSampler(self.model, n_steps=n_steps, ddim_eta=ddim_eta)
    elif sampler_name == "ddpm":
      self.sampler = DDPMSampler(self.model)
 
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument()
  
  args = parser.parse_args()
  set_seed(42)
   
if __name__ == "__main__":
  main()
