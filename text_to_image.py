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
  
  def __call__(self, *, dest_path: str, batch_size: int = 3, prompt: str, h: int = 512, w: int = 512, uncond_scale: float = 7.5):
    c = 4 # channels in a image
    f = 8 # image to latent space resolution reduction
    prompts = batch_size * [prompt]
    
    with torch.cuda.amp.autocast():
      if uncond_scale != 1:
        un_cond = self.model.get_text_conditioning(batch_size * [""]) # empty string
      else:
        un_cond = None
        
      cond = self.model.get_text_conditioning(prompts)

      x = self.sampler.sample(cond=cond, shape=(batch_size, c, h // f, w //f), uncond_scale, uncond_cond=un_cond)
      images = self.model.autoencoder_decode(x)
      
    save_images(images, dest_path, "txt_")
    
        
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument()
  
  args = parser.parse_args()
  set_seed(42)
   
if __name__ == "__main__":
  main()
