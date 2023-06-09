import argparse
from pathlib import Path
from typing import Optional

import torch

from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.sampler.ddim import DDIMSampler
from stable_diffusion.util import load_model, save_images, load_img, set_seed

class Img2Img:
  def __init__(self, *, checkpoint_path: Path, ddim_steps: int = 50, ddim_eta: float = 0.0): 
    self.ddim_steps = ddim_steps
    self.model = load_model(checkpoint_path)
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.model.to(self.device)
    self.sampler = DDIMSampler(self.model, n_steps=n_steps, ddim_eta=ddim_eta)
  
  @torch.no_grad()
  def __call__(self, *, dest_path: str, orig_img: str, strength: float, batch_size: int = 3, prompt: str, uncond_scale: float = 5.0):
    prompts = batch_size * [prompt]
    orig_image = load_img(orig_img).to(self.device)
    orig = self.model.autoencoder_encode(orig_image).repeat(batch_size, 1, 1, 1)
    
    assert 0. <= strength <= 1., "can only work with strength in (0.0, 1.0)"
    t_index = int(strength * self.ddim_steps)
    
    with torch.cuda.amp.autocast():
      if uncond_scale != 1:
        un_cond = self.model.get_text_conditioning(batch_size * [""]) # empty string
      else:
        un_cond = None
        
      cond = self.model.get_text_conditioning(prompts)

      x = self.sampler.q_sample(orig, t_index)
      x = self.sampler.paint(x, cond, t_index, uncond_scale=uncond_scale, uncond_cond=un_cond)
      images = self.model.autoencoder_decode(x)
      
    save_images(images, dest_path, "img_")
    
        
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--prompt", type=str, nargs="?", default="a painting of a cute monkey playing guitar", help="The prompt to render")
  parser.add_argumnet("--orig_img", type=str, nargs="?", help="path to the imput image")
  parser.add_argument("--batch_size", type=int, default=4, help="batch size")
  parser.add_argument("--steps", type=int, default=50, help="number of sampling steps")
  parser.add_argument("--scale", type=float, default=5.0, help="unconditional guidance scale: ", help="unconditional guidance scale: ", 
                      "eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
  parser.add_argumnet("--strength", type=float, default=0.75, help="strength for noise: ", "1.0 corresponds to full destruction of init image.")
 
  args = parser.parse_args()
  set_seed(42)
  
  checkpoints = Path("/checkpoints/")
  
  img_2_img = Img2Img(checkpoint_path=checkpoints / "sd-v1-4.ckpt", ddim_steps=args.steps)
  img_2_img(dest_path="outputs", orig_img=args.orig_img, strength=args.strength, batch_size=args.batch_size, prompt=args.prompt, uncond_scale=args.scale) 
   
if __name__ == "__main__":
  main()
