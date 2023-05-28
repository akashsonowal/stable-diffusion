import argparse
from pathlib import Path
from typing import Optional

import torch

from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.sampler.ddim import DDIMSampler
from stable_diffusion.util import load_model, save_images, load_img, set_seed

class InPaint:
  model: LatentDiffusion
  sampler: DiffusionSampler
  
  def __init__(self, checkpoint_path: Path, ddim_steps: int = 50, ddim_eta: float = 0.0): 
    self.ddim_steps = ddim_steps
    self.model = load_model(checkpoint_path)
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.model.to(self.device)
    self.sampler = DDIMSampler(self.model, n_steps=n_steps, ddim_eta=ddim_eta)
  
  @torch.no_grad()
  def __call__(self, *, dest_path: str, orig_img: str, strength: float, batch_size: int = 3, prompt: str, uncond_scale: float = 7.5, mask: Optional[torch.Tensor] = None):
    prompts = batch_size * [prompt]
    orig_image = load_img(orig_img).to(self.device)
    orig = self.model.autoencoder_encode(orig_image).repeat(batch_size, 1, 1, 1)
    
    if mask is None:
      mask = torch.zeros_like(orig, device=self.device)
      mask[:, :, mask.shape[2] // 2, :] = 1. # preserve the bottom half of the image
    else:
      mask = mask.to(self.device)
    
    orig_noise = torch.randn(orig.shape, device=self.device)
    
    assert 0. <= strength <= 1., "can only work with strength in (0.0, 1.0)"
    t_index = int(strength * self.ddim_steps)
    
    with torch.cuda.amp.autocast():
      if uncond_scale != 1:
        un_cond = self.model.get_text_conditioning(batch_size * [""]) # empty string
      else:
        un_cond = None
        
      cond = self.model.get_text_conditioning(prompts)

      x = self.sampler.q_sample(orig, t_index, noise=orig_noise)
      x = self.sampler.paint(x, cond, t_index, orig=orig, mask=mask, orig_noise=orig_noise, uncond_scale, uncond_cond=un_cond)
      images = self.model.autoencoder_decode(x)
      
    save_images(images, dest_path, "paint_")
    
        
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar", help="The prompt to render")
  parser.add_argument("--batch_size", type=int, default=4, help="batch size")
  parser.add_argument("--sampler", dest="sampler_name", choices=["ddim", "ddpm"], default="ddim", help=f"Set the sampler")
  
  parser.add_argument("--flash", action="store_true", help="whether to use flash attention")
  parser.add_argument("--steps", type=int, default=50, help="number of sampling steps")
  parser.add_argument("--scale", type=float, default=7.5, help="unconditional guidance scale: ", help="unconditional guidance scale: ", 
                      "eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
  
  args = parser.parse_args()
  set_seed(42)
  
  checkpoints = Path("/checkpoints/")
  
  from stable_diffusion.models.unet_attention import CrossAttention
  CrossAttention.use_flash_attention = args.flash
  
  
  txt2img = Txt2Img(checkpoint_path=checkpoints / "sd-v1-4.ckpt", sampler_name=args.sampler_name, n_steps=args.steps)
  txt2img(dest_path="outputs", batch_size=args.batch_size, prompt=args.prompt, uncond_scale=args.scale) 
   
if __name__ == "__main__":
  main()
