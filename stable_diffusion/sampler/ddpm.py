from typing import Tuple, Optional

import numpy as np
import torch

from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.sampler import DiffusionSampler

class DDPMSampler(DiffusionSampler):
  model: LatentDiffusion
