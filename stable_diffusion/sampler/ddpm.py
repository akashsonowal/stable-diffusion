from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from stable_diffusion.util import gather

class DenoiseDiffusion:
  def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
    super().__init__()
