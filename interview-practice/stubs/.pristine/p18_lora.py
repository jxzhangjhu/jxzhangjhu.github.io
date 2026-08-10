"""p18 · LoRA with a lossless merge   —   budget 10 min  [cold-start set]

LoRA: a low-rank adapter that is the identity at initialisation and merges losslessly.

Fill in the body. Run:  python run.py p18
Stuck? hints/p18_lora.md has three levels, in increasing order of spoiler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def merged_weight(self):
        """Return the single weight matrix equivalent to base + adapter."""
        raise NotImplementedError
