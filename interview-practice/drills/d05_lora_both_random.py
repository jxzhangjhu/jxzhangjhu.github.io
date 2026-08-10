"""d05 · LoRA that is not identity at init   —   budget 3 min

A freshly initialised adapter has to leave the base model bit-for-bit unchanged,
or your first training step starts from a different model than you evaluated.

One line in this file is wrong. Run:  python -m pytest tests/test_d05_lora_both_random.py -q
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base, self.scaling = base, alpha / r
        base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(base.out_features, r) * 0.01)

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling
