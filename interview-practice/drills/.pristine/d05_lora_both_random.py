"""d05 · LoRA initialisation failure   —   budget 3 min

Symptom: a fresh adapter changes the base model

One line in this file is wrong. Run:  python -m pytest tests/test_d05_lora_both_random.py -q
Stuck? Read hints/d05_lora_both_random.md one level at a time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base, self.scaling = base, alpha / r
        for p in base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(base.out_features, r) * 0.01)

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling
