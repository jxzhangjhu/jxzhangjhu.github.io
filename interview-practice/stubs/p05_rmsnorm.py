"""p05 · RMSNorm   —   budget 5 min  [cold-start set]

RMSNorm: normalise by the root mean square, no mean subtraction, no bias.

Fill in the body. Run:  python run.py p05
Stuck? hints/p05_rmsnorm.md has three levels, in increasing order of spoiler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
