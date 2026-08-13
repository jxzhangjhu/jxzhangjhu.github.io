"""p06 · SwiGLU feed-forward   —   budget 5 min

SwiGLU feed-forward: three matrices, with the 8/3 sizing.

Fill in the body. Run:  python run.py p06
Stuck? hints/p06_swiglu.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
