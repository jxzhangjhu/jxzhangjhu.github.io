"""p07 · A full pre-norm block   —   budget 15 min

Assemble the completed attention, RMSNorm, and SwiGLU exercises into a pre-norm block.

Fill in the body. Run:  python run.py p07
Stuck? hints/p07_transformer_block.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stubs.p01_mha import CausalSelfAttention
from stubs.p05_rmsnorm import RMSNorm
from stubs.p06_swiglu import SwiGLU


class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
