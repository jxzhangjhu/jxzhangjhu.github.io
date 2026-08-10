"""p03 · Grouped-query attention   —   budget 10 min  [reported: Datadog]

Grouped-query attention: n_kv_heads < n_heads, each group sharing one K/V head.

Fill in the body. Run:  python run.py p03
Stuck? hints/p03_gqa.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, cache=None):
        """cache: dict with 'k','v', mutated in place. None disables caching."""
        raise NotImplementedError
