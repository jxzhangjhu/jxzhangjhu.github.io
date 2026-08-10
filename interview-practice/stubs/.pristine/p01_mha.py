"""p01 · Causal multi-head attention   —   budget 20 min  [cold-start set]

Write multi-head causal self-attention from scratch. No nn.MultiheadAttention,
no F.scaled_dot_product_attention.

Fill in the body. Run:  python run.py p01
Stuck? hints/p01_mha.md has three levels, in increasing order of spoiler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        # TODO: n_heads, d_head, one fused qkv projection, an output projection,
        # and a causal mask registered as a buffer.
        raise NotImplementedError

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, d_model)"""
        raise NotImplementedError
