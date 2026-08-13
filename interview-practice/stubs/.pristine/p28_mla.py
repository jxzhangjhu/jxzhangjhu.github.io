"""p28 · Multi-head latent attention and compressed cache   —   budget 25 min

Multi-head latent attention: cache a low-rank KV latent plus a small positional key.

Fill in the body. Run:  python run.py p28
Stuck? hints/p28_mla.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, kv_rank, rope_dim):
        super().__init__()
        # TODO: split each query/key head into non-positional and RoPE parts.
        # Cache only the shared latent c and the shared rotated positional key.
        raise NotImplementedError

    def forward(self, x, cache=None):
        """x: (B,T,D); cache is a mutable dict with compressed c and k_rope."""
        raise NotImplementedError
