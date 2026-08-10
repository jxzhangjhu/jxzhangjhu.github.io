"""p02 · KV cache and incremental decode   —   budget 15 min  [cold-start set]

Add a KV cache so decoding step t costs O(t) instead of O(t^2).
The cached path must be numerically identical to a full recompute.

Fill in the body. Run:  python run.py p02
Stuck? hints/p02_kv_cache.md has three levels, in increasing order of spoiler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class CachedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x, cache=None):
        """x: (B, T_new, d_model). cache: (k, v) from previous steps, or None.

        Returns (output, new_cache). During decode T_new == 1.
        """
        raise NotImplementedError
