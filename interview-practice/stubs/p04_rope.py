"""p04 · Rotary position embeddings   —   budget 15 min  [cold-start set]

Implement rotary position embeddings: build the cos/sin table, then apply it to q and k.

Fill in the body. Run:  python run.py p04
Stuck? hints/p04_rope.md has three levels, in increasing order of spoiler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def rope_cache(seq_len, d_head, base=10000.0):
    """Return (cos, sin), each (seq_len, d_head // 2)."""
    raise NotImplementedError


def apply_rope(x, cos, sin):
    """x: (..., T, d_head) -> same shape, rotated pairwise."""
    raise NotImplementedError
