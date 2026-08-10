"""d02 · view() on a transposed tensor   —   budget 3 min

Merging heads back together after a transpose. The tensor is no longer contiguous,
so view() cannot reinterpret its strides.

One line in this file is wrong. Run:  python -m pytest tests/test_d02_missing_contiguous.py -q
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def merge_heads(x):
    """x: (B, H, T, Dh) -> (B, T, H*Dh), heads concatenated per position."""
    B, H, T, Dh = x.shape
    return x.transpose(1, 2).view(B, T, H * Dh)
