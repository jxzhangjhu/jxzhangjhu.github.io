"""d02 · Head-merge failure   —   budget 3 min

Symptom: head merging raises or interleaves values

One line in this file is wrong. Run:  python -m pytest tests/test_d02_missing_contiguous.py -q
Stuck? Read hints/d02_missing_contiguous.md one level at a time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def merge_heads(x):
    """x: (B, H, T, Dh) -> (B, T, H*Dh), heads concatenated per position."""
    B, H, T, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * Dh)
