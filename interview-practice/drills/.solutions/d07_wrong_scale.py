"""d07 · Attention-scale failure   —   budget 3 min

Symptom: attention disagrees with the scaled-dot-product definition

One line in this file is wrong. Run:  python -m pytest tests/test_d07_wrong_scale.py -q
Stuck? Read hints/d07_wrong_scale.md one level at a time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(q, k, v, n_heads):
    """q, k, v: (B, T, D) with D = n_heads * d_head. Non-causal, single call."""
    B, T, D = q.shape
    d_head = D // n_heads
    shape = (B, T, n_heads, d_head)
    q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)
    out = F.softmax(scores, dim=-1) @ v
    return out.transpose(1, 2).contiguous().view(B, T, D)
