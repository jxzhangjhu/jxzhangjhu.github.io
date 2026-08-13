"""d01 · Causal-mask failure   —   budget 3 min

Symptom: attention can see the future

One line in this file is wrong. Run:  python -m pytest tests/test_d01_mask_inverted.py -q
Stuck? Read hints/d01_mask_inverted.md one level at a time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_scores(q, k):
    """q, k: (B, T, D). Returns pre-softmax scores with the future masked out."""
    T = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    allowed = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    return scores.masked_fill(allowed, float("-inf"))
