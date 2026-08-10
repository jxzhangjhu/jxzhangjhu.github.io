"""d01 · Inverted attention mask   —   budget 3 min

The mask marks the positions a query is allowed to see. The scores come back with
every allowed position at -inf, so softmax puts all the weight on the future.

One line in this file is wrong. Run:  python -m pytest tests/test_d01_mask_inverted.py -q
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_scores(q, k):
    """q, k: (B, T, D). Returns pre-softmax scores with the future masked out."""
    T = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    allowed = torch.tril(torch.ones(T, T, dtype=torch.bool))
    return scores.masked_fill(allowed, float("-inf"))
