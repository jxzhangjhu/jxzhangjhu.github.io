"""d06 · softmax without the max subtraction   —   budget 3 min

Works on toy logits, overflows to inf on the logit magnitudes a real model produces.

One line in this file is wrong. Run:  python -m pytest tests/test_d06_softmax_overflow.py -q
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax(x):
    """x: (..., V) -> the same shape, rows summing to 1."""
    e = (x - x.max(dim=-1, keepdim=True).values).exp()
    return e / e.sum(dim=-1, keepdim=True)
