"""d06 · Softmax stability failure   —   budget 3 min

Symptom: large finite logits produce non-finite probabilities

One line in this file is wrong. Run:  python -m pytest tests/test_d06_softmax_overflow.py -q
Stuck? Read hints/d06_softmax_overflow.md one level at a time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax(x):
    """x: (..., V) -> the same shape, rows summing to 1."""
    e = (x - x.max(dim=-1, keepdim=True).values).exp()
    return e / e.sum(dim=-1, keepdim=True)
