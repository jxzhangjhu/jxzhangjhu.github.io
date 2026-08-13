"""d03 · Nucleus-support failure   —   budget 4 min

Symptom: the nucleus has the wrong support

One line in this file is wrong. Run:  python -m pytest tests/test_d03_top_p_off_by_one.py -q
Stuck? Read hints/d03_top_p_off_by_one.md one level at a time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def nucleus(logits, top_p):
    """logits: (V,), 0 < top_p <= 1. Returns logits outside the nucleus as -inf."""
    if not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    srt, idx = torch.sort(logits, descending=True)
    probs = F.softmax(srt, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    drop = cum - probs >= top_p
    srt = srt.masked_fill(drop, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(0, idx, srt)
