"""d03 · top-p drops the crossing token   —   budget 4 min

Nucleus sampling should keep the shortest prefix whose cumulative mass reaches p,
which means the token that crosses the threshold stays in.

One line in this file is wrong. Run:  python -m pytest tests/test_d03_top_p_off_by_one.py -q
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def nucleus(logits, top_p):
    """logits: (V,). Returns logits with everything outside the nucleus set to -inf."""
    srt, idx = torch.sort(logits, descending=True)
    probs = F.softmax(srt, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    drop = cum >= top_p
    srt = srt.masked_fill(drop, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(0, idx, srt)
