"""p17 · Tiled FlashAttention forward   —   budget 25 min

Implement tiled exact attention with the online-softmax recurrence.

Fill in the body. Run:  python run.py p17
Stuck? hints/p17_flash_attention.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def flash_attention_forward(q, k, v, block_q=16, block_kv=16, causal=True):
    """q,k,v: (B,H,T,D). Return (output, row_logsumexp)."""
    raise NotImplementedError
