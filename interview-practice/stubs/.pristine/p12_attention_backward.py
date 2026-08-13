"""p12 · Attention backward by hand   —   budget 25 min

Derive and implement the attention backward pass without autograd.

Fill in the body. Run:  python run.py p12
Stuck? hints/p12_attention_backward.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def attention_backward(d_out, cache):
    """cache is returned by reference.attention_forward: (q, k, v, p, scale)."""
    raise NotImplementedError
