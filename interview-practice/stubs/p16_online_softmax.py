"""p16 · Streaming softmax   —   budget 15 min

Streaming (online) softmax over blocks.

Fill in the body. Run:  python run.py p16
Stuck? hints/p16_online_softmax.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def online_softmax_weighted_sum(*args, **kwargs):
    """Signature matches reference.online_softmax_weighted_sum."""
    raise NotImplementedError
