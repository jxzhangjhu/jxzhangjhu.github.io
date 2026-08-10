"""p08 · Cross entropy with log-sum-exp   —   budget 10 min  [cold-start set]

Cross entropy from logits, with the log-sum-exp trick and ignore_index support.

Fill in the body. Run:  python run.py p08
Stuck? hints/p08_cross_entropy.md has three levels, in increasing order of spoiler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy(logits, targets, ignore_index=-100):
    """logits: (N, V) raw scores. targets: (N,) int64. Returns a scalar mean loss."""
    raise NotImplementedError
