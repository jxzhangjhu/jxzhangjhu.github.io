"""p14 · Temperature, top-k, top-p   —   budget 15 min  [cold-start set]

Sampling with temperature, top-k and top-p. Order matters.

Fill in the body. Run:  python run.py p14
Stuck? hints/p14_sampling.md has three levels, in increasing order of spoiler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """logits: (V,) -> an int token id."""
    raise NotImplementedError
