"""p22 · Byte-pair encoding   —   budget 20 min  [cold-start set]

Byte-pair encoding: train the merges, then encode with them.

Fill in the body. Run:  python run.py p22
Stuck? hints/p22_bpe.md has three levels, in increasing order of spoiler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def bpe_train(text, num_merges):
    """Return a dict mapping (a, b) -> new_id, in the order the merges were learned."""
    raise NotImplementedError


def bpe_encode(text, merges):
    """Apply merges in learned order. Returns a list of ints."""
    raise NotImplementedError
