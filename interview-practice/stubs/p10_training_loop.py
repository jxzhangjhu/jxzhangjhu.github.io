"""p10 · Overfit a tiny batch   —   budget 20 min  [cold-start set]

Write a training loop that drives a tiny fixed batch to near-zero loss.
This is the smoke test every real run should start with.

Fill in the body. Run:  python run.py p10
Stuck? hints/p10_training_loop.md has three levels, in increasing order of spoiler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def overfit_tiny(steps=2000, lr=0.5):
    """Build a small model + fixed batch, train it, and report the result.

    Returns (final_loss, accuracy) as plain floats.
    """
    raise NotImplementedError
