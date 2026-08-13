"""p21 · Generalised advantage estimation   —   budget 15 min

Implement generalised advantage estimation and its returns.

Fill in the body. Run:  python run.py p21
Stuck? hints/p21_gae.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_gae(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    """rewards, values: (T,). Return (advantages, returns)."""
    raise NotImplementedError
