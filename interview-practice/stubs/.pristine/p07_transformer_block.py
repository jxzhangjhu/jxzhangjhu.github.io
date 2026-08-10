"""p07 · A full pre-norm block   —   budget 15 min

A full pre-norm transformer block.

Fill in the body. Run:  python run.py p07
Stuck? hints/p07_transformer_block.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block:
    def __init__(self, *a, **kw):
        raise NotImplementedError
