"""p11 · A 40-line autograd   —   budget 30 min  [reported: OpenAI 2+]

A minimal reverse-mode autograd.

Fill in the body. Run:  python run.py p11
Stuck? hints/p11_autograd.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Value:
    def __init__(self, *a, **kw):
        raise NotImplementedError
