"""p11 · A minimal scalar autograd   —   budget 30 min

Implement scalar reverse-mode autodiff in the style of micrograd.

Fill in the body. Run:  python run.py p11
Stuck? hints/p11_autograd.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Value:
    def __init__(self, data, _children=(), _op=""):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __pow__(self, k):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else Value(other) ** -1)

    def __hash__(self):
        return id(self)
