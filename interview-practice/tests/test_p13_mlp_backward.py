"""Tests for p13 · MLP backward by hand. Run: python run.py p13"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p13_mlp_backward as stub  # noqa: E402

def test_every_gradient_matches_autograd():
    torch.manual_seed(0)
    kw = dict(dtype=torch.float64, requires_grad=True)
    x, W1, b1 = torch.randn(5, 4, **kw), torch.randn(4, 9, **kw), torch.randn(9, **kw)
    W2, b2 = torch.randn(9, 4, **kw), torch.randn(4, **kw)
    y, cache = R.mlp_forward(x, W1, b1, W2, b2)
    upstream = torch.randn_like(y)
    y.backward(upstream)
    got = stub.mlp_backward(upstream, cache)
    for mine, want in zip(got, (x.grad, W1.grad, b1.grad, W2.grad, b2.grad)):
        assert mine.shape == want.shape
        assert torch.allclose(mine, want, atol=1e-9)
