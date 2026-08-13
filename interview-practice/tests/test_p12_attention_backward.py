"""Tests for p12 · Attention backward by hand. Run: python run.py p12"""

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
from stubs import p12_attention_backward as stub  # noqa: E402

def test_all_three_gradients_match_autograd():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 6, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 6, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 6, 8, dtype=torch.float64, requires_grad=True)
    out, cache = R.attention_forward(q, k, v, causal=True)
    upstream = torch.randn_like(out)
    out.backward(upstream)
    got = stub.attention_backward(upstream, cache)
    for name, mine, want in zip(("dQ", "dK", "dV"), got, (q.grad, k.grad, v.grad)):
        assert mine.shape == want.shape, f"{name} has shape {mine.shape}, expected {want.shape}"
        assert torch.allclose(mine, want, atol=1e-9), name
