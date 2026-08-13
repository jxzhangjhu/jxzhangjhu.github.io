"""Tests for p11 · A minimal scalar autograd. Run: python run.py p11"""

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
from stubs import p11_autograd as stub  # noqa: E402

def test_forward_and_reused_node_gradients_match_torch():
    a, b = stub.Value(-4.0), stub.Value(2.0)
    c = a * b + b.tanh()
    out = c * c + a / b
    out.backward()

    ta = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    tc = ta * tb + torch.tanh(tb)
    tout = tc * tc + ta / tb
    tout.backward()
    assert abs(out.data - tout.detach().item()) < 1e-9
    assert abs(a.grad - ta.grad.detach().item()) < 1e-6
    assert abs(b.grad - tb.grad.detach().item()) < 1e-6
