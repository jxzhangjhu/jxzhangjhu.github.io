"""Tests for p06 · SwiGLU feed-forward. Run: python run.py p06"""

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
from stubs import p06_swiglu as stub  # noqa: E402

def test_matches_reference_numerically():
    torch.manual_seed(0)
    mine, ref = stub.SwiGLU(96), R.SwiGLU(96)
    ref.load_state_dict(mine.state_dict())
    x = torch.randn(2, 5, 96)
    assert torch.allclose(mine(x), ref(x), atol=1e-6)


def test_default_width_matches_four_x_ffn_parameter_count():
    d = 96
    m = stub.SwiGLU(d)
    got = sum(p.numel() for p in m.parameters())
    want = 2 * d * (4 * d)
    assert abs(got - want) / want < 0.02
