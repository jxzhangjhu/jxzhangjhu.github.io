"""Tests for d06 · softmax without the max subtraction. Run: python -m pytest tests/test_d06_softmax_overflow.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d06_softmax_overflow as d  # noqa: E402

def test_survives_realistic_logit_magnitudes():
    x = torch.tensor([[1e4, 2e4, 3e4]])
    got = d.softmax(x)
    assert torch.isfinite(got).all(), \
        "exp() overflowed to inf, then inf/inf gave nan: subtract the row max first"
    assert torch.allclose(got, F.softmax(x, dim=-1), atol=1e-6)


def test_still_correct_on_ordinary_logits():
    x = torch.randn(3, 7)
    assert torch.allclose(d.softmax(x), F.softmax(x, dim=-1), atol=1e-6)
