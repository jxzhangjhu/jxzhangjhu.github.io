"""Tests for p07 · A full pre-norm block. Run: python run.py p07"""

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
from stubs import p07_transformer_block as stub  # noqa: E402

def test_matches_reference_and_preserves_shape():
    torch.manual_seed(0)
    mine, ref = stub.Block(64, 8), R.Block(64, 8)
    ref.load_state_dict(mine.state_dict())
    x = torch.randn(2, 6, 64)
    assert mine(x).shape == x.shape
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_block_is_causal():
    torch.manual_seed(1)
    m = stub.Block(32, 4).eval()
    x = torch.randn(1, 6, 32)
    y = m(x)
    x[:, -1] += 10
    assert torch.allclose(y[:, :-1], m(x)[:, :-1], atol=1e-5)
