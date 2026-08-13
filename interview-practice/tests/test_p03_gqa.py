"""Tests for p03 · Grouped-query attention. Run: python run.py p03"""

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
from stubs import p03_gqa as stub  # noqa: E402

def test_matches_reference():
    torch.manual_seed(0)
    mine = stub.GroupedQueryAttention(64, 8, 2).eval()
    ref = R.GroupedQueryAttention(64, 8, 2).eval()
    ref.load_state_dict(mine.state_dict())
    x = torch.randn(2, 7, 64)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_degenerates_to_mha():
    torch.manual_seed(0)
    m = stub.GroupedQueryAttention(64, 8, 8).eval()
    y = m(torch.randn(1, 5, 64))
    assert y.shape == (1, 5, 64), "n_kv_heads == n_heads should just be MHA"


def test_cached_decode_matches_full():
    torch.manual_seed(0)
    m = stub.GroupedQueryAttention(32, 4, 2).eval()
    x = torch.randn(1, 6, 32)
    full = m(x)
    cache, outs = {}, []
    for t in range(6):
        outs.append(m(x[:, t:t + 1], cache))
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-5)
