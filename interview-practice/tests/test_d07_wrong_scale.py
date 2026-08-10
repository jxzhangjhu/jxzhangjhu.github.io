"""Tests for d07 · Attention scaled by the wrong dimension. Run: python -m pytest tests/test_d07_wrong_scale.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d07_wrong_scale as d  # noqa: E402

def test_matches_pytorch_sdpa():
    torch.manual_seed(0)
    B, T, D, H = 2, 6, 32, 8
    q, k, v = (torch.randn(B, T, D) for _ in range(3))
    got = d.attention(q, k, v, H)

    shape = (B, T, H, D // H)
    qq, kk, vv = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
    want = F.scaled_dot_product_attention(qq, kk, vv)
    want = want.transpose(1, 2).contiguous().view(B, T, D)
    assert torch.allclose(got, want, atol=1e-5), \
        "scores are scaled by sqrt(d_model) instead of sqrt(d_head)"
