"""Tests for p04 · Rotary position embeddings. Run: python run.py p04"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402
from stubs import p04_rope as stub  # noqa: E402

def test_matches_reference():
    cos, sin = stub.rope_cache(8, 16)
    rc, rs = R.rope_cache(8, 16)
    assert torch.allclose(cos, rc, atol=1e-6) and torch.allclose(sin, rs, atol=1e-6)
    x = torch.randn(2, 3, 8, 16)
    assert torch.allclose(stub.apply_rope(x, cos, sin), R.apply_rope(x, rc, rs), atol=1e-6)


def test_logits_depend_only_on_relative_offset():
    torch.manual_seed(0)
    d, T = 16, 12
    cos, sin = stub.rope_cache(T, d)
    q, k = torch.randn(1, 1, T, d), torch.randn(1, 1, T, d)
    qr, kr = stub.apply_rope(q, cos, sin), stub.apply_rope(k, cos, sin)
    # same offset at two different absolute positions must give the same logit
    a = (qr[0, 0, 5] * kr[0, 0, 3]).sum()
    q2 = torch.roll(q, shifts=2, dims=2); k2 = torch.roll(k, shifts=2, dims=2)
    qr2, kr2 = stub.apply_rope(q2, cos, sin), stub.apply_rope(k2, cos, sin)
    b = (qr2[0, 0, 7] * kr2[0, 0, 5]).sum()
    assert torch.allclose(a, b, atol=1e-4), f"offset-2 logits differ: {a} vs {b}"
