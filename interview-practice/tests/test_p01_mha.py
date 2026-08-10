"""Tests for p01 · Causal multi-head attention. Run: python run.py p01"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402
from stubs import p01_mha as stub  # noqa: E402

def test_matches_pytorch_sdpa():
    torch.manual_seed(0)
    m = stub.CausalSelfAttention(64, 8).eval()
    x = torch.randn(2, 7, 64)
    got = m(x)

    B, T, C = x.shape
    q, k, v = m.qkv(x).split(C, dim=2)
    shape = (B, T, 8, 8)
    q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
    want = m.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True)
                  .transpose(1, 2).contiguous().view(B, T, C))
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_is_actually_causal():
    torch.manual_seed(0)
    m = stub.CausalSelfAttention(32, 4).eval()
    x = torch.randn(2, 6, 32)
    y = m(x)
    x2 = x.clone(); x2[:, -1, :] += 10.0
    assert torch.allclose(y[:, :-1], m(x2)[:, :-1], atol=1e-6), \
        "perturbing the last token changed earlier outputs: the mask leaks the future"


def test_scaling_uses_head_dim():
    torch.manual_seed(0)
    m = stub.CausalSelfAttention(64, 8).eval()
    x = torch.randn(1, 4, 64) * 3
    ref = R.CausalSelfAttention(64, 8).eval()
    ref.load_state_dict(m.state_dict())
    assert torch.allclose(m(x), ref(x), atol=1e-5), \
        "output differs from reference — check you divide by sqrt(d_head), not sqrt(d_model)"
