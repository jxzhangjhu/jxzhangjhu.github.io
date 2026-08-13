"""Tests for p05 · RMSNorm. Run: python run.py p05"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402
from stubs import p05_rmsnorm as stub  # noqa: E402

def test_matches_torch_rms_norm():
    torch.manual_seed(0)
    m = stub.RMSNorm(32)
    with torch.no_grad():
        m.weight.copy_(torch.randn(32))
    x = torch.randn(4, 7, 32)
    want = F.rms_norm(x, (32,), m.weight, eps=1e-6)
    assert torch.allclose(m(x), want, atol=1e-5)


def test_reduction_runs_in_fp32_under_bf16():
    m = stub.RMSNorm(64).to(torch.bfloat16)
    x = (torch.randn(2, 3, 64) * 100).to(torch.bfloat16)
    y = m(x)
    assert y.dtype == x.dtype, "output dtype should follow the input (cast back at the end)"
    # a bf16 reduction over 64 squared values of magnitude ~1e4 can drift substantially
    want = F.rms_norm(x.float(), (64,), m.weight.float(), eps=1e-6)
    err = (y.float() - want).abs().max() / want.abs().max()
    assert err < 5e-3, f"relative error {err:.2%}: is the reduction running in fp32?"


def test_float64_is_not_silently_demoted():
    m = stub.RMSNorm(32).double()
    x = torch.randn(2, 3, 32, dtype=torch.float64)
    want = F.rms_norm(x, (32,), m.weight, eps=1e-6)
    assert torch.allclose(m(x), want, atol=1e-12, rtol=1e-12)
