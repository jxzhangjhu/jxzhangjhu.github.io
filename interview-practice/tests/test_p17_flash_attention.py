"""Tests for p17 · Tiled FlashAttention forward. Run: python run.py p17"""

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
from stubs import p17_flash_attention as stub  # noqa: E402

def test_matches_sdpa_and_is_block_size_invariant():
    torch.manual_seed(0)
    q = torch.randn(2, 2, 23, 8, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    for bq, bkv in ((1, 1), (7, 5), (16, 16), (32, 9)):
        got, lse = stub.flash_attention_forward(q, k, v, bq, bkv, causal=True)
        assert lse.shape == (2, 2, 23)
        assert torch.allclose(got, want, atol=1e-10), (bq, bkv)


@pytest.mark.parametrize("dtype,atol", [
    (torch.float16, 2e-3),
    (torch.bfloat16, 2e-2),
])
def test_mixed_precision_accumulates_in_float32_and_casts_output(dtype, atol):
    torch.manual_seed(1)
    q = torch.randn(1, 2, 19, 8).to(dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    want = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True
    ).to(dtype)
    got, lse = stub.flash_attention_forward(q, k, v, 7, 5, causal=True)
    assert got.dtype == dtype
    assert lse.dtype == torch.float32
    assert torch.allclose(got, want, atol=atol, rtol=atol)
