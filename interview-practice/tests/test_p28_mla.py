"""Tests for p28 · Multi-head latent attention and compressed cache. Run: python run.py p28"""

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
from stubs import p28_mla as stub  # noqa: E402

def test_cached_decode_matches_full_recompute():
    torch.manual_seed(0)
    m = stub.MultiHeadLatentAttention(64, 4, kv_rank=12, rope_dim=8).eval()
    x = torch.randn(1, 7, 64)
    full = m(x)
    cache, pieces = {}, []
    for t in range(x.shape[1]):
        pieces.append(m(x[:, t:t + 1], cache))
    step = torch.cat(pieces, dim=1)
    assert torch.allclose(full, step, atol=1e-5), (
        f"cached decode differs from full recompute by {(full - step).abs().max():.2e}")


def test_cache_is_compressed():
    torch.manual_seed(1)
    m = stub.MultiHeadLatentAttention(64, 4, kv_rank=12, rope_dim=8).eval()
    cache = {}
    m(torch.randn(2, 3, 64), cache)
    assert set(cache) == {"c", "k_rope"}
    assert cache["c"].shape == (2, 3, 12)
    assert cache["k_rope"].shape == (2, 1, 3, 8)
    cached = cache["c"].shape[-1] + cache["k_rope"].shape[-1]
    mha = 2 * 4 * (64 // 4)
    assert cached < mha, f"cached {cached} values/token; plain MHA needs {mha}"


def test_matches_validated_reference():
    torch.manual_seed(2)
    mine = stub.MultiHeadLatentAttention(64, 4, 12, 8).eval()
    ref = R.MultiHeadLatentAttention(64, 4, 12, 8).eval()
    ref.load_state_dict(mine.state_dict())
    x = torch.randn(2, 5, 64)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)
