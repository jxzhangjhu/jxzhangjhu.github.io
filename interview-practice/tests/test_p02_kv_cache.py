"""Tests for p02 · KV cache and incremental decode. Run: python run.py p02"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402
from stubs import p02_kv_cache as stub  # noqa: E402

def test_cached_decode_equals_full_recompute():
    torch.manual_seed(0)
    m = stub.CachedAttention(64, 8).eval()
    x = torch.randn(1, 6, 64)
    full = m(x)[0]

    out, cache = [], None
    for t in range(x.shape[1]):
        y, cache = m(x[:, t:t + 1], cache)
        out.append(y)
    inc = torch.cat(out, dim=1)
    assert torch.allclose(full, inc, atol=1e-5), \
        f"incremental decode diverges from teacher forcing by {(full - inc).abs().max():.2e}"


def test_cache_grows_by_one_per_step():
    torch.manual_seed(0)
    m = stub.CachedAttention(32, 4).eval()
    cache = None
    for t in range(1, 5):
        _, cache = m(torch.randn(1, 1, 32), cache)
        assert cache[0].shape[-2] == t, f"after {t} steps the cache holds {cache[0].shape[-2]}"
