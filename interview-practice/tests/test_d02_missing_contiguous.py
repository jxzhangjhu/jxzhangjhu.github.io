"""Tests for d02 · view() on a transposed tensor. Run: python -m pytest tests/test_d02_missing_contiguous.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d02_missing_contiguous as d  # noqa: E402

def test_merge_matches_a_manual_concat():
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).view(2, 3, 4, 5)
    got = d.merge_heads(x)
    want = torch.cat([x[:, h] for h in range(3)], dim=-1)
    assert got.shape == want.shape, f"expected {tuple(want.shape)}, got {tuple(got.shape)}"
    assert torch.equal(got, want), \
        "head values were not restored in per-token order"
