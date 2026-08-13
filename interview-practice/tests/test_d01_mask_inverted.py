"""Tests for d01 · Inverted attention mask. Run: python -m pytest tests/test_d01_mask_inverted.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d01_mask_inverted as d  # noqa: E402

def test_future_is_masked_and_past_is_not():
    torch.manual_seed(0)
    q, k = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    s = d.causal_scores(q, k)[0]
    assert torch.isinf(s[0, 1:]).all(), \
        "query 0 must not see any later position"
    assert torch.isfinite(s[3, :4]).all(), \
        "query 3 must see positions 0..3"
    w = F.softmax(s, dim=-1)
    assert torch.allclose(w[0, 0], torch.tensor(1.0)), \
        "row 0 should put all its weight on itself"
