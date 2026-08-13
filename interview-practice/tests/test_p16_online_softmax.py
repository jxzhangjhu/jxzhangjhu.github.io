"""Tests for p16 · Streaming softmax. Run: python run.py p16"""

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
from stubs import p16_online_softmax as stub  # noqa: E402

def test_matches_naive_for_large_scores_and_many_block_sizes():
    torch.manual_seed(0)
    scores = torch.randn(37, dtype=torch.float64) * 20
    values = torch.randn(37, 5, dtype=torch.float64)
    want = F.softmax(scores, dim=0) @ values
    for block in (1, 4, 11, 64):
        got = stub.online_softmax_weighted_sum(scores, values, block)
        assert torch.allclose(got, want, atol=1e-10), block
