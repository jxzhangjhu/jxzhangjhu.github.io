"""Tests for p21 · Generalised advantage estimation. Run: python run.py p21"""

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
from stubs import p21_gae as stub  # noqa: E402

def test_lambda_limits_and_reference_match():
    rewards = torch.tensor([1.0, 0.0, 2.0, 1.0], dtype=torch.float64)
    values = torch.tensor([0.5, 0.4, 0.3, 0.2], dtype=torch.float64)
    for lam in (0.0, 0.37, 1.0):
        got = stub.compute_gae(rewards, values, gamma=0.99, lam=lam)
        want = R.compute_gae(rewards, values, gamma=0.99, lam=lam)
        assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(got, want))
    adv0, _ = stub.compute_gae(rewards, values, gamma=0.99, lam=0.0)
    td = rewards + 0.99 * torch.cat([values[1:], values.new_zeros(1)]) - values
    assert torch.allclose(adv0, td)
