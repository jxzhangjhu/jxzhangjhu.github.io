"""Tests for p20 · DPO loss. Run: python run.py p20"""

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
from stubs import p20_dpo_loss as stub  # noqa: E402

def test_reference_policy_is_log_two_and_ordering_is_right():
    z = torch.zeros(4)
    at_ref = stub.dpo_loss(z, z, z, z)
    assert torch.allclose(at_ref, torch.tensor(math.log(2.0)), atol=1e-6)
    better = stub.dpo_loss(torch.full((4,), 2.0), z, z, z)
    worse = stub.dpo_loss(z, torch.full((4,), 2.0), z, z)
    assert better < at_ref < worse


def test_matches_reference():
    torch.manual_seed(0)
    xs = [torch.randn(9) for _ in range(4)]
    assert torch.allclose(stub.dpo_loss(*xs, beta=0.3), R.dpo_loss(*xs, beta=0.3))
