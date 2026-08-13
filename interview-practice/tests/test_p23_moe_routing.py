"""Tests for p23 · Top-1 MoE routing with capacity. Run: python run.py p23"""

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
from stubs import p23_moe_routing as stub  # noqa: E402

def test_capacity_keeps_the_most_confident_tokens():
    logits = torch.tensor([
        [3.0, 0.0], [1.0, 0.0], [5.0, 0.0], [0.0, 4.0], [0.0, 2.0]
    ])
    expert, gate, kept = stub.top1_route(logits, capacity=2)
    assert torch.equal(expert, torch.tensor([0, 0, 0, 1, 1]))
    assert kept.tolist() == [True, False, True, True, True]
    assert torch.all((gate >= 0) & (gate <= 1))


def test_balance_loss_prefers_uniform_routing():
    torch.manual_seed(0)
    balanced = torch.randn(400, 4) * 0.01
    skewed = torch.full((400, 4), -5.0)
    skewed[:, 0] = 5.0
    assert stub.load_balancing_loss(balanced) < stub.load_balancing_loss(skewed)
    assert torch.allclose(stub.load_balancing_loss(balanced),
                          R.load_balancing_loss(balanced), atol=1e-6)
