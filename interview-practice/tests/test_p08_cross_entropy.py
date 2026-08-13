"""Tests for p08 · Cross entropy with log-sum-exp. Run: python run.py p08"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402
from stubs import p08_cross_entropy as stub  # noqa: E402

def test_matches_f_cross_entropy():
    torch.manual_seed(0)
    logits, targets = torch.randn(16, 10), torch.randint(0, 10, (16,))
    assert torch.allclose(stub.cross_entropy(logits, targets),
                          F.cross_entropy(logits, targets), atol=1e-6)


def test_respects_ignore_index():
    torch.manual_seed(0)
    logits, targets = torch.randn(16, 10), torch.randint(0, 10, (16,))
    targets[:5] = -100
    assert torch.allclose(stub.cross_entropy(logits, targets),
                          F.cross_entropy(logits, targets, ignore_index=-100), atol=1e-6)


def test_no_overflow_on_huge_logits():
    logits = torch.tensor([[1e4, 2e4, 3e4]])
    loss = stub.cross_entropy(logits, torch.tensor([2]))
    assert torch.isfinite(loss), "overflowed: subtract the row max before exponentiating"


def test_fully_masked_batch_is_zero_and_stays_on_graph():
    logits = torch.randn(4, 7, requires_grad=True)
    targets = torch.full((4,), -100)
    loss = stub.cross_entropy(logits, targets)
    assert loss.shape == () and loss.detach().item() == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.equal(logits.grad, torch.zeros_like(logits))
