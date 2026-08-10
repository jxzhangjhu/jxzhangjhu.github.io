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


def test_fully_masked_batch_is_zero_not_nan():
    logits = torch.randn(4, 10, requires_grad=True)
    loss = stub.cross_entropy(logits, torch.full((4,), -100))
    assert torch.isfinite(loss) and abs(loss.detach().item()) < 1e-9, \
        f"a fully masked microbatch must contribute nothing, got {loss.detach().item()}"
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), \
        "the zero has to stay attached to the graph, or .backward() breaks"
