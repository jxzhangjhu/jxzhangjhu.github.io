"""Tests for p25 · BatchNorm forward, gradients, and eval mode. Run: python run.py p25"""

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
from stubs import p25_batchnorm as stub  # noqa: E402

def test_matches_pytorch_in_train_mode():
    torch.manual_seed(0)
    mine, ref = stub.BatchNorm1dScratch(6), nn.BatchNorm1d(6)
    for _ in range(5):
        x = torch.randn(32, 6) * 3 + 1
        assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_running_stats_match():
    torch.manual_seed(0)
    mine, ref = stub.BatchNorm1dScratch(6), nn.BatchNorm1d(6)
    for _ in range(5):
        x = torch.randn(32, 6) * 3 + 1
        mine(x); ref(x)
    assert torch.allclose(mine.running_mean, ref.running_mean, atol=1e-5)
    assert torch.allclose(mine.running_var, ref.running_var, atol=1e-4), \
        "running_var differs: normalise with the biased variance, accumulate the unbiased one"


def test_eval_mode_uses_running_stats():
    torch.manual_seed(0)
    mine, ref = stub.BatchNorm1dScratch(6), nn.BatchNorm1d(6)
    for _ in range(5):
        x = torch.randn(32, 6) * 3 + 1
        mine(x); ref(x)
    mine.eval(); ref.eval()
    x = torch.randn(8, 6)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_input_and_affine_gradients_match_pytorch():
    torch.manual_seed(1)
    mine, ref = stub.BatchNorm1dScratch(6), nn.BatchNorm1d(6)
    with torch.no_grad():
        mine.gamma.copy_(ref.weight)
        mine.beta.copy_(ref.bias)
    x1 = torch.randn(16, 6, dtype=torch.float64, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    mine = mine.double()
    ref = ref.double()
    upstream = torch.randn_like(x1)
    mine(x1).backward(upstream)
    ref(x2).backward(upstream)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-9)
    assert torch.allclose(mine.gamma.grad, ref.weight.grad, atol=1e-9)
    assert torch.allclose(mine.beta.grad, ref.bias.grad, atol=1e-9)


def test_running_stats_are_buffers_not_parameters():
    m = stub.BatchNorm1dScratch(4)
    names = {n for n, _ in m.named_parameters()}
    assert "running_mean" not in names, "running stats must be buffers, not parameters"
    assert "running_mean" in dict(m.named_buffers())


def test_singleton_training_batch_is_rejected_cleanly():
    m = stub.BatchNorm1dScratch(4)
    try:
        m(torch.randn(1, 4))
    except ValueError:
        return
    raise AssertionError("a singleton training batch cannot estimate per-channel variance")
