"""Tests for p19 · GRPO objective. Run: python run.py p19"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p19_grpo_loss as stub  # noqa: E402

def _batch(B=8, L=5, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, L), torch.randn(B, L), torch.randn(B, L),
            torch.rand(B), torch.ones(B, L))


def test_tied_group_gives_no_gradient_and_no_nan():
    logp, _, ref, _, mask = _batch()
    rewards = torch.ones(8)                       # every completion scores the same
    loss = stub.grpo_loss(logp, logp.clone(), ref, rewards, mask, group_size=4, beta=0.0)
    assert torch.isfinite(loss), "a tied group divided by std=0 — add an epsilon"
    assert abs(float(loss)) < 1e-4, \
        f"advantage should be identically zero for a tied group, got loss {float(loss)}"


def test_singleton_group_is_finite():
    logp, _, ref, rewards, mask = _batch()
    loss = stub.grpo_loss(logp, logp.clone(), ref, rewards, mask, group_size=1, beta=0.0)
    assert torch.isfinite(loss), \
        "group_size=1 has no relative signal; the unbiased std is NaN before the epsilon lands"


def test_ratio_is_one_on_policy():
    logp, _, ref, rewards, mask = _batch()
    loss = stub.grpo_loss(logp, logp.clone(), ref, rewards, mask, group_size=4, beta=0.0)
    r = rewards.view(-1, 4)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).reshape(-1)
    assert torch.allclose(loss, -adv.mean(), atol=1e-3), \
        "at ratio 1 the clipped surrogate must equal the advantage"


def test_kl_term_is_non_negative():
    logp, old, ref, rewards, mask = _batch()
    with_kl = stub.grpo_loss(logp, old, ref, rewards, mask, 4, beta=1.0)
    without = stub.grpo_loss(logp, old, ref, rewards, mask, 4, beta=0.0)
    assert float(with_kl) >= float(without) - 1e-6, \
        "the k3 estimator is non-negative per sample, so adding it cannot lower the loss"
