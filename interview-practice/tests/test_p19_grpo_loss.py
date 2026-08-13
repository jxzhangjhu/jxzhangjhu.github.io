"""Tests for p19 · GRPO objective. Run: python run.py p19"""

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


def test_singleton_groups_are_finite_and_carry_zero_relative_signal():
    logp, _, ref, rewards, mask = _batch(B=4)
    loss = stub.grpo_loss(logp, logp.clone(), ref, rewards, mask, group_size=1, beta=0.0)
    assert torch.isfinite(loss) and abs(loss.detach().item()) < 1e-6


def test_ratio_is_one_on_policy():
    logp, _, ref, rewards, mask = _batch()
    loss = stub.grpo_loss(logp, logp.clone(), ref, rewards, mask, group_size=4, beta=0.0)
    r = rewards.view(-1, 4)
    adv = ((r - r.mean(1, keepdim=True))
           / (r.std(1, keepdim=True, correction=0) + 1e-4)).reshape(-1)
    assert torch.allclose(loss, -adv.mean(), atol=1e-3), \
        "at ratio 1 the clipped surrogate must equal the advantage"


def test_completions_are_weighted_equally_despite_different_lengths():
    z = torch.zeros(2, 4)
    rewards = torch.tensor([0.0, 1.0])
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    loss = stub.grpo_loss(z, z, z, rewards, mask, group_size=2, beta=0.0)
    assert abs(loss.detach().item()) < 1e-6, (
        "this exercise implements original GRPO: average within each completion, then "
        "average completions. A global-token reduction is a distinct length-weighting variant"
    )


def test_kl_term_is_non_negative():
    logp, old, ref, rewards, mask = _batch()
    with_kl = stub.grpo_loss(logp, old, ref, rewards, mask, 4, beta=1.0)
    without = stub.grpo_loss(logp, old, ref, rewards, mask, 4, beta=0.0)
    assert float(with_kl) >= float(without) - 1e-6, \
        "the k3 estimator is non-negative per sample, so adding it cannot lower the loss"
