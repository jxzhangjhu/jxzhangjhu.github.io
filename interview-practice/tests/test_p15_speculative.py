"""Tests for p15 · Speculative decoding accept/reject. Run: python run.py p15"""

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
from stubs import p15_speculative as stub  # noqa: E402

def test_accepts_when_target_dominates_draft():
    p = torch.tensor([0.7, 0.3])
    q = torch.tensor([0.2, 0.8])
    assert stub.speculative_accept(p, q, token=0, u=0.99) == (0, True)


def test_rejection_samples_the_positive_residual():
    p = torch.tensor([0.7, 0.3])
    q = torch.tensor([0.2, 0.8])
    token, accepted = stub.speculative_accept(p, q, token=1, u=0.99)
    assert not accepted and token == 0, "the residual distribution has all its mass on token 0"


def test_zero_acceptance_probability_rejects_even_when_u_is_zero():
    p = torch.tensor([0.0, 1.0])
    q = torch.tensor([0.5, 0.5])
    assert stub.speculative_accept(p, q, token=0, u=0.0) == (1, False), (
        "acceptance compares u < p/q strictly; <= incorrectly accepts a zero-probability token"
    )


def test_sampled_draft_token_requires_positive_q():
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.0, 1.0])
    with pytest.raises((ValueError, AssertionError)):
        stub.speculative_accept(p, q, token=0, u=0.5)
