"""Tests for p14 · Temperature, top-k, top-p. Run: python run.py p14"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402
from stubs import p14_sampling as stub  # noqa: E402

def test_temperature_zero_is_greedy():
    logits = torch.tensor([0.1, 5.0, 0.3, 2.0])
    assert stub.sample_next(logits, temperature=0.0) == 1


def test_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([5.0, 4.0, 0.1, 0.0, -1.0])
    got = {stub.sample_next(logits, temperature=1.0, top_k=2) for _ in range(300)}
    assert got <= {0, 1}, f"top-k=2 sampled outside the top two: {got}"


def test_top_k_keeps_exactly_k_slots_when_logits_tie():
    torch.manual_seed(1)
    logits = torch.zeros(4)
    got = {stub.sample_next(logits, top_k=2) for _ in range(300)}
    assert len(got) == 2, f"top-k=2 should retain exactly two tied slots, sampled {got}"


def test_top_p_keeps_the_crossing_token():
    # probs approx [0.5, 0.3, 0.15, 0.05]; p=0.9 must keep exactly the first three
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    got = {stub.sample_next(logits, temperature=1.0, top_p=0.9) for _ in range(600)}
    assert got <= {0, 1, 2} and 2 in got, \
        f"expected the nucleus to be exactly {{0,1,2}}, sampled {got} (off-by-one in the shift?)"


def test_top_p_one_preserves_finite_logit_support(monkeypatch):
    captured = {}

    def capture(probs, num_samples, generator=None):
        captured["probs"] = probs
        return torch.tensor([0], device=probs.device)

    monkeypatch.setattr(torch, "multinomial", capture)
    logits = torch.tensor([700.0, 0.0, -700.0], dtype=torch.float64)
    for top_p in (1.0, 1.5):
        stub.sample_next(logits, top_p=top_p)
        assert captured["probs"][1] > 0, (
            "top_p>=1 must bypass nucleus filtering; cumulative sums can round to one "
            "early for extreme finite logits"
        )
