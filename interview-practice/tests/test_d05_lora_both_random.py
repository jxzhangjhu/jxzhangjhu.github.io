"""Tests for d05 · LoRA that is not identity at init. Run: python -m pytest tests/test_d05_lora_both_random.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d05_lora_both_random as d  # noqa: E402

def test_adapter_is_identity_before_training():
    torch.manual_seed(0)
    base = nn.Linear(16, 32)
    x = torch.randn(4, 16)
    want = base(x).clone()
    got = d.LoRALinear(base, r=4, alpha=8)(x)
    assert torch.allclose(got, want, atol=1e-7), \
        "a fresh adapter changed the base model's output"


def test_the_adapter_is_not_dead_at_init():
    torch.manual_seed(0)
    m = d.LoRALinear(nn.Linear(16, 32), r=4, alpha=8)
    m(torch.randn(4, 16)).sum().backward()
    assert m.B.grad is not None and m.B.grad.abs().sum() > 0, \
        "the adapter receives no useful first-step gradient"
