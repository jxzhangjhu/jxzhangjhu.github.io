"""Tests for p18 · LoRA with a lossless merge. Run: python run.py p18"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402
from stubs import p18_lora as stub  # noqa: E402

def test_identity_at_init():
    torch.manual_seed(0)
    base = nn.Linear(32, 16, bias=False)
    lora = stub.LoRALinear(base, r=4)
    x = torch.randn(3, 32)
    assert torch.allclose(lora(x), base(x), atol=1e-6), \
        "not the identity at step 0 — B must start at zero"


def test_merge_is_lossless():
    torch.manual_seed(0)
    base = nn.Linear(32, 16, bias=False)
    lora = stub.LoRALinear(base, r=4)
    with torch.no_grad():          # pretend we trained
        for p in lora.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
    x = torch.randn(3, 32)
    merged = F.linear(x, lora.merged_weight())
    assert torch.allclose(lora(x), merged, atol=1e-5), \
        "merged weight does not reproduce the adapter path"


def test_base_is_frozen():
    base = nn.Linear(8, 8, bias=False)
    lora = stub.LoRALinear(base, r=2)
    assert not lora.base.weight.requires_grad, "the base must be frozen — that is the memory win"
