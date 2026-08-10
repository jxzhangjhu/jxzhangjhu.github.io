"""Tests for d03 · top-p drops the crossing token. Run: python -m pytest tests/test_d03_top_p_off_by_one.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d03_top_p_off_by_one as d  # noqa: E402

def test_the_crossing_token_survives():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    kept = torch.isfinite(d.nucleus(logits, 0.9)).nonzero().flatten().tolist()
    assert kept == [0, 1, 2], \
        f"p=0.9 on [.5,.3,.15,.05] should keep three tokens, kept {kept}: " \
        "use the exclusive cumulative sum, cum - probs"


def test_a_single_dominant_token_is_never_dropped():
    logits = torch.log(torch.tensor([0.99, 0.005, 0.005]))
    kept = torch.isfinite(d.nucleus(logits, 0.5)).nonzero().flatten().tolist()
    assert kept == [0], f"expected only the argmax, kept {kept}"
