"""Tests for p17 · Tiled FlashAttention forward. Run: python run.py p17"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p17_flash_attention as stub  # noqa: E402

def test_exists():
    assert hasattr(stub, "flash_attention_forward"), "define flash_attention_forward"


def test_matches_reference():
    """Compare against the validated reference. See reference.py for the exact API."""
    import inspect
    assert inspect.signature(stub.flash_attention_forward) == inspect.signature(R.flash_attention_forward), (
        "signature differs from the reference; match it so the harness can call yours")
