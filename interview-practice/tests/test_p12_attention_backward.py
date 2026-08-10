"""Tests for p12 · Attention backward by hand. Run: python run.py p12"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p12_attention_backward as stub  # noqa: E402

def test_exists():
    assert hasattr(stub, "attention_backward"), "define attention_backward"


def test_matches_reference():
    """Compare against the validated reference. See reference.py for the exact API."""
    import inspect
    assert inspect.signature(stub.attention_backward) == inspect.signature(R.attention_backward), (
        "signature differs from the reference; match it so the harness can call yours")
