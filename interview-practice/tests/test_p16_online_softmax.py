"""Tests for p16 · Streaming softmax. Run: python run.py p16"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p16_online_softmax as stub  # noqa: E402

def test_exists():
    assert hasattr(stub, "online_softmax_weighted_sum"), "define online_softmax_weighted_sum"


def test_matches_reference():
    """Compare against the validated reference. See reference.py for the exact API."""
    import inspect
    assert inspect.signature(stub.online_softmax_weighted_sum) == inspect.signature(R.online_softmax_weighted_sum), (
        "signature differs from the reference; match it so the harness can call yours")
