"""Tests for p15 · Speculative decoding accept/reject. Run: python run.py p15"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p15_speculative as stub  # noqa: E402

def test_exists():
    assert hasattr(stub, "speculative_accept"), "define speculative_accept"


def test_matches_reference():
    """Compare against the validated reference. See reference.py for the exact API."""
    import inspect
    assert inspect.signature(stub.speculative_accept) == inspect.signature(R.speculative_accept), (
        "signature differs from the reference; match it so the harness can call yours")
