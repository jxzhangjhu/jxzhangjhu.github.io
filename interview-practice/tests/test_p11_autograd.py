"""Tests for p11 · A 40-line autograd. Run: python run.py p11"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p11_autograd as stub  # noqa: E402

def test_exists():
    assert hasattr(stub, "Value"), "define Value"


def test_matches_reference():
    """Compare against the validated reference. See reference.py for the exact API."""
    import inspect
    assert inspect.signature(stub.Value) == inspect.signature(R.Value), (
        "signature differs from the reference; match it so the harness can call yours")
