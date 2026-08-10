"""Tests for p21 · Generalised advantage estimation. Run: python run.py p21"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p21_gae as stub  # noqa: E402

def test_exists():
    assert hasattr(stub, "compute_gae"), "define compute_gae"


def test_matches_reference():
    """Compare against the validated reference. See reference.py for the exact API."""
    import inspect
    assert inspect.signature(stub.compute_gae) == inspect.signature(R.compute_gae), (
        "signature differs from the reference; match it so the harness can call yours")
