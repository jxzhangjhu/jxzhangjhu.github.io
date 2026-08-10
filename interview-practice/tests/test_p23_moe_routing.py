"""Tests for p23 · Top-1 MoE routing with capacity. Run: python run.py p23"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p23_moe_routing as stub  # noqa: E402

def test_exists():
    assert hasattr(stub, "top1_route"), "define top1_route"


def test_matches_reference():
    """Compare against the validated reference. See reference.py for the exact API."""
    import inspect
    assert inspect.signature(stub.top1_route) == inspect.signature(R.top1_route), (
        "signature differs from the reference; match it so the harness can call yours")
