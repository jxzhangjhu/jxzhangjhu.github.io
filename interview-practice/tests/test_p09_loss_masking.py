"""Tests for p09 · SFT loss masking and packing. Run: python run.py p09"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p09_loss_masking as stub  # noqa: E402

def test_exists():
    assert hasattr(stub, "build_sft_labels"), "define build_sft_labels"


def test_matches_reference():
    """Compare against the validated reference. See reference.py for the exact API."""
    import inspect
    assert inspect.signature(stub.build_sft_labels) == inspect.signature(R.build_sft_labels), (
        "signature differs from the reference; match it so the harness can call yours")
