"""Tests for p26 · Filter bad human annotations. Run: python run.py p26"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p26_data_filtering as stub  # noqa: E402

def test_exists():
    assert hasattr(stub, "filter_annotations"), "define filter_annotations"


def test_matches_reference():
    """Compare against the validated reference. See reference.py for the exact API."""
    import inspect
    assert inspect.signature(stub.filter_annotations) == inspect.signature(R.filter_annotations), (
        "signature differs from the reference; match it so the harness can call yours")
