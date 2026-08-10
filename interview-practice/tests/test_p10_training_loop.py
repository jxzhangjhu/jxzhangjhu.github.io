"""Tests for p10 · Overfit a tiny batch. Run: python run.py p10"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402
from stubs import p10_training_loop as stub  # noqa: E402

def test_reaches_near_zero_loss():
    loss, acc = stub.overfit_tiny()
    assert isinstance(loss, float), "return floats, not tensors still attached to the graph"
    assert acc == 1.0, f"accuracy {acc}: ten fixed examples should be memorisable exactly"
    assert loss < 1e-3, f"final loss {loss:.4f}: not actually converged"
