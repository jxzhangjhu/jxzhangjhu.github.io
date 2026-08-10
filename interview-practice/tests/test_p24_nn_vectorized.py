"""Tests for p24 · 1-NN in pure NumPy, no loops. Run: python run.py p24"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402,F401
from stubs import p24_nn_vectorized as stub  # noqa: E402

def test_matches_bruteforce():
    rng = np.random.default_rng(0)
    tr_x, te_x = rng.normal(size=(40, 5)), rng.normal(size=(17, 5))
    tr_y = rng.integers(0, 3, 40)
    want = np.array([tr_y[np.argmin(((tr_x - t) ** 2).sum(1))] for t in te_x])
    assert (stub.nearest_neighbour(tr_x, tr_y, te_x) == want).all()


def test_no_python_loop_over_test_points():
    import inspect
    src = inspect.getsource(stub.nearest_neighbour)
    body = "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))
    assert "for " not in body, "vectorise it: the point of the question is the matmul trick"


def test_scales_to_a_large_input():
    rng = np.random.default_rng(1)
    tr_x, te_x = rng.normal(size=(3000, 20)), rng.normal(size=(2000, 20))
    tr_y = rng.integers(0, 5, 3000)
    out = stub.nearest_neighbour(tr_x, tr_y, te_x)
    assert out.shape == (2000,)
