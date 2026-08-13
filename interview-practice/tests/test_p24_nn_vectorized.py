"""Tests for p24 · 1-NN in pure NumPy, no loops. Run: python run.py p24"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
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
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(stub.nearest_neighbour))
    loop_nodes = (ast.For, ast.AsyncFor, ast.ListComp, ast.SetComp,
                  ast.DictComp, ast.GeneratorExp)
    assert not any(isinstance(node, loop_nodes) for node in ast.walk(tree)), (
        "vectorise pairwise differences instead of looping over test points"
    )


def test_scales_to_a_large_input():
    rng = np.random.default_rng(1)
    tr_x, te_x = rng.normal(size=(500, 20)), rng.normal(size=(300, 20))
    tr_y = rng.integers(0, 5, 500)
    out = stub.nearest_neighbour(tr_x, tr_y, te_x)
    assert out.shape == (300,)


def test_float32_nearby_large_coordinates_do_not_cancel():
    tr_x = np.array([[100.02], [100.001]], dtype=np.float32)
    tr_y = np.array([0, 1])
    te_x = np.array([[100.0]], dtype=np.float32)
    assert stub.nearest_neighbour(tr_x, tr_y, te_x).item() == 1, (
        "expanded squared norms cancel in float32 here; square direct differences instead"
    )
