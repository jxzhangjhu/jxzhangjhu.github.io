"""Tests for p27 · Spinning light source → Cauchy. Run: python run.py p27"""

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
from stubs import p27_cauchy_simulation as stub  # noqa: E402

def test_pdf_is_standard_cauchy():
    x = np.linspace(-4, 4, 17)
    assert np.allclose(stub.cauchy_pdf(x), 1 / (np.pi * (1 + x ** 2)), atol=1e-12)


def test_histogram_matches_pdf_after_truncation_correction():
    x = stub.light_source_samples(400_000)
    L = 5.0
    edges = np.linspace(-L, L, 41)
    hist, _ = np.histogram(x, bins=edges, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    in_range = 2 * np.arctan(L) / np.pi           # only 87.4% of the mass is in view
    err = np.abs(hist - stub.cauchy_pdf(centres) / in_range).max()
    assert err < 0.01, (
        f"max density error {err:.3f}. If it is around 0.05, you compared against the "
        "untruncated PDF — density=True normalises over the plotted range only")


def test_median_is_stable_while_fixed_mean_diagnostics_are_not():
    x = stub.light_source_samples(400_000)
    assert abs(np.median(x)) < 0.02, "the median should estimate the location parameter"
    means = [abs(stub.light_source_samples(n, seed=s).mean())
             for s, n in enumerate([10_000, 100_000, 400_000])]
    assert min(means) > 0.5, (
        f"sample means {means} were unexpectedly small for this fixed diagnostic")
