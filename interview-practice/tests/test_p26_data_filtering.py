"""Tests for p26 · Filter bad human annotations. Run: python run.py p26"""

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
from stubs import p26_data_filtering as stub  # noqa: E402

def test_flags_adversarial_but_not_sparse_annotator():
    ann = ["good1", "good2", "bad", "sparse"]
    labels = [
        ["a", "a", "b", None],
        ["a", "a", "b", None],
        ["b", "b", "a", "b"],
        ["a", "a", "b", None],
        ["b", "b", "a", None],
    ]
    got = stub.filter_annotations(labels, ann)
    want = R.filter_annotations(labels, ann)
    assert got == want
    clean, flagged = got
    assert flagged == {"bad"} and len(clean) == len(labels)


def test_empty_rows_are_skipped_safely():
    clean, flagged = stub.filter_annotations([[None, None], ["x", None]], ["a", "b"],
                                                min_items=2)
    assert clean == [(1, "x")] and flagged == set()
