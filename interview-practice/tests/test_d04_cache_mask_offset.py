"""Tests for d04 · Cached decode with an unshifted mask. Run: python -m pytest tests/test_d04_cache_mask_offset.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d04_cache_mask_offset as d  # noqa: E402

def test_each_new_query_sees_the_whole_cache_up_to_itself():
    m = d.cached_causal_mask(2, 5)
    assert m.shape == (2, 5)
    assert m[0].tolist() == [True, True, True, True, False], \
        f"the first new query is at absolute position 3, so it sees keys 0..3; got {m[0].tolist()}"
    assert m[1].tolist() == [True] * 5, \
        f"the second new query sees the whole cache; got {m[1].tolist()}"


def test_prefill_is_the_plain_triangle():
    m = d.cached_causal_mask(4, 4)
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool))), \
        "with no cache the shift must be zero"
