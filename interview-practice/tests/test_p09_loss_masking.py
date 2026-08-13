"""Tests for p09 · SFT loss masking and packing. Run: python run.py p09"""

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
from stubs import p09_loss_masking as stub  # noqa: E402

def test_variable_prompt_lengths_and_padding_are_masked():
    ids = torch.arange(32).reshape(4, 8)
    mask = torch.ones_like(ids)
    mask[2, 6:] = 0
    mask[3, 5:] = 0
    lens = [1, 3, 4, 2]
    got = stub.build_sft_labels(ids, lens, mask)
    want = ids.clone()
    for i, n in enumerate(lens):
        want[i, :n] = -100
    want[mask == 0] = -100
    assert torch.equal(got, want)
    assert torch.equal(ids, torch.arange(32).reshape(4, 8)), "do not mutate input_ids"


def test_packed_labels_keep_only_response_tokens():
    ids = torch.tensor([[10, 11, 12, 20, 21, 0]])
    response = torch.tensor([[0, 1, 1, 0, 1, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 0]])
    got = stub.build_packed_sft_labels(ids, response, attention)
    assert torch.equal(got, torch.tensor([[-100, 11, 12, -100, 21, -100]]))


def test_packing_resets_positions_and_blocks_cross_document_attention():
    segments = torch.tensor([[0, 0, 0, 1, 1, -1]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 0]])
    positions, allowed = stub.build_packed_attention(segments, attention)
    assert torch.equal(positions, torch.tensor([[0, 1, 2, 0, 1, 0]]))
    assert allowed.dtype == torch.bool and allowed.shape == (1, 6, 6)
    assert allowed[0, 2, :3].all()             # causal history inside document zero
    assert not allowed[0, 3, :3].any()         # document one cannot see document zero
    assert allowed[0, 4, 3:5].all()            # legal history inside document one
    assert not allowed[0, 5].any() and not allowed[0, :, 5].any()  # padding

    # Segment labels may be reused later; boundaries are contiguous runs, not label values.
    reused = torch.tensor([[0, 0, 1, 1, 0, 0]])
    pos2, allowed2 = stub.build_packed_attention(reused, torch.ones_like(reused))
    assert torch.equal(pos2, torch.tensor([[0, 1, 0, 1, 0, 1]]))
    assert not allowed2[0, 4, :4].any()
