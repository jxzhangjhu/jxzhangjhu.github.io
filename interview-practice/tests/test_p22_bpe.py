"""Tests for p22 · Byte-pair encoding. Run: python run.py p22"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as R  # noqa: E402
from stubs import p22_bpe as stub  # noqa: E402

def test_matches_reference():
    text = "the cat sat on the mat, the cat sat again" * 6
    m_stub, m_ref = stub.bpe_train(text, 20), R.bpe_train(text, 20)
    assert list(m_stub.items()) == list(m_ref.items()), "merge order differs from the reference"
    assert stub.bpe_encode(text, m_stub) == R.bpe_encode(text, m_ref)


def test_compresses_and_round_trips():
    text = "abababab " * 40
    merges = stub.bpe_train(text, 10)
    ids = stub.bpe_encode(text, merges)
    raw = list(text.encode("utf-8"))
    assert len(ids) < len(raw) * 0.7, "ten merges on a repetitive string should compress a lot"

    table = {i: bytes([i]) for i in range(256)}
    for (a, b), new in merges.items():
        table[new] = table[a] + table[b]
    assert b"".join(table[i] for i in ids).decode("utf-8") == text
