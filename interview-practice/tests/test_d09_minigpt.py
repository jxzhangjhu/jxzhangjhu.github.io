"""Tests for d09 · Debug miniGPT. Run: python -m pytest tests/test_d09_minigpt.py -q

Each test names a symptom the way an interviewer would describe it, and never the line.
Bugs in a real codebase mask each other, so fix them one at a time and re-run.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d09_minigpt as m  # noqa: E402


def build(seed=0):
    torch.manual_seed(seed)
    return m.MiniGPT(vocab_size=32, d_model=64, n_heads=4, n_layers=2).eval()


def correct_attention(attn, x):
    """What the attention module is supposed to compute."""
    B, T, C = x.shape
    q, k, v = attn.qkv(x).split(C, dim=2)
    q = q.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    k = k.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    v = v.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(attn.d_head)
    att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float("-inf"))
    y = F.softmax(att, dim=-1) @ v
    return attn.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class IndexProbe(nn.Module):
    """Wraps an embedding and records the indices it is asked for."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.seen = None

    def forward(self, idx):
        self.seen = idx.detach().clone()
        return self.inner(idx)


def test_bug1_positions_are_actually_indexed():
    """Every token must be given its own position.

    Feeding a constant index means all positions share one embedding, which leaves the
    model permutation-equivariant apart from the causal mask.
    """
    model = build()
    probe = IndexProbe(model.pos_emb)
    model.pos_emb = probe
    T = 6
    model(torch.randint(0, 32, (1, T)))
    assert probe.seen is not None, "the positional embedding was never used"
    want = torch.arange(T)
    assert torch.equal(probe.seen.flatten()[:T], want), (
        f"positional embedding was indexed with {probe.seen.flatten().tolist()} "
        f"instead of {want.tolist()}: every token is getting the same position")


def test_bug2_and_3_attention_matches_the_definition():
    """The attention module must equal softmax(mask(QK^T/sqrt(d)))V, heads merged properly.

    Two independent things break this and both are silent:
      - masking AFTER the softmax (rows then sum to less than 1, unevenly)
      - reshaping (B, n_heads, T, d_head) to (B, T, C) without transposing first,
        which interleaves the head and time axes without raising
    """
    model = build()
    attn = model.blocks[0].attn
    x = torch.randn(1, 6, 64)
    got, want = attn(x), correct_attention(attn, x)
    assert torch.allclose(got, want, atol=1e-5), (
        f"attention output is off by {(got - want).abs().max():.3e}. Two usual suspects: "
        "the mask is applied after the softmax, or the heads are merged without a transpose")


def test_bug4_training_actually_updates_parameters():
    """A step that never calls optimizer.step() leaves every parameter untouched."""
    torch.manual_seed(0)
    model = m.MiniGPT(vocab_size=32, d_model=64, n_heads=4, n_layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = torch.randint(0, 32, (2, 8))
    before = [p.detach().clone() for p in model.parameters()]
    m.train_step(model, opt, idx, idx)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), (
        "no parameter changed after a training step: the optimiser never stepped")


def test_end_to_end_can_memorise_one_sequence():
    """With everything fixed, the model overfits a single sequence. This is the real bar."""
    torch.manual_seed(0)
    model = m.MiniGPT(vocab_size=32, d_model=64, n_heads=4, n_layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, 32, (1, 16))
    losses = [m.train_step(model, opt, idx, idx) for _ in range(300)]
    assert losses[-1] < 0.1, (
        f"loss only reached {losses[-1]:.3f} after 300 steps on a single sequence — "
        "at least one bug is still there")


def test_followup_kv_cache_matches_uncached():
    """The stated correctness invariant: cached decode == full recompute."""
    model = build()
    idx = torch.randint(0, 32, (1, 10))
    try:
        full = model(idx)
        caches, outs = None, []
        for t in range(idx.shape[1]):
            logits, caches = m.forward_cached(model, idx[:, t:t + 1], caches)
            outs.append(logits)
        inc = torch.cat(outs, dim=1)
    except NotImplementedError:
        pytest.skip("forward_cached not implemented yet — this is the follow-up")
    assert torch.allclose(full, inc, atol=1e-4), (
        f"cached decode diverges by {(full - inc).abs().max():.2e}: "
        "check the positional index used for the new token")
