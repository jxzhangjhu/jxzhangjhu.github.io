"""Symptom-oriented tests for d09. Fix one invariant at a time and re-run."""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d09_minigpt as m  # noqa: E402


def build(seed=0):
    torch.manual_seed(seed)
    return m.MiniGPT(vocab_size=32, d_model=64, n_heads=4, n_layers=2).eval()


def correct_attention(attn, x):
    """Oracle built from PyTorch's independently tested attention primitive."""
    B, T, C = x.shape
    q, k, v = attn.qkv(x).split(C, dim=2)
    q = q.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    k = k.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    v = v.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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


def test_each_token_receives_its_absolute_position():
    model = build()
    probe = IndexProbe(model.pos_emb)
    model.pos_emb = probe
    T = 6
    model(torch.randint(0, 32, (1, T)))
    assert probe.seen is not None, "the positional embedding was never used"
    want = torch.arange(T)
    assert torch.equal(probe.seen.flatten()[:T], want), (
        f"position indices were {probe.seen.flatten().tolist()}, expected {want.tolist()}")


def test_attention_matches_the_definition():
    model = build()
    attn = model.blocks[0].attn
    x = torch.randn(1, 6, 64)
    got, want = attn(x), correct_attention(attn, x)
    assert torch.allclose(got, want, atol=1e-5), (
        f"attention output is off by {(got - want).abs().max():.3e}")


def test_training_step_changes_at_least_one_parameter():
    torch.manual_seed(0)
    model = m.MiniGPT(vocab_size=32, d_model=64, n_heads=4, n_layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = torch.randint(0, 32, (2, 8))
    before = [p.detach().clone() for p in model.parameters()]
    m.train_step(model, opt, idx, idx)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), (
        "no parameter changed after a reported training step")


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
    full = model(idx)
    caches, outs = None, []
    for t in range(idx.shape[1]):
        logits, caches = m.forward_cached(model, idx[:, t:t + 1], caches)
        outs.append(logits)
    inc = torch.cat(outs, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (
        f"cached decode diverges from full recompute by {(full - inc).abs().max():.2e}")
