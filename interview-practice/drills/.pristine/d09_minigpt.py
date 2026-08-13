"""d09 · Debug miniGPT — budget 35 min

This OpenAI-style drill is synthesised from public, anecdotal interview reports; it is
not an official or verbatim question. You are handed a nanoGPT-style causal LM that runs
but violates four behavioural invariants. Then implement a KV cache and prove it matches
the uncached path.

The bugs here are logical, not syntactic. Nothing raises. `python -m pytest
tests/test_d09_minigpt.py -q` tells you which invariant is violated, not where.

Your job:
  1. find and fix the four bugs (the tests name the symptom, not the line)
  2. implement forward_cached below and make the last test pass

Stuck? Read hints/d09_minigpt.md one level at a time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = F.softmax(att, dim=-1)
        att = att * mask

        y = att @ v                                   # (B, n_heads, T, d_head)
        y = y.reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=64):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.zeros(T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


def train_step(model, opt, idx, targets):
    """One optimisation step of next-token prediction."""
    logits = model(idx)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1)
    )
    opt.zero_grad()
    loss.backward()
    return loss.item()


# ----------------------------------------------------------------------------------
# Follow-up: implement a per-layer K/V cache. Derive its positional and masking contracts
# from the behavioural test; use the graded hints only if needed.
# ----------------------------------------------------------------------------------
def forward_cached(model, idx, caches=None):
    """idx: (B, T_new). caches: list of (k, v) per layer, or None.

    Returns (logits, caches). Must match the uncached path within floating-point tolerance.
    """
    raise NotImplementedError
