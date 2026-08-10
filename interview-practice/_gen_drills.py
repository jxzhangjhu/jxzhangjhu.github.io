"""Generate the micro debug drills d01-d08 from one spec. Rerun after editing SPECS.

Writes drills/<id>_<name>.py (with the bug planted), drills/.solutions/<id>_<name>.py
(the same file with the bug fixed), and tests/test_<id>_<name>.py.

Each drill is one function with exactly one wrong line, so buggy and fixed are the same
source with a single substitution. That keeps the two copies from drifting apart, which
is the usual failure mode of hand-maintained "find the bug" exercises.

d09 and d10 are the full-length drills and are hand-written; this only owns d01-d08.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from problems import DRILLS_BY_ID  # noqa: E402

HEADER = '''"""{id} · {title}   —   budget {minutes} min

{brief}

One line in this file is wrong. Run:  python -m pytest tests/test_{id}_{name}.py -q
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

'''

# id -> (title, brief, source with a {impl} hole, buggy line, fixed line, test body)
SPECS = {}

SPECS["d01"] = (
    "Inverted attention mask",
    "The mask marks the positions a query is allowed to see. The scores come back with\n"
    "every allowed position at -inf, so softmax puts all the weight on the future.",
    '''
def causal_scores(q, k):
    """q, k: (B, T, D). Returns pre-softmax scores with the future masked out."""
    T = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    allowed = torch.tril(torch.ones(T, T, dtype=torch.bool))
{impl}
''',
    '    return scores.masked_fill(allowed, float("-inf"))',
    '    return scores.masked_fill(~allowed, float("-inf"))',
    '''
def test_future_is_masked_and_past_is_not():
    torch.manual_seed(0)
    q, k = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    s = d.causal_scores(q, k)[0]
    assert torch.isinf(s[0, 1:]).all(), \\
        "query 0 must not see any later position"
    assert torch.isfinite(s[3, :4]).all(), \\
        "query 3 must see positions 0..3; masked_fill fills where the mask is True"
    w = F.softmax(s, dim=-1)
    assert torch.allclose(w[0, 0], torch.tensor(1.0)), \\
        "row 0 should put all its weight on itself"
''',
)

SPECS["d02"] = (
    "view() on a transposed tensor",
    "Merging heads back together after a transpose. The tensor is no longer contiguous,\n"
    "so view() cannot reinterpret its strides.",
    '''
def merge_heads(x):
    """x: (B, H, T, Dh) -> (B, T, H*Dh), heads concatenated per position."""
    B, H, T, Dh = x.shape
{impl}
''',
    '    return x.transpose(1, 2).view(B, T, H * Dh)',
    '    return x.transpose(1, 2).contiguous().view(B, T, H * Dh)',
    '''
def test_merge_matches_a_manual_concat():
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).view(2, 3, 4, 5)
    got = d.merge_heads(x)
    want = torch.cat([x[:, h] for h in range(3)], dim=-1)
    assert got.shape == want.shape, f"expected {tuple(want.shape)}, got {tuple(got.shape)}"
    assert torch.equal(got, want), \\
        "heads came back interleaved: view() reused the pre-transpose strides"
''',
)

SPECS["d03"] = (
    "top-p drops the crossing token",
    "Nucleus sampling should keep the shortest prefix whose cumulative mass reaches p,\n"
    "which means the token that crosses the threshold stays in.",
    '''
def nucleus(logits, top_p):
    """logits: (V,). Returns logits with everything outside the nucleus set to -inf."""
    srt, idx = torch.sort(logits, descending=True)
    probs = F.softmax(srt, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
{impl}
    srt = srt.masked_fill(drop, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(0, idx, srt)
''',
    '    drop = cum >= top_p',
    '    drop = cum - probs >= top_p',
    '''
def test_the_crossing_token_survives():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    kept = torch.isfinite(d.nucleus(logits, 0.9)).nonzero().flatten().tolist()
    assert kept == [0, 1, 2], \\
        f"p=0.9 on [.5,.3,.15,.05] should keep three tokens, kept {kept}: " \\
        "use the exclusive cumulative sum, cum - probs"


def test_a_single_dominant_token_is_never_dropped():
    logits = torch.log(torch.tensor([0.99, 0.005, 0.005]))
    kept = torch.isfinite(d.nucleus(logits, 0.5)).nonzero().flatten().tolist()
    assert kept == [0], f"expected only the argmax, kept {kept}"
''',
)

SPECS["d04"] = (
    "Cached decode with an unshifted mask",
    "During incremental decode the query block is short but the key block is the whole\n"
    "cache, so a plain lower triangle points at the wrong keys.",
    '''
def cached_causal_mask(n_new, n_total):
    """True where a query in the last n_new positions may attend to a key in n_total."""
{impl}
''',
    '    return torch.tril(torch.ones(n_new, n_total, dtype=torch.bool))',
    '    return torch.tril(torch.ones(n_new, n_total, dtype=torch.bool),\n'
    '                      diagonal=n_total - n_new)',
    '''
def test_each_new_query_sees_the_whole_cache_up_to_itself():
    m = d.cached_causal_mask(2, 5)
    assert m.shape == (2, 5)
    assert m[0].tolist() == [True, True, True, True, False], \\
        f"the first new query is at absolute position 3, so it sees keys 0..3; got {m[0].tolist()}"
    assert m[1].tolist() == [True] * 5, \\
        f"the second new query sees the whole cache; got {m[1].tolist()}"


def test_prefill_is_the_plain_triangle():
    m = d.cached_causal_mask(4, 4)
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool))), \\
        "with no cache the shift must be zero"
''',
)

SPECS["d05"] = (
    "LoRA that is not identity at init",
    "A freshly initialised adapter has to leave the base model bit-for-bit unchanged,\n"
    "or your first training step starts from a different model than you evaluated.",
    '''
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base, self.scaling = base, alpha / r
        base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
{impl}

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling
''',
    '        self.B = nn.Parameter(torch.randn(base.out_features, r) * 0.01)',
    '        self.B = nn.Parameter(torch.zeros(base.out_features, r))',
    '''
def test_adapter_is_identity_before_training():
    torch.manual_seed(0)
    base = nn.Linear(16, 32)
    x = torch.randn(4, 16)
    want = base(x).clone()
    got = d.LoRALinear(base, r=4, alpha=8)(x)
    assert torch.allclose(got, want, atol=1e-7), \\
        "a fresh adapter changed the output: one of A, B must start at zero"


def test_the_adapter_is_not_dead_at_init():
    """With B = 0 the gradient reaches B but not A, which is why zeroing both is fatal."""
    torch.manual_seed(0)
    m = d.LoRALinear(nn.Linear(16, 32), r=4, alpha=8)
    m(torch.randn(4, 16)).sum().backward()
    assert m.B.grad is not None and m.B.grad.abs().sum() > 0, \\
        "no gradient reaches B: zeroing both factors makes the adapter untrainable"
''',
)

SPECS["d06"] = (
    "softmax without the max subtraction",
    "Works on toy logits, overflows to inf on the logit magnitudes a real model produces.",
    '''
def softmax(x):
    """x: (..., V) -> the same shape, rows summing to 1."""
{impl}
    return e / e.sum(dim=-1, keepdim=True)
''',
    '    e = x.exp()',
    '    e = (x - x.max(dim=-1, keepdim=True).values).exp()',
    '''
def test_survives_realistic_logit_magnitudes():
    x = torch.tensor([[1e4, 2e4, 3e4]])
    got = d.softmax(x)
    assert torch.isfinite(got).all(), \\
        "exp() overflowed to inf, then inf/inf gave nan: subtract the row max first"
    assert torch.allclose(got, F.softmax(x, dim=-1), atol=1e-6)


def test_still_correct_on_ordinary_logits():
    x = torch.randn(3, 7)
    assert torch.allclose(d.softmax(x), F.softmax(x, dim=-1), atol=1e-6)
''',
)

SPECS["d07"] = (
    "Attention scaled by the wrong dimension",
    "The scale is the square root of the *per-head* dimension. With eight heads this is\n"
    "off by a factor of sqrt(8), which just looks like a model that trains badly.",
    '''
def attention(q, k, v, n_heads):
    """q, k, v: (B, T, D) with D = n_heads * d_head. Non-causal, single call."""
    B, T, D = q.shape
    d_head = D // n_heads
    shape = (B, T, n_heads, d_head)
    q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
{impl}
    out = F.softmax(scores, dim=-1) @ v
    return out.transpose(1, 2).contiguous().view(B, T, D)
''',
    '    scores = q @ k.transpose(-2, -1) / math.sqrt(D)',
    '    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)',
    '''
def test_matches_pytorch_sdpa():
    torch.manual_seed(0)
    B, T, D, H = 2, 6, 32, 8
    q, k, v = (torch.randn(B, T, D) for _ in range(3))
    got = d.attention(q, k, v, H)

    shape = (B, T, H, D // H)
    qq, kk, vv = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
    want = F.scaled_dot_product_attention(qq, kk, vv)
    want = want.transpose(1, 2).contiguous().view(B, T, D)
    assert torch.allclose(got, want, atol=1e-5), \\
        "scores are scaled by sqrt(d_model) instead of sqrt(d_head)"
''',
)

SPECS["d08"] = (
    "SFT loss counting the prompt",
    "Supervised fine-tuning should only score the completion. Scoring the prompt too\n"
    "trains the model to generate the instructions it is supposed to follow.",
    '''
def sft_loss(logits, input_ids, prompt_len):
    """logits: (B, T, V), input_ids: (B, T). Only completion tokens should be scored."""
    logits, targets = logits[:, :-1], input_ids[:, 1:]
    B, Tm1, V = logits.shape
{impl}
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
''',
    '    targets = targets.clone()',
    '    targets = targets.clone()\n'
    '    positions = torch.arange(Tm1).expand(B, Tm1)\n'
    '    targets[positions < prompt_len - 1] = -100',
    '''
def test_prompt_tokens_do_not_affect_the_loss():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 50)
    ids = torch.randint(0, 50, (2, 8))
    base = d.sft_loss(logits, ids, prompt_len=4)

    moved = ids.clone()
    moved[:, :3] = (moved[:, :3] + 7) % 50        # rewrite prompt tokens only
    assert torch.allclose(base, d.sft_loss(logits, moved, prompt_len=4)), \\
        "changing prompt tokens changed the loss, so the prompt is still being scored"


def test_matches_a_hand_masked_reference():
    torch.manual_seed(0)
    logits, ids = torch.randn(2, 8, 50), torch.randint(0, 50, (2, 8))
    lg, tg = logits[:, :-1], ids[:, 1:].clone()
    tg[:, :3] = -100                               # prompt_len=4 -> targets 0..2 are prompt
    want = F.cross_entropy(lg.reshape(-1, 50), tg.reshape(-1), ignore_index=-100)
    assert torch.allclose(d.sft_loss(logits, ids, prompt_len=4), want)
''',
)

TEST_HEADER = '''"""Tests for {id} · {title}. Run: python -m pytest tests/test_{id}_{name}.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import {id}_{name} as d  # noqa: E402

'''


def main():
    drills = HERE / "drills"
    sols = drills / ".solutions"
    sols.mkdir(parents=True, exist_ok=True)
    written = 0

    for did, (title, brief, body, buggy, fixed, test) in sorted(SPECS.items()):
        meta = DRILLS_BY_ID[did]
        head = HEADER.format(id=did, name=meta.name, title=title,
                             minutes=meta.minutes, brief=brief)
        stem = f"{did}_{meta.name}"
        (drills / f"{stem}.py").write_text(head + body.format(impl=buggy).lstrip("\n"),
                                           encoding="utf-8")
        (sols / f"{stem}.py").write_text(head + body.format(impl=fixed).lstrip("\n"),
                                         encoding="utf-8")
        (HERE / "tests" / f"test_{stem}.py").write_text(
            TEST_HEADER.format(id=did, name=meta.name, title=title) + test.lstrip("\n"),
            encoding="utf-8")
        written += 3
        print(f"  {did}  {meta.name}")

    print(f"\n{written} files written for {len(SPECS)} micro-drills")


if __name__ == "__main__":
    main()
