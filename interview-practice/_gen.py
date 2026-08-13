"""Generate stubs, tests and hints from one spec. Rerun after editing SPECS.

Writes stubs/<id>_<name>.py (and a pristine copy for `run.py --reset`),
tests/test_<id>_<name>.py, and hints/<id>_<name>.md.
Existing stubs are NOT overwritten unless --force, so your work-in-progress is safe.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from problems import PROBLEMS  # noqa: E402

HEADER = '''"""{id} · {title}   —   budget {minutes} min{cold}

{brief}

Fill in the body. Run:  python run.py {id}
Stuck? hints/{id}_{name}.md has three levels, in increasing order of spoiler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

'''

# id -> (brief, stub_body, test_body, [hint1, hint2, hint3])
SPECS = {}

SPECS["p01"] = (
"Write multi-head causal self-attention from scratch. No nn.MultiheadAttention,\nno F.scaled_dot_product_attention.",
'''
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        # TODO: n_heads, d_head, one fused qkv projection, an output projection,
        # and a causal mask registered as a buffer.
        raise NotImplementedError

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, d_model)"""
        raise NotImplementedError
''',
'''
def test_matches_pytorch_sdpa():
    torch.manual_seed(0)
    m = stub.CausalSelfAttention(64, 8).eval()
    x = torch.randn(2, 7, 64)
    got = m(x)

    B, T, C = x.shape
    q, k, v = m.qkv(x).split(C, dim=2)
    shape = (B, T, 8, 8)
    q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
    want = m.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True)
                  .transpose(1, 2).contiguous().view(B, T, C))
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_is_actually_causal():
    torch.manual_seed(0)
    m = stub.CausalSelfAttention(32, 4).eval()
    x = torch.randn(2, 6, 32)
    y = m(x)
    x2 = x.clone(); x2[:, -1, :] += 10.0
    assert torch.allclose(y[:, :-1], m(x2)[:, :-1], atol=1e-6), \\
        "perturbing the last token changed earlier outputs: the mask leaks the future"


def test_scaling_uses_head_dim():
    torch.manual_seed(0)
    m = stub.CausalSelfAttention(64, 8).eval()
    x = torch.randn(1, 4, 64) * 3
    ref = R.CausalSelfAttention(64, 8).eval()
    ref.load_state_dict(m.state_dict())
    assert torch.allclose(m(x), ref(x), atol=1e-5), \\
        "output differs from reference — check you divide by sqrt(d_head), not sqrt(d_model)"
''',
["The shape journey is (B,T,C) -> (B,T,n_heads,d_head) -> (B,n_heads,T,d_head), and back again at the end.",
 "Mask before the softmax, additively, with -inf. Multiplying by zero after the softmax leaves the masked positions in the denominator.",
 "After `att @ v` you have (B,n_heads,T,d_head). `.transpose(1,2)` makes it non-contiguous, so `.view()` raises — call `.contiguous()` first, or use `.reshape()`."])

SPECS["p02"] = (
"Add a KV cache so decoding step t costs O(t) instead of O(t^2).\nThe cached path must be numerically identical to a full recompute.",
'''
class CachedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x, cache=None):
        """x: (B, T_new, d_model). cache: (k, v) from previous steps, or None.

        Returns (output, new_cache). During decode T_new == 1.
        """
        raise NotImplementedError
''',
'''
def test_cached_decode_equals_full_recompute():
    torch.manual_seed(0)
    m = stub.CachedAttention(64, 8).eval()
    x = torch.randn(1, 6, 64)
    full = m(x)[0]

    out, cache = [], None
    for t in range(x.shape[1]):
        y, cache = m(x[:, t:t + 1], cache)
        out.append(y)
    inc = torch.cat(out, dim=1)
    assert torch.allclose(full, inc, atol=1e-5), \\
        f"incremental decode diverges from teacher forcing by {(full - inc).abs().max():.2e}"


def test_cache_grows_by_one_per_step():
    torch.manual_seed(0)
    m = stub.CachedAttention(32, 4).eval()
    cache = None
    for t in range(1, 5):
        _, cache = m(torch.randn(1, 1, 32), cache)
        assert cache[0].shape[-2] == t, f"after {t} steps the cache holds {cache[0].shape[-2]}"
''',
["The query is only for the new token; keys and values are the concatenation of the cache and the new step.",
 "During decode there is exactly one query row attending to all t keys, so no causal mask is needed at all.",
 "For T_new > 1 with a non-empty cache the mask must start at row offset T_full - T_new: `torch.tril(ones(T_new, T_full), diagonal=T_full - T_new)`."])

SPECS["p04"] = (
"Implement rotary position embeddings: build the cos/sin table, then apply it to q and k.",
'''
def rope_cache(seq_len, d_head, base=10000.0):
    """Return (cos, sin), each (seq_len, d_head // 2)."""
    raise NotImplementedError


def apply_rope(x, cos, sin):
    """x: (..., T, d_head) -> same shape, rotated pairwise."""
    raise NotImplementedError
''',
'''
def test_matches_reference():
    cos, sin = stub.rope_cache(8, 16)
    rc, rs = R.rope_cache(8, 16)
    assert torch.allclose(cos, rc, atol=1e-6) and torch.allclose(sin, rs, atol=1e-6)
    x = torch.randn(2, 3, 8, 16)
    assert torch.allclose(stub.apply_rope(x, cos, sin), R.apply_rope(x, rc, rs), atol=1e-6)


def test_logits_depend_only_on_relative_offset():
    torch.manual_seed(0)
    d, T = 16, 12
    cos, sin = stub.rope_cache(T, d)
    q, k = torch.randn(1, 1, T, d), torch.randn(1, 1, T, d)
    qr, kr = stub.apply_rope(q, cos, sin), stub.apply_rope(k, cos, sin)
    # same offset at two different absolute positions must give the same logit
    a = (qr[0, 0, 5] * kr[0, 0, 3]).sum()
    q2 = torch.roll(q, shifts=2, dims=2); k2 = torch.roll(k, shifts=2, dims=2)
    qr2, kr2 = stub.apply_rope(q2, cos, sin), stub.apply_rope(k2, cos, sin)
    b = (qr2[0, 0, 7] * kr2[0, 0, 5]).sum()
    assert torch.allclose(a, b, atol=1e-4), f"offset-2 logits differ: {a} vs {b}"
''',
["Frequencies are `base ** (-arange(0, d, 2) / d)`; the angle at position m is m * freq.",
 "Rotating the pair (x0, x1) by theta gives (x0*cos - x1*sin, x0*sin + x1*cos).",
 "Slice the pairs with `x[..., 0::2]` and `x[..., 1::2]`, then interleave back with `torch.stack([...], -1).flatten(-2)`."])

SPECS["p05"] = (
"RMSNorm: normalise by the root mean square, no mean subtraction, no bias.",
'''
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
''',
'''
def test_matches_torch_rms_norm():
    torch.manual_seed(0)
    m = stub.RMSNorm(32)
    with torch.no_grad():
        m.weight.copy_(torch.randn(32))
    x = torch.randn(4, 7, 32)
    want = F.rms_norm(x, (32,), m.weight, eps=1e-6)
    assert torch.allclose(m(x), want, atol=1e-5)


def test_reduction_runs_in_fp32_under_bf16():
    m = stub.RMSNorm(64).to(torch.bfloat16)
    x = (torch.randn(2, 3, 64) * 100).to(torch.bfloat16)
    y = m(x)
    assert y.dtype == x.dtype, "output dtype should follow the input (cast back at the end)"
    # a bf16 reduction over 64 squared values of magnitude ~1e4 can drift substantially
    want = F.rms_norm(x.float(), (64,), m.weight.float(), eps=1e-6)
    err = (y.float() - want).abs().max() / want.abs().max()
    assert err < 5e-3, f"relative error {err:.2%}: is the reduction running in fp32?"


def test_float64_is_not_silently_demoted():
    m = stub.RMSNorm(32).double()
    x = torch.randn(2, 3, 32, dtype=torch.float64)
    want = F.rms_norm(x, (32,), m.weight, eps=1e-6)
    assert torch.allclose(m(x), want, atol=1e-12, rtol=1e-12)
''',
["RMS(x) = sqrt(mean(x^2) + eps) over the last dimension, keepdim=True.",
 "There is no mean subtraction and no bias term — that is the whole difference from LayerNorm.",
 "Promote fp16/bf16 reductions to fp32, but do not demote float64; cast the normalised activations back to the input dtype."])

SPECS["p08"] = (
"Cross entropy from logits, with the log-sum-exp trick and ignore_index support.",
'''
def cross_entropy(logits, targets, ignore_index=-100):
    """logits: (N, V) raw scores. targets: (N,) int64. Returns a scalar mean loss."""
    raise NotImplementedError
''',
'''
def test_matches_f_cross_entropy():
    torch.manual_seed(0)
    logits, targets = torch.randn(16, 10), torch.randint(0, 10, (16,))
    assert torch.allclose(stub.cross_entropy(logits, targets),
                          F.cross_entropy(logits, targets), atol=1e-6)


def test_respects_ignore_index():
    torch.manual_seed(0)
    logits, targets = torch.randn(16, 10), torch.randint(0, 10, (16,))
    targets[:5] = -100
    assert torch.allclose(stub.cross_entropy(logits, targets),
                          F.cross_entropy(logits, targets, ignore_index=-100), atol=1e-6)


def test_no_overflow_on_huge_logits():
    logits = torch.tensor([[1e4, 2e4, 3e4]])
    loss = stub.cross_entropy(logits, torch.tensor([2]))
    assert torch.isfinite(loss), "overflowed: subtract the row max before exponentiating"


def test_fully_masked_batch_is_zero_and_stays_on_graph():
    logits = torch.randn(4, 7, requires_grad=True)
    targets = torch.full((4,), -100)
    loss = stub.cross_entropy(logits, targets)
    assert loss.shape == () and loss.detach().item() == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.equal(logits.grad, torch.zeros_like(logits))
''',
["log_softmax(x)[t] = x[t] - logsumexp(x). Never build probabilities and then take a log.",
 "logsumexp needs the max subtracted first: m + log(sum(exp(x - m))).",
 "For ignore_index, average only kept rows. If none remain, return `logits.sum() * 0.0`: finite zero, still attached to the graph."])

SPECS["p10"] = (
"Write a training loop that drives a tiny fixed batch to near-zero loss.\nThis is the smoke test every real run should start with.",
'''
def overfit_tiny(steps=2000, lr=0.5):
    """Build a small model + fixed batch, train it, and report the result.

    Returns (final_loss, accuracy) as plain floats.
    """
    raise NotImplementedError
''',
'''
def test_reaches_near_zero_loss():
    loss, acc = stub.overfit_tiny()
    assert isinstance(loss, float), "return floats, not tensors still attached to the graph"
    assert acc == 1.0, f"accuracy {acc}: ten fixed examples should be memorisable exactly"
    assert loss < 1e-3, f"final loss {loss:.4f}: not actually converged"
''',
["Fixed random inputs and targets, a two-layer MLP, cross entropy, Adam. No dropout, no shuffling.",
 "The three lines that must be in this order every step: `opt.zero_grad()`, `loss.backward()`, `opt.step()`.",
 "Call `.item()` outside any `torch.no_grad()` block, and remember the loss you return should come from the final forward pass, not a stale variable."])

SPECS["p14"] = (
"Sampling with temperature, top-k and top-p. Order matters.",
'''
def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """logits: (V,) -> an int token id."""
    raise NotImplementedError
''',
'''
def test_temperature_zero_is_greedy():
    logits = torch.tensor([0.1, 5.0, 0.3, 2.0])
    assert stub.sample_next(logits, temperature=0.0) == 1


def test_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([5.0, 4.0, 0.1, 0.0, -1.0])
    got = {stub.sample_next(logits, temperature=1.0, top_k=2) for _ in range(300)}
    assert got <= {0, 1}, f"top-k=2 sampled outside the top two: {got}"


def test_top_k_keeps_exactly_k_slots_when_logits_tie():
    torch.manual_seed(1)
    logits = torch.zeros(4)
    got = {stub.sample_next(logits, top_k=2) for _ in range(300)}
    assert len(got) == 2, f"top-k=2 should retain exactly two tied slots, sampled {got}"


def test_top_p_keeps_the_crossing_token():
    # probs approx [0.5, 0.3, 0.15, 0.05]; p=0.9 must keep exactly the first three
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    got = {stub.sample_next(logits, temperature=1.0, top_p=0.9) for _ in range(600)}
    assert got <= {0, 1, 2} and 2 in got, \\
        f"expected the nucleus to be exactly {{0,1,2}}, sampled {got} (off-by-one in the shift?)"


def test_top_p_one_preserves_finite_logit_support(monkeypatch):
    captured = {}

    def capture(probs, num_samples, generator=None):
        captured["probs"] = probs
        return torch.tensor([0], device=probs.device)

    monkeypatch.setattr(torch, "multinomial", capture)
    logits = torch.tensor([700.0, 0.0, -700.0], dtype=torch.float64)
    for top_p in (1.0, 1.5):
        stub.sample_next(logits, top_p=top_p)
        assert captured["probs"][1] > 0, (
            "top_p>=1 must bypass nucleus filtering; cumulative sums can round to one "
            "early for extreme finite logits"
        )
''',
["Apply temperature first — it changes the distribution the truncations then act on.",
 "For top-p below 1, sort descending, take the cumulative sum, and keep the shortest prefix whose mass reaches p. At top_p>=1, skip nucleus filtering so every finite-logit slot remains in support.",
 "The exclusive cumulative sum is `cum - probs`; drop where that is already >= top_p, which keeps the token that crosses the threshold."])

SPECS["p18"] = (
"LoRA: a low-rank adapter that is the identity at initialisation and merges losslessly.",
'''
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def merged_weight(self):
        """Return the single weight matrix equivalent to base + adapter."""
        raise NotImplementedError
''',
'''
def test_identity_at_init():
    torch.manual_seed(0)
    base = nn.Linear(32, 16, bias=False)
    lora = stub.LoRALinear(base, r=4)
    x = torch.randn(3, 32)
    assert torch.allclose(lora(x), base(x), atol=1e-6), \\
        "not the identity at step 0 — B must start at zero"


def test_merge_is_lossless():
    torch.manual_seed(0)
    base = nn.Linear(32, 16, bias=False)
    lora = stub.LoRALinear(base, r=4)
    with torch.no_grad():          # pretend we trained
        for p in lora.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
    x = torch.randn(3, 32)
    merged = F.linear(x, lora.merged_weight())
    assert torch.allclose(lora(x), merged, atol=1e-5), \\
        "merged weight does not reproduce the adapter path"


def test_base_is_frozen():
    base = nn.Linear(8, 8, bias=True)
    lora = stub.LoRALinear(base, r=2)
    assert all(not p.requires_grad for p in lora.base.parameters()), \
        "every base parameter, including bias, must be frozen"


def test_adapter_inherits_base_dtype():
    base = nn.Linear(8, 8, bias=False, dtype=torch.float64)
    lora = stub.LoRALinear(base, r=2)
    assert lora.A.dtype == base.weight.dtype and lora.B.dtype == base.weight.dtype
    assert lora(torch.randn(3, 8, dtype=torch.float64)).dtype == torch.float64
''',
["W' = W + (alpha/r) * B @ A, with A of shape (r, in) and B of shape (out, r).",
 "Initialise A randomly (kaiming) and B to zeros, so B@A = 0 and the adapter starts as a no-op.",
 "Freeze every base parameter; create A/B from `base.weight` so device and dtype match. The merge is `W + (alpha/r) * B @ A`."])

SPECS["p22"] = (
"Byte-pair encoding: train the merges, then encode with them.",
'''
def bpe_train(text, num_merges):
    """Return a dict mapping (a, b) -> new_id, in the order the merges were learned."""
    raise NotImplementedError


def bpe_encode(text, merges):
    """Apply merges in learned order. Returns a list of ints."""
    raise NotImplementedError
''',
'''
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
''',
["Start from raw bytes (`text.encode('utf-8')`), so there is never an out-of-vocabulary case.",
 "Each round: count adjacent pairs, take the most frequent, replace every occurrence with a fresh id starting at 256.",
 "Encoding applies merges in the order they were LEARNED, not by frequency in the string being encoded. Iterate the merge dict in insertion order."])


def write(force=False):
    (HERE / "stubs" / ".pristine").mkdir(parents=True, exist_ok=True)
    n_s = n_t = n_h = 0
    for p in PROBLEMS:
        if p.id not in SPECS:
            continue
        brief, body, test, hints = SPECS[p.id]
        base = f"{p.id}_{p.name}"

        head = HEADER.format(id=p.id, title=p.title, minutes=p.minutes,
                             cold="  [cold-start set]" if p.cold else "",
                             brief=brief, name=p.name)
        stub_txt = head + body.strip() + "\n"
        pristine = HERE / "stubs" / ".pristine" / f"{base}.py"
        pristine.write_text(stub_txt, encoding="utf-8")
        target = HERE / "stubs" / f"{base}.py"
        if force or not target.exists():
            target.write_text(stub_txt, encoding="utf-8"); n_s += 1

        t = (f'"""Tests for {p.id} · {p.title}. Run: python run.py {p.id}"""\n\n'
             "import sys\nfrom pathlib import Path\n\n"
             "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n"
             "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
             f"import reference as R  # noqa: E402\n"
             f"from stubs import {base} as stub  # noqa: E402\n\n" + test.strip() + "\n")
        (HERE / "tests" / f"test_{base}.py").write_text(t, encoding="utf-8"); n_t += 1

        h = (f"# Hints · {p.id} {p.title}\n\n"
             "Read one at a time. If the first is enough, stop.\n\n"
             + "\n\n".join(f"## Level {i}\n\n{x}" for i, x in enumerate(hints, 1)) + "\n")
        (HERE / "hints" / f"{base}.md").write_text(h, encoding="utf-8"); n_h += 1

    (HERE / "stubs" / "__init__.py").write_text("", encoding="utf-8")
    print(f"stubs written {n_s} (skipped existing), tests {n_t}, hints {n_h}")
    print(f"specs present for {len(SPECS)}/{len(PROBLEMS)} problems")


if __name__ == "__main__":
    write(force="--force" in sys.argv)
