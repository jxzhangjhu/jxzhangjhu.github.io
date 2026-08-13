"""Generate the remaining stubs/tests/hints from the validated reference implementations.

Most of these tests are "your version must match the reference numerically", which is
exactly the bar an interviewer applies. Where a problem has a property worth asserting on
its own (an identity, a limit, an invariant), that is added as a second test.

    python3 _gen2.py [--force]
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from problems import BY_ID  # noqa: E402

HEADER = '''"""{id} · {title}   —   budget {minutes} min{cold}{seen}

{brief}

Fill in the body. Run:  python run.py {id}
Stuck? hints/{id}_{name}.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''

# id: (brief, stub, test, [hint x3])
S = {}

S["p24"] = ("1-nearest-neighbour classification in NumPy. No Python loops over test points.",
'''
def nearest_neighbour(train_x, train_y, test_x):
    """train_x (n, d), train_y (n,), test_x (m, d) -> predicted labels (m,)."""
    raise NotImplementedError
''',
'''
def test_matches_bruteforce():
    rng = np.random.default_rng(0)
    tr_x, te_x = rng.normal(size=(40, 5)), rng.normal(size=(17, 5))
    tr_y = rng.integers(0, 3, 40)
    want = np.array([tr_y[np.argmin(((tr_x - t) ** 2).sum(1))] for t in te_x])
    assert (stub.nearest_neighbour(tr_x, tr_y, te_x) == want).all()


def test_no_python_loop_over_test_points():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(stub.nearest_neighbour))
    loop_nodes = (ast.For, ast.AsyncFor, ast.ListComp, ast.SetComp,
                  ast.DictComp, ast.GeneratorExp)
    assert not any(isinstance(node, loop_nodes) for node in ast.walk(tree)), (
        "vectorise pairwise differences instead of looping over test points"
    )


def test_scales_to_a_large_input():
    rng = np.random.default_rng(1)
    tr_x, te_x = rng.normal(size=(500, 20)), rng.normal(size=(300, 20))
    tr_y = rng.integers(0, 5, 500)
    out = stub.nearest_neighbour(tr_x, tr_y, te_x)
    assert out.shape == (300,)


def test_float32_nearby_large_coordinates_do_not_cancel():
    tr_x = np.array([[100.02], [100.001]], dtype=np.float32)
    tr_y = np.array([0, 1])
    te_x = np.array([[100.0]], dtype=np.float32)
    assert stub.nearest_neighbour(tr_x, tr_y, te_x).item() == 1, (
        "expanded squared norms cancel in float32 here; square direct differences instead"
    )
''',
["Broadcast direct differences to shape (m, n, d): `test_x[:, None, :] - train_x[None, :, :]`.",
 "Square those differences and sum over the feature axis. This avoids cancellation from expanding two large nearby squared norms.",
 "You never need sqrt because argmin is invariant to it. This teaching version materialises (m,n,d); production code can chunk rows or use a vetted distance kernel."])

S["p25"] = ("BatchNorm1d from scratch: forward, running statistics, and eval mode.",
'''
class BatchNorm1dScratch(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        """x: (B, d) -> (B, d)"""
        raise NotImplementedError
''',
'''
def test_matches_pytorch_in_train_mode():
    torch.manual_seed(0)
    mine, ref = stub.BatchNorm1dScratch(6), nn.BatchNorm1d(6)
    for _ in range(5):
        x = torch.randn(32, 6) * 3 + 1
        assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_running_stats_match():
    torch.manual_seed(0)
    mine, ref = stub.BatchNorm1dScratch(6), nn.BatchNorm1d(6)
    for _ in range(5):
        x = torch.randn(32, 6) * 3 + 1
        mine(x); ref(x)
    assert torch.allclose(mine.running_mean, ref.running_mean, atol=1e-5)
    assert torch.allclose(mine.running_var, ref.running_var, atol=1e-4), \\
        "running_var differs: normalise with the biased variance, accumulate the unbiased one"


def test_eval_mode_uses_running_stats():
    torch.manual_seed(0)
    mine, ref = stub.BatchNorm1dScratch(6), nn.BatchNorm1d(6)
    for _ in range(5):
        x = torch.randn(32, 6) * 3 + 1
        mine(x); ref(x)
    mine.eval(); ref.eval()
    x = torch.randn(8, 6)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_input_and_affine_gradients_match_pytorch():
    torch.manual_seed(1)
    mine, ref = stub.BatchNorm1dScratch(6), nn.BatchNorm1d(6)
    with torch.no_grad():
        mine.gamma.copy_(ref.weight)
        mine.beta.copy_(ref.bias)
    x1 = torch.randn(16, 6, dtype=torch.float64, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    mine = mine.double()
    ref = ref.double()
    upstream = torch.randn_like(x1)
    mine(x1).backward(upstream)
    ref(x2).backward(upstream)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-9)
    assert torch.allclose(mine.gamma.grad, ref.weight.grad, atol=1e-9)
    assert torch.allclose(mine.beta.grad, ref.bias.grad, atol=1e-9)


def test_running_stats_are_buffers_not_parameters():
    m = stub.BatchNorm1dScratch(4)
    names = {n for n, _ in m.named_parameters()}
    assert "running_mean" not in names, "running stats must be buffers, not parameters"
    assert "running_mean" in dict(m.named_buffers())


def test_singleton_training_batch_is_rejected_cleanly():
    m = stub.BatchNorm1dScratch(4)
    try:
        m(torch.randn(1, 4))
    except ValueError:
        return
    raise AssertionError("a singleton training batch cannot estimate per-channel variance")
''',
["Train mode uses batch statistics; eval mode uses the running ones. They compute different functions.",
 "register_buffer, not nn.Parameter — they move with .to(device) and are saved, but get no gradient.",
 "PyTorch normalises with the biased variance (/n) but accumulates the unbiased one (/(n-1)). Reject n < 2 in training instead of dividing by zero."])

S["p03"] = ("Grouped-query attention: n_kv_heads < n_heads, each group sharing one K/V head.",
'''
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, cache=None):
        """cache: dict with 'k','v', mutated in place. None disables caching."""
        raise NotImplementedError
''',
'''
def test_matches_reference():
    torch.manual_seed(0)
    mine = stub.GroupedQueryAttention(64, 8, 2).eval()
    ref = R.GroupedQueryAttention(64, 8, 2).eval()
    ref.load_state_dict(mine.state_dict())
    x = torch.randn(2, 7, 64)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_degenerates_to_mha():
    torch.manual_seed(0)
    m = stub.GroupedQueryAttention(64, 8, 8).eval()
    y = m(torch.randn(1, 5, 64))
    assert y.shape == (1, 5, 64), "n_kv_heads == n_heads should just be MHA"


def test_cached_decode_matches_full():
    torch.manual_seed(0)
    m = stub.GroupedQueryAttention(32, 4, 2).eval()
    x = torch.randn(1, 6, 32)
    full = m(x)
    cache, outs = {}, []
    for t in range(6):
        outs.append(m(x[:, t:t + 1], cache))
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-5)
''',
["Q gets n_heads, K and V get n_kv_heads. Only the projection output sizes differ.",
 "repeat_interleave(n_rep, dim=1) expands the KV heads to match the query heads before the matmul.",
 "With a cache the query block starts at T_full - T, so the mask needs diagonal=T_full - T."])

S["p28"] = ("Multi-head latent attention: cache a low-rank KV latent plus a small positional key.",
'''
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, kv_rank, rope_dim):
        super().__init__()
        # TODO: split each query/key head into non-positional and RoPE parts.
        # Cache only the shared latent c and the shared rotated positional key.
        raise NotImplementedError

    def forward(self, x, cache=None):
        """x: (B,T,D); cache is a mutable dict with compressed c and k_rope."""
        raise NotImplementedError
''',
'''
def test_cached_decode_matches_full_recompute():
    torch.manual_seed(0)
    m = stub.MultiHeadLatentAttention(64, 4, kv_rank=12, rope_dim=8).eval()
    x = torch.randn(1, 7, 64)
    full = m(x)
    cache, pieces = {}, []
    for t in range(x.shape[1]):
        pieces.append(m(x[:, t:t + 1], cache))
    step = torch.cat(pieces, dim=1)
    assert torch.allclose(full, step, atol=1e-5), (
        f"cached decode differs from full recompute by {(full - step).abs().max():.2e}")


def test_cache_is_compressed():
    torch.manual_seed(1)
    m = stub.MultiHeadLatentAttention(64, 4, kv_rank=12, rope_dim=8).eval()
    cache = {}
    m(torch.randn(2, 3, 64), cache)
    assert set(cache) == {"c", "k_rope"}
    assert cache["c"].shape == (2, 3, 12)
    assert cache["k_rope"].shape == (2, 1, 3, 8)
    cached = cache["c"].shape[-1] + cache["k_rope"].shape[-1]
    mha = 2 * 4 * (64 // 4)
    assert cached < mha, f"cached {cached} values/token; plain MHA needs {mha}"


def test_matches_validated_reference():
    torch.manual_seed(2)
    mine = stub.MultiHeadLatentAttention(64, 4, 12, 8).eval()
    ref = R.MultiHeadLatentAttention(64, 4, 12, 8).eval()
    ref.load_state_dict(mine.state_dict())
    x = torch.randn(2, 5, 64)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)
''',
["The content cache is `c = w_down(x)` with shape (B,T,kv_rank); reconstruct non-positional K and V from c only when needed.",
 "Use a separate shared `k_rope` of shape (B,1,T,rope_dim). Split Q into `q_nope` and `q_rope`, rotate only the RoPE parts, and concatenate before QK^T.",
 "For cached decode, rotate new Q/K with positions `past:past+T`, append c along axis 1 and k_rope along axis 2, then use a causal mask with diagonal=T_full-T."])

S["p19"] = ("The GRPO objective: group-relative advantage, clipped ratio, per-token k3 KL.",
'''
def grpo_loss(logp, logp_old, logp_ref, rewards, mask, group_size, clip_eps=0.2, beta=0.04):
    """logp/logp_old/logp_ref: (B, L). rewards: (B,). mask: (B, L). Returns a scalar."""
    raise NotImplementedError
''',
'''
def _batch(B=8, L=5, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, L), torch.randn(B, L), torch.randn(B, L),
            torch.rand(B), torch.ones(B, L))


def test_tied_group_gives_no_gradient_and_no_nan():
    logp, _, ref, _, mask = _batch()
    rewards = torch.ones(8)                       # every completion scores the same
    loss = stub.grpo_loss(logp, logp.clone(), ref, rewards, mask, group_size=4, beta=0.0)
    assert torch.isfinite(loss), "a tied group divided by std=0 — add an epsilon"
    assert abs(float(loss)) < 1e-4, \\
        f"advantage should be identically zero for a tied group, got loss {float(loss)}"


def test_singleton_groups_are_finite_and_carry_zero_relative_signal():
    logp, _, ref, rewards, mask = _batch(B=4)
    loss = stub.grpo_loss(logp, logp.clone(), ref, rewards, mask, group_size=1, beta=0.0)
    assert torch.isfinite(loss) and abs(loss.detach().item()) < 1e-6


def test_ratio_is_one_on_policy():
    logp, _, ref, rewards, mask = _batch()
    loss = stub.grpo_loss(logp, logp.clone(), ref, rewards, mask, group_size=4, beta=0.0)
    r = rewards.view(-1, 4)
    adv = ((r - r.mean(1, keepdim=True))
           / (r.std(1, keepdim=True, correction=0) + 1e-4)).reshape(-1)
    assert torch.allclose(loss, -adv.mean(), atol=1e-3), \\
        "at ratio 1 the clipped surrogate must equal the advantage"


def test_completions_are_weighted_equally_despite_different_lengths():
    z = torch.zeros(2, 4)
    rewards = torch.tensor([0.0, 1.0])
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    loss = stub.grpo_loss(z, z, z, rewards, mask, group_size=2, beta=0.0)
    assert abs(loss.detach().item()) < 1e-6, (
        "this exercise implements original GRPO: average within each completion, then "
        "average completions. A global-token reduction is a distinct length-weighting variant"
    )


def test_kl_term_is_non_negative():
    logp, old, ref, rewards, mask = _batch()
    with_kl = stub.grpo_loss(logp, old, ref, rewards, mask, 4, beta=1.0)
    without = stub.grpo_loss(logp, old, ref, rewards, mask, 4, beta=0.0)
    assert float(with_kl) >= float(without) - 1e-6, \\
        "the k3 estimator is non-negative per sample, so adding it cannot lower the loss"
''',
["Advantage: reshape rewards to (-1, G), standardise within the group with population std (`correction=0`), reshape back, and broadcast over tokens.",
 "ratio = (logp - logp_old).exp(); the clipped surrogate is -min(ratio*adv, clamp(ratio)*adv).",
 "k3 KL: with log_ratio = logp_ref - logp, it is log_ratio.exp() - log_ratio - 1. Average valid tokens within each completion, then average valid completions so length does not change example weight."])

S["p27"] = ("Simulate a spinning light source hitting a wall, then verify the distribution.",
'''
def light_source_samples(n, seed=0):
    """A lamp at distance 1 from an infinite wall, pointing uniformly at random.

    Return n sample positions along the wall.
    """
    raise NotImplementedError


def cauchy_pdf(x):
    """The analytic density the samples should follow."""
    raise NotImplementedError
''',
'''
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
''',
["The angle is Uniform(-pi/2, pi/2) and the hit position is tan(theta).",
 "Transform the density: f_X(x) = f_theta(theta) |d theta / dx| = 1 / (pi (1 + x^2)).",
 "When you compare a histogram to the PDF on a finite window, density=True normalises over that window while the true Cauchy has only (2/pi)arctan(L) of its mass there."])


S["p06"] = ("SwiGLU feed-forward: three matrices, with the 8/3 sizing.",
'''
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
''',
'''
def test_matches_reference_numerically():
    torch.manual_seed(0)
    mine, ref = stub.SwiGLU(96), R.SwiGLU(96)
    ref.load_state_dict(mine.state_dict())
    x = torch.randn(2, 5, 96)
    assert torch.allclose(mine(x), ref(x), atol=1e-6)


def test_default_width_matches_four_x_ffn_parameter_count():
    d = 96
    m = stub.SwiGLU(d)
    got = sum(p.numel() for p in m.parameters())
    want = 2 * d * (4 * d)
    assert abs(got - want) / want < 0.02
''',
["Gate, up, down: three projections, not two.",
 "Forward is `w_down(silu(w_gate(x)) * w_up(x))`.",
 "`d_ff = 8*d_model/3` keeps the parameter count equal to a 4x ReLU FFN."])

S["p07"] = ("Assemble the completed attention, RMSNorm, and SwiGLU exercises into a pre-norm block.",
'''
from stubs.p01_mha import CausalSelfAttention
from stubs.p05_rmsnorm import RMSNorm
from stubs.p06_swiglu import SwiGLU


class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
''',
'''
def test_matches_reference_and_preserves_shape():
    torch.manual_seed(0)
    mine, ref = stub.Block(64, 8), R.Block(64, 8)
    ref.load_state_dict(mine.state_dict())
    x = torch.randn(2, 6, 64)
    assert mine(x).shape == x.shape
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_block_is_causal():
    torch.manual_seed(1)
    m = stub.Block(32, 4).eval()
    x = torch.randn(1, 6, 32)
    y = m(x)
    x[:, -1] += 10
    assert torch.allclose(y[:, :-1], m(x)[:, :-1], atol=1e-5)
''',
["Complete p01, p05, and p06 first; this exercise deliberately reuses those modules.",
 "Two residual lines: `x = x + attn(norm1(x))`, then `x = x + mlp(norm2(x))`.",
 "Pre-norm normalises each sublayer input. A complete language model also needs a final norm before `lm_head`."])

S["p09"] = ("Build SFT labels plus position ids and a block-diagonal mask for packing.",
'''
def build_sft_labels(input_ids, prompt_lens, attention_mask, ignore_index=-100):
    """Return labels with prompts and padding replaced by ignore_index."""
    raise NotImplementedError


def build_packed_sft_labels(input_ids, response_mask, attention_mask, ignore_index=-100):
    """Label response tokens in packed rows; mask prompts, separators, and padding."""
    raise NotImplementedError


def build_packed_attention(segment_ids, attention_mask):
    """Return (position_ids, allowed_mask) with positions reset at each segment."""
    raise NotImplementedError
''',
'''
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
''',
["There are three independent invariants: only response tokens are labels; position ids restart at each packed segment; attention is causal and cannot cross a segment boundary.",
 "For labels, intersect response_mask with attention_mask. For attention, first identify contiguous segment runs, then compare each query/key run and AND with a lower-triangular mask and validity on both axes.",
 "Reset an offset whenever the segment id changes. Build run ids from boundary indicators and return `same_run & causal & query_valid & key_valid`; padding gets position zero and no allowed row or column."])

S["p11"] = ("Implement scalar reverse-mode autodiff in the style of micrograd.",
'''
class Value:
    def __init__(self, data, _children=(), _op=""):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __pow__(self, k):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else Value(other) ** -1)

    def __hash__(self):
        return id(self)
''',
'''
def test_forward_and_reused_node_gradients_match_torch():
    a, b = stub.Value(-4.0), stub.Value(2.0)
    c = a * b + b.tanh()
    out = c * c + a / b
    out.backward()

    ta = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    tc = ta * tb + torch.tanh(tb)
    tout = tc * tc + ta / tb
    tout.backward()
    assert abs(out.data - tout.detach().item()) < 1e-9
    assert abs(a.grad - ta.grad.detach().item()) < 1e-6
    assert abs(b.grad - tb.grad.detach().item()) < 1e-6
''',
["Each result stores its parents and a closure that pushes `out.grad` into them.",
 "Use `+=`, not `=`: one node may receive gradient through several graph paths.",
 "`backward()` builds a DFS post-order, seeds the output gradient with 1, then executes closures in reverse topological order."])

S["p12"] = ("Derive and implement the attention backward pass without autograd.",
'''
def attention_backward(d_out, cache):
    """cache is returned by reference.attention_forward: (q, k, v, p, scale)."""
    raise NotImplementedError
''',
'''
def test_all_three_gradients_match_autograd():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 6, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 6, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 6, 8, dtype=torch.float64, requires_grad=True)
    out, cache = R.attention_forward(q, k, v, causal=True)
    upstream = torch.randn_like(out)
    out.backward(upstream)
    got = stub.attention_backward(upstream, cache)
    for name, mine, want in zip(("dQ", "dK", "dV"), got, (q.grad, k.grad, v.grad)):
        assert mine.shape == want.shape, f"{name} has shape {mine.shape}, expected {want.shape}"
        assert torch.allclose(mine, want, atol=1e-9), name
''',
["From `O = P V`: `dV = P^T dO`, `dP = dO V^T`.",
 "Softmax VJP: `dS = P * (dP - rowsum(dP * P))`.",
 "From `S = scale QK^T`: `dQ = scale dS K`, `dK = scale dS^T Q`. Masked entries already have P=0."])

S["p13"] = ("Implement the backward pass of a two-layer ReLU MLP.",
'''
def mlp_backward(d_y, cache):
    """cache is returned by reference.mlp_forward: (x, W1, W2, h, a)."""
    raise NotImplementedError
''',
'''
def test_every_gradient_matches_autograd():
    torch.manual_seed(0)
    kw = dict(dtype=torch.float64, requires_grad=True)
    x, W1, b1 = torch.randn(5, 4, **kw), torch.randn(4, 9, **kw), torch.randn(9, **kw)
    W2, b2 = torch.randn(9, 4, **kw), torch.randn(4, **kw)
    y, cache = R.mlp_forward(x, W1, b1, W2, b2)
    upstream = torch.randn_like(y)
    y.backward(upstream)
    got = stub.mlp_backward(upstream, cache)
    for mine, want in zip(got, (x.grad, W1.grad, b1.grad, W2.grad, b2.grad)):
        assert mine.shape == want.shape
        assert torch.allclose(mine, want, atol=1e-9)
''',
["Work backwards through the down projection, ReLU, then the up projection.",
 "Every gradient must have the shape of the tensor it belongs to; use that to determine each transpose.",
 "A broadcast bias becomes a sum over the broadcast batch dimension in backward."])

S["p15"] = ("Implement one exact speculative-decoding accept/reject step.",
'''
def speculative_accept(p_target, q_draft, token, u):
    """Return (emitted_token, accepted_draft: bool)."""
    raise NotImplementedError
''',
'''
def test_accepts_when_target_dominates_draft():
    p = torch.tensor([0.7, 0.3])
    q = torch.tensor([0.2, 0.8])
    assert stub.speculative_accept(p, q, token=0, u=0.99) == (0, True)


def test_rejection_samples_the_positive_residual():
    p = torch.tensor([0.7, 0.3])
    q = torch.tensor([0.2, 0.8])
    token, accepted = stub.speculative_accept(p, q, token=1, u=0.99)
    assert not accepted and token == 0, "the residual distribution has all its mass on token 0"


def test_zero_acceptance_probability_rejects_even_when_u_is_zero():
    p = torch.tensor([0.0, 1.0])
    q = torch.tensor([0.5, 0.5])
    assert stub.speculative_accept(p, q, token=0, u=0.0) == (1, False), (
        "acceptance compares u < p/q strictly; <= incorrectly accepts a zero-probability token"
    )


def test_sampled_draft_token_requires_positive_q():
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.0, 1.0])
    with pytest.raises((ValueError, AssertionError)):
        stub.speculative_accept(p, q, token=0, u=0.5)
''',
["Require `q_draft[token] > 0`, then accept exactly when `u < min(1, p_target[token] / q_draft[token])`; the strict inequality matters at a zero threshold.",
 "On rejection, form `clamp(p_target - q_draft, min=0)` and normalise it.",
 "Sample from that residual with `torch.multinomial`; the accepted mass plus residual mass equals the target distribution exactly."])

S["p16"] = ("Compute softmax(scores) @ values block by block without materialising all probabilities.",
'''
def online_softmax_weighted_sum(scores, values, block=4):
    """scores: (N,), values: (N,D) -> (D,)."""
    raise NotImplementedError
''',
'''
def test_matches_naive_for_large_scores_and_many_block_sizes():
    torch.manual_seed(0)
    scores = torch.randn(37, dtype=torch.float64) * 20
    values = torch.randn(37, 5, dtype=torch.float64)
    want = F.softmax(scores, dim=0) @ values
    for block in (1, 4, 11, 64):
        got = stub.online_softmax_weighted_sum(scores, values, block)
        assert torch.allclose(got, want, atol=1e-10), block
''',
["Carry a running max `m`, denominator `l`, and weighted numerator `acc`.",
 "When the max changes, rescale old state with `exp(m_old - m_new)`.",
 "Apply that correction to both `l` and `acc`; initialise tensors from `scores`/`values` so device and dtype are preserved."])

S["p17"] = ("Implement tiled exact attention with the online-softmax recurrence.",
'''
def flash_attention_forward(q, k, v, block_q=16, block_kv=16, causal=True):
    """q,k,v: (B,H,T,D). Return (output, row_logsumexp)."""
    raise NotImplementedError
''',
'''
def test_matches_sdpa_and_is_block_size_invariant():
    torch.manual_seed(0)
    q = torch.randn(2, 2, 23, 8, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    for bq, bkv in ((1, 1), (7, 5), (16, 16), (32, 9)):
        got, lse = stub.flash_attention_forward(q, k, v, bq, bkv, causal=True)
        assert lse.shape == (2, 2, 23)
        assert torch.allclose(got, want, atol=1e-10), (bq, bkv)


@pytest.mark.parametrize("dtype,atol", [
    (torch.float16, 2e-3),
    (torch.bfloat16, 2e-2),
])
def test_mixed_precision_accumulates_in_float32_and_casts_output(dtype, atol):
    torch.manual_seed(1)
    q = torch.randn(1, 2, 19, 8).to(dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    want = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True
    ).to(dtype)
    got, lse = stub.flash_attention_forward(q, k, v, 7, 5, causal=True)
    assert got.dtype == dtype
    assert lse.dtype == torch.float32
    assert torch.allclose(got, want, atol=atol, rtol=atol)
''',
["Outer-loop over query tiles, inner-loop over KV tiles; keep `m`, `l`, and `acc` per query row.",
 "Use the recurrence from p16 and place V in the numerator update. For fp16/bf16 inputs, compute block scores and keep `m`, `l`, and `acc` in float32.",
 "For causality, skip tiles wholly above the diagonal and element-mask overlapping tiles; guard the `-inf - -inf` fully-masked case, then cast the output back to the input dtype."])

S["p20"] = ("Implement DPO from four sequence log-probabilities.",
'''
def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """All inputs have shape (B,); return a scalar mean."""
    raise NotImplementedError
''',
'''
def test_reference_policy_is_log_two_and_ordering_is_right():
    z = torch.zeros(4)
    at_ref = stub.dpo_loss(z, z, z, z)
    assert torch.allclose(at_ref, torch.tensor(math.log(2.0)), atol=1e-6)
    better = stub.dpo_loss(torch.full((4,), 2.0), z, z, z)
    worse = stub.dpo_loss(z, torch.full((4,), 2.0), z, z)
    assert better < at_ref < worse


def test_matches_reference():
    torch.manual_seed(0)
    xs = [torch.randn(9) for _ in range(4)]
    assert torch.allclose(stub.dpo_loss(*xs, beta=0.3), R.dpo_loss(*xs, beta=0.3))
''',
["Compute policy and reference chosen-minus-rejected log-ratios.",
 "The margin is `(pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)`.",
 "Return `-logsigmoid(beta * margin).mean()`; zero margin must give exactly log(2)."])

S["p21"] = ("Implement generalised advantage estimation and its returns.",
'''
def compute_gae(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    """rewards, values: (T,). Return (advantages, returns)."""
    raise NotImplementedError
''',
'''
def test_lambda_limits_and_reference_match():
    rewards = torch.tensor([1.0, 0.0, 2.0, 1.0], dtype=torch.float64)
    values = torch.tensor([0.5, 0.4, 0.3, 0.2], dtype=torch.float64)
    for lam in (0.0, 0.37, 1.0):
        got = stub.compute_gae(rewards, values, gamma=0.99, lam=lam)
        want = R.compute_gae(rewards, values, gamma=0.99, lam=lam)
        assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(got, want))
    adv0, _ = stub.compute_gae(rewards, values, gamma=0.99, lam=0.0)
    td = rewards + 0.99 * torch.cat([values[1:], values.new_zeros(1)]) - values
    assert torch.allclose(adv0, td)
''',
["`delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`.",
 "`A_t = delta_t + gamma * lambda * A_{t+1}`, so iterate backwards.",
 "Return both advantages and `advantages + values`; preserve the input device and dtype."])

S["p23"] = ("Implement top-1 MoE routing with capacity and the Switch balancing loss.",
'''
def top1_route(logits, capacity):
    """Return (expert_index, selected_gate, kept_mask), each with leading token axis."""
    raise NotImplementedError


def load_balancing_loss(logits):
    """Switch loss E * sum_e fraction_e * mean_probability_e."""
    raise NotImplementedError
''',
'''
def test_capacity_keeps_the_most_confident_tokens():
    logits = torch.tensor([
        [3.0, 0.0], [1.0, 0.0], [5.0, 0.0], [0.0, 4.0], [0.0, 2.0]
    ])
    expert, gate, kept = stub.top1_route(logits, capacity=2)
    assert torch.equal(expert, torch.tensor([0, 0, 0, 1, 1]))
    assert kept.tolist() == [True, False, True, True, True]
    assert torch.all((gate >= 0) & (gate <= 1))


def test_balance_loss_prefers_uniform_routing():
    torch.manual_seed(0)
    balanced = torch.randn(400, 4) * 0.01
    skewed = torch.full((400, 4), -5.0)
    skewed[:, 0] = 5.0
    assert stub.load_balancing_loss(balanced) < stub.load_balancing_loss(skewed)
    assert torch.allclose(stub.load_balancing_loss(balanced),
                          R.load_balancing_loss(balanced), atol=1e-6)
''',
["Softmax router logits, then take each token's maximum probability and expert index.",
 "For each expert, keep at most `capacity` assigned tokens — the most confident ones, not merely the first ones.",
 "Balance loss is `E * sum(fraction_routed.detach-like * mean_gate_probability)`; hard routing supplies load while gradients flow through probabilities."])

S["p26"] = ("Filter unreliable annotators, while protecting sparse but plausible annotators.",
'''
def filter_annotations(labels, annotators, min_agreement=0.6, min_items=3):
    """Return (clean_items, flagged_annotators)."""
    raise NotImplementedError
''',
'''
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
''',
["Take the non-None per-item majority, then measure each annotator's agreement with it.",
 "Only flag below-threshold annotators who labelled at least `min_items`; otherwise the estimate is too noisy.",
 "Recompute each retained item label after removing flagged annotators, and skip rows with no remaining votes."])


def write(force=False):
    (HERE / "stubs" / ".pristine").mkdir(parents=True, exist_ok=True)
    n = 0
    for pid, (brief, body, test, hints) in S.items():
        p = BY_ID[pid]
        base = f"{p.id}_{p.name}"
        head = HEADER.format(id=p.id, title=p.title, minutes=p.minutes, name=p.name,
                             cold="  [cold-start set]" if p.cold else "",
                             seen=f"  [reported: {p.seen}]" if p.seen else "",
                             brief=brief)
        txt = head + body.strip() + "\n"
        (HERE / "stubs" / ".pristine" / f"{base}.py").write_text(txt, encoding="utf-8")
        t = HERE / "stubs" / f"{base}.py"
        if force or not t.exists():
            t.write_text(txt, encoding="utf-8")

        tf = (f'"""Tests for {p.id} · {p.title}. Run: python run.py {p.id}"""\n\n'
              "import math\nimport sys\nfrom pathlib import Path\n\n"
              "import numpy as np\nimport pytest\nimport torch\nimport torch.nn as nn\n"
              "import torch.nn.functional as F\n\n"
              "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
              "import reference as R  # noqa: E402,F401\n"
              f"from stubs import {base} as stub  # noqa: E402\n\n" + test.strip() + "\n")
        (HERE / "tests" / f"test_{base}.py").write_text(tf, encoding="utf-8")

        h = (f"# Hints · {p.id} {p.title}\n\nRead one at a time.\n\n"
             + "\n\n".join(f"## Level {i}\n\n{x}" for i, x in enumerate(hints, 1)) + "\n")
        (HERE / "hints" / f"{base}.md").write_text(h, encoding="utf-8")
        n += 1
    print(f"generated {n} problems (stubs skipped where they already exist)")


if __name__ == "__main__":
    write(force="--force" in sys.argv)
