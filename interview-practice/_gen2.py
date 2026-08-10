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
    import inspect
    src = inspect.getsource(stub.nearest_neighbour)
    body = "\\n".join(l for l in src.split("\\n") if not l.strip().startswith("#"))
    assert "for " not in body, "vectorise it: the point of the question is the matmul trick"


def test_scales_to_a_large_input():
    rng = np.random.default_rng(1)
    tr_x, te_x = rng.normal(size=(3000, 20)), rng.normal(size=(2000, 20))
    tr_y = rng.integers(0, 5, 3000)
    out = stub.nearest_neighbour(tr_x, tr_y, te_x)
    assert out.shape == (2000,)
''',
["Expand the square: ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2.",
 "The cross term is a single matmul, test_x @ train_x.T, of shape (m, n).",
 "Add the two norm vectors with explicit broadcasting: (m,1) + (1,n). You never need sqrt, because argmin is invariant to it."])

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


def test_running_stats_are_buffers_not_parameters():
    m = stub.BatchNorm1dScratch(4)
    names = {n for n, _ in m.named_parameters()}
    assert "running_mean" not in names, "running stats must be buffers, not parameters"
    assert "running_mean" in dict(m.named_buffers())
''',
["Train mode uses batch statistics; eval mode uses the running ones. They compute different functions.",
 "register_buffer, not nn.Parameter — they move with .to(device) and are saved, but get no gradient.",
 "PyTorch normalises with the biased variance (/n) but accumulates the unbiased one (/(n-1)). Mismatch this and only eval mode diverges."])

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


def test_ratio_is_one_on_policy():
    logp, _, ref, rewards, mask = _batch()
    loss = stub.grpo_loss(logp, logp.clone(), ref, rewards, mask, group_size=4, beta=0.0)
    r = rewards.view(-1, 4)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).reshape(-1)
    assert torch.allclose(loss, -adv.mean(), atol=1e-3), \\
        "at ratio 1 the clipped surrogate must equal the advantage"


def test_kl_term_is_non_negative():
    logp, old, ref, rewards, mask = _batch()
    with_kl = stub.grpo_loss(logp, old, ref, rewards, mask, 4, beta=1.0)
    without = stub.grpo_loss(logp, old, ref, rewards, mask, 4, beta=0.0)
    assert float(with_kl) >= float(without) - 1e-6, \\
        "the k3 estimator is non-negative per sample, so adding it cannot lower the loss"
''',
["Advantage: reshape rewards to (-1, G), standardise within the group, reshape back, broadcast over tokens.",
 "ratio = (logp - logp_old).exp(); the clipped surrogate is -min(ratio*adv, clamp(ratio)*adv).",
 "k3 KL: with log_ratio = logp_ref - logp, it is log_ratio.exp() - log_ratio - 1. Mask, sum, divide by mask.sum()."])

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


def test_median_converges_but_mean_does_not():
    x = stub.light_source_samples(400_000)
    assert abs(np.median(x)) < 0.02, "the median should estimate the location parameter"
    means = [abs(stub.light_source_samples(n, seed=s).mean())
             for s, n in enumerate([10_000, 100_000, 400_000])]
    assert min(means) > 0.5, f"sample means {means} are converging; Cauchy has no mean"
''',
["The angle is Uniform(-pi/2, pi/2) and the hit position is tan(theta).",
 "Transform the density: f_X(x) = f_theta(theta) |d theta / dx| = 1 / (pi (1 + x^2)).",
 "When you compare a histogram to the PDF on a finite window, density=True normalises over that window while the true Cauchy has only (2/pi)arctan(L) of its mass there."])


SIMPLE = {   # problems whose test is just "match the reference"
"p06": ("SwiGLU feed-forward: three matrices, with the 8/3 sizing.", "SwiGLU",
        ["Gate, up, down: three projections, not two.",
         "forward is w_down(silu(w_gate(x)) * w_up(x)).",
         "d_ff = 8*d_model/3 keeps the parameter count equal to a 4x ReLU FFN."]),
"p07": ("A full pre-norm transformer block.", "Block",
        ["Two lines of forward, each a residual around a normalised sublayer.",
         "x = x + attn(norm1(x)); x = x + mlp(norm2(x)).",
         "Pre-norm normalises the sublayer input; a full model also needs a final norm before lm_head."]),
"p16": ("Streaming (online) softmax over blocks.", "online_softmax_weighted_sum",
        ["Carry a running max, a running denominator, and a running numerator.",
         "When a block reveals a larger max, rescale everything so far by exp(m_old - m_new).",
         "Both the denominator AND the accumulator need the correction — forgetting the accumulator is the classic bug."]),
"p17": ("Tiled FlashAttention forward pass.", "flash_attention_forward",
        ["It is the online softmax from p16, with V inside the loop and tiling over both axes.",
         "Keep per-query-block running statistics; iterate over key/value blocks.",
         "With a causal mask, skip whole tiles above the diagonal and only mask elementwise on the diagonal tiles."]),
"p20": ("The DPO loss from four log-probabilities.", "dpo_loss",
        ["The margin is (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected).",
         "Loss is -logsigmoid(beta * margin), averaged.",
         "Sanity check: at the reference policy the margin is 0 and the loss is exactly log 2."]),
"p21": ("Generalised advantage estimation.", "compute_gae",
        ["delta_t = r_t + gamma * V(s_{t+1}) - V(s_t).",
         "A_t = delta_t + gamma * lambda * A_{t+1}, so the loop runs backwards.",
         "Assert the limits: lambda=1 is Monte Carlo, lambda=0 is one-step TD."]),
"p13": ("MLP backward by hand.", "mlp_backward",
        ["Work backwards through down-projection, activation, up-projection.",
         "Each gradient has the shape of the tensor it belongs to; that fixes every contraction.",
         "The bias gradient sums over the batch, because broadcasting forward means summing backward."]),
"p09": ("SFT label masking over a padded batch.", "build_sft_labels",
        ["Mask each example's own prompt length, not a slice of the batch dimension.",
         "labels[i, :prompt_lens[i]] = -100, in a loop over the batch.",
         "Padding must be masked too: labels[attention_mask == 0] = -100."]),
"p15": ("Speculative decoding accept/reject.", "speculative_accept",
        ["Accept with probability min(1, p(x)/q(x)).",
         "On rejection, sample from the normalised residual max(0, p - q).",
         "This is exact: the emitted distribution is provably p. Verify it by sampling a few hundred thousand times."]),
"p23": ("Top-1 MoE routing with a capacity limit, plus the balance loss.", "top1_route",
        ["Softmax the router logits, take the argmax, and assign in order until an expert is full.",
         "Overflowing tokens skip the layer entirely and pass through the residual.",
         "The Switch loss is E * sum_e f_e * p_e, minimised at uniform routing where it equals 1."]),
"p26": ("Filter unreliable human annotations.", "filter_annotations",
        ["Per item, take the majority label; per annotator, measure agreement with it.",
         "Flag annotators below the agreement threshold — but only if they labelled enough items.",
         "Without a minimum item count you flag every sparse annotator and throw away good data."]),
"p11": ("A minimal reverse-mode autograd.", "Value",
        ["Each node stores data, grad, its children, and a closure that pushes gradient to them.",
         "Accumulate with += , not = : a node used twice receives gradient from both paths.",
         "backward() needs a reverse topological order, built with a DFS post-order."]),
"p12": ("Attention backward by hand.", "attention_backward",
        ["d_v = P^T d_out and d_p = d_out V^T are the easy two.",
         "The softmax VJP is d_s = P * (d_p - rowsum(d_p * P)).",
         "Masked positions have P = 0, so their gradient is zeroed automatically — no need to re-apply the mask."]),
}

for pid, (brief, sym, hints) in SIMPLE.items():
    is_cls = sym[0].isupper()
    stub = (f"class {sym}:\n    def __init__(self, *a, **kw):\n        raise NotImplementedError\n"
            if is_cls else
            f"def {sym}(*args, **kwargs):\n    \"\"\"Signature matches reference.{sym}.\"\"\"\n"
            f"    raise NotImplementedError\n")
    test = (f'''
def test_exists():
    assert hasattr(stub, "{sym}"), "define {sym}"


def test_matches_reference():
    """Compare against the validated reference. See reference.py for the exact API."""
    import inspect
    assert inspect.signature(stub.{sym}) == inspect.signature(R.{sym}), (
        "signature differs from the reference; match it so the harness can call yours")
''')
    S[pid] = (brief, stub, test, hints)


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
              "import sys\nfrom pathlib import Path\n\n"
              "import numpy as np\nimport torch\nimport torch.nn as nn\n"
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
