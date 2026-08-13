"""Reference solutions used by the blog and the practice tests.

Do not read this file while practising — that is what hints/ is for. Implementations are
checked against PyTorch or an independent behavioural oracle where one exists.

Run: python reference.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn

    return deco


# --------------------------------------------------------------------------------------
# 1. Causal multi-head attention
# --------------------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # buffer, not parameter: saved with the module but gets no gradient
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)).view(
                1, 1, max_len, max_len
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        if T > self.mask.shape[-1]:
            raise ValueError(f"sequence length {T} exceeds max_len {self.mask.shape[-1]}")
        # one fused qkv projection is typically more efficient than three small projections
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))

        y = att @ v  # (B, n_heads, T, d_head)
        # .contiguous() is required: transpose leaves a non-contiguous view that .view() rejects
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


@check("causal MHA matches F.scaled_dot_product_attention")
def _t_mha():
    B, T, C, H = 2, 6, 32, 4
    m = CausalSelfAttention(C, H, max_len=T).eval()
    x = torch.randn(B, T, C)
    mine = m(x)

    q, k, v = m.qkv(x).split(C, dim=2)
    q = q.view(B, T, H, C // H).transpose(1, 2)
    k = k.view(B, T, H, C // H).transpose(1, 2)
    v = v.view(B, T, H, C // H).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = m.proj(ref.transpose(1, 2).contiguous().view(B, T, C))
    assert torch.allclose(mine, ref, atol=1e-5), (mine - ref).abs().max()


@check("causal mask actually blocks the future")
def _t_causal():
    """Perturbing token t must not change outputs at positions < t."""
    B, T, C, H = 1, 8, 16, 2
    m = CausalSelfAttention(C, H, max_len=T).eval()
    x = torch.randn(B, T, C)
    y1 = m(x)
    x2 = x.clone()
    x2[:, -1, :] += 10.0
    y2 = m(x2)
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-6)
    assert not torch.allclose(y1[:, -1], y2[:, -1])


# --------------------------------------------------------------------------------------
# 2. Grouped-query attention + a KV cache (the "now make it fast" follow-up)
# --------------------------------------------------------------------------------------
class GroupedQueryAttention(nn.Module):
    """n_kv_heads < n_heads. n_kv_heads == 1 is MQA; == n_heads is plain MHA."""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x, cache=None):
        """cache: dict with 'k','v' of shape (B, n_kv_heads, T_past, d_head), mutated in place."""
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # expand kv heads to match q heads: repeat_interleave along the head axis
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        T_full = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # query at absolute position (T_full - T + i) may attend to keys 0..(T_full - T + i)
        causal = torch.ones(T, T_full, dtype=torch.bool, device=x.device).tril(
            diagonal=T_full - T
        )
        att = att.masked_fill(~causal, float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


@check("KV-cached incremental decode == full recompute")
def _t_kv_cache():
    B, T, C = 1, 7, 32
    m = GroupedQueryAttention(C, n_heads=4, n_kv_heads=2).eval()
    x = torch.randn(B, T, C)

    full = m(x)  # teacher forcing, one shot

    cache, outs = {}, []
    with torch.no_grad():
        for t in range(T):
            outs.append(m(x[:, t : t + 1, :], cache=cache))
    step = torch.cat(outs, dim=1)
    assert torch.allclose(full, step, atol=1e-5), (full - step).abs().max()


@check("GQA matches expanded scaled-dot-product attention")
def _t_gqa():
    B, T, D, H, Hkv = 2, 5, 32, 4, 2
    m = GroupedQueryAttention(D, n_heads=H, n_kv_heads=Hkv).eval()
    x = torch.randn(B, T, D)
    q = m.wq(x).view(B, T, H, D // H).transpose(1, 2)
    k = m.wk(x).view(B, T, Hkv, D // H).transpose(1, 2)
    v = m.wv(x).view(B, T, Hkv, D // H).transpose(1, 2)
    k = k.repeat_interleave(H // Hkv, dim=1)
    v = v.repeat_interleave(H // Hkv, dim=1)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    want = m.wo(want.transpose(1, 2).contiguous().view(B, T, D))
    assert torch.allclose(m(x), want, atol=1e-5)

    degenerate = GroupedQueryAttention(D, n_heads=H, n_kv_heads=H).eval()
    assert degenerate.n_rep == 1 and degenerate(x).shape == x.shape


# --------------------------------------------------------------------------------------
# 2b. Multi-head latent attention: cache a low-rank latent, not expanded K and V
# --------------------------------------------------------------------------------------
class MultiHeadLatentAttention(nn.Module):
    """Interview-sized MLA with a compressed KV cache and decoupled RoPE.

    This keeps the defining MLA mechanism from DeepSeek-V2: content keys and values are
    reconstructed from one low-rank latent, while a small shared positional key is cached
    separately. Production MLA additionally absorbs projection matrices into the query
    path during decode; this readable version optimises cache size, not kernel launches.
    """

    def __init__(self, d_model, n_heads, kv_rank, rope_dim):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope_dim = rope_dim
        self.nope_dim = self.d_head - rope_dim
        assert self.nope_dim > 0 and rope_dim % 2 == 0

        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.w_down = nn.Linear(d_model, kv_rank, bias=False)
        self.wk_up = nn.Linear(kv_rank, n_heads * self.nope_dim, bias=False)
        self.wv_up = nn.Linear(kv_rank, n_heads * self.d_head, bias=False)
        self.wk_rope = nn.Linear(d_model, rope_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x, cache=None):
        """x: (B,T,D); cache is a mutable dict containing compressed ``c`` and ``k_rope``."""
        B, T, _ = x.shape
        past = 0 if cache is None or "c" not in cache else cache["c"].shape[1]

        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q_nope, q_rope = q.split([self.nope_dim, self.rope_dim], dim=-1)
        c_new = self.w_down(x)                              # (B,T,kv_rank)
        k_rope_new = self.wk_rope(x).view(B, T, 1, self.rope_dim).transpose(1, 2)

        cos, sin = rope_cache(past + T, self.rope_dim)
        cos = cos[past:past + T].to(device=x.device, dtype=q.dtype)
        sin = sin[past:past + T].to(device=x.device, dtype=q.dtype)
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope_new = apply_rope(k_rope_new, cos, sin)

        if cache is not None and "c" in cache:
            c_all = torch.cat([cache["c"], c_new], dim=1)
            k_rope_all = torch.cat([cache["k_rope"], k_rope_new], dim=2)
        else:
            c_all, k_rope_all = c_new, k_rope_new
        if cache is not None:
            cache["c"], cache["k_rope"] = c_all, k_rope_all

        T_full = c_all.shape[1]
        k_nope = self.wk_up(c_all).view(
            B, T_full, self.n_heads, self.nope_dim
        ).transpose(1, 2)
        v = self.wv_up(c_all).view(
            B, T_full, self.n_heads, self.d_head
        ).transpose(1, 2)
        k = torch.cat([k_nope, k_rope_all.expand(-1, self.n_heads, -1, -1)], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.ones(T, T_full, dtype=torch.bool, device=x.device).tril(
            diagonal=T_full - T
        )
        probs = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        out = probs @ v
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


@check("MLA compressed-cache decode == full recompute")
def _t_mla():
    B, T, D, H = 1, 7, 64, 4
    m = MultiHeadLatentAttention(D, H, kv_rank=12, rope_dim=8).eval()
    x = torch.randn(B, T, D)
    full = m(x)
    cache, outs = {}, []
    with torch.no_grad():
        for t in range(T):
            outs.append(m(x[:, t:t + 1], cache))
    step = torch.cat(outs, dim=1)
    assert torch.allclose(full, step, atol=1e-5), (full - step).abs().max()
    cached_per_token = cache["c"].shape[-1] + cache["k_rope"].shape[-1]
    mha_per_token = 2 * H * (D // H)
    assert cached_per_token < mha_per_token, (cached_per_token, mha_per_token)


# --------------------------------------------------------------------------------------
# 3. RoPE (asked as "how does the model know about position?")
# --------------------------------------------------------------------------------------
def rope_cache(seq_len, d_head, base=10000.0):
    assert d_head % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (T, d_head/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, d_head). Rotates coordinate pairs (2i, 2i+1) by angle m*theta_i."""
    T = x.shape[-2]
    cos, sin = cos[:T], sin[:T]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)


@check("RoPE attention logits depend only on relative offset")
def _t_rope():
    d = 8
    cos, sin = rope_cache(64, d)
    q = torch.randn(1, 1, 1, d)
    k = torch.randn(1, 1, 1, d)

    def logit(pos_q, pos_k):
        qr = apply_rope(q, cos[pos_q : pos_q + 1], sin[pos_q : pos_q + 1])
        kr = apply_rope(k, cos[pos_k : pos_k + 1], sin[pos_k : pos_k + 1])
        return (qr * kr).sum()

    # same relative distance (3) at two different absolute positions
    assert torch.allclose(logit(5, 2), logit(20, 17), atol=1e-4)
    assert not torch.allclose(logit(5, 2), logit(5, 4), atol=1e-4)


# --------------------------------------------------------------------------------------
# 4. Online softmax -> the idea FlashAttention is built on
# --------------------------------------------------------------------------------------
def online_softmax_weighted_sum(scores, values, block=4):
    """Streaming softmax(scores) @ values without materialising the full prob vector.

    Carries (running max m, running denominator l, running numerator acc) and rescales
    on every block. This is exactly the FlashAttention recurrence, minus the tiling
    over queries and the GPU kernel.
    """
    if scores.ndim != 1 or values.ndim != 2 or values.shape[0] != scores.shape[0]:
        raise ValueError("expected scores (N,) and values (N, D)")
    if scores.numel() == 0 or block <= 0:
        raise ValueError("scores must be non-empty and block must be positive")
    n = scores.shape[0]
    m = scores.new_tensor(float("-inf"))
    l = scores.new_tensor(0.0)
    acc = values.new_zeros(values.shape[1])
    for start in range(0, n, block):
        s = scores[start : start + block]
        v = values[start : start + block]
        m_new = torch.maximum(m, s.max())
        correction = torch.exp(m - m_new)  # exp(-inf - finite) = 0 on the first block
        p = torch.exp(s - m_new)
        l = l * correction + p.sum()
        acc = acc * correction + p @ v
        m = m_new
    return acc / l


@check("online (streaming) softmax == naive softmax")
def _t_online_softmax():
    n, d = 37, 5
    scores = torch.randn(n) * 8  # large scale: catches numerical-stability bugs
    values = torch.randn(n, d)
    ref = F.softmax(scores, dim=0) @ values
    got = online_softmax_weighted_sum(scores, values, block=4)
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


# --------------------------------------------------------------------------------------
# 5. LoRA
# --------------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(base.weight.new_zeros((r, base.in_features)))
        self.B = nn.Parameter(base.weight.new_zeros((base.out_features, r)))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B starts at zero so the adapter is an exact no-op at step 0
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.T @ self.B.T * self.scaling

    def merged_weight(self):
        """Return the weight equivalent to the adapter path in eval mode."""
        return self.base.weight + (self.B @ self.A) * self.scaling

    @torch.no_grad()
    def merge(self):
        """Fold BA into the frozen weight for inference without extra LoRA matmuls."""
        self.base.weight.copy_(self.merged_weight())
        self.A.zero_()
        self.B.zero_()
        return self.base


@check("LoRA is identity at init, and merging preserves outputs")
def _t_lora():
    base = nn.Linear(16, 32, bias=False)
    x = torch.randn(4, 16)
    lora = LoRALinear(base, r=4, alpha=8)
    assert torch.allclose(lora(x), base(x), atol=1e-6), "B must init to zero"

    nn.init.normal_(lora.B, std=0.02)  # simulate a training step
    before = lora(x).clone()
    lora.merge()
    assert torch.allclose(before, lora(x), atol=1e-5)

    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    assert trainable == 4 * 16 + 32 * 4, trainable
    assert trainable < base.weight.numel()

    base64 = nn.Linear(8, 8, bias=False, dtype=torch.float64)
    lora64 = LoRALinear(base64, r=2)
    assert lora64.A.dtype == torch.float64 and lora64.B.dtype == torch.float64


# --------------------------------------------------------------------------------------
# 6. Sampling: temperature / top-k / top-p
# --------------------------------------------------------------------------------------
def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """logits: (V,). Order matters: temperature -> top-k -> top-p -> sample."""
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits must be a non-empty 1D tensor")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:  # greedy; guard against division by zero
        return int(logits.argmax())
    logits = logits / temperature

    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        k = min(top_k, logits.numel())
        kept, idx = torch.topk(logits, k)
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, kept)

    if top_p is not None:
        if top_p < 0:
            raise ValueError("top_p must be non-negative")
        if top_p < 1:
            srt, idx = torch.sort(logits, descending=True)
            probs = F.softmax(srt, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            # keep the smallest prefix whose mass exceeds p; shift so the crossing token stays
            drop = cum - probs >= top_p
            drop[0] = False  # the argmax always survives, so top_p=0 cannot empty the support
            srt = srt.masked_fill(drop, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(0, idx, srt)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@check("sampling: temperature=0 is greedy, top-k/top-p restrict support")
def _t_sampling():
    logits = torch.tensor([5.0, 4.0, 1.0, 0.5, -3.0])
    assert sample_next(logits, temperature=0) == 0

    g = torch.Generator().manual_seed(0)
    picks = {sample_next(logits, temperature=1.0, top_k=2, generator=g) for _ in range(400)}
    assert picks <= {0, 1}, picks

    # softmax([5,4,1,0.5,-3]) ~ [.71,.26,.013,.008,.0004]; p=0.9 keeps the top two
    picks = {sample_next(logits, temperature=1.0, top_p=0.9, generator=g) for _ in range(400)}
    assert picks <= {0, 1}, picks

    # top_p=1 is a no-op even when cumulative sums round to exactly one early.
    extreme = torch.tensor([700.0, 0.0, -700.0], dtype=torch.float64)
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert sample_next(extreme, generator=g1) == sample_next(extreme, top_p=1.5, generator=g2)


# --------------------------------------------------------------------------------------
# 7. Cross entropy from scratch (log-sum-exp trick)
# --------------------------------------------------------------------------------------
def cross_entropy(logits, targets, ignore_index=-100):
    """logits: (N, V), targets: (N,). Numerically stable, mean over non-ignored."""
    keep = targets != ignore_index
    if not keep.any():
        # F.cross_entropy returns NaN here; a fully masked microbatch should contribute
        # nothing instead of poisoning the whole accumulation
        return logits.sum() * 0.0
    logits, targets = logits[keep], targets[keep]
    m = logits.max(dim=-1, keepdim=True).values
    logsumexp = m.squeeze(-1) + (logits - m).exp().sum(-1).log()
    picked = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (logsumexp - picked).mean()


@check("cross entropy matches F.cross_entropy, including ignore_index")
def _t_ce():
    logits = torch.randn(12, 7) * 20  # big logits: unstable implementations overflow
    targets = torch.randint(0, 7, (12,))
    targets[3] = targets[8] = -100
    ref = F.cross_entropy(logits, targets, ignore_index=-100)
    assert torch.allclose(cross_entropy(logits, targets), ref, atol=1e-5)


# --------------------------------------------------------------------------------------
# 8. RMSNorm
# --------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """No mean subtraction and no bias -> one fewer reduction than LayerNorm."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # Promote low-precision inputs for the reduction without demoting float64.
        reduction_dtype = (
            torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        )
        xf = x.to(reduction_dtype)
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * rms).type_as(x) * self.weight.type_as(x)


@check("RMSNorm matches F.rms_norm")
def _t_rmsnorm():
    x = torch.randn(3, 4, 16)
    m = RMSNorm(16)
    assert torch.allclose(m(x), F.rms_norm(x, (16,), m.weight, 1e-6), atol=1e-6)

    x64 = torch.randn(2, 3, 16, dtype=torch.float64)
    m64 = RMSNorm(16).double()
    assert torch.allclose(
        m64(x64), F.rms_norm(x64, (16,), m64.weight, 1e-6),
        atol=1e-12, rtol=1e-12,
    )


# --------------------------------------------------------------------------------------
# 9. Top-1 MoE routing with capacity (the "how does a sparse layer work" question)
# --------------------------------------------------------------------------------------
def top1_route(logits, capacity):
    """logits: (T, E). Returns (expert_idx, gate, kept_mask) with per-expert capacity."""
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have non-empty shape (tokens, experts)")
    if not isinstance(capacity, int) or capacity < 0:
        raise ValueError("capacity must be a non-negative integer")
    gates = F.softmax(logits, dim=-1)
    gate, expert = gates.max(dim=-1)
    kept = torch.zeros_like(expert, dtype=torch.bool)
    for e in range(logits.shape[1]):
        idx = (expert == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        # under overflow, keep the most confident tokens; the rest are dropped
        order = idx[torch.argsort(gate[idx], descending=True)][:capacity]
        kept[order] = True
    return expert, gate, kept


def load_balancing_loss(logits):
    """Switch Transformer aux loss: E * sum_e (fraction of tokens to e) * (mean prob of e)."""
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have non-empty shape (tokens, experts)")
    gates = F.softmax(logits, dim=-1)
    E = logits.shape[-1]
    expert = gates.argmax(dim=-1)
    frac = torch.bincount(expert, minlength=E).float() / expert.numel()
    prob = gates.mean(dim=0)
    return E * (frac * prob).sum()


@check("MoE routing respects capacity; aux loss penalises confident collapse")
def _t_moe():
    T, E, cap = 20, 4, 3
    logits = torch.randn(T, E)
    expert, _, kept = top1_route(logits, cap)
    for e in range(E):
        assert int(((expert == e) & kept).sum()) <= cap

    balanced = load_balancing_loss(torch.zeros(400, E) + torch.randn(400, E) * 0.01)
    skewed = torch.full((400, E), -5.0)
    skewed[:, 0] = 5.0
    assert load_balancing_loss(skewed) > balanced
    assert abs(float(balanced) - 1.0) < 0.1  # uniform routing -> aux loss ~ 1


# --------------------------------------------------------------------------------------
# 10. GRPO objective
# --------------------------------------------------------------------------------------
def grpo_loss(
    logp, logp_old, logp_ref, rewards, mask, clip_eps=0.2, beta=0.04, group_size=None
):
    """logp/logp_old/logp_ref: (B, L). rewards: (B,). mask: (B, L) of 1.0 on completion tokens.

    B is (n_prompts * group_size) laid out contiguously per prompt. Advantage is a single
    scalar per completion, standardised within its group, then broadcast to every token.
    """
    B = rewards.shape[0]
    g = B if group_size is None else group_size
    if g < 1 or B % g:
        raise ValueError("group_size must be positive and divide the batch size")
    r = rewards.view(-1, g)
    # Population std is the within-sampled-group scale; a singleton has zero signal.
    spread = r.std(dim=1, keepdim=True, correction=0)
    adv = (r - r.mean(dim=1, keepdim=True)) / (spread + 1e-4)
    adv = adv.reshape(B, 1)  # broadcast over L

    ratio = (logp - logp_old).exp()
    unclipped = ratio * adv
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
    policy = -torch.min(unclipped, clipped)

    # k3 estimator: unbiased in expectation and non-negative per sample
    log_ratio = logp_ref - logp
    kl = log_ratio.exp() - log_ratio - 1.0

    per_token = policy + beta * kl
    token_count = mask.sum(dim=1)
    valid = token_count > 0
    if not valid.any():
        return per_token.sum() * 0.0
    per_completion = (per_token * mask).sum(dim=1) / token_count.clamp(min=1.0)
    return per_completion[valid].mean()


@check("GRPO: zero advantage spread -> ~zero loss; k3 KL is non-negative")
def _t_grpo():
    B, L = 8, 12
    logp = torch.randn(B, L) * 0.1
    mask = torch.ones(B, L)

    # all rewards equal -> advantage 0 -> policy term vanishes, and with logp_ref == logp
    # the KL term vanishes too. Such a group carries no relative policy signal.
    flat = grpo_loss(logp, logp.clone(), logp.clone(), torch.ones(B), mask, group_size=4)
    assert abs(float(flat)) < 1e-6, float(flat)

    log_ratio = torch.randn(500) * 0.5
    k3 = log_ratio.exp() - log_ratio - 1.0
    assert (k3 >= -1e-6).all()

    # a spread of rewards must produce a non-trivial gradient
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    logp_train = logp.clone().requires_grad_(True)
    loss = grpo_loss(logp_train, logp.detach(), logp.detach(), rewards, mask, group_size=4)
    loss.backward()
    assert logp_train.grad.abs().sum() > 0


# --------------------------------------------------------------------------------------
# 11. DPO loss
# --------------------------------------------------------------------------------------
def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """All args are summed sequence log-probs, shape (B,). No rollouts, no reward model."""
    pi_logratio = pi_chosen - pi_rejected
    ref_logratio = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()


@check("DPO: loss falls as the policy prefers chosen more than the reference does")
def _t_dpo():
    ref_c, ref_r = torch.zeros(4), torch.zeros(4)
    at_ref = dpo_loss(torch.zeros(4), torch.zeros(4), ref_c, ref_r)
    assert abs(float(at_ref) - math.log(2)) < 1e-5  # margin 0 -> -log sigmoid(0) = log 2

    better = dpo_loss(torch.full((4,), 2.0), torch.zeros(4), ref_c, ref_r)
    worse = dpo_loss(torch.zeros(4), torch.full((4,), 2.0), ref_c, ref_r)
    assert better < at_ref < worse


# --------------------------------------------------------------------------------------
# 12. GAE (PPO's other half)
# --------------------------------------------------------------------------------------
def compute_gae(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    """rewards, values: (T,). Returns (advantages, returns), both (T,)."""
    T = len(rewards)
    adv = torch.zeros_like(rewards)
    gae = rewards.new_tensor(0.0)
    last_value = torch.as_tensor(last_value, dtype=values.dtype, device=values.device)
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + values


@check("GAE: lambda=1 is Monte-Carlo, lambda=0 is one-step TD")
def _t_gae():
    rewards = torch.tensor([1.0, 0.0, 2.0, 1.0])
    values = torch.tensor([0.5, 0.4, 0.3, 0.2])
    g = 0.99

    adv1, _ = compute_gae(rewards, values, gamma=g, lam=1.0)
    mc = torch.zeros(4)
    run = 0.0
    for t in reversed(range(4)):
        run = rewards[t] + g * run
        mc[t] = run
    assert torch.allclose(adv1, mc - values, atol=1e-5)

    adv0, _ = compute_gae(rewards, values, gamma=g, lam=0.0)
    td = rewards + g * torch.cat([values[1:], torch.zeros(1)]) - values
    assert torch.allclose(adv0, td, atol=1e-5)


# --------------------------------------------------------------------------------------
# 13. A 40-line autograd (the question Yuan Meng calls "God forbid")
# --------------------------------------------------------------------------------------
class Value:
    """Scalar reverse-mode autodiff, in the spirit of Karpathy's micrograd."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad  # += , not = : a node can be used more than once
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo, seen = [], set()

        def build(v):
            if v in seen:
                return
            seen.add(v)
            for c in v._prev:
                build(c)
            topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):  # reverse topological order = correct dependency order
            v._backward()

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other**-1 if isinstance(other, Value) else Value(other) ** -1)

    def __hash__(self):
        return id(self)


@check("micrograd-style autograd matches torch.autograd, incl. reused nodes")
def _t_autograd():
    a, b = Value(-4.0), Value(2.0)
    c = a * b + b.tanh()
    d = c * c + a / b  # `a` and `b` each feed several paths
    d.backward()

    # float64 to match Python floats; float32 alone leaves a ~1e-6 gap that is not a bug
    ta = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    tc = ta * tb + torch.tanh(tb)
    (tc * tc + ta / tb).backward()

    assert abs(a.grad - float(ta.grad)) < 1e-6, (a.grad, float(ta.grad))
    assert abs(b.grad - float(tb.grad)) < 1e-6, (b.grad, float(tb.grad))


# --------------------------------------------------------------------------------------
# 14. Byte-pair encoding training loop
# --------------------------------------------------------------------------------------
def bpe_train(text, num_merges):
    ids = list(text.encode("utf-8"))
    merges = {}
    for i in range(num_merges):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best = max(counts, key=counts.get)
        if counts[best] < 2:
            break
        new_id = 256 + i
        merges[best] = new_id
        merged, j = [], 0
        while j < len(ids):
            if j + 1 < len(ids) and (ids[j], ids[j + 1]) == best:
                merged.append(new_id)
                j += 2
            else:
                merged.append(ids[j])
                j += 1
        ids = merged
    return merges


def bpe_encode(text, merges):
    ids = list(text.encode("utf-8"))
    # apply merges in the order they were learned, not by frequency in this string
    for pair, new_id in merges.items():
        out, j = [], 0
        while j < len(ids):
            if j + 1 < len(ids) and (ids[j], ids[j + 1]) == pair:
                out.append(new_id)
                j += 2
            else:
                out.append(ids[j])
                j += 1
        ids = out
    return ids


@check("BPE compresses and round-trips through a decode table")
def _t_bpe():
    text = "the theory of the theatre is the theory of the theatre" * 4
    merges = bpe_train(text, 12)
    ids = bpe_encode(text, merges)
    assert len(ids) < len(text.encode("utf-8"))

    vocab = {i: bytes([i]) for i in range(256)}
    for (p0, p1), new_id in merges.items():
        vocab[new_id] = vocab[p0] + vocab[p1]
    assert b"".join(vocab[i] for i in ids).decode("utf-8") == text


# --------------------------------------------------------------------------------------
# 15. Attention backward pass by hand (named as baseline by Sapora)
# --------------------------------------------------------------------------------------
def attention_forward(q, k, v, causal=True):
    """q,k,v: (B, H, T, D). Returns (out, cache) for the manual backward."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    s = (q @ k.transpose(-2, -1)) * scale
    if causal:
        T = q.shape[-2]
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
    p = F.softmax(s, dim=-1)
    return p @ v, (q, k, v, p, scale)


def attention_backward(d_out, cache):
    """The whole derivation is four lines plus the softmax Jacobian.

    O = P V           -> dV = P^T dO ,  dP = dO V^T
    P = softmax(S)    -> dS = P * (dP - rowsum(dP * P))
    S = QK^T * scale  -> dQ = dS K * scale ,  dK = dS^T Q * scale
    """
    q, k, v, p, scale = cache
    d_v = p.transpose(-2, -1) @ d_out
    d_p = d_out @ v.transpose(-2, -1)
    # softmax Jacobian, applied row-wise without materialising the (T, T, T) tensor
    d_s = p * (d_p - (d_p * p).sum(dim=-1, keepdim=True))
    d_q = (d_s @ k) * scale
    d_k = (d_s.transpose(-2, -1) @ q) * scale
    return d_q, d_k, d_v


@check("hand-derived attention backward matches autograd")
def _t_attn_backward():
    B, H, T, D = 2, 3, 7, 8
    q = torch.randn(B, H, T, D, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, H, T, D, dtype=torch.float64, requires_grad=True)

    out, cache = attention_forward(q, k, v, causal=True)
    d_out = torch.randn_like(out)
    out.backward(d_out)

    dq, dk, dv = attention_backward(d_out, cache)
    for mine, ref, name in ((dq, q.grad, "dQ"), (dk, k.grad, "dK"), (dv, v.grad, "dV")):
        assert torch.allclose(mine, ref, atol=1e-9), (name, (mine - ref).abs().max())


# --------------------------------------------------------------------------------------
# 16. MLP forward + backward by hand
# --------------------------------------------------------------------------------------
def mlp_forward(x, W1, b1, W2, b2):
    h = x @ W1 + b1
    a = torch.relu(h)
    return a @ W2 + b2, (x, W1, W2, h, a)


def mlp_backward(d_y, cache):
    x, W1, W2, h, a = cache
    d_W2 = a.transpose(-2, -1) @ d_y
    d_b2 = d_y.sum(dim=0)
    d_a = d_y @ W2.transpose(-2, -1)
    d_h = d_a * (h > 0)                       # ReLU'(h); the subgradient at 0 is taken as 0
    d_W1 = x.transpose(-2, -1) @ d_h
    d_b1 = d_h.sum(dim=0)
    d_x = d_h @ W1.transpose(-2, -1)
    return d_x, d_W1, d_b1, d_W2, d_b2


@check("hand-derived MLP backward matches autograd")
def _t_mlp_backward():
    N, D, Hd = 5, 4, 9
    kw = dict(dtype=torch.float64, requires_grad=True)
    x = torch.randn(N, D, **kw)
    W1 = torch.randn(D, Hd, **kw)
    b1 = torch.randn(Hd, **kw)
    W2 = torch.randn(Hd, D, **kw)
    b2 = torch.randn(D, **kw)

    y, cache = mlp_forward(x, W1, b1, W2, b2)
    d_y = torch.randn_like(y)
    y.backward(d_y)

    grads = mlp_backward(d_y, cache)
    refs = (x.grad, W1.grad, b1.grad, W2.grad, b2.grad)
    for mine, ref in zip(grads, refs):
        assert torch.allclose(mine, ref, atol=1e-9), (mine - ref).abs().max()


# --------------------------------------------------------------------------------------
# 17. Tiled FlashAttention forward (blocks over queries AND keys)
# --------------------------------------------------------------------------------------
def flash_attention_forward(q, k, v, block_q=16, block_kv=16, causal=True):
    """Never materialises the (T, T) score matrix.

    Outer loop over query blocks, inner loop over key blocks, carrying the running
    max m and denominator l per query row. This is FlashAttention's structure; a real
    kernel additionally keeps the tiles in SRAM and fuses everything into one launch.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape (B, H, T, D)")
    if q.shape[-2] == 0 or q.shape[-1] == 0 or block_q <= 0 or block_kv <= 0:
        raise ValueError("sequence/head dimensions and block sizes must be positive")
    B, H, T, D = q.shape
    acc_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros(B, H, T, D, dtype=acc_dtype, device=q.device)
    logsumexp = torch.zeros(B, H, T, dtype=acc_dtype, device=q.device)

    for i in range(0, T, block_q):
        qi = q[:, :, i : i + block_q].to(acc_dtype)        # (B, H, bq, D)
        bq = qi.shape[2]
        m = torch.full((B, H, bq), float("-inf"), dtype=acc_dtype, device=q.device)
        l = torch.zeros(B, H, bq, dtype=acc_dtype, device=q.device)
        acc = torch.zeros(B, H, bq, D, dtype=acc_dtype, device=q.device)

        for j in range(0, T, block_kv):
            if causal and j > i + bq - 1:
                break                                       # whole block is in the future
            kj = k[:, :, j : j + block_kv].to(acc_dtype)
            vj = v[:, :, j : j + block_kv].to(acc_dtype)
            s = (qi @ kj.transpose(-2, -1)) * scale         # (B, H, bq, bkv)
            if causal:
                rows = torch.arange(i, i + bq, device=q.device).unsqueeze(1)
                cols = torch.arange(j, j + kj.shape[2], device=q.device).unsqueeze(0)
                s = s.masked_fill(cols > rows, float("-inf"))

            m_new = torch.maximum(m, s.amax(dim=-1))
            # a fully-masked block leaves m at -inf; guard so -inf - -inf never becomes nan
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            correction = torch.exp(torch.where(torch.isinf(m), torch.full_like(m, -1e30), m) - m_safe)
            p = torch.exp(s - m_safe.unsqueeze(-1))
            p = torch.nan_to_num(p, nan=0.0)

            l = l * correction + p.sum(dim=-1)
            acc = acc * correction.unsqueeze(-1) + p @ vj
            m = m_new

        out[:, :, i : i + bq] = acc / l.clamp(min=1e-30).unsqueeze(-1)
        logsumexp[:, :, i : i + bq] = m + l.clamp(min=1e-30).log()

    return out.to(q.dtype), logsumexp


@check("tiled FlashAttention forward == exact attention")
def _t_flash():
    B, H, T, D = 2, 2, 40, 16
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.randn(B, H, T, D, dtype=torch.float64)
    v = torch.randn(B, H, T, D, dtype=torch.float64)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got, _ = flash_attention_forward(q, k, v, block_q=16, block_kv=16, causal=True)
    assert torch.allclose(ref, got, atol=1e-10), (ref - got).abs().max()

    # block size must not change the answer
    got2, _ = flash_attention_forward(q, k, v, block_q=7, block_kv=5, causal=True)
    assert torch.allclose(ref, got2, atol=1e-10)

    for dtype, atol in ((torch.float16, 2e-3), (torch.bfloat16, 2e-2)):
        q_low, k_low, v_low = q.float().to(dtype), k.float().to(dtype), v.float().to(dtype)
        want = F.scaled_dot_product_attention(
            q_low.float(), k_low.float(), v_low.float(), is_causal=True
        ).to(dtype)
        got, lse = flash_attention_forward(q_low, k_low, v_low, block_q=7, block_kv=5)
        assert got.dtype == dtype and lse.dtype == torch.float32
        assert torch.allclose(got, want, atol=atol, rtol=atol), dtype


# --------------------------------------------------------------------------------------
# 18. A training loop that overfits a tiny batch
# --------------------------------------------------------------------------------------
def overfit_tiny(steps=2000, lr=0.5):
    """Sanity check: a small network should memorise one fixed ten-example batch."""
    torch.manual_seed(0)
    x = torch.randn(10, 4)
    y = torch.randint(0, 3, (10,))
    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3))
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(x)
        return F.cross_entropy(logits, y).item(), (logits.argmax(-1) == y).float().mean().item()


@check("training loop drives a tiny batch to zero loss")
def _t_overfit():
    loss, acc = overfit_tiny()
    assert acc == 1.0, acc
    assert loss < 1e-3, loss


# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# 21. SwiGLU feed-forward (three matrices, not two)
# --------------------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        # 8/3 approximately matches a 4x ReLU FFN: 3*d*F ~= 2*d*4d
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


@check("SwiGLU: 8/3 sizing matches a 4x FFN parameter count")
def _t_swiglu():
    d = 96
    ff = SwiGLU(d)
    swiglu_params = sum(p.numel() for p in ff.parameters())
    vanilla = 2 * d * (4 * d)
    assert abs(swiglu_params - vanilla) / vanilla < 0.02, (swiglu_params, vanilla)
    y = ff(torch.randn(2, 5, d))
    assert y.shape == (2, 5, d)


# --------------------------------------------------------------------------------------
# 22. A full pre-norm transformer block
# --------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))     # pre-norm: residual stays an identity path
        x = x + self.mlp(self.norm2(x))
        return x


@check("pre-norm block preserves shape and stays causal")
def _t_block():
    torch.manual_seed(0)
    blk = Block(64, 8).eval()
    x = torch.randn(2, 6, 64)
    y1 = blk(x)
    assert y1.shape == x.shape
    x2 = x.clone(); x2[:, -1, :] += 10.0
    assert torch.allclose(y1[:, :-1], blk(x2)[:, :-1], atol=1e-5)


# --------------------------------------------------------------------------------------
# 23. SFT loss masking and packed-sequence metadata
# --------------------------------------------------------------------------------------
def build_sft_labels(input_ids, prompt_lens, attention_mask, ignore_index=-100):
    """Supervise response tokens only; prompt and padding are excluded."""
    labels = input_ids.clone()
    for i, n in enumerate(prompt_lens):
        labels[i, :n] = ignore_index
    labels[attention_mask == 0] = ignore_index
    return labels


def build_packed_sft_labels(input_ids, response_mask, attention_mask, ignore_index=-100):
    """Label response tokens in a packed row; response_mask is supplied by preprocessing."""
    keep = response_mask.bool() & attention_mask.bool()
    return input_ids.masked_fill(~keep, ignore_index)


def build_packed_attention(segment_ids, attention_mask):
    """Return reset position ids and a block-diagonal causal mask for packed examples."""
    B, T = segment_ids.shape
    valid = attention_mask.bool()
    positions = torch.zeros_like(segment_ids)
    for b in range(B):
        last_segment, offset = None, 0
        for t in range(T):
            if not valid[b, t]:
                last_segment, offset = None, 0
                continue
            segment = int(segment_ids[b, t])
            if segment != last_segment:
                offset = 0
                last_segment = segment
            positions[b, t] = offset
            offset += 1

    starts = valid.clone()
    starts[:, 1:] = valid[:, 1:] & (
        ~valid[:, :-1] | (segment_ids[:, 1:] != segment_ids[:, :-1])
    )
    run_ids = starts.long().cumsum(dim=1)
    same_segment = run_ids[:, :, None] == run_ids[:, None, :]
    causal = torch.ones(T, T, dtype=torch.bool, device=segment_ids.device).tril()
    allowed = (
        same_segment
        & causal[None]
        & valid[:, :, None]
        & valid[:, None, :]
    )
    return positions, allowed


@check("SFT masking and packing preserve response and document boundaries")
def _t_loss_masking():
    ids = torch.arange(24).reshape(3, 8)
    am = torch.ones(3, 8, dtype=torch.long); am[2, 6:] = 0
    lens = [3, 5, 2]
    lab = build_sft_labels(ids, lens, am)
    for i, n in enumerate(lens):
        assert (lab[i, :n] == -100).all()
        keep = lab[i, n:][am[i, n:] == 1]
        assert (keep != -100).all() and keep.numel() > 0
    assert (lab[2, 6:] == -100).all()

    packed_ids = torch.tensor([[10, 11, 12, 20, 21, 0]])
    segments = torch.tensor([[0, 0, 0, 1, 1, -1]])
    packed_am = torch.tensor([[1, 1, 1, 1, 1, 0]])
    response = torch.tensor([[0, 1, 1, 0, 1, 0]])
    packed_labels = build_packed_sft_labels(packed_ids, response, packed_am)
    assert torch.equal(packed_labels, torch.tensor([[-100, 11, 12, -100, 21, -100]]))
    positions, allowed = build_packed_attention(segments, packed_am)
    assert torch.equal(positions, torch.tensor([[0, 1, 2, 0, 1, 0]]))
    assert not allowed[0, 3, 2] and allowed[0, 4, 3] and not allowed[0, 5].any()


# --------------------------------------------------------------------------------------
# 24. Speculative decoding accept/reject (exactness is the whole point)
# --------------------------------------------------------------------------------------
def speculative_accept(p_target, q_draft, token, u):
    """Accept with prob min(1, p/q); on reject sample from the normalised residual."""
    q_token = q_draft[token]
    if q_token <= 0:
        raise ValueError("the sampled draft token must have positive probability")
    if u < min(1.0, (p_target[token] / q_token).item()):
        return int(token), True
    resid = torch.clamp(p_target - q_draft, min=0)
    resid = resid / resid.sum()
    return int(torch.multinomial(resid, 1)), False


@check("speculative decoding reproduces the target distribution exactly")
def _t_speculative():
    torch.manual_seed(0)
    V = 6
    p = torch.softmax(torch.randn(V), -1)
    q = torch.softmax(torch.randn(V), -1)
    counts = torch.zeros(V)
    N = 200_000
    draft = torch.multinomial(q, N, replacement=True)
    us = torch.rand(N)
    for t, u in zip(draft.tolist(), us.tolist()):
        tok, _ = speculative_accept(p, q, t, u)
        counts[tok] += 1
    emp = counts / N
    assert torch.allclose(emp, p, atol=0.01), (emp, p)

    p_edge = torch.tensor([0.0, 1.0])
    q_edge = torch.tensor([0.5, 0.5])
    assert speculative_accept(p_edge, q_edge, token=0, u=0.0) == (1, False)


# --------------------------------------------------------------------------------------
# 25. 1-NN in pure NumPy, no loops
# --------------------------------------------------------------------------------------
def nearest_neighbour(train_x, train_y, test_x):
    """1-NN classification, fully vectorised.

    Direct differences avoid the cancellation in ||a||^2 + ||b||^2 - 2 a.b for nearby,
    large float32 coordinates. Never loop over test points.
    """
    import numpy as np

    diff = test_x[:, None, :] - train_x[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    return train_y[np.argmin(d2, axis=1)]


@check("1-NN matches a brute-force loop and needs no explicit loop")
def _t_nn():
    import numpy as np

    rng = np.random.default_rng(0)
    tr_x, te_x = rng.normal(size=(40, 5)), rng.normal(size=(17, 5))
    tr_y = rng.integers(0, 3, 40)
    got = nearest_neighbour(tr_x, tr_y, te_x)
    want = np.array([tr_y[np.argmin(((tr_x - t) ** 2).sum(1))] for t in te_x])
    assert (got == want).all()

    # Expanded norms cancel in float32 here; direct differences retain the nearest point.
    tr_x = np.array([[100.02], [100.001]], dtype=np.float32)
    te_x = np.array([[100.0]], dtype=np.float32)
    assert nearest_neighbour(tr_x, np.array([0, 1]), te_x).item() == 1


# --------------------------------------------------------------------------------------
# 26. BatchNorm: forward, backward, and the train/eval split (Datadog)
# --------------------------------------------------------------------------------------
class BatchNorm1dScratch(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.eps, self.momentum = eps, momentum
        # buffers, not parameters: updated by a running average, not by gradient
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))

    def forward(self, x):
        if self.training:
            if x.shape[0] < 2:
                raise ValueError("BatchNorm training needs at least two values per channel")
            mean = x.mean(0)
            var = x.var(0, unbiased=False)          # biased for normalising...
            with torch.no_grad():
                n = x.shape[0]
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                # ...but the running estimate uses the unbiased variance
                self.running_var.mul_(1 - self.momentum).add_(
                    self.momentum * var * n / (n - 1))
        else:
            mean, var = self.running_mean, self.running_var
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


@check("BatchNorm matches nn.BatchNorm1d in both train and eval mode")
def _t_batchnorm():
    torch.manual_seed(0)
    mine, ref = BatchNorm1dScratch(6), nn.BatchNorm1d(6)
    for _ in range(5):
        x = torch.randn(32, 6) * 3 + 1
        assert torch.allclose(mine(x), ref(x), atol=1e-5)
    assert torch.allclose(mine.running_mean, ref.running_mean, atol=1e-5)
    assert torch.allclose(mine.running_var, ref.running_var, atol=1e-4)
    mine.eval(); ref.eval()
    x = torch.randn(8, 6)
    assert torch.allclose(mine(x), ref(x), atol=1e-5), "eval mode must use running stats"


# --------------------------------------------------------------------------------------
# 27. Filtering bad human annotations
# --------------------------------------------------------------------------------------
def filter_annotations(labels, annotators, min_agreement=0.6, min_items=3):
    """labels[i][j] is annotator j's label for item i, or None if unlabelled.

    Returns (clean_items, flagged_annotators). An annotator is flagged when they agree
    with the per-item majority less than `min_agreement` of the time, provided they
    labelled at least `min_items` — below that the estimate is noise, not evidence.
    """
    from collections import Counter, defaultdict

    majority = []
    for row in labels:
        votes = Counter(v for v in row if v is not None)
        majority.append(votes.most_common(1)[0][0] if votes else None)

    agree, total = defaultdict(int), defaultdict(int)
    for i, row in enumerate(labels):
        for j, v in enumerate(row):
            if v is None or majority[i] is None:
                continue
            total[j] += 1
            agree[j] += int(v == majority[i])

    flagged = {annotators[j] for j in total
               if total[j] >= min_items and agree[j] / total[j] < min_agreement}

    clean = []
    for i, row in enumerate(labels):
        votes = Counter(v for j, v in enumerate(row)
                        if v is not None and annotators[j] not in flagged)
        if votes:
            clean.append((i, votes.most_common(1)[0][0]))
    return clean, flagged


@check("annotation filtering drops an adversarial labeller and keeps a sparse good one")
def _t_filter():
    ann = ["good1", "good2", "bad", "sparse"]
    labels = [
        ["a", "a", "b", None],
        ["a", "a", "b", None],
        ["b", "b", "a", "b"],
        ["a", "a", "b", None],
        ["b", "b", "a", None],
    ]
    clean, flagged = filter_annotations(labels, ann)
    assert flagged == {"bad"}, flagged        # sparse has 1 item, below min_items
    assert [c[1] for c in clean] == ["a", "a", "b", "a", "b"]


# --------------------------------------------------------------------------------------
# 28. Cauchy from a spinning light source
# --------------------------------------------------------------------------------------
def light_source_samples(n, seed=0):
    """A lamp at distance 1 from an infinite wall points at a uniformly random angle.

    Where it hits the wall is x = tan(theta) with theta ~ Uniform(-pi/2, pi/2), which is
    the standard Cauchy. Its population mean does not exist, and the n-sample mean has
    the same standard Cauchy distribution for every n.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    return np.tan(theta)


def cauchy_pdf(x):
    import numpy as np

    return 1.0 / (np.pi * (1.0 + x ** 2))


@check("light-source simulation matches the Cauchy PDF and shows mean instability")
def _t_cauchy():
    import numpy as np

    x = light_source_samples(400_000)

    # The empirical density matches the analytic PDF on a bounded window -- but only
    # after correcting for truncation. density=True normalises over the bins shown,
    # while the true Cauchy puts mass (2/pi)*arctan(5) = 0.874 inside [-5, 5]; the tails
    # are heavy enough that ignoring this inflates the histogram by 14%.
    L = 5.0
    edges = np.linspace(-L, L, 41)
    hist, _ = np.histogram(x, bins=edges, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    in_range = 2 * np.arctan(L) / np.pi
    assert np.abs(hist - cauchy_pdf(centres) / in_range).max() < 0.01

    # the median is a fine estimator of the location parameter...
    assert abs(np.median(x)) < 0.02, np.median(x)

    # One deterministic illustration of the unstable mean. The proof is analytic:
    # integral |x| f(x) dx diverges; one finite simulation cannot prove non-convergence.
    means = [np.abs(light_source_samples(n, seed=s).mean())
             for s, n in enumerate([10_000, 100_000, 400_000])]
    assert min(means) > 0.5, (
        f"sample means {means} were unexpectedly small for this fixed diagnostic")

if __name__ == "__main__":
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  pass   {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL   {name}\n         {type(exc).__name__}: {exc}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    raise SystemExit(1 if failures else 0)
