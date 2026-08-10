"""The problem set: one row per exercise, used by run.py, the tests, and build.py.

`minutes` is the interview budget, not how long it takes to understand the answer.
`cold` marks the subset you should be able to write from an empty file; those are the
ones worth repeating weekly. Everything else only needs to be recallable in outline.
"""

from collections import namedtuple

# `seen` records what public interview reports say about this question: which lab was
# reported asking it and roughly how often. Empty means "standard, not specifically
# attributed". Sorting your practice by this column is the whole point of collecting it.
P = namedtuple("P", "id name minutes cold section title seen")

PROBLEMS = [
    # --- B1 numpy / pytorch fundamentals -----------------------------------------------
    P("p24", "nn_vectorized",     15, True,  "B1", "1-NN in pure NumPy, no loops", "OpenAI 3+"),
    P("p25", "batchnorm",         20, False, "B1", "BatchNorm forward, backward, eval mode", "Datadog"),
    # --- B2 transformer components -----------------------------------------------------
    P("p01", "mha",               20, True,  "B2", "Causal multi-head attention", "universal"),
    P("p02", "kv_cache",          15, True,  "B2", "KV cache and incremental decode", "OpenAI 7+ (as follow-up)"),
    P("p03", "gqa",               10, False, "B2", "Grouped-query attention", "Datadog"),
    P("p04", "rope",              15, True,  "B2", "Rotary position embeddings", ""),
    P("p05", "rmsnorm",            5, True,  "B2", "RMSNorm", ""),
    P("p06", "swiglu",             5, False, "B2", "SwiGLU feed-forward", ""),
    P("p07", "transformer_block", 15, False, "B2", "A full pre-norm block", ""),
    # --- B3 training loop --------------------------------------------------------------
    P("p08", "cross_entropy",     10, True,  "B3", "Cross entropy with log-sum-exp", ""),
    P("p09", "loss_masking",      10, False, "B3", "SFT loss masking and packing", ""),
    P("p10", "training_loop",     20, True,  "B3", "Overfit a tiny batch", ""),
    P("p26", "data_filtering",    20, False, "B3", "Filter bad human annotations", "OpenAI 2+"),
    # --- B4 backward by hand -----------------------------------------------------------
    P("p11", "autograd",          30, False, "B4", "A 40-line autograd", "OpenAI 2+"),
    P("p12", "attention_backward",25, False, "B4", "Attention backward by hand", "OpenAI"),
    P("p13", "mlp_backward",      15, False, "B4", "MLP backward by hand", ""),
    # --- B5 inference and sampling -----------------------------------------------------
    P("p14", "sampling",          15, True,  "B5", "Temperature, top-k, top-p", ""),
    P("p15", "speculative",       20, False, "B5", "Speculative decoding accept/reject", ""),
    # --- B6 efficiency -----------------------------------------------------------------
    P("p16", "online_softmax",    15, False, "B6", "Streaming softmax", ""),
    P("p17", "flash_attention",   25, False, "B6", "Tiled FlashAttention forward", ""),
    # --- B7 post-training --------------------------------------------------------------
    P("p18", "lora",              10, True,  "B7", "LoRA with a lossless merge", ""),
    P("p19", "grpo_loss",         20, True,  "B7", "GRPO objective", "OpenAI + Anthropic 4+"),
    P("p20", "dpo_loss",          15, False, "B7", "DPO loss", ""),
    P("p21", "gae",               15, False, "B7", "Generalised advantage estimation", ""),
    # --- B8 data and tokenization ------------------------------------------------------
    P("p22", "bpe",               20, True,  "B8", "Byte-pair encoding", ""),
    P("p23", "moe_routing",       20, False, "B8", "Top-1 MoE routing with capacity", ""),
    # --- C2 simulate then verify -------------------------------------------------------
    P("p27", "cauchy_simulation", 20, False, "C2", "Spinning light source -> Cauchy", "OpenAI"),
]

BY_ID = {p.id: p for p in PROBLEMS}
BY_NAME = {p.name: p for p in PROBLEMS}
COLD = [p for p in PROBLEMS if p.cold]

# --- debug drills: read the code, find the bug -----------------------------------------
D = namedtuple("D", "id name minutes bug seen")

DRILLS = [
    # The two flagship drills are full reproductions of the most-reported questions.
    D("d09", "minigpt",           35, "four planted bugs in a nanoGPT, then add a KV cache",
      "OpenAI 7+ — the single most reported ML-coding question"),
    D("d10", "grpo_loop",         30, "three planted bugs in a GRPO training script",
      "Anthropic 3+, also OpenAI"),
    # Micro-drills: one wrong line each, about three minutes apiece.
    D("d01", "mask_inverted",      3, "masked_fill fills where the mask is True", ""),
    D("d02", "missing_contiguous", 3, "view() after transpose on a non-contiguous tensor", ""),
    D("d03", "top_p_off_by_one",   4, "the token that crosses the threshold gets dropped", ""),
    D("d04", "cache_mask_offset",  5, "tril without diagonal=T_full-T during cached decode", ""),
    D("d05", "lora_both_random",   3, "B initialised randomly, so the adapter is not identity", ""),
    D("d06", "softmax_overflow",   3, "exp without subtracting the row max", ""),
    D("d07", "wrong_scale",        3, "dividing by sqrt(d_model) instead of sqrt(d_head)", ""),
    D("d08", "prompt_not_masked",  4, "loss computed over prompt tokens as well", ""),
]

DRILLS_BY_ID = {d.id: d for d in DRILLS}
