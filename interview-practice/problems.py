"""Single source of truth for every practice problem and debug drill.

The blog's tables and exercise links, the timed runner, and the repository validator all
import this module. ``minutes`` is an interview-sized practice budget, not a claim about
any employer's exact format. ``cold`` marks the small weekly, from-an-empty-file set.

Attribution is deliberately conservative. ``reported_*`` only says that a public,
anecdotal account associated the prompt with a lab; it never means an official question
bank or a verified frequency count.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class P:
    id: str
    name: str
    minutes: int
    cold: bool
    section: str
    title_en: str
    title_zh: str
    reported_en: str = ""
    reported_zh: str = ""

    # Backwards-compatible names used by the runner and generators.
    @property
    def title(self):
        return self.title_en

    @property
    def seen(self):
        return self.reported_en


PROBLEMS = [
    # --- B1 numpy / pytorch fundamentals -----------------------------------------------
    P("p24", "nn_vectorized", 15, True, "B1",
      "1-NN in pure NumPy, no loops", "纯 NumPy 的 1-NN，不许循环"),
    P("p25", "batchnorm", 20, False, "B1",
      "BatchNorm forward, gradients, and eval mode", "BatchNorm 前向、梯度与 eval 模式",
      "Personal anecdotal report: Datadog", "作者个人轶事性面经：Datadog"),
    # --- B2 transformer components -----------------------------------------------------
    P("p01", "mha", 20, True, "B2",
      "Causal multi-head attention", "因果多头注意力"),
    P("p02", "kv_cache", 15, True, "B2",
      "KV cache and incremental decode", "KV cache 与增量解码"),
    P("p03", "gqa", 10, False, "B2",
      "Grouped-query attention", "分组查询注意力",
      "Personal anecdotal report: Datadog", "作者个人轶事性面经：Datadog"),
    P("p28", "mla", 25, False, "B2",
      "Multi-head latent attention and compressed cache", "多头潜在注意力与压缩缓存"),
    P("p04", "rope", 15, True, "B2",
      "Rotary position embeddings", "旋转位置编码"),
    P("p05", "rmsnorm", 5, True, "B2", "RMSNorm", "RMSNorm"),
    P("p06", "swiglu", 5, False, "B2",
      "SwiGLU feed-forward", "SwiGLU 前馈层"),
    P("p07", "transformer_block", 15, False, "B2",
      "A full pre-norm block", "完整的 pre-norm block"),
    # --- B3 training loop --------------------------------------------------------------
    P("p08", "cross_entropy", 10, True, "B3",
      "Cross entropy with log-sum-exp", "交叉熵与 log-sum-exp"),
    P("p09", "loss_masking", 20, False, "B3",
      "SFT loss masking and packing", "SFT loss masking 与 packing"),
    P("p10", "training_loop", 20, True, "B3",
      "Overfit a tiny batch", "把一个小 batch 过拟合"),
    P("p26", "data_filtering", 20, False, "B3",
      "Filter bad human annotations", "过滤劣质人工标注"),
    # --- B4 backward by hand -----------------------------------------------------------
    P("p11", "autograd", 30, False, "B4",
      "A minimal scalar autograd", "最小标量 autograd"),
    P("p12", "attention_backward", 25, False, "B4",
      "Attention backward by hand", "手写 attention 反向"),
    P("p13", "mlp_backward", 15, False, "B4",
      "MLP backward by hand", "手写 MLP 反向"),
    # --- B5 inference and sampling -----------------------------------------------------
    P("p14", "sampling", 15, True, "B5",
      "Temperature, top-k, top-p", "temperature / top-k / top-p"),
    P("p15", "speculative", 20, False, "B5",
      "Speculative decoding accept/reject", "投机解码的接受/拒绝"),
    # --- B6 efficiency -----------------------------------------------------------------
    P("p16", "online_softmax", 15, False, "B6",
      "Streaming softmax", "流式 softmax"),
    P("p17", "flash_attention", 25, False, "B6",
      "Tiled FlashAttention forward", "分块 FlashAttention 前向"),
    # --- B7 post-training --------------------------------------------------------------
    P("p18", "lora", 10, True, "B7",
      "LoRA with a lossless merge", "LoRA 与无损合并"),
    P("p19", "grpo_loss", 20, True, "B7",
      "GRPO objective", "GRPO 目标"),
    P("p20", "dpo_loss", 15, False, "B7", "DPO loss", "DPO 损失"),
    P("p21", "gae", 15, False, "B7",
      "Generalised advantage estimation", "广义优势估计（GAE）"),
    # --- B8 data and tokenization ------------------------------------------------------
    P("p22", "bpe", 20, True, "B8", "Byte-pair encoding", "Byte-pair encoding"),
    P("p23", "moe_routing", 20, False, "B8",
      "Top-1 MoE routing with capacity", "带容量的 top-1 MoE 路由"),
    # --- C2 simulate then verify -------------------------------------------------------
    P("p27", "cauchy_simulation", 20, False, "C2",
      "Spinning light source → Cauchy", "旋转光源 → Cauchy 分布"),
]

BY_ID = {p.id: p for p in PROBLEMS}
BY_NAME = {p.name: p for p in PROBLEMS}
COLD = [p for p in PROBLEMS if p.cold]


@dataclass(frozen=True)
class D:
    id: str
    name: str
    minutes: int
    title_en: str
    title_zh: str
    symptom_en: str
    symptom_zh: str
    reported_en: str = ""
    reported_zh: str = ""

    @property
    def title(self):
        return self.title_en

    # Legacy attribute used by the table generator; this is a symptom, not the answer.
    @property
    def bug(self):
        return self.symptom_en

    @property
    def seen(self):
        return self.reported_en


DRILLS = [
    D("d09", "minigpt", 35, "Debug miniGPT", "调试 miniGPT",
      "four model invariants fail; then implement a KV cache",
      "四条模型不变量失败；随后实现 KV cache",
      "OpenAI-style; based on anecdotal reports", "OpenAI 风格；基于轶事性面经"),
    D("d10", "grpo_loop", 30, "Debug a GRPO loop", "调试 GRPO 循环",
      "sampling, advantages, and policy ratios violate invariants",
      "采样、优势值和策略比率违反不变量",
      "Anthropic-style; based on anecdotal reports", "Anthropic 风格；基于轶事性面经"),
    D("d01", "mask_inverted", 3, "Causal-mask failure", "因果 mask 故障",
      "attention can see the future", "attention 看到了未来"),
    D("d02", "missing_contiguous", 3, "Head-merge failure", "合并注意力头故障",
      "head merging raises or interleaves values", "合并注意力头时报错或数值交错"),
    D("d03", "top_p_off_by_one", 4, "Nucleus-support failure", "nucleus 支持集故障",
      "the nucleus has the wrong support", "nucleus 的支持集不正确"),
    D("d04", "cache_mask_offset", 5, "Cached-mask failure", "缓存 mask 故障",
      "cached decode cannot attend to legal history", "缓存解码看不到本应可见的历史"),
    D("d05", "lora_both_random", 3, "LoRA initialisation failure", "LoRA 初始化故障",
      "a fresh adapter changes the base model", "新建 adapter 改变了基础模型"),
    D("d06", "softmax_overflow", 3, "Softmax stability failure", "softmax 稳定性故障",
      "large finite logits produce non-finite probabilities", "有限的大 logits 产生非有限概率"),
    D("d07", "wrong_scale", 3, "Attention-scale failure", "attention 缩放故障",
      "attention disagrees with the scaled-dot-product definition",
      "attention 与缩放点积定义不一致"),
    D("d08", "prompt_not_masked", 4, "SFT objective failure", "SFT 目标故障",
      "changing prompt tokens changes the completion loss", "改动 prompt token 会改变 completion loss"),
]

DRILLS_BY_ID = {d.id: d for d in DRILLS}
