---
layout: post
title: "Interview Bank II · Coding + Math: write it, do not read it"
date: 2026-08-09 12:00:00
author: Jiaxin Zhang
description: "Complete, tested implementations for frontier-lab-style coding practice — attention, KV cache, RoPE, sampling, GRPO, BPE — plus the probability and linear algebra that comes with them, and a timed practice harness."
tags: interviews llm coding math pytorch qbank
categories: research-notes
giscus_comments: true
related_posts: false
ai_assisted: true
---

<div class="lang-switch"><strong>English</strong> · <a href="/blog/2026/interview-coding-zh/">中文</a></div>

<div class="lang-switch"><a href="/blog/2026/interview-knowledge/">I · Knowledge</a> · <strong>II · Coding + Math</strong> · <a href="/blog/2026/interview-discussion/">III · Discussion + BQ</a></div>

Part I asked whether you can *retrieve* something. This part asks whether you can
*produce* it, from an empty file, with a clock running.

Those are different skills, and the gap between them is the entire reason this page
comes with a repository attached. Reading an attention implementation until it feels
obvious does almost nothing for your ability to write one in twenty minutes. So the
code below is the explanation layer, and [`interview-practice/`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/tree/master/interview-practice) is where you
actually train — 28 problems with stubs and tests, 10 debug drills,
and a timed runner. The 11-problem cold-start set is generated from the same
manifest as this page, so those counts cannot drift.

> **How this is organised.** Each section states the concept briefly, gives a complete
> annotated implementation, lists common mistakes, and names what
> the interviewer is watching for. Then it points you at the exercise.

---

### Table of contents

- **[B0 · How to practise this](#section-b0)**
  - [B0.1 The two layers](#b0-1)
  - [B0.2 Three things that make practice work](#b0-2)
  - [B0.3 The problem set](#b0-3)
- **[B9 · Debugging](#section-b9)**
  - [B9.1 A method that works under a clock](#b9-1)
  - [B9.2 The miniGPT drill](#b9-2)
  - [B9.3 The GRPO loop drill](#b9-3)
  - [B9.4 Micro-drills](#b9-4)
- **[B1 · NumPy and PyTorch fundamentals](#section-b1)** — 2 exercises
  - [B1.1 Vectorisation: the one trick worth memorising](#b1-1)
  - [B1.2 BatchNorm, and why it has two modes](#b1-2)
  - [B1.3 The tensor semantics that cause silent bugs](#b1-3)
- **[B2 · Transformer components](#section-b2)** — 8 exercises
  - [B2.1 Causal multi-head attention](#b2-1)
  - [B2.2 KV cache and incremental decode](#b2-2)
  - [B2.3 MHA, MQA, GQA, and MLA](#b2-3)
  - [B2.4 Rotary position embeddings](#b2-4)
  - [B2.5 RMSNorm](#b2-5)
  - [B2.6 SwiGLU](#b2-6)
  - [B2.7 The block, assembled](#b2-7)
- **[B3 · The training loop](#section-b3)** — 4 exercises
  - [B3.1 Cross entropy, and why it takes logits](#b3-1)
  - [B3.2 Loss masking and packing](#b3-2)
  - [B3.3 Overfit ten examples before anything else](#b3-3)
  - [B3.4 Filtering bad annotations](#b3-4)
- **[B4 · Backward passes by hand](#section-b4)** — 3 exercises
  - [B4.1 The constraint that checks every backward](#b4-1)
  - [B4.2 A minimal scalar autograd](#b4-2)
  - [B4.3 Attention backward](#b4-3)
- **[B5 · Inference and sampling](#section-b5)** — 2 exercises
  - [B5.1 Temperature, top-k, top-p](#b5-1)
  - [B5.2 Speculative decoding](#b5-2)
- **[B6 · Efficient implementations](#section-b6)** — 2 exercises
  - [B6.1 Streaming softmax](#b6-1)
  - [B6.2 Tiled FlashAttention forward](#b6-2)
- **[B7 · Post-training algorithms](#section-b7)** — 4 exercises
  - [B7.1 LoRA](#b7-1)
  - [B7.2 The GRPO objective](#b7-2)
  - [B7.3 DPO](#b7-3)
  - [B7.4 GAE](#b7-4)
- **[B8 · Data and tokenization](#section-b8)** — 2 exercises
  - [B8.1 Byte-pair encoding](#b8-1)
  - [B8.2 Top-1 MoE routing with capacity](#b8-2)
- **[C1 · Probability: five reusable patterns](#section-c1)**
  - [C1.1 First-step analysis](#c1-1)
  - [C1.2 Indicators plus linearity of expectation](#c1-2)
  - [C1.3 Max and min of $$n$$ variables — go through the CDF](#c1-3)
  - [C1.4 Symmetry as a proof technique](#c1-4)
  - [C1.5 Which inequality to reach for](#c1-5)
- **[C2 · Simulate, then verify](#section-c2)** — 1 exercise
  - [C2.1 The spinning light source](#c2-1)
  - [C2.2 The general recipe](#c2-2)
- **[C3 · Linear algebra](#section-c3)**
  - [C3.1 The four facts everything else follows from](#c3-1)
  - [C3.2 Positive semi-definiteness, and why it keeps appearing](#c3-2)
  - [C3.3 Norms, conditioning, and the things that blow up](#c3-3)
  - [C3.4 The matrix calculus you actually need](#c3-4)
- **[C4 · Counting](#section-c4)**
  - [C4.1 The one decision that determines the formula](#c4-1)
  - [C4.2 Overcount, then divide](#c4-2)
  - [C4.3 Inclusion–exclusion](#c4-3)
  - [C4.4 Where counting meets ML](#c4-4)
- **[C5 · Markov chains and random walks](#section-c5)**
  - [C5.1 What the Markov property actually buys you](#c5-1)
  - [C5.2 Gambler's ruin](#c5-2)
  - [C5.3 Random walks, and the dimension surprise](#c5-3)
  - [C5.4 Portable worked examples](#c5-4)
- **[C6 · Statistics and estimation](#section-c6)**
  - [C6.1 Maximum likelihood, and what your loss function really is](#c6-1)
  - [C6.2 Bias, variance, and when a biased estimator is useful](#c6-2)
  - [C6.3 Concentration: how many samples do you need](#c6-3)
  - [C6.4 Hypothesis testing, briefly, and its ML failure mode](#c6-4)
- **[References](#section-refs)**

---
<a id="section-b0"></a>

## B0 · How to practise this

Reading a correct implementation tests recognition, not recall. Timed practice adds a
different requirement: produce the implementation and its tests without relying on the
answer in front of you.

So this page is deliberately only half the material. The other half is a repository.

---

<a id="b0-1"></a>
### B0.1 The two layers

**This page is the explanation layer.** Coding sections give tested implementations,
common mistakes, and the invariant each exercise checks; math sections add derivations
and closed-book self-tests. Read once to build the model, then return when an exercise
defeats you.

**[`interview-practice/`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/tree/master/interview-practice)
is the training layer.** It contains the same problems as stubs — signature and
docstring, body removed — with focused behavioural tests and a validated reference.
The clickable entry points are [`run.py`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py),
[`README.md`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/README.md),
and the generated problem table in B0.3.

Python blocks excerpted from `reference.py` assume its shared prelude:

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
```

Functions that use NumPy import it locally. The full file is directly executable; the
blog blocks and bilingual copies are synchronised from it during the build.

```bash
git clone https://github.com/jxzhangjhu/jxzhangjhu.github.io.git
cd jxzhangjhu.github.io/interview-practice

python run.py                 # the problem set, budgets, and your history
python run.py p01             # start the clock, run that problem's tests
python run.py --cold          # the from-an-empty-file set, in order
python run.py --reset p01     # restore the pristine stub and redo it
```

Every **Exercise** row links the stub, all three hint levels, the symptom-oriented tests,
and the runner while retaining the copyable shell command. The `pNN` identifiers are
what `run.py` takes as arguments.

Tests are written to diagnose rather than merely fail. When causal masking leaks you get
*"perturbing the last token changed earlier outputs: the mask leaks the future"*, not a
tensor dump.

> **On the reference solutions.** They live in `interview-practice/reference.py`.
> Its own numerical self-checks and `_validate.py` must both pass before the page is
> built. Do not open that file while practising; that is what the graded hints are for.

---

<a id="b0-2"></a>
### B0.2 Three things that make practice work

**Put a clock on it.** Every problem carries a budget — 20 minutes for multi-head
attention, 5 for RMSNorm, 30 for a small autograd. Timed practice additionally exposes
retrieval and implementation errors: forgetting `.contiguous()`, inverting a mask, or
dividing by the wrong square root.

**Keep a small cold-start set.** Only the manifest entries marked `cold` need to come
from an empty file, and they are worth repeating weekly; `python run.py list` prints the
current set and count. For everything else the bar is lower — recall the shape of the
solution and reconstruct it with a hint. The entire bank need not be maintained at the
same from-memory intensity.

**Drill the bugs separately, and do it first.** First-hand preparation accounts explicitly
include ML debugging rounds, so the bank trains construction and fault recognition
separately and puts B9 immediately after this setup. `d09` and `d10` are **OpenAI-style**
and **Anthropic-style** syntheses of anecdotal reports, not official or verbatim
questions. The remaining entries are one-line micro-drills. Section B9 explains the planted
bugs after you have attempted them.

---

<a id="b0-3"></a>
### B0.3 The problem set

`cold` marks the from-an-empty-file set. Budgets are interview budgets, not how long it
takes to understand the answer.

**Treat "reported in" as provenance, not frequency.** It records only that a public,
anecdotal account associated a similar prompt with a lab. It is not an official bank and
does not justify a probability estimate. A blank means "standard exercise, not
specifically attributed here," not "unimportant."

<!-- TABLE -->

| | Problem | Budget | Cold | Reported in |
|---|---|---|---|---|
| **B1 · NumPy and PyTorch** | | | | |
| [p24](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p24_nn_vectorized.py) | [1-NN in pure NumPy, no loops](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p24_nn_vectorized.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p24_nn_vectorized.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p24_nn_vectorized.py) · [`python run.py p24`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 min | ● |  |
| [p25](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p25_batchnorm.py) | [BatchNorm forward, gradients, and eval mode](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p25_batchnorm.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p25_batchnorm.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p25_batchnorm.py) · [`python run.py p25`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 min |  | Personal anecdotal report: Datadog |
| **B2 · Components** | | | | |
| [p01](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p01_mha.py) | [Causal multi-head attention](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p01_mha.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p01_mha.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p01_mha.py) · [`python run.py p01`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 min | ● |  |
| [p02](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p02_kv_cache.py) | [KV cache and incremental decode](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p02_kv_cache.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p02_kv_cache.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p02_kv_cache.py) · [`python run.py p02`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 min | ● |  |
| [p03](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p03_gqa.py) | [Grouped-query attention](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p03_gqa.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p03_gqa.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p03_gqa.py) · [`python run.py p03`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 10 min |  | Personal anecdotal report: Datadog |
| [p28](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p28_mla.py) | [Multi-head latent attention and compressed cache](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p28_mla.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p28_mla.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p28_mla.py) · [`python run.py p28`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 25 min |  |  |
| [p04](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p04_rope.py) | [Rotary position embeddings](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p04_rope.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p04_rope.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p04_rope.py) · [`python run.py p04`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 min | ● |  |
| [p05](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p05_rmsnorm.py) | [RMSNorm](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p05_rmsnorm.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p05_rmsnorm.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p05_rmsnorm.py) · [`python run.py p05`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 5 min | ● |  |
| [p06](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p06_swiglu.py) | [SwiGLU feed-forward](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p06_swiglu.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p06_swiglu.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p06_swiglu.py) · [`python run.py p06`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 5 min |  |  |
| [p07](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p07_transformer_block.py) | [A full pre-norm block](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p07_transformer_block.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p07_transformer_block.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p07_transformer_block.py) · [`python run.py p07`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 min |  |  |
| **B3 · Training** | | | | |
| [p08](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p08_cross_entropy.py) | [Cross entropy with log-sum-exp](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p08_cross_entropy.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p08_cross_entropy.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p08_cross_entropy.py) · [`python run.py p08`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 10 min | ● |  |
| [p09](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p09_loss_masking.py) | [SFT loss masking and packing](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p09_loss_masking.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p09_loss_masking.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p09_loss_masking.py) · [`python run.py p09`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 min |  |  |
| [p10](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p10_training_loop.py) | [Overfit a tiny batch](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p10_training_loop.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p10_training_loop.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p10_training_loop.py) · [`python run.py p10`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 min | ● |  |
| [p26](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p26_data_filtering.py) | [Filter bad human annotations](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p26_data_filtering.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p26_data_filtering.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p26_data_filtering.py) · [`python run.py p26`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 min |  |  |
| **B4 · Backward** | | | | |
| [p11](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p11_autograd.py) | [A minimal scalar autograd](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p11_autograd.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p11_autograd.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p11_autograd.py) · [`python run.py p11`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 30 min |  |  |
| [p12](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p12_attention_backward.py) | [Attention backward by hand](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p12_attention_backward.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p12_attention_backward.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p12_attention_backward.py) · [`python run.py p12`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 25 min |  |  |
| [p13](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p13_mlp_backward.py) | [MLP backward by hand](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p13_mlp_backward.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p13_mlp_backward.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p13_mlp_backward.py) · [`python run.py p13`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 min |  |  |
| **B5 · Inference** | | | | |
| [p14](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p14_sampling.py) | [Temperature, top-k, top-p](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p14_sampling.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p14_sampling.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p14_sampling.py) · [`python run.py p14`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 min | ● |  |
| [p15](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p15_speculative.py) | [Speculative decoding accept/reject](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p15_speculative.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p15_speculative.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p15_speculative.py) · [`python run.py p15`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 min |  |  |
| **B6 · Efficiency** | | | | |
| [p16](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p16_online_softmax.py) | [Streaming softmax](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p16_online_softmax.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p16_online_softmax.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p16_online_softmax.py) · [`python run.py p16`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 min |  |  |
| [p17](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p17_flash_attention.py) | [Tiled FlashAttention forward](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p17_flash_attention.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p17_flash_attention.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p17_flash_attention.py) · [`python run.py p17`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 25 min |  |  |
| **B7 · Post-training** | | | | |
| [p18](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p18_lora.py) | [LoRA with a lossless merge](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p18_lora.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p18_lora.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p18_lora.py) · [`python run.py p18`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 10 min | ● |  |
| [p19](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p19_grpo_loss.py) | [GRPO objective](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p19_grpo_loss.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p19_grpo_loss.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p19_grpo_loss.py) · [`python run.py p19`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 min | ● |  |
| [p20](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p20_dpo_loss.py) | [DPO loss](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p20_dpo_loss.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p20_dpo_loss.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p20_dpo_loss.py) · [`python run.py p20`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 min |  |  |
| [p21](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p21_gae.py) | [Generalised advantage estimation](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p21_gae.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p21_gae.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p21_gae.py) · [`python run.py p21`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 min |  |  |
| **B8 · Data** | | | | |
| [p22](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p22_bpe.py) | [Byte-pair encoding](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p22_bpe.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p22_bpe.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p22_bpe.py) · [`python run.py p22`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 min | ● |  |
| [p23](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p23_moe_routing.py) | [Top-1 MoE routing with capacity](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p23_moe_routing.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p23_moe_routing.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p23_moe_routing.py) · [`python run.py p23`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 min |  |  |
| **C2 · Simulation** | | | | |
| [p27](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p27_cauchy_simulation.py) | [Spinning light source → Cauchy](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p27_cauchy_simulation.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p27_cauchy_simulation.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p27_cauchy_simulation.py) · [`python run.py p27`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 min |  |  |

| | Drill | Budget | Symptom | Reported in |
|---|---|---|---|---|
| [d09](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d09_minigpt.py) | [Debug miniGPT](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d09_minigpt.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d09_minigpt.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d09_minigpt.py) · [`python run.py --drill d09`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 35 min | four model invariants fail; then implement a KV cache | OpenAI-style; based on anecdotal reports |
| [d10](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d10_grpo_loop.py) | [Debug a GRPO loop](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d10_grpo_loop.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d10_grpo_loop.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d10_grpo_loop.py) · [`python run.py --drill d10`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 30 min | sampling, advantages, and policy ratios violate invariants | Anthropic-style; based on anecdotal reports |
| [d01](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d01_mask_inverted.py) | [Causal-mask failure](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d01_mask_inverted.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d01_mask_inverted.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d01_mask_inverted.py) · [`python run.py --drill d01`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 3 min | attention can see the future |  |
| [d02](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d02_missing_contiguous.py) | [Head-merge failure](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d02_missing_contiguous.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d02_missing_contiguous.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d02_missing_contiguous.py) · [`python run.py --drill d02`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 3 min | head merging raises or interleaves values |  |
| [d03](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d03_top_p_off_by_one.py) | [Nucleus-support failure](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d03_top_p_off_by_one.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d03_top_p_off_by_one.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d03_top_p_off_by_one.py) · [`python run.py --drill d03`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 4 min | the nucleus has the wrong support |  |
| [d04](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d04_cache_mask_offset.py) | [Cached-mask failure](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d04_cache_mask_offset.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d04_cache_mask_offset.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d04_cache_mask_offset.py) · [`python run.py --drill d04`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 5 min | cached decode cannot attend to legal history |  |
| [d05](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d05_lora_both_random.py) | [LoRA initialisation failure](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d05_lora_both_random.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d05_lora_both_random.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d05_lora_both_random.py) · [`python run.py --drill d05`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 3 min | a fresh adapter changes the base model |  |
| [d06](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d06_softmax_overflow.py) | [Softmax stability failure](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d06_softmax_overflow.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d06_softmax_overflow.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d06_softmax_overflow.py) · [`python run.py --drill d06`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 3 min | large finite logits produce non-finite probabilities |  |
| [d07](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d07_wrong_scale.py) | [Attention-scale failure](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d07_wrong_scale.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d07_wrong_scale.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d07_wrong_scale.py) · [`python run.py --drill d07`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 3 min | attention disagrees with the scaled-dot-product definition |  |
| [d08](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d08_prompt_not_masked.py) | [SFT objective failure](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d08_prompt_not_masked.py) · [hint](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d08_prompt_not_masked.md) · [test](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d08_prompt_not_masked.py) · [`python run.py --drill d08`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 4 min | changing prompt tokens changes the completion loss |  |

<!-- TABLE -->

**A four-week rotation that fits around a job.** Week one, the two flagship drills
(`d09`, `d10`) plus the cold set once each with hints allowed — coverage, not speed.
Week two, everything outside the cold set, same rules. Week three, the cold set again with no hints
and the clock enforced; anything you miss goes on a short list. Week four, the short list
plus the micro-drills, and one full pass of the cold set the day before.

> **Avoid practising only solved items.** The runner records local outcomes and elapsed
> time in `interview-practice/attempts.local.json`; use that history to revisit failures
> and slow passes first.

---

<a id="section-b9"></a>

## B9 · Debugging

Put this section first if you are short on time.

First-hand preparation accounts explicitly mention ML debugging rounds:
[Alisa Liu](https://alisawuffles.github.io/blog/job-search/) calls out implementing and
debugging a transformer, while
[Silvia Sapora](https://silviasapora.github.io/blog/ml-interviews.html) lists spotting
bugs in training loops. This public evidence is anecdotal and does **not** support a
reliable company-by-company frequency ranking. The practical conclusion is still useful:
debugging is a distinct skill, and from-scratch implementation does not train it alone.

The drills here use a reproducible format: code runs but violates behavioural invariants;
bugs are logical rather than syntactic; and the longer drill ends with an extension after
the repairs. `d09` and `d10` are syntheses, not leaked or official questions.

---

<a id="b9-1"></a>
### B9.1 A method that works under a clock

The following method is robust under a clock; "read the code more carefully" is not.

**Reproduce deterministically first.** Seed every RNG and make the train/eval mode
explicit. For an inference mismatch, put the model in `eval()` and use greedy decoding;
for a training bug, keep `train()` but control data order and randomness. You cannot tell
whether a change helped if the output moves on its own.

**Localise with assertions, not with reading.** Print shapes at every stage. Assert the
invariants you know must hold: attention rows sum to one, cached decode equals full
recompute, and position indices equal `arange(T)` in a full forward. Each assertion
narrows the search space and leaves a reproducible failure boundary.

**Fix one bug at a time and re-run.** Bugs mask each other. In the drill below, an
end-to-end loss cannot tell you whether a failure comes from position indices or the
head/time layout; focused assertions separate them. If you fix three things then run,
you will not know which one mattered.

**Clarify the search boundary.** If regions are marked, ask whether code outside them is
in scope. Under time pressure, do not silently assume either that every comment is true
or that the entire repository must be re-audited.

**Say what class of bug you found.** "The mask is applied after the softmax, so rows do
not sum to one" communicates the mechanism and the violated invariant; "fixed it" only
describes the diff.

> **Strong preparation for this format** is to write a nanoGPT-style model end to end at
> least once, from the embedding table to the training loop. Liu explicitly recommends
> turning transformer implementation and debugging into muscle memory; Karpathy's
> [nanoGPT](https://github.com/karpathy/nanoGPT) provides a compact reference.

---

<a id="b9-2"></a>
### B9.2 The miniGPT drill

This is an **OpenAI-style synthesis** from anecdotal reports: a small decoder-only LM
with four planted bugs and a KV-cache follow-up. It is not a verbatim company prompt.

[`drill`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d09_minigpt.py)
· [`three-level hints`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d09_minigpt.md)
· [`tests`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d09_minigpt.py)
· [`runner`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py)

```bash
python run.py --drill d09                          # 35 min budget
```

The four planted bug classes are chosen to exercise distinct invariants:

| Bug | Symptom you can test for |
|---|---|
| Positional embedding indexed with a constant | Every token receives position index zero |
| Causal mask applied *after* the softmax | Attention rows no longer sum to 1 |
| Heads merged without transposing time back | Silently wrong output, no exception |
| Training loop never steps the optimiser | No parameter changes after a step |

**The third one deserves attention** because shape checks do not catch it.
The attention output is `(B, n_heads, T, d_head)` and you want `(B, T, C)`. Reshaping
directly *works* — the element count matches — and interleaves head and time. Nothing
raises. The model can train with the wrong connectivity and gives you no exception to
chase. This explains why `.transpose(1, 2).contiguous().view(...)` is written in that
order and why shape-suffixed variable names (`y_BHTD`) are useful.

**The follow-up is a KV cache.** The new token's **positional index is the cache length**,
not zero. Decode step $$t$$ must embed position $$t$$. Get that wrong and generation
degrades while a teacher-forced-only evaluation can stay clean. State the invariant out
loud — cached decode must agree numerically with a full recompute — and then test it.

---

<a id="b9-3"></a>
### B9.3 The GRPO loop drill

This is an **Anthropic-style synthesis**: a complete toy GRPO loop with two numerical
failures and one algorithmic failure. Again, the attribution is anecdotal, not official.

[`drill`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d10_grpo_loop.py)
· [`three-level hints`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d10_grpo_loop.md)
· [`tests`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d10_grpo_loop.py)
· [`runner`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py)

```bash
python run.py --drill d10                          # 30 min budget
```

**Shifted logits passed to `torch.multinomial` as weights.** The planted code subtracts
the minimum so it stays non-negative and runs, but that does not turn logits into policy
probabilities. `torch.multinomial` accepts non-negative weights, not logits. Fix:
softmax first.

**Advantage normalised by a bare standard deviation.** When every completion in a group
earns the same reward, the standard deviation is zero and the advantage is NaN, which
can propagate non-finite gradients through the next update. This can happen whenever a group is
all-correct or all-wrong; that group also carries no within-group relative reward signal
(Part I, A9.5). Use `std(correction=0) + 1e-5`: the population correction also keeps a
singleton group finite.

**The ratio computed as a log difference.** The importance ratio is
$$\exp(\log \pi_\theta - \log \pi_{\text{old}})$$. Using the difference itself is not a
ratio, and the tell is sharp: on-policy, where new and old log-probs are equal, the ratio
must be exactly 1 and the unclipped surrogate must equal the advantage. A log difference
gives zero there, putting the surrogate value and clipping regime in the wrong state at
the first update.

**A useful follow-up** is a discussion question rather than a code one:

> At the first update of an on-policy rollout, when should the importance ratio be exactly
> 1? Why can a logged minibatch mean later differ from 1?

A good answer names several causes and, for each, what you would check:

- **More than one optimiser step per rollout batch.** After the first step the policy has
  moved, so the remaining mini-epochs are off-policy by construction. Check the number of
  inner epochs. Even when the full-distribution expectation of a valid importance ratio
  is 1, a finite minibatch mean need not be.
- **The sampling engine is not the training engine.** Rollouts from vLLM and log-probs
  recomputed in HF may not agree bit for bit — different kernels, different attention
  implementations, different precision. Check by recomputing log-probs for the same
  tokens in both and diffing.
- **Sampling parameters applied at generation but not at scoring.** Temperature, top-p,
  and logit bias change the distribution you actually sampled from. If you score with the
  raw distribution, your "old" log-probs are of the wrong policy.
- **Precision and non-determinism.** fp32 versus bf16 log-prob accumulation, or fused
  versus eager attention, moves the ratio slightly even with identical weights.

The diagnostic goal is to separate **expected-by-design drift** from **an actual bug**.

---

<a id="b9-4"></a>
### B9.4 Micro-drills

The micro-drills have exactly one wrong line each and take only a few minutes apiece. They
are cheap enough to do while a build runs, and they map directly onto the bug classes
above.

| Drill | The one wrong line |
|---|---|
| `d01` | `masked_fill` fills where the mask is **True**, and the mask is inverted |
| `d02` | `.view()` after a transpose, on a non-contiguous tensor |
| `d03` | Top-p drops the token that crosses the threshold |
| `d04` | Cached decode uses `tril` without `diagonal=T_full - T` |
| `d05` | LoRA initialises both `A` and `B` randomly |
| `d06` | Softmax without subtracting the row maximum |
| `d07` | Scaling by $$\sqrt{d_\text{model}}$$ instead of $$\sqrt{d_\text{head}}$$ |
| `d08` | SFT loss computed over prompt tokens as well as the response |

> **Why micro-drills complement re-implementation.** Writing attention from scratch
> trains construction; finding an inverted mask trains fault recognition. Short drills
> provide more repetitions, while the longer drills test whether those local checks
> compose into a debugging strategy.

---

<a id="section-b1"></a>

## B1 · NumPy and PyTorch fundamentals

This section starts with vectorised 1-NN and a stateful BatchNorm implementation.
The Datadog label records the author's personal anecdotal interview experience, not an
official question bank or frequency estimate. Every later implementation depends on the same transposes,
broadcasting, and dtype discipline.

---

<a id="b1-1"></a>
### B1.1 Vectorisation: the one trick worth memorising

Implement 1-nearest-neighbour in NumPy with no loops. The point is not the classifier;
it is whether you can express all pairwise distances as array operations.

$$\|a - b\|^2 = \|a\|^2 - 2\,a \cdot b + \|b\|^2$$

The expansion turns the cross term into a fast matmul, but subtracting large, nearly
equal float32 terms can erase the very distance being compared. The teaching reference
therefore broadcasts direct differences and squares those.

```python
def nearest_neighbour(train_x, train_y, test_x):
    """1-NN classification, fully vectorised.

    Direct differences avoid the cancellation in ||a||^2 + ||b||^2 - 2 a.b for nearby,
    large float32 coordinates. Never loop over test points.
    """
    import numpy as np

    diff = test_x[:, None, :] - train_x[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    return train_y[np.argmin(d2, axis=1)]
```

**Three things to say while writing it.** You never take a square root, because
`argmin` is invariant to monotone transforms. The direct-difference broadcast has shape
`(n_test, n_train, d)`, then the feature-axis sum leaves the distance matrix. And this
choice is numerical, not algebraic: for float32 coordinates `100.02`, `100.001`, and
query `100.0`, expanded squared norms can cancel enough to select the wrong neighbour.

> **The scaling check is memory.** Direct broadcasting also materialises an
> `n_test × n_train × d` temporary. A practical exact implementation chunks test rows;
> `scipy.spatial.distance.cdist` is another vetted option. The expanded matmul uses less
> temporary memory and may be faster, but for nearby large float32 coordinates it needs
> a deliberate accuracy policy such as float64 accumulation.

<!-- EXERCISE p24 -->
**Exercise** — [`p24` · 1-NN in pure NumPy, no loops](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p24_nn_vectorized.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p24_nn_vectorized.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p24_nn_vectorized.py) · [`python run.py p24`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 min · cold-start set
<!-- /EXERCISE -->

---

<a id="b1-2"></a>
### B1.2 BatchNorm, and why it has two modes

The author personally encountered this prompt in a Datadog interview; that is anecdotal
experience, not an official Datadog question bank. It looks like a warm-up and it is not,
because the interesting part is the state, not the formula.
The technical source is [Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167).

```python
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
```

This interview-sized implementation contracts `x` as `(N, D)`. PyTorch's
`BatchNorm1d` also supports `(N, C, L)`, where statistics aggregate over `N` and `L`.

**Four semantic details.**

**Train and eval compute different functions.** In training the statistics come from the
batch; in evaluation they come from the running estimate. Training with only one value
per channel cannot estimate variance, whereas evaluation uses stored statistics.
Distributed training also needs an explicit policy for which workers share statistics;
LayerNorm has neither dependency.

**`register_buffer`, not `nn.Parameter`.** The running statistics move with `.to(device)`
and are saved in the state dict, but they receive no gradient. Making them parameters is
a semantic error even if the forward pass initially looks right.

**Biased for normalising, unbiased for the running estimate.** PyTorch normalises with
the biased variance ($$/n$$) and accumulates the unbiased one ($$/(n-1)$$). Match this or
training outputs can match while the running state, and therefore evaluation outputs,
diverge. Tests must cover both modes.

**Epsilon is inside the square root** in PyTorch's definition. Putting it outside can
still avoid division by zero, but implements a different scale, especially at small
variance, and will not match the reference layer.

> **The architectural comparison:** why do transformers generally use LayerNorm rather
> than BatchNorm? LayerNorm operates per token, has identical train/eval statistics, does
> not couple examples in a batch, and needs no cross-device batch-statistic
> synchronisation. Part I, A1.7 has the full version.

<!-- EXERCISE p25 -->
**Exercise** — [`p25` · BatchNorm forward, gradients, and eval mode](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p25_batchnorm.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p25_batchnorm.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p25_batchnorm.py) · [`python run.py p25`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 min · *Personal anecdotal report: Datadog*
<!-- /EXERCISE -->

---

<a id="b1-3"></a>
### B1.3 The tensor semantics that cause silent bugs

Four recurring semantics are worth testing explicitly.

**`view` versus `reshape`.** `view` requires contiguous memory and refuses otherwise;
`reshape` falls back to a copy. After `transpose` you are non-contiguous, so `view`
raises — which is the *good* case, because it tells you. The bad case is when the element
count happens to line up and a reshape silently interleaves the wrong axes, which is
exactly bug three in the miniGPT drill (B9.2).

**Broadcasting aligns from the right.** `(B, T, C) * (C,)` works; `(B, T, C) * (B,)` does
not in general — and can silently align to `C` if `B == C`. When you mean a per-batch
scale, write `(B, 1, 1)`. Explicit singleton dimensions make the intended axis testable.

**In-place operations and autograd.** `x += 1` on a value needed for the backward pass
can raise a leaf- or version-counter error; `x = x + 1` creates a new tensor. In-place
updates are appropriate for deliberately non-differentiated state, such as optimiser
buffers and running statistics, when wrapped with the right gradient context.

**dtype promotion is silent.** bf16 times fp32 gives fp32. That is how a normalisation
layer can quietly return the wrong dtype (B2.5), and how a "bf16" training run ends up
with fp32 activations in places you did not intend.

**Debug drills** — the micro-drills `d02`, `d06`, `d07` in B9.4 target exactly these.

---

<a id="section-b2"></a>

## B2 · Transformer components

This is the reusable baseline: implement each component under a clock, then verify its
behavioural invariants rather than relying on shape checks alone.

Everything here is in `interview-practice/reference.py` and checked against PyTorch
primitives or explicit behavioural invariants.

---

<a id="b2-1"></a>
### B2.1 Causal multi-head attention

Primary source: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

Under the usual independent, unit-variance component model, the scaling keeps logit
variance order-one so the softmax does not saturate merely as head width grows. The mask
is additive $$-\infty$$ *before* the softmax so masked positions contribute nothing to
the denominator. Both arguments are in Part I (A2.3).

```python
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
```

**Four failure modes and design choices.**

1. **`.contiguous()`.** After `transpose(1, 2)` the tensor is a view with non-contiguous
   strides, and `.view()` raises. Use `.reshape()` if you prefer, but know the difference:
   `view` never copies and therefore refuses; `reshape` falls back to a copy.
2. **Scaling by $$\sqrt{d_\text{model}}$$ instead of $$\sqrt{d_\text{head}}$$.** The dot
   product runs over the head dimension, so that is the variance being corrected. The
   wrong denominator silently changes the effective softmax temperature.
3. **Masking after the softmax.** Zeroing masked positions afterwards leaves them in the
   denominator, so surviving weights no longer sum to one, and the error varies by row.
4. **One fused versus three separate projections.** They are mathematically identical.
   One larger projection is often more efficient, but compiler and hardware determine
   the measured difference.

**Write a causality test.** Put the model in evaluation mode so dropout does not make the
test stochastic:

```python
model.eval()
y1 = model(x)
x2 = x.clone(); x2[:, -1, :] += 10.0
assert torch.allclose(y1[:, :-1], model(x2)[:, :-1])   # the past cannot see the future
```

> **What to demonstrate.** State the expected shapes, explain the stride change after a
> transpose, and test causality numerically.

<!-- EXERCISE p01 -->
**Exercise** — [`p01` · Causal multi-head attention](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p01_mha.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p01_mha.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p01_mha.py) · [`python run.py p01`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 min · cold-start set
<!-- /EXERCISE -->

---

<a id="b2-2"></a>
### B2.2 KV cache and incremental decode

At decode step $$t$$ you have one new query, but need every previous key and value.
Q is transient; K and V accumulate. At one layer, recomputing every full-prefix
attention matrix for a length-$$T$$ generation costs cubic attention work in aggregate;
cached one-row attention reduces that aggregate to quadratic work and avoids repeating
historical K/V projections.

```python
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
```

The `torch.cat` calls keep the contract readable but reallocate and copy the cache.
Production serving preallocates storage or uses paged blocks, and a grouped kernel avoids
physically repeating K/V heads.

**The mask offset is the whole question.** During prefill $$T = T_\text{full}$$ and a
plain `tril` is right. During cached decode your query block starts partway down the
matrix, so you need `diagonal=T_full - T`. Get this wrong and the model is fine in
teacher-forced evaluation and quietly degrades during generation; an evaluation suite
that only uses teacher forcing will not expose it.

**The correctness property to state out loud:** cached incremental decode must be
**numerically close** to a full recompute; floating-point kernel order can prevent
bitwise identity. That is testable, so test it.

> **What to demonstrate.** Make the cache concatenation and the offset causal mask two
> separate invariants; testing only the former misses this failure.

<!-- EXERCISE p02 -->
**Exercise** — [`p02` · KV cache and incremental decode](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p02_kv_cache.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p02_kv_cache.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p02_kv_cache.py) · [`python run.py p02`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 min · cold-start set
<!-- /EXERCISE -->

---

<a id="b2-3"></a>
### B2.3 MHA, MQA, GQA, and MLA

The author personally encountered GQA in a Datadog interview. This is an anecdotal
provenance label, not an official Datadog question bank or frequency claim.

One design axis connects the four variants: **what is retained per token in the KV
cache?** Plain MHA keeps a separate key and value for every query head. MQA shares one
K/V head across all query heads. GQA partitions query heads into groups and shares one
K/V head inside each group. With $$H$$ query heads, $$H_{kv}$$ KV heads and head width
$$d_h$$, the per-layer cache is $$2H_{kv}d_h$$ values per token.

```python
k = k.repeat_interleave(self.n_rep, dim=1)   # n_rep = n_heads // n_kv_heads
v = v.repeat_interleave(self.n_rep, dim=1)
```

$$H_{kv}=1$$ is MQA, $$H_{kv}=H$$ is plain MHA, and everything between is
[GQA](https://arxiv.org/abs/2305.13245). This is a tunable quality-versus-cache knob.

**The follow-up that catches people: this does not reduce attention FLOPs.** K and V are
expanded back to the full head count before the matmuls, so $$QK^\top$$ and $$AV$$ are
unchanged. What shrinks is the cache and the bandwidth needed to read it, and since
decode is often bandwidth-bound that is where the speedup comes from. (Be precise if
pushed: the K/V *projections* do get smaller, from $$2D^2$$ to
$$2D H_{kv} d_h$$ parameters per layer.)

<!-- EXERCISE p03 -->
**Exercise** — [`p03` · Grouped-query attention](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p03_gqa.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p03_gqa.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p03_gqa.py) · [`python run.py p03`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 10 min · *Personal anecdotal report: Datadog*
<!-- /EXERCISE -->

**[MLA](https://arxiv.org/abs/2405.04434) changes the object being cached.** Instead of
storing expanded content K and V, it stores one low-rank latent
$$c_t=W^{DKV}x_t\in\mathbb R^r$$ and reconstructs content keys and values from it. RoPE
cannot in general be pushed through that low-rank projection, so MLA also retains a
small decoupled positional key. In the interview-sized implementation below, the cache
contains `c: (B,T,r)` and `k_rope: (B,1,T,d_rope)`: $$r+d_\text{rope}$$ values per
token, versus $$2Hd_h$$ for MHA.

```python
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
```

This is a faithful implementation of the **cache representation and attention
semantics**, not a claim to reproduce DeepSeek's production kernel. Production MLA
absorbs parts of the up-projection into the query path during decode so it need not
materialise expanded keys. The readable version reconstructs them, making cache
compression easy to test while leaving kernel fusion out of scope.

> **The invariant is unchanged:** full-sequence evaluation and token-by-token cached
> decoding must agree numerically. A compressed cache that changes logits is an
> approximation; MLA itself is not.

<!-- EXERCISE p28 -->
**Exercise** — [`p28` · Multi-head latent attention and compressed cache](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p28_mla.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p28_mla.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p28_mla.py) · [`python run.py p28`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 25 min
<!-- /EXERCISE -->

---

<a id="b2-4"></a>
### B2.4 Rotary position embeddings

Primary source: [Su et al., 2021](https://arxiv.org/abs/2104.09864).

RoPE rotates each coordinate pair by an angle proportional to position, which makes the
attention logit depend only on the relative offset. The three-line proof is in Part I
(A2.6); here it is the implementation that matters.

```python
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
```

**Three details.** It is applied to **Q and K only**, after the head split and before the
dot product — not to V. With a KV cache you store the **post-rotation** keys. And the
pairing convention (`0::2, 1::2` versus split-half) must match between the table and the
application, or the logits silently implement a different positional transform.

<!-- EXERCISE p04 -->
**Exercise** — [`p04` · Rotary position embeddings](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p04_rope.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p04_rope.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p04_rope.py) · [`python run.py p04`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 min · cold-start set
<!-- /EXERCISE -->

---

<a id="b2-5"></a>
### B2.5 RMSNorm

Primary source: [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467).

```python
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
```

The interesting choice is the reduction dtype. Low-precision inputs are promoted to
fp32, while float64 inputs remain float64. There is no mean subtraction and no bias —
the RMSNorm paper reports that re-scaling alone can match LayerNorm in its experiments,
while dropping re-centering saves a reduction.

> **This is a numerical choice, not a stylistic one.** An implementation that reduces in
> bf16 can pass fp32-only tests while accumulating much larger rounding error in mixed
> precision. Conversely, blindly calling `x.float()` demotes float64. The exercise tests
> both directions, including bf16 inputs whose squared values have magnitude around
> $$10^4$$.

<!-- EXERCISE p05 -->
**Exercise** — [`p05` · RMSNorm](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p05_rmsnorm.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p05_rmsnorm.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p05_rmsnorm.py) · [`python run.py p05`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 5 min · cold-start set
<!-- /EXERCISE -->

---

<a id="b2-6"></a>
### B2.6 SwiGLU

Primary source: [Shazeer, 2020](https://arxiv.org/abs/2002.05202).

```python
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
```

**Three matrices, not two.** Choosing $$F \approx \tfrac83 D$$ approximately matches the
matrix parameter count of a bias-free vanilla $$4D$$ FFN. The integer truncation in this
reference makes it approximate; production implementations often round width to a
hardware-friendly multiple.

<!-- EXERCISE p06 -->
**Exercise** — [`p06` · SwiGLU feed-forward](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p06_swiglu.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p06_swiglu.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p06_swiglu.py) · [`python run.py p06`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 5 min
<!-- /EXERCISE -->

---

<a id="b2-7"></a>
### B2.7 The block, assembled

```python
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
```

Two lines of forward. Pre-norm normalises the sublayer *input*, leaving a clean identity
path that improves gradient flow through deep stacks. The residual stream can still grow
with depth, so a standard full model includes a **final norm before `lm_head`**.
Pre-norm reduces optimisation fragility; it does not by itself guarantee that warmup is
unnecessary.

<!-- EXERCISE p07 -->
**Exercise** — [`p07` · A full pre-norm block](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p07_transformer_block.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p07_transformer_block.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p07_transformer_block.py) · [`python run.py p07`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 min
<!-- /EXERCISE -->

---

<a id="section-b3"></a>

## B3 · The training loop

A training loop often serves as the harness around a component or as the object of a
debugging exercise. One of this bank's four miniGPT bugs lives here, so the loop is worth
being able to audit even when it is not the primary implementation task.

---

<a id="b3-1"></a>
### B3.1 Cross entropy, and why it takes logits

```python
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
```

**Why the API takes logits rather than probabilities.** A hand-written
`log(exp(x) / exp(x).sum())` can overflow on large logits; even a stable softmax can
underflow small probabilities before the following log. Working in log space subtracts
the row maximum and avoids materialising probabilities. With logits around $$10^4$$,
the reference remains finite and agrees with `F.cross_entropy` within floating-point
tolerance.

**The `ignore_index` reduction:** divide by the number of *kept*
tokens, not by $$N$$. Masking then averaging over everything silently scales your loss by
the keep fraction, which then interacts with your learning rate.

**And the case where nothing is kept.** A packed microbatch can end up fully masked, and
without the early guard the mean reduction returns NaN — as does `F.cross_entropy` —
which can poison the next update through non-finite gradients. Return a zero that is still
attached to the graph instead.

<!-- EXERCISE p08 -->
**Exercise** — [`p08` · Cross entropy with log-sum-exp](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p08_cross_entropy.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p08_cross_entropy.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p08_cross_entropy.py) · [`python run.py p08`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 10 min · cold-start set
<!-- /EXERCISE -->

---

<a id="b3-2"></a>
### B3.2 Loss masking and packing

Two things that look like plumbing and are actually correctness.

```python
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
```

**The whiteboard slip to avoid:** `labels[:len(prompt_ids)] = -100` on a `(B, T)` tensor
slices the **batch** dimension, wiping out the first few examples entirely rather than
masking each example's prompt. It runs, it trains, and it is wrong.

**Packing** concatenates short examples into one fixed-length sequence to avoid padding
waste, which can dominate length-skewed batches. The catch is that tokens can then attend
across the document boundary. Either use a varlen kernel (FlashAttention with
`cu_seqlens`) or a block-diagonal mask; and reset `position_ids` per document, or
document two starts at position 512.

The reference makes that contract executable. `response_mask` comes from preprocessing
and marks only target-side tokens; `segment_ids` marks which contiguous document run each
position belongs to; `build_packed_attention` returns reset positions and a `(B, T, T)` boolean mask
that is both causal and block diagonal. Padding has neither an allowed row nor column.
Production kernels encode the same boundaries compactly instead of materialising this
teaching mask. Reusing a numeric segment ID later still starts a new run. These label
helpers also assume the causal-LM implementation shifts logits and labels internally;
in a manual loss, align token $$t$$ logits with token $$t+1$$ labels explicitly.

<!-- EXERCISE p09 -->
**Exercise** — [`p09` · SFT loss masking and packing](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p09_loss_masking.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p09_loss_masking.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p09_loss_masking.py) · [`python run.py p09`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 min
<!-- /EXERCISE -->

---

<a id="b3-3"></a>
### B3.3 Overfit ten examples before anything else

```python
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
```

**Use memorisation as a controlled diagnostic.** With a sufficiently expressive model,
a fixed tiny batch, and sane optimiser settings, failure to approach zero loss points to
the data, objective, gradient path, update step, or model state. Because it removes most
data and capacity uncertainty, this is a cheap, high-signal check before a full run.

**The three-line order has semantics.** `zero_grad` → `backward` → `step`.
Gradients *accumulate* by default, so skipping `zero_grad` sums every step's gradient;
skipping `step` means nothing updates and your loss curve is flat; and calling `step`
before `backward` either has no fresh gradient or consumes one left from an earlier step.

> **Why gradients accumulate:** one parameter can receive contributions along multiple
> graph paths, and micro-batch gradient accumulation relies on the same semantics. The
> training loop must therefore decide explicitly when to clear them.

<!-- EXERCISE p10 -->
**Exercise** — [`p10` · Overfit a tiny batch](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p10_training_loop.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p10_training_loop.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p10_training_loop.py) · [`python run.py p10`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 min · cold-start set
<!-- /EXERCISE -->

---

<a id="b3-4"></a>
### B3.4 Filtering bad annotations

This is not mainly a modelling question — it is whether you can reason about label noise
without over-engineering.

```python
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
```

**The key statistical guard is `min_items`.** An annotator who labelled two items and
disagreed on both looks terrible, but two samples is noise, not evidence. Without a
minimum the heuristic can flag sparse annotators on too little evidence.

**This is a teaching heuristic, not a production estimator.** Scoring each annotator
against a majority that includes their own vote is circular; leave-one-annotator-out
agreement or gold items reduce that leakage. If bad annotators form a majority, consensus
is wrong. Disagreement is not the same as error on ambiguous items, so item difficulty
and annotator quality are confounded; Dawid–Skene-style models estimate them jointly.
Groups of systematically correlated annotators can also survive majority averaging.

<!-- EXERCISE p26 -->
**Exercise** — [`p26` · Filter bad human annotations](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p26_data_filtering.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p26_data_filtering.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p26_data_filtering.py) · [`python run.py p26`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 min
<!-- /EXERCISE -->

---

<a id="section-b4"></a>

## B4 · Backward passes by hand

The useful challenge is not "do you remember the chain rule" — it is whether you can
track shapes while applying it and explain what a framework is doing for you.

---

<a id="b4-1"></a>
### B4.1 The constraint that checks every backward

Use the local derivative and chain rule; then enforce one non-negotiable check:

> **The gradient with respect to any tensor has that tensor's shape**, and it is
> assembled from the incoming gradient and local operands.

Shape alone does not prove a derivative, but it rules out many incorrect transposes and
reductions. Differentials supply the values; shapes constrain the legal contractions.

For $$Z = XW + b$$ with $$X: (m, n_\text{in})$$ and $$W: (n_\text{in}, n_\text{out})$$:

$$\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Z}W^\top,\qquad
\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Z},\qquad
\frac{\partial L}{\partial b}=\sum_i \frac{\partial L}{\partial z_{i}}$$

Check them by shape: $$(m, n_\text{out}) \times (n_\text{out}, n_\text{in})$$ gives
$$X$$'s shape; $$(n_\text{in}, m) \times (m, n_\text{out})$$ gives $$W$$'s. The bias
gradient sums over the batch axis because broadcasting in the forward means summation
over broadcast axes in the backward.

**Why dense backward costs about twice dense forward.** A linear layer computes two
similar-size products in backward instead of one: gradients for the input and weights.
Counting one multiply-add as two FLOPs gives the common core-model estimate
$$2N + 4N = 6N$$ FLOPs per token for forward plus backward (Part I, A10.0), before
optimizer work, recomputation, sparsity, or non-matmul operations.

---

<a id="b4-2"></a>
### B4.2 A minimal scalar autograd

```python
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
```

This is deliberately a scalar teaching engine: it omits tensor broadcasting, gradient
reset APIs, graph retention, and higher-order derivatives.

**Two things carry the whole answer.**

**`+=` rather than `=`.** A node used in two places receives gradient from both paths,
and the multivariable chain rule says they add. Assignment silently keeps only the last
one — and the bug is invisible on any graph where every node is used once, which is
exactly the graph you would test with.

**Reverse topological order.** A node's backward can only run once every consumer has
contributed. Executing callbacks immediately during a traversal fails on diamond graphs;
the code uses DFS only to build a postorder, then reverses that list.

> **The follow-up: why does PyTorch build the graph dynamically?** Because the graph is
> just whatever operations ran, recorded as they run — which is why control flow, loops,
> and data-dependent shapes work eagerly. `torch.compile` captures and optimises guarded
> regions of that execution; changed guards can trigger graph breaks or recompilation.

<!-- EXERCISE p11 -->
**Exercise** — [`p11` · A minimal scalar autograd](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p11_autograd.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p11_autograd.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p11_autograd.py) · [`python run.py p11`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 30 min
<!-- /EXERCISE -->

---

<a id="b4-3"></a>
### B4.3 Attention backward

The point of doing this by hand is the softmax Jacobian, which is the only non-obvious
piece.

For $$P = \text{softmax}(S)$$ row-wise, the Jacobian is
$$\partial p_i/\partial s_j = p_i(\delta_{ij} - p_j)$$, so the vector-Jacobian product
collapses to something you can write in one line without ever materialising the
$$T \times T \times T$$ Jacobian:

$$\frac{\partial L}{\partial S} = P \odot \left(\frac{\partial L}{\partial P}
- \Big(\sum_j \frac{\partial L}{\partial p_j} p_j\Big)\right)$$

```python
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
```

**What the masked positions do.** They have $$p = 0$$, so `p * (...)` zeroes their
gradient automatically. Provided every query has at least one valid key, you do not need
to re-apply the mask in this backward.

**Why this matters beyond the interview:** FlashAttention uses the same backward algebra
in tiles, regenerating $$P$$ on chip from $$Q$$ and $$K$$ rather than storing the full
attention matrix in HBM (B6.2).

<!-- EXERCISE p12 -->
**Exercise** — [`p12` · Attention backward by hand](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p12_attention_backward.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p12_attention_backward.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p12_attention_backward.py) · [`python run.py p12`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 25 min
<!-- /EXERCISE -->

**The MLP case is the same method with fewer axes.** Cache pre-activation `h` because
ReLU's derivative depends on its sign; sum bias gradients over the broadcast batch
dimension; and check every returned gradient against the shape of its primal tensor.
This compact implementation contracts `x` and `d_y` as two-dimensional `(N, D)` tensors.

```python
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
```

The useful self-check is not one hand-computed example. Generate random float64 tensors,
feed a random upstream gradient to both implementations, and compare all five gradients
against `torch.autograd`. That catches transposes and missing bias reductions.

<!-- EXERCISE p13 -->
**Exercise** — [`p13` · MLP backward by hand](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p13_mlp_backward.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p13_mlp_backward.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p13_mlp_backward.py) · [`python run.py p13`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 min
<!-- /EXERCISE -->

---

<a id="section-b5"></a>

## B5 · Inference and sampling

Short section, two questions, and both are more subtle than they look.

---

<a id="b5-1"></a>
### B5.1 Temperature, top-k, top-p

```python
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
```

**Order matters: temperature, then top-k, then top-p.** Temperature changes the
distribution the truncations act on, so applying it last selects a nucleus from the wrong
distribution.

**The top-p off-by-one.** You want the
*shortest prefix whose cumulative mass reaches p*, which means the token that crosses the
threshold is **kept**. `cum - probs` is the exclusive cumulative sum — the mass strictly
before this token — and dropping where that already exceeds `p` gets it right. Writing
`cum >= top_p` instead drops the crossing token, and with `p = 0.9` on a distribution like
`[0.5, 0.3, 0.15, 0.05]` you silently sample from two tokens instead of three.

**`top_p >= 1` is a no-op.** Skip nucleus filtering rather than trusting a cumulative
sum: with extreme finite logits, floating-point cumulative mass can round to one before
the last token and accidentally remove finite-logit support.

**`temperature == 0` needs an explicit branch**, or you divide by zero.

**Top-k should keep exactly k indices.** Masking everything below the kth value keeps
more than k tokens when logits tie at the boundary. `topk` plus `scatter` makes the
tie-breaking explicit and preserves an exact support-size contract.

**Top-p's adaptive support:** when probability mass is concentrated, the nucleus is
small; when it is diffuse, the nucleus widens. Top-k fixes support size instead. Neither
dominates universally—the useful comparison is which failure mode the application can
tolerate.

<!-- EXERCISE p14 -->
**Exercise** — [`p14` · Temperature, top-k, top-p](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p14_sampling.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p14_sampling.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p14_sampling.py) · [`python run.py p14`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 min · cold-start set
<!-- /EXERCISE -->

---

<a id="b5-2"></a>
### B5.2 Speculative decoding

Primary sources: [Leviathan et al. (2022)](https://arxiv.org/abs/2211.17192) and
[Chen et al. (2023)](https://arxiv.org/abs/2302.01318).

The interesting property is that it is **exact** — it does not approximate the target
model's distribution under the acceptance-and-correction assumptions below.

```python
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
```

The function is one position's correction kernel. The full algorithm has a draft model
propose $$K$$ tokens autoregressively, has the target score those positions in one
forward pass, and walks left to right until the first rejection. At position $$i$$,
$$p_i$$ and $$q_i$$ are distributions conditioned on the already accepted prefix. A
rejection emits from the residual and discards the remaining draft suffix; if every draft
token is accepted, the target emits one additional token.

The kernel assumes normalised `p_target` and `q_draft`, a draft token with positive
`q_draft[token]`, and a uniform `u` in `[0, 1)`. It enforces positive draft mass before
division and uses the strict event `u < p/q`; with `p[token] = 0` and `u = 0`, `<=`
would incorrectly emit a token to which the target assigns zero probability.

Accept the draft's token with probability $$\min(1, p(x)/q(x))$$; on rejection, sample
from the normalised residual $$\propto \max(0, p - q)$$. This is rejection sampling, and
the resulting samples are provably distributed as $$p$$.

**Prove it in one line if asked.** The probability of emitting $$x$$ is
$$q(x)\min(1, p/q) + P(\text{reject})\cdot\frac{\max(0, p-q)}{\sum_y \max(0, p-q)}$$. The
first term is $$\min(q, p)$$, and the second supplies exactly the missing $$\max(0, p-q)$$,
summing to $$p(x)$$. The normaliser works because normalised $$p$$ and $$q$$ satisfy
$$\sum (p-q)_+ = \sum (q-p)_+ = P(\text{reject})$$.

> **The test worth writing.** Sample 200,000 times and compare the empirical distribution
> to the target. A finite Monte Carlo test cannot prove exactness, but it checks the
> implementation against the derived target; the reference does this.

**Where the speedup comes from, and where it goes.** Decode is bandwidth-bound with idle
FLOPs at small batch sizes, so verifying $$k$$ draft tokens in one parallel forward can
cost far less than $$k$$ serial target steps. As batch size grows, target verification
becomes more compute-bound and the acceptance-adjusted speedup can shrink or turn
negative. Latency and throughput gains therefore depend on the draft, acceptance rate,
batching policy, sequence length, and hardware; benchmark the serving regime you need.

<!-- EXERCISE p15 -->
**Exercise** — [`p15` · Speculative decoding accept/reject](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p15_speculative.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p15_speculative.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p15_speculative.py) · [`python run.py p15`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 min
<!-- /EXERCISE -->

---

<a id="section-b6"></a>

## B6 · Efficient implementations

The two problems share one recurrence: softmax normalisation can be updated block by
block. IO-aware kernels combine that recurrence with tiling and fusion to avoid storing
the full attention matrix.

---

<a id="b6-1"></a>
### B6.1 Streaming softmax

Softmax looks like it needs a full pass before you can normalise anything — you need the
max for stability and the sum for the denominator. It does not. Keep a running max
$$m$$, a running denominator $$\ell$$, and a running numerator, and rescale whenever a
new block reveals a larger maximum.

```python
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
```

**The correction factor is the whole algorithm.** Everything accumulated so far was
computed relative to the old maximum; multiplying by $$e^{m_\text{old}-m_\text{new}}$$
re-expresses it relative to the new one. Both $$\ell$$ and the accumulator need it, and
forgetting the accumulator is a useful failure test — the denominator is then right and the
numerator is not, which produces a plausible-looking but wrong result.

**This is algebraically exact**, not an approximation. Floating-point block order can
still change the last bits, so assert agreement with a naive softmax within a dtype-aware
tolerance; the reference implementation does.

> **Provenance:** this recurrence is from
> [Milakov & Gimelshein (2018)](https://arxiv.org/abs/1805.02867) and
> predates FlashAttention. FlashAttention's contribution is not the recurrence, it is the
> IO-aware tiling built on top of it.

<!-- EXERCISE p16 -->
**Exercise** — [`p16` · Streaming softmax](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p16_online_softmax.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p16_online_softmax.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p16_online_softmax.py) · [`python run.py p16`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 min
<!-- /EXERCISE -->

---

<a id="b6-2"></a>
### B6.2 Tiled FlashAttention forward

Now apply the recurrence with $$V$$ in the loop, tiling over both query and key blocks,
and you have the structure of
[FlashAttention's](https://arxiv.org/abs/2205.14135) forward pass.

```python
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
```

This readable contract covers dense self-attention with equal `q`, `k`, and `v` shapes
and no padding mask. For fp16 or bf16 input it computes block logits and keeps `m`,
`l`, and `acc` in fp32, then casts the attention output back to the input dtype;
row log-sum-exp remains fp32. Production kernels apply the same accumulation policy
inside fused operations.

**Three things to say about it.**

**Auxiliary memory goes from $$O(N^2)$$ to $$O(N)$$** for fixed head width because the
score matrix is never materialised. This Python implementation expresses the tiling but
does not control memory placement; a fused GPU kernel keeps the active tiles on chip.

**FLOPs go *up*, not down.** The backward pass recomputes the attention matrix on-chip
rather than reading a stored $$N\times N$$ matrix. FlashAttention reduces memory traffic,
not the arithmetic count of exact dense attention.

**It can still be faster when the operation is bound by HBM traffic, not arithmetic.**
Trading FLOPs for memory traffic is a win on the memory-bound side of the roofline; the
speedup depends on shape, precision, kernel, and hardware.

**The causal optimisation worth mentioning:** with a causal mask, tiles entirely above
the diagonal can be skipped outright, tiles entirely below it need no mask, and only
tiles that cross the causal boundary need an elementwise mask. For long square sequences,
this nearly halves score-tile work, though end-to-end speedup is smaller and
kernel-dependent.

<!-- EXERCISE p17 -->
**Exercise** — [`p17` · Tiled FlashAttention forward](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p17_flash_attention.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p17_flash_attention.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p17_flash_attention.py) · [`python run.py p17`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 25 min
<!-- /EXERCISE -->

---

<a id="section-b7"></a>

## B7 · Post-training algorithms

Section B9.3 supplies the explicitly labelled Anthropic-style GRPO debugging drill.
Write the objective once from scratch first — you cannot reliably debug an objective
you have never assembled.

---

<a id="b7-1"></a>
### B7.1 LoRA

Primary source: [Hu et al., 2021](https://arxiv.org/abs/2106.09685).

```python
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
```

**Two properties are being checked.** Identity at initialisation, which requires `B = 0`
in this parameterisation — initialising both factors randomly changes the starting
function. And a **lossless merge**: the adapted layer is just a weight matrix, so after
merging there are no additional LoRA matrix multiplications, unlike adapter layers which
add depth. With nonzero LoRA dropout, this equivalence is the evaluation-mode path;
training remains stochastic.

**Where the memory saving comes from** — not the weights. The base still has to be
resident. Under a common mixed-precision AdamW accounting—bf16 weight and gradient,
fp32 master weight, and two fp32 moments—full fine-tuning uses about 16 bytes per
parameter. With the base frozen, only its 2-byte bf16 weight remains while the other
states apply to the adapter. For a 70B base, that changes the base-related footprint from
about 1,120 GB to 140 GB, plus adapter state. This back-of-the-envelope figure excludes
activations, temporary buffers, quantisation, and distributed sharding.

<!-- EXERCISE p18 -->
**Exercise** — [`p18` · LoRA with a lossless merge](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p18_lora.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p18_lora.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p18_lora.py) · [`python run.py p18`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 10 min · cold-start set
<!-- /EXERCISE -->

---

<a id="b7-2"></a>
### B7.2 The GRPO objective

Primary source: [DeepSeekMath](https://arxiv.org/abs/2402.03300).

```python
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
```

**Four implementation details.** The Anthropic-style drill in B9.3
exercises the first two here plus sampling from logits; the KL and credit-assignment
points are conceptual follow-ups.

**The epsilon in the denominator is not cosmetic.** A group where every completion earns
the same reward has zero standard deviation; this can happen when a policy solves all or
none of a prompt's samples. Without the epsilon you get NaN. Use population
`std(correction=0)`: then a singleton group has zero spread and zero relative advantage,
rather than producing NaN before epsilon is added.

**The ratio is `exp` of the log difference.** On-policy, where new and old log-probs
coincide, the ratio must be exactly 1 and the unclipped surrogate must equal the
advantage. A bare log difference gives zero there, so both the surrogate value and
clipping regime are wrong at the first update.

**This formulation uses a per-token KL term in the loss**, rather than folding it into a
shaped reward, and it
uses Schulman's k3 estimator: with $$r = \pi_\text{ref}/\pi_\theta$$ sampled from
$$\pi_\theta$$, $$\widehat{\mathrm{KL}} = r - \log r - 1$$. It is unbiased *and*
non-negative per sample, whereas the naive $$-\log r$$ can come out negative on a single
sample even though its expectation is the KL.

**The advantage is bandit-shaped.** One scalar per completion, broadcast to every token —
there is no reward-derived per-token credit assignment.

**Preserve original GRPO's sequence-level reduction.** The reference averages valid
tokens within each completion, then averages completions with at least one valid token.
DAPO's global-token reduction is a distinct, intentional variant: it divides by the
total number of valid response tokens across the batch. That is not inherently wrong,
but it changes length weighting by giving longer completions more influence than under
the original sequence-level mean.

<!-- EXERCISE p19 -->
**Exercise** — [`p19` · GRPO objective](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p19_grpo_loss.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p19_grpo_loss.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p19_grpo_loss.py) · [`python run.py p19`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 min · cold-start set
<!-- /EXERCISE -->

---

<a id="b7-3"></a>
### B7.3 DPO

Primary source: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290).

```python
def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """All args are summed sequence log-probs, shape (B,). No rollouts, no reward model."""
    pi_logratio = pi_chosen - pi_rejected
    ref_logratio = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()
```

Four log-probs, one sigmoid. No reward model, critic, or online generation is required in
the training loop. If the frozen reference model stays resident, it adds roughly another
model-weight footprint; precomputing reference log-probs trades that memory for storage
and less flexibility.

**The sanity check:** if the policy is initialised exactly from the reference, the margin
is zero and the loss is $$\log 2 \approx 0.693$$. A mismatch then points to masking,
sequence-log-prob aggregation, or reference/policy drift. Sum response-token log-probs
only; prompt and padding tokens are not part of the preference margin.

**What it trades away.** It is off-policy — it learns from preferences collected on a
distribution the policy drifts away from — and it is vulnerable to *likelihood
displacement*, where the margin grows because the rejected response's probability falls
rather than the chosen one's rising, sometimes driving both down
([Raz et al., 2024](https://arxiv.org/abs/2410.08847)).

<!-- EXERCISE p20 -->
**Exercise** — [`p20` · DPO loss](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p20_dpo_loss.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p20_dpo_loss.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p20_dpo_loss.py) · [`python run.py p20`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 min
<!-- /EXERCISE -->

---

<a id="b7-4"></a>
### B7.4 GAE

Primary source: [Schulman et al., 2015](https://arxiv.org/abs/1506.02438).

```python
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
```

$$\lambda = 0$$ recovers one-step TD; $$\lambda = 1$$ gives the full discounted
residual sum, equivalent to a Monte Carlo-style return minus the baseline when the
trajectory is terminal. At a time-limit truncation it still bootstraps from `last_value`.
Assert both limits as correctness checks.

**The loop runs backwards** because $$\hat A_t$$ depends on $$\hat A_{t+1}$$. Writing it
forwards uses an unavailable future accumulator. This compact function handles one
uninterrupted trajectory: pass `last_value=0` for a true terminal and the critic's
bootstrap value for a truncation; batched episodes need done masks.

<!-- EXERCISE p21 -->
**Exercise** — [`p21` · Generalised advantage estimation](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p21_gae.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p21_gae.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p21_gae.py) · [`python run.py p21`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 min
<!-- /EXERCISE -->

---

<a id="section-b8"></a>

## B8 · Data and tokenization

Two compact implementations. BPE exercises deterministic tokenisation and merge replay;
MoE routing exercises sparse dispatch, capacity, and weighted accumulation.

---

<a id="b8-1"></a>
### B8.1 Byte-pair encoding

Primary source: [Sennrich et al., 2015](https://arxiv.org/abs/1508.07909).

```python
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
```

This is a minimal byte-stream BPE. Production tokenizers usually pre-tokenize text,
prevent merges across selected boundaries, reserve special tokens, and define explicit
tie-breaking. Those policies sit around the same learn-and-replay core.

**A key bug source:** encoding applies merges in the order
they were **learned**, not by frequency in the string being encoded. Python dicts preserve
insertion order, so iterating `merges` is correct — but if you store them in a `set`, sort
them, or rebuild the dict, the token IDs no longer match those used to train the model.
Decoding may still recover the original bytes, so round-tripping alone will not catch
this; keep golden token-ID tests too.

**Why bytes rather than characters.** A byte-level vocabulary can represent any input, so
there is no out-of-vocabulary byte sequence. Non-Latin characters begin as more UTF-8
bytes; whether that becomes more tokens depends on how the finite merge budget is
allocated across scripts.

**The decode table is worth writing** even if not asked, because it is how you test
round-tripping:

```python
table = {i: bytes([i]) for i in range(256)}
for (a, b), new in merges.items():
    table[new] = table[a] + table[b]
assert b"".join(table[i] for i in ids).decode("utf-8") == text
```

**A useful follow-up:** why can character-count questions be brittle? The model receives
token IDs, not an explicit character sequence; one token may contain several letters.
The model can learn spelling information statistically, but exact character access is
not built into the representation.

<!-- EXERCISE p22 -->
**Exercise** — [`p22` · Byte-pair encoding](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p22_bpe.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p22_bpe.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p22_bpe.py) · [`python run.py p22`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 min · cold-start set
<!-- /EXERCISE -->

---

<a id="b8-2"></a>
### B8.2 Top-1 MoE routing with capacity

Primary source for this routing style:
[Switch Transformer](https://arxiv.org/abs/2101.03961).

```python
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
```

**Token dropping is the easy-to-miss part.** In this Switch-style policy, each expert has
a capacity; overflow tokens **skip the expert computation** and continue on the residual
stream. Other MoE systems reroute or avoid dropping. With dropping, the same input can
produce different outputs depending on what else shares its batch.

**The auxiliary loss is not there because the router lacks gradient** — it has one. The
gate probability multiplies the chosen expert's output, so the LM loss backpropagates into
the router; only the top-$$k$$ *selection* is non-differentiable. The problem is that this
gradient is self-reinforcing: experts that receive more tokens train faster, so the router
prefers them more, and routing can collapse.

**Why the loss is $$E\sum_e f_e p_e$$:** $$f$$ is non-differentiable (it counts
assignments) and $$p$$ is differentiable, so the gradient flows through $$p$$ weighted by
observed load. Under balanced assignments and mean probabilities it equals 1, while a
confidently collapsed router is penalised. It is still a surrogate with degeneracies:
exactly uniform logits also score 1 even though deterministic `argmax` tie-breaking sends
all tokens to one index.

<!-- EXERCISE p23 -->
**Exercise** — [`p23` · Top-1 MoE routing with capacity](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p23_moe_routing.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p23_moe_routing.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p23_moe_routing.py) · [`python run.py p23`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 min
<!-- /EXERCISE -->

---

<a id="section-c1"></a>

## C1 · Probability: five reusable patterns

Alisa Liu's public math notes provide the source map for distributions, expectations,
inequalities, and limit theorems. This section selects five reusable patterns and makes
each one self-contained: recognition cue, derivation, worked example, trap, and
closed-book self-test. The catalogue is a study scaffold, not a claim about interview
frequency.

---

<a id="c1-1"></a>
### C1.1 First-step analysis

**The tell:** a process that repeats, and you want the expected time until something
happens.

Condition on the first step and write the unknown expectations in terms of themselves.
Expected flips to see two heads in a row:

$$E_0=\underbrace{\tfrac 12 (1+E_0)}_\text{tails, no progress}+\underbrace{\tfrac 12(1+E_1)}_\text{heads},
\qquad E_1=\underbrace{\tfrac 12(1+E_0)}_\text{tails, start over}+\underbrace{\tfrac 12\cdot 1}_\text{heads, done}$$

Two equations, two unknowns, $$E_0 = 6$$.

**What makes it work is choosing the right state.** Here the state is "how much progress
toward HH do I have," which has three values (none, one H, done). Get the state wrong and
the equations do not close. That choice is the whole skill; the algebra is trivial.

**Same pattern, harder dressings:** gambler's ruin (state = current fortune), random walk
return times, expected steps for a Markov chain to hit a set.

<details><summary><strong>Self-test · derive before opening</strong></summary>
For a fair coin, find the expected number of flips until `HTH`. Use states for the
longest matched prefix. The equations are `E0 = 1 + (E0 + E1)/2`,
`E1 = 1 + (E1 + E2)/2`, and `E2 = 1 + E0/2`; solving gives `E0 = 10`.
</details>

---

<a id="c1-2"></a>
### C1.2 Indicators plus linearity of expectation

**The tell:** "how many X are expected to..." — a count.

Define $$X_i = \mathbb 1[\text{item } i \text{ has the property}]$$, sum them, and use
$$\mathbb E[\sum X_i] = \sum \mathbb E[X_i] = \sum P(X_i = 1)$$.

**Linearity holds whether or not the indicators are independent**, and that is the entire
reason this technique is so powerful. Dependence usually makes the joint distribution
hopeless while leaving each marginal easy.

Expected number of fixed points in a random permutation: by symmetry
$$P(X_i = 1) = 1/n$$, so the answer is $$n \cdot 1/n = 1$$, for every $$n$$. Computing
that from the distribution of the number of fixed points is genuinely hard; with
indicators it is one line.

**Coupon collector** is the same idea run backwards: decompose the total time into the
waiting times between new coupons, each geometric with mean $$n/(n-k)$$, giving
$$n H_n \approx n \ln n$$.

<details><summary><strong>Self-test · dependence is allowed</strong></summary>
In a fair random bit string of length `n`, what is the expected number of adjacent equal
pairs? There are `n - 1` indicators, each with probability `1/2`, so the expectation is
`(n - 1) / 2`. Adjacent indicators need not be independent.
</details>

---

<a id="c1-3"></a>
### C1.3 Max and min of $$n$$ variables — go through the CDF

**The tell:** anything about the largest or smallest of several draws.

For iid variables, start with the CDF. "The max is at most $$x$$" means "all of them are
at most $$x$$", and independence turns the intersection into a product:

$$F_M(x) = P(\max_i X_i \le x) = [F_X(x)]^n$$

For the minimum, complement it: $$P(\min > x) = [1 - F_X(x)]^n$$. Differentiate at the
end if you actually need a density.

**Worth having memorised:** for $$n$$ iid uniforms on $$[0,1]$$,
$$\mathbb E[\max] = n/(n+1)$$ and $$\mathbb E[\min] = 1/(n+1)$$ — and the symmetry between
them is a good sanity check on any answer of this type.

<details><summary><strong>Self-test · CDF first</strong></summary>
For three iid `Uniform(0, 1)` draws, the maximum has CDF `x^3` on `[0, 1]`.
Integrating its density `3x^2` times `x`, or integrating `1 - x^3`, gives expected
maximum `3/4`.
</details>

---

<a id="c1-4"></a>
### C1.4 Symmetry as a proof technique

**The tell:** the answer feels like it should not depend on something.

Two examples that come up. In a random permutation, the probability that item $$i$$ lands
in position $$j$$ is $$1/n$$ for every pair — that is what makes C1.2 work. And in the
secretary problem, the probability that the best candidate appears in any given position
is uniform, which is why the analysis only involves *where* the maximum falls.

**Monty Hall is the anti-example.** The host's choice is *not* symmetric — he never opens
the prize door — and that asymmetry is exactly where the 2/3 comes from. When you invoke
symmetry, say which transformation the problem is invariant under; if you cannot name
it, you are guessing.

<details><summary><strong>Self-test · name the transformation</strong></summary>
In a uniformly random permutation, what is the probability that item `a` appears before
item `b`? It is `1/2`: swapping the positions of `a` and `b` pairs every ordering of one
kind with exactly one ordering of the other kind.
</details>

---

<a id="c1-5"></a>
### C1.5 Which inequality to reach for

| You know | Use | Gives |
|---|---|---|
| Mean only, $$X \ge 0$$ | Markov | $$P(X \ge a) \le \mathbb E[X]/a$$ |
| Mean and variance | Chebyshev | $$P(\lvert X-\mu\rvert \ge k\sigma) \le 1/k^2$$ |
| Bounded, independent sum | Hoeffding | Exponential tail |
| Independent terms with a controlled MGF | Chernoff | Exponential tail |
| Convex function of an expectation | Jensen | $$f(\mathbb E[X]) \le \mathbb E[f(X)]$$ |

**Markov makes the fewest assumptions** in this list: non-negativity and a finite mean.
Chebyshev is Markov applied to $$(X-\mu)^2$$.

**Jensen is the one that shows up in ML rather than in puzzles.** It is why the ELBO is a
lower bound, why $$\log \mathbb E[\cdot] \ge \mathbb E[\log \cdot]$$ matters in importance
sampling, and why the KL divergence is non-negative.

<details><summary><strong>Self-test · choose the weakest sufficient tool</strong></summary>
If `X` is non-negative with mean 4, Markov gives `P(X >= 20) <= 1/5`.
If you additionally know mean `mu` and standard deviation 3, Chebyshev gives
`P(|X - mu| >= 6) <= 1/4`. Neither bound assumes a particular distribution.
</details>

---

<a id="section-c2"></a>

## C2 · Simulate, then verify

This hybrid prompt asks you to simulate a physical setup, derive its distribution
analytically, and verify that the samples agree. It deliberately joins coding and
probability, so both halves must be checked.

---

<a id="c2-1"></a>
### C2.1 The spinning light source

> A lamp sits at distance 1 from an infinitely long wall and points in a uniformly random
> direction. Simulate where it hits the wall. What distribution is that? Verify it.

**The derivation.** With $$\theta \sim \text{Uniform}(-\pi/2, \pi/2)$$, the hit position
is $$x = \tan\theta$$. Transform the density:

$$f_X(x) = f_\Theta(\theta)\left|\frac{d\theta}{dx}\right|
= \frac{1}{\pi}\cdot\frac{1}{1+x^2}$$

That is the standard **Cauchy** distribution, and it is chosen for the question because
it is pathological in a way that is easy to demonstrate and hard to fake.

```python
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
```

**The point of the question is that the mean does not exist.** $$\int |x| f(x)\,dx$$
diverges, so the usual law of large numbers does not apply. More sharply, the mean of
$$n$$ independent standard Cauchy samples is itself standard Cauchy for every $$n$$:
it does not concentrate as $$n$$ grows. Illustrate this with running means at $$10^4$$,
$$10^5$$ and $$4\times10^5$$ samples, but do not mistake one path for a proof. For a
finite-variance distribution the standard error would fall like $$1/\sqrt n$$; Cauchy
has neither a finite mean nor finite variance.

**The median is well behaved**, and estimating the location parameter with it instead is
the correct practical response.

**One trap in the verification, and it is a good one.** Comparing a histogram to the
analytic PDF on a window like $$[-5, 5]$$ *fails* if you are careless, and the reason is
not a coding error. NumPy's `density=True` normalises over the bins you plotted, but the
true Cauchy puts only $$\tfrac{2}{\pi}\arctan 5 = 0.874$$ of its mass in that window. The
tails are heavy enough that ignoring the truncation inflates your histogram by 14% and
makes a correct simulation look wrong. Compare against the *conditional* density
$$f(x)/P(|X|\leq L)$$, where
$$P(|X|\leq L)=2\arctan(L)/\pi$$, and choose a tolerance from the sample count rather than
hard-coding one independent of $$n$$.

> **Why the exercise is useful.** The implementation is short, but verification requires
> specifying the distribution actually represented by the plotted bins. The truncation
> correction is the same kind of conditioning issue that can invalidate a real
> experiment.

<!-- EXERCISE p27 -->
**Exercise** — [`p27` · Spinning light source → Cauchy](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p27_cauchy_simulation.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p27_cauchy_simulation.md) · [tests](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p27_cauchy_simulation.py) · [`python run.py p27`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 min
<!-- /EXERCISE -->

---

<a id="c2-2"></a>
### C2.2 The general recipe

The setup generalises, and having a recipe means you are not improvising under time
pressure.

**Derive first, simulate second.** If you simulate first you have nothing to check
against, and "the histogram looks about right" is not an answer. The transformation rule
for a monotone $$g$$ is $$f_Y(y) = f_X(g^{-1}(y))\,|dg^{-1}/dy|$$; for non-monotone
$$g$$, sum over the branches.

**Verify at three levels, cheapest first.** Moments where they exist (mean, variance) are
one line. A Kolmogorov–Smirnov statistic gives a bin-free global CDF discrepancy.
A histogram is visually useful but sensitive to bins, range, and the truncation issue
above.

**Say what you would do about variance.** Sample size sets your resolution: with $$n$$
samples a bin holding probability $$p$$ has relative error about $$1/\sqrt{np}$$, so
tail bins are noisy and you should not tighten your tolerance uniformly across them.

**Name the standard transformations**, since one of them is usually the intended answer:
inverse-CDF sampling for anything with a closed-form quantile, Box–Muller for Gaussians,
$$-\log U/\lambda$$ for exponentials, and the ratio of two standard normals for a Cauchy —
which is a nice cross-check on this very question, since it should produce the same
distribution as the tangent construction.

<details><summary><strong>Self-test · transform and verify</strong></summary>
Let `U` be uniform on `(0, 1)` and define `Y = -log(1 - U) / lambda`.
Then `P(Y <= y) = P(U <= 1 - exp(-lambda*y)) = 1 - exp(-lambda*y)`, the exponential
CDF. Verify a simulation with quantiles or a KS statistic, not only a histogram.
</details>

---

<a id="section-c3"></a>

## C3 · Linear algebra

This section fills the linear-algebra side of the source notes: matrices as maps, rank and
SVD, positive semi-definiteness, conditioning, and matrix calculus. Each idea is tied to
an operation already used in the coding half.

---

<a id="c3-1"></a>
### C3.1 The four facts everything else follows from

**A matrix is a linear map, and its shape tells you between which spaces.** $$W$$ of shape
$$(m, n)$$ maps $$\mathbb R^n \to \mathbb R^m$$. Most shape bugs dissolve if you read
matrices this way rather than as grids of numbers.

**Rank is the dimension of the output space actually reached.** A low-rank matrix
squashes its input into a subspace. For LoRA, $$\Delta W=BA$$ with inner dimension $$r$$
satisfies $$\operatorname{rank}(\Delta W)\le r$$: each input passes through an
$$r$$-dimensional bottleneck before being expanded back to the output space.

**Eigenvectors are directions a map only scales**, $$Av = \lambda v$$. The definition
requires a square matrix; a general real matrix may need complex eigenvectors and may not
have a full eigenbasis. A real symmetric matrix, however, has an orthonormal basis of real
eigenvectors — which is why Hessians and covariance matrices are especially tractable.

**The SVD applies to every matrix**, square or not: $$A = U\Sigma V^\top$$, an orthogonal
change of coordinates, scaling along axes, then another orthogonal change. Singular
values are the scale factors, and truncating the small ones is the best rank-$$k$$
approximation in the Frobenius norm.
That single fact underlies PCA, low-rank compression, and the way people reason about
whether a weight update is "really" low rank.

<details><summary><strong>Self-test · connect shape, rank, and SVD</strong></summary>
A `6 x 4` matrix has rank 3. Its map has a one-dimensional null space by rank-nullity,
exactly three non-zero singular values, and a three-dimensional column space. Its best
rank-2 Frobenius approximation keeps the two largest singular triplets.
</details>

---

<a id="c3-2"></a>
### C3.2 Positive semi-definiteness, and why it keeps appearing

A real symmetric $$M$$ is PSD when $$x^\top M x \ge 0$$ for all $$x$$, equivalently when
all its eigenvalues are non-negative.

**Three places it decides something in ML.** A covariance matrix is PSD by construction,
because $$x^\top \Sigma x$$ is the variance of a projection and variances are
non-negative. For a twice-differentiable objective on a convex domain, a Hessian that is
PSD everywhere implies convexity, so any stationary point is a global minimum; neural
network objectives generally do not satisfy that condition. And a kernel matrix must be
PSD for the kernel trick to correspond to an inner product in some space.

**How the Hessian classifies a critical point.** A positive-definite Hessian gives a
strict local minimum; a negative-definite one gives a strict local maximum; an
indefinite Hessian gives a saddle because there are both ascent and descent directions.
Hessian eigenvalue signs are not independent coin flips, so the tempting $$2^{-d}$$
argument for the prevalence of minima or saddles is not a valid general proof.

<details><summary><strong>Self-test · prove a workhorse matrix is PSD</strong></summary>
For any rectangular `A`, `A.T @ A` is PSD because
`x.T @ A.T @ A @ x = ||A @ x||_2^2 >= 0`. It is positive definite exactly when `A` has
trivial null space, equivalently full column rank.
</details>

---

<a id="c3-3"></a>
### C3.3 Norms, conditioning, and the things that blow up

**Which norm, and why it matters.** Global-norm gradient clipping commonly measures the
$$\ell_2$$ norm; the Frobenius norm is the $$\ell_2$$ norm of a flattened matrix; the spectral
norm is the largest singular value, i.e. the most a matrix can stretch any vector. The
spectral norm bounds how much a perturbation can grow through a linear layer, making it a
useful component of stability and Lipschitz analyses.

**Condition number** $$\kappa = \sigma_\max/\sigma_\min$$ bounds how much relative
perturbations can be amplified when solving a full-rank linear system; it is infinite
when $$\sigma_\min=0$$. Feature scaling and
preconditioning can reduce anisotropy; Adam's coordinate-wise scaling can be viewed as
a diagonal, adaptive preconditioner, though it is not the inverse Hessian.

**Two numerical rules that come from this.** Never form $$X^\top X$$ to solve least
squares if you can avoid it: it squares the condition number, so you lose twice the
digits. And never invert a matrix to solve $$Ax = b$$ — use a factorisation
(`np.linalg.solve`, not `inv(A) @ b`), which is both faster and more numerically stable.
The problem's condition number is a property of $$A$$ and does not improve merely
because a better algorithm is used.

<details><summary><strong>Self-test · quantify the damage</strong></summary>
If a matrix has singular values 10 and 0.1, its 2-norm condition number is 100.
Forming the normal-equation matrix squares it to 10,000, which is why QR or SVD is safer
for least squares.
</details>

---

<a id="c3-4"></a>
### C3.4 The matrix calculus you actually need

Four identities regenerate almost everything, and you can check each by shape.

$$\frac{\partial}{\partial x}(a^\top x) = a,\qquad
\frac{\partial}{\partial x}(x^\top A x) = (A + A^\top)x,\qquad
\frac{\partial}{\partial X}\operatorname{tr}(AX) = A^\top,\qquad
\frac{\partial}{\partial X}\|X\|_F^2 = 2X$$

**The quadratic form gives $$2Ax$$ when $$A$$ is symmetric**, which is the case you meet,
but stating the general form shows you did not just memorise the special case.

**The rule that beats memorisation** is the one from B4.1: the gradient with respect to a
tensor has that tensor's shape, and there is usually exactly one contraction of the
available operands that produces it. If you can derive $$\partial L/\partial W = X^\top
\partial L/\partial Z$$ by shape alone, you do not need a table.

> **Where this shows up unannounced:** deriving the linear-layer backward (B4.1), the
> softmax Jacobian (B4.3), the normal equations, and any question about why a particular
> update rule has the form it does.

<details><summary><strong>Self-test · derive by differentials and shape</strong></summary>
For `y = X @ w` and `L = 0.5 * ||y - t||^2`, let `r = X @ w - t`.
Then `dL = r.T @ X @ dw`, so the gradient is `X.T @ r`, which has the same shape as `w`.
</details>

---

<a id="section-c4"></a>

## C4 · Counting

The first step is to state *what you are counting*. Typical errors count the same object
twice or count orderings when the question did not ask for them.

---

<a id="c4-1"></a>
### C4.1 The one decision that determines the formula

Before writing anything, answer two questions: **does order matter**, and **can items
repeat**. That is a 2×2 table and it fixes the formula.

| | Order matters | Order does not |
|---|---|---|
| **No repeats** | $$\dfrac{n!}{(n-k)!}$$ | $$\dbinom{n}{k}$$ |
| **Repeats allowed** | $$n^k$$ | $$\dbinom{n+k-1}{k}$$ |

The bottom-right cell follows from **stars and bars**:
distributing $$k$$ identical items into $$n$$ labelled bins is the same as arranging
$$k$$ stars and $$n-1$$ bars in a row, so you choose which $$k$$ of the $$n+k-1$$
positions are stars.

**Say the mapping, not the formula.** "Each arrangement of stars and bars corresponds to
exactly one distribution" is a proof; quoting $$\binom{n+k-1}{k}$$ is a memory claim.

<details><summary><strong>Self-test · classify before calculating</strong></summary>
Eight-character lowercase strings allow repetition and care about order, so there are
`26^8`. Choosing eight distinct letters without order gives `C(26, 8)`. State which
object is being counted before choosing either expression.
</details>

---

<a id="c4-2"></a>
### C4.2 Overcount, then divide

One reusable technique is to count something easy that overcounts by a known factor,
then divide.

**Arrangements of a multiset.** The letters of MISSISSIPPI: pretend all 11 letters are
distinct ($$11!$$), then divide by the orderings within each repeated group,
$$4!\,4!\,2!$$ for the S's, I's and P's.

**Circular arrangements.** If rotations are equivalent but reflections are distinct,
$$n$$ people around a round table gives $$(n-1)!$$ seatings, because each seating was
counted $$n$$ times, once per rotation.

**Choosing then ordering.** $$\binom{n}{k}k! = n!/(n-k)!$$ recovers the permutation
formula, which is a good check that you have the right mental model rather than two
memorised formulas.

> **A debugging heuristic:** when labelled and unlabelled counts differ by a symmetry
> group, the overcount factor is often a factorial. Three interchangeable objects, for
> example, produce a factor of `3!`.

<details><summary><strong>Self-test · identify every symmetry factor</strong></summary>
Partition 12 distinct people into three **unlabelled** teams of four. Order all people,
then divide by `4!` within each team and by `3!` for the order of the teams:
`12! / ((4!)^3 3!)`.
</details>

---

<a id="c4-3"></a>
### C4.3 Inclusion–exclusion

**The tell:** "at least one", or a union of overlapping conditions.

$$|A \cup B \cup C| = \sum|A_i| - \sum|A_i \cap A_j| + |A_1 \cap A_2 \cap A_3|$$

The alternating signs are there because adding the singles double-counts the pairs,
subtracting the pairs then removes the triple one time too many, and so on.

**The canonical application is derangements** — permutations with no fixed point:

$$D_n = n!\sum_{k=0}^{n}\frac{(-1)^k}{k!} \approx \frac{n!}{e}$$

so the probability that a random permutation has no fixed point tends to $$1/e \approx
0.368$$. The finite-$$n$$ probability is not constant, but it converges rapidly.

**Check the complement first.** Many "at least one" problems become "1 − none", where
"none" is a single product. Use inclusion–exclusion when overlapping events remain after
that reformulation.

<details><summary><strong>Self-test · use the complement</strong></summary>
The probability that a random permutation has at least one fixed point is
`1 - D_n/n!`, which tends to `1 - 1/e`. This is the complement of the derangement event,
not a sum of independent fixed-point probabilities.
</details>

---

<a id="c4-4"></a>
### C4.4 Where counting meets ML

Two places these questions get dressed up in ML clothing.

**The birthday problem, applied to hash collisions and duplicate detection.** With
$$n\leq d$$ independent uniform items in $$d$$ buckets, the exact probability of no
collision is $$\prod_{i=0}^{n-1}(1 - i/d)$$. The approximation is
$$\exp[-n(n-1)/(2d)]$$, so collisions become likely once $$n \sim \sqrt d$$ in the
order-of-magnitude sense. In the birthday regime $$n=O(\sqrt d)$$ the approximation is
asymptotically accurate; more generally, dropping the next log-expansion term requires
$$n^3/d^2 \ll 1$$. The exact product's condition $$n\leq d$$ is separate—when $$n>d$$,
the pigeonhole principle makes the no-collision probability zero. This is why a 64-bit
hash is not enough to deduplicate a trillion documents; the approximation is also the
right first check for any hash-based deduplication design.

**Counting parameters.** Deriving a transformer's parameter count is a counting problem
with a shape argument attached (Part I, A10). The discipline is the same: say what you
are counting — per layer, per head, embedding versus
unembedding — before you multiply anything.

<details><summary><strong>Self-test · estimate before multiplying exactly</strong></summary>
For a uniform 128-bit hash, the collision scale is the square root of the bucket count,
about `2^64` items. At one trillion items the pair-collision approximation is roughly
`10^24 / 2^129`, about `1.5e-15`.
</details>

---

<a id="section-c5"></a>

## C5 · Markov chains and random walks

The natural home of first-step analysis (C1.1), and the setting for several of the
classic puzzles. MCMC and diffusion use Markov processes directly; an MDP is their
controlled-state extension, and a fixed policy induces a Markov chain.

---

<a id="c5-1"></a>
### C5.1 What the Markov property actually buys you

$$P(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = P(X_{t+1} \mid X_t)$$

The future is conditionally independent of the past given the present. The practical
consequence for a finite, time-homogeneous chain is that **one transition matrix**
$$P$$ specifies the dynamics, and $$n$$-step behaviour is $$P^n$$ — which turns dynamics
questions into linear algebra.

**The modelling skill is choosing the state**, exactly as in first-step analysis. Almost
any process can be made Markov by enlarging the state to include whatever history
matters. For the target `HTH`, the last flip alone is insufficient; the length of the
longest target prefix matched so far (0, 1, 2, or absorbed) is a sufficient state.

**The stationary distribution** $$\pi$$ satisfies $$\pi P = \pi$$ — a left eigenvector
with eigenvalue 1. For a finite irreducible chain it is unique; aperiodicity additionally
ensures $$P^n$$ converges to it. For finite reversible chains, the nontrivial eigenvalues
yield spectral-gap bounds on convergence. The inverse gap is a mixing-time scale, not
literally the mixing time; constants and the smallest stationary mass also matter.

<details><summary><strong>Self-test · solve the left-eigenvector equation</strong></summary>
For transition matrix `[[0.9, 0.1], [0.2, 0.8]]`, solve `pi @ P = pi` together with
`pi.sum() = 1`. Detailed flow balance gives `0.1*pi0 = 0.2*pi1`, hence
`pi = [2/3, 1/3]`.
</details>

---

<a id="c5-2"></a>
### C5.2 Gambler's ruin

You have $$i$$ dollars, bet one at a time on a fair coin, and stop at $$0$$ or $$N$$.

**The probability of reaching $$N$$ first is $$i/N$$**, and the fastest derivation is
that your fortune is a martingale: its expectation never changes, so
$$i = 0\cdot(1-p) + N\cdot p$$, giving $$p = i/N$$ immediately. First-step analysis gets
the same answer via $$p_i = \tfrac12 p_{i-1} + \tfrac12 p_{i+1}$$, whose solution is
linear in $$i$$ with the given boundary conditions.

**The expected duration is $$i(N-i)$$**, which is worth knowing because it is
surprisingly long — starting at half of $$N = 100$$, the expected number of bets before
absorption is 2,500.

**The biased case is where the intuition breaks.** With win probability $$q \ne 1/2$$ the
answer becomes a ratio of geometric terms, and the ruin probability approaches 1
exponentially fast even for $$q$$ slightly below $$1/2$$. The takeaway to state: a small
persistent edge against you is not a small disadvantage over many rounds.

<details><summary><strong>Self-test · use both boundary-value solutions</strong></summary>
For the fair walk with `i = 3` and `N = 10`, the chance of reaching 10 is `3/10`, while
the expected absorption time is `3 * (10 - 3) = 21` bets.
</details>

---

<a id="c5-3"></a>
### C5.3 Random walks, and the dimension surprise

**Symmetric walk on $$\mathbb Z$$.** After $$n$$ steps, $$\mathbb E[X_n] = 0$$ and
$$\operatorname{Var}(X_n) = n$$, so the typical displacement grows like $$\sqrt n$$.
The same square-root scaling appears in standard errors and Brownian diffusion because
independent increments add variance rather than standard deviation.

**Pólya's theorem:** the symmetric random walk
is recurrent in one and two dimensions — it returns to the origin with probability 1 —
and **transient in three or more**. In three dimensions the eventual return probability
is about 0.34; it is smaller in higher dimensions.

**Expected return time.** For an irreducible positive-recurrent chain with stationary
distribution $$\pi$$, Kac's formula gives mean return time $$1/\pi_i$$. On $$\mathbb Z$$
the walk returns with probability 1 but has no stationary probability distribution and
its expected return time is infinite — recurrent, yet null recurrent.

<details><summary><strong>Self-test · separate mean from scale</strong></summary>
After 100 steps of a one-dimensional symmetric walk, the mean position is 0 and the
standard deviation is 10. The probability of being exactly at the origin is
`C(100, 50) / 2^100`, about `0.0796`; zero mean does not mean likely to be at zero.
</details>

---

<a id="c5-4"></a>
### C5.4 Portable worked examples

Each example isolates one modelling move that transfers to less familiar problems.

**Coupon collector.** $$n H_n \approx n\ln n$$ draws to see all $$n$$ coupons, from
decomposing into geometric waiting times (C1.2). The variance is $$O(n^2)$$, so the tail
still fluctuates on an order-$$n$$ scale — relevant whenever you sample to cover a space.

**Birthday problem.** Collisions become likely at $$n \sim \sqrt d$$, not $$n \sim d$$
(C4.4).

**Monty Hall.** Switching wins 2/3. The reason is that the host's action is not
independent of the truth — he never opens the prize door — so his choice transfers
information. Frame it as conditioning on the host's *rule*, not on the door.

**Secretary problem.** Asymptotically, reject roughly the first $$n/e$$ candidates, then
take the next one better than everything seen. The success probability tends to
$$1/e \approx 0.37$$; the same $$1/e$$ appears in derangements (C4.3) for unrelated
reasons.

**Reservoir sampling.** For a one-item reservoir, replace the current item with stream
item $$k$$ with probability $$1/k$$. A one-line induction shows every one of the
$$n$$ stream items ends with probability $$1/n$$. This is the right primitive for a
uniform sample from a stream whose length is not known in advance.

**Two-heads-in-a-row.** Expected 6 flips (C1.1). The follow-up — expected flips for HT —
is 4, and the fact that HH and HT differ at all is the interesting part: after a failed
HT attempt caused by another H, you retain partial progress; after an HH attempt fails
with T, you do not. The overlap structure of the target pattern changes the answer.

<details><summary><strong>Self-test · prove reservoir uniformity</strong></summary>
Item `j` enters a one-item reservoir with probability `1/j`. It then survives updates
`j+1` through `n` with probability
`(j/(j+1)) * ((j+1)/(j+2)) * ... * ((n-1)/n) = j/n`.
The product with `1/j` is `1/n`, independent of `j`.
</details>

---

<a id="section-c6"></a>

## C6 · Statistics and estimation

This section connects the probability above to losses, estimators, and evaluation
decisions used in ML.

---

<a id="c6-1"></a>
### C6.1 Maximum likelihood, and what your loss function really is

MLE picks the parameters that make the observed data most probable:

$$\hat\theta = \arg\max_\theta \prod_i p(x_i \mid \theta)
= \arg\max_\theta \sum_i \log p(x_i \mid \theta)$$

You take the log because products of many small numbers underflow, and because sums
differentiate cleanly.

**The three derivations worth being able to produce on demand**, because each one shows
that a familiar loss *is* an MLE under a particular noise assumption:

**Gaussian with known variance gives least squares.** $$\log p \propto -(x-\mu)^2$$, so
maximising likelihood is minimising squared error. Under a likelihood interpretation,
using MSE corresponds to a fixed-variance Gaussian observation model.

**Bernoulli gives cross entropy.** $$\log p = y\log\hat y + (1-y)\log(1-\hat y)$$, which
is the negative of binary cross entropy. Minimising negative log-likelihood therefore
gives BCE; a categorical likelihood gives multi-class cross entropy.

**Laplace gives absolute error.** $$\log p \propto -|x - \mu|$$, so L1 loss corresponds
to noise with heavier tails than a Gaussian — the probabilistic reason L1 is less
sensitive to large residuals.

> **The follow-up that goes with all three:** MAP estimation adds a prior, and a Gaussian
> prior on the weights gives L2 regularisation while a Laplace prior gives L1. Classical
> SGD weight decay is equivalent to L2 regularisation; decoupled weight decay in adaptive
> optimisers is not generally identical. L1 encourages exact zeros because its penalty is
> non-differentiable at zero.

<details><summary><strong>Self-test · likelihood versus prior</strong></summary>
For 8 successes and 2 failures, the Bernoulli MLE is `p = 0.8`. With a `Beta(2, 2)`
prior, the MAP mode is `(8 + 2 - 1) / (10 + 2 + 2 - 2) = 0.75`.
</details>

---

<a id="c6-2"></a>
### C6.2 Bias, variance, and when a biased estimator is useful

For an estimator $$\hat\theta$$:

$$\mathbb E[(\hat\theta - \theta)^2] = \underbrace{(\mathbb E[\hat\theta]-\theta)^2}_{\text{bias}^2}
+ \underbrace{\operatorname{Var}(\hat\theta)}_{\text{variance}}$$

**The $$n-1$$ question.** The sample variance with $$1/n$$
is biased low, because you measured deviations from the *sample* mean, which is itself
fitted to the data and therefore sits closer to the points than the true mean does.
Dividing by $$n-1$$ corrects it — you spent one degree of freedom estimating the mean.

**Then the part that makes it an ML answer:** unbiasedness is not the goal, mean squared
error is, and a biased estimator with lower variance often wins. That is why
many normalisation layers use the equivalent of `torch.var(unbiased=False)` in their
forward pass — they normalise the current activation set rather than estimate a
population variance. It is also the subtlety in BatchNorm, which normalises with the biased variance and
accumulates the unbiased one (B1.2).

**What the classical cartoon leaves out.** The decomposition itself always holds for
squared error; what can fail is the simple U-shaped story that model size monotonically
trades bias for variance. Double descent shows that test risk can fall again after the
interpolation threshold, depending on optimisation, data, and implicit regularisation
(Part I, A1.8).

<details><summary><strong>Self-test · distinguish target from estimator</strong></summary>
For observations `1, 2, 3`, the squared deviations from the sample mean are `1, 0, 1`.
Dividing by 3 gives the empirical variance `2/3`; dividing by `n - 1 = 2` gives the
unbiased estimator 1 for an iid population variance.
</details>

---

<a id="c6-3"></a>
### C6.3 Concentration: how many samples do you need

This is the practical form of the inequalities in C1.5, and it is what you actually use
when someone asks how many eval examples are enough.

**For a proportion**, the standard error is $$\sqrt{p(1-p)/n}$$, worst case
$$1/(2\sqrt n)$$. Under an iid normal approximation, a 95% confidence interval is roughly
$$\pm 1/\sqrt n$$ at the worst case: 100 examples
gives $$\pm 10\%$$, 1,000 gives $$\pm 3\%$$, 10,000 gives $$\pm 1\%$$. **Memorise those
three** as worst-case scales. A 2-point difference on a 500-example benchmark may be
within sampling noise; decide with a paired interval, not the point estimates alone.

**Hoeffding** gives the same shape without a normal approximation: for independent
variables in `[0, 1]`,
$$P(|\bar X - \mu| \ge t) \le 2\exp(-2nt^2)$$, so $$n \sim \log(1/\delta)/t^2$$. The
$$1/t^2$$ is the expensive part — one more digit of precision costs 100× the samples.

**The paired-comparison trick worth knowing.** When comparing two models on the same
examples, compare per-example differences rather than the two means. The variance of the
difference can be much smaller when example difficulty is positively correlated across
models. Report the confidence interval of those paired differences rather than assuming
that a point estimate is real.

<details><summary><strong>Self-test · invert a concentration bound</strong></summary>
Hoeffding with error `t = 0.03` and failure probability `delta = 0.05` requires
`n >= log(2/delta) / (2t^2)`, so `n >= 2050` bounded independent examples is sufficient.
Here the examples are independent and `[0, 1]`-valued. The bound can be conservative;
the algebra is the point.
</details>

---

<a id="c6-4"></a>
### C6.4 Hypothesis testing, briefly, and its ML failure mode

The practical question is whether a benchmark improvement is real, not whether you can
recite the mechanics of a t-test.

**The vocabulary you need:** a p-value is $$P(\text{data this extreme} \mid H_0)$$ —
*not* the probability that the null is true, and getting that backwards is the classic
error. A corresponding confidence interval also reports an effect-size range rather than
only thresholding evidence against a null.

**A major ML failure mode is multiple comparisons.** Under twenty
independent true nulls tested at $$p < 0.05$$, the expected number of false positives is
one and the chance of at least one is about 64%. Selecting among hyperparameters or
variants using the same benchmark creates the same problem; responses include a held-out
set touched once, a correction, or pre-registering what you will measure.

> **The connection to make** is that this is the same failure as
> overfitting to a validation set, seen through a statistical lens. Selecting a model on
> a benchmark makes that benchmark an optimistic estimate of its performance, and the
> final estimate should come from data that did not drive the selection.

<details><summary><strong>Self-test · family-wise error</strong></summary>
For 20 independent null tests at level 0.05, the probability of at least one false
positive is `1 - 0.95^20`, about `0.642`. Bonferroni tests each at `0.05/20 = 0.0025`
to keep family-wise error at most 0.05 without requiring independence.
</details>

---

<a id="section-refs"></a>

## References

Technical claims in this article use the primary papers below. Interview-shape claims
are labelled *reported* or *anecdotal* and come from the named first-hand preparation
accounts; no company name denotes an official question bank.

1. Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
2. Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
3. DeepSeek-AI. (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
4. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
5. Jaiswal, M. (2024–2025). *LLM (ML) Job Interviews — Resources.* [mimansajaiswal.github.io](https://mimansajaiswal.github.io/posts/llm-ml-job-interviews-resources/)
6. Karpathy, A. *micrograd* and *nanoGPT.* [micrograd](https://github.com/karpathy/micrograd) · [nanoGPT](https://github.com/karpathy/nanoGPT)
7. Leviathan, Y., Kalman, M., & Matias, Y. (2022). *Fast Inference from Transformers via Speculative Decoding.* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
8. Liu, A. (2026). *Notes on the Industry Job Search.* [alisawuffles.github.io](https://alisawuffles.github.io/blog/job-search/)
9. Meng, Y. (2026). *MLE Interview 2.0: Research Engineering and Scary Rounds.* [yuan-meng.com](https://www.yuan-meng.com/posts/mle_interviews_2.0/)
10. Milakov, M., & Gimelshein, N. (2018). *Online Normalizer Calculation for Softmax.* [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
11. Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
12. Sapora, S. (2026). *ML Job Interviews: The Ultimate Guide.* [silviasapora.github.io](https://silviasapora.github.io/blog/ml-interviews.html)
13. Schulman, J., et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
14. Sennrich, R., Haddow, B., & Birch, A. (2015). *Neural Machine Translation of Rare Words with Subword Units.* [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
15. Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
16. Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All You Need.* [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
17. Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
18. Vaswani, A., et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
19. Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
20. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
21. Shazeer, N. (2020). *GLU Variants Improve Transformer.* [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
22. Raz, G., et al. (2024). *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization.* [arXiv:2410.08847](https://arxiv.org/abs/2410.08847)
23. Chen, C., et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling.* [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
24. Fedus, W., Zoph, B., & Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
