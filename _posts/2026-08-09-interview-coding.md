---
layout: post
title: "Interview Bank II · Coding + Math: write it, do not read it"
date: 2026-08-09 12:00:00
author: Jiaxin Zhang
description: "Complete, tested implementations of the components a frontier-lab coding round asks for — attention, KV cache, RoPE, sampling, GRPO, BPE — plus the probability and linear algebra that comes with them, and a timed practice harness."
tags: interviews llm coding math pytorch qbank
categories: research-notes
giscus_comments: true
related_posts: false
ai_assisted: true
---

<div class="lang-switch"><strong>English</strong> · <a href="/blog/2026/interview-coding-zh/">中文</a></div>

<div class="lang-switch"><a href="/blog/2026/interview-knowledge/">I · Knowledge</a> · <strong>II · Coding + Math</strong> · <span class="text-muted">III · Discussion + BQ</span></div>

Part I asked whether you can *retrieve* something. This part asks whether you can
*produce* it, from an empty file, with a clock running.

Those are different skills, and the gap between them is the entire reason this page
comes with a repository attached. Reading an attention implementation until it feels
obvious does almost nothing for your ability to write one in twenty minutes. So the
code below is the explanation layer, and [`interview-practice/`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/tree/master/interview-practice) is where you
actually train — 27 problems with stubs and tests, 10 debug drills, and a timed runner.

> **How this is organised.** Each section states the concept briefly, gives a complete
> annotated implementation, lists the mistakes that show up every time, and names what
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
- **[B1 · NumPy and PyTorch fundamentals](#section-b1)** — 3 exercises
  - [B1.1 Vectorisation: the one trick worth memorising](#b1-1)
  - [B1.2 BatchNorm, and why it has two modes](#b1-2)
  - [B1.3 The tensor semantics that cause silent bugs](#b1-3)
- **[B2 · Transformer components](#section-b2)** — 7 exercises
  - [B2.1 Causal multi-head attention](#b2-1)
  - [B2.2 KV cache and incremental decode](#b2-2)
  - [B2.3 Grouped-query attention](#b2-3)
  - [B2.4 Rotary position embeddings](#b2-4)
  - [B2.5 RMSNorm](#b2-5)
  - [B2.6 SwiGLU](#b2-6)
  - [B2.7 The block, assembled](#b2-7)
- **[B3 · The training loop](#section-b3)** — 4 exercises
  - [B3.1 Cross entropy, and why it takes logits](#b3-1)
  - [B3.2 Loss masking and packing](#b3-2)
  - [B3.3 Overfit ten examples before anything else](#b3-3)
  - [B3.4 Filtering bad annotations](#b3-4)
- **[B4 · Backward passes by hand](#section-b4)** — 2 exercises
  - [B4.1 The rule that generates every backward](#b4-1)
  - [B4.2 A 40-line autograd](#b4-2)
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
- **[C1 · Probability: four patterns that cover most of it](#section-c1)**
  - [C1.1 First-step analysis](#c1-1)
  - [C1.2 Indicators plus linearity of expectation](#c1-2)
  - [C1.3 Max and min of $$n$$ variables — go through the CDF](#c1-3)
  - [C1.4 Symmetry as a proof technique](#c1-4)
  - [C1.5 Which inequality to reach for](#c1-5)
- **[C2 · Simulate, then verify](#section-c2)** — 1 exercises
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
  - [C5.4 The puzzles worth having pre-solved](#c5-4)
- **[C6 · Statistics and estimation](#section-c6)**
  - [C6.1 Maximum likelihood, and what your loss function really is](#c6-1)
  - [C6.2 Bias, variance, and why the biased estimator is the default](#c6-2)
  - [C6.3 Concentration: how many samples do you need](#c6-3)
  - [C6.4 Hypothesis testing, briefly, and its ML failure mode](#c6-4)

---
<a id="section-b0"></a>

## B0 · How to practise this

Most coding-prep material fails in the same way: it shows you a correct implementation,
you nod, and you conclude that you know it. You do not. You have confirmed you can
*read* it. Under a clock, in an editor with no autocomplete, with someone watching, the
thing that fails is not your understanding — it is your ability to produce twenty
correct lines without stopping.

So this page is deliberately only half the material. The other half is a repository.

---

<a id="b0-1"></a>
### B0.1 The two layers

**This page is the explanation layer.** Every section gives the complete implementation,
annotated, plus the mistakes that recur and what the interviewer is actually checking.
Read it once to build the model, and return to it when an exercise defeats you.

**`practice/` is the training layer.** It contains the same problems as stubs — signature
and docstring, body removed — with a test suite that compares your version numerically
against PyTorch and against a reference solution.

```bash
git clone https://github.com/jxzhangjhu/jxzhangjhu.github.io
cd jxzhangjhu.github.io/interview-practice

python run.py                 # the problem set, budgets, and your history
python run.py p01             # start the clock, run that problem's tests
python run.py --cold          # the from-an-empty-file set, in order
python run.py --reset p01     # wipe your attempt and redo it
```

Every section below ends with an **Exercise** line linking straight to that problem's
stub and its graded hints, so you can read one on GitHub without cloning anything. The
`pNN` identifiers are what `run.py` takes as arguments.

Tests are written to diagnose rather than merely fail. When causal masking leaks you get
*"perturbing the last token changed earlier outputs: the mask leaks the future"*, not a
tensor dump.

> **On the reference solutions.** They live in `interview-practice/reference.py` and every one is
> checked against a PyTorch ground truth — 28 self-checks, all green, or the page is
> wrong. Do not open that file while practising; that is what the graded hints are for.

---

<a id="b0-2"></a>
### B0.2 Three things that make practice work

**Put a clock on it.** Every problem carries a budget — 20 minutes for multi-head
attention, 5 for RMSNorm, 30 for a small autograd. Practising untimed builds confidence
that evaporates in the room, because the failure mode under pressure is not conceptual.
It is forgetting `.contiguous()`, inverting a mask, or dividing by the wrong square root.

**Keep a small cold-start set.** Eleven of the twenty-six problems are marked `cold`:
these you should be able to write from an empty file, and they are worth repeating
weekly. For everything else the bar is lower — recall the shape of the solution, and be
able to reconstruct it with a hint. Trying to hold all twenty-six at muscle-memory level
is how people burn out a week before the loop.

**Drill the bugs separately, and do it first.** Public interview reports say debugging
is the *most common* ML-coding format, not writing from scratch — so `drills/` is not a
side dish. Two of them (`d09`, `d10`) are full reproductions of the most-reported
questions at OpenAI and Anthropic; the other eight are micro-drills with exactly one
wrong line, about three minutes each. Section B9 covers all of it, and it comes first in
this page for a reason.

---

<a id="b0-3"></a>
### B0.3 The problem set

`cold` marks the from-an-empty-file set. Budgets are interview budgets, not how long it
takes to understand the answer.

**The "reported in" column is the most useful one.** It records what public interview
accounts say about each question — which lab was reported asking it, and roughly how
often. Practise in that order rather than in mine. A blank means the question is standard
but not specifically attributed, not that it is unimportant.

<!-- TABLE -->
| | Problem | Budget | Cold | Reported in |
|---|---|---|---|---|
| **B1 · NumPy and PyTorch** | | | | |
| p24 | 1-NN in pure NumPy, no loops | 15 min | ● | OpenAI 3+ |
| p25 | BatchNorm forward, backward, eval mode | 20 min |  | Datadog |
| **B2 · Components** | | | | |
| p01 | Causal multi-head attention | 20 min | ● | universal |
| p02 | KV cache and incremental decode | 15 min | ● | OpenAI 7+ (as follow-up) |
| p03 | Grouped-query attention | 10 min |  | Datadog |
| p04 | Rotary position embeddings | 15 min | ● |  |
| p05 | RMSNorm | 5 min | ● |  |
| p06 | SwiGLU feed-forward | 5 min |  |  |
| p07 | A full pre-norm block | 15 min |  |  |
| **B3 · Training** | | | | |
| p08 | Cross entropy with log-sum-exp | 10 min | ● |  |
| p09 | SFT loss masking and packing | 10 min |  |  |
| p10 | Overfit a tiny batch | 20 min | ● |  |
| p26 | Filter bad human annotations | 20 min |  | OpenAI 2+ |
| **B4 · Backward** | | | | |
| p11 | A 40-line autograd | 30 min |  | OpenAI 2+ |
| p12 | Attention backward by hand | 25 min |  | OpenAI |
| p13 | MLP backward by hand | 15 min |  |  |
| **B5 · Inference** | | | | |
| p14 | Temperature, top-k, top-p | 15 min | ● |  |
| p15 | Speculative decoding accept/reject | 20 min |  |  |
| **B6 · Efficiency** | | | | |
| p16 | Streaming softmax | 15 min |  |  |
| p17 | Tiled FlashAttention forward | 25 min |  |  |
| **B7 · Post-training** | | | | |
| p18 | LoRA with a lossless merge | 10 min | ● |  |
| p19 | GRPO objective | 20 min | ● | OpenAI + Anthropic 4+ |
| p20 | DPO loss | 15 min |  |  |
| p21 | Generalised advantage estimation | 15 min |  |  |
| **B8 · Data** | | | | |
| p22 | Byte-pair encoding | 20 min | ● |  |
| p23 | Top-1 MoE routing with capacity | 20 min |  |  |
| **C2 · Simulation** | | | | |
| p27 | Spinning light source -> Cauchy | 20 min |  | OpenAI |

| | Drill | Budget | The one wrong thing | Reported in |
|---|---|---|---|---|
| d09 | minigpt | 35 min | four planted bugs in a nanoGPT, then add a KV cache | OpenAI 7+ — the single most reported ML-coding question |
| d10 | grpo_loop | 30 min | three planted bugs in a GRPO training script | Anthropic 3+, also OpenAI |
| d01 | mask_inverted | 3 min | masked_fill fills where the mask is True |  |
| d02 | missing_contiguous | 3 min | view() after transpose on a non-contiguous tensor |  |
| d03 | top_p_off_by_one | 4 min | the token that crosses the threshold gets dropped |  |
| d04 | cache_mask_offset | 5 min | tril without diagonal=T_full-T during cached decode |  |
| d05 | lora_both_random | 3 min | B initialised randomly, so the adapter is not identity |  |
| d06 | softmax_overflow | 3 min | exp without subtracting the row max |  |
| d07 | wrong_scale | 3 min | dividing by sqrt(d_model) instead of sqrt(d_head) |  |
| d08 | prompt_not_masked | 4 min | loss computed over prompt tokens as well |  |
<!-- TABLE -->

**A four-week rotation that fits around a job.** Week one, the two flagship drills
(`d09`, `d10`) plus the cold set once each with hints allowed — coverage, not speed.
Week two, the remaining fifteen, same rules. Week three, the cold set again with no hints
and the clock enforced; anything you miss goes on a short list. Week four, the short list
plus the eight micro-drills, and one full pass of the cold set the day before.

> **The honest failure mode of any scheme like this** is grinding the problems you
> already pass because they feel good. Your history is recorded in
> `practice/attempts.local.json`; sort by "last failed" and start there.

---

<a id="section-b9"></a>

## B9 · Debugging

Put this section first if you are short on time.

Public interview reports make the ranking unambiguous: **debugging is the most common
ML-coding format**, not writing from scratch. At OpenAI, "debug this transformer" appears
in seven or more independent accounts, more than any other question. At Anthropic the top
two are both debugging — a GRPO training loop and a NumPy bug hunt. The from-scratch
questions in the rest of this page are real, but they are not where the mass is.

The format is consistent enough to prepare for specifically: you get code that **runs
without raising** and produces wrong results. The bugs are logical, not syntactic. There
is usually a stated count ("about four"), and usually a follow-up that extends the code
rather than fixing it.

---

<a id="b9-1"></a>
### B9.1 A method that works under a clock

Reports from people who passed converge on the same approach, and it is not "read the
code more carefully."

**Reproduce deterministically first.** Seed everything, put the model in `eval()`, use
greedy decoding. You cannot tell whether a change helped if the output moves on its own.

**Localise with assertions, not with reading.** Print shapes at every stage. Assert the
invariants you know must hold: attention rows sum to one, cached decode equals full
recompute, a permuted input produces a permuted output when position is disabled. Each
assertion halves the search space; re-reading does not.

**Fix one bug at a time and re-run.** Bugs mask each other. In the drill below, the
scrambled head/time reshape hides the fact that the model cannot see position, because
scrambling also breaks permutation equivariance. If you fix three things then run, you
will not know which one mattered.

**Trust the comments if the file has them.** Several reports mention the buggy regions
are marked, and passing candidates say the same thing: under time pressure, do not
re-audit unmarked code. If there is no markup, ask — it is a reasonable question.

**Say what class of bug you found.** "The mask is applied after the softmax, so rows do
not sum to one" is a much stronger signal than "fixed it." The interviewer is scoring
your model of the system, not your diff.

> **The single best preparation for this format** is to have written a nanoGPT-style
> model end to end at least once, from the embedding table to the training loop. Multiple
> reports say familiarity with nanoGPT was sufficient on its own.

---

<a id="b9-2"></a>
### B9.2 The miniGPT drill

This reproduces the OpenAI question directly: a small decoder-only LM that runs, trains,
and generates garbage, with four planted bugs and a KV cache follow-up.

```
python -m pytest tests/test_d09_minigpt.py -q      # 35 min budget
```

The four bug classes, which are the same ones the reports name:

| Bug | Symptom you can test for |
|---|---|
| Positional embedding indexed with a constant | Permuting the input just permutes the output |
| Causal mask applied *after* the softmax | Attention rows no longer sum to 1 |
| Heads merged without transposing time back | Silently wrong output, no exception |
| Training loop never steps the optimiser | No parameter changes after a step |

**The third one deserves attention** because it is the one that teaches you something.
The attention output is `(B, n_heads, T, d_head)` and you want `(B, T, C)`. Reshaping
directly *works* — the element count matches — and interleaves head and time. Nothing
raises. The model trains to a worse loss and you have no error to chase. This is the
canonical example of why `.transpose(1, 2).contiguous().view(...)` is written the way it
is, and why shape-suffixed variable names (`y_BHTD`) pay for themselves.

**The follow-up is a KV cache**, and there is one detail almost everyone misses: the new
token's **positional index is the cache length**, not zero. Decode step $$t$$ must embed
position $$t$$. Get that wrong and generation degrades in a way teacher-forced evaluation
never shows you. State the invariant out loud — cached decode must be numerically
identical to a full recompute — and then test it.

---

<a id="b9-3"></a>
### B9.3 The GRPO loop drill

The Anthropic version: a complete GRPO training script, roughly 150 lines, that runs end
to end. Two bugs are numerical and one is algorithmic.

```
python -m pytest tests/test_d10_grpo_loop.py -q    # 30 min budget
```

**Raw logits passed to `torch.multinomial`.** That function takes unnormalised *weights*,
not logits, so you silently sample from the wrong distribution — and negative logits make
it worse. Fix: softmax first.

**Advantage normalised by a bare standard deviation.** When every completion in a group
earns the same reward, the standard deviation is zero and the advantage is NaN, which
propagates into every parameter on the next step. This is not a corner case: a group that
is all-correct or all-wrong is the *common* case both early and late in training, and it
is the same zero-variance situation that makes those groups worthless for learning
(Part I, A9.5). Fix: `+ 1e-5`.

**The ratio computed as a log difference.** The importance ratio is
$$\exp(\log \pi_\theta - \log \pi_{\text{old}})$$. Using the difference itself is not a
ratio, and the tell is sharp: on-policy, where new and old log-probs are equal, the ratio
must be exactly 1 and the unclipped surrogate must equal the advantage. A log difference
gives zero there, so the objective has no gradient at exactly the point where training
starts.

**Then comes the actual question**, and it is a discussion question rather than a code
one:

> This loop is strictly on-policy. Why is the mean importance ratio not exactly 1?

A good answer names several causes and, for each, what you would check:

- **More than one optimiser step per rollout batch.** After the first step the policy has
  moved, so the remaining mini-epochs are off-policy by construction. Check the number of
  inner epochs.
- **The sampling engine is not the training engine.** Rollouts from vLLM and log-probs
  recomputed in HF will not agree bit for bit — different kernels, different attention
  implementations, different precision. Check by recomputing log-probs for the same
  tokens in both and diffing.
- **Sampling parameters applied at generation but not at scoring.** Temperature, top-p,
  and logit bias change the distribution you actually sampled from. If you score with the
  raw distribution, your "old" log-probs are of the wrong policy.
- **Precision and non-determinism.** fp32 versus bf16 log-prob accumulation, or fused
  versus eager attention, moves the ratio slightly even with identical weights.

Being able to separate **expected-by-design drift** from **an actual bug** is what this
question is testing.

---

<a id="b9-4"></a>
### B9.4 Micro-drills

Eight implementations with exactly one wrong line each, about three minutes apiece. These
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

> **Why micro-drills beat re-implementation for this skill.** Writing attention from
> scratch trains production; finding an inverted mask trains recognition. The debugging
> round tests recognition, and you can get twenty repetitions of it in the time one
> from-scratch attempt takes.

---

<a id="section-b1"></a>

## B1 · NumPy and PyTorch fundamentals

Two questions in this section are reported by name — 1-NN in pure NumPy at OpenAI, and
BatchNorm at Datadog — but the real reason it comes first is that everything later
depends on it. Most failures in the from-scratch questions are not conceptual. They are
a transpose, a broadcast, or a dtype.

---

<a id="b1-1"></a>
### B1.1 Vectorisation: the one trick worth memorising

The question, reported at OpenAI three or more times, is 1-nearest-neighbour in NumPy
with no loops. The point is not the classifier. It is whether you know how to turn a
pairwise distance computation into a matrix multiply.

$$\|a - b\|^2 = \|a\|^2 - 2\,a \cdot b + \|b\|^2$$

Expand the square and the cross term becomes a single matmul, which BLAS executes far
faster than anything you can write with loops.

```python
def nearest_neighbour(train_x, train_y, test_x):
    # (n_test, 1) + (1, n_train) - 2 * (n_test, n_train)
    d2 = (
        (test_x ** 2).sum(1)[:, None]
        + (train_x ** 2).sum(1)[None, :]
        - 2 * test_x @ train_x.T
    )
    return train_y[np.argmin(d2, axis=1)]
```

**Three things to say while writing it.** You never take a square root, because
`argmin` is invariant to monotone transforms and `sqrt` costs a pass over the whole
matrix. The broadcast is `(n_test, 1)` against `(1, n_train)`, and being explicit with
`[:, None]` rather than relying on implicit alignment is what keeps this readable. And
the expansion can go slightly negative from floating-point cancellation when points
nearly coincide — harmless for `argmin`, but `np.maximum(d2, 0)` before a `sqrt`.

> **The follow-up in the reports is memory.** This materialises an
> `n_test × n_train` matrix. At 100k × 100k that is 80 GB, so you chunk over test rows
> and keep a running best. Say that before you are asked.

**Exercise** — [`p24` · 1-NN in pure NumPy, no loops](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p24_nn_vectorized.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p24_nn_vectorized.md) · 15 min · cold-start set · *OpenAI 3+*

---

<a id="b1-2"></a>
### B1.2 BatchNorm, and why it has two modes

Reported at Datadog. It looks like a warm-up and it is not, because the interesting part
is the state, not the formula.

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

**The four details that separate answers.**

**Train and eval compute different functions.** In training the statistics come from the
batch; in evaluation they come from the running estimate. This is the only common layer
where forgetting `.eval()` changes the output, and it is why BatchNorm breaks
batch-size-1 inference and distributed training in ways LayerNorm does not.

**`register_buffer`, not `nn.Parameter`.** The running statistics move with `.to(device)`
and are saved in the state dict, but they receive no gradient. Making them parameters is
a common and quietly wrong answer.

**Biased for normalising, unbiased for the running estimate.** PyTorch normalises with
the biased variance ($$/n$$) and accumulates the unbiased one ($$/(n-1)$$). Match this or
your implementation diverges from `nn.BatchNorm1d` in eval mode only — an excellent
example of a test that must cover both modes.

**Epsilon is inside the square root**, not outside. Outside, it fails to protect against
a zero variance at all.

> **The follow-up worth pre-loading:** why did transformers move to LayerNorm? Variable
> sequence length, coupling between examples in a batch (which breaks autoregressive
> generation at batch 1), and a cross-device synchronisation on every forward pass in
> distributed training. Part I, A1.7 has the full version.

**Exercise** — [`p25` · BatchNorm forward, backward, eval mode](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p25_batchnorm.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p25_batchnorm.md) · 20 min · *Datadog*

---

<a id="b1-3"></a>
### B1.3 The tensor semantics that cause silent bugs

Four things account for most of the damage.

**`view` versus `reshape`.** `view` requires contiguous memory and refuses otherwise;
`reshape` falls back to a copy. After `transpose` you are non-contiguous, so `view`
raises — which is the *good* case, because it tells you. The bad case is when the element
count happens to line up and a reshape silently interleaves the wrong axes, which is
exactly bug three in the miniGPT drill (B9.2).

**Broadcasting aligns from the right.** `(B, T, C) * (C,)` works; `(B, T, C) * (B,)` does
not. When you mean a per-batch scale you must write `(B, 1, 1)`. Being explicit with
`None` indexing rather than relying on the alignment rule is the habit that prevents this.

**In-place operations and autograd.** `x += 1` on a tensor needed for the backward pass
raises a version-counter error; `x = x + 1` does not. In-place is worth it for optimiser
state and running statistics, and rarely worth it elsewhere.

**dtype promotion is silent.** bf16 times fp32 gives fp32. That is how a normalisation
layer can quietly return the wrong dtype (B2.5), and how a "bf16" training run ends up
with fp32 activations in places you did not intend.

**Exercise** — the micro-drills `d02`, `d06`, `d07` in B9.4 target exactly these.

---

<a id="section-b2"></a>

## B2 · Transformer components

The baseline round. Every lab asks for some subset of this, and the bar is not
"do you know what attention is" — it is whether twenty lines come out clean under a
clock, and whether you catch your own mistakes before the interviewer does.

Everything here is in `interview-practice/reference.py` and validated against PyTorch.

---

<a id="b2-1"></a>
### B2.1 Causal multi-head attention

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

The scaling keeps logits at unit variance so the softmax does not saturate at
initialisation; the mask is additive $$-\infty$$ *before* the softmax so masked positions
contribute nothing to the denominator. Both arguments are in Part I (A2.3).

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
        # buffer, not parameter: moves with .to(device) and is saved, but gets no gradient
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x):
        B, T, C = x.shape
        # one fused matmul for q, k, v is cheaper than three separate ones
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))

        y = att @ v                                   # (B, n_heads, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))
```

**The four things that go wrong, in order of frequency.**

1. **`.contiguous()`.** After `transpose(1, 2)` the tensor is a view with non-contiguous
   strides, and `.view()` raises. Use `.reshape()` if you prefer, but know the difference:
   `view` never copies and therefore refuses; `reshape` falls back to a copy.
2. **Scaling by $$\sqrt{d_\text{model}}$$ instead of $$\sqrt{d_\text{head}}$$.** The dot
   product runs over the head dimension, so that is the variance you are correcting. This
   still trains, just worse — which is exactly why it is asked.
3. **Masking after the softmax.** Zeroing masked positions afterwards leaves them in the
   denominator, so surviving weights no longer sum to one, and the error varies by row.
4. **Three separate projections.** Mathematically identical, measurably slower: one large
   GEMM beats three small ones at the same FLOP count.

**Offer the causality test before you are asked.** It takes three lines and it is the
single strongest signal available in this question:

```python
y1 = model(x)
x2 = x.clone(); x2[:, -1, :] += 10.0
assert torch.allclose(y1[:, :-1], model(x2)[:, :-1])   # the past cannot see the future
```

> **What they are watching for.** Shape discipline, whether you notice the contiguity
> problem without being prompted, and whether you verify your own work. Candidates who
> write the test unprompted are in a different category from candidates who need to be
> asked "how would you check this?"

**Exercise** — [`p01` · Causal multi-head attention](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p01_mha.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p01_mha.md) · 20 min · cold-start set

---

<a id="b2-2"></a>
### B2.2 KV cache and incremental decode

At decode step $$t$$ you have exactly one query, but you need every previous key and
value. Q is transient; K and V accumulate. Without a cache you recompute all of history
every step, which is $$O(T^2)$$ wasted work.

```python
def forward(self, x, cache=None):
    B, T, _ = x.shape
    q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    k = self.wk(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
    v = self.wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

    if cache is not None:
        if "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        cache["k"], cache["v"] = k, v

    k = k.repeat_interleave(self.n_rep, dim=1)        # expand kv heads to match q heads
    v = v.repeat_interleave(self.n_rep, dim=1)

    T_full = k.shape[2]
    att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
    # query i sits at absolute position T_full - T + i, so that is the diagonal offset
    causal = torch.ones(T, T_full, dtype=torch.bool, device=x.device).tril(
        diagonal=T_full - T
    )
    att = att.masked_fill(~causal, float("-inf"))
    y = F.softmax(att, dim=-1) @ v
    return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))
```

**The mask offset is the whole question.** During prefill $$T = T_\text{full}$$ and a
plain `tril` is right. During cached decode your query block starts partway down the
matrix, so you need `diagonal=T_full - T`. Get this wrong and the model is fine in
teacher-forced evaluation and quietly degrades during generation — one of the nastier
bugs to find in production because your eval never sees it.

**The correctness property to state out loud:** cached incremental decode must be
**numerically identical** to a full recompute. That is testable, so test it.

> **What they are watching for.** Whether you realise the mask changes. Most candidates
> write the cache concatenation correctly and then reuse the prefill mask.

**Exercise** — [`p02` · KV cache and incremental decode](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p02_kv_cache.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p02_kv_cache.md) · 15 min · cold-start set

---

<a id="b2-3"></a>
### B2.3 Grouped-query attention

One line separates GQA from MHA: query heads are partitioned into groups, and each group
shares one K/V head.

```python
k = k.repeat_interleave(self.n_rep, dim=1)   # n_rep = n_heads // n_kv_heads
v = v.repeat_interleave(self.n_rep, dim=1)
```

$$n_\text{kv} = 1$$ is MQA, $$n_\text{kv} = n_\text{heads}$$ is plain MHA, and everything
between is a tunable knob on the KV cache — which is the only reason GQA exists.

**The follow-up that catches people: this does not reduce attention FLOPs.** K and V are
expanded back to the full head count before the matmuls, so $$QK^\top$$ and $$AV$$ are
unchanged. What shrinks is the cache and the bandwidth needed to read it, and since
decode is bandwidth-bound that is where the speedup comes from. (Be precise if pushed:
the K/V *projections* do get smaller, from $$2D^2$$ to $$2DKH$$ per layer.)

**Exercise** — [`p03` · Grouped-query attention](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p03_gqa.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p03_gqa.md) · 10 min

---

<a id="b2-4"></a>
### B2.4 Rotary position embeddings

RoPE rotates each coordinate pair by an angle proportional to position, which makes the
attention logit depend only on the relative offset. The three-line proof is in Part I
(A2.6); here it is the implementation that matters.

```python
def rope_cache(seq_len, d_head, base=10000.0):
    theta = base ** (-torch.arange(0, d_head, 2).float() / d_head)   # (d_head/2,)
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, theta)                                  # (T, d_head/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (..., T, d_head). Rotate the pairs (0,1), (2,3), ... independently.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)
```

**Three details.** It is applied to **Q and K only**, after the head split and before the
dot product — never to V, which carries content rather than position. With a KV cache you
store the **post-rotation** keys. And the pairing convention (`0::2, 1::2` versus split-half)
must match between the table and the application, or you get a model that trains to a
worse loss with no error message.

**Exercise** — [`p04` · Rotary position embeddings](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p04_rope.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p04_rope.md) · 15 min · cold-start set

---

<a id="b2-5"></a>
### B2.5 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # The reduction genuinely has to run in fp32: summing d squared values in bf16
        # accumulates enough rounding error to shift the norm. Cast back at the end so
        # the layer is dtype-transparent.
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * rms).type_as(x) * self.weight.type_as(x)
```

Five lines, and the interesting one is the `float()`. No mean subtraction and no bias —
ablations show the re-scaling is what matters and the re-centering is not, and dropping
it saves a reduction, which matters when you do it twice per layer across eighty layers.

> **This is a real trap, not a stylistic one.** An implementation that does the reduction
> in bf16 passes every fp32 test you write and silently degrades in a bf16 training run.
> The exercise feeds it bf16 inputs of magnitude $$10^4$$ specifically to catch that.

**Exercise** — [`p05` · RMSNorm](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p05_rmsnorm.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p05_rmsnorm.md) · 5 min · cold-start set

---

<a id="b2-6"></a>
### B2.6 SwiGLU

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        # 8/3 keeps the parameter count equal to a 4x ReLU FFN: 3*d*F == 2*d*4d
        d_ff = d_ff or int(8 * d_model / 3)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

**Three matrices, not two** — that is the whole question, along with knowing why
$$F = \tfrac83 D$$: it is what makes the parameter count match a vanilla $$4D$$ FFN, so
the architectural comparison is at equal parameters and therefore means something.

**Exercise** — [`p06` · SwiGLU feed-forward](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p06_swiglu.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p06_swiglu.md) · 5 min

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
        x = x + self.attn(self.norm1(x))     # pre-norm: the residual stays an identity path
        x = x + self.mlp(self.norm2(x))
        return x
```

Two lines of forward. Pre-norm normalises the sublayer *input*, leaving a clean identity
path from embedding to output, which is what removes the architectural need for warmup.
The cost is that the residual stream grows with depth, so **a full model needs a final
norm before `lm_head`** — the single most commonly forgotten line when writing a model
from scratch.

**Exercise** — [`p07` · A full pre-norm block](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p07_transformer_block.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p07_transformer_block.md) · 15 min

---

<a id="section-b3"></a>

## B3 · The training loop

Nobody asks you to write a training loop as the main question. It shows up as the thing
you have to fix in a debugging round, or as the harness around a component you just
implemented — and one of the four canonical miniGPT bugs lives here.

---

<a id="b3-1"></a>
### B3.1 Cross entropy, and why it takes logits

```python
def cross_entropy(logits, targets, ignore_index=-100):
    keep = targets != ignore_index
    logits, targets = logits[keep], targets[keep]
    # log_softmax(x)[t] = x[t] - logsumexp(x); never build probabilities then log them
    logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -logprobs.gather(1, targets[:, None]).mean()
```

**Why the API takes logits rather than probabilities.** `log(softmax(x))` computes an
exponential, normalises, and then takes a logarithm — three chances to lose precision,
and `exp` of a large logit overflows to `inf` before you ever reach the log. The fused
form subtracts `logsumexp` directly, and `logsumexp` itself subtracts the row max first.
With logits around $$10^4$$ the naive version returns `nan` and the fused one is exact.

**The `ignore_index` detail that gets missed:** you must divide by the number of *kept*
tokens, not by $$N$$. Masking then averaging over everything silently scales your loss by
the keep fraction, which then interacts with your learning rate.

**And the case where nothing is kept.** A packed microbatch can end up fully masked, and
then this returns NaN — as does `F.cross_entropy` — which propagates into every parameter
on the next step. Return a zero that is still attached to the graph instead.

**Exercise** — [`p08` · Cross entropy with log-sum-exp](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p08_cross_entropy.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p08_cross_entropy.md) · 10 min · cold-start set

---

<a id="b3-2"></a>
### B3.2 Loss masking and packing

Two things that look like plumbing and are actually correctness.

```python
labels = input_ids.clone()                    # input_ids: (B, T)
for i, n in enumerate(prompt_lens):
    labels[i, :n] = -100                      # each example's own prompt length
labels[attention_mask == 0] = -100            # padding too
```

**The whiteboard slip to avoid:** `labels[:len(prompt_ids)] = -100` on a `(B, T)` tensor
slices the **batch** dimension, wiping out the first few examples entirely rather than
masking each example's prompt. It runs, it trains, and it is wrong.

**Packing** concatenates short examples into one fixed-length sequence to avoid padding
waste — which is often 50% of your compute. The catch is that tokens can then attend
across the document boundary. Either use a varlen kernel (FlashAttention with
`cu_seqlens`) or a block-diagonal mask; and reset `position_ids` per document, or
document two starts at position 512.

**Exercise** — [`p09` · SFT loss masking and packing](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p09_loss_masking.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p09_loss_masking.md) · 10 min

---

<a id="b3-3"></a>
### B3.3 Overfit ten examples before anything else

```python
def overfit_tiny(steps=2000, lr=0.5):
    torch.manual_seed(0)
    x, y = torch.randn(10, 4), torch.randint(0, 3, (10,))
    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3))
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(steps):
        opt.zero_grad()          # forgetting this is the single most common bug
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(x)
        return F.cross_entropy(logits, y).item(), (logits.argmax(-1) == y).float().mean().item()
```

**If a model cannot memorise ten examples, the bug is in your code, not your
hyperparameters.** This is the cheapest diagnostic in machine learning and the one most
people skip. Say it out loud in an interview — it signals that you have debugged real
training runs rather than only read about them.

**The three-line order matters and is asked about.** `zero_grad` → `backward` → `step`.
Gradients *accumulate* by default, so skipping `zero_grad` sums every step's gradient;
skipping `step` means nothing updates and your loss curve is flat; and calling `step`
before `backward` updates on stale gradients.

> **Why gradients accumulate at all**, since it causes so many bugs: it is what makes
> gradient accumulation across micro-batches work, which is how you get a large effective
> batch on limited memory. The default serves the harder case.

**Exercise** — [`p10` · Overfit a tiny batch](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p10_training_loop.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p10_training_loop.md) · 20 min · cold-start set

---

<a id="b3-4"></a>
### B3.4 Filtering bad annotations

Reported at OpenAI twice. It is not a modelling question — it is whether you can reason
about label noise without over-engineering.

```python
def filter_annotations(labels, annotators, min_agreement=0.6, min_items=3):
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
    ...
```

**The judgement being tested is `min_items`.** An annotator who labelled two items and
disagreed on both looks terrible, but two samples is noise, not evidence. Without a
minimum you flag every sparse annotator and throw away good data. Say this before you are
asked; it is the difference between a filter and a heuristic that eats your dataset.

**Three follow-ups worth having ready.** Majority voting is circular when the bad
annotators are a majority on an item — real pipelines use gold-standard items with known
answers as an independent check. Disagreement is not the same as error: on genuinely
ambiguous items everyone disagrees, so item difficulty and annotator quality are
confounded, which is what Dawid-Skene-style models estimate jointly. And a systematically
biased annotator is more dangerous than a random one, because their errors correlate and
survive averaging.

**Exercise** — [`p26` · Filter bad human annotations](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p26_data_filtering.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p26_data_filtering.md) · 20 min · *OpenAI 2+*

---

<a id="section-b4"></a>

## B4 · Backward passes by hand

Reported at OpenAI twice for autograd and at least once for attention backward. The
question is never "do you remember the chain rule" — it is whether you can keep track of
shapes while applying it, and whether you know what a framework is doing for you.

---

<a id="b4-1"></a>
### B4.1 The rule that generates every backward

You do not need to memorise per-layer formulas. One rule regenerates all of them:

> **The gradient with respect to any tensor has that tensor's shape**, and it is
> assembled from the incoming gradient and the other operands so that the shapes work
> out. There is usually exactly one way to contract them.

For $$Z = XW + b$$ with $$X: (m, n_\text{in})$$ and $$W: (n_\text{in}, n_\text{out})$$:

$$\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Z}W^\top,\qquad
\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Z},\qquad
\frac{\partial L}{\partial b}=\sum_i \frac{\partial L}{\partial z_{i}}$$

Check them by shape: $$(m, n_\text{out}) \times (n_\text{out}, n_\text{in})$$ gives
$$X$$'s shape; $$(n_\text{in}, m) \times (m, n_\text{out})$$ gives $$W$$'s. The bias
gradient sums over the batch because broadcasting in the forward means summation in the
backward — that pairing is worth stating explicitly, because it generalises to every
broadcast you will ever write.

**Why backward costs about twice forward.** You compute two products per layer instead of
one: the gradient with respect to the input, to keep propagating, and with respect to the
weights, to update. Hence $$2N + 4N = 6N$$ FLOPs per token in total (Part I, A10.0).

---

<a id="b4-2"></a>
### B4.2 A 40-line autograd

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad      # += , not =
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = 1.0
        for v in reversed(topo):                    # reverse topological order
            v._backward()
```

**Two things carry the whole answer.**

**`+=` rather than `=`.** A node used in two places receives gradient from both paths,
and the multivariable chain rule says they add. Assignment silently keeps only the last
one — and the bug is invisible on any graph where every node is used once, which is
exactly the graph you would test with.

**Reverse topological order.** A node's backward can only run once every consumer of it
has contributed. A plain DFS or BFS gives the wrong order on any diamond-shaped graph.

> **The follow-up: why does PyTorch build the graph dynamically?** Because the graph is
> just whatever operations ran, recorded as they run — which is why control flow, loops,
> and data-dependent shapes work without a compilation step. The price is that you rebuild
> it every iteration, which is what `torch.compile` claws back.

**Exercise** — [`p11` · A 40-line autograd](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p11_autograd.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p11_autograd.md) · 30 min · *OpenAI 2+*

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
def attention_backward(d_out, cache):
    q, k, v, p, scale = cache
    d_v = p.transpose(-2, -1) @ d_out
    d_p = d_out @ v.transpose(-2, -1)
    # softmax VJP: elementwise, with the row-sum term subtracted
    d_s = p * (d_p - (d_p * p).sum(-1, keepdim=True))
    d_s = d_s * scale
    d_q = d_s @ k
    d_k = d_s.transpose(-2, -1) @ q
    return d_q, d_k, d_v
```

**What the masked positions do.** They have $$p = 0$$, so `p * (...)` zeroes their
gradient automatically. You do not need to re-apply the mask in the backward — a detail
that surprises people and is worth mentioning.

**Why this matters beyond the interview:** this backward is exactly what FlashAttention
recomputes on-chip rather than reading from HBM, and the reason it can is that $$P$$ is
cheap to regenerate from $$Q$$ and $$K$$ but expensive to store (B6.2).

**Exercise** — [`p12` · Attention backward by hand](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p12_attention_backward.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p12_attention_backward.md) · 25 min · *OpenAI* · and `p13` for the MLP version

---

<a id="section-b5"></a>

## B5 · Inference and sampling

Short section, two questions, and both are more subtle than they look.

---

<a id="b5-1"></a>
### B5.1 Temperature, top-k, top-p

```python
def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0:                       # greedy; also guards the division
        return int(logits.argmax())
    logits = logits / temperature

    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        srt, idx = torch.sort(logits, descending=True)
        probs = F.softmax(srt, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        drop = cum - probs >= top_p            # exclusive cumsum: keep the crosser
        drop[0] = False                        # argmax always survives, so p=0 is still valid
        srt = srt.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, srt)

    return int(torch.multinomial(F.softmax(logits, dim=-1), 1))
```

**Order matters: temperature, then top-k, then top-p.** Temperature changes the
distribution the truncations act on, so applying it last selects a nucleus from the wrong
distribution.

**The off-by-one that changes your sampling distribution silently.** You want the
*shortest prefix whose cumulative mass reaches p*, which means the token that crosses the
threshold is **kept**. `cum - probs` is the exclusive cumulative sum — the mass strictly
before this token — and dropping where that already exceeds `p` gets it right. Writing
`cum >= top_p` instead drops the crossing token, and with `p = 0.9` on a distribution like
`[0.5, 0.3, 0.15, 0.05]` you silently sample from two tokens instead of three.

**`temperature == 0` needs an explicit branch**, or you divide by zero. This has shipped
in real inference servers.

**The follow-up:** why does top-p usually beat top-k? Because the support size adapts to
the model's confidence. When the model is sure, the nucleus is one or two tokens; when it
is unsure, it widens. A fixed $$k$$ is either too permissive on confident steps or too
restrictive on uncertain ones.

**Exercise** — [`p14` · Temperature, top-k, top-p](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p14_sampling.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p14_sampling.md) · 15 min · cold-start set

---

<a id="b5-2"></a>
### B5.2 Speculative decoding

The interesting property is that it is **exact** — it does not approximate the target
model's distribution, it reproduces it.

```python
def speculative_accept(p_target, q_draft, token, u):
    if u <= min(1.0, (p_target[token] / q_draft[token]).item()):
        return int(token), True
    resid = torch.clamp(p_target - q_draft, min=0)
    return int(torch.multinomial(resid / resid.sum(), 1)), False
```

Accept the draft's token with probability $$\min(1, p(x)/q(x))$$; on rejection, sample
from the normalised residual $$\propto \max(0, p - q)$$. This is rejection sampling, and
the resulting samples are provably distributed as $$p$$.

**Prove it in one line if asked.** The probability of emitting $$x$$ is
$$q(x)\min(1, p/q) + P(\text{reject})\cdot\frac{\max(0, p-q)}{\sum_y \max(0, p-q)}$$. The
first term is $$\min(q, p)$$, and the second supplies exactly the missing $$\max(0, p-q)$$,
summing to $$p(x)$$.

> **The test worth writing.** Sample 200,000 times and compare the empirical distribution
> to the target. Exactness is a claim you can check, so check it — the reference
> implementation does.

**Where the speedup comes from, and where it goes.** Decode is bandwidth-bound with idle
FLOPs, so verifying $$k$$ draft tokens in one parallel forward costs roughly the same
wall-clock as generating one. As batch size grows you are no longer bandwidth-starved,
verification competes for compute that is now scarce, and the benefit shrinks to zero and
then goes negative. It is a latency optimisation for interactive serving, not a
throughput one.

**Exercise** — [`p15` · Speculative decoding accept/reject](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p15_speculative.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p15_speculative.md) · 20 min

---

<a id="section-b6"></a>

## B6 · Efficient implementations

Two problems, and they are really one idea seen twice: you can compute a softmax without
ever holding all of its inputs, and that single fact is what makes long-context attention
tractable.

---

<a id="b6-1"></a>
### B6.1 Streaming softmax

Softmax looks like it needs a full pass before you can normalise anything — you need the
max for stability and the sum for the denominator. It does not. Keep a running max
$$m$$, a running denominator $$\ell$$, and a running numerator, and rescale whenever a
new block reveals a larger maximum.

```python
m = -inf; l = 0.0; acc = 0.0
for s_block, v_block in blocks:
    m_new = max(m, s_block.max())
    correction = exp(m - m_new)          # rescales everything accumulated so far
    l = l * correction + exp(s_block - m_new).sum()
    acc = acc * correction + exp(s_block - m_new) @ v_block
    m = m_new
out = acc / l
```

**The correction factor is the whole algorithm.** Everything accumulated so far was
computed relative to the old maximum; multiplying by $$e^{m_\text{old}-m_\text{new}}$$
re-expresses it relative to the new one. Both $$\ell$$ and the accumulator need it, and
forgetting the accumulator is the classic bug — the denominator is then right and the
numerator is not, which produces a plausible-looking but wrong result.

**This is exact**, not an approximation. Assert it against a naive softmax; the reference
implementation does.

> **Worth knowing for credit:** this recurrence is Milakov & Gimelshein (2018) and
> predates FlashAttention. FlashAttention's contribution is not the recurrence, it is the
> IO-aware tiling built on top of it.

**Exercise** — [`p16` · Streaming softmax](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p16_online_softmax.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p16_online_softmax.md) · 15 min

---

<a id="b6-2"></a>
### B6.2 Tiled FlashAttention forward

Now apply the recurrence with $$V$$ in the loop, tiling over both query and key blocks,
and you have FlashAttention's forward pass.

```python
for i, q_block in enumerate(query_blocks):
    m_i = full(-inf); l_i = zeros(); acc_i = zeros()
    for j, (k_block, v_block) in enumerate(kv_blocks):
        if causal and j_start > i_end:        # whole tile is in the future
            continue
        s = q_block @ k_block.T * scale
        if causal and tiles_overlap:
            s = s.masked_fill(future_mask, -inf)
        m_new = maximum(m_i, s.max(-1))
        corr = exp(m_i - m_new)
        l_i = l_i * corr + exp(s - m_new).sum(-1)
        acc_i = acc_i * corr[:, None] + exp(s - m_new) @ v_block
        m_i = m_new
    out[i] = acc_i / l_i[:, None]
```

**Three things to say about it.**

**Memory goes from $$O(N^2)$$ to $$O(N)$$** because the score matrix is never
materialised — only one tile exists at a time, in SRAM.

**FLOPs go *up*, not down.** The backward pass recomputes the attention matrix on-chip
rather than reading the stored one. Saying "FlashAttention reduces computation" is the
answer that marks you as having read a summary rather than the paper.

**It is faster anyway because the operation was bound by HBM traffic, not arithmetic.**
Trading FLOPs for memory traffic is a win on the memory-bound side of the roofline, and
knowing which side you are on is the actual skill.

**The causal optimisation worth mentioning:** with a causal mask, tiles entirely above
the diagonal can be skipped outright, and only the diagonal tiles need an elementwise
mask. That is close to a 2× saving that falls out of the tiling for free.

**Exercise** — [`p17` · Tiled FlashAttention forward](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p17_flash_attention.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p17_flash_attention.md) · 25 min

---

<a id="section-b7"></a>

## B7 · Post-training algorithms

GRPO is the highest-frequency item in this whole page after transformer debugging:
reported at both OpenAI and Anthropic, four or more accounts, and usually as a *debugging*
exercise rather than a from-scratch one (B9.3). Write it once from scratch anyway — you
cannot debug an objective you have never assembled.

---

<a id="b7-1"></a>
### B7.1 LoRA

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False              # the base is frozen: that is the win
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B stays zero so B @ A == 0 and the adapter is an exact no-op at step 0

    def forward(self, x):
        return self.base(x) + x @ self.A.T @ self.B.T * self.scaling

    def merged_weight(self):
        return self.base.weight + (self.B @ self.A) * self.scaling
```

**Two properties are being checked.** Identity at initialisation, which requires `B = 0`
— initialising both randomly corrupts your starting point and is the tell that you have
used LoRA through a library but never read it. And a **lossless merge**: the adapted
layer is just a weight matrix, so after training there is zero inference overhead, unlike
adapter layers which add depth. That is the actual reason LoRA won.

**Where the memory saving comes from** — not the weights. The base still has to be
resident. It comes from optimizer state and gradients: full AdamW fine-tuning is ~16
bytes per parameter, and with the base frozen only its 2 bytes of bf16 weights count,
while the other 14 apply to the adapter alone. For a 70B model that is 1,120 GB of state
down to roughly 140 GB.

**Exercise** — [`p18` · LoRA with a lossless merge](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p18_lora.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p18_lora.md) · 10 min · cold-start set

---

<a id="b7-2"></a>
### B7.2 The GRPO objective

```python
r = rewards.view(-1, G)
adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
adv = adv.reshape(B, 1)                    # one scalar per completion, broadcast to tokens

ratio  = (logp - logp_old).exp()
policy = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)

log_ratio = logp_ref - logp
kl = log_ratio.exp() - log_ratio - 1.0     # k3: unbiased AND non-negative

loss = ((policy + beta * kl) * mask).sum() / mask.sum()
```

**Four details, and each one is a place a bug hides** — the same four the Anthropic
debugging question plants (B9.3).

**The epsilon in the denominator is not cosmetic.** A group where every completion earns
the same reward has zero standard deviation, and that is the *common* case: prompts the
policy always solves or always fails. Without the epsilon you get NaN, which poisons every
parameter on the next step. The epsilon does not cover $$G = 1$$, though: the unbiased
`std` of a single sample is NaN before the epsilon is ever added, so guard that case
separately or pass `correction=0`.

**The ratio is `exp` of the log difference.** On-policy, where new and old log-probs
coincide, the ratio must be exactly 1 and the unclipped surrogate must equal the
advantage. A bare log difference gives zero there — no gradient at precisely the point
training begins.

**The KL is a per-token term in the loss**, not folded into the reward as in PPO, and it
uses Schulman's k3 estimator: with $$r = \pi_\text{ref}/\pi_\theta$$ sampled from
$$\pi_\theta$$, $$\widehat{\mathrm{KL}} = r - \log r - 1$$. It is unbiased *and*
non-negative per sample, whereas the naive $$-\log r$$ can come out negative on a single
sample, which is a meaningless KL estimate.

**The advantage is bandit-shaped.** One scalar per completion, broadcast to every token —
there is no per-token credit assignment at all. State this limitation before you are
asked.

**Exercise** — [`p19` · GRPO objective](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p19_grpo_loss.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p19_grpo_loss.md) · 20 min · cold-start set · *OpenAI + Anthropic 4+*

---

<a id="b7-3"></a>
### B7.3 DPO

```python
def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()
```

Four log-probs, one sigmoid. No reward model, no critic, no generation in the training
loop — it runs on SFT infrastructure at roughly 2× memory.

**The sanity check to state:** at the reference policy the margin is zero and the loss is
exactly $$\log 2 \approx 0.693$$. If your first-step loss is not that, your reference
log-probs are wrong.

**What it trades away.** It is off-policy — it learns from preferences collected on a
distribution the policy drifts away from — and it is vulnerable to *likelihood
displacement*, where the margin grows because the rejected response's probability falls
rather than the chosen one's rising, sometimes driving both down.

**Exercise** — [`p20` · DPO loss](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p20_dpo_loss.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p20_dpo_loss.md) · 15 min

---

<a id="b7-4"></a>
### B7.4 GAE

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages, gae, next_value = [], 0.0, last_value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.append(gae)
        next_value = values[t]
    return list(reversed(advantages))
```

$$\lambda = 0$$ recovers one-step TD (low variance, biased by the critic's error);
$$\lambda = 1$$ recovers Monte Carlo (unbiased, high variance). **Assert both limits** if
you implement it — it is the cheapest correctness check available, and offering it
unprompted is the point of the exercise.

**The loop runs backwards** because $$\hat A_t$$ depends on $$\hat A_{t+1}$$. Writing it
forwards is a common slip that produces plausible numbers.

**Exercise** — [`p21` · Generalised advantage estimation](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p21_gae.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p21_gae.md) · 15 min

---

<a id="section-b8"></a>

## B8 · Data and tokenization

Two implementations. BPE is the one that actually gets asked; MoE routing is here because
it is the only place a from-scratch question touches sparse layers.

---

<a id="b8-1"></a>
### B8.1 Byte-pair encoding

```python
def bpe_train(text, num_merges):
    ids = list(text.encode("utf-8"))          # bytes: no out-of-vocabulary, ever
    merges = {}
    for i in range(num_merges):
        counts = Counter(zip(ids, ids[1:]))
        if not counts:
            break
        best = max(counts, key=counts.get)
        new_id = 256 + i
        merges[best] = new_id
        ids = replace_pair(ids, best, new_id)
    return merges


def bpe_encode(text, merges):
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges.items():       # LEARNED order, not frequency order
        ids = replace_pair(ids, pair, new_id)
    return ids
```

**The one thing that is genuinely a bug source:** encoding applies merges in the order
they were **learned**, not by frequency in the string being encoded. Python dicts preserve
insertion order, so iterating `merges` is correct — but if you store them in a `set`, sort
them, or rebuild the dict, you get a tokenizer that round-trips inconsistently. That is a
nasty production bug because it is data-dependent.

**Why bytes rather than characters.** A byte-level vocabulary can represent any input, so
there is no out-of-vocabulary case ever. The cost is that non-Latin scripts consume more
tokens per character, which is a real cost and fairness issue worth raising unprompted.

**The decode table is worth writing** even if not asked, because it is how you test
round-tripping:

```python
table = {i: bytes([i]) for i in range(256)}
for (a, b), new in merges.items():
    table[new] = table[a] + table[b]
assert b"".join(table[i] for i in ids).decode("utf-8") == text
```

**The follow-up that always comes:** why can't the model count the r's in "strawberry"?
Because it never sees characters. The word may be three tokens and nothing in the
representation exposes the letters inside them. It is an artefact of the input
representation, not of reasoning.

**Exercise** — [`p22` · Byte-pair encoding](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p22_bpe.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p22_bpe.md) · 20 min · cold-start set

---

<a id="b8-2"></a>
### B8.2 Top-1 MoE routing with capacity

```python
def top1_route(logits, capacity):
    gates = F.softmax(logits, dim=-1)          # (T, E)
    gate, expert = gates.max(dim=-1)
    # each expert takes tokens in order until it is full; the rest overflow
    assignment = torch.full_like(expert, -1)
    counts = torch.zeros(logits.shape[-1], dtype=torch.long)
    for t in range(logits.shape[0]):
        e = expert[t]
        if counts[e] < capacity:
            assignment[t] = e
            counts[e] += 1
    return assignment, gate


def load_balancing_loss(logits):
    gates = F.softmax(logits, dim=-1)
    E = gates.shape[-1]
    f = F.one_hot(gates.argmax(-1), E).float().mean(0)   # fraction routed to each
    p = gates.mean(0)                                     # mean gate probability
    return E * (f * p).sum()                              # minimised at uniform, = 1
```

**Token dropping is the part people have not thought about.** All-to-all needs fixed-size
buffers, so each expert has a capacity limit; overflow tokens **skip the layer entirely**
and pass through on the residual stream. The consequence to state unprompted: the same
input can produce different outputs depending on what else is in the batch.

**The auxiliary loss is not there because the router lacks gradient** — it has one. The
gate probability multiplies the chosen expert's output, so the LM loss backpropagates into
the router; only the top-$$k$$ *selection* is non-differentiable. The problem is that this
gradient is self-reinforcing: experts that receive more tokens train faster, so the router
prefers them more, and routing collapses. Getting this right is a genuine differentiator,
because the "no gradient" version is repeated everywhere.

**Why the loss is $$E\sum_e f_e p_e$$:** $$f$$ is non-differentiable (it counts
assignments) and $$p$$ is differentiable, so the gradient flows through $$p$$ weighted by
observed load. It is minimised at uniform routing, where it equals 1.

**Exercise** — [`p23` · Top-1 MoE routing with capacity](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p23_moe_routing.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p23_moe_routing.md) · 20 min

---

<a id="section-c1"></a>

## C1 · Probability: four patterns that cover most of it

Alisa Liu's public math notes are the reference for the formulas — distributions,
expectations, inequalities, limit theorems, all with proofs. There is no point
reproducing them here. What they do not have is a **self-test layer**, so this section is
the pattern catalogue: four techniques that between them solve the large majority of
probability questions asked in these loops, each with the tell that says "use this one."

---

<a id="c1-1"></a>
### C1.1 First-step analysis

**The tell:** a process that repeats, and you want the expected time until something
happens.

Condition on the first step and write the unknown expectations in terms of themselves.
Expected flips to see two heads in a row:

$$E_0=\underbrace{\tfrac 12 (1+E_0)}_\text{tails, no progress}+\underbrace{\tfrac 12(1+E_1)}_\text{heads},
\qquad E_1=\underbrace{\tfrac 12(1+E_0)}_\text{tails, start over}+\underbrace{\tfrac 12\cdot 0}_\text{done}$$

Two equations, two unknowns, $$E_0 = 6$$.

**What makes it work is choosing the right state.** Here the state is "how much progress
toward HH do I have," which has three values (none, one H, done). Get the state wrong and
the equations do not close. That choice is the whole skill; the algebra is trivial.

**Same pattern, harder dressings:** gambler's ruin (state = current fortune), random walk
return times, expected steps for a Markov chain to hit a set.

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

---

<a id="c1-3"></a>
### C1.3 Max and min of $$n$$ variables — go through the CDF

**The tell:** anything about the largest or smallest of several draws.

Never attack the density directly. The maximum has a trivially simple CDF, because "the
max is at most $$x$$" means "all of them are at most $$x$$":

$$F_M(x) = P(\max_i X_i \le x) = [F_X(x)]^n$$

For the minimum, complement it: $$P(\min > x) = [1 - F_X(x)]^n$$. Differentiate at the
end if you actually need a density.

**Worth having memorised:** for $$n$$ iid uniforms on $$[0,1]$$,
$$\mathbb E[\max] = n/(n+1)$$ and $$\mathbb E[\min] = 1/(n+1)$$ — and the symmetry between
them is a good sanity check on any answer of this type.

---

<a id="c1-4"></a>
### C1.4 Symmetry as a proof technique

**The tell:** the answer feels like it should not depend on something.

Two examples that come up. In a random permutation, the probability that item $$i$$ lands
in position $$j$$ is $$1/n$$ for every pair — that is what makes C1.2 work. And in the
secretary problem, the probability that the best candidate appears in any given position
is uniform, which is why the analysis only involves *where* the maximum falls.

**Monty Hall is the anti-example**, and it is asked because it punishes the symmetry
instinct. The host's choice is *not* symmetric — he never opens the prize door — and that
asymmetry is exactly where the 2/3 comes from. When you invoke symmetry, say which
transformation the problem is invariant under; if you cannot name it, you are guessing.

---

<a id="c1-5"></a>
### C1.5 Which inequality to reach for

| You know | Use | Gives |
|---|---|---|
| Mean only, $$X \ge 0$$ | Markov | $$P(X \ge a) \le \mathbb E[X]/a$$ |
| Mean and variance | Chebyshev | $$P(\|X-\mu\| \ge k\sigma) \le 1/k^2$$ |
| Bounded, independent sum | Hoeffding / Chernoff | Exponential tail |
| Convex function of an expectation | Jensen | $$f(\mathbb E[X]) \le \mathbb E[f(X)]$$ |

**Markov is the weakest and the most useful**, because it needs almost nothing — and
Chebyshev is just Markov applied to $$(X-\mu)^2$$, which is worth being able to say.

**Jensen is the one that shows up in ML rather than in puzzles.** It is why the ELBO is a
lower bound, why $$\log \mathbb E[\cdot] \ge \mathbb E[\log \cdot]$$ matters in importance
sampling, and why the KL divergence is non-negative.

---

<a id="section-c2"></a>

## C2 · Simulate, then verify

A distinct question genre, reported at OpenAI: you are given a physical setup, asked to
simulate it, and then asked to verify that the samples match a distribution you derive
analytically. It sits exactly between the coding and the math rounds, which is why people
prepare for neither half of it.

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
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    return np.tan(theta)
```

**The point of the question is that the mean does not exist.** $$\int |x| f(x)\,dx$$
diverges, so the law of large numbers does not apply and the sample mean never settles —
it wanders forever, with occasional enormous jumps as a sample lands far out in the tail.
Demonstrate it rather than assert it: compute the running mean at $$10^4$$, $$10^5$$ and
$$4\times10^5$$ samples and show it is not shrinking. For any distribution with a finite
mean the error would fall like $$1/\sqrt n$$.

**The median is well behaved**, and estimating the location parameter with it instead is
the correct practical response.

**One trap in the verification, and it is a good one.** Comparing a histogram to the
analytic PDF on a window like $$[-5, 5]$$ *fails* if you are careless, and the reason is
not a coding error. NumPy's `density=True` normalises over the bins you plotted, but the
true Cauchy puts only $$\tfrac{2}{\pi}\arctan 5 = 0.874$$ of its mass in that window. The
tails are heavy enough that ignoring the truncation inflates your histogram by 14% and
makes a correct simulation look wrong. Compare against the *conditional* density:

```python
in_range = 2 * np.arctan(L) / np.pi
assert np.abs(hist - cauchy_pdf(centres) / in_range).max() < 0.01
```

> **Why this is a good interview question.** It has a clean derivation, a simulation any
> competent candidate can write, and then a verification step that separates people who
> check their work carefully from people who check it approximately. The truncation
> correction is not a trick — it is the kind of thing that makes real experiments
> reproduce or not.

**Exercise** — [`p27` · Spinning light source -> Cauchy](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p27_cauchy_simulation.py) · [hints](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p27_cauchy_simulation.md) · 20 min · *OpenAI*

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
one line. The CDF via a Kolmogorov–Smirnov statistic is more sensitive than a histogram
and needs no binning choices. A histogram is the most convincing to look at and the most
error-prone, for the truncation reason above.

**Say what you would do about variance.** Sample size sets your resolution: with $$n$$
samples a bin holding probability $$p$$ has relative error about $$1/\sqrt{np}$$, so
tail bins are noisy and you should not tighten your tolerance uniformly across them.

**Name the standard transformations**, since one of them is usually the intended answer:
inverse-CDF sampling for anything with a closed-form quantile, Box–Muller for Gaussians,
$$-\log U/\lambda$$ for exponentials, and the ratio of two standard normals for a Cauchy —
which is a nice cross-check on this very question, since it should produce the same
distribution as the tangent construction.

---

<a id="section-c3"></a>

## C3 · Linear algebra

The gap in most people's preparation, including Alisa's otherwise excellent notes, which
cover probability and calculus but not this. It gets asked because every object you touch
all day is a matrix, and because the questions double as a check on whether you
understand the methods you use.

---

<a id="c3-1"></a>
### C3.1 The four facts everything else follows from

**A matrix is a linear map, and its shape tells you between which spaces.** $$W$$ of shape
$$(m, n)$$ maps $$\mathbb R^n \to \mathbb R^m$$. Most shape bugs dissolve if you read
matrices this way rather than as grids of numbers.

**Rank is the dimension of the output space actually reached.** A low-rank matrix
squashes its input into a subspace, which is exactly what LoRA exploits: $$BA$$ with inner
dimension $$r$$ can only move the weights within an $$r$$-dimensional subspace, which is
both why it is cheap and why it cannot install substantial new knowledge.

**Eigenvectors are the directions a map only scales**, $$Av = \lambda v$$. They exist for
square matrices and, for symmetric ones, form an orthogonal basis with real eigenvalues —
which is why the Hessian and covariance matrices are so tractable.

**The SVD applies to every matrix**, square or not: $$A = U\Sigma V^\top$$, an orthogonal
rotation, a scaling along axes, another rotation. Singular values are the scale factors,
and truncating the small ones is the best rank-$$k$$ approximation in the Frobenius norm.
That single fact underlies PCA, low-rank compression, and the way people reason about
whether a weight update is "really" low rank.

---

<a id="c3-2"></a>
### C3.2 Positive semi-definiteness, and why it keeps appearing

$$M$$ is PSD when $$x^\top M x \ge 0$$ for all $$x$$, equivalently when all eigenvalues
are non-negative.

**Three places it decides something in ML.** A covariance matrix is PSD by construction,
because $$x^\top \Sigma x$$ is the variance of a projection and variances are
non-negative. A Hessian that is PSD everywhere means the objective is convex, so any
stationary point is a global minimum — and neural network losses are emphatically not
that, which is why we discuss saddle points instead. And a kernel matrix must be PSD for
the kernel trick to correspond to an inner product in some space.

**The follow-up worth pre-loading: why is a saddle point the typical critical point in
high dimensions?** At a random critical point each eigenvalue of the Hessian is
positive or negative with roughly even odds, so all $$d$$ agreeing in sign has
probability like $$2^{-d}$$. With $$d$$ in the millions, local minima are vanishingly
rare relative to saddles, which is why second-order intuition from two-dimensional
pictures misleads.

---

<a id="c3-3"></a>
### C3.3 Norms, conditioning, and the things that blow up

**Which norm, and why it matters.** The $$\ell_2$$ norm is what gradient clipping
measures; the Frobenius norm is the $$\ell_2$$ norm of a flattened matrix; the spectral
norm is the largest singular value, i.e. the most a matrix can stretch any vector. The
spectral norm is the one that governs stability, because it bounds how much a
perturbation grows when it passes through a layer.

**Condition number** $$\kappa = \sigma_\max/\sigma_\min$$ tells you how much relative
error is amplified. It is the reason we normalise inputs, the reason ill-conditioned
problems need small learning rates, and the reason Adam's per-parameter scaling helps —
it is approximating a diagonal preconditioner.

**Two numerical rules that come from this.** Never form $$X^\top X$$ to solve least
squares if you can avoid it: it squares the condition number, so you lose twice the
digits. And never invert a matrix to solve $$Ax = b$$ — use a factorisation
(`np.linalg.solve`, not `inv(A) @ b`), which is both faster and better conditioned.

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

---

<a id="section-c4"></a>

## C4 · Counting

Counting questions are graded on whether you can state *what you are counting* before you
count it. Most wrong answers are not arithmetic errors — they are counting the same object
twice, or counting orderings when the question did not ask for them.

---

<a id="c4-1"></a>
### C4.1 The one decision that determines the formula

Before writing anything, answer two questions: **does order matter**, and **can items
repeat**. That is a 2×2 table and it fixes the formula.

| | Order matters | Order does not |
|---|---|---|
| **No repeats** | $$\dfrac{n!}{(n-k)!}$$ | $$\dbinom{n}{k}$$ |
| **Repeats allowed** | $$n^k$$ | $$\dbinom{n+k-1}{k}$$ |

The bottom-right cell is the one people cannot reconstruct. It is **stars and bars**:
distributing $$k$$ identical items into $$n$$ labelled bins is the same as arranging
$$k$$ stars and $$n-1$$ bars in a row, so you choose which $$k$$ of the $$n+k-1$$
positions are stars.

**Say the mapping, not the formula.** "Each arrangement of stars and bars corresponds to
exactly one distribution" is a proof; quoting $$\binom{n+k-1}{k}$$ is a memory claim.

---

<a id="c4-2"></a>
### C4.2 Overcount, then divide

The most useful technique in counting: count something easy that overcounts by a known
factor, then divide.

**Arrangements of a multiset.** The letters of MISSISSIPPI: pretend all 11 letters are
distinct ($$11!$$), then divide by the orderings within each repeated group,
$$4!\,4!\,2!$$ for the S's, I's and P's.

**Circular arrangements.** $$n$$ people around a round table is $$(n-1)!$$, because each
distinct seating was counted $$n$$ times, once per rotation.

**Choosing then ordering.** $$\binom{n}{k}k! = n!/(n-k)!$$ recovers the permutation
formula, which is a good check that you have the right mental model rather than two
memorised formulas.

> **The tell that you are overcounting:** your answer is an integer multiple of the
> expected one, and the multiple is usually a factorial. If you get exactly 6× too many
> and there are 3 identical items, you know what happened.

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
0.368$$, essentially independent of $$n$$. That is worth remembering because it is
counter-intuitive and it comes up in the "hat check" family of questions.

**Almost always easier: count the complement.** "At least one" problems are usually
"1 − none", and "none" is a single product rather than an alternating sum. Reach for
inclusion–exclusion only when the complement is not simpler.

---

<a id="c4-4"></a>
### C4.4 Where counting meets ML

Two places these questions get dressed up in ML clothing.

**The birthday problem, applied to hash collisions and duplicate detection.** With
$$n$$ items into $$d$$ buckets, the probability of no collision is
$$\prod_{i=0}^{n-1}(1 - i/d) \approx e^{-n^2/2d}$$, so collisions become likely once
$$n \sim \sqrt d$$. This is why a 64-bit hash is not enough to deduplicate a trillion
documents, and it is the standard framing for the MinHash question in a data-pipeline
interview.

**Counting parameters.** Deriving a transformer's parameter count is a counting problem
with a shape argument attached, and it is asked constantly (Part I, A10). The discipline
is the same: say what you are counting — per layer, per head, embedding versus
unembedding — before you multiply anything.

---

<a id="section-c5"></a>

## C5 · Markov chains and random walks

The natural home of first-step analysis (C1.1), and the setting for several of the
classic puzzles. It also underlies more of modern ML than it gets credit for — MCMC,
diffusion, and the MDP framing of RL are all Markov chains wearing different hats.

---

<a id="c5-1"></a>
### C5.1 What the Markov property actually buys you

$$P(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = P(X_{t+1} \mid X_t)$$

The future is conditionally independent of the past given the present. The practical
consequence is that **the entire process is described by one transition matrix** $$P$$,
and $$n$$-step behaviour is $$P^n$$ — which turns dynamics questions into linear algebra.

**The modelling skill is choosing the state**, exactly as in first-step analysis. Almost
any process can be made Markov by enlarging the state to include whatever history
matters. "Expected flips to see HH" is not Markov in the last flip alone; it is Markov in
"how many H's of progress do I have."

**The stationary distribution** $$\pi$$ satisfies $$\pi P = \pi$$ — a left eigenvector
with eigenvalue 1. It exists and is unique for an irreducible aperiodic chain, and
$$P^n$$ converges to it at a rate set by the second-largest eigenvalue modulus. That
spectral gap is the mixing time, and it is why MCMC diagnostics talk about
autocorrelation.

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

---

<a id="c5-3"></a>
### C5.3 Random walks, and the dimension surprise

**Symmetric walk on $$\mathbb Z$$.** After $$n$$ steps, $$\mathbb E[X_n] = 0$$ and
$$\operatorname{Var}(X_n) = n$$, so the typical displacement grows like $$\sqrt n$$.
That $$\sqrt n$$ is the same one in the standard error of a mean and in the diffusion
scaling of noise schedules, and noticing the connection is worth a sentence.

**Pólya's theorem, which is the part people find memorable:** the symmetric random walk
is recurrent in one and two dimensions — it returns to the origin with probability 1 —
and **transient in three or more**, where the return probability is about 0.34. As the
saying goes, a drunk man finds his way home but a drunk bird does not.

**Expected return time.** For a recurrent chain the expected return time to state $$i$$
is $$1/\pi_i$$. On $$\mathbb Z$$ the walk returns with probability 1 but the expected
time is infinite — recurrent, yet null recurrent. Being able to hold both of those
statements at once is the actual test here.

---

<a id="c5-4"></a>
### C5.4 The puzzles worth having pre-solved

These recur, and each takes minutes if you have seen it and much longer if you have not.

**Coupon collector.** $$n H_n \approx n\ln n$$ draws to see all $$n$$ coupons, from
decomposing into geometric waiting times (C1.2). The variance is $$O(n^2)$$, so the tail
is heavy — relevant whenever you are sampling to cover a space.

**Birthday problem.** Collisions become likely at $$n \sim \sqrt d$$, not $$n \sim d$$
(C4.4).

**Monty Hall.** Switching wins 2/3. The reason is that the host's action is not
independent of the truth — he never opens the prize door — so his choice transfers
information. Frame it as conditioning on the host's *rule*, not on the door.

**Secretary problem.** Reject the first $$n/e$$ candidates, then take the next one better
than everything seen. Success probability $$1/e \approx 0.37$$, and the same $$1/e$$
shows up in derangements (C4.3) for unrelated reasons — do not claim a connection.

**Reservoir sampling.** Keep item $$k$$ with probability $$1/k$$, replacing a uniformly
chosen incumbent. A one-line induction shows every item ends with probability $$k/n$$.
This one is *also* a coding question, and it is the right answer whenever an interviewer
says "a stream you cannot store."

**Two-heads-in-a-row.** Expected 6 flips (C1.1). The follow-up — expected flips for HT —
is 4, and the fact that HH and HT differ at all is the interesting part: after a failed
HH attempt you may retain partial progress, and the overlap structure of the pattern is
what changes the answer.

---

<a id="section-c6"></a>

## C6 · Statistics and estimation

The section that connects the probability above to the losses you actually minimise.
Every question here has an ML answer attached, and giving it is what separates this from
an undergraduate exam.

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
maximising likelihood is minimising squared error. Whenever you use MSE you are assuming
Gaussian noise, whether or not you meant to.

**Bernoulli gives cross entropy.** $$\log p = y\log\hat y + (1-y)\log(1-\hat y)$$, which
is exactly binary cross entropy. Categorical gives the multi-class version.

**Laplace gives absolute error.** $$\log p \propto -|x - \mu|$$, so L1 loss corresponds
to heavy-tailed noise — which is the principled reason L1 is more robust to outliers, and
a much better answer than "L1 is robust."

> **The follow-up that goes with all three:** MAP estimation adds a prior, and a Gaussian
> prior on the weights is exactly L2 regularisation while a Laplace prior is L1. That is
> where weight decay comes from, and it explains why L1 induces sparsity — the Laplace
> density has a spike at zero.

---

<a id="c6-2"></a>
### C6.2 Bias, variance, and why the biased estimator is the default

For an estimator $$\hat\theta$$:

$$\mathbb E[(\hat\theta - \theta)^2] = \underbrace{(\mathbb E[\hat\theta]-\theta)^2}_{\text{bias}^2}
+ \underbrace{\operatorname{Var}(\hat\theta)}_{\text{variance}}$$

**The $$n-1$$ question, and the answer people fumble.** The sample variance with $$1/n$$
is biased low, because you measured deviations from the *sample* mean, which is itself
fitted to the data and therefore sits closer to the points than the true mean does.
Dividing by $$n-1$$ corrects it — you spent one degree of freedom estimating the mean.

**Then the part that makes it an ML answer:** unbiasedness is not the goal, mean squared
error is, and a biased estimator with lower variance often wins. That is why
`torch.var(unbiased=False)` is what normalisation layers use — you want the actual
variance of these activations, not an estimate of a population parameter. It is also
exactly the subtlety in BatchNorm, which normalises with the biased variance and
accumulates the unbiased one (B1.2).

**Where bias-variance stops working.** It is a statement about a fixed hypothesis class,
and it predicts that very large models overfit badly. They do not — double descent is the
empirical refutation, and the modern picture is that over-parameterised models find
low-complexity interpolants rather than sitting at the classical high-variance end
(Part I, A1.8).

---

<a id="c6-3"></a>
### C6.3 Concentration: how many samples do you need

This is the practical form of the inequalities in C1.5, and it is what you actually use
when someone asks how many eval examples are enough.

**For a proportion**, the standard error is $$\sqrt{p(1-p)/n}$$, worst case
$$1/(2\sqrt n)$$. So a 95% confidence interval is roughly $$\pm 1/\sqrt n$$: 100 examples
gives $$\pm 10\%$$, 1,000 gives $$\pm 3\%$$, 10,000 gives $$\pm 1\%$$. **Memorise those
three**, because they let you say immediately that a 2-point difference on a 500-example
benchmark is noise.

**Hoeffding** gives the same shape without a normal approximation: for bounded variables,
$$P(|\bar X - \mu| \ge t) \le 2\exp(-2nt^2)$$, so $$n \sim \log(1/\delta)/t^2$$. The
$$1/t^2$$ is the expensive part — one more digit of precision costs 100× the samples.

**The paired-comparison trick worth knowing.** When comparing two models on the same
examples, compare per-example differences rather than the two means. The variance of the
difference is usually far smaller because example difficulty cancels, and you can detect
a real gap with an order of magnitude fewer examples. This is the standard answer to "our
eval is too noisy to tell these apart."

---

<a id="c6-4"></a>
### C6.4 Hypothesis testing, briefly, and its ML failure mode

You will not be asked to run a t-test. You may well be asked why a benchmark improvement
is or is not real.

**The vocabulary you need:** a p-value is $$P(\text{data this extreme} \mid H_0)$$ —
*not* the probability that the null is true, and getting that backwards is the classic
error. A confidence interval is more informative than a p-value because it carries the
effect size.

**The failure mode that actually matters in ML is multiple comparisons.** Evaluate twenty
checkpoints on the same benchmark and one will look significant at $$p < 0.05$$ by
construction. Every hyperparameter sweep, every "we tried a few variants," is a multiple
comparison problem, and the honest responses are a held-out set touched once, a
correction (Bonferroni is crude but defensible), or pre-registering what you will measure.

> **The strongest thing you can say here** is that this is the same failure as
> overfitting to a validation set, seen through a statistical lens. Selecting a model on
> a benchmark makes that benchmark an optimistic estimate of its performance, and the
> only durable fix is a set you have not selected on.
