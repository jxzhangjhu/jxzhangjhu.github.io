---
layout: post
title: "Calibrating Long-Horizon Agents: Confidence and Uncertainty at Inference Time"
date: 2026-06-27 10:00:00
author: Jiaxin Zhang
description: "An inference-time guide to uncertainty quantification and confidence calibration for long-horizon LLM agents: measuring trajectory uncertainty, acting on it (abstain, ask, reflect, route), and evaluating it — across tool-use, coding, and deep-research agents."
tags: agents uncertainty calibration confidence llm reliability
categories: research-notes
giscus_comments: true
related_posts: false
og_image: https://jxzhangjhu.github.io/assets/img/blog/calibrating-long-horizon-agents/fig1_agentic_reliability_loop.png
---

<div class="lang-switch"><strong>English</strong> · <a href="/blog/2026/calibrating-long-horizon-agents-zh/">中文</a></div>

### Table of Contents

- [Why long-horizon agents need calibrated uncertainty](#why-long-horizon-agents-need-calibrated-uncertainty)
- [Problem setup: a formal vocabulary for agentic uncertainty](#problem-setup-a-formal-vocabulary-for-agentic-uncertainty)
  - [The agent as a partially observed process](#the-agent-as-a-partially-observed-process)
  - [Aleatoric, epistemic, and why they entangle](#aleatoric-epistemic-and-why-they-entangle)
  - [Turn-level vs trajectory-level uncertainty](#turn-level-vs-trajectory-level-uncertainty)
  - [Forward propagation and inverse calibration](#forward-propagation-and-inverse-calibration)
  - [Calibration vs discrimination, defined](#calibration-vs-discrimination-defined)
  - [The shape of a long-horizon agent](#the-shape-of-a-long-horizon-agent)
- [Measure and calibrate trajectory uncertainty](#measure-and-calibrate-trajectory-uncertainty)
  - [What can be a confidence signal?](#what-can-be-a-confidence-signal)
  - [Aggregating step uncertainty into trajectory uncertainty](#aggregating-step-uncertainty-into-trajectory-uncertainty)
  - [Propagation: inheriting uncertainty from the past](#propagation-inheriting-uncertainty-from-the-past)
  - [Holistic trajectory calibration](#holistic-trajectory-calibration)
  - [Tail risk and proper scores](#tail-risk-and-proper-scores)
  - [The right signal depends on the agent type](#the-right-signal-depends-on-the-agent-type)
- [Act on uncertainty](#act-on-uncertainty)
  - [The control menu](#the-control-menu)
  - [Dual-process agentic UQ](#dual-process-agentic-uq)
  - [The economics of reflection](#the-economics-of-reflection)
  - [When confidence gates fail](#when-confidence-gates-fail)
- [Self-improve and self-evolve with uncertainty](#self-improve-and-self-evolve-with-uncertainty)
- [Evaluate agentic calibration](#evaluate-agentic-calibration)
- [Open challenges](#open-challenges)
- [Summary](#summary)
- [How to cite](#how-to-cite)
- [References](#references)

---

For a single language-model answer, "confidence" is a number: how likely is this output to be correct? You can compute it, calibrate it with temperature scaling, and move on.

For a long-horizon agent, confidence is not a number you report — it is a **control variable you act on**. At every step the agent faces a decision that has nothing to do with the next token and everything to do with reliability: *should I commit this action, ask the user a question, call a verifier, reflect, stop, escalate to a human — or remember this moment as a failure to learn from later?* The quality of those decisions, accumulated over dozens of steps, is what separates a demo from a deployable system.

This shift is easy to underestimate because the agent still looks like a model emitting text. But the failure modes are different. An early mistake does not stay local: once it is written into the trajectory, it becomes part of the context the agent reasons from at every later step. A 90%-reliable step, repeated twenty times, is a ~12%-reliable task. And the agent will often be *most* confident exactly when it has quietly gone off the rails, because the local reasoning looks fluent inside a mistaken world.

So the single-turn question — *is the final answer calibrated?* — is the wrong size. For agents, reliability is a **trajectory-level, propagating, and actionable** property.

This post is about the **inference-time** view: the base model is **frozen**. We are not changing weights. We build reliability *around* the model — measuring uncertainty over the trajectory, calibrating it so it is trustworthy, acting on it, and feeding it back into the agent's memory and skills. The companion post takes the **post-training** view: how to bake calibrated confidence *into* the weights.

> **Thesis.** With the model frozen, uncertainty is the control plane for a reliable — and self-improving — long-horizon agent: measure it over the trajectory, calibrate it, act on it, and let it drive self-evolution.

![A loop diagram: measure trajectory uncertainty, calibrate confidence, act on uncertainty, write the signal into memory, and self-evolve the agent system.](/assets/img/blog/calibrating-long-horizon-agents/fig1_agentic_reliability_loop.png)
*Figure 1. The agentic reliability loop. The same loop recurs in tool-use, coding, computer-use, deep-research, and reasoning agents; what changes between them is the signal that is actually trustworthy.*

> **Three running examples.** We carry these through the whole post to keep the abstractions concrete:
> - **E1 — Deep research (in the style of [GAIA](https://arxiv.org/abs/2311.12983) / enterprise deep research).** An agent must answer an open-ended research question by searching, reading, and synthesizing a report over many steps. *State:* external evidence (web/docs). *Verifier:* report-quality rubric / reference answer — often non-verifiable.
> - **E2 — Retail tool agent (in the style of [τ²-bench](https://arxiv.org/abs/2506.07982)).** A customer wants to return an item; the agent follows a written policy, calls typed tools over an order database, and converses with a simulated user. *State:* SQL database. *Verifier:* database end-state + policy/communication checks — verifiable.
> - **E3 — Coding / SWE agent (in the style of [SWE-bench](https://arxiv.org/abs/2310.06770)).** The agent localizes a bug across a repo, edits files, and runs tests. *State:* repository + filesystem. *Verifier:* fail-to-pass tests — verifiable, but only where tests exist.

---

## Why long-horizon agents need calibrated uncertainty

Benchmark accuracy is not agent reliability. A benchmark asks one question — did the final answer match a label? A deployed agent has to answer harder ones: Will it behave consistently across reruns? Degrade gracefully when a tool fails or evidence is stale? Notice that a task is underspecified? Pause before an irreversible action? Surface a warning *mid-trajectory* instead of wasting forty more tool calls? Will its confidence predict trajectory success rather than token fluency?

Reliability, in other words, is better described as a **profile** than a scalar. A useful one has four parts — **consistency**, **robustness**, **predictability**, and **safety** ([Rabanser et al., 2026](https://arxiv.org/abs/2602.16666)). Confidence and uncertainty mostly attack the last two: can the system predict its own failure, and can it route risky cases to safe fallback behavior?

![A reliability profile for long-horizon agents: consistency, robustness, predictability, and safety.](/assets/img/blog/calibrating-long-horizon-agents/fig2_reliability_profile.png)
*Figure 2. A reliability profile. Calibration is not all of reliability, but it is the lever that turns uncertainty into predictable, risk-aware behavior — Predictability and Safety.*

### Errors compound over a horizon

Start with the simplest possible model. Suppose each step succeeds independently with probability $$p$$, and the task needs $$T$$ correct steps with no self-correction. Then

$$P(\text{task success}) \approx p^{T},$$

and the horizon length the agent can sustain at a target success rate $$s$$ is

$$H_s(p) \;\approx\; \frac{\ln s}{\ln p}.$$

This is the point of *The Illusion of Diminishing Returns* ([Sinha et al., 2025](https://arxiv.org/abs/2509.09677)): because $$H_s$$ grows hyperbolically as $$p \to 1$$, a *small* gain in per-step reliability buys a *large* gain in reliable horizon length once you are past ~80%. The flip side is the warning for this post: a confident final step says little about a long trajectory, because $$P(\text{success})$$ was set by the weakest steps, not the last one.

![A curve showing reliable horizon length rising sharply as per-step accuracy approaches 100%.](/assets/img/blog/calibrating-long-horizon-agents/fig3_horizon_length.png)
*Figure 3. Small per-step reliability gains produce large horizon gains — which is exactly why trajectory-level calibration has leverage, and why a final-answer score is the wrong thing to trust.*

### History becomes future context

The independence assumption above is charitable. Real agents **self-condition** on their own history. If the agent writes a wrong intermediate result into memory, every later step is conditioned on that wrong result, so the per-step error rate *rises* as the trajectory grows — the opposite of a human, who tends to improve with practice ([Sinha et al., 2025](https://arxiv.org/abs/2509.09677)).

AUQ names this the **Spiral of Hallucination**: an early epistemic error becomes part of the effective environment for later steps ([Zhang et al., 2026a](https://arxiv.org/abs/2601.15703)). *History-Echoes* gives a complementary, mechanistic view — prior hallucinations geometrically trap later generations by biasing the hidden-state trajectory ([Simhi et al., 2026](https://arxiv.org/abs/2603.03308)). It is the same failure across substrates:

- *E1:* the agent misreads one source early and builds the whole report around it.
- *E2:* it assumes the wrong order ID, after which every "correct" tool call operates on the wrong object.
- *E3:* it misdiagnoses the failing test and edits the wrong module.

> **Insight — the unit of reliability is the trajectory.** A single overconfident *intermediate* step can be more damaging than an overconfident final sentence, because the intermediate step gets written into the context that everything downstream depends on.

### Multi-source uncertainty

In a single-turn model, uncertainty is about the model's own output distribution. In an agent, the *environment* is a second source. A tool can fail (auth, rate limit, timeout), return noisy or stale data, or be silently wrong. The agent's next-token uncertainty over its own text is simply not the same object as the reliability of an external tool result — and conflating them is a common bug. MESA-S makes this split explicit, separating **self-confidence** (parametric certainty) from **source-confidence** (trust in retrieved/external evidence) ([Unlu, 2026](https://arxiv.org/abs/2604.16753)).

### Single-turn UQ does not transfer cleanly

The LLM uncertainty toolbox — token probabilities, semantic entropy, self-consistency, verbalized confidence, hidden-state probes, conformal wrappers — is a good *foundation*, but agentic settings stress it in four concrete ways:

1. **No logprobs.** Many frontier models expose no token probabilities, ruling out probability-based methods for the strongest agents.
2. **Length dilution.** A low-probability token in a critical tool argument is averaged away by hundreds of fluent reasoning tokens.
3. **Sampling cost.** Sampling 10 answers to one question is cheap; sampling 10 complete tool trajectories is often prohibitive.
4. **Heterogeneous observations.** Inputs arrive from users, APIs, web pages, filesystems, and other agents, each with its own distribution.

Two position papers now make this case directly: Kirchhof et al. argue the aleatoric/epistemic dichotomy must be reassessed for interactive agents ([Kirchhof et al., 2025](https://arxiv.org/abs/2505.22655)), and Oh et al. give a general formulation of agent UQ and flag the lack of fine-grained benchmarks ([Oh et al., 2026](https://arxiv.org/abs/2602.05073)). A companion survey frames the broader arc this post builds on: uncertainty moving from a **passive metric** to an **active control signal** ([Zhang et al., 2026c](https://arxiv.org/abs/2601.15690)).

### Confidence is not action

Finally, even a calibrated number is useless if the agent ignores it — and the raw number is often not calibrated to begin with. *Agentic Overconfidence* elicits an agent's self-estimated success probability before, during, and after execution and finds a large overconfidence gap: agents stay confident on tasks they go on to fail ([Kaddour et al., 2026](https://arxiv.org/abs/2602.06948)). *RiskEval* shows the second half of the problem: even when models verbalize uncertainty, their decisions are often *not faithful to it* — they fail to abstain when abstention is utility-optimal ([Wang et al., 2026a](https://arxiv.org/abs/2601.07767)), a point echoed by work showing confidences do not align with actions ([Pal et al., 2025](https://arxiv.org/abs/2511.13240)).

That gives us the spine of the post: **measure → calibrate → act → feed back**.

**Takeaway.** Long-horizon agents need calibrated uncertainty because errors compound, history traps later reasoning, the environment is a second source of uncertainty, single-turn confidence breaks under interaction, and a score only matters when it changes what the agent does.

---

## Problem setup: a formal vocabulary for agentic uncertainty

Before methods, we need definitions precise enough to compare them. This section is the one place we lean on notation; the rest of the post stays in plain language.

### The agent as a partially observed process

Model a single agent as a partially observable Markov decision process (POMDP) with state space $$\mathcal{S}$$, actions $$\mathcal{A}$$, observations $$\mathcal{O}$$, transition kernel $$\mathcal{T}(s_{t+1}\mid s_t,a_t)$$, and a reward/verifier $$R$$. The agent never sees $$s_t$$ directly; it acts on the **history**

$$h_t = (o_0, a_0, o_1, a_1, \dots, o_t),$$

through a history-conditioned policy $$\pi(a_t \mid h_t)$$. Because it cannot observe the true state, it maintains an implicit **belief** $$b_t(s_t) = P(s_t \mid h_t)$$.

This gives a crisp definition of what "reliability failure" even means: it is when the belief $$b_t$$ diverges from the true state $$s_t$$ while the policy keeps acting as if it had not ([Zhang et al., 2026a](https://arxiv.org/abs/2601.15703)). The job of agentic uncertainty quantification is to estimate how much to trust $$b_t$$, and how that trust propagates over time.

One generalization matters in practice. In customer-service settings (E2) the **user is also an actor** with their own tools and policy, so the right model is a dual-control decentralized POMDP (Dec-POMDP), which [τ²-bench](https://arxiv.org/abs/2506.07982) formalizes ([Barres et al., 2025](https://arxiv.org/abs/2506.07982)). Dual control is strictly harder: now the agent is uncertain about the *user's* state and actions too.

### Aleatoric, epistemic, and why they entangle

Classical UQ splits uncertainty into two kinds ([Kendall & Gal, 2017](https://arxiv.org/abs/1703.04977)):

- **Aleatoric** — irreducible randomness in the environment (a flaky tool, a stochastic user). More data does not remove it.
- **Epistemic** — the agent's own ignorance (a knowledge gap, a reasoning error). Reducible with better information or reasoning.

In a one-shot prediction these are stable categories. In an agent they **entangle over time**. Suppose E2's agent is unsure which item the user means — that is epistemic. If it guesses and writes the guess into $$h_t$$, then for every later step that guess is part of the "given" context: the agent's own epistemic error has been promoted into an effective *aleatoric* constraint on the rest of the trajectory. This temporal coupling is exactly why a per-step scalar is not enough, and it is the formal seed of the Spiral of Hallucination.

This is also why several authors argue the aleatoric/epistemic split is the wrong design vocabulary for agents. A more actionable taxonomy asks *what the agent should do next* ([Ojewale et al., 2026](https://arxiv.org/abs/2606.02965)):

- **Specification gap** — required information about intent is missing. → *ask.*
- **Verification gap** — a precondition or outcome cannot be confirmed. → *verify.*
- **Authority gap** — the next action is a high-impact/irreversible commitment without authorization. → *pause / escalate.*
- (and the implicit fourth, **tool-failure uncertainty** — the tool returned something wrong or empty).

A second axis is **distribution shift**. A confidence model calibrated on one task mix can break on another: domain shift (QA → coding), tool shift (a search API gets noisier), temporal shift (evidence goes stale), or agent-architecture shift (the same base model under a different planner/memory). We will see this again when we ask whether a calibrator *transfers* (it sometimes does, [Zhang et al., 2026b](https://arxiv.org/abs/2601.15778)) and when post-training calibration *forgets* under shift (the companion post).

### Turn-level vs trajectory-level uncertainty

Let $$F_t$$ denote the agent's decision at turn $$t$$ (action, or action + observation). Following the autoregressive chain rule for sequence uncertainty ([Malinin & Gales, 2021](https://arxiv.org/abs/2002.07650)) and its agentic generalization ([Oh et al., 2026](https://arxiv.org/abs/2602.05073)), trajectory uncertainty decomposes as a sum of **turn-level** terms:

$$U(F_{1:T}) \;=\; \sum_{t=1}^{T} U\!\left(F_t \mid F_{<t}\right),$$

where each conditional term mixes the agent's *intrinsic* uncertainty at step $$t$$ with uncertainty *inherited* from the history. This is the cleanest way to see why single-turn UQ is a special case ($$T=1$$) and why agents need something more: the cross-term that single-turn methods cannot express.

### Forward propagation and inverse calibration

AUQ frames the agent-UQ problem as two coupled problems ([Zhang et al., 2026a](https://arxiv.org/abs/2601.15703)).

**Forward problem (propagation).** Let $$V_t \in \{0,1\}$$ indicate that the trajectory up to step $$t$$ is still valid (free of a critical error). By the chain rule,

$$P(V_t = 1 \mid h_t) \;=\; \underbrace{P(\text{correct}(a_t)\mid h_t)}_{\text{local confidence } c_t} \;\cdot\; \underbrace{P(V_{t-1}=1 \mid h_{t-1})}_{\text{historical validity}}.$$

Two things fall out immediately. First, $$P(V_t)$$ is **monotonically non-increasing** in $$t$$: a single near-zero step ($$c_k \approx 0$$) drives the whole product to zero for all $$t > k$$ — the Spiral of Hallucination in one line. Second, it suggests a simple trajectory belief, $$P(V_t \mid h_t) \approx \prod_{i\le t} c_i$$, or a conservative $$\min_{i\le t} c_i$$ (the weakest-link reading we return to under aggregation).

> **Worked example (E2).** The retail agent runs six steps to process a return, with local confidences $$c = (0.97,\, 0.62,\, 0.95,\, 0.96,\, 0.98,\, 0.99)$$ — one shaky step (step 2, "identify which order the customer means") hidden among confident ones. The naive **average** is $$0.91$$, which *looks* safe. But the forward recursion says the trajectory belief is $$\prod_i c_i \approx 0.53$$, and the **weakest link** is $$\min_i c_i = 0.62$$ — both flag the run as close to a coin flip. Step 2 caps everything after it: once the agent acts on the wrong order, no amount of downstream confidence can raise $$P(V_T)$$. This single example is why later we aggregate with a product/min, calibrate the result, and gate reflection on it.

**Inverse problem (calibration).** When the forward estimate drops below a threshold, $$P(V_t\mid h_t) < \delta$$, the agent should not just *report* low confidence — it should *act* to recover. Treating success as an optimality variable $$\mathcal{O}$$, the corrected action solves a posterior optimization,

$$a^\star \;=\; \arg\max_{a}\; \int P(a \mid z, h_t)\,P(z \mid \mathcal{O}{=}1, h_t)\,dz,$$

over a latent reasoning path $$z$$ — which, in practice, is approximated by inference-time compute (reflection, best-of-$$N$$, retrieval expansion). This is the formal bridge from *measuring* uncertainty to *acting* on it.

### Calibration vs discrimination, defined

Two different things are often both called "calibration."

**Calibration** asks whether a confidence of 0.8 corresponds to ~80% empirical success. The standard summary is Expected Calibration Error over $$M$$ bins,

$$\text{ECE} \;=\; \sum_{m=1}^{M} \frac{|B_m|}{N}\,\big|\,\text{acc}(B_m) - \text{conf}(B_m)\,\big|.$$

**Discrimination** asks whether the score *ranks* successes above failures, summarized by AUROC (threshold-free). The **Brier score**, $$\frac{1}{N}\sum_i (C_i - y_i)^2$$, is a proper scoring rule that captures both at once.

The distinction matters because the two can diverge: a model can be calibrated on average yet unable to separate a successful trajectory from a failed one (useless for routing), or sharply discriminative yet numerically miscalibrated (useless for risk budgets). For agents, all of these must be computed at the **trajectory level**, replacing the per-answer outcome with trajectory success $$y \in \{0,1\}$$ and the per-answer confidence with an aggregated trajectory belief $$C(\tau) = \Phi(c_{1:T})$$.

### The shape of a long-horizon agent

A last piece of vocabulary: what we mean by "agent." The first wave was the ReAct loop — reason, act, observe, repeat ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)) — later extended with deliberate search such as Tree of Thoughts ([Yao et al., 2023](https://arxiv.org/abs/2305.10601)) and verbal self-reflection ([Shinn et al., 2023](https://arxiv.org/abs/2303.11366)). But the modern landscape is broader, and the *useful uncertainty signal differs by type*:

| Agent type | Trustworthy signal | Typical action | Self-improvement artifact |
|---|---|---|---|
| Coding / SWE (E3) | tests, symbolic clusters, patch validators | validate, abstain, ask | reusable fix / rubric |
| Tool-use (E2) | verbalized confidence (tool-type dependent), missing params | ask, call, skip, verify | tool-schema memory |
| Reasoning / long-CoT | self-certainty, local confidence, entropy | early-stop, filter, vote | better search policy |
| Computer-use / web | spatial dispersion of click samples | execute, abstain, cascade | UI memory / safer policy |
| Deep research (E1) | calibrated stopping confidence, evidence coverage | continue, stop, verify | research memory / source heuristics |
| Self-evolving | failure / instability / low confidence | write memory, add skill, explore | skill library / agentic memory |

*Table 1. The control loop is shared across agent types; the reliable signal is not. Raw token logits are useful for some types and nearly useless for others (code, GUI clicks).*

> **Insight — the substrate determines the signal.** A general theory of agent calibration has to respect the agent's action space. "Just read the logprobs" is good advice for a reasoning agent and bad advice for a browser-clicking one.

**Takeaway.** Agentic uncertainty is uncertainty over intent, state, tools, action consequences, and trajectory validity — decomposable turn-by-turn, non-increasing in validity, and only meaningful once you fix the granularity (turn vs trajectory) and the metric (calibration vs discrimination).

---

## Measure and calibrate trajectory uncertainty

With the setup in hand, the first half of the loop is: produce a number $$C(\tau)$$ that actually tracks trajectory success. This breaks into *what to read* (signal), *how to combine it over time* (aggregation/propagation), and *how to make it trustworthy* (calibration). ACC/HTC is the worked example that ties these together.

### What can be a confidence signal?

| Signal | Black-box API? | Agent-ready? | Main failure mode |
|---|---|---|---|
| Verbalized confidence | yes | sometimes | overconfident; can be ignored by the policy |
| Token logprob / perplexity | no (often hidden) | limited | length bias; critical tokens averaged out |
| Semantic entropy | yes (needs sampling) | costly | trajectory-scale sampling is expensive |
| Self-consistency / self-certainty | yes | costly | agreement can reflect shared bias |
| Hidden-state probes | no | promising | model-specific; needs labeled data |
| Process / trajectory features | partial | strong | needs trajectory logs + labels |

*Table 2. Single-turn signals, and how they survive the jump to agents.*

**Verbalized confidence.** Ask the model to emit a scalar (and, crucially for agents, an *explanation*). Formally it is an elicitation map $$\Phi: h_t \mapsto (a_t, \hat{c}_t, \hat{e}_t)$$ producing an action, a confidence $$\hat{c}_t \in [0,1]$$, and a natural-language reason $$\hat{e}_t$$. It works with black-box frontier APIs, which is why it is attractive in production; models can be taught to express calibrated uncertainty in words ([Lin et al., 2022](https://arxiv.org/abs/2205.14334)), "just ask" strategies are decent for RLHF models ([Tian et al., 2023](https://arxiv.org/abs/2305.14975)), and the reliability of the elicitation depends heavily on *how* you ask ([Yang et al., 2024](https://arxiv.org/abs/2412.14737)). Self-evaluation in the same family — asking for $$P(\text{True})$$ — goes back to [Kadavath et al., 2022](https://arxiv.org/abs/2207.05221). But it is a model output and can be distorted: TASR finds verbalized 1–5 confidence *collapses* on RLHF models and instead uses calibrated logit margins for its stopping rule ([Kieback et al., 2026](https://arxiv.org/abs/2606.13814)).

**Token logprobs.** When available, a sequence confidence is often the length-normalized log-likelihood $$\frac{1}{L}\sum_{j=1}^{L}\log p(y_j\mid y_{<j})$$. Cheap, but frequently misaligned with semantic correctness — most visibly in code, where token confidence poorly predicts correctness and *symbolic* equivalence classes work far better ([Sharma & David, 2025](https://arxiv.org/abs/2502.11620)).

**Semantic entropy.** Sample several answers, cluster them by meaning, and take entropy over the meaning clusters rather than surface strings ([Kuhn et al., 2023](https://arxiv.org/abs/2302.09664)). Elegant and strong for single-turn QA; the cost multiplies when each "sample" is a full multi-step trajectory.

**Self-certainty / self-consistency.** Agreement across sampled reasoning paths is an implicit confidence ([Wang et al., 2022](https://arxiv.org/abs/2203.11171)); self-certainty measures how peaked the output distribution is via its KL divergence from uniform, a reward-model-free quality signal that scales with the number of samples ([Kang et al., 2025](https://arxiv.org/abs/2502.18581)).

The upshot: an agent needs a signal that is both *meaningful* and *affordable* under its interface and cost constraints — and, ideally, that carries enough semantics ($$\hat{e}_t$$, not just $$\hat{c}_t$$) to tell the controller *what* to do about it.

### Aggregating step uncertainty into trajectory uncertainty

Given step confidences $$c_1, \dots, c_T$$, how do we get a trajectory belief $$C(\tau)$$? The common operators:

- **Last-step**, $$C=c_T$$ — easy to inspect, blind to early failures.
- **Average**, $$C=\frac{1}{T}\sum_t c_t$$ — smooth, but smoothness is a bug here.
- **Weakest-link / min**, $$C=\min_t c_t$$ — conservative; matches the validity recursion.
- **Product**, $$C=\prod_t c_t$$ — the literal $$P(V_T)$$ under step independence.
- **Learned**, $$C=f(\phi(\tau))$$ — features of the whole trajectory (next subsection).

> **Insight — averaging hides decisive failures.** Agent failures are *sparse but decisive*. If one tool call silently poisoned the trajectory, nine confident later steps should not be allowed to wash it out. This is why the validity recursion uses a product/min, not a mean.

*E2:* if the agent's confidence cratered at the "look up the order" step, the trajectory belief should crater too, even if the refund-issuing step looks confident.

### Propagation: inheriting uncertainty from the past

Aggregation treats steps symmetrically; **propagation** models how uncertainty is *inherited*. SAUP propagates per-step uncertainty with **situational weights** that up-weight pivotal steps, improving failure ranking over final-step-only baselines ([Zhao et al., 2024](https://arxiv.org/abs/2412.01033)). UProp gives the information-theoretic version, decomposing a step's uncertainty into an **intrinsic** term and an **extrinsic** term inherited from prior steps,

$$U(d_t) \;=\; \underbrace{H(d_t \mid h_t)}_{\text{intrinsic}} \;+\; \underbrace{I(d_t; d_{<t})}_{\text{extrinsic (inherited)}},$$

and estimates the mutual-information term over trajectory-dependent decision processes ([Duan et al., 2025](https://arxiv.org/abs/2506.17419)). This is the formal heart of "distrust this step *because* of earlier steps."

### Holistic trajectory calibration

Rather than hand-pick one aggregation rule, ACC asks a supervised question: **given the whole trajectory, predict whether it will succeed** ([Zhang et al., 2026b](https://arxiv.org/abs/2601.15778)). It names the problem **Agentic Confidence Calibration** and solves it with **Holistic Trajectory Calibration (HTC)**: turn the raw confidence trace into a compact set of process features in four families,

1. **Dynamics** — how confidence evolves across steps (trend, reversals, gradients).
2. **Stability** — within-step volatility of the token/confidence distribution.
3. **Position** — early and late indicators (first/last-step signals).
4. **Structure** — step counts, token-length patterns (proxies for complexity).

These map through a deliberately simple, interpretable calibrator $$\mathcal{C}_\tau = \sigma(\mathbf{w}^\top \phi(\tau) + b)$$ with L2 (full) or L1 (sparse) regularization. The simplicity is a feature, not a limitation: agent trajectory datasets are small and expensive, so a low-capacity model overfits less and — because the weights are inspectable — tells you *which* signals predict failure.

![Holistic Trajectory Calibration overview: per-step token-confidence sequences become trajectory-level features (dynamics, stability, position, structure) feeding a lightweight interpretable calibrator.](/assets/img/blog/calibrating-long-horizon-agents/acc_fig1_htc.png)
*Figure 4. Holistic Trajectory Calibration (HTC): the whole trajectory's confidence trace becomes process-level features for an interpretable calibrator; a pretrained General Agent Calibrator transfers to held-out tasks. (Image source: [Zhang et al., 2026b](https://arxiv.org/abs/2601.15778), Figure 1.)*

**Does seeing the process actually help? Yes — most where it matters.** The advantage over last-step confidence widens on the hardest tasks, exactly where overconfidence is most dangerous. On Humanity's Last Exam (HLE), raw verbalized confidence is essentially noise (ECE 0.656), and even last-step token-probability with temperature scaling only reaches ECE 0.436 — while HTC-Reduced reaches **ECE 0.031** (a ~14× reduction over the tuned last-step baseline) at an equal-or-better Brier score.

| Calibration error (ECE ↓) | SimpleQA | GPQA | HLE |
|---|---|---|---|
| Verbalized confidence | 0.121 | 0.454 | 0.656 |
| Last-step token-prob + temperature | 0.071 | 0.139 | 0.436 |
| **HTC-Reduced (ACC)** | **0.068** | **0.102** | **0.031** |

*Table 3. A slice of ACC's main results (ECE, lower is better): process-level features dominate last-step confidence, and the gap grows on harder tasks. (Source: [Zhang et al., 2026b](https://arxiv.org/abs/2601.15778), Table 1.)*

Three findings generalize beyond the specific feature set:

- **Interpretability.** The most predictive signal is task-dependent — positional features dominate long, hard reasoning chains; multi-step QA leans on a more even mix of dynamics/stability/position — but a hierarchy recurs: *start/end first, then process stability.* Because the calibrator is linear, you can read this straight off the weights.
- **Transferability.** A calibrator trained on one task transfers to related ones when the *output format* matches — trained on SimpleQA, it transfers to HotpotQA well enough to *beat* an in-domain calibrator — but degrades across genuinely different paradigms (e.g., multiple-choice → open-ended).
- **Generalization.** A pretrained **General Agent Calibrator (GAC)** gives the best *zero-shot* calibration on out-of-domain GAIA (ECE 0.118), beating five learning-based baselines (LSTM, Transformer, …) with far lower variance in the small-data regime — a portable "uncertainty grammar."

The durable claim: **calibration should see the process**, not just the last token — and a small, legible model is enough to read it.

### Tail risk and proper scores

Two refinements close the loop. TRACER argues that since failures are sparse and decisive, you should aggregate with a **tail-risk** functional (CVaR / max-composite over situational-awareness signals like looping or coherence gaps) rather than an average ([Tayebati et al., 2026](https://arxiv.org/abs/2602.11409)). And TPS points out that scalar trajectory-ECE is *resolution-blind*, proposing a strictly **proper** trajectory score that elicits the full prefix-conditioned success-probability trace ([Raghu et al., 2026](https://arxiv.org/abs/2605.24756)). The practical stance: use trajectory-ECE/Brier/AUROC for comparison and reliability diagrams for inspection, but do not claim a single scalar "solves" trajectory calibration.

### The right signal depends on the agent type

Finally, instantiate Table 1. For **coding** agents, logits are weak; use tests, symbolic clusters, or patch validators, and prefer *validate-or-abstain* over editing blindly ([Sharma & David, 2025](https://arxiv.org/abs/2502.11620); [Cambronero et al., 2025](https://arxiv.org/abs/2510.03217)). For **tool-use** agents, the same verbalized confidence means different things depending on the tool: *evidence* tools (web search) inject noise and breed overconfidence, while *verification* tools (a code interpreter) ground the trajectory ([Xuan et al., 2026](https://arxiv.org/abs/2601.07264)). For **computer-use** agents, the trustworthy signal is *spatial* — the dispersion of sampled GUI grounding points — not token confidence ([Wang et al., 2026b](https://arxiv.org/abs/2602.02419)). For **reasoning** agents, local confidence drives test-time scaling ([Fu et al., 2025](https://arxiv.org/abs/2508.15260)). For **deep-research** agents (E1), the decisive quantity is a calibrated *stopping* confidence — is the evidence enough to answer? ([Kieback et al., 2026](https://arxiv.org/abs/2606.13814)).

**Takeaway.** Trajectory-level, process-aware signals beat last-step confidence; ACC/HTC gives a general, interpretable way to calibrate the whole trajectory; and the *local* signal that feeds it must be chosen for the agent's substrate.

---

## Act on uncertainty

A calibrated number is inert until it changes behavior. The agent's uncertainty policy is much richer than answer/refuse.

### The control menu

| Action | Trigger | Cost | Risk if wrong |
|---|---|---|---|
| Continue / commit | high confidence, low risk | low | overconfident failure |
| Ask / clarify | specification gap | user friction | needless interruption |
| Verify / gather | verification gap, shaky tool result | tool cost | verification loop |
| Reflect / self-correct | uncertain reasoning or plan | model calls | delusional confirmation |
| Allocate compute | promising but unresolved | latency / tokens | wasted compute |
| Abstain / defer | low confidence or high risk | lower coverage | over-refusal |
| Escalate | beyond model/tool budget; authority gap | human / strong-model cost | bottleneck |

*Table 4. Uncertainty as a typed router. The agentic move is to pick among these by the kind of uncertainty, not a single scalar threshold.*

**Abstain.** AbstentionBench shows abstention is unsolved even for frontier models, and — a warning for the companion post — that reasoning fine-tuning can *degrade* it ([Kirichenko et al., 2025](https://arxiv.org/abs/2506.09038)); Abstain-R1 treats abstention and post-refusal clarification as first-class behaviors ([Zhai et al., 2026](https://arxiv.org/abs/2604.17073)).

**Ask.** When uncertainty is a specification gap, the right move is to ask, not to silently reflect. UoT chooses *what* to ask by expected information gain — roughly $$q^\star = \arg\max_q\, \big[H(\text{answer}) - \mathbb{E}_{r\sim q}H(\text{answer}\mid r)\big]$$ ([Hu et al., 2024](https://arxiv.org/abs/2402.03271)); SAGE-Agent moves clarification into tool-parameter space and uses expected value of perfect information to decide when to stop asking ([Suri et al., 2025](https://arxiv.org/abs/2511.08798)); in coding (E3), Ambig-SWE shows targeted clarifying questions recover a large fraction of underspecified tasks ([Vijayvargiya et al., 2025](https://arxiv.org/abs/2502.13069)).

**Reflect.** Reflexion ([Shinn et al., 2023](https://arxiv.org/abs/2303.11366)) and related self-refinement methods improve agents via verbal feedback — but blind reflection is inefficient or harmful. The question is *when* to reflect, which is exactly where calibrated uncertainty earns its keep.

**Allocate compute / stop.** DeepConf spends inference where local confidence is high and early-stops where it is not ([Fu et al., 2025](https://arxiv.org/abs/2508.15260)); TASR's calibrated stopping rule governs iterative retrieval without any training ([Kieback et al., 2026](https://arxiv.org/abs/2606.13814)).

**Route / escalate.** AutoMix routes uncertain queries to a stronger model via a self-verification meta-controller ([Aggarwal et al., 2024](https://arxiv.org/abs/2310.12963)); KnowNo gives the robotics version — conformal *ask-for-help* with a statistical task-completion guarantee, acting only when the prediction set is a singleton ([Ren et al., 2023](https://arxiv.org/abs/2307.01928)).

### Dual-process agentic UQ

ACC tells you *how reliable* a trajectory is; AUQ turns that signal into *action* — the cleanest instantiation of "uncertainty as a switch," and notably **training-free** ([Zhang et al., 2026a](https://arxiv.org/abs/2601.15703)). It implements the forward/inverse decomposition from the problem setup with a dual-process controller:

- **System 1 — Uncertainty-Aware Memory (UAM).** At each step the agent emits $$(a_t, \hat{c}_t, \hat{e}_t)$$ and writes the confidence and explanation into memory, $$\mathcal{M}_t = \{(o_i, a_i, \hat{c}_i, \hat{e}_i)\}_{i<t}$$. Keeping prior *doubts* in context acts as a soft damper on overconfidence — the forward propagation of uncertainty.
- **System 2 — Uncertainty-Aware Reflection (UAR).** A switching function $$S(h_t)=\mathbb{I}[\hat{c}_t < \tau]$$ triggers reflection only when confidence falls below $$\tau$$. Reflection is *guided* by the explanation $$\hat{e}_t$$ (the inverse problem), and the corrected action is chosen by a consistency-weighted vote over $$N$$ samples,

$$S_{\text{cons}}(a) \;=\; \frac{1}{N}\sum_{k=1}^{N} \hat{c}^{(k)} \cdot \mathbb{I}\!\left[a^{(k)} \equiv a\right],$$

with an adaptive memory-expansion fallback when local reflection fails.

![Dual-process AUQ: System 1 uncertainty-aware memory propagates verbalized confidence forward; System 2 uncertainty-aware reflection triggers when confidence falls below threshold.](/assets/img/blog/calibrating-long-horizon-agents/auq_fig1_dual_process.png)
*Figure 5. AUQ turns uncertainty into a switch: fast System-1 execution when confident, targeted System-2 reflection when confidence signals a likely failure point. (Image source: [Zhang et al., 2026a](https://arxiv.org/abs/2601.15703), Figure 1.)*

**The two halves do different jobs — and the split shows up cleanly in ablation.** UAM-only gives the best *calibration* (lowest trajectory-ECE: it aligns confidence with reality), while UAR-only gives the best *resolution* (lowest Brier: it actively resolves gaps and polarizes belief toward the truth). AUQ keeps both, and the combination is what pays off downstream:

- On closed-loop tasks it reaches **74.3% success on ALFWorld (+10.7%)** and **42.5% on WebShop (+13.6%)** over a ReAct System-1 baseline — and beats self-consistency (CoT-SC), so the win is the *targeted* reflection, not just extra samples.
- On open-ended **DeepResearch Bench** it scores **52.09 overall**, above the strongest deployed closed agent (49.71) and the best open-source framework (50.62).
- Its confidence is also more *discriminative* (higher AUROC), which is why it spends the System-2 budget on the trajectories that need it instead of reflecting blindly.

The difference from blind reflection is the *trigger* and the *cue*. AUQ does not reflect every step (expensive, and prone to "double-checking fatigue"); it reflects when its own uncertainty says a step is risky, and the verbal explanation tells System 2 *what* to fix. Concretely, in E1 a planner might emit:

> Confidence: 0.58. Concern: I haven't separated Japan's elderly *population* forecast from *spending per capita*; the current plan may conflate demographics with market size.

That is far more actionable than a bare `0.58`: it names the query to run and the decomposition to revise.

### The economics of reflection

Reflection costs extra inference — but failed trajectories are expensive too. A naive agent that makes a wrong assumption at step 5 can burn 45 more steps chasing it. The right accounting is not raw token cost but **effective cost per success**,

$$\text{Cost}_{\text{eff}} \;=\; \frac{\text{total cost of all attempts}}{\text{number of successful tasks}} \;=\; \frac{\text{avg cost per trajectory}}{\text{success rate}}.$$

Under this metric, spending more per trajectory can *lower* cost per success, because it converts long futile failures into solved tasks. AUQ quantifies both sides of the ledger: its targeted reflection corrects **14.3%** of a ReAct baseline's failed trajectories with far more repairs than regressions (net-positive), yet the returns are sharply diminishing — past a point, accuracy plateaus while cost grows roughly exponentially, which is exactly why the trigger threshold $$\tau$$ has to be *chosen*, not maximized.

![Real AUQ results: internal belief dynamics of UAM-only vs AUQ across trajectory steps, and a Pareto frontier of success rate versus computational cost.](/assets/img/blog/calibrating-long-horizon-agents/slide_auq_dynamics_pareto.png)
*Figure 6. The economics, with real numbers. Left: across a long trajectory, AUQ keeps the confidence of eventual successes and failures separated, where UAM-only drifts and tangles. Right: the success-rate-vs-compute Pareto frontier — gated reflection buys accuracy up to a point, then "over-verifies" with diminishing returns. (Image source: author's slides; AUQ, [Zhang et al., 2026a](https://arxiv.org/abs/2601.15703).)*

### When confidence gates fail

Three honest caveats keep this from being a free lunch.

- **Delusional confirmation.** Reflection can *raise* confidence in a wrong plan when the model fabricates a plausible justification; AUQ documents this failure mode, which is why the post-reflection confidence must not be trusted blindly and the trigger should be conservative (net repair $$\gg$$ regression).
- **Unstable gates.** *Same Signal, Opposite Meaning* shows the same confidence/difficulty signal can predict that extra compute *helps* in one setting and *hurts* in another — compute *need* is not compute *suitability* ([Li et al., 2026](https://arxiv.org/abs/2605.06908)).
- **Confidence ≠ utility.** RiskEval again: a controller that does not weigh the cost of errors will act wrongly even with a correct risk estimate ([Wang et al., 2026a](https://arxiv.org/abs/2601.07767)).

> **Insight — more reflection is not better.** The goal is not to maximize deliberation; it is to fire the *right* intervention at the *right* step. An uncertainty controller is only as good as the calibration of the signal it gates on.

**Takeaway.** Uncertainty becomes valuable only as an action — ask, verify, reflect, allocate compute, abstain, or route — selected by the *kind* of uncertainty, and the policy must be conservative because the gate itself can be wrong.

---

## Self-improve and self-evolve with uncertainty

The loop has one more arc. Beyond improving the *current* decision, uncertainty can make the agent *system* better over time — without touching weights. This is the application-layer payoff: non-parametric self-evolution through memory, skills, and tools.

**Uncertainty-aware memory.** AUQ's UAM is the simplest instance: store metacognitive state $$(\hat{c}_i, \hat{e}_i)$$, not just observations and actions. The explanation matters because it gives uncertainty semantics — "evidence conflicts" vs "the user didn't specify a date" lead to different future behavior. Oblivion extends this into self-adaptive memory, where uncertainty gates *when* to consult memory and how memories decay or are reinforced ([Rana et al., 2026](https://arxiv.org/abs/2604.00131)). And Confidence Laundering is the systems-level warning: if uncertainty is not carried across component handoffs, downstream modules treat shaky artifacts as clean facts ([Shi et al., 2026](https://arxiv.org/abs/2606.20662)). The design principle:

> uncertainty should be a persistent metadata field in the agent system, not a transient string in one prompt.

**Uncertainty-driven exploration.** Where should a frozen agent spend its next action? Search more when the expected information gain is high and uncertainty is reducible; stop when more search will not change the answer; abstain or escalate when uncertainty is irreducible under the available tools. UoT is the inference-time version of "what to ask" ([Hu et al., 2024](https://arxiv.org/abs/2402.03271)); TASR is the version of "when to stop" for deep research (E1) ([Kieback et al., 2026](https://arxiv.org/abs/2606.13814)).

**Skills and tools from failure.** Voyager is the canonical non-parametric self-evolving agent: an automatic curriculum plus an ever-growing, self-verified skill library, all on a frozen GPT-4 ([Wang et al., 2023](https://arxiv.org/abs/2305.16291)). ExpeL distills cross-task experience into reusable natural-language insights ([Zhao et al., 2023](https://arxiv.org/abs/2308.10144)); A-MEM builds an evolving agentic memory ([Xu et al., 2025](https://arxiv.org/abs/2502.12110)). The connection to uncertainty is direct: low confidence marks where to gather evidence, repeated failure marks where to write a new skill, and a verified recovery is exactly what to store.

This is also the bridge to the companion post. Once a system repeatedly learns *which* uncertainties matter, the natural next question is whether to internalize those behaviors into the weights — which is where post-training comes in.

**Takeaway.** A frozen model can still become a more reliable agent over time, if uncertainty drives what it remembers, asks, verifies, and turns into reusable skills.

---

## Evaluate agentic calibration

Evaluation is the weakest link in the field, and worth a clear-eyed section.

**Metrics.** Use the trajectory-level versions of ECE, Brier, and AUROC defined in the problem setup, plus risk–coverage curves for selective execution. The trajectory belief $$C(\tau)$$ can be $$c_T$$, $$\text{mean}_t c_t$$, $$\min_t c_t$$, or a learned $$f(\phi(\tau))$$ — and which one you pick changes the numbers, so report it. A reliability diagram is the most informative single plot.

![A trajectory reliability diagram comparing raw overconfident agent confidence against a calibrated curve near the diagonal.](/assets/img/blog/calibrating-long-horizon-agents/fig7_trajectory_reliability.png)
*Figure 7. A trajectory reliability diagram asks whether trajectory confidence matches trajectory success. Raw agent confidence sits well below the diagonal (overconfident); calibration pulls it back.*

**Benchmarks.** There is, as of writing, no purpose-built agentic-UQ benchmark; people repurpose agent benchmarks that lack uncertainty labels.

| Benchmark | What it provides | UQ limitation |
|---|---|---|
| [τ²-bench](https://arxiv.org/abs/2506.07982) | dual-control tool-use tasks | no uncertainty labels (the default substrate) |
| [GAIA](https://arxiv.org/abs/2311.12983) | deep-research / general-assistant tasks | trajectory-level success only |
| [AbstentionBench](https://arxiv.org/abs/2506.09038) | abstention scenarios | mostly single-turn |
| [Ambig-SWE](https://arxiv.org/abs/2502.13069) | underspecified coding tasks | SWE-specific |
| Argus / computer-use UQ ([Kumar et al., 2026](https://arxiv.org/abs/2606.25760)) | GUI grounding uncertainty | domain-specific |

*Table 5. The benchmarks the field calibrates on, and why none of them was built for agentic UQ.*

Oh et al. survey dozens of agent benchmarks and find only a small fraction give turn-level labels ([Oh et al., 2026](https://arxiv.org/abs/2602.05073)). A benchmark built for this should log every observation, action, tool output and failure, the model's confidence and uncertainty explanation, whether each step was reversible, milestone/partial-success labels, final success, and — the hard part — whether the agent *should have* asked, verified, or escalated. It should score not just task success but whether the agent made the right uncertainty-controlled decisions along the way.

**Takeaway.** We can measure final success; we are still learning to measure whether an agent's confidence tracked the trajectory and improved its decisions. The benchmark gap is an opportunity.

---

## Open challenges

The recipe is useful but young; here is where I would point a skeptical eye.

**The signal itself can be unreliable.** Every controller in this post amplifies its input signal. If the confidence is miscalibrated, the controller will reflect on easy cases, skip hard ones, or escalate the wrong tasks — and *Same Signal, Opposite Meaning* shows the signal's meaning is context-dependent ([Li et al., 2026](https://arxiv.org/abs/2605.06908)). Calibrating the *gate* is as important as building it.

**Tool-output uncertainty is under-modeled.** Most methods decide *whether* to call a tool; far fewer decide whether to *trust its output*. Yet agents increasingly depend on tools whose failures are silent — stale pages, partial results, schema drift, auth errors, ambiguous records.

**Multi-agent uncertainty gets laundered.** One agent's uncertain claim becomes another's confident premise. DebUnc studies communicating uncertainty in debate via attention scaling ([Yoffe et al., 2024](https://arxiv.org/abs/2407.06426)); Confidence Laundering generalizes the leak to any component handoff ([Shi et al., 2026](https://arxiv.org/abs/2606.20662)). Reliable multi-agent systems likely need uncertainty metadata that survives message boundaries.

**Reflection has latency and regression risk.** It adds tokens and wall-clock time, and can overturn a correct plan (delusional confirmation). A good controller optimizes net repair, not intervention count — and real-time agents may not be able to afford best-of-$$N$$ at all.

**Metrics and benchmarks are immature.** Trajectory-ECE is resolution-blind ([Raghu et al., 2026](https://arxiv.org/abs/2605.24756)); we need proper trajectory scores, turn-level labels, abstention-competence metrics, and a purpose-built benchmark.

**Orchestration is not internalization.** Everything here builds reliability *around* a frozen model — deployable, and often necessary. But a model that is *natively* calibrated would make all of this cheaper and simpler. That is the subject of the companion post, **Calibrating Long-Horizon Agents: A Post-Training Perspective**, which builds on calibration-aware on-policy distillation ([CaOPD; Zhang et al., 2026d](https://arxiv.org/abs/2604.16830)).

**Takeaway.** Inference-time uncertainty control already works, but the field still needs robust signals, tool-aware uncertainty, multi-agent propagation, better metrics, and — eventually — training-time internalization.

---

## Summary

Long-horizon agents fail differently from single-turn models: their errors compound, their history becomes future context, and their confidence must guide decisions rather than describe outputs. The inference-time playbook follows the reliability loop — **measure** uncertainty across the trajectory, **calibrate** it to success, **act** on it (abstain, ask, verify, reflect, allocate compute, route), and feed it back into **memory and skills**. ACC/HTC shows how to calibrate the whole trajectory; AUQ shows how to turn uncertainty into a runtime control signal; and the surrounding 2025–2026 literature supplies the evidence, the caveats, and the agent-type-specific signals. The frozen-model view takes you a long way. The next step — internalizing these behaviors into the weights — is the companion post.

![A three-stage spine: Measure (AUQ, inference-time), Align (ACC, post-hoc calibration), Internalize (CaOPD, training-time).](/assets/img/blog/calibrating-long-horizon-agents/slide_measure_align_internalize.png)
*Figure 8. The bigger arc this post sits inside: **Measure** trajectory uncertainty at inference time (AUQ), **Align** confidence to success with a transferable calibrator (ACC), and — in the companion post — **Internalize** it into the weights (CaOPD). Uncertainty as an actionable control variable, measured, calibrated, and eventually trained into the policy. (Image source: author's slides.)*

---

*Acknowledgements / sources: figures marked "Image source" are reproduced from the cited papers; all other figures are original (generation scripts in `figures/`).*

---

## How to cite

> Zhang, Jiaxin. (Jun 2026). Calibrating Long-Horizon Agents: Confidence and Uncertainty at Inference Time. *Jiaxin Zhang's Blog.*
> https://jxzhangjhu.github.io/blog/2026/calibrating-long-horizon-agents/

Or in BibTeX:

```bibtex
@article{zhang2026calibratingagents,
  title   = "Calibrating Long-Horizon Agents: Confidence and Uncertainty at Inference Time",
  author  = "Zhang, Jiaxin",
  journal = "Jiaxin Zhang's Blog",
  year    = "2026",
  month   = "Jun",
  url     = "https://jxzhangjhu.github.io/blog/2026/calibrating-long-horizon-agents/"
}
```

---

## References

[1] Pranjal Aggarwal, et al. ["AutoMix: Automatically Mixing Language Models."](https://arxiv.org/abs/2310.12963) arXiv:2310.12963, 2024.

[2] Victor Barres, et al. ["τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment."](https://arxiv.org/abs/2506.07982) arXiv:2506.07982, 2025.

[3] José Cambronero, et al. ["Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair."](https://arxiv.org/abs/2510.03217) arXiv:2510.03217, 2025.

[4] Jinhao Duan, et al. ["UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step Agentic Decision-Making."](https://arxiv.org/abs/2506.17419) arXiv:2506.17419, 2025.

[5] Yichao Fu, et al. ["Deep Think with Confidence."](https://arxiv.org/abs/2508.15260) arXiv:2508.15260, 2025.

[6] Zhiyuan Hu, et al. ["Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models."](https://arxiv.org/abs/2402.03271) NeurIPS 2024. arXiv:2402.03271.

[7] Carlos E. Jimenez, et al. ["SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"](https://arxiv.org/abs/2310.06770) ICLR 2024. arXiv:2310.06770.

[8] Saurav Kadavath, et al. ["Language Models (Mostly) Know What They Know."](https://arxiv.org/abs/2207.05221) arXiv:2207.05221, 2022.

[9] Jean Kaddour, et al. ["Agentic Uncertainty Reveals Agentic Overconfidence."](https://arxiv.org/abs/2602.06948) arXiv:2602.06948, 2026.

[10] Zhewei Kang, et al. ["Scalable Best-of-N Selection for Large Language Models via Self-Certainty."](https://arxiv.org/abs/2502.18581) arXiv:2502.18581, 2025.

[11] Alex Kendall, Yarin Gal. ["What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"](https://arxiv.org/abs/1703.04977) NeurIPS 2017. arXiv:1703.04977.

[12] Adrian Kieback, et al. ["TASR: Training-Free Adaptive Stopping for Iterative Retrieval."](https://arxiv.org/abs/2606.13814) arXiv:2606.13814, 2026.

[13] Michael Kirchhof, Gjergji Kasneci, Enkelejda Kasneci. ["Position: Uncertainty Quantification Needs Reassessment for Large Language Model Agents."](https://arxiv.org/abs/2505.22655) ICML 2025. arXiv:2505.22655.

[14] Polina Kirichenko, et al. ["AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions."](https://arxiv.org/abs/2506.09038) arXiv:2506.09038, 2025.

[15] Lorenz Kuhn, Yarin Gal, Sebastian Farquhar. ["Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation."](https://arxiv.org/abs/2302.09664) ICLR 2023. arXiv:2302.09664.

[16] Divake Kumar, et al. ["Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets."](https://arxiv.org/abs/2606.25760) arXiv:2606.25760, 2026.

[17] Ziming Li, et al. ["Same Signal, Opposite Meaning: Direction-Informed Adaptive Learning for LLM Agents."](https://arxiv.org/abs/2605.06908) arXiv:2605.06908, 2026.

[18] Stephanie Lin, Jacob Hilton, Owain Evans. ["Teaching Models to Express Their Uncertainty in Words."](https://arxiv.org/abs/2205.14334) TMLR 2022. arXiv:2205.14334.

[19] Andrey Malinin, Mark Gales. ["Uncertainty Estimation in Autoregressive Structured Prediction."](https://arxiv.org/abs/2002.07650) ICLR 2021. arXiv:2002.07650.

[20] Grégoire Mialon, et al. ["GAIA: A Benchmark for General AI Assistants."](https://arxiv.org/abs/2311.12983) ICLR 2024. arXiv:2311.12983.

[21] Changdae Oh, et al. ["Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities."](https://arxiv.org/abs/2602.05073) arXiv:2602.05073, 2026.

[22] Victor Ojewale, et al. ["What Benchmarks Don't Measure: The Case for Evaluating Abstention Competence in Autonomous Agents."](https://arxiv.org/abs/2606.02965) arXiv:2606.02965, 2026.

[23] Arka Pal, et al. ["Knowing What You Know Is Not Enough: Large Language Model Confidences Don't Align With Their Actions."](https://arxiv.org/abs/2511.13240) arXiv:2511.13240, 2025.

[24] Stephan Rabanser, et al. ["Towards a Science of AI Agent Reliability."](https://arxiv.org/abs/2602.16666) arXiv:2602.16666, 2026.

[25] Suresh Raghu, Satwik Pandey, Shashwat Pandey. ["Proper Scoring Rules for Agentic Uncertainty Quantification."](https://arxiv.org/abs/2605.24756) arXiv:2605.24756, 2026.

[26] Ashish Rana, et al. ["Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation."](https://arxiv.org/abs/2604.00131) arXiv:2604.00131, 2026.

[27] Allen Z. Ren, et al. ["Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners."](https://arxiv.org/abs/2307.01928) CoRL 2023. arXiv:2307.01928.

[28] Arindam Sharma, Cristina David. ["Assessing Correctness in LLM-Based Code Generation via Uncertainty Estimation."](https://arxiv.org/abs/2502.11620) arXiv:2502.11620, 2025.

[29] Kaiwen Shi, et al. ["Confidence Laundering in Agent Systems: Why Uncertainty Needs a Latent Carrier."](https://arxiv.org/abs/2606.20662) arXiv:2606.20662, 2026.

[30] Noah Shinn, et al. ["Reflexion: Language Agents with Verbal Reinforcement Learning."](https://arxiv.org/abs/2303.11366) NeurIPS 2023. arXiv:2303.11366.

[31] Adi Simhi, et al. ["Old Habits Die Hard: How Conversational History Geometrically Traps LLMs."](https://arxiv.org/abs/2603.03308) arXiv:2603.03308, 2026.

[32] Akshit Sinha, et al. ["The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs."](https://arxiv.org/abs/2509.09677) arXiv:2509.09677, 2025.

[33] Manan Suri, et al. ["Structured Uncertainty guided Clarification for LLM Agents."](https://arxiv.org/abs/2511.08798) arXiv:2511.08798, 2025.

[34] Sina Tayebati, et al. ["TRACER: Trajectory Risk Aggregation for Critical Episodes in Agentic Reasoning."](https://arxiv.org/abs/2602.11409) ICML 2026. arXiv:2602.11409.

[35] Katherine Tian, et al. ["Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback."](https://arxiv.org/abs/2305.14975) EMNLP 2023. arXiv:2305.14975.

[36] Eren Unlu. ["Know When to Trust the Skill: Delayed Appraisal and Epistemic Vigilance for Single-Agent LLMs."](https://arxiv.org/abs/2604.16753) arXiv:2604.16753, 2026.

[37] Sanidhya Vijayvargiya, et al. ["Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering."](https://arxiv.org/abs/2502.13069) arXiv:2502.13069, 2025.

[38] Guanzhi Wang, et al. ["Voyager: An Open-Ended Embodied Agent with Large Language Models."](https://arxiv.org/abs/2305.16291) arXiv:2305.16291, 2023.

[39] Jiawei Wang, et al. ["Are LLM Decisions Faithful to Verbal Confidence?"](https://arxiv.org/abs/2601.07767) arXiv:2601.07767, 2026a.

[40] Qingni Wang, et al. ["SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration."](https://arxiv.org/abs/2602.02419) arXiv:2602.02419, 2026b.

[41] Xuezhi Wang, et al. ["Self-Consistency Improves Chain of Thought Reasoning in Language Models."](https://arxiv.org/abs/2203.11171) ICLR 2023. arXiv:2203.11171.

[42] Wujiang Xu, et al. ["A-MEM: Agentic Memory for LLM Agents."](https://arxiv.org/abs/2502.12110) arXiv:2502.12110, 2025.

[43] Weihao Xuan, et al. ["The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents."](https://arxiv.org/abs/2601.07264) ACL 2026. arXiv:2601.07264.

[44] Daniel Yang, Yao-Hung Hubert Tsai, Makoto Yamada. ["On Verbalized Confidence Scores for LLMs."](https://arxiv.org/abs/2412.14737) arXiv:2412.14737, 2024.

[45] Shunyu Yao, et al. ["ReAct: Synergizing Reasoning and Acting in Language Models."](https://arxiv.org/abs/2210.03629) ICLR 2023. arXiv:2210.03629.

[46] Shunyu Yao, et al. ["Tree of Thoughts: Deliberate Problem Solving with Large Language Models."](https://arxiv.org/abs/2305.10601) NeurIPS 2023. arXiv:2305.10601.

[47] Luke Yoffe, et al. ["DebUnc: Improving Large Language Model Agent Communication With Uncertainty Metrics."](https://arxiv.org/abs/2407.06426) arXiv:2407.06426, 2024.

[48] Skylar Zhai, et al. ["Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL."](https://arxiv.org/abs/2604.17073) arXiv:2604.17073, 2026.

[49] Jiaxin Zhang, Prafulla Kumar Choubey, Kung-Hsiang Huang, Caiming Xiong, Chien-Sheng Wu. ["Agentic Uncertainty Quantification."](https://arxiv.org/abs/2601.15703) arXiv:2601.15703, 2026a.

[50] Jiaxin Zhang, Caiming Xiong, Chien-Sheng Wu. ["Agentic Confidence Calibration."](https://arxiv.org/abs/2601.15778) arXiv:2601.15778, 2026b.

[51] Jiaxin Zhang, et al. ["From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models."](https://arxiv.org/abs/2601.15690) arXiv:2601.15690, 2026c.

[52] Jiaxin Zhang, et al. ["The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation."](https://arxiv.org/abs/2604.16830) arXiv:2604.16830, 2026d.

[53] Andrew Zhao, et al. ["ExpeL: LLM Agents Are Experiential Learners."](https://arxiv.org/abs/2308.10144) AAAI 2024. arXiv:2308.10144.

[54] Qiwei Zhao, et al. ["SAUP: Situation Awareness Uncertainty Propagation on LLM Agent."](https://arxiv.org/abs/2412.01033) ACL 2025. arXiv:2412.01033.
