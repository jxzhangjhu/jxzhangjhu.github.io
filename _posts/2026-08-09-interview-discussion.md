---
layout: post
title: "Interview Bank III · Discussion + BQ: think aloud, decide, defend"
date: 2026-08-09 13:00:00
author: Jiaxin Zhang
description: "A practical bank for frontier-lab research discussion, project deep dives, paper critique, ML system design, experiment design, incident debugging, and behavioral interviews—with spoken answer structures, follow-ups, traps, and worksheets."
tags: interviews llm ml research systems behavioral qbank
categories: research-notes
giscus_comments: true
related_posts: false
ai_assisted: true
---

<div class="lang-switch"><strong>English</strong> · <a href="/blog/2026/interview-discussion-zh/">中文</a></div>

<div class="lang-switch"><a href="/blog/2026/interview-knowledge/">I · Knowledge</a> · <a href="/blog/2026/interview-coding/">II · Coding + Math</a> · <strong>III · Discussion + BQ</strong></div>

Part I asks whether you can retrieve the knowledge. Part II asks whether you can produce the code or
derivation under a clock. This part asks a different question: **can another person follow your
judgment while the problem is still underspecified?**

This is not a list of polished speeches. Each section starts with a mental model, then turns it into
spoken prompts, answer shapes, follow-ups, traps, and drills. The sample language is explicitly
illustrative. Any claim about *your* ownership, scale, cost, failure, conflict, or impact must come
from you.

> **How to use it.** Pick one prompt, take thirty seconds to draw a map, and answer aloud. Stop after
> two minutes unless the prompt specifies otherwise. Then reopen the answer, compare structures rather
> than sentences, and let a mock interviewer press one branch three levels deep.
>
> **The reusable spine.** Clarify the decision and constraints → establish a baseline → state competing
> hypotheses → propose the cheapest discriminating test → define success and guardrails → choose and
> explain the trade-off → name failure modes and the next decision.

The structure extends the existing
[**three gates / seven buckets**](/blog/2026/frontier-lab-interview/) synthesis. It expands Bucket 5
(ML systems design) and Bucket 6 (research taste, deep dives, and values), with scientific reasoning
and verbal debugging as cross-cutting skills. It is a study index, not a claim that lab loops are
standardized. Frequency is stated only when a named source supports it; otherwise a prompt is marked
**commonly useful**.

---

### Table of contents

- **[D0 · How to train discussion, not scripts](#section-d0)** — 5 prompts
  - [D0.1 · Build a map before sentences](#d0-1)
  - [D0.2 · Calibrate claims and familiarity](#d0-2)
  - [D0.3 · Practise retrieval under pressure](#d0-3)
- **[D1 · A reusable technical-discussion frame](#section-d1)** — 8 prompts
  - [D1.1 · Clarify the decision and constraints](#d1-1)
  - [D1.2 · Establish the baseline and competing hypotheses](#d1-2)
  - [D1.3 · Design the cheapest discriminating experiment](#d1-3)
  - [D1.4 · Define metrics, choose, and name failure modes](#d1-4)
- **[D2 · Project deep dive: prove depth without giving a chronology](#section-d2)** — 10 prompts
  - [D2.1 · Prepare two-, five-, and fifteen-minute versions](#d2-1)
  - [D2.2 · Separate personal contribution, team contribution, and influence](#d2-2)
  - [D2.3 · Defend the decisive choice and the failed path](#d2-3)
  - [D2.4 · Discuss debugging, scale, and cost as decisions](#d2-4)
  - [D2.5 · Establish result credibility and future work](#d2-5)
- **[D3 · Research taste and the paper discussion](#section-d3)** — 11 prompts
  - [D3.1 · Read to the depth the conversation needs](#d3-1)
  - [D3.2 · Critique the claim, not the paper's aesthetics](#d3-2)
  - [D3.3 · Design a reproduction before designing an extension](#d3-3)
  - [D3.4 · Use ablations to test mechanisms](#d3-4)
  - [D3.5 · Judge novelty, impact, and the next experiment](#d3-5)
- **[D4 · Open-ended ML and LLM system design](#section-d4)** — 13 prompts
  - [D4.1 · Turn the vague prompt into an end-to-end contract](#d4-1)
  - [D4.2 · Make the data and objective do real work](#d4-2)
  - [D4.3 · Choose model and training only after locating the bottleneck](#d4-3)
  - [D4.4 · Resolve capability, latency, cost, and safety as a Pareto problem](#d4-4)
  - [D4.5 · Design evaluation, rollout, and monitoring together](#d4-5)
  - [D4.6 · Design a long-running coding and research agent as a durable system](#d4-6)
- **[D5 · Experiment design and scientific reasoning](#section-d5)** — 12 prompts
  - [D5.1 · Turn a claim into an estimand and control](#d5-1)
  - [D5.2 · Find leakage and confounding before optimizing](#d5-2)
  - [D5.3 · Reason about variance, power, and multiple comparisons](#d5-3)
  - [D5.4 · Connect offline metrics to online decisions](#d5-4)
  - [D5.5 · Learn from negative and ambiguous results](#d5-5)
  - [D5.6 · Design scaling experiments that identify regime changes](#d5-6)
- **[D6 · Debugging and incident discussion](#section-d6)** — 12 prompts
  - [D6.1 · Verify, bound, and preserve evidence](#d6-1)
  - [D6.2 · Discuss a training loss spike](#d6-2)
  - [D6.3 · Debug a model or product regression](#d6-3)
  - [D6.4 · Find training-serving skew](#d6-4)
  - [D6.5 · Suspect the evaluation when the conclusion is impossible](#d6-5)
  - [D6.6 · Localize distributed failures by topology and boundary](#d6-6)
  - [D6.7 · Debug agent side effects and prompt injection together](#d6-7)
- **[D7 · Behavioral questions: show decisions, not adjectives](#section-d7)** — 18 prompts
  - [D7.1 · Answer the question in the first sentence](#d7-1)
  - [D7.2 · Motivation and “why this lab”](#d7-2)
  - [D7.3 · Ownership, ambiguity, and prioritization](#d7-3)
  - [D7.4 · Conflict, collaboration, and feedback](#d7-4)
  - [D7.5 · Failure and learning speed](#d7-5)
  - [D7.6 · Leadership and influence](#d7-6)
  - [D7.7 · Ethics and safety under uncertainty](#d7-7)
  - [D7.8 · Show collaboration across different functions](#d7-8)
- **[D8 · Build a story bank without becoming mechanical](#section-d8)** — 9 prompts
  - [D8.1 · Create an eight-story matrix](#d8-1)
  - [D8.2 · Build the follow-up tree before polishing prose](#d8-2)
  - [D8.3 · Use public artifacts only as candidate story prompts](#d8-3)
  - [D8.4 · Keep the delivery natural and interruptible](#d8-4)
- **[D9 · Interviewer questions, communication rubric, mocks, and debrief](#section-d9)** — 9 prompts
  - [D9.1 · Ask questions that can change your decision](#d9-1)
  - [D9.2 · Score communication with observable behavior](#d9-2)
  - [D9.3 · Run mocks that reproduce the round](#d9-3)
  - [D9.4 · Debrief into a small error taxonomy](#d9-4)
  - [D9.5 · Run a final readiness review](#d9-5)
- **[D10 · Lab-specific preparation, bounded by evidence](#section-d10)** — 11 prompts
  - [D10.1 · Build the recruiter packet first](#d10-1)
  - [D10.2 · OpenAI: follow the official process, then the team](#d10-2)
  - [D10.3 · Anthropic: policy is explicit, so follow it literally](#d10-3)
  - [D10.4 · Meta: role-specific materials and AI policy are the source of truth](#d10-4)
  - [D10.5 · Keep company differences evidence-backed](#d10-5)
  - [D10.6 · Google DeepMind: use the role-specific packet, not a fixed question bank](#d10-6)
- **[References](#section-refs)**

---
<a id="section-d0"></a>

## D0 · How to train discussion, not scripts

Discussion rounds are not the oral version of Part I. Alisa Liu separates rapid-fire breadth from an
extended technical discussion in which the interviewer keeps changing the evidence and asking the
candidate to defend a choice [[R9]](#ref-r9). Yong Zheng-Xin's account adds the warning that round
formats are genuinely diverse [[R11]](#ref-r11). The transferable skill is therefore not memorizing a
perfect answer. It is keeping a decision process legible while the problem moves.

The unit of practice in this bank is a **spoken decision**:

1. What decision are we trying to make?
2. What is known, assumed, or missing?
3. What is the cheapest useful baseline?
4. Which competing explanations matter?
5. What evidence would separate them?
6. What would make us ship, stop, or change course?

If an answer contains facts but never reaches a decision, it is still incomplete.

> **Minimum viable path.** Do not try to finish all 110 prompts. Read D1 once; prepare two projects
> from D2; choose one prompt each from D3–D6; privately fill four real stories from D7–D8; run one
> mock from D9; then use the [D9.5 readiness review](#d9-5). Add prompts only when a mock, recruiter
> packet, or factual gap gives you a reason. A small set recalled under follow-up is more useful than
> a fully highlighted bank.

---

<a id="d0-1"></a>
### D0.1 · Build a map before sentences

Use the first twenty to thirty seconds to write five nouns, not a paragraph:

`goal · constraints · baseline · test · risks`

Then say the map out loud: “I’ll first pin down what success means, then choose a baseline, then
compare two hypotheses with one discriminating experiment, and finish with rollout risks.” This does
three things. It buys thinking time without silence, gives the interviewer a chance to redirect, and
makes later detail feel like depth rather than wandering. OpenAI's official guidance explicitly asks
candidates to expose how they consider and solve problems, and its recruiting event recommends
showing the work and validating assumptions with the interviewer [[R4]](#ref-r4)
[[R5]](#ref-r5).

The map is provisional. If the interviewer says “focus on evaluation,” acknowledge the redirect:
“Great; I’ll hold the architecture constant and spend the time on the eval design.” Good
collaboration is not stubbornly completing the outline you announced.

<a id="d0-1-1"></a>
#### D0.1.1 · Open an underspecified technical prompt

**Status.** Commonly useful.

**Prompt.** “How would you improve this model?”

**Illustrative spoken opening.**

> “Before proposing a method, I want to turn ‘improve’ into a decision. Who uses the model, on what
> distribution, and which failure currently matters: capability, latency, cost, calibration, or
> safety? I’ll assume for now that quality on a fixed production-like set is primary, latency is a
> guardrail, and we can change data and post-training but not the base architecture. I’d reproduce the
> current system as the baseline, slice its errors, form one hypothesis per dominant slice, and run
> the cheapest experiment that could disprove each hypothesis. If those assumptions are wrong, I’ll
> update the plan.”

That answer has not solved the problem. It has made the next question answerable.

**Practice follow-ups**

- “You cannot ask the user; choose assumptions.” State them and attach a sensitivity check.
- “What if the offline metric disagrees with users?” Name the online decision and guardrails.
- “You have one week.” Reduce the plan to instrumentation, a baseline, and one high-information test.

**Traps**

- Listing fine-tuning methods before defining the failure.
- Asking ten questions and never choosing defaults.
- Treating “accuracy” as a complete metric without a distribution or decision threshold.

**Drill.** Take five vague verbs—improve, scale, align, debug, evaluate—and give a forty-five-second
opening for each. The opening passes only if it ends with a concrete first comparison.

<a id="d0-1-2"></a>
#### D0.1.2 · Recover when the interviewer changes direction

**Status.** Commonly useful.

**Prompt.** Halfway through your design, the interviewer says, “I care less about training. Tell me
how this fails after launch.”

**Answer shape.**

> “Understood. I’ll freeze the training choice and switch to deployment risk. I see four layers:
> input drift, model-quality regressions, system reliability, and harmful or costly actions. For each
> I’ll name an observable signal, a threshold, and a rollback or containment action. I’ll start with
> input drift because it can invalidate every offline result.”

**Practice follow-ups**

- “Which signal alerts first?” Separate leading indicators from lagging user outcomes.
- “What can you roll back?” Distinguish weights, prompts, retrieval index, routing, and policy.
- “What if the metric itself is broken?” Add shadow labels, audits, and an independent sentinel.

**Trap.** Defending the abandoned branch because you prepared it. The redirection is part of the
collaboration signal.

---

<a id="d0-2"></a>
### D0.2 · Calibrate claims and familiarity

Use three verbal labels:

- **Observed:** “In our experiment, under this setup, we saw…”
- **Supported:** “The paper reports…, and I believe the mechanism is…”
- **Hypothesized:** “I have not tested this; my current hypothesis is…”

Mimansa Jaiswal maintains an intentionally broad “I know of these papers” ledger and explicitly tells
interviewers whether knowledge came from a deep read, a blog, or a discussion [[R14]](#ref-r14).
That is not weakness. It prevents a skim from being presented as implementation experience and gives
the interviewer a useful place to probe.

Calibrated language still needs a position. “I do not know” is the beginning:
“I have not used that method. Based on the objective you described, I would expect X because Y. The
first thing I would check is Z.” Refusing to reason is not calibration.

<a id="d0-2-1"></a>
#### D0.2.1 · Answer when you do not know the named method

**Status.** Commonly useful.

**Prompt.** “Would you use Method X here?” You have heard the name but have not read the paper.

**Illustrative spoken answer.**

> “I know of it, but I have not read it closely enough to claim its details. If you give me its key
> objective or constraint, I can compare it with the baseline from first principles. My current
> understanding is that it trades additional inference work for better search. If that is right, I
> would expect gains on tasks with verifiable alternatives and weaker returns when latency dominates.
> I would verify the exact claim before making the design depend on it.”

**Practice follow-ups**

- “Suppose it does X; now compare it.” Restate the new fact, then reason normally.
- “What would you read first?” Objective, evaluation setup, strongest baseline, and ablation.
- “Have you implemented anything similar?” Answer literally; do not convert conceptual familiarity
  into hands-on experience.

**Traps**

- Bluffing a mechanism.
- Stopping at “I don't know.”
- Apologizing for so long that no reasoning follows.

<a id="d0-2-2"></a>
#### D0.2.2 · Separate a result from its explanation

**Status.** Commonly useful.

**Prompt.** “The larger model improves the benchmark. Does that prove the proposed mechanism?”

**Answer shape.**

> “It supports the empirical claim that this configuration scores higher. It does not isolate the
> mechanism yet. Scale changes capacity, optimization, and often data or compute at the same time. I
> would need a matched-compute control and an intervention that changes the proposed mechanism while
> holding the obvious alternatives fixed. I would also ask whether the gain is broad or concentrated
> in one slice.”

**Follow-ups.** Propose the exact control; say what a null result means; explain whether the mechanism
is necessary, sufficient, both, or neither.

**Trap.** Treating a plausible story as a measured causal explanation.

---

<a id="d0-3"></a>
### D0.3 · Practise retrieval under pressure

Reading an answer creates recognition. A discussion round needs retrieval, compression, and branch
control. Use a four-pass practice loop:

1. **Cold answer:** two minutes, no notes.
2. **Branch:** a partner chooses one claim and asks “why?” three times.
3. **Compression:** answer the same prompt in thirty seconds.
4. **Repair:** record one missing decision, one unsupported claim, and one sentence to remove.

Do not score charisma. Score whether the listener can recover the goal, assumptions, evidence,
choice, and caveat. Keep the error log small enough to revisit. Jaiswal's graded recall categories are
useful precisely because they separate “I saw this” from “I could produce it” [[R14]](#ref-r14).

<a id="d0-3-1"></a>
#### D0.3.1 · Run a useful self-debrief

**Status.** Commonly useful.

**Prompt.** You finish a mock answer and it felt bad. What do you write down?

**Answer shape.**

> “I would not write ‘be clearer.’ I would identify the first observable failure. Did I start without
> a decision? Did I use an undefined metric? Did a follow-up reveal that I had no baseline, no
> ownership detail, or no failure mode? Then I would turn that into one next drill. For example:
> ‘Tomorrow, answer three prompts with the baseline in the first sixty seconds.’”

**Useful log fields**

- prompt and role;
- first point where the answer became hard to follow;
- missing fact versus missing reasoning;
- strongest follow-up and whether it exposed a real gap;
- one claim that needs a source;
- one sentence that sounded memorized;
- next repetition date and one constrained drill.

**Trap.** Rewriting the whole answer into polished prose. That trains editing, not speaking.

---

<a id="section-d1"></a>

## D1 · A reusable technical-discussion frame

The extended technical discussion described in the first-hand sources usually starts with a research
or product question, then changes one condition at a time: defend the baseline, interpret a
hypothetical result, choose the next experiment, or explain a trade-off [[R9]](#ref-r9). A useful
answer therefore needs a control loop, not a static outline:

`frame the decision → model the possibilities → choose evidence → update the decision`

The seven-move spine below is a detailed version of that loop. Do not recite all seven labels. Use
them to notice what is missing.

| Move | Question it answers | Spoken artifact |
|---|---|---|
| Clarify | What are we deciding, for whom, and by when? | objective, unit, constraints |
| Baseline | What is the simplest credible comparison? | current system or cheap heuristic |
| Hypotheses | What different worlds could explain the failure? | two or three falsifiable stories |
| Experiment | What is the cheapest test that separates those worlds? | intervention and control |
| Metrics | What counts as success, and what must not regress? | primary metric, slices, guardrails |
| Trade-off | Which option do I choose under these constraints? | explicit default and sensitivity |
| Failure modes | How could the conclusion or system be wrong? | monitoring, rollback, next test |

Google's production-ML guidance repeatedly recommends a simple first model, solid metrics, and
instrumentation before complexity [[R18]](#ref-r18). That is not merely production advice. It is a
strong discussion order because every later idea has something concrete to beat.

---

<a id="d1-1"></a>
### D1.1 · Clarify the decision and constraints

Weak clarification collects context. Strong clarification changes the design space.

Ask questions in descending order of leverage:

1. **Decision:** Are we choosing whether to launch, what to build, or what experiment to run?
2. **User and harm:** Who receives the output, and what does a false positive or false negative cost?
3. **Unit and horizon:** Per token, request, session, task, customer, or month?
4. **Distribution:** What traffic or data must generalize, including the hard tail?
5. **Budget:** Latency, compute, labels, engineer time, deadline, and reversibility.
6. **Control:** What can be changed, and what must stay fixed?

After two or three questions, choose assumptions. Clarification without commitment becomes
avoidance.

<a id="d1-1-1"></a>
#### D1.1.1 · Turn “build a safe assistant” into a decision

**Status.** Commonly useful.

**Prompt.** “Design a safer LLM assistant.”

**Illustrative spoken answer.**

> “I need to narrow both ‘safer’ and ‘assistant.’ I’ll assume a general consumer assistant that can
> browse but cannot execute irreversible actions. The primary decision is whether a candidate policy
> is safe enough for a staged launch. I would define safety by a risk taxonomy and severity-weighted
> violation rate, measured on adversarial and natural traffic; helpfulness is a co-primary metric,
> and latency plus over-refusal are guardrails. We can change policy, classifiers, and tool
> permissions, but not the base model this quarter. Under those assumptions I would start from the
> current model with no new filter, because every intervention must beat its risk–helpfulness curve.”

**Practice follow-ups**

- “Why is over-refusal a guardrail rather than the primary metric?” Explain the product objective and
  what decision would change that choice.
- “How do you combine severity and frequency?” Keep both visible; do not hide rare catastrophic
  events in one mean.
- “Now allow money transfer.” Move from model-only safety to permissions, confirmation, limits,
  audit, and recovery.

**Traps**

- Saying “safety is important” without defining observable outcomes.
- Starting with an alignment algorithm before defining the product's action surface.
- Producing one scalar that lets a common minor issue erase a rare severe one.

<a id="d1-1-2"></a>
#### D1.1.2 · Choose assumptions when the interviewer will not

**Status.** Commonly useful.

**Prompt.** “There is no more information. Make reasonable assumptions and proceed.”

**Answer shape.**

> “I’ll pick a concrete operating point and flag the two assumptions that can reverse the answer.
> Assume interactive traffic, a p95 first-token target of two seconds, and a fixed GPU fleet. I will
> optimize goodput under that SLO. If traffic is batch rather than interactive, or if quality can be
> traded for a smaller model, the architecture changes; I’ll return to those in sensitivity checks.”

**Follow-ups.** Change one assumption and update only the affected branches. State which parts remain
invariant.

**Trap.** Inventing precise numbers and later speaking as if the interviewer supplied them. Keep
assumptions visibly yours.

---

<a id="d1-2"></a>
### D1.2 · Establish the baseline and competing hypotheses

A baseline is not the weakest thing you can beat. It is the simplest alternative a serious owner
would deploy. Depending on the question, that may be:

- the current production system;
- a rule or retrieval-only solution;
- a smaller model with more data;
- an existing method at matched compute and labels;
- no change, when intervention has cost or risk.

Then create hypotheses that imply different observations. “The model needs more data” is not yet a
hypothesis. “Failures concentrate on unseen schema combinations; adding compositionally diverse data
should improve that slice while leaving familiar schemas unchanged” is.

Keep the set small. Two well-separated explanations are more useful than ten causes that all predict
the same metric movement.

<a id="d1-2-1"></a>
#### D1.2.1 · Build a baseline that can win

**Status.** Commonly useful.

**Prompt.** “Use an LLM to route support tickets.”

**Illustrative spoken answer.**

> “My baseline is not a random classifier. It is the current rules plus a supervised linear or small
> encoder classifier on the existing labels. Routing has a closed label set and usually rewards
> consistency and latency, so a generative model must earn its complexity. I would compare the LLM
> only after fixing the label taxonomy and measuring the current handoff cost. The LLM may win on
> sparse classes or explanations, but if a small model meets the routing target, I would keep it and
> reserve the LLM for ambiguous tickets.”

**Practice follow-ups**

- “What if labels change weekly?” Compare retraining, prompt updates, and retrieval over label
  definitions.
- “What if a ticket can have several destinations?” Recast as ranking or multilabel prediction.
- “What is the cost of a mistake?” Separate delay, privacy exposure, and irreversible action.

**Trap.** Choosing an LLM because the prompt contains “LLM.”

<a id="d1-2-2"></a>
#### D1.2.2 · Produce hypotheses that disagree

**Status.** Commonly useful.

**Prompt.** A coding agent passes unit tests but fails user acceptance. Give a useful hypothesis set.

**Answer shape.**

> “I would start with three explanations that predict different evidence. First, the tests are
> incomplete: success rises with test coverage but not with model changes. Second, the agent is
> exploiting the tests: suspicious special cases and brittle patches rise while hidden tests fail.
> Third, the task specification is underdetermined: independent human reviewers disagree on what
> counts as correct. I would not tune the policy until I know which world I am in, because each needs
> a different fix—better evaluation, stronger isolation, or a clearer task.”

**Follow-ups.** Name a test for each hypothesis; prioritize by expected information per unit cost;
explain what evidence would make you abandon one.

**Trap.** Listing “data, model, optimization” as categories without predictions that distinguish
them.

---

<a id="d1-3"></a>
### D1.3 · Design the cheapest discriminating experiment

An experiment is useful when its possible outcomes lead to different next actions. Before describing
implementation, complete this sentence:

> “If the result is X, I will believe A and do Y; if it is not, I will believe B more and do Z.”

Prefer interventions that isolate one uncertain link:

- a tiny controlled slice before a full retrain;
- an oracle component to estimate the ceiling of one subsystem;
- shadow traffic before user-visible rollout;
- a matched-budget ablation before comparing unconstrained systems;
- a hand audit before building an automated metric.

The cheapest experiment is not necessarily the smallest run. It is the one that buys the most
decision-relevant information, including engineer time and the cost of a misleading result.

<a id="d1-3-1"></a>
#### D1.3.1 · Pick the next experiment after an ambiguous gain

**Status.** Commonly useful.

**Prompt.** Adding synthetic data improved aggregate accuracy by two points. What next?

**Illustrative spoken answer.**

> “I would not scale the recipe yet. The gain could come from more tokens, duplicated test-like
> content, label quality, or the intended diversity. My next run is a matched-token comparison:
> real-only resampling versus synthetic augmentation, with deduplication against evaluation data and
> results sliced by examples the generator could not have copied. If the synthetic arm wins mainly
> on novel slices and across seeds, I would invest in the generator. If the gain disappears at matched
> tokens, the causal story is volume, not synthesis.”

**Practice follow-ups**

- “You can run only one more job.” Choose the comparison with the highest value of information.
- “The gain is real but only on easy examples.” Decide whether the product objective values them.
- “The generator used the same teacher as the evaluator.” Identify correlated error and use an
  independent judge or human audit.

**Traps**

- Treating two points as meaningful without uncertainty or slice behavior.
- Running a full factorial grid before checking leakage and a matched-token control.

<a id="d1-3-2"></a>
#### D1.3.2 · Use an oracle to locate the ceiling

**Status.** Commonly useful.

**Prompt.** A RAG system is weak. Should you improve retrieval or generation?

**Answer shape.**

> “I would separate the stages with two oracle tests. Feed the generator gold evidence: that estimates
> the generation ceiling if retrieval were perfect. Then keep the retriever's documents but replace
> generation with a human or exact evidence check: that measures whether the answer was present.
> If gold evidence barely helps, retrieval is not the first bottleneck. If answers become strong with
> gold evidence but the current retriever rarely contains them, retrieval is.”

**Follow-ups.** Discuss unanswerable queries, stale corpora, ranking versus recall, and whether the
oracle distribution is unrealistically clean.

**Trap.** Changing retriever and generator together, then being unable to attribute the gain.

---

<a id="d1-4"></a>
### D1.4 · Define metrics, choose, and name failure modes

A complete metric set has at least four layers:

1. **Primary outcome:** the quantity tied to the decision.
2. **Guardrails:** what may not get materially worse.
3. **Slices:** where the mean can hide unacceptable behavior.
4. **Measurement health:** agreement, coverage, contamination, and latency of the metric itself.

Then choose. A trade-off answer without a default sounds informed but not responsible:

> “Given interactive use and a strict tail-latency SLO, I choose the smaller model with retrieval.
> I would switch to the larger model if the high-severity error reduction exceeds X at matched
> goodput.”

Finally, distinguish **system failure** from **inference failure**. The system can be bad; the
experiment can also be unable to tell whether it is good. Monitoring only the model leaves both
measurement drift and operational failures invisible. The ML Test Score treats data, model,
infrastructure, and monitoring tests as separate production-readiness layers
[[R19]](#ref-r19).

<a id="d1-4-1"></a>
#### D1.4.1 · Resolve capability, latency, cost, and safety

**Status.** Commonly useful.

**Prompt.** Model A is better but twice as slow and expensive. Which do you ship?

**Illustrative spoken answer.**

> “I cannot answer from aggregate quality alone. I would first translate the quality delta into user
> or risk value by slice. My default is a cascade: Model B handles traffic where its calibrated
> confidence and risk score are acceptable; Model A receives the ambiguous or high-consequence tail.
> I would compare that cascade with both single-model baselines on goodput, cost per successful task,
> severe-error rate, and p95 latency. I ship the simplest system on the Pareto frontier, not
> automatically the highest-quality model.”

**Practice follow-ups**

- “The router is wrong.” Include router error in end-to-end evaluation and a conservative fallback.
- “Caching changes the economics.” Recompute under the real request distribution.
- “Safety cannot be averaged.” Keep severity classes and worst-case slices explicit.

**Trap.** Saying “it depends” and listing axes without selecting an operating point.

<a id="d1-4-2"></a>
#### D1.4.2 · Finish with a pre-mortem and decision rule

**Status.** Commonly useful.

**Prompt.** End any open-ended design discussion well.

**Answer shape.**

> “Before launch I would try to kill this design in four ways: break the data assumptions, saturate
> the resource bottleneck, attack the metric, and force an irreversible high-impact action. I would
> launch first in shadow mode, then to a reversible low-risk slice. The go condition is improvement
> on the primary outcome with guardrails inside their bounds; the rollback condition is any severe
> safety event, sustained tail-latency breach, or measurement-health failure. The first post-launch
> question is whether the traffic distribution now differs from the one that justified the choice.”

**Follow-ups.** Set owners and alert thresholds; distinguish automatic rollback from human review;
say what evidence is retained for incident reconstruction.

**Trap.** Ending with architecture. A real design ends with how the decision is validated and
reversed.

---

<a id="section-d2"></a>

## D2 · Project deep dive: prove depth without giving a chronology

A project deep dive is the one round where the raw material is entirely yours. First-hand accounts
describe it as starting from past work and then following whichever thread reveals judgment
[[R9]](#ref-r9). Yuan Meng's useful distinction is that coding can clear a bar, while a deep dive
shows why this particular person fits the team [[R13]](#ref-r13).

The interviewer is usually trying to recover five things:

1. **Problem ownership:** did you understand why the work existed?
2. **Technical depth:** can you descend from the claim into implementation and evidence?
3. **Decision quality:** did you choose among real alternatives under constraints?
4. **Epistemic honesty:** do you know what the result establishes and what it does not?
5. **Collaboration:** can you separate individual contribution from the team without erasing either?

Do not tell the project in calendar order. Tell it as an argument:

`problem and stakes → your claim → decisive choice → evidence → limitation → next move`

Before using any template below, replace every bracket with a fact you can defend. If a number, role,
failure, or decision is not documented or remembered, leave it blank and recover the evidence. A
plausible placeholder is still a fabricated story.

---

<a id="d2-1"></a>
### D2.1 · Prepare two-, five-, and fifteen-minute versions

The versions are not the same talk at different speeds.

| Version | Job | What survives |
|---|---|---|
| 2 minutes | earn the next question | problem, one claim, your role, strongest evidence, one limitation |
| 5 minutes | expose decision quality | add baseline, decisive alternative, experiment, result interpretation |
| 15 minutes | withstand a defense | add architecture, failed paths, scale/cost, collaboration, future work |

Do not confuse a **prepared job talk** with a **conversational deep dive**. A job talk has a fixed
audience, time budget, visual narrative, and research agenda; you control the sequence. In a deep
dive, the interviewer controls the branch, may interrupt after two minutes, and can stay on one
decision for the rest of the round. Prepare the same evidence base for both, but rehearse the talk for
coherent delivery and the deep dive for random access.

The short version should contain a deliberate **handle**: one decision or surprising result that you
want the interviewer to pull. Do not end with “and that was the project.” End with a live edge:
“The decision I would most like to unpack is why we abandoned X despite its better offline score.”

<a id="d2-1-1"></a>
#### D2.1.1 · Give a two-minute project overview

**Status.** Reported as a recurring research-discussion shape; exact timing is a practice device.

**Worksheet, not a completed answer**

> “The problem was **[user/scientific need]**, and the existing approach failed because **[specific
> bottleneck]**. Our main claim was **[one falsifiable contribution]**. I personally owned
> **[artifact/decision/experiment]**, while **[team roles]** owned **[their work]**. The decision that
> shaped the project was **[choice]** over **[alternative]**, because **[constraint]**. On
> **[evaluation setting]**, we observed **[verified result with uncertainty or slice]**. The result
> supports **[narrow conclusion]**, but not **[limitation]**. If useful, I can go deeper on
> **[decision handle]** or **[failure handle]**.”

**Practice follow-ups**

- “What exactly did you do?”
- “Why was the prior approach insufficient?”
- “Which result most changed your mind?”
- “What would fail at ten times the scale?”

**Traps**

- Spending ninety seconds on background and rushing the contribution.
- Saying “we” everywhere, which hides personal ownership.
- Saying “I” everywhere, which makes a team project implausible.
- Giving a metric without the baseline, dataset, or uncertainty that gives it meaning.

**Drill.** Record the answer. At two minutes, a listener should be able to write one sentence each for
problem, claim, ownership, evidence, and caveat. If not, remove details; do not speak faster.

<a id="d2-1-2"></a>
#### D2.1.2 · Expand from two minutes to five

**Status.** Commonly useful.

**Prompt.** “Give me more detail.”

**Answer shape.**

> “I’ll add the decision and evidence layers rather than repeat the overview. The baseline we had to
> beat was **[baseline]**. We considered **[A]**, **[B]**, and **[C]**. The uncertainty was
> **[unknown]**, so I designed **[test]**. The result ruled out **[hypothesis]** but left
> **[remaining ambiguity]**. That is why we chose **[decision]**. The strongest counterevidence was
> **[failure or slice]**, and we handled it by **[action]**.”

**Follow-ups.** Ask for architecture; matched-budget fairness; an ablation; deployment evidence; the
person who disagreed.

**Trap.** Expanding by adding every component. Expand the causal argument, not the inventory.

---

<a id="d2-2"></a>
### D2.2 · Separate personal contribution, team contribution, and influence

Ownership is not “I wrote most of the code.” It can be:

- identifying the problem or reframing the objective;
- designing the decisive experiment;
- building a critical component;
- resolving a failure no one understood;
- aligning several teams on a technical decision;
- taking a prototype through deployment and measurement;
- deciding to stop a line of work.

Use an ownership ledger:

| Layer | Fill with facts |
|---|---|
| I directly built or decided | [specific artifacts and decisions] |
| I influenced but did not own | [proposal, review, alignment, unblock] |
| Teammates owned | [names/roles if public and appropriate] |
| Existing work I inherited | [baseline, infrastructure, prior research] |
| Evidence of my role | [design doc, commits, experiment log, presentation, decision record] |

The purpose is not legalistic attribution. It lets you use “we” for the outcome and “I” for your
actions without confusion.

<a id="d2-2-1"></a>
#### D2.2.1 · Answer “What did you personally do?”

**Status.** Commonly reported follow-up.

**Illustrative structure.**

> “The team outcome was **[outcome]**. My direct ownership had three parts: I framed **[decision]**,
> built **[artifact]**, and ran **[experiment]** that changed us from **[old direction]** to
> **[new direction]**. **[Collaborator or team]** owned **[their component]**, which my work depended
> on. I also influenced **[cross-team decision]**, but I would not describe that component as mine.”

**Practice follow-ups**

- “Could the project have succeeded without your part?”
- “Who made the final decision?”
- “What did you inherit?”
- “Show me one artifact that demonstrates the ownership.”

**Traps**

- Converting leadership into “I told people what to do.”
- Claiming an entire platform because you led one workstream.
- Minimizing collaboration until the story no longer sounds credible.

<a id="d2-2-2"></a>
#### D2.2.2 · Describe leadership without formal authority

**Status.** Commonly useful for senior research and engineering roles.

**Answer shape.**

> “I did not own **[people/team]**. I owned the decision process. The groups disagreed about
> **[technical choice]** because they optimized different constraints. I made those constraints
> explicit, proposed **[shared experiment or decision document]**, and got agreement in advance on
> what result would choose each path. The evidence favored **[path]**. The influence was not that I
> won an argument; it was that the team had a decision rule everyone trusted.”

**Follow-ups.** What if the result had favored the other side? Who still disagreed? What relationship
cost did the process create? What would you do faster now?

**Trap.** Describing consensus as the goal. Sometimes the job is a clear decision with documented
dissent.

---

<a id="d2-3"></a>
### D2.3 · Defend the decisive choice and the failed path

Prepare two or three **decision cards** per project:

- decision and date;
- constraints known at the time;
- alternatives considered;
- evidence available then—not evidence learned later;
- expected upside and failure mode of each;
- who decided and who disagreed;
- result and what you learned.

This prevents hindsight narration. A good past decision can lead to a bad outcome; a lucky result does
not make a weak decision rigorous.

Also prepare one failed experiment in technical detail. “It did not work” is not a failure analysis.
Explain what it predicted, what happened, how you ruled out implementation error, and how it changed
the project.

<a id="d2-3-1"></a>
#### D2.3.1 · Explain why you chose A over B

**Status.** Commonly reported follow-up.

**Worksheet.**

> “At that point we needed to decide **[decision]**. A optimized **[axis]** but risked **[failure]**;
> B optimized **[other axis]** but cost **[trade-off]**. The binding constraint was **[constraint]**,
> not **[tempting but secondary metric]**. The evidence we had then was **[evidence]**, so I
> recommended **[choice]**. I would have switched to the other option if **[observable threshold]**.
> In hindsight, **[what the result says about the decision, without pretending you knew it]**.”

**Practice follow-ups**

- “Why not run both?”
- “What did you underweight?”
- “Who argued for B, and what were they right about?”
- “Would the decision change with twice the compute or half the deadline?”

**Trap.** Using the eventual result as if it was available at decision time.

<a id="d2-3-2"></a>
#### D2.3.2 · Walk through a failed experiment

**Status.** Commonly useful and often a deep follow-up.

**Answer shape.**

> “The hypothesis was **[mechanism]**, which predicted **[specific movement]**. We changed
> **[intervention]** and held **[controls]** fixed. Instead we saw **[result]**. Before rejecting the
> idea, I checked **[implementation/data/eval checks]**. The failure was informative because it ruled
> out **[narrow claim]** and exposed **[new bottleneck]**. We stopped **[work]**, changed
> **[direction]**, and kept **[reusable artifact]**. What I would change now is **[earlier or cheaper
> discriminating test]**.”

**Follow-ups.** How much did it cost? When did you know to stop? Did anyone predict it? What negative
result remains unexplained?

**Traps**

- Choosing a fake failure that ends as an unqualified success.
- Blaming only data, infrastructure, or another team.
- Claiming a broad theory was disproved by one noisy run.

---

<a id="d2-4"></a>
### D2.4 · Discuss debugging, scale, and cost as decisions

Scale is not a prestige number. It changes failure modes. Prepare a small **resource ledger** using
only figures you can disclose and verify:

- data units and distribution, not just total count;
- model size and active parameters if relevant;
- training and inference compute;
- latency distribution and throughput;
- storage, memory, and network bottlenecks;
- label and engineer time;
- direct cost and opportunity cost;
- recovery point: checkpoint frequency, rollback window, and lost work.

If a number is confidential, do not improvise a substitute. Use an approved range, normalize to a
baseline, or say what scales asymptotically: “I cannot share the fleet size, but the relevant point is
that checkpoint time grew linearly with state size and exceeded our recovery budget.”

<a id="d2-4-1"></a>
#### D2.4.1 · Tell a debugging story as hypothesis reduction

**Status.** Commonly useful.

**Answer shape.**

> “The observable failure was **[symptom]**, first detected by **[signal]**. I separated data,
> numerical, code, and infrastructure hypotheses. The highest-information check was **[check]**
> because it split **[hypotheses]**. That localized the issue to **[boundary]**. I mitigated immediate
> impact with **[containment]**, fixed **[root cause]**, validated with **[reproduction and regression
> test]**, and added **[prevention]**. The lesson was not ‘monitor more’; it was **[specific missing
> invariant or ownership gap]**.”

**Follow-ups.** Why did the existing alert miss it? What evidence could have falsified your diagnosis?
What was the blast radius? How did you decide rollback versus forward fix?

**Trap.** Narrating every command. The story is the sequence of hypotheses and decisions.

<a id="d2-4-2"></a>
#### D2.4.2 · Answer “What happens at ten times the scale?”

**Status.** Commonly useful.

**Illustrative structure.**

> “I would not multiply every number by ten. I would identify the first regime change. At ten times
> traffic, **[queue/cache/network/labeling]** becomes the binding resource; at ten times model or data
> size, **[memory/communication/evaluation]** may change the architecture. I would load-test the
> predicted knee, measure tail rather than mean behavior, and define the degradation mode. The design
> is acceptable only if overload fails in a controlled way—admission control, a smaller fallback, or
> delayed batch processing—rather than corrupting results.”

**Follow-ups.** Which estimate is least certain? What is the cheapest load test? What becomes an
organizational bottleneck rather than a compute bottleneck?

**Trap.** Answering “add more machines” without identifying state, coordination, or cost.

---

<a id="d2-5"></a>
### D2.5 · Establish result credibility and future work

Prepare a **claim–evidence boundary** for every headline result:

- exact claim;
- evaluation population;
- baseline fairness;
- variance or confidence information;
- ablations that support mechanism;
- negative and neutral slices;
- known contamination or measurement risk;
- what was observed after deployment, if anything;
- what remains unknown.

“We ran three seeds” is not reliability by itself. Stronger evidence pre-registers the primary
comparison, uses genuinely independent runs or experimental units, reports an effect interval rather
than only a mean, and shows task-level or slice-level variation. Seeds address one source of training
randomness; they do not repair a dependent test set, evaluator bias, or a claim that changes after the
results. Equally, do not decorate a project with statistical language it did not actually use.

End with future work that follows from the limitation. “Try larger models” is a budget request.
“The method assumes calibrated uncertainty, so I would first test whether recalibration transfers
under distribution shift; a failure would change the control policy” is a research plan.

<a id="d2-5-1"></a>
#### D2.5.1 · Answer “How do you know the result is real?”

**Status.** Commonly reported follow-up.

**Worksheet.**

> “I trust the narrow result because **[control]**, **[repeat/uncertainty]**, and **[independent
> validation]** address the three largest alternative explanations: **[alternatives]**. The strongest
> remaining threat is **[threat]**. The claim is therefore **[bounded claim]**, not **[overclaim]**.
> If I had one more week, I would run **[test]** because its result would most change that boundary.”

**Practice follow-ups**

- “Was the baseline tuned equally?”
- “How many seeds or independent units?”
- “Could the evaluator prefer your method's style?”
- “What result did you omit from the headline?”

**Trap.** Answering with volume—many benchmarks, many runs—without identifying which alternative
explanations they eliminate.

<a id="d2-5-2"></a>
#### D2.5.2 · Answer “What would you do differently?”

**Status.** Commonly reported deep-dive question.

**Illustrative structure.**

> “I would change **[specific early decision]**, not the part that merely produced a bad outcome. At
> the time I assumed **[assumption]** and did not instrument **[signal]**. That made us spend
> **[disclosable cost or qualitative consequence]** before learning **[fact]**. Now I would run
> **[cheaper test]** first and set **[stop rule]**. The underlying lesson I have reused is
> **[decision principle]**.”

**Follow-ups.** Why did you not know then? What process changed afterward? Give a later example where
you applied the lesson.

**Trap.** A disguised boast—“I would have scaled sooner”—that contains no real correction.

Nathan Lambert's first-hand candidate advice to invest in the job talk because it communicates a
vision and story applies equally here [[R12]](#ref-r12). The story is not marketing varnish. It is
the causal structure that lets another researcher inspect your judgment.

---

<a id="section-d3"></a>

## D3 · Research taste and the paper discussion

A paper round is not a memory test about the PDF. It asks whether you can turn a claim into a
research program: identify what evidence bears on it, find the weakest assumption, and choose the
next experiment.

Use five lenses:

1. **Claim:** what is asserted, and at what scope?
2. **Evidence:** which result supports which part of the claim?
3. **Mechanism:** what explanation predicts something not already plotted?
4. **Boundary:** where should it stop working?
5. **Next decision:** what one experiment most changes belief or action?

Keep empirical findings separate from the author's interpretation. A result can be real while the
mechanism is wrong; a mechanism can be plausible while the reported experiment is too weak to test it.

---

<a id="d3-1"></a>
### D3.1 · Read to the depth the conversation needs

Keshav's three-pass method is a useful time allocator: first recover category, context, correctness,
contributions, and clarity; then inspect figures and evidence; finally attempt a virtual
reimplementation and challenge assumptions [[R20]](#ref-r20). For interview preparation, map it to
three products:

| Pass | Product | You should be able to say |
|---|---|---|
| 1 | claim card | problem, headline claim, comparison, scope |
| 2 | evidence map | which figure/ablation supports each claim |
| 3 | reproduction plan | hidden choices, dependencies, failure points |

Do not perform a third pass on every paper. Maintain a familiarity ledger: title-only, skimmed,
discussable, reproduced, or built upon. Jaiswal's public preparation notes model this calibrated
inventory explicitly [[R14]](#ref-r14).

<a id="d3-1-1"></a>
#### D3.1.1 · Summarize a paper in ninety seconds

**Status.** Reported paper-round shape; the time box is a practice device.

**Answer shape.**

> “The paper asks **[question]**. Its main claim is **[one sentence, bounded]**. The key change over
> **[strongest relevant baseline]** is **[mechanism or system choice]**. The strongest evidence is
> **[figure/experiment]**, because it isolates **[factor]**. The result matters if
> **[scope/use case]**. My main reservation is **[assumption or measurement threat]**, so the next
> experiment I would run is **[discriminating test]**.”

**Practice follow-ups**

- “What did you omit?”
- “Is the contribution an objective, algorithm, system, dataset, or measurement result?”
- “What baseline would make the claim disappear?”
- “Would you build on it?”

**Traps**

- Replaying the abstract section by section.
- Calling every item in the contribution list a separate conceptual contribution.
- Giving criticism before establishing that you understood the strongest claim.

<a id="d3-1-2"></a>
#### D3.1.2 · Read an unfamiliar paper under a short deadline

**Status.** Commonly useful.

**Prompt.** You receive a paper thirty minutes before the discussion.

**Answer shape.**

> “I would spend five minutes on the title, abstract, introduction, conclusion, headings, and
> references to recover the claim and context. Then I would inspect every figure and table, writing
> the comparison, axes, uncertainty, and takeaway in my own words. I would spend the remaining time
> on the objective, strongest baseline, and one central experiment rather than reading linearly. I
> would enter the interview with a claim card, two questions, one plausible failure mode, and an
> explicit list of details I did not verify.”

**Follow-ups.** What if there are no error bars? What if the method section is central? What if you
lack prerequisite knowledge?

**Trap.** Spending twenty minutes decoding the first equation and never reaching the evidence.

---

<a id="d3-2"></a>
### D3.2 · Critique the claim, not the paper's aesthetics

A useful critique names an alternative explanation and a test. “The datasets are small” is a concern.
“The gain may come from a domain-specific preprocessing step because all positive datasets share that
format; evaluate on a matched dataset without the step” is research.

Audit in this order:

1. **Construct:** does the metric measure the thing named in the claim?
2. **Comparison:** is the strongest baseline present and equally tuned?
3. **Budget:** are data, compute, latency, parameters, and test-time work matched?
4. **Dependence:** are examples, users, tasks, and seeds truly independent units?
5. **Scope:** do the datasets and conditions support the generalization in the title?
6. **Uncertainty:** is variance visible, and were many choices tried?
7. **Failure evidence:** are negative slices, costs, and safety effects reported?

The NeurIPS paper checklist is a useful external audit because it asks authors to connect claims to
assumptions, uncertainty, compute, and reproducibility details [[R21]](#ref-r21).

<a id="d3-2-1"></a>
#### D3.2.1 · Give one strong criticism

**Status.** Commonly useful.

**Prompt.** “What is the weakest part of this paper?”

**Illustrative structure.**

> “The weakest link is not that the evaluation is small in the abstract; it is that the evaluator
> shares the same model family as the method's teacher. That creates correlated preference for its
> style, so the measured gain may not reflect user quality. I would keep the outputs fixed and rerun
> evaluation with blinded humans plus an independent judge family. If the ranking survives, my
> concern falls substantially.”

**Practice follow-ups**

- “How serious is it—fatal or a limitation?”
- “Why did the authors make this choice?”
- “What result would change your mind?”
- “What is the strongest defense of the paper?”

**Trap.** Maximizing the number of complaints. Research taste is prioritizing the one that most
changes the claim.

<a id="d3-2-2"></a>
#### D3.2.2 · Steelman before disagreeing

**Status.** Commonly useful.

**Answer shape.**

> “The strongest version of the authors' position is **[claim under favorable assumptions]**. The
> design is compelling because **[reason]**, and Figure **[X]** rules out **[alternative]**. My
> disagreement begins at **[specific extrapolation]**: the evidence establishes **[narrow claim]**,
> while the discussion assumes **[broader claim]**. I would preserve the method and narrow the
> conclusion unless **[new evidence]** appears.”

**Follow-ups.** Defend the paper against your own criticism; identify a limitation that does not
matter for the intended use.

**Trap.** Treating generosity as agreement. A steelman makes the disagreement more precise.

---

<a id="d3-3"></a>
### D3.3 · Design a reproduction before designing an extension

A reproduction has levels:

1. **Artifact reproduction:** run released code and recover the table.
2. **Independent reimplementation:** reconstruct the method from the paper.
3. **Robustness reproduction:** vary seeds, environments, versions, and data samples.
4. **Conceptual reproduction:** test the claim in a different but relevant setting.

State which one you mean. Reproducing a number from the authors' container tests packaging, not the
full scientific claim.

Build a reproduction manifest:

- exact claim and acceptance tolerance;
- code, model, data, and license availability;
- preprocessing and split construction;
- hidden external services or judge versions;
- compute and wall-clock budget;
- seed and variance plan;
- expected intermediate invariants;
- stopping and escalation rules;
- divergence log.

<a id="d3-3-1"></a>
#### D3.3.1 · Plan a one-week reproduction

**Status.** Commonly useful.

**Answer shape.**

> “In one week I would target the central claim, not every table. Day one is an environment and data
> manifest plus a tiny smoke run. Day two reproduces one baseline and checks intermediate invariants.
> Days three and four run the proposed method and matched baseline under the same budget. Day five is
> variance and one robustness perturbation. I reserve the final time to audit divergences and write
> what reproduced, what did not, and whether the claim boundary changed. I would predefine an
> acceptable numerical tolerance rather than deciding after seeing the result.”

**Practice follow-ups**

- “The released checkpoint matches, training does not.”
- “The API model has changed.”
- “You cannot afford the full scale.”
- “The code has a preprocessing bug.”

**Traps**

- Starting the largest run before validating a tiny end-to-end path.
- Quietly modifying the method until the number matches.
- Calling environment drift a failure of the scientific idea without localization.

<a id="d3-3-2"></a>
#### D3.3.2 · Handle a failed reproduction

**Status.** Commonly useful.

**Answer shape.**

> “I would report the failure at the narrowest layer I can defend. First verify the artifact,
> environment, data split, and intermediate values. Then compare our divergence point with the
> authors' logs. I would contact the authors with a minimal reproduction and exact versions, not an
> accusation. If the result remains different, I would distinguish ‘the artifact is not portable,’
> ‘the reported setting is underspecified,’ and ‘the claim does not replicate.’ Those are different
> conclusions.”

**Follow-ups.** When do you stop? Would you publish a negative result? How do you avoid confirmation
bias if you dislike the paper?

**Trap.** Treating agreement as success and disagreement as author error. Both outcomes require audit.

---

<a id="d3-4"></a>
### D3.4 · Use ablations to test mechanisms

An ablation should answer a question, not merely subtract a module.

Four useful forms:

- **necessity:** remove X; does the effect disappear?
- **sufficiency:** add only X to the baseline; does the effect appear?
- **dose response:** vary X continuously; does the predicted shape emerge?
- **substitution:** replace X with a simpler mechanism that produces the same proposed effect.

Match training budget and tuning effort. If removing a component also reduces parameters, data, or
compute, the ablation confounds mechanism with capacity. Include interactions when components may
only work together; one-at-a-time deletion cannot reveal that.

<a id="d3-4-1"></a>
#### D3.4.1 · Design the decisive ablation

**Status.** Commonly useful.

**Prompt.** A method has retrieval, reflection, and a verifier; the full system wins.

**Answer shape.**

> “The first question is not which component has the largest deletion drop. It is which causal claim
> the paper makes. If the claim is that reflection uses verifier feedback to repair retrieval errors,
> the key test is an interaction: retrieval plus verifier without reflection, retrieval plus
> reflection with shuffled verifier feedback, and the full combination. Shuffling preserves message
> volume while destroying information. If the full system still wins only with informative feedback,
> that supports the proposed pathway.”

**Follow-ups.** Match extra tokens; use an oracle verifier; test false feedback; inspect which errors
are repaired.

**Trap.** Reporting three component deletions that all change compute and calling the mechanism
isolated.

<a id="d3-4-2"></a>
#### D3.4.2 · Interpret an ablation with no drop

**Status.** Commonly useful.

**Answer shape.**

> “No drop does not immediately prove the component is useless. It may be redundant at this scale,
> compensated by retraining, active only on a rare slice, or measured by an insensitive metric. I
> would check whether removal changed compute or optimization, inspect the predicted slice, and try a
> dose response. If the component still has no measurable effect under conditions where the mechanism
> predicts one, I would remove it and narrow the paper's claim.”

**Trap.** Inventing endless rescue hypotheses. State in advance how many checks the component gets
before it loses.

---

<a id="d3-5"></a>
### D3.5 · Judge novelty, impact, and the next experiment

Novelty is not one axis:

- new capability or problem;
- new mechanism or objective;
- new evidence that changes a belief;
- new system combination that changes feasibility;
- new dataset, measurement, or negative result.

Impact is also not the benchmark delta. Ask:

- Does it unlock a previously infeasible regime?
- Does it survive replacement by a stronger baseline?
- Is the gain on a binding constraint?
- Can others adopt it at reasonable cost?
- Does it change what researchers build or believe?

A simple method can be high impact; a technically novel method can solve a non-binding problem.

<a id="d3-5-1"></a>
#### D3.5.1 · Decide whether a paper is novel and important

**Status.** Commonly useful.

**Answer shape.**

> “I would separate novelty from importance. The novel piece is **[specific delta from closest
> work]**. Its importance depends on **[binding constraint or changed belief]**. Right now the evidence
> supports **[scope]**, but impact is uncertain because **[adoption cost / stronger baseline / scale
> question]**. The experiment that would most update me is **[test]**. A positive result would move it
> from an interesting technique to a changed operating point.”

**Follow-ups.** Name the closest prior work; say whether the novelty is technical or empirical; judge
what the field may remember in three years.

**Trap.** Using “first” as a synonym for useful, or citation count as a substitute for causal impact.

<a id="d3-5-2"></a>
#### D3.5.2 · Discuss a paper you have not read

**Status.** Commonly useful.

**Illustrative spoken answer.**

> “I have not read that paper, so I do not want to invent its contribution. From the title and what
> you said, I think the underlying question is **[restate]**. The baseline I would compare against is
> **[baseline]**, and the result I would need before believing the broad claim is **[evidence]**. If
> you give me the core mechanism, I can reason through likely trade-offs and propose an ablation.”

**Practice follow-ups**

- The interviewer supplies the mechanism; update your model explicitly.
- Compare it with work you do know.
- State what prerequisite you would read first.

**Trap.** Name-dropping adjacent papers to hide that you do not know this one. Calibrated curiosity is
stronger than counterfeit familiarity.

<a id="d3-5-3"></a>
#### D3.5.3 · Propose a future research agenda and the first ninety days

**Status.** Commonly useful research-direction prompt.

**Prompt.** “What would you work on here? What changes with unlimited compute, and what would you do
in your first ninety days?”

**Answer shape.**

> “My current thesis is **[bounded research question]** because **[evidence and open bottleneck]**.
> Unlimited compute would let me test **[scale-sensitive hypothesis]**, but it would not remove data
> validity, evaluation, safety, or coordination constraints; I would still start with
> **[discriminating smaller experiment]**. In the first thirty days I would learn the team's actual
> assets and reproduce **[baseline]**. By day sixty I would align on one high-information wedge and
> its stop rule. By day ninety I would aim to deliver **[reproducible artifact or result]**, not claim
> a finished agenda. I would change direction if **[observable evidence]**.”

**Practice follow-ups.** Name the first experiment; explain why this team is necessary; cut the
compute budget by one hundred; identify the collaborator or capability the plan depends on; say what
result would kill the agenda.

**Trap.** Treating unlimited compute as unlimited evidence. More runs cannot repair the wrong
construct, inaccessible data, or an untestable claim.

---

<a id="section-d4"></a>

## D4 · Open-ended ML and LLM system design

An ML system-design answer is not an architecture diagram with a model box in the middle. It is a
chain of contracts:

`user decision → data → objective → model → training → evaluation → deployment → monitoring`

Every arrow can break while the adjacent boxes look healthy. A retriever may optimize recall while
the generator needs calibrated evidence; an offline evaluator may reward verbosity while users need
fast action; a training pipeline may compute the right feature differently from serving. Hidden
Technical Debt in Machine Learning Systems describes these entanglements as data dependencies,
feedback loops, configuration debt, and undeclared consumers [[R24]](#ref-r24).

Start end-to-end, then choose one bottleneck to deepen. A strong interview is not the one that names
the most components. It is the one where each component exists because of an explicit requirement and
has an observable failure mode.

For the underlying mechanisms, cross-check Part I
[A8 · Inference and serving](/blog/2026/interview-knowledge/#section-a8),
[A9 · Data](/blog/2026/interview-knowledge/#section-a9), and
[A10 · Estimation](/blog/2026/interview-knowledge/#section-a10). This section practises selecting and
connecting those mechanisms under an open-ended requirement.

---

<a id="d4-1"></a>
### D4.1 · Turn the vague prompt into an end-to-end contract

Use this opening sequence:

1. **User and action:** what output changes whose decision?
2. **Error asymmetry:** what do false positives, false negatives, delay, and abstention cost?
3. **Traffic and horizon:** volume, tail, context, freshness, and interaction length.
4. **Success and guardrails:** offline proxy, online outcome, safety, cost, and latency.
5. **Baseline and control surface:** current process; what may change.
6. **Then** sketch data through monitoring.

Chip Huyen's interview book emphasizes that ML system questions require choices about objectives,
data, metrics, and deployment, not just model families [[R16]](#ref-r16). The order above makes those
choices visible before implementation detail consumes the hour.

<a id="d4-1-1"></a>
#### D4.1.1 · Open “design an enterprise support assistant”

**Status.** Commonly useful system-design prompt.

**Illustrative spoken opening.**

> “I’ll assume employees ask policy and product questions over a private corpus, and the assistant may
> recommend actions but cannot execute them. A wrong confident answer is worse than an abstention,
> freshness matters, and citations are required. The primary outcome is resolved tasks with verified
> evidence; severe unsupported claims and data-access violations are hard guardrails; p95 time to
> useful answer and cost per resolved task are operational metrics. The baseline is search plus a
> document viewer. I’ll walk from corpus and permissions through retrieval, generation, evaluation,
> serving, and monitoring, then go deeper on evidence grounding.”

**Practice follow-ups**

- “Why not fine-tune?”
- “How do permissions propagate into retrieval and caches?”
- “What if the corpus cannot answer?”
- “How do you measure resolution without waiting for a support ticket?”

**Traps**

- Starting with vector database selection.
- Treating citations as proof that the cited text entails the answer.
- Ignoring that access control is part of the data path, not a final filter.

<a id="d4-1-2"></a>
#### D4.1.2 · Draw the minimum viable pipeline

**Status.** Commonly useful.

**Answer shape.**

> “For the first version: ingest versioned documents with ACL metadata; hybrid retrieve and rerank;
> pass a small evidence set to the model; require answer spans to cite document versions; abstain when
> retrieval or entailment confidence is low; log query, authorized document IDs, model version,
> citations, latency, cost, and feedback. Offline evaluation uses frozen time-aware queries with
> answerability labels. Launch in shadow mode, then a low-risk cohort. I would not add agents,
> fine-tuning, or a learned router until the baseline reveals a binding failure.”

**Follow-ups.** Index freshness; deletion; multilingual documents; long tables; feedback loops.

**Trap.** Calling this “simple” while leaving logging, versioning, permissions, and abstention out.

---

<a id="d4-2"></a>
### D4.2 · Make the data and objective do real work

Discuss data as a lifecycle:

- source and consent;
- inclusion/exclusion rules;
- unit of independence;
- labeling and disagreement;
- deduplication and contamination;
- temporal split and freshness;
- hard-negative and tail coverage;
- versioning, deletion, and lineage;
- feedback after launch.

Then define the training objective as an approximation to a decision. Ask what behavior it rewards,
what it leaves indifferent, and how it can be gamed. A loss can improve while the product worsens
because the label, sampling distribution, or aggregation unit is wrong.

<a id="d4-2-1"></a>
#### D4.2.1 · Design data for a coding agent

**Status.** Commonly useful.

**Answer shape.**

> “The unit is a repository state plus issue, not an isolated code snippet. I need the environment,
> tests, dependency lock, and a clean reset. I would split by repository and time to reduce
> contamination, keep hidden tests outside the agent's observation, and label more than final pass:
> compile status, tests changed, files touched, and policy violations. Training data should include
> failed and recovered trajectories, not only clean demonstrations. The first baseline is retrieval
> of relevant files plus a single patch attempt; multi-step planning earns its complexity only if it
> improves hidden-test success at controlled cost.”

**Practice follow-ups**

- “Public repositories may be in pretraining.”
- “The agent edits tests.”
- “A passing patch is ugly or insecure.”
- “Long tasks make iteration too slow.”

**Traps**

- Randomly splitting issues from the same repository.
- Rewarding visible tests while letting the agent modify them.
- Counting generated tokens rather than successful, reviewable tasks.

<a id="d4-2-2"></a>
#### D4.2.2 · Detect an objective that invites reward hacking

**Status.** Commonly useful.

**Prompt.** “The agent's reward is the number of tests passed.”

**Illustrative answer.**

> “That objective is incomplete. It rewards deleting or weakening tests, hard-coding fixtures, and
> broad changes that pass locally but violate the issue. I would make the environment immutable where
> possible, use hidden tests, enforce a change policy, and add static and security checks. More
> importantly, I would audit high-reward trajectories before trusting the curve. The metric is an
> attack surface.”

**Follow-ups.** What if hidden tests leak? How do you reward partial progress? How do you prevent the
guardrails from dominating useful learning?

**Trap.** Solving reward hacking only by adding more weighted terms. Every new term creates another
boundary to exploit.

---

<a id="d4-3"></a>
### D4.3 · Choose model and training only after locating the bottleneck

Organize options by what they can fix:

- **prompt/context/tool contract:** instruction, formatting, and available information;
- **retrieval or tools:** missing or changing external knowledge;
- **supervised tuning:** repeated behavior and format from demonstrations;
- **preference or RL training:** choosing among model behaviors under a reward;
- **continued pretraining:** domain distribution and knowledge, at higher cost;
- **architecture/scale:** capacity or efficiency limits that cheaper layers cannot remove.

Then compare options at a fixed budget and state the reversal condition. “Fine-tune” is not a
decision until you specify data, objective, base policy, expected changed behavior, and evaluation.

<a id="d4-3-1"></a>
#### D4.3.1 · Decide RAG, fine-tuning, or both

**Status.** Commonly useful.

**Answer shape.**

> “I separate knowledge from behavior. If facts change, must be cited, or have per-user permissions,
> retrieval is the default. If the model repeatedly mishandles format, tool choice, or domain
> language despite having the right evidence, supervised tuning may help. I would first test the
> generator with oracle evidence. If it succeeds, retrieval is the bottleneck; if it still fails in a
> systematic, teachable way, add tuning. Both are justified only when both failure classes are
> measured.”

**Practice follow-ups**

- “Can fine-tuning add knowledge?”
- “Can RAG teach style?”
- “What if retrieval raises latency?”
- “How do you update or delete knowledge?”

**Trap.** Presenting RAG versus fine-tuning as mutually exclusive brand choices rather than different
contracts.

<a id="d4-3-2"></a>
#### D4.3.2 · Propose a training ladder with stop rules

**Status.** Commonly useful.

**Answer shape.**

> “I would climb only when the measured failure survives the cheaper rung: prompt and tool contract,
> then better data/retrieval, then SFT, then preference or RL optimization, and only then larger-scale
> training or architecture change. Each rung has a go/no-go test. For example, SFT proceeds only if a
> small clean set produces a repeatable gain on the target slice without eroding general capability.
> This avoids spending a full training budget to repair an evaluation or context bug.”

**Follow-ups.** Interactions between rungs; catastrophic forgetting; stale preference data; when a
larger model is actually cheaper operationally.

**Trap.** Treating the ladder as universal. If the base model lacks the capability entirely, early
rungs may only hide the gap.

---

<a id="d4-4"></a>
### D4.4 · Resolve capability, latency, cost, and safety as a Pareto problem

Do not collapse all constraints into a weighted score before showing them separately. Build a small
frontier:

- quality by consequential slice;
- p50 and tail latency;
- throughput or goodput under SLO;
- cost per successful task, not merely per token;
- severe-error and policy-violation rates;
- human-review or escalation load;
- reversibility and blast radius.

Then select an operating point for the stated user. Common levers include model routing, cascades,
batching, caching, quantization, constrained tools, retrieval, early exit, and human escalation. Each
lever moves more than one axis.

<a id="d4-4-1"></a>
#### D4.4.1 · Cut cost in half without hiding quality loss

**Status.** Commonly useful.

**Answer shape.**

> “First I would decompose cost by prefill, decode, retrieval, external tools, and retries under the
> real traffic mix. Then I would rank reversible levers: remove wasted context, cache shared prefixes
> and retrieval, raise batching until the latency guardrail binds, route easy requests to a smaller
> model, and quantize only after slice evaluation. I would report cost per successfully resolved task,
> because a cheaper call that causes more retries is not cheaper. The final proposal includes the
> quality and safety movement for every lever.”

**Follow-ups.** Traffic is sparse; prompts are unique; quality labels arrive late; the smaller model
is miscalibrated.

**Trap.** Promising a percentage from generic optimization folklore before measuring the cost stack.

<a id="d4-4-2"></a>
#### D4.4.2 · Add human review without creating a bottleneck

**Status.** Commonly useful.

**Answer shape.**

> “Human review is a queue, not a magic safety box. I would define which decisions are reviewable,
> route by calibrated risk and consequence, give reviewers evidence and a clear action, and measure
> agreement, handling time, overrides, and missed severe cases. Capacity and fatigue set a hard
> budget, so the system must abstain or degrade safely when the queue saturates. Review outcomes can
> become training data only after correcting for selective sampling.”

**Follow-ups.** Reviewer disagreement; adversarial users; privacy; delayed labels; automation bias.

**Trap.** Sending every uncertain case to people. That is neither scalable nor necessarily safer.

<a id="d4-4-3"></a>
#### D4.4.3 · Size capacity on the back of an envelope

**Status.** Commonly useful system-design drill.

**Prompt.** “Estimate the first serving footprint before you have a load test.”

**Answer shape.**

> “I would expose assumptions rather than guess a fleet count. Suppose arrival rate is
> **[requests/second]**, mean input and output are **[tokens]**, and service time is **[seconds]**.
> Little's law gives roughly `arrival rate × service time` in-flight requests. Prefill demand is
> `arrival rate × input tokens`; decode demand is `arrival rate × output tokens`; peak concurrency
> times per-sequence KV bytes bounds cache memory. I divide each demand by benchmarked per-replica
> goodput at the target tail-latency SLO, then add headroom for burst, failure, and rollout. I size
> tool quotas, sandbox slots, log ingestion, and human approvals separately because GPU capacity does
> not remove those queues. The first load test targets whichever estimate has the widest interval.”

**Practice follow-ups.** Traffic is bursty; sessions last hours; retries double tool work; prefill and
decode are disaggregated; one external API has a strict rate limit; quality requires a larger model.

**Trap.** Reporting a precise machine count without model-specific throughput, KV layout, utilization,
or a tail-latency target. Part I [A10](/blog/2026/interview-knowledge/#section-a10) supplies the
parameter, FLOP, memory, and KV-cache estimation drills.

---

<a id="d4-5"></a>
### D4.5 · Design evaluation, rollout, and monitoring together

Offline evaluation, online experiments, and monitoring answer different questions:

- **offline:** can we detect likely improvement cheaply and repeatedly?
- **online:** does the system improve the real decision under live behavior?
- **monitoring:** is the deployed contract still valid?

Create evaluation sets by source and purpose: regression, capability, adversarial safety, temporal
holdout, and a rotating fresh set. Freeze a core set for comparability, but keep fresh data to detect
contamination and drift. Measure the evaluator itself through agreement, stability, and targeted
human audits.

Rollout should move by reversibility: replay → shadow → internal users → low-risk cohort → broader
traffic. For each stage, name the entry condition, evidence window, owner, and rollback.

<a id="d4-5-1"></a>
#### D4.5.1 · Monitor a system whose labels arrive weeks later

**Status.** Commonly useful.

**Answer shape.**

> “I would separate fast proxies from delayed truth. Fast signals include input drift, abstention,
> retrieval coverage, judge disagreement, policy violations, latency, and user correction behavior.
> They trigger investigation, not a declaration of quality. When delayed labels arrive, I backfill
> cohort metrics and estimate which proxy actually predicted harm. I also maintain a small continuously
> labeled sentinel sample so the system is not blind for weeks.”

**Practice follow-ups**

- “The sentinel sample is biased.”
- “User feedback is sparse and selected.”
- “The judge and model drift together.”
- “What automatically rolls back?”

**Trap.** Calling an embedding-distance alert “model quality.” It is evidence of change, not evidence
of harm.

<a id="d4-5-2"></a>
#### D4.5.2 · Design the rollback before launch

**Status.** Commonly useful.

**Answer shape.**

> “I would version weights, prompts, retrieval index, feature definitions, tool policy, and evaluator.
> A rollback must restore a compatible bundle, not only old weights. I would decide whether ongoing
> sessions drain or switch, preserve enough sampled traces for diagnosis, and test rollback as part of
> release. The automatic triggers are limited to high-confidence operational or severe-safety
> failures; ambiguous quality movement pages an owner and freezes expansion.”

**Follow-ups.** Database migrations; non-reversible actions; corrupted feedback; multi-region
version skew.

**Trap.** A rollback command that has never been exercised. Untested recovery is a hypothesis.

The AI Engineering Field Guide is useful for seeing the range of reported ML-design shapes, but its
own variation across companies reinforces the rule: use the recruiter's actual loop and the role's
constraints as the source of truth. It is a living repository, accessed 2026-08-13
[[R15]](#ref-r15).

---

<a id="d4-6"></a>
### D4.6 · Design a long-running coding and research agent as a durable system

The model is one replaceable component. The product is a control plane around untrusted model output,
external data, tools, and side effects. Anthropic's Managed Agents architecture separates a durable
append-only session log, a replaceable harness, and isolated sandboxes; its long-running harness work
uses explicit progress artifacts across sessions [[R27]](#ref-r27) [[R28]](#ref-r28). These are
useful design examples, not the only implementation.

Protocol boundaries do not erase system obligations. MCP standardizes host/client/server connections
to resources and tools, while A2A supports communication with independent agents
[[R29]](#ref-r29) [[R30]](#ref-r30). Neither protocol supplies your authorization model,
idempotency, release policy, or task-level correctness. NIST's agent identity concept paper frames
identification, authorization, auditing, non-repudiation, prompt injection, and least privilege as
deployment concerns and open standardization questions, not as guarantees supplied by a finished
standard [[R31]](#ref-r31).

<a id="d4-6-1"></a>
#### D4.6.1 · Design a long-running coding and research agent

**Status.** Commonly useful agent-system design prompt.

**Prompt.** “Design an agent that works for hours across repositories, papers, tests, and external
services, can recover from failure, and may eventually propose or perform consequential actions.”

**Illustrative spoken answer.**

> “I would separate task state from model context. A durable task ID owns an append-only event log:
> user intent, plan revisions, model and prompt version, tool calls and results, approvals, budget,
> artifacts, checkpoints, and side-effect receipts. Stateless harness workers lease the next step and
> can resume from the log after a crash. Code and untrusted content run in disposable sandboxes with
> no ambient credentials; typed tool proxies hold narrow, short-lived identity outside the sandbox.
>
> “The harness enforces wall-clock, token, tool, retry, and cost budgets. Reads may retry with bounded
> backoff; writes require an idempotency key or a read-before-write reconciliation plan. A timeout is
> unknown outcome, not permission to repeat. High-impact actions—publishing, messaging, merging,
> spending, changing access, or sending data—cross an explicit approval boundary.
>
> “Observability links every model turn and tool call to task, principal, sandbox, artifact version,
> latency, cost, and resulting state. Completion is not the model saying ‘done’: an isolated verifier
> gets the original requirement and produced artifacts, runs hidden tests or independent evidence
> checks, and controls a release gate. Reconciliation compares intended with observed external state
> before close. The minimum baseline is one agent, one sandbox, typed tools, a durable log, and a
> verifier; parallel agents earn their coordination cost only on measured bottlenecks.”

**Practice follow-ups**

- The harness crashes after an external write but before recording success.
- A tool returns an untrusted webpage containing instructions to reveal credentials.
- The verifier shares the same model family and failure mode.
- A user changes the goal while two workers are active.
- One task exceeds budget but has produced a promising partial artifact.
- A remote specialist is exposed through A2A and tools through MCP.

**Traps**

- Treating chat history as the durable source of truth.
- Giving the sandbox broad cloud or source-control credentials.
- Retrying side effects as though they were pure model calls.
- Letting self-reflection serve as the only verifier or release gate.
- Drawing a multi-agent graph before defining ownership, state, and recovery.

<a id="d4-6-2"></a>
#### D4.6.2 · Choose an API or computer use when no stable API exists

**Status.** Commonly useful follow-up.

**Answer shape.**

> “I prefer a stable API when it exists because typed requests, scoped authorization, idempotency
> keys, structured errors, and audit logs make effects easier to bound and reconcile. If no stable
> API exists, computer use is an adapter of last resort: run it in an isolated browser or VM, expose
> only approved sites and accounts, treat page content as untrusted input, and require confirmation
> for high-impact actions. Before acting, capture the target and expected postcondition; afterward,
> verify the actual external state. Pixel actions are not inherently idempotent, so a timeout enters
> reconciliation rather than blind replay.”

OpenAI's official computer-use guide recommends an isolated browser or VM, human review for
high-impact actions, and treating page content as untrusted [[R32]](#ref-r32).

**Practice follow-ups.** The site redesigns; an upload partially succeeds; CAPTCHA or MFA appears;
the page contains indirect prompt injection; the same action is available through an undocumented
endpoint.

**Trap.** Calling computer use “just another tool” while ignoring weaker schemas, ambiguous outcomes,
UI drift, and a larger injection surface.

---

<a id="section-d5"></a>

## D5 · Experiment design and scientific reasoning

An experiment is a decision instrument. Start with the decision and estimand—not the model:

- **Decision:** what will change if the evidence is positive, negative, or ambiguous?
- **Unit:** user, task, conversation, document, run, seed, cluster, or time window?
- **Intervention:** what exactly differs?
- **Counterfactual:** what would have happened to the same population without it?
- **Estimand:** average effect, tail effect, treatment-on-the-treated, cost-adjusted effect?
- **Validity:** what could create the same observation without the claimed mechanism?

The NIST/SEMATECH statistical handbook is useful background for controls, variance, power, and
experimental design [[R23]](#ref-r23). In an interview, however, formulas are secondary to identifying
the independent unit, the leakage path, and the decision threshold.

Cross-check Part I
[A11.7 · Designing an eval](/blog/2026/interview-knowledge/#a11-7) and
[A11.12 · A/B testing and online metrics](/blog/2026/interview-knowledge/#a11-12), plus Part II
[C6 · Statistics and estimation](/blog/2026/interview-coding/#section-c6). This section turns those
tools into spoken experimental decisions.

---

<a id="d5-1"></a>
### D5.1 · Turn a claim into an estimand and control

“Method A improves reasoning” is not yet testable. A testable version might be:

> On prompts drawn from a defined post-cutoff distribution, at a fixed inference-compute budget,
> Method A changes pass rate by a specified amount relative to a tuned baseline, while severe-format
> violations remain below a guardrail.

That sentence fixes population, budget, comparison, outcome, and guardrail. It also exposes choices
that can reverse the conclusion.

A post-cutoff sample only reduces known exposure risk. It does **not** prove absence of
benchmark-specific post-training, retrieval exposure during the task, or undisclosed training data.
Treat the cutoff as one control alongside provenance review, access-path checks, fresh or private
tasks, overlap audits, and explicit uncertainty about what cannot be observed.

Choose a control that answers the actual causal question:

- no intervention;
- current production policy;
- active placebo with equal tokens or latency;
- strongest tuned alternative;
- randomized order or blinded evaluation;
- matched-compute or matched-data control.

For an online randomized experiment, default to the **intention-to-treat (ITT)** effect of assignment.
Conditioning on who actually complied can break randomization. A treatment-on-the-treated or
complier-average causal effect (CACE) needs assignment as an instrument and explicit assumptions:
assignment changes treatment uptake (**relevance**), affects outcome only through treatment
(**exclusion**), does not make anyone systematically move opposite the assignment
(**monotonicity**), and is independent of potential outcomes by design. If these are not defensible,
report ITT and compliance separately rather than relabeling a selected comparison as causal.

<a id="d5-1-1"></a>
#### D5.1.1 · Evaluate a new reasoning-time strategy

**Status.** Commonly useful.

**Answer shape.**

> “The decision is whether to spend extra inference compute on this strategy. I would sample prompts
> from the target distribution after the training-data cutoff, randomize strategy versus a
> matched-compute baseline, and use task-level success as the unit. Primary outcome is solved tasks;
> latency, token cost, and severe invalid outputs are guardrails. A plain greedy baseline is
> insufficient if the new method uses four times the budget; compare against best-of-N or another use
> of the same compute. I would predefine the minimum gain that justifies the added cost.”

**Practice follow-ups**

- “Some tasks have no verifier.”
- “The strategy helps only hard prompts.”
- “Compute usage is adaptive.”
- “Prompts share templates.”

**Traps**

- Comparing unequal test-time budgets and attributing the entire gain to the algorithm.
- Treating multiple samples from one prompt as independent tasks.
- Selecting “hard prompts” based on the same model outcome being evaluated.

<a id="d5-1-2"></a>
#### D5.1.2 · Choose the unit of randomization

**Status.** Commonly useful.

**Prompt.** You want to A/B test an assistant that users visit repeatedly.

**Answer shape.**

> “I would randomize by user, not request, if treatment can change future behavior or expectations.
> Request-level randomization risks contamination within a user and an incoherent experience. The
> outcome may be measured per session, but inference must account for clustering by user. If users
> collaborate or share workspaces, the workspace may be the true interference unit.”

**Follow-ups.** New versus existing users; network effects; crossover design; cookie loss.

**Trap.** Confusing the row in the event table with the independent experimental unit.

---

<a id="d5-2"></a>
### D5.2 · Find leakage and confounding before optimizing

Leakage means the experiment exposes information that would not be available at the intended decision
time or allows train and evaluation units to communicate. Common paths:

- duplicate or near-duplicate examples across splits;
- future labels or post-outcome features;
- users, repositories, documents, or templates shared across splits;
- teacher or evaluator trained on the benchmark;
- prompt selection based on test performance;
- retrieval over the answer or benchmark explanations;
- repeated tuning on a nominal holdout;
- human raters recognizing treatment from style.

Confounding means treatment moves with another cause: model A uses more tokens, fresher data, a
different prompt, more tuning, or a different serving stack. Write a treatment manifest listing every
difference. If there are twelve, the experiment tests a system package, not one mechanism.

<a id="d5-2-1"></a>
#### D5.2.1 · Audit a suspiciously large offline gain

**Status.** Commonly useful.

**Answer shape.**

> “Before celebrating, I would freeze the run and inspect provenance. Check exact and semantic
> overlap, split by the highest-level entity such as repository or customer, verify label timestamps,
> and confirm retrieval cannot see evaluation answers. Then compare data cutoff and evaluator family.
> I would rerun on a fresh post-cutoff sample before changing the model, while stating that the cutoff
> cannot rule out benchmark-specific post-training, retrieval exposure, or undisclosed data. A large
> gain is a reason to increase skepticism, not reduce it.”

**Practice follow-ups**

- “Semantic deduplication has false positives.”
- “The benchmark is public and likely in pretraining.”
- “No fresh labels exist.”
- “The evaluator wrote some training data.”

**Traps**

- Running more seeds on contaminated data. Repetition reduces random error, not bias.
- Treating removal of exact duplicates as a complete contamination audit.

<a id="d5-2-2"></a>
#### D5.2.2 · Separate model change from serving change

**Status.** Commonly useful.

**Prompt.** A new model improves conversion after launch, but the rollout also changed latency and
UI copy.

**Answer shape.**

> “That rollout estimates the effect of the package, not the model. I would first decide whether the
> package effect is sufficient for the product decision. If mechanism matters, use a factorial or
> sequential design: hold UI and serving fixed while randomizing model, then test latency or copy
> separately. I would also check sample-ratio mismatch and exposure logging, because a latency change
> can alter who remains in the observed sample.”

**Follow-ups.** Interaction effects; limited traffic; cannot serve old model; novelty effect.

**Trap.** Retroactively attributing the package result to the component you like.

---

<a id="d5-3"></a>
### D5.3 · Reason about variance, power, and multiple comparisons

Power planning begins with an effect worth acting on. With enough traffic, a meaningless effect
becomes statistically detectable; with too little, a valuable effect remains ambiguous. State:

- minimum decision-relevant effect;
- baseline rate and expected variance;
- independent unit and clustering;
- desired false-positive and false-negative tolerance;
- planned duration and seasonality;
- stopping rule.

Variance reduction can come from better pairing, stratification, pre-experiment covariates, repeated
measures, or a lower-noise metric. More samples are not the only lever.

Multiple comparisons appear when you try many prompts, checkpoints, datasets, slices, metrics, seeds,
and stopping times, then report the best. Pre-specify one primary analysis; treat the rest as
exploratory, correct when formal claims require it, and validate selected findings on untouched data.

<a id="d5-3-1"></a>
#### D5.3.1 · Respond to “the result is not significant”

**Status.** Commonly useful.

**Answer shape.**

> “First I would avoid translating ‘not significant’ into ‘no effect.’ I would report the effect
> estimate and interval relative to the minimum useful effect. If the interval excludes any useful
> gain, that supports stopping. If it spans meaningful harm and benefit, the experiment is
> inconclusive; I would inspect variance, compliance, and whether more data is worth the decision
> value. If the estimate is tiny with a narrow interval, more traffic will not make it important.”

**Follow-ups.** Bayesian framing; equivalence or non-inferiority; sequential test; underpowered
subgroups.

**Trap.** Running until a conventional threshold appears without a valid sequential design.

<a id="d5-3-2"></a>
#### D5.3.2 · Handle twenty metrics and fifty slices

**Status.** Commonly useful.

**Answer shape.**

> “Before the run I would define one primary metric, a small set of guardrails, and a short list of
> confirmatory slices tied to explicit hypotheses. The remaining slices are diagnostic. If a surprise
> slice appears, I treat it as hypothesis generation and confirm it on fresh data. I would show the
> full family of tested outcomes rather than present the winner alone.”

**Follow-ups.** False-discovery control; hierarchical metrics; rare severe events; fairness slices
that cannot be dismissed as exploratory.

**Trap.** Correcting away safety obligations. Statistical multiplicity does not make a severe event
irrelevant; it changes how confidently you generalize from it.

---

<a id="d5-4"></a>
### D5.4 · Connect offline metrics to online decisions

Offline metrics are valuable when they are fast, stable, decomposable, and predictive of what matters
online. They fail when:

- the benchmark distribution differs from traffic;
- the metric rewards a proxy such as length or style;
- users adapt to treatment;
- system latency changes exposure;
- errors have asymmetric consequences;
- post-launch feedback changes future data.

Build a metric ladder:

1. component metric;
2. end-to-end task outcome;
3. user or business outcome;
4. safety and system guardrails;
5. long-term feedback effects.

Trustworthy Online Controlled Experiments emphasizes randomized online evidence, guardrails, and
diagnostics such as sample-ratio mismatch [[R22]](#ref-r22). Use online tests to validate the
offline–online link, not to rescue every weak offline idea.

<a id="d5-4-1"></a>
#### D5.4.1 · Explain an offline–online disagreement

**Status.** Commonly useful.

**Prompt.** Offline answer quality rises, but task completion falls online.

**Answer shape.**

> “I would not immediately choose one metric. I would enumerate mechanisms that predict the
> disagreement: added latency causes abandonment, answers are longer but less actionable, offline
> prompts omit the hard live tail, or the evaluator rewards style. First validate exposure and
> experiment integrity, then slice completion by latency, query type, and answer length. Replay live
> failures into the offline harness. The goal is to repair the measurement model, not merely choose
> the preferred result.”

**Follow-ups.** Online label is noisy; long-term retention versus short-term completion; treatment
changes query mix.

**Trap.** Declaring the benchmark useless without identifying the missing mechanism.

<a id="d5-4-2"></a>
#### D5.4.2 · Choose guardrails for an online LLM experiment

**Status.** Commonly useful.

**Answer shape.**

> “The primary metric follows the product decision, such as successful task completion. Guardrails
> cover severe unsupported claims, policy violations, privacy events, user corrections or
> escalations, p95 latency, error rate, and cost. For each rare severe event I report the exposure
> denominator, event count and rate, plus a one-sided upper confidence bound; zero observed events is
> not zero risk. I also keep case review rather than dilute severity into an average, and monitor
> sample-ratio mismatch and logging health so an apparently clean outcome is not built on a broken
> experiment.”

**Trap.** Having thirty guardrails with no escalation rule. Every guardrail needs an owner and
response.

---

<a id="d5-5"></a>
### D5.5 · Learn from negative and ambiguous results

A negative result can mean:

- the mechanism is wrong;
- the intervention was too weak;
- the implementation failed;
- the measurement is insensitive;
- the effect exists only in a slice;
- variance is too high;
- the baseline is stronger than expected.

Do not choose the explanation after seeing the outcome. Predefine sanity checks and positive controls.
A negative result becomes useful when it narrows the claim and changes resource allocation.

<a id="d5-5-1"></a>
#### D5.5.1 · Decide whether to stop after a negative result

**Status.** Commonly useful.

**Answer shape.**

> “I would ask whether the experiment had the capacity to see the predicted effect. Did the
> intervention move its intermediate target? Did a positive control register? Is the interval narrow
> enough to exclude a useful gain? If yes, I stop or narrow the hypothesis. If no, I allow one
> diagnostic experiment chosen in advance. I do not grant the idea unlimited retries because every
> failure can be explained away.”

**Follow-ups.** Sunk cost; senior sponsor; promising qualitative examples; publication bias.

**Trap.** Calling the run “a learning” without naming what belief or plan changed.

<a id="d5-5-2"></a>
#### D5.5.2 · Present a negative result constructively

**Status.** Commonly useful.

**Answer shape.**

> “The original claim predicted **[effect]** under **[conditions]**. With a validated intervention and
> sensitivity sufficient to detect **[minimum useful effect]**, we did not observe it. That rules out
> **[bounded claim]** in this regime, not the whole idea. The result saved **[next investment]** and
> exposed **[stronger baseline or hidden constraint]**. The remaining open case is **[specific
> boundary]**.”

**Trap.** Rebranding failure as success. The value comes from a defensible boundary, not positive
spin.

---

<a id="d5-6"></a>
### D5.6 · Design scaling experiments that identify regime changes

A scaling experiment is not “train small, medium, large and fit a line.” Decide what remains fixed:

- total compute;
- tokens per parameter;
- data quality and mixture;
- optimizer and schedule;
- architecture ratios;
- inference budget;
- evaluation contamination and difficulty.

Then distinguish interpolation from extrapolation. Small-scale rankings can reverse because of
optimization stability, data saturation, communication overhead, or emergent system bottlenecks.
Use pilot scales to estimate trends and diagnose invariants, but reserve an intermediate validation
point before the expensive endpoint.

<a id="d5-6-1"></a>
#### D5.6.1 · Decide whether an idea will survive scale

**Status.** Commonly useful.

**Answer shape.**

> “I would identify why scale could change the comparison. If the proposed gain is algorithmic, I
> want matched compute and data across several sizes, plus intermediate metrics tied to the mechanism.
> I would fit the trend without using the final validation scale, predict that point, then test the
> prediction. I would also measure systems cost, because a method with equal FLOPs can have worse
> communication or utilization. I scale further only if both the quality trend and operational trend
> support the target regime.”

**Practice follow-ups**

- “Only one large run is affordable.”
- “Hyperparameters do not transfer.”
- “The dataset saturates.”
- “The small model cannot express the mechanism.”

**Traps**

- Tuning each scale differently and then attributing the frontier to scale alone.
- Extrapolating through a known regime change.
- Selecting the scaling law after seeing the large point.

<a id="d5-6-2"></a>
#### D5.6.2 · Allocate one expensive run

**Status.** Commonly useful.

**Answer shape.**

> “I would spend the expensive run only after cheap runs have eliminated implementation, evaluation,
> and obvious hyperparameter failures. I would choose the run that most separates the leading
> hypotheses, not automatically the largest model. Before launch I write the predicted range, kill
> criteria, checkpoint and evaluation schedule, and what each outcome means. During the run I inspect
> predeclared health signals, not chase the test metric with ad hoc changes.”

**Trap.** Treating the large run as both experiment and production artifact, then becoming unable to
stop it when its premise fails.

---

<a id="section-d6"></a>

## D6 · Debugging and incident discussion

This section is about verbal localization. Part II owns code drills. Here the interviewer is watching
whether you can reduce uncertainty without causing a second incident.

Use the incident loop:

1. **Stabilize:** stop irreversible harm; preserve evidence.
2. **Verify:** confirm the symptom is real, not logging or aggregation.
3. **Bound:** first bad time, affected slices, versions, regions, ranks, and blast radius.
4. **Compare:** last-known-good versus first-known-bad; list every changed dependency.
5. **Hypothesize:** data, code/configuration, numerical/model, infrastructure, and measurement.
6. **Discriminate:** run the cheapest safe check that splits the leading hypotheses.
7. **Repair:** mitigate first, then fix the root cause.
8. **Validate:** replay, canary, regression test, and monitor delayed effects.
9. **Prevent:** restore an invariant, test, owner, alert, or recovery path.

Say which phase you are in. Jumping straight to a cause sounds fast but usually reveals anchoring.
Stas Bekman's engineering notes are useful here because they treat large-training failures as
observable shapes, preserve evidence around the first bad point, and distinguish recovery from root
cause [[R17]](#ref-r17).

For implementation practice, cross-check Part I
[A5.5 · Diagnosing training instability](/blog/2026/interview-knowledge/#a5-5) and Part II
[B9 · Debugging](/blog/2026/interview-coding/#section-b9). This section keeps the focus on verbal
localization, containment, and decision order.

---

<a id="d6-1"></a>
### D6.1 · Verify, bound, and preserve evidence

Before changing the system, ask:

- Is the alert computed correctly and from complete traffic?
- Does the symptom appear in an independent signal?
- What is the earliest bad event, not merely when the alert fired?
- Which cohorts are unaffected?
- What changed in code, data, configuration, dependencies, hardware, and traffic?
- Will rollback destroy evidence or make state incompatible?

Negative space is powerful. “Only one region,” “only long contexts,” or “only resumed jobs” can
remove entire hypothesis families.

<a id="d6-1-1"></a>
#### D6.1.1 · Open an incident answer

**Status.** Commonly useful.

**Prompt.** “Production quality suddenly dropped. What do you do?”

**Illustrative spoken opening.**

> “First I would separate containment from diagnosis. If users face high-consequence harm, route to
> the last-known-good bundle or disable the affected action while preserving sampled traces and
> versions. Then verify the drop with an independent metric and bound the first bad time, cohorts,
> regions, model versions, and request shapes. I would diff every change around that boundary before
> proposing causes. My initial buckets are traffic/data, model or prompt, serving/configuration,
> external dependencies, and evaluation. The first test should split several of those at once.”

**Practice follow-ups**

- “Rollback is expensive.”
- “The last model version is also bad.”
- “Only a proxy metric moved.”
- “No deployment happened.”

**Traps**

- Saying “roll back” without checking compatibility, data migration, or irreversible actions.
- Starting a brainstorm before bounding the incident.
- Changing multiple variables and erasing the original state.

<a id="d6-1-2"></a>
#### D6.1.2 · Build a change ledger

**Status.** Commonly useful.

**Answer shape.**

> “I would construct a timeline around the first bad event, not the page time. For each change I log
> owner, rollout percentage, affected surface, and rollback status: weights, prompt, tokenizer,
> feature code, retrieval index, data source, dependency, hardware image, autoscaling policy, and
> evaluator. I compare affected and unaffected cohorts to rank changes. A change is evidence, not
> guilt; silent external or data changes remain hypotheses.”

**Follow-ups.** Multiple changes landed together; clocks disagree; gradual rollout; hidden feature
flag.

**Trap.** Assuming “no code deploy” means “nothing changed.”

---

<a id="d6-2"></a>
### D6.2 · Discuss a training loss spike

Classify the shape:

- brief spike and recovery;
- slow recovery;
- permanent level shift;
- divergence or NaN;
- global versus one rank, shard, or data slice;
- aligned movement in gradient norm, learning rate, throughput, and validation.

Then inspect the time window before the visible spike. In large runs, a bad data pocket, optimizer
state, numerical problem, or hardware error can develop before the dashboard crosses a threshold.
Preserve checkpoints, sampler state, recent data IDs, per-rank metrics, and environment versions.

<a id="d6-2-1"></a>
#### D6.2.1 · Localize a loss spike at step 42,000

**Status.** Synthetic practice incident informed by the failure categories in the engineering source;
the exact step and scenario are illustrative, not reported as a recurring interview question.

**Illustrative spoken answer.**

> “I would not lower the learning rate first. I would classify the shape and check whether validation
> loss, gradient norm, throughput, and per-rank loss move with it. Then ask whether step 42,000 follows
> a resume, data-shard transition, schedule boundary, code/config change, or hardware event. I would
> compare the last good checkpoint with a replay using the same sampler state and captured batch IDs.
> Data, numerics, optimizer state, and hardware make different predictions about replayability and
> rank locality. Containment depends on recovery: a brief spike may be logged; persistent damage may
> require rollback and a different data order.”

**Practice follow-ups**

- “The batch before the spike looks normal.”
- “Only one rank has NaNs.”
- “The spike appears after resume.”
- “Training loss recovers but validation does not.”

**Traps**

- Inspecting only the immediately preceding batch.
- Treating gradient clipping as a root-cause fix.
- Restarting without preserving sampler and optimizer state, making the event irreproducible.

<a id="d6-2-2"></a>
#### D6.2.2 · Decide continue, skip, or roll back

**Status.** Commonly useful.

**Answer shape.**

> “Continue if the event is brief, state returns to the prior trajectory, validation is unaffected,
> and no corruption signal remains. Skip or quarantine a data window if the issue is reproducible and
> localized there. Roll back when parameters or optimizer state remain damaged, validation shifts,
> or the cause is unresolved with a credible recurrence risk. The decision uses expected lost compute
> versus expected damage, not the emotional cost of discarding progress.”

**Follow-ups.** Checkpoint cadence; repeated bad batch; non-deterministic kernels; deadline pressure.

**Trap.** Continuing because the run is expensive. Sunk compute does not make corrupted compute
valuable.

---

<a id="d6-3"></a>
### D6.3 · Debug a model or product regression

Decompose the end-to-end delta:

`traffic → preprocessing → context/retrieval → model → postprocessing/tools → UI → outcome measurement`

Replay the same frozen inputs through last-known-good and candidate bundles. If outputs differ,
bisect components. If outputs match but live outcomes differ, look at traffic, latency, exposure,
external state, and measurement.

Use a **golden corpus** for deterministic or bounded regression checks, but never mistake it for the
whole production distribution.

<a id="d6-3-1"></a>
#### D6.3.1 · Debug an accuracy regression after a release

**Status.** Commonly useful.

**Answer shape.**

> “I first reproduce the regression on frozen requests with full versioned traces. If old and new
> bundles differ, bisect weights, prompt, tokenizer, retrieval snapshot, feature code, and decoding
> configuration. If they do not differ offline, inspect exposure and live dependencies: traffic mix,
> permissions, latency timeouts, tool responses, or measurement. I slice by the smallest dimension
> that separates good and bad—language, context length, tenant, region, or task type—then test the
> corresponding boundary.”

**Practice follow-ups**

- “The aggregate drop comes from one large customer.”
- “Only long context regresses.”
- “Old traces contain private data.”
- “The release cannot be replayed exactly.”

**Trap.** Retraining before proving where the regression enters the pipeline.

<a id="d6-3-2"></a>
#### D6.3.2 · Handle a silent gradual regression

**Status.** Commonly useful.

**Answer shape.**

> “A gradual regression shifts suspicion toward input drift, index staleness, feedback loops,
> dependency changes, calibration drift, or resource saturation rather than a single deployment. I
> would compare time cohorts on fixed sentinel inputs and current traffic, inspect leading component
> metrics, and locate the first derivative change. Then I would ask what state accumulates—cache,
> corpus, labels, user adaptation, or model-generated training data.”

**Follow-ups.** Seasonality; logging schema changes; growing context; evaluator drift.

**Trap.** Looking only for a discrete bad commit because the incident ticket has a start date.

---

<a id="d6-4"></a>
### D6.4 · Find training-serving skew

Training-serving skew means the policy sees a different feature, transformation, context, tool
contract, or distribution in production. Prevention is architectural: share transformations, version
contracts, log served features, and replay production examples through training code. Google's Rules
of ML explicitly recommends measuring skew and logging serving features [[R18]](#ref-r18).

Useful invariants:

- same tokenizer and normalization;
- same feature definition and default;
- event-time rather than processing-time semantics where required;
- point-in-time correct joins;
- same prompt/tool schema;
- same retrieval and permission contract;
- train/eval mode and numerical precision explicit;
- no feature available only after the prediction.

<a id="d6-4-1"></a>
#### D6.4.1 · Model is good offline and bad in serving

**Status.** Commonly useful.

**Answer shape.**

> “I would capture a privacy-safe sample of exact served inputs and intermediate features, then replay
> them through the offline pipeline. Compare hashes or values at each boundary: raw event,
> preprocessing, tokenizer, feature join, context construction, model version, decoding, and
> postprocessing. If model outputs match, the gap is downstream or in traffic/measurement; if they
> diverge, the first mismatching boundary localizes skew. The durable fix is one versioned
> transformation or contract, plus a parity test in release.”

**Practice follow-ups**

- “Cannot log raw features.”
- “The online store updates faster.”
- “Batch normalization or dropout differs.”
- “Only a rare default value is wrong.”

**Trap.** Comparing only final predictions. The first differing intermediate is much more useful.

---

<a id="d6-5"></a>
### D6.5 · Suspect the evaluation when the conclusion is impossible

A bad eval can produce:

- correct-looking but misaligned labels;
- overlap or temporal leakage;
- wrong answer normalization;
- evaluator preference for length, style, or self-family;
- non-deterministic judge versions;
- denominator or aggregation bugs;
- silent task failures counted as model failures;
- selection of only completed or parseable outputs.

Validate the harness with known positive and negative controls. Hand-score a random sample and the
highest-impact disagreements. Version every prompt, judge, parser, and dataset. The ML Test Score's
separation of data, model, infrastructure, and monitoring tests is useful precisely because a model
metric can fail outside the model [[R19]](#ref-r19).

<a id="d6-5-1"></a>
#### D6.5.1 · Debug a sudden benchmark jump

**Status.** Commonly useful.

**Answer shape.**

> “A discontinuous jump with no plausible model change makes the harness a leading hypothesis. I
> would freeze artifacts, rerun the previous model in the current harness and the current model in
> the previous harness, then compare dataset version, parser, normalization, judge prompt/model, retry
> policy, and denominator. Known-answer controls and a blinded hand audit can reveal whether we are
> measuring more capability or merely accepting more outputs.”

**Follow-ups.** The old judge is unavailable; API retries differ; parser failure rate fell; data set
was refreshed.

**Trap.** Publishing the gain while the investigation runs because “the model did not change much.”

<a id="d6-5-2"></a>
#### D6.5.2 · Evaluate the evaluator

**Status.** Commonly useful.

**Answer shape.**

> “I would define the construct the evaluator is meant to approximate, sample cases across score and
> risk strata, collect blinded independent human judgments, and measure agreement plus systematic
> errors—not only correlation. Perturb length, order, style, and model identity while preserving
> content to test bias. I also inspect stability across repeated calls and versions. If the evaluator
> is used for training, I maintain a separate audit channel because optimization will target its
> weaknesses.”

**Trap.** Reporting one agreement number without the consequential disagreement slices.

---

<a id="d6-6"></a>
### D6.6 · Localize distributed failures by topology and boundary

Distributed failures often look like model failures because a single bad worker can contaminate a
collective. Bound by:

- rank, host, rack, region, and accelerator type;
- data shard and pipeline stage;
- collective operation and tensor size;
- deterministic versus intermittent;
- load, temperature, memory pressure, and job age;
- before or after resume/checkpoint.

Use controlled substitution: move the same workload to different hardware, different workload to the
same hardware, shrink world size, or replace one communication path. Preserve per-rank logs and avoid
letting aggregate means erase a single poison source.

<a id="d6-6-1"></a>
#### D6.6.1 · One distributed job hangs intermittently

**Status.** Commonly useful.

**Answer shape.**

> “First distinguish slow progress from a deadlock and capture stack plus collective state before
> killing the job. Bound the hang by rank and operation using per-rank timestamps. Compare whether the
> same rank, host, tensor size, data batch, or pipeline boundary recurs. Then use substitution: move
> the rank, reduce world size, or run a communication diagnostic on the suspect topology. Immediate
> mitigation may exclude a host; the root fix depends on whether the cause is mismatched collectives,
> data-dependent control flow, network, or hardware.”

**Practice follow-ups**

- “Watchdog kills before logs flush.”
- “The bad rank changes every time.”
- “Only large messages hang.”
- “It started after adding conditional routing.”

**Traps**

- Calling every hang an NCCL issue.
- Restarting until it works and losing the evidence.
- Ignoring that divergent control flow can make ranks call different collectives.

<a id="d6-6-2"></a>
#### D6.6.2 · Finish an incident with prevention

**Status.** Commonly useful.

**Answer shape.**

> “I would not end at ‘fixed the node.’ I would state the violated invariant, why detection was late,
> and which control now prevents or shortens recurrence: preflight health test, per-rank sentinel,
> timeout with state capture, compatible checkpoint, canary topology, or owner/runbook. Then verify
> the control by recreating the failure safely. An action item without an owner, trigger, and test is
> not prevention.”

**Trap.** A generic promise to improve monitoring. Name the signal, threshold, response, and person
who owns it.

---

<a id="d6-7"></a>
### D6.7 · Debug agent side effects and prompt injection together

An agent incident can cross model, harness, sandbox, identity, and external-system boundaries in one
trace. Do not choose “the model” or “the infrastructure” before reconstructing the state machine.
Treat tool output and retrieved content as untrusted evidence, and treat a timed-out write as an
unknown outcome.

<a id="d6-7-1"></a>
#### D6.7.1 · A timeout duplicated a write while indirect injection attempted exfiltration

**Status.** Synthetic practice incident combining documented agent-security and reliability failure
modes; not claimed as a reported interview question.

**Prompt.** A research agent submitted an external job. The tool timed out, the harness retried, and
two jobs were created. In the same trace, a retrieved document instructed the agent to upload prior
attachments to an attacker-controlled site.

**Illustrative spoken answer.**

> “I would first revoke or suspend the task's credentials and leases, freeze new writes and egress,
> preserve the append-only session, tool-proxy, approval, network, and external audit logs, and avoid
> deleting the sandbox. Then reconcile side effects against the external system: search by task,
> principal, payload fingerprint, time, and any idempotency key; identify both jobs; stop or
> compensate the duplicate through the service's supported path; and notify affected owners.
>
> “Next I would reconstruct the first divergence. At the model layer, did untrusted document text
> alter the plan or request data outside the user's goal? At the harness layer, why could retrieved
> content reach an egress-capable tool, why did the retry reuse no stable operation ID, and why was
> approval absent? At the tool layer, did the API accept idempotency keys, and did its timeout mean
> unknown commit rather than failure? I would replay the captured inputs in an isolated sandbox with
> fake credentials and mock services to test each boundary without repeating harm.
>
> “The durable fixes are a write-ahead intent plus one idempotency key across retries; timeout as
> `UNKNOWN` followed by read/reconcile; least-privileged, task-bound identity; egress allowlists and
> data-classification checks; explicit approval for new destinations or sensitive transfer; strict
> separation of tool data from instructions; and regression tests that combine injection with
> ambiguous tool outcomes. Recovery is complete only when intended and observed external state
> match.”

**Practice follow-ups**

- The external service has no lookup or idempotency support.
- One duplicate action is irreversible.
- The event log records the request but not which tool worker sent it.
- The model never quoted the injection, but its next action changed.
- A human approved the original job, not the retry or data upload.

**Traps**

- Rotating credentials before preserving enough evidence to reconstruct scope.
- Assuming the timeout means the first write failed.
- Patching the prompt while leaving broad identity, egress, and retry semantics unchanged.
- Calling the incident one root cause when several controls failed independently.

---

<a id="section-d7"></a>

## D7 · Behavioral questions: show decisions, not adjectives

Behavioral rounds can trip up strong technical candidates because memory retrieval and answer
construction happen at the same time. Alisa Liu describes entering one assuming she was obviously
well-behaved, then going blank on simple prompts and failing to answer the actual question
[[R9]](#ref-r9). Preparation is not inventing polished stories. It is indexing real evidence before
the clock starts.

Every answer should let the interviewer inspect:

- the situation and stakes;
- your responsibility and authority;
- the decision or tension;
- what **you** said or did;
- how other people responded;
- the result, including cost or unresolved parts;
- what changed in your later behavior.

Traits are conclusions. “I have high ownership” is weak. A specific moment when you noticed an
unowned failure, chose a bounded action, aligned the right people, and left a durable control is
evidence.

---

<a id="d7-1"></a>
### D7.1 · Answer the question in the first sentence

Start with a headline:

> “Yes. The clearest example is **[event]**; I made **[decision]**, and the result was **[outcome]**.”

Then provide only context needed to understand the tension. Spend most time on actions and reasoning.
End with a genuine reflection, not “communication is important.”

If the prompt asks about conflict, a technically difficult project with no interpersonal disagreement
does not answer it. If it asks about failure, a delayed success does not answer it. Name the question
you are answering.

<a id="d7-1-1"></a>
#### D7.1.1 · Give a ninety-second behavioral answer

**Status.** Behavioral questions are widely reported; this time box is a practice device.

**Worksheet.**

> “The short answer is **[direct answer]**. The situation was **[two sentences]**. I was responsible
> for **[scope]**, while **[others]** owned **[scope]**. The hard choice was **[tension]**. I did
> **[specific actions in order]** because **[reasoning]**. The result was **[verified outcome,
> including what did not resolve]**. The part I would change is **[real correction]**; since then I
> have **[later behavior or control]**.”

**Practice follow-ups**

- “What did you say in the room?”
- “How did the other person see it?”
- “What evidence shows your action caused the result?”
- “What would they say you did poorly?”

**Traps**

- Five minutes of setup.
- “We” in every action sentence.
- A moral at the end that is not connected to changed behavior.

<a id="d7-1-2"></a>
#### D7.1.2 · Correct yourself when the story does not answer

**Status.** Commonly useful.

**Answer shape.**

> “I realize that example shows technical difficulty, not conflict. Let me switch to a better one:
> **[real example]**. The disagreement was **[decision]**, and my role was **[role]**.”

**Trap.** Continuing a mismatched story because changing examples feels awkward. Self-correction is
better evidence than polished irrelevance.

---

<a id="d7-2"></a>
### D7.2 · Motivation and “why this lab”

A credible motivation answer has four links:

`problem you care about → evidence from your work → why this team now → mutual test`

Do not praise a brand. Name a problem, a recent artifact or direction, the capability or constraint
unique to the role, and what you hope to learn or contribute. OpenAI's official guide asks candidates
to discuss motivations and goals and to study work related to the interviewing team
[[R4]](#ref-r4). Anthropic's careers page says candidates will be asked about experience and
motivation [[R2]](#ref-r2). This is preparation they openly expect.

<a id="d7-2-1"></a>
#### D7.2.1 · Answer “Why do you want to work here?”

**Status.** Official guidance supports this prompt across multiple labs.

**Worksheet.**

> “The problem I want to spend the next few years on is **[specific problem]**. In **[real project]**
> I learned **[evidence-backed lesson]**, which made me believe **[current view]**. This team is
> unusually relevant because **[specific current work, infrastructure, deployment surface, or
> collaborators from official sources]**. I could contribute **[verified capability]**, and the
> question I want to test in conversations here is **[honest uncertainty about role/team]**.”

**Practice follow-ups**

- “Why not your current organization?”
- “Which team or paper specifically?”
- “What do you disagree with us about?”
- “If this role changes, what remains motivating?”

**Traps**

- “Best people, biggest models, most impact.”
- Reciting a mission with no connection to decisions you have made.
- Claiming a team-specific fit without knowing the actual team.

<a id="d7-2-2"></a>
#### D7.2.2 · Give an honest criticism of the lab

**Status.** Commonly useful; not asserted as a universal question.

**Answer shape.**

> “From public information, the tension I would want to understand is **[specific technical or
> organizational tension]**. My current view is **[position with confidence]** because
> **[evidence]**. I may be missing **[inside context]**, and evidence that would change my mind is
> **[evidence]**. I would ask the team how it decides **[related trade-off]**.”

**Follow-ups.** Take the opposite side; identify a decision you admire; say what would be a deal
breaker.

**Trap.** Choosing a fake weakness that is actually praise, or making a strong accusation from rumor.

<a id="d7-2-3"></a>
#### D7.2.3 · Give a ninety-second career spine

**Status.** Private worksheet. Required before personal use; do not publish private answers.

**Answer shape.**

> “My background began in **[real starting point]**, where I learned **[verified capability or
> question]**. The pivot toward **[current direction]** happened because **[specific evidence or
> experience]**, not because I need the chronology to look inevitable. Since then, **[two real
> choices]** built the through-line: **[problem you repeatedly chose]**. Now is the right moment to
> move because **[what changed in your capability, problem, or opportunity]**. I am leaving
> **[current context]** for **[honest pull and bounded constraint, without disparagement]**. The next
> logical step is **[role/team type]** because it lets me contribute **[evidence-backed strength]**
> while testing **[real growth question]**.”

**Practice follow-ups.** Why that pivot; why now rather than a year ago; what you are giving up; why
leave if the current work is going well; what next step would also make sense; which fact in the
spine would a former collaborator emphasize differently.

**Trap.** Turning a private fact worksheet into a public autobiography. Fill exact employers,
timelines, reasons, and constraints privately; publish only material you deliberately approve.

---

<a id="d7-3"></a>
### D7.3 · Ownership, ambiguity, and prioritization

Ownership has boundaries. It is noticing the unowned problem, establishing who is affected, choosing
a reversible first move, and creating durable responsibility. It is not bypassing every owner or
personally absorbing unlimited work.

Under ambiguity, show how you reduce uncertainty:

1. define the decision and deadline;
2. separate reversible and irreversible choices;
3. identify the assumption with highest value of information;
4. take a bounded step;
5. set a checkpoint or stop rule.

Prioritization should expose opportunity cost. “Everything was important” means no prioritization
happened.

<a id="d7-3-1"></a>
#### D7.3.1 · Describe taking ownership outside your scope

**Status.** Commonly useful.

**Worksheet.**

> “I noticed **[unowned problem]** because **[signal]**. It affected **[stakeholders]**, but authority
> sat with **[owner]**. I first **[containment or evidence step]**, then aligned with **[people]** on
> scope and decision rights. I owned **[bounded work]**, not **[what stayed elsewhere]**. The result
> was **[outcome]**, and the durable change was **[owner, process, test, or documentation]**.”

**Practice follow-ups**

- “Did anyone think you were overstepping?”
- “What did you stop doing to make room?”
- “Who owned it afterward?”

**Trap.** A hero story whose system remains dependent on the hero.

<a id="d7-3-2"></a>
#### D7.3.2 · Prioritize three urgent projects

**Status.** Commonly useful.

**Answer shape.**

> “I would compare consequence, urgency, reversibility, dependency, and information value. I first
> protect the item with irreversible or severe downside, then unblock work that gates several teams,
> and defer the reversible optimization. I would make the non-choice explicit to stakeholders:
> **[project]** pauses until **[date/evidence]**. If all three truly require immediate work, the
> constraint is capacity and I escalate with options rather than silently overcommit.”

**Follow-ups.** An executive disagrees; priorities change tomorrow; your favorite project is cut.

**Trap.** Giving a scoring framework but never saying which project loses.

<a id="d7-3-3"></a>
#### D7.3.3 · Move under deep ambiguity

**Status.** Commonly useful.

**Worksheet.**

> “The ambiguous part was **[unknown]**, but the decision had to be made by **[deadline]**. I separated
> the reversible choice **[choice]** from the irreversible one **[choice]**. I ran **[small test or
> stakeholder check]**, set **[checkpoint]**, and proceeded with **[bounded action]**. When we learned
> **[new evidence]**, I updated **[decision]**. What I did not do was **[unjustified large commitment]**.”

**Trap.** Turning ambiguity into a story about working harder rather than reducing uncertainty.

---

<a id="d7-4"></a>
### D7.4 · Conflict, collaboration, and feedback

A conflict answer needs legitimate interests on both sides. If the other person is simply irrational,
the story says little about collaboration. Explain their model well enough that they might endorse
your description.

Strong conflict behavior includes:

- naming the actual decision rather than personality;
- asking what evidence each side trusts;
- separating values, facts, and incentives;
- creating a reversible test or escalation rule;
- preserving the relationship after the decision;
- recognizing what the other side got right.

<a id="d7-4-1"></a>
#### D7.4.1 · Describe a serious technical disagreement

**Status.** Commonly useful.

**Worksheet.**

> “We disagreed about **[decision]**. Their concern was **[steelman]**; mine was **[concern]**. We were
> optimizing different constraints: **[constraints]**. I first **[listened/clarified]**, then proposed
> **[experiment, decision rule, or escalation]**. We chose **[outcome]**. They were right about
> **[point]**, and I changed **[part of approach]**. The relationship afterward was **[evidence]**.”

**Practice follow-ups**

- “What words did you use?”
- “What if they had more authority?”
- “Did you ever escalate?”
- “What would they say about you?”

**Traps**

- A disagreement that ends because the other person realizes you were right.
- Hiding the emotional or incentive layer when it affected the decision.
- Equating politeness with avoiding a hard conversation.

<a id="d7-4-2"></a>
#### D7.4.2 · Receive difficult feedback

**Status.** Commonly useful.

**Answer shape.**

> “The feedback was **[specific behavior and impact]**. My first reaction was **[honest reaction]**,
> but I asked for examples and checked the pattern with **[source]**. I changed **[observable
> behavior]** and created **[measurement/check-in]**. Evidence it improved is **[evidence]**. The part
> I still disagree with is **[bounded disagreement]**, and here is how I handle it.”

**Follow-ups.** Give the exact example; explain why you had not seen it; describe feedback you chose
not to act on.

**Trap.** Selecting praise disguised as feedback, such as “I care too much.”

<a id="d7-4-3"></a>
#### D7.4.3 · Give feedback to a strong peer

**Status.** Commonly useful.

**Worksheet.**

> “The peer was strong at **[strength]**, and the recurring behavior was **[observable behavior]**,
> which affected **[impact]**. I checked that I had a pattern rather than one frustrating event. I
> gave the feedback privately with examples, asked for their view, and agreed on **[next behavior or
> check]**. I also changed **[my contribution to the dynamic]**. The result was **[honest result,
> including if incomplete]**.”

**Trap.** Centering your courage rather than whether the feedback was accurate and useful.

---

<a id="d7-5"></a>
### D7.5 · Failure and learning speed

Choose a failure where:

- you had meaningful agency;
- the cost was real but discussable;
- the mistake is specific;
- you can separate decision quality from outcome;
- a later behavior changed.

Do not manufacture a clean redemption arc. Some consequences remain. Owning them without
self-dramatization is stronger than turning every failure into a triumph.

Learning speed is demonstrated by a loop: gap → plan → feedback → changed output. “I learn fast” is
not evidence.

<a id="d7-5-1"></a>
#### D7.5.1 · Describe a real failure

**Status.** Commonly useful.

**Worksheet.**

> “I failed to **[responsibility]**. The immediate cause was **[your decision or omission]**, not only
> **[external factor]**. At the time I believed **[assumption]**; I should have checked
> **[evidence]** earlier. The consequence was **[real cost]**. I first **[repair/accountability]**, then
> changed **[process or behavior]**. A later example where that change mattered is **[real event]**.
> The limitation is **[what was not fully repaired]**.”

**Practice follow-ups**

- “When did you know?”
- “Who did you tell first?”
- “Did you repeat it?”
- “What would your manager say the failure was?”

**Traps**

- Choosing a team failure while your own action remains invisible.
- Choosing a failure so old or small that no current learning is tested.
- Ending at an abstract lesson.

<a id="d7-5-2"></a>
#### D7.5.2 · Learn an unfamiliar area quickly

**Status.** Commonly useful.

**Answer shape.**

> “The gap was **[specific capability]**, and the deadline/output was **[deliverable]**. I built a
> dependency map, found **[primary sources/experts]**, and made a small artifact by **[early date]** so
> feedback arrived before I felt ready. I tracked unknowns and tested understanding with
> **[implementation, prediction, or review]**. The output changed from **[before]** to **[after]**.
> What I still would not claim expertise in is **[boundary]**.”

**Follow-ups.** What did you deliberately not learn? How did you identify a credible source? What was
your first wrong model?

**Trap.** A reading list with no feedback-producing artifact.

---

<a id="d7-6"></a>
### D7.6 · Leadership and influence

Leadership answers should reveal the mechanism of influence:

- technical credibility;
- a clearer decision frame;
- a coalition or cross-functional translation;
- resource allocation;
- conflict resolution;
- coaching and delegation;
- taking accountability for a call.

At senior levels, the unit of output shifts from personal artifact to a system that lets others make
better decisions. That does not mean removing yourself from technical detail; it means choosing where
your depth changes the whole system.

<a id="d7-6-1"></a>
#### D7.6.1 · Lead a project through a change in direction

**Status.** Commonly useful.

**Worksheet.**

> “The original direction was **[direction]**. Evidence **[evidence]** made it no longer defensible,
> but changing course cost **[cost]** and affected **[people]**. I made the evidence visible, invited
> the strongest counterargument, and proposed **[new path plus transition]**. I decided
> **[decision]** within **[authority]**, while **[other decision]** belonged to **[owner]**. The result
> was **[outcome]**; the people cost or unresolved issue was **[honest cost]**.”

**Follow-ups.** Who resisted? How did you protect morale? Were you too late? What did you stop?

**Trap.** Describing communication but omitting the hard allocation decision.

<a id="d7-6-2"></a>
#### D7.6.2 · Help another person succeed

**Status.** Commonly useful.

**Answer shape.**

> “The person wanted **[goal]** and was blocked by **[observed gap]**. I did not take over the task. We
> agreed on **[ownership]**, I provided **[context/feedback/opportunity]**, and reduced support as
> **[evidence of independence]** appeared. Their outcome was **[outcome]**. I learned that my initial
> approach **[mistake]**, so I changed **[coaching behavior]**.”

**Trap.** Making the other person's story entirely about your mentorship.

---

<a id="d7-7"></a>
### D7.7 · Ethics and safety under uncertainty

Ethics questions are not solved by saying “safety first.” Show how you:

1. identify affected parties, including people outside the room;
2. separate legal, policy, technical, and moral questions;
3. estimate severity, likelihood, reversibility, and distribution;
4. preserve evidence and seek appropriate review;
5. choose containment proportional to uncertainty;
6. define what would change your view;
7. accept responsibility for the decision.

Avoid pretending the trade-off disappears. Sometimes delaying a beneficial system also causes harm;
sometimes a low-probability severe event dominates expected value; sometimes you lack authority and
must escalate.

<a id="d7-7-1"></a>
#### D7.7.1 · Respond to a safety concern before launch

**Status.** First-hand sources report occasional safety or societal-impact questions.

**Worksheet.**

> “I observed **[specific concern]**, with possible impact on **[affected parties]**. The uncertain
> pieces were **[unknowns]**, but **[part]** was serious and reversible enough to justify
> **[containment]**. I documented evidence, informed **[appropriate owner/review body]**, and proposed
> **[test and decision rule]**. I would support launch only if **[conditions]**; otherwise
> **[alternative]**. If overruled, I would **[real escalation path consistent with role and law]**.”

**Practice follow-ups**

- “The evidence is weak.”
- “Delay has a large opportunity cost.”
- “Your manager says launch.”
- “The concern affects a marginalized small group.”

**Traps**

- A theatrical answer with no proportionate action.
- Treating policy compliance as the complete ethical analysis.
- Claiming you would take an escalation action you would not actually take.

<a id="d7-7-2"></a>
#### D7.7.2 · Balance openness and misuse risk

**Status.** Commonly useful research judgment prompt.

**Answer shape.**

> “I would separate the artifact: high-level findings, evaluation method, code, weights, data, and
> operational details have different risk and scientific value. For each, assess plausible misuse,
> marginal capability enabled, reproducibility benefit, existing access, and whether staged or
> controlled release helps. My default is not ‘open’ or ‘closed’; it is the least restrictive release
> that keeps risk within a defensible bound, with a review and update plan.”

**Follow-ups.** Who decides? How do you avoid security through obscurity? What if competitors release
it anyway?

**Trap.** Giving a universal slogan instead of analyzing the particular artifact and threat model.

---

<a id="d7-8"></a>
### D7.8 · Show collaboration across different functions

Cross-functional collaboration is translation between valid but different objective functions:
research wants identifiable evidence, engineering wants reliability and maintainability, product
wants user value and timing, legal/security wants bounded risk, and operations wants a process that
can run repeatedly.

Do not say you “aligned everyone” without naming the disagreement, decision rights, and artifact that
made coordination possible.

<a id="d7-8-1"></a>
#### D7.8.1 · Align research, engineering, and product

**Status.** Commonly useful.

**Worksheet.**

> “The shared goal was **[goal]**, but research optimized **[constraint]**, engineering
> **[constraint]**, and product **[constraint]**. The conflict appeared as **[decision]**. I created
> **[joint metric, interface contract, experiment, or milestone]** and clarified who decided
> **[which layer]**. We chose **[choice]**, explicitly accepting **[cost]**. The durable artifact was
> **[artifact]**, and the unresolved tension was **[tension]**.”

**Practice follow-ups**

- “Who was unhappy?”
- “What did you concede?”
- “How did you prevent research prototypes from becoming production debt?”
- “How did production evidence return to research?”

**Trap.** Presenting another function as a constraint to route around instead of a source of required
information.

Meta's official hiring page says candidates may meet peers, cross-functional partners, and leaders,
and should prepare their relevant experience and questions [[R6]](#ref-r6). The safest general
preparation is therefore not guessing a value keyword. It is having real evidence for how you make
decisions with other people.

---

<a id="section-d8"></a>

## D8 · Build a story bank without becoming mechanical

STAR, CAR, and CARL are compression tools, not scripts. Use whichever labels help you recover:

- **Context:** only what makes the tension intelligible;
- **Responsibility:** what you were accountable for and what authority you had;
- **Decision and action:** alternatives, reasoning, exact behavior, and collaboration;
- **Result:** outcome, evidence, cost, and unresolved part;
- **Learning:** a later change in behavior, not a slogan.

The answer should sound like a person remembering a real decision, not a template being executed. A
good test is whether you can leave the order, answer an interruption, and return without losing the
story.

Anthropic's candidate guidance is unusually explicit: use AI to refine authentic experience, never to
create experiences, and be transparent [[R1]](#ref-r1). Apply that rule everywhere. This section
deliberately contains no completed personal story.

---

<a id="d8-1"></a>
### D8.1 · Create an eight-story matrix

Six to eight stories are usually enough if they contain different decisions and can be viewed from
several angles. Do not force one heroic project to answer every prompt.

Fill this matrix with event names only first. Then add evidence. A blank cell is a preparation task,
not permission to invent.

| ID | Core evidence | Useful prompts | Facts that must be supplied |
|---|---|---|---|
| S1 | technical depth and decisive trade-off | hardest project, judgment, scale | exact decision, alternatives, your role, measured result |
| S2 | failure and recovery | mistake, failed experiment, changed belief | your error, consequence, repair, later changed behavior |
| S3 | ownership under ambiguity | initiative, unclear scope, urgency | missing owner, authority boundary, what you stopped |
| S4 | conflict and disagreement | collaboration, dissent, influence | other side's valid concern, exact interaction, decision |
| S5 | leadership and direction change | vision, prioritization, delegation | who decided, resources moved, people cost |
| S6 | feedback and growth | received/gave feedback, coaching | actual feedback, initial reaction, observable change |
| S7 | ethics or safety judgment | responsible release, escalation | affected parties, evidence, authority, real escalation path |
| S8 | cross-functional delivery | research-to-production, partnership | objective conflict, interface, launch evidence, limitation |

For every story, complete a one-page evidence card:

```text
Event and date:
Question this story truly answers:
My responsibility / authority:
Team and inherited work:
Tension or decision:
Alternatives considered:
What I said or did, in order:
Verified result and uncertainty:
Cost / unresolved part:
What I would change:
Later evidence that behavior changed:
Confidentiality boundary:
Artifacts that refresh memory:
```

<a id="d8-1-1"></a>
#### D8.1.1 · Decide whether two prompts can share one story

**Status.** Commonly useful.

**Answer shape.**

> “They can share the event only if the evidence differs. For an ownership prompt I center how I
> noticed and bounded an unowned problem. For a conflict prompt I center the other person's model,
> our interaction, and the decision rule. If I merely change the moral at the end, I do not have two
> answers; I have one memorized story being stretched.”

**Drill.** Take one event and give two sixty-second answers to different prompts. Highlight sentences
that are identical. More than half identical means the angle has not truly changed.

**Trap.** Reusing the same story in several interviews in one loop without realizing interviewers may
compare notes.

<a id="d8-1-2"></a>
#### D8.1.2 · Choose a failure story

**Status.** Commonly useful.

**Answer shape.**

> “I choose a story where I had agency, the consequence was real, and I can name a later behavioral
> change. I reject stories where the ‘failure’ is only an ambitious target, another person's error,
> or a success delayed by circumstances. I also reject confidential incidents I cannot explain
> concretely.”

**Drill.** For each candidate, answer: “What did I do wrong?” in one sentence. If the sentence has no
first-person verb, choose another story.

<a id="d8-1-3"></a>
#### D8.1.3 · Build the private evidence card behind the career spine

**Status.** Private worksheet. Required before personal use; do not publish private answers.

```text
Background through-line:
Pivot and the evidence that caused it:
Why now:
Why leaving / what is being left behind:
Why this next step follows:
Two choices that prove the through-line:
Facts, dates, and artifacts that verify each claim:
What a former collaborator might phrase differently:
Confidentiality boundary:
```

**Drill.** Give the ninety-second spine from D7.2.3, then let a partner choose any transition and ask
for the real evidence. If the transition is only a polished narrative, revise it or remove it. Keep
the completed card private.

**Trap.** Making the career story look inevitable. A real pivot includes uncertainty, alternatives,
and something you stopped pursuing.

---

<a id="d8-2"></a>
### D8.2 · Build the follow-up tree before polishing prose

Interviewers test whether detail continues below the prepared surface. For each story, answer these
branches in notes—not as a speech:

1. **Chronology:** what happened immediately before and after?
2. **Role:** what exactly was yours, inherited, delegated, or shared?
3. **Alternatives:** what else was considered, and by whom?
4. **Evidence:** what did you know then; what did you learn later?
5. **Interaction:** what words did you use; how did others respond?
6. **Counterfactual:** what if the result or authority were reversed?
7. **Cost:** time, trust, quality, opportunity, compute, or user impact?
8. **Boundary:** what remains uncertain or confidential?
9. **Transfer:** where did you apply the lesson later?

If a branch has no detail because memory is hazy, recover it from documents or choose another story.
Do not fill it with plausible connective tissue.

<a id="d8-2-1"></a>
#### D8.2.1 · Survive three levels of “why”

**Status.** Commonly useful.

**Prompt chain.** “Why did you choose A?” → “Why did that constraint dominate?” → “Why did you trust
that evidence?”

**Answer shape.**

> “At the first level, name the trade-off. At the second, connect it to the stakeholder or system
> consequence. At the third, expose evidence quality and uncertainty. If the real answer at any level
> was authority, habit, or deadline, say so. Do not retrofit a technical theory.”

**Drill.** A partner selects any action verb in your story and asks why three times. Score whether the
third answer reaches evidence, value, or constraint—not another restatement.

**Trap.** Treating every “why” as an attack. It is often an invitation to show the bedrock.

<a id="d8-2-2"></a>
#### D8.2.2 · Answer a counterfactual follow-up

**Status.** Commonly useful.

**Prompt.** “What would you have done if the experiment favored the other team?”

**Answer shape.**

> “Use the decision rule you claim existed. If result B crossed **[threshold]**, I would have supported
> **[other path]**, because my commitment was to **[goal]**, not my proposal. The transition cost
> would have been **[cost]**, so I had prepared **[reversible step]**. If no such rule existed, I
> should admit the process was less rigorous than I now describe.”

**Trap.** Saying the other outcome was impossible. That makes the experiment sound ceremonial.

---

<a id="d8-3"></a>
### D8.3 · Use public artifacts only as candidate story prompts

This repository's public homepage and bibliography expose several possible artifacts
[[R25]](#ref-r25) [[R26]](#ref-r26). They establish titles, links, authorship, and a few public
outcome statements. They **do not** establish the user's personal decision, exact ownership,
failed paths, team dynamics, or lesson. Publication authorship is not a behavioral story.

The table is an inventory to investigate:

| Candidate artifact from the public site | Possible angle to verify | Missing facts the user must supply |
|---|---|---|
| CaOPD / calibration-aware on-policy distillation | research taste, mechanism, failed alternative | exact personal role, decisive experiment, negative results, compute/cost |
| Agentic Confidence Calibration | new problem definition, evaluation design | ownership split, key disagreement, metric choices, limitations |
| Agentic Uncertainty Quantification | long-horizon reliability, system design | direct contribution, failed approaches, collaboration, deployment boundary |
| SAC3 hallucination detection | research-to-production, measurable impact | exact deployment role, adoption evidence, incident/iteration details |
| PhaseEvo / SEE prompt optimization | optimization and productization | which artifact was personally owned, trade-offs, user feedback, failures |
| Enterprise deep research | multi-agent/system design, cross-functional work | role, architecture decisions, production constraints, unresolved risks |
| Large-scale distributed learning on supercomputers | scale, incident, infrastructure judgment | exact project, hardware/run facts, bottleneck, personal actions |
| DOE Generative-AI-for-Science work | leadership, portfolio prioritization | which project, decision rights, team/resource facts, outcome |

Before using an artifact, verify:

- the public description is current and accurate;
- your exact authorship and leadership role;
- which figures may be disclosed;
- whether the event contains the tension required by the prompt;
- whether collaborators would recognize the attribution as fair.

<a id="d8-3-1"></a>
#### D8.3.1 · Turn a publication into a deep-dive candidate

**Status.** Personal worksheet.

**Answer shape.**

> “I would not begin from the paper abstract. I would locate one decision I personally owned:
> **[decision]**. Then recover the alternatives, the evidence available at the time, the failed or
> surprising result, and what collaborators owned. If I cannot fill those from memory and artifacts,
> the paper stays on the inventory and does not become an interview story.”

**Required before personal use; do not publish private answers.** Privately supply decision records,
experiment logs, exact ownership, disclosure boundary, cost/scale, and one limitation not already
polished for publication.

<a id="d8-3-2"></a>
#### D8.3.2 · Turn a public impact number into defensible evidence

**Status.** Personal worksheet.

**Answer shape.**

> “I first verify what the number counts, over what period, and from which source. Then I state my
> causal contribution narrowly. Adoption is not automatically quality, and team impact is not
> automatically my impact. I pair the number with a mechanism or independent outcome and disclose
> uncertainty. If the figure is internal, stale, or not approved, I use a qualitative or normalized
> description instead.”

**Trap.** Repeating a public number without being able to define its denominator.

---

<a id="d8-4"></a>
### D8.4 · Keep the delivery natural and interruptible

Do not memorize sentences. Memorize five landmarks:

`headline · tension · two actions · result · reflection`

Practise with interruptions:

- begin at the decision, not the background;
- answer a follow-up directly;
- say “I’ll return to the result” and actually return;
- compress any branch to one sentence;
- stop when the question is answered.

Use names only when appropriate and public. Otherwise use roles. Remove confidential customer,
security, personnel, or unreleased research detail before rehearsal with external tools or people.

<a id="d8-4-1"></a>
#### D8.4.1 · Recover after an interruption

**Status.** Commonly useful.

**Answer shape.**

> “Direct answer to the interruption: **[answer]**. The reason it matters to the story is
> **[connection]**. Returning to the original question, that evidence changed my action from
> **[before]** to **[after]**.”

**Drill.** Ask a partner to interrupt at random after thirty seconds. The answer passes if you address
the new question in the first sentence and return within another thirty seconds.

**Trap.** Saying “I’ll get to that” when the follow-up is the interviewer's chosen branch. Follow
their signal.

<a id="d8-4-2"></a>
#### D8.4.2 · Audit a story for fabrication risk

**Status.** Required before public use.

**Checklist**

- Every number has a source and denominator.
- Every “I” action is personally owned.
- Every “we” outcome credits the team.
- Dialogue is exact enough to be honest or clearly paraphrased.
- Causal language does not exceed the evidence.
- Confidential details are removed, not replaced with invented specifics.
- Reflection is supported by a later behavioral change.
- The story can survive “who else was there?” and “what would they say?”

**Trap.** Assuming a story is safe because each individual sentence sounds plausible. Consistency
across follow-ups is the real test.

---

<a id="section-d9"></a>

## D9 · Interviewer questions, communication rubric, mocks, and debrief

The last ten minutes and the practice week before them serve the same purpose: reduce uncertainty
about the actual work. Good questions test a hypothesis about the team. Good mocks test a hypothesis
about your failure mode.

Silvia Sapora reports using targeted LLM mock interviews before specific rounds and seeing surprising
overlap with real questions, while also emphasizing that preparation was highly role-specific
[[R10]](#ref-r10). OpenAI's official recruiting event likewise recommends practising career examples
and showing the problem-solving process [[R5]](#ref-r5). Use mock tools to generate pressure and
follow-ups, not personal content.

---

<a id="d9-1"></a>
### D9.1 · Ask questions that can change your decision

Build questions from unknowns:

| Unknown | Better question | Evidence to listen for |
|---|---|---|
| success | “What would make this hire clearly successful after six and twelve months?” | concrete outcomes, not traits |
| work mix | “What did the team spend time on last month that is absent from the job description?” | actual versus advertised work |
| decision rights | “Tell me about a recent research/engineering trade-off and who decided.” | operating model |
| evaluation | “How do you know a research result is ready to affect the product?” | evidence and handoff |
| failure | “What is a recent project the team stopped, and what evidence stopped it?” | learning and sunk-cost behavior |
| collaboration | “Where do researchers, engineers, product, and safety most often disagree?” | real interfaces |
| resources | “Which constraint binds first today: data, compute, eval, deployment, or headcount?” | opportunity surface |
| management | “How are priorities changed, and how does dissent travel?” | psychological and decision safety |
| role fit | “Which strength would be unusually valuable here, and which gap would be hard to support?” | honest fit |

Avoid questions answered on the website unless you are testing how the public statement works in
practice. Ask the same core uncertainty of several interviewers; variance in answers is data.

<a id="d9-1-1"></a>
#### D9.1.1 · Ask a researcher about team quality

**Status.** Commonly useful.

**Illustrative questions**

- “What important belief has the team changed in the last year, and what evidence changed it?”
- “Which result looked promising offline but failed to influence deployment?”
- “How are negative results preserved and reused?”
- “What technical disagreement is still live?”
- “What makes a project easy or difficult to get adopted here?”

**Follow-up move.** Ask for a concrete recent example after a general answer.

**Trap.** Asking “what is the culture like?” and accepting adjectives.

<a id="d9-1-2"></a>
#### D9.1.2 · Ask a hiring manager about the role

**Status.** Commonly useful.

**Illustrative questions**

- “What problem would you hope I own first, and why is it unowned now?”
- “At twelve months, what outcome distinguishes a strong hire from an acceptable one?”
- “Which part of my background are you hiring for, and which concern are you still evaluating?”
- “What decision authority comes with the role before formal management responsibility?”
- “What changed between writing this job description and today?”

**Trap.** Turning the question period into a second sales pitch. Ask, listen, and update.

---

<a id="d9-2"></a>
### D9.2 · Score communication with observable behavior

Use a six-axis rubric. Score from 0 to 3 after the answer:

| Axis | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Frame | no decision | context only | decision and assumptions | frame changes correctly with new evidence |
| Structure | untraceable | list of facts | clear map and transitions | compresses/expands without losing map |
| Depth | slogans | one technical layer | mechanism plus evidence | survives three why/how follow-ups |
| Judgment | “it depends” | axes without choice | choice plus trade-off | reversal condition and sensitivity |
| Calibration | bluff/refusal | vague caveats | labels fact versus hypothesis | updates confidence explicitly |
| Collaboration | monologue/defense | occasional check-in | uses interviewer signal | co-solves and redirects cleanly |

Add two binary checks:

- **Answered the literal question?**
- **Stopped when the answer was complete?**

Do not average away a zero. A technically deep answer that never answers the question is not a
passing discussion answer.

<a id="d9-2-1"></a>
#### D9.2.1 · Diagnose “clear but shallow”

**Status.** Commonly useful.

**Answer shape.**

> “The map is working, but every branch ends in a label. I would choose one claim and require
> mechanism, evidence, alternative, and failure mode. The next drill is not another full mock; it is
> five minutes of follow-ups on that branch, with no new top-level topics.”

**Trap.** Adding more categories to sound comprehensive. Depth is a longer chain, not a wider list.

<a id="d9-2-2"></a>
#### D9.2.2 · Diagnose “deep but hard to follow”

**Status.** Commonly useful.

**Answer shape.**

> “The technical content is present, but the listener cannot locate the decision. I would practise
> headline-first answers, announce no more than three branches, and checkpoint after each:
> ‘That rules out data leakage; next I would test optimization.’ The drill is to compress the answer
> to thirty seconds before expanding it.”

**Trap.** Removing all detail. The fix is navigation, not shallowness.

---

<a id="d9-3"></a>
### D9.3 · Run mocks that reproduce the round

Three useful formats:

**Thirty-minute technical discussion**

- 2 minutes: prompt and clarification;
- 15 minutes: one design or experiment branch;
- 8 minutes: hypothetical result changes twice;
- 5 minutes: feedback and one repeat.

**Forty-five-minute project deep dive**

- 5 minutes: project version;
- 25 minutes: interviewer chooses decisions, ownership, failure, and evidence;
- 10 minutes: counterfactual and future work;
- 5 minutes: factual gaps and rubric.

**Forty-five-minute behavioral round**

- 4 prompts, one at a time;
- at least two follow-ups per prompt;
- interviewer rejects one mismatched story and asks for another;
- no feedback until the end, so retrieval pressure remains realistic.

Rotate interviewer personas: technical peer, skeptical researcher, hiring manager, cross-functional
partner. The rubric stays fixed; the follow-up style changes.

<a id="d9-3-1"></a>
#### D9.3.1 · Prompt an AI mock interviewer safely

**Status.** Practice template.

**Template**

```text
Act as a skeptical but collaborative interviewer for [role].
Use only the job description, round description, and source material I provide.
Ask one question at a time. Do not write answers or invent experiences for me.
After my answer, choose one claim and follow it three levels deep.
At least once, change a constraint or provide a hypothetical result.
Do not give feedback until I say the interview is over.
Then score only observable behavior: framing, structure, technical depth,
judgment, calibration, collaboration, literal question answered, and concision.
Quote the exact sentence that caused each deduction.
```

**Safety rule.** Remove confidential data and unreleased project detail. The mock does not need it to
test structure.

**Trap.** Asking the model to create an ideal answer before you attempt retrieval. Recognition will
replace practice.

<a id="d9-3-2"></a>
#### D9.3.2 · Use a human mock well

**Status.** Commonly useful.

**Answer shape.**

> “Give the interviewer a one-page contract: target role, round type, time, rubric, and permission to
> interrupt. Do not give them my prepared answer. Ask them to note the first moment they lost the
> thread and the strongest unsupported claim. Afterward I ask for evidence—what sentence or behavior
> produced the judgment—before discussing advice.”

**Trap.** Selecting only close collaborators who already understand your project context. Include
someone who needs the explanation to work.

---

<a id="d9-4"></a>
### D9.4 · Debrief into a small error taxonomy

Within ten minutes, record facts before memory rewrites them:

- actual questions and follow-ups;
- answer chosen and where it branched;
- interviewer redirects or confusion;
- facts you could not retrieve;
- claims that lacked evidence;
- moments of overconfidence or unnecessary apology;
- questions you asked and the evidence received;
- energy and timing, without turning feelings into a performance score.

Classify the first failure:

- **retrieval:** knew it but could not access it;
- **knowledge:** did not know it;
- **framing:** solved the wrong decision;
- **depth:** labels ended under follow-up;
- **evidence:** claim outran support;
- **communication:** listener lost the map;
- **story fit:** example did not answer;
- **calibration:** bluffed, refused, or failed to update;
- **logistics:** environment, audio, timing, or fatigue.

Each class has a different repair. Reading more does not fix story fit. More mocks do not fix a
missing technical concept.

<a id="d9-4-1"></a>
#### D9.4.1 · Turn a failed answer into one drill

**Status.** Commonly useful.

**Answer shape.**

> “Write the first observable failure as a behavior: ‘I spent two minutes before naming the
> objective,’ not ‘I was nervous.’ Then choose a constrained repair: five forty-five-second openings
> with a decision and baseline. Repeat the original prompt once after the drill. If the behavior does
> not change, revise the diagnosis.”

**Trap.** Creating a ten-item improvement plan after every interview. The log becomes another place
to avoid practising.

<a id="d9-4-2"></a>
#### D9.4.2 · Avoid overfitting to one interview

**Status.** Commonly useful.

**Answer shape.**

> “Separate a one-off question from a recurring failure. I add a new topic to the core plan only if it
> appears in the stated upcoming loop, recurs across credible sources, or exposes a foundational gap.
> Otherwise it enters a wildcard log. I always repair a process failure—such as not clarifying the
> decision—even if the exact prompt never returns.”

**Trap.** Rebuilding the entire study plan around the most emotionally vivid question.

---

<a id="d9-5"></a>
### D9.5 · Run a final readiness review

Forty-eight hours before the round, verify:

- exact format, tool policy, time, and interviewer roles from the recruiter;
- two-, five-, and fifteen-minute versions of two project candidates;
- eight story cards with no fabricated blanks;
- one current paper or system you can summarize, critique, and extend;
- one system-design and one experiment-design mock;
- one incident framework cold;
- questions tailored to interviewer type;
- confidentiality redactions;
- sleep, environment, and a hard stop on last-minute expansion.

The objective is not feeling ready. It is removing preventable failure modes.

<a id="d9-5-1"></a>
#### D9.5.1 · Decide what not to study at the end

**Status.** Commonly useful.

**Answer shape.**

> “I stop adding broad topics. I review the recruiter-specified loop, factual gaps in my own stories,
> and the short list of errors from mocks. A new paper or question is admitted only if it repairs a
> known gap with high transfer. The night before, retrieval quality and sleep dominate one more
> chapter.”

**Trap.** Confusing anxiety relief with expected interview value.

---

<a id="section-d10"></a>

## D10 · Lab-specific preparation, bounded by evidence

Company preparation should be an **overlay**, not a second personality. Start from the same projects,
scientific reasoning, and story bank. Then update:

- the exact role and team;
- the recruiter's round list;
- official tool and AI policy;
- recent official work relevant to the team;
- the action surface and risk profile of the product;
- interviewer roles;
- questions that test mutual fit.

Use a source hierarchy:

1. recruiter instructions for your scheduled loop;
2. official candidate and careers guidance;
3. exact job description and team publications/products;
4. named first-hand accounts, labelled as experiences rather than policy;
5. aggregated reports as weak priors;
6. ignore anonymous precision and content-farm “values” or question lists.

Policies change. The facts below were checked on 2026-08-13 and must be rechecked before a real loop.

---

<a id="d10-1"></a>
### D10.1 · Build the recruiter packet first

Ask once, in one concise message:

- What are the named rounds, duration, and interviewer functions?
- What does each round evaluate?
- Which tools, languages, editors, whiteboards, or AI assistants are allowed?
- Is the project discussion a presentation, conversational deep dive, or both?
- Is a paper provided in advance?
- What may be prepared or brought?
- Is the loop tailored to this role or team?
- Are there accommodations or scheduling constraints to address now?

Record the answer with a date. If a public guide conflicts with your recruiter, clarify; do not choose
the version you prefer.

<a id="d10-1-1"></a>
#### D10.1.1 · Ask for the exact loop without sounding entitled

**Status.** Commonly useful logistics template.

**Template**

> “To prepare for the actual work you want to evaluate, could you share the round names, approximate
> duration, and the main signal for each? I would also like to confirm the tool/AI policy, whether the
> project or paper discussion requires prepared material, and any role-specific guidance. I
> understand interviewers may vary the details.”

**Trap.** Asking for leaked questions. The goal is format and evaluation signal.

<a id="d10-1-2"></a>
#### D10.1.2 · Reconcile conflicting reports

**Status.** Commonly useful.

**Answer shape.**

> “I treat the recruiter as authoritative for my scheduled round. Official public guidance supplies
> defaults, and named accounts suggest possible follow-ups. I do not convert one person's OOP,
> system-design, or values experience into company policy. I keep a wildcard budget rather than
> preparing a counterfeit universal loop.”

**Trap.** Averaging contradictory reports into a detailed process no candidate actually experienced.

---

<a id="d10-2"></a>
### D10.2 · OpenAI: follow the official process, then the team

OpenAI publishes an interview guide. It says candidates should prepare to discuss work and academic
experience, motivations, and goals; review recent work, especially that related to the interviewing
team; and expect skills-based formats that vary by team, including pair coding, take-homes, and
technical tests. It describes final interviews as typically four to six hours with four to six people
over one or two days. For engineering, it names design quality, code quality, performance, and test
coverage; across interviews it emphasizes communication, collaboration, and exposing how you solve
problems [[R4]](#ref-r4).

An official OpenAI recruiting event adds practical communication guidance: be clear and intentional,
show the work, ask for assumptions, solve with the interviewer rather than disappearing into a
monologue, prepare career examples, and use thoughtful questions to show curiosity
[[R5]](#ref-r5).

This supports a preparation method. It does **not** support a universal OpenAI question list.

<a id="d10-2-1"></a>
#### D10.2.1 · Build an OpenAI discussion overlay

**Status.** Based on official guidance.

**Worksheet**

- **Role/team:** [exact job and current team, confirmed]
- **Recent official work:** [two artifacts directly relevant to the role]
- **Your connection:** [one real project or decision per artifact]
- **Project handles:** [ownership, hard choice, failure, evidence, scale]
- **Communication drill:** [show map, ask assumption, make choice, test, update]
- **Questions:** [team success, decision rights, evidence-to-deployment]
- **Unknowns for recruiter:** [format, tools, presentation, AI policy]

**Trap.** Studying “OpenAI” as one topic while ignoring the team named in the job and recruiter call.

<a id="d10-2-2"></a>
#### D10.2.2 · Answer “Why OpenAI?” without inventing fit

**Status.** Personal worksheet; official guidance confirms motivation is discussed.

**Answer shape.**

> “Start from **[problem you have repeatedly chosen]**, cite **[real evidence from your work]**, connect
> it to **[specific recent official work from the interviewing team]**, and state
> **[capability you can contribute]**. End
> with **[uncertainty you want to test about the role]**. Do not claim access to scale, people, or
> research freedom you have not confirmed.”

**Required before personal use; do not publish private answers.** Privately supply the actual target
role/team, current official artifacts, personal evidence, and honest reason for leaving or changing
context.

---

<a id="d10-3"></a>
### D10.3 · Anthropic: policy is explicit, so follow it literally

Anthropic's candidate AI guidance says:

- create the first application draft yourself, then AI may refine it;
- complete take-homes without AI unless explicitly permitted;
- use AI to research, rehearse, and prepare questions;
- use no AI during live interviews unless explicitly permitted;
- do not use AI to create experiences; preserve authentic thought and transparency
  [[R1]](#ref-r1).

Its careers page says technical interviews use live coding tools, candidates may look up basic
information, and interviews include experience, motivation, and candidate questions. It also says
the company values demonstrated ability over credentials and that researchers do engineering while
engineers do research [[R2]](#ref-r2).

Anthropic currently publishes principles including acting for global good, holding benefit and risk
together, doing the simple thing that works, being helpful/honest/harmless, and putting the mission
first [[R3]](#ref-r3). Read the official wording and current work. Do not force every story into a
slogan; show real decisions, including tensions and counterevidence.

<a id="d10-3-1"></a>
#### D10.3.1 · Use AI correctly in Anthropic preparation

**Status.** Official policy.

**Answer shape.**

> “I may use AI to map the job description, generate follow-up pressure, find gaps, and critique my
> own draft. I write the underlying experience and first draft. I remove confidential content, verify
> every factual claim, and practise without assistance. For any assessment or live round, I follow
> the explicit instruction for that round; silence means no AI for the live interview and no AI for a
> take-home.”

**Trap.** Assuming a company that builds AI always wants AI used during evaluation.

<a id="d10-3-2"></a>
#### D10.3.2 · Prepare for Anthropic's research–engineering boundary

**Status.** Based on the official careers page.

**Worksheet**

> “For each project, prepare both the research claim and the operational artifact: hypothesis,
> evidence, implementation, failure, scale, and maintenance. Select one example where a simple method
> beat a sophisticated one and one where safety or reliability changed the design. These must be real
> examples; if none exists, leave the slot empty rather than reverse-engineer a company principle.”

**Practice follow-ups.** Why the simpler method worked; what engineering revealed about the research;
what safety cost was accepted; what you disagree with in a public approach.

**Trap.** Performing the published principles instead of reasoning with them.

---

<a id="d10-4"></a>
### D10.4 · Meta: role-specific materials and AI policy are the source of truth

Meta's public hiring page says interview formats and preparation are role-specific; recruiters and the
Career Profile provide the applicable schedule and guides. It also says **select** technical roles use
an authorized AI assistant inside CoderPad, and that outside AI tools are not allowed. Candidates
should practise in the supplied environment when applicable [[R6]](#ref-r6).

Meta provides an official ML initial-interview landing page describing a 45-minute conversation with
an engineer [[R7]](#ref-r7), and an ML full-loop landing page describing up to six 45-minute
conversations [[R8]](#ref-r8). These pages establish available preparation, not a universal research
loop. The candidate's Career Profile and recruiter remain authoritative.

<a id="d10-4-1"></a>
#### D10.4.1 · Build a Meta discussion overlay

**Status.** Based on official guidance.

**Worksheet**

- download and read the exact Career Profile guide;
- list each scheduled conversation and its signal;
- use the supplied practice environment;
- confirm whether AI is authorized in each round;
- prepare coding/system discussion with and without AI until policy is explicit;
- map project stories to technical depth, impact, collaboration, and role fit;
- prepare questions for peers, cross-functional partners, and leaders.

**Trap.** Generalizing the public “select roles” AI policy to every research interview.

<a id="d10-4-2"></a>
#### D10.4.2 · Adjust for an authorized AI-native round

**Status.** Official Meta guidance says this applies to select roles only.

**Answer shape.**

> “I would practise the exact tool and authorized models, while keeping the reasoning observable. I
> define the problem and tests before prompting, inspect generated code, run targeted checks, and
> explain what I accept or reject. I treat the assistant as an unreliable collaborator whose output I
> own. I use no outside tool. If the round is not explicitly AI-enabled, I practise without AI.”

**Follow-ups.** Model gives plausible wrong code; tool latency; interviewer asks why you prompted;
design discussion uses Mermaid.

**Trap.** Optimizing prompt cleverness while surrendering problem decomposition, verification, or
ownership.

---

<a id="d10-5"></a>
### D10.5 · Keep company differences evidence-backed

The supported comparison is narrow:

| Lab | Officially supported preparation facts | Deliberately not inferred |
|---|---|---|
| OpenAI | motivations/goals, interviewing team's recent work, variable assessments, communication and collaboration | universal OOP, project, or values question set |
| Anthropic | explicit AI-use policy, experience/motivation, live coding, published principles, research–engineering overlap | that every interviewer scores slogans or one standard values round |
| Meta | role-specific guides, up to six ML full-loop conversations, authorized AI in select technical roles | that every ML/RS loop is identical or AI-enabled |

Named first-hand accounts can add hypotheses, but label them. “One candidate reported X” is honest;
“Lab Y always asks X” is not.

<a id="d10-5-1"></a>
#### D10.5.1 · Make a one-page lab brief

**Status.** Required personal worksheet.

```text
Role, level, team:
Recruiter-confirmed rounds and dates:
Official tool / AI policy, checked on:
Two recent team artifacts:
One real point of agreement:
One real, evidence-backed question or disagreement:
Two projects that best match the role:
Four stories most likely to be useful:
Technical gap to repair:
Questions by interviewer type:
Facts still unconfirmed:
```

**Required before personal use; do not publish private answers.** Complete every field privately. A
generic company brief cannot answer a team-specific loop.

<a id="d10-5-2"></a>
#### D10.5.2 · Update the brief when the recruiter changes the loop

**Status.** Commonly useful.

**Answer shape.**

> “I treat the new schedule as evidence, not inconvenience. I remap preparation to the actual rounds,
> preserve only high-transfer work, and move unsupported company lore out of the core plan. If a
> wildcard appears, I use the reusable discussion frame rather than inventing a company-specific
> script.”

**Trap.** Continuing the old plan because time already invested in it feels valuable.

---

<a id="d10-6"></a>
### D10.6 · Google DeepMind: use the role-specific packet, not a fixed question bank

Google DeepMind's official careers page describes a typical process while explicitly saying exact
steps differ by role and invited candidates receive detailed role-specific preparation. The current
overview is:

1. a 30-minute recruiter call about background, experience, motivation, questions, and next stages;
2. possibly a hiring-manager interview about relevant capabilities and the team's work;
3. two or three skills interviews against competencies required for the role, with potential peers;
4. final interviews with Team Leads and leadership, including a potential manager, viewed through
   team goals, future plans, culture, mission, and values;
5. decision and offer.

That supports process preparation, not a fixed Google DeepMind question set
[[R33]](#ref-r33). The recruiter and role-specific packet are authoritative for the scheduled loop.

<a id="d10-6-1"></a>
#### D10.6.1 · Build a Google DeepMind discussion overlay

**Status.** Based on official careers guidance.

**Worksheet**

- **Recruiter call:** [ninety-second career spine, why this role now, questions]
- **Possible hiring manager:** [two matching projects, capability evidence, team unknowns]
- **Skills interviews:** [packet-named competencies, one cold mock per format]
- **Final interviews:** [team goals, future research direction, collaboration and judgment stories]
- **Interviewing team's recent work:** [two current official artifacts]
- **Authority:** [recruiter-confirmed rounds, tools, presentation, AI policy, checked date]
- **Deliberate unknowns:** [anything the official page does not specify]

**Practice follow-ups.** The hiring-manager stage is omitted; the skills packet differs from public
accounts; the team changes before finals; a leadership interviewer asks about a research direction
outside your prepared projects.

**Trap.** Turning “typical overview” into a guaranteed sequence or inferring exact technical,
behavioral, coding, or values questions from the stage names.

---

<a id="section-refs"></a>

## References

Lab policies and interview formats change. Official pages below were checked on 2026-08-13; re-check
the recruiter instructions and current page immediately before an interview. First-hand accounts
describe individual experiences, not universal company policy.

<a id="ref-r1"></a>
[R1] Anthropic. *Guidance on Candidates' AI Usage.* Last updated July 10, 2025. [Official candidate guidance](https://www.anthropic.com/candidate-ai-guidance)

<a id="ref-r2"></a>
[R2] Anthropic. *Careers: How We Hire.* [Official careers and interview guidance](https://www.anthropic.com/careers)

<a id="ref-r3"></a>
[R3] Anthropic. *Company.* [Official mission and published principles](https://www.anthropic.com/company)

<a id="ref-r4"></a>
[R4] OpenAI. *OpenAI Interview Guide.* [Official interview guidance](https://openai.com/interview-guide/)

<a id="ref-r5"></a>
[R5] OpenAI Forum. *Careers at the Frontier: Hiring the Future at OpenAI.* Official recruiting event replay. [Event page and transcript](https://forum.openai.com/en/public/videos/event-replay-careers-at-the-frontier-hiring-the-future-at-openai)

<a id="ref-r6"></a>
[R6] Meta Careers. *Meta Hiring Process.* [Official hiring and AI-interview guidance](https://www.metacareers.com/hiring-process/)

<a id="ref-r7"></a>
[R7] Meta Careers. *Preparing for Your Initial Machine Learning Interview.* [Official guide landing page](https://www.metacareers.com/ML-prep-initial)

<a id="ref-r8"></a>
[R8] Meta Careers. *Preparing for Your Full Loop Machine Learning Interview.* [Official guide landing page](https://www.metacareers.com/ML-prep-onsite/)

<a id="ref-r9"></a>
[R9] Liu, Alisa. *Notes on the Industry Job Search.* June 2026. [Named first-hand account](https://alisawuffles.github.io/blog/job-search/)

<a id="ref-r10"></a>
[R10] Sapora, Silvia. *ML Job Interviews: The Ultimate Guide.* June 2026. [Named first-hand account](https://silviasapora.github.io/blog/ml-interviews.html)

<a id="ref-r11"></a>
[R11] Zheng-Xin, Yong. *Surprising Lessons from My Research Scientist Job Search.* June 24, 2026. [Named first-hand account](https://yongzx.github.io/blog/2026/06/24/job-search/)

<a id="ref-r12"></a>
[R12] Lambert, Nathan. *Job Hunt as a PhD in AI / ML / RL: How It Actually Happens.* July 2022. [Named first-hand candidate advice](https://natolambert.com/writing/ai-phd-job-hunt)

<a id="ref-r13"></a>
[R13] Meng, Yuan. *MLE Interview 2.0: Research Engineering and Scary Rounds.* 2026. [Named first-hand account](https://www.yuan-meng.com/posts/mle_interviews_2.0/)

<a id="ref-r14"></a>
[R14] Jaiswal, Mimansa. *LLM (ML) Job Interviews — Resources.* 2024–2025. [Named preparation record](https://mimansajaiswal.github.io/posts/llm-ml-job-interviews-resources/)

<a id="ref-r15"></a>
[R15] Grigorev, Alexey. *AI Engineering Field Guide.* Living repository; accessed 2026-08-13. [Traceable aggregated field guide](https://github.com/alexeygrigorev/ai-engineering-field-guide)

<a id="ref-r16"></a>
[R16] Huyen, Chip. *Introduction to Machine Learning Interviews.* 2021. [Open interview book](https://huyenchip.com/ml-interviews-book/)

<a id="ref-r17"></a>
[R17] Bekman, Stas. *Machine Learning Engineering Open Book.* [Open engineering reference](https://github.com/stas00/ml-engineering)

<a id="ref-r18"></a>
[R18] Google. *Rules of Machine Learning: Best Practices for ML Engineering.* [Official engineering guide](https://developers.google.com/machine-learning/guides/rules-of-ml)

<a id="ref-r19"></a>
[R19] Breck, Eric, Shanqing Cai, Eric Nielsen, Michael Salib, and D. Sculley. *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.* Proceedings of IEEE Big Data, 2017. DOI: 10.1109/BigData.2017.8258038. [Google Research publication page](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)

<a id="ref-r20"></a>
[R20] Keshav, S. *How to Read a Paper.* ACM SIGCOMM Computer Communication Review, 2007. [Author manuscript](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)

<a id="ref-r21"></a>
[R21] NeurIPS. *Paper Checklist Guidelines.* [Official author checklist](https://neurips.cc/public/guides/PaperChecklist)

<a id="ref-r22"></a>
[R22] Kohavi, Ron, Diane Tang, and Ya Xu. *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing.* Cambridge University Press, 2020. [Authors' companion site](https://experimentguide.com/)

<a id="ref-r23"></a>
[R23] NIST/SEMATECH. *Engineering Statistics Handbook.* [Official stable project page](https://www.nist.gov/programs-projects/nistsematech-engineering-statistics-handbook)

<a id="ref-r24"></a>
[R24] Sculley, D., et al. *Hidden Technical Debt in Machine Learning Systems.* NeurIPS, 2015. [Proceedings page](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)

<a id="ref-r25"></a>
[R25] Zhang, Jiaxin. *Personal Research Homepage.* [Public biography and selected work](https://jxzhangjhu.github.io/)

<a id="ref-r26"></a>
[R26] Zhang, Jiaxin. *Publications.* [Public bibliography](https://jxzhangjhu.github.io/publications/)

<a id="ref-r27"></a>
[R27] Anthropic. *Scaling Managed Agents: Decoupling the Brain from the Hands.* April 8, 2026. [Official engineering article](https://www.anthropic.com/engineering/managed-agents)

<a id="ref-r28"></a>
[R28] Anthropic. *Effective Harnesses for Long-Running Agents.* November 26, 2025. [Official engineering article](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

<a id="ref-r29"></a>
[R29] Model Context Protocol. *Specification.* Version 2026-07-28. [Official specification](https://modelcontextprotocol.io/specification/2026-07-28)

<a id="ref-r30"></a>
[R30] A2A Protocol Project. *Agent2Agent Protocol Specification.* Version 1.0.0. [Official specification](https://a2a-protocol.org/v1.0.0/specification/)

<a id="ref-r31"></a>
[R31] Booth, Harold, William Fisher, Ryan Galluzzo, and Joshua Roberts. *Accelerating the Adoption of Software and Artificial Intelligence Agent Identity and Authorization.* NIST NCCoE concept paper, February 5, 2026. [Official NIST page](https://csrc.nist.gov/pubs/other/2026/02/05/accelerating-the-adoption-of-software-and-ai-agent/ipd)

<a id="ref-r32"></a>
[R32] OpenAI. *Computer Use.* [Official developer guide](https://developers.openai.com/api/docs/guides/tools-computer-use)

<a id="ref-r33"></a>
[R33] Google DeepMind. *Careers: Our Interview Process.* Accessed 2026-08-13. [Official careers page](https://deepmind.google/careers/)
