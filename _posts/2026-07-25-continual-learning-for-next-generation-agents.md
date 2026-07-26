---
layout: post
title: "Continual Learning for Next-Generation Agents"
date: 2026-07-25 10:00:00
author: Jiaxin Zhang
description: "How next-generation agents should route deployment experience across context, memory, harnesses, weights, and the learning mechanism itself."
tags: agents continual-learning llm self-improvement
categories: research-notes
giscus_comments: true
related_posts: false
ai_assisted: true
og_image: https://jxzhangjhu.github.io/assets/img/blog/continual-learning-for-next-generation-agents/fig1_multitimescale_stack.png
---

<div class="lang-switch"><strong>English</strong> · <a href="/blog/2026/continual-learning-for-next-generation-agents-zh/">中文</a></div>

### Table of Contents

- [Why agents must learn after deployment](#why-agents-must-learn-after-deployment)
- [A vocabulary that does not collapse everything together](#a-vocabulary-that-does-not-collapse-everything-together)
- [The multi-timescale learning stack](#the-multi-timescale-learning-stack)
  - [Context: learn now, forget safely](#context-learn-now-forget-safely)
  - [Memory: persist with provenance](#memory-persist-with-provenance)
  - [Harness and skills: turn lessons into procedures](#harness-and-skills-turn-lessons-into-procedures)
  - [Adapters and weights: consolidate what transfers](#adapters-and-weights-consolidate-what-transfers)
  - [The updater: learn how to learn](#the-updater-learn-how-to-learn)
- [The deployment learning loop](#the-deployment-learning-loop)
- [What survives from classical continual learning](#what-survives-from-classical-continual-learning)
- [What works today](#what-works-today)
- [Failure modes worth naming](#failure-modes-worth-naming)
- [From continual learning to self-improvement and RSI](#from-continual-learning-to-self-improvement-and-rsi)
- [How to evaluate compounding learning](#how-to-evaluate-compounding-learning)
- [A research agenda](#a-research-agenda)
- [Summary](#summary)
- [How to cite](#how-to-cite)
- [References](#references)

---

## Why agents must learn after deployment

Imagine hiring a brilliant new research assistant. On day one, they know machine learning, can code, and have
read more papers than anyone in the lab. But they do not yet know that your group reports uncertainty with
bootstrap intervals, that an internal dataset changed its schema last month, that one benchmark has a hidden
leak, or that “make the figure cleaner” means preserving the color convention used in the last three papers.
They become useful not because their general intelligence changes every afternoon, but because they
**accumulate situated experience**: corrections, outcomes, exceptions, routines, and a model of how this
particular organization works.

Now compare that with a modern agent. It can spend an hour searching, coding, calling tools, failing,
recovering, and finally solving a task—and begin the next session as if none of that happened. We spend
enormous inference-time compute producing experience, then throw most of it away.

The waste is easy to underestimate because each individual loss looks small. A long agent session discovers
which of five plausible APIs actually exists, which test is flaky, which internal document is out of date,
and which three-step recovery works after a specific error. None of that is in the training data, none of it
is in the final answer, and by default none of it survives the session. The next run rediscovers the same
facts at the same cost, and a user who watched the agent learn something yesterday reasonably expects not to
teach it again today.

That gap is showing up in conversations that otherwise have little in common. A
[July 2026 *Daily Economic News* report](https://www.nbd.com.cn/articles/2026-07-23/4504670.html)
on an unofficial, AI-organized transcript of a May 2026 investor meeting attributes to Liang Wenfeng the
view that continual learning is the problem to solve after agents: a model should keep learning over long
periods, more like an employee adapting inside an organization. The outlet said an investment institution
participating in DeepSeek's financing confirmed that a May meeting took place and regarded the content as
credible, but no audio or official DeepSeek transcript is public; treat it as a
**media-reported, institution-corroborated unofficial record**, not a verbatim roadmap. DeepSeek's
[official recruiting page](https://talent.deepseek.com/) nevertheless provides independent evidence that the
topic is real organizational work: it lists a `Frontier（持续学习/自进化/新范式）研究员` role.

David Silver and Richard Sutton approach the same gap from reinforcement learning. Their
[**Era of Experience**](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
argues that the next generation of agents should learn predominantly from long-lived, environmentally grounded streams of
interaction—not only from fixed corpora of human-produced data. Andrej Karpathy is more cautious about the
timeline: in his [Dwarkesh interview](https://www.dwarkesh.com/p/andrej-karpathy), the fact that “you can't
just tell them something and they'll remember it” is one reason the *year* of agents looks more like a
*decade* of engineering. Dario Amodei offers a useful counterpoint in a
[separate interview](https://www.dwarkesh.com/p/dario-amodei-2): perhaps very long context and stronger
in-context learning will cover much of on-the-job learning before persistent weight updates are solved. A
million tokens, after all, can hold days of reading.

These positions are less contradictory than they appear. They concern **different clocks**:

- context can adapt behavior within a session;
- memory can carry selected experience across sessions;
- skills and harness code can encode verified procedures;
- adapters and weights can compress patterns that transfer broadly;
- an outer learning mechanism can improve how all of those updates are chosen.

The question is therefore not *“Should the agent learn in context or in its weights?”* The question is:

> **Given one piece of deployment experience, where should it be written, at what scope, for how long, and
> with what evidence before it is allowed to affect more users?**

This is the central claim of the post:

> **Thesis.** Continual learning for agents is no longer only “update parameters without catastrophic
> forgetting.” It is the problem of **routing experience to the right update surface and consolidating it
> across multiple timescales**—while preserving transfer, privacy, calibration, safety, and rollback.

![One agent, five learning surfaces, and multiple clocks.](/assets/img/blog/continual-learning-for-next-generation-agents/fig1_multitimescale_stack.png)
*Figure 1. One agent has several learning surfaces. Fast surfaces are local and reversible; slow surfaces
are shared and powerful but need stronger validation and governance. “Real-time feedback” should enter
quickly, not necessarily reach shared weights quickly. The clocks and scopes shown are typical design
choices, not intrinsic constraints: a memory can be fleet-wide, an adapter can be personal, and a router can
run on the hot path.*

We will carry one running example throughout: **a research agent joining an AI lab**.

- In its first session, a user corrects a project-specific citation rule.
- During its first week, it learns recurring project facts, people, datasets, and preferences.
- During its first month, repeated successful workflows become a verified literature-search or
  ablation-analysis skill.
- Across many projects, some pattern survives held-out evaluation and becomes adapter- or weight-training
  material.
- Across many releases, the system learns which feedback sources and update paths actually predict
  transfer; now the *updater itself* changes.

Putting every correction into shared weights would be unsafe, slow, and often wrong. Keeping everything in
the current context would make every new session the agent's first day at work. The design problem is the
space between those two failures.

---

## A vocabulary that does not collapse everything together

“Continual learning,” “online learning,” “lifelong learning,” “test-time learning,” “self-improvement,” and
“RSI” are often used as if they describe one escalating capability. They do not. A useful vocabulary asks a
different question for each term.

**Online learning** asks about *cadence*: does the learner process examples sequentially and update as the
stream arrives? An online algorithm may optimize cumulative reward or regret without preserving every old
skill. Conversely, a system can do continual learning through periodic batches rather than by updating after each
example.

**Continual learning (CL)** asks about *competence over a changing stream*: can the system acquire useful
new behavior, retain what still matters, and transfer experience forward under bounded resources? The classic
objective combines the **stability–plasticity trade-off**, intra/inter-task generalization, and efficiency
([Wang et al., 2024](https://arxiv.org/abs/2302.00487)). Catastrophic forgetting is the stability failure.
There is an equally important mirror image: **loss of plasticity**, where a repeatedly trained network retains
old behavior but gradually loses its ability to learn new behavior
([Dohare et al., 2024](https://arxiv.org/abs/2306.13812)).

**Continuous learning** and **continuous training** are usually engineering terms for a recurring pipeline:
collect data, retrain, validate, and release. A weekly batch can be “continuous” operationally while using no
continual-learning algorithm. It can even retrain from scratch each time. Seen from industry, this is really
an update-and-release problem inside a versioned model ecosystem, which is the framing
[Jiang et al. (2026)](https://arxiv.org/abs/2606.24901) use to argue that plasticity headroom, capability
inheritance, and accountability are lifecycle concerns rather than purely algorithmic ones.

**Lifelong learning** is the broader aspiration: an open-ended learner accumulates knowledge and skills over
its operating life. Much of the literature uses *lifelong*, *continual*, and *incremental* learning
interchangeably; the
[agent-focused roadmap by Zheng et al. (2026)](https://doi.org/10.1109/TPAMI.2025.3650546) is a useful map of
that literature and of how perception, memory, and action couple in changing environments. Here, *lifelong*
names the destination and *continual learning* names the technical problem.

**In-context learning, test-time adaptation, and test-time learning** ask whether behavior changes during
deployment. The change may live only in context, in an external memory, in a prompt, or in temporary
parameters. Each of those counts as a learning mechanism, but none is automatically persistent. A system
that does better on its fifth attempt than its first has adapted; it has not necessarily become better for
tomorrow's different task.

**Self-improvement** asks who produces the candidate improvement. A model that critiques and rewrites one
answer performs bounded self-refinement. An agent that edits a reusable skill and keeps the better version
performs persistent self-improvement. Neither is recursive merely because the same model appears on both
sides of the loop.

**Self-evolution** has no single settled definition. It often denotes persistent structural change—to
memory, skills, workflows, code, populations, or environments—selected by an outer loop. In the companion
post on [self-evolving agentic harnesses](https://jxzhangjhu.github.io/blog/2026/self-evolving-agentic-harnesses/),
the mutable object is the software around a frozen model.

**Recursive self-improvement (RSI)** asks about the *topology of the improvement loop*: does the system
improve the updater, evaluator, curriculum, research process, or successor-building machinery that produces
its next improvement? Continual learning describes how competence changes over time. RSI describes whether
the improvement process has become one of its own improvement targets.

Two axes prevent most category errors:

1. **Persistence:** transient within one trajectory → durable across sessions → inherited by future versions.
2. **Object of change:** current knowledge/behavior → memory/harness/weights → the learning mechanism itself.

Placing the terms on those two axes makes the distinctions mechanical rather than rhetorical:

| Term | Question it answers | Typical persistence | Object of change |
|---|---|---|---|
| Online learning | cadence of updates | varies | usually weights |
| Continual learning | competence on a changing stream | across sessions or versions | any durable surface |
| Continuous training | pipeline schedule | across releases | weights, by retraining |
| Lifelong learning | the long-run aspiration | operating lifetime | any surface |
| In-context / test-time learning | does behavior change during deployment | often within one episode | context or temporary parameters |
| Self-improvement | who proposes the change | bounded or durable | artifact, memory, harness, weights |
| Self-evolution | what an outer loop selects over | durable | structure: memory, skills, code, population |
| RSI | topology of the improvement loop | inherited by successors | the updater and the research process itself |

Retrieval, reflection, fine-tuning, and self-revision can all participate in continual learning. None alone
guarantees cumulative learning. And continual learning is plausible infrastructure for RSI, not evidence that
RSI has happened.

> **A practical test.** After an apparent improvement, reset the context and present a novel future task.
> What state carries the improvement forward? Does it transfer? What did it forget? If the answer is
> “nothing,” you observed refinement, not continual learning.

---

## The multi-timescale learning stack

Represent the deployed agent at time $$t$$ as

$$A_t = (C_t, M_t, H_t, \theta_t, U_t),$$

where:

- $$C_t$$ is context and working state;
- $$M_t$$ is episodic, semantic, and procedural memory;
- $$H_t$$ is the harness—prompts, tools, control flow, skills, and validators;
- $$\theta_t$$ is adapters and shared model weights;
- $$U_t$$ is the update mechanism: router, evaluator, curriculum, and training/search recipe.

This is not a claim that every product needs five separate databases. It is a way to make the **location of
learning** explicit. Each surface has a different write latency, scope, capacity, rollback mechanism, and
failure mode.

| Surface | Typical write latency | Natural scope | How you undo it | Characteristic failure |
|---|---|---|---|---|
| Context $$C_t$$ | immediate | one session | end the session | dilution; an early error becomes a premise |
| Memory $$M_t$$ | seconds to minutes | user, project, org | delete or supersede the entry | stale, contradictory, or distracting recall |
| Harness $$H_t$$ | hours to days | product surface | revert the commit | overfitted procedures; brittle tool contracts |
| Weights $$\theta_t$$ | days to weeks; hours for narrow signals | everyone on the checkpoint | roll back the checkpoint | forgetting; calibration and safety drift |
| Updater $$U_t$$ | weeks to months | every future update | restore the previous policy | systematically wrong promotions |

Read the table as a cost gradient rather than a hierarchy. Moving up, writes get cheaper and easier to
justify; moving down, they get more powerful, more dangerous, and more demanding of evidence. Most
production disagreements about "should the agent learn this?" are really disagreements about which row is
being proposed.

Three shorthand names for that gradient recur below. The **hot path** writes into context or isolated memory
within seconds. The **warm path** promotes a verified lesson into durable memory, a skill, or harness code.
The **cold path** changes adapters, shared weights, or the updater itself. A later section gives each one its
release requirements.

### Context: learn now, forget safely

Context is the fastest surface. A correction can change the next action without any training job or durable
write:

> “For this project, use the preregistered split, not the public leaderboard split.”

For the research agent, the right immediate response is to place that instruction in working state, revisit
the current plan, and perhaps repair work already done. That last step is the one most systems skip: a
correction that only changes future actions leaves behind an inconsistent artifact—half the results table
computed on the wrong split. Fast learning includes retroactive repair within the current episode.

Context is ideal for tentative beliefs, local tool state, and one-off constraints because it is cheap and
reversible. In practice, four things belong in working state: the current plan and its open questions, the
constraints the user has asserted, the environment facts discovered this session (which command works, which
path exists), and the failures already ruled out. Each is either cheap to rediscover or too unverified to
promote.

Its limitations are equally important. Context ends, long histories dilute relevant evidence, and an early
mistake can become the premise for every later step. Longer context increases the amount a model *can*
condition on; it does not decide what deserves persistence, resolve contradictions, or guarantee that a fresh
session starts with the lesson. This is why long context and continual learning are complements rather than
substitutes.

The economics reinforce the same conclusion. Re-reading a large history on every turn is a recurring cost
paid per session and per user, whereas consolidation is a cost paid once. When the same 40 lines of hard-won
project convention are re-derived in every session, the cheaper design is to write them down; when a
constraint applies to exactly one conversation, paying for context is correct. "Where should this live?" is
partly a caching question with an accuracy penalty attached.

### Memory: persist with provenance

Memory crosses the session boundary. It should not be a bag of old messages. A useful agent memory has at
least three roles:

- **episodic:** what happened in a particular run, with the model/harness version and outcome;
- **semantic:** relatively stable facts and preferences abstracted across episodes;
- **procedural:** a reusable lesson about how to act.

The distinction between recall and learning matters. [Evo-Memory](https://arxiv.org/abs/2511.20857) puts it
cleanly: conversational recall retrieves *what was said*; experience reuse retrieves *what was learned*—for
example, the quadratic formula rather than a previous equation's roots. The latter requires abstraction,
selection, and revision.

Every durable memory should carry metadata: source, timestamp, scope, confidence, evidence, expiry, and
links to contradicting or superseding entries. [ARIA](https://arxiv.org/abs/2507.17131) is a useful concrete
pattern: a deployed agent identifies a knowledge gap, asks a human expert a targeted question, writes the
guidance into a timestamped repository, compares it with related entries, and requests clarification when
rules conflict.

Written out, a durable entry looks less like a chat log and more like a small record with an owner and an
expiry date:

```yaml
id: mem_4193
kind: procedural
claim: "Run the leakage check before comparing retrieval methods on the internal QA set."
scope: {project: qa-retrieval, users: all}
source: {type: human_correction, actor: reviewer, session: s_2291}
evidence: [run_8842_failed_review, run_8871_passed_after_check]
confidence: 0.82
created: 2026-05-02
review_by: 2026-11-02
supersedes: mem_3907
status: active
```

The metadata is not bureaucracy; each field answers a question the learning loop must eventually ask.
`scope` decides who is affected, `source` and `evidence` decide how much the entry should be trusted when it
conflicts with another, `review_by` forces stale rules to expire rather than accumulate, and `supersedes`
keeps the history auditable when a rule changes. Without those fields, "resolve the conflict" degrades into
"whichever entry the retriever happened to return."

Four operations then define memory quality, and only one of them is retrieval:

- **write:** decide that something is worth persisting at all, and at what scope;
- **revise:** merge, sharpen, or contradict an existing entry when new evidence arrives;
- **retrieve:** surface the few entries relevant to the current step, not everything similar;
- **forget:** expire, prune, or delete on request, and propagate that deletion to anything derived from it.

For our lab agent, “Jiaxin prefers blue figures” might be scoped to a user; “Dataset v3 renamed `label` to
`target`” to a project and time range; “run leakage checks before comparing retrieval methods” might become
a procedural memory. Scope is part of correctness.

Memory is attractive because it is editable and deletable. It is also dangerous because retrieval feels like
free improvement. [LifelongAgentBench](https://arxiv.org/abs/2505.11942) finds that adding more experience
can *reduce* performance when irrelevant history consumes context or distracts the agent. Agent memory needs
garbage collection, conflict resolution, and evaluation—not just vector search.

### Harness and skills: turn lessons into procedures

Some lessons are best expressed as inspectable artifacts:

- a checklist for reviewing a paper;
- a script that verifies bibliography links;
- a tool wrapper that returns smaller, structured output;
- a routing rule that invokes a second model after repeated failures;
- a control-flow change that runs tests before claiming completion.

These belong to the **harness and skill layer**. Compared with weights, a harness change is cheap, readable,
diffable, and easy to roll back. Compared with raw memory, it can encode an executable procedure and its
validator. This is the warm path between “remember this” and “retrain the model.”

The companion [harness-evolution post](https://jxzhangjhu.github.io/blog/2026/self-evolving-agentic-harnesses/)
studies this layer in detail. Here the key point is its place in the larger stack: traces can first become
scoped memories, then a repeated and verified pattern can become a skill or tool. The promotion should
require evidence that the procedure fixes more than the triggering episode.

A workable promotion rule from memory to skill has four conditions, and the last one is the one usually
missing:

1. **recurrence:** the situation has appeared often enough that a procedure beats a lookup;
2. **determinism:** the right response is a sequence of steps, not a judgment that depends on the case;
3. **checkability:** the procedure can be validated by something other than the model that proposed it;
4. **non-interference:** running it in situations where it does not apply is harmless, or the skill knows
   its own preconditions.

Violating condition 4 is how skill libraries decay. A skill written to fix one failure often fires in
adjacent situations it was never tested on, and the cost lands on unrelated tasks. Stating preconditions explicitly
inside the skill—and testing the skill on cases where it should *not* fire—is what keeps a growing library
from becoming a slow, contradictory second model.

### Adapters and weights: consolidate what transfers

Weights are valuable because they compress behavior. A lesson internalized into an adapter or shared model
requires no retrieval and can generalize in ways a literal memory cannot. But a weight update has a far
larger blast radius than a memory or harness change, and the weakest local reversibility.

This surface is appropriate only when a pattern is:

1. supported by many attributable experiences;
2. useful beyond one user or project;
3. stable enough not to expire next week;
4. legal and private enough to train on;
5. able to survive replay, held-out transfer, safety, and calibration checks.

The biological analogy is
[**complementary learning systems**](https://doi.org/10.1037/0033-295X.102.3.419): fast episodic storage
protects individual experiences while slower, interleaved consolidation extracts structure into semantic
memory. The analogy
does not specify an algorithm, but it gives the right systems principle. The memory-first position paper by
[Dorovatas et al. (2026)](https://arxiv.org/abs/2603.01761) makes the agent version explicit: combine rapid
in-context adaptation with low-frequency in-weight learning rather than forcing either surface to do both.

For the research agent, one successful way of formatting an ablation table should remain a skill or memory.
If the same procedure transfers across hundreds of projects and models, it may be ready for adapter or
weight consolidation.

It also helps to be concrete about what a consolidation job contains, because "update the weights" hides
most of the work. A realistic cold-path job assembles training targets from accepted experiences and their
verified outcomes; mixes in replay data representing capabilities that must not move; chooses a surface
(LoRA per scope, a merged adapter, or the shared checkpoint); trains; then evaluates against local repair,
held-out transfer, regression, safety, and calibration suites before any traffic moves. The training step is
often the cheapest and least uncertain part of that list.

Scope matters here too. A per-user or per-organization adapter keeps interference local and makes deletion
tractable, at the price of serving complexity and weaker sharing. A single shared checkpoint gets maximum
transfer and minimum isolation. The choice is not purely technical: it determines whether one customer's
idiosyncratic convention can degrade another customer's results.

### The updater: learn how to learn

The final surface is easy to miss because it is not “knowledge” in the usual sense. The system must decide:

- which feedback is trustworthy;
- which failure belongs to the model, memory, harness, tool, or environment;
- which experiences deserve replay;
- which candidate update is evaluated on which future tasks;
- when an update should be rolled back;
- how exploration and training resources should be allocated.

Call this mechanism $$U_t$$. A hand-written rule—“save every user correction to memory”—is an updater. A
learned model that routes feedback is an updater. So is an evolutionary process that optimizes a
test-time adaptation prompt across tasks.

A **calibrated update router** can be written as

$$R(e_t, q_t) \rightarrow (s, \sigma, \tau, a, c),$$

where $$e_t$$ is an experience; $$q_t$$ summarizes credibility, causal attribution, novelty, recurrence,
transferability, privacy, regression risk, and cost; $$s$$ is the target surface; $$\sigma$$ is scope; $$\tau$$ is a
time-to-live or review horizon; $$a$$ is write, quarantine, request evidence, stage, merge, or discard; and
$$c$$ is confidence in that route.

The features in $$q_t$$ are worth spelling out, because they are what a router can actually be trained or
evaluated on:

- **credibility:** who or what produced the signal, and does independent evidence agree?
- **attribution:** is the observed outcome caused by the model, memory, harness, tool, or environment?
- **novelty and recurrence:** is this new information, and has it now appeared enough times to generalize?
- **transferability:** does the lesson plausibly apply outside the case that produced it?
- **privacy and legality:** may this content persist, and at what scope?
- **regression risk:** what existing behavior could this change break?
- **cost:** what does writing, storing, retrieving, or training on it cost over its lifetime?

Framed this way, the router is a decision policy with an explicit budget rather than a classifier. Each route
carries an expected benefit (future tasks improved), an expected cost (compute, latency, review time), and an
expected risk (regression, leakage, poisoning). Choosing the smallest reversible write that captures most of
the benefit is a bandit-style problem with delayed, partially observed rewards—which is exactly why routing
deserves to be learned and measured rather than hard-coded once.

This connects to the [calibration post](https://jxzhangjhu.github.io/blog/2026/calibrating-long-horizon-agents/):
uncertainty should not only decide whether an agent acts. It should decide whether the resulting experience
is safe to remember and how far it may propagate.

The updater surface is also the bridge to meta-learning and RSI. Learning a new fact changes what the agent
knows. Learning which update policy produces future gains changes how the agent learns.

> **Takeaway.** The stack is not a ladder in which every memory eventually becomes a weight. Most
> experiences should expire locally. Promotion to a slower, broader, less reversible surface must demand
> stronger evidence.

---

## The deployment learning loop

The stack says *where* learning can happen. The loop says *how an experience earns a durable write*.

![The gated deployment learning loop.](/assets/img/blog/continual-learning-for-next-generation-agents/fig2_deployment_learning_loop.png)
*Figure 2. Capture is not learning, and feedback is not automatically truth. A deployment experience becomes
a release only after validation, routing, consolidation, and regression/safety gates. Rejected evidence can
be quarantined without blocking fast local adaptation.*

Across memory systems, production engineering reports, and weight-update papers, the same skeleton appears.

### 1. Capture the complete experience

Store more than the final answer. A useful record includes:

- task and relevant environment state;
- full action/tool trajectory;
- model, prompt, memory, tool, and harness versions;
- final and intermediate outcomes;
- explicit feedback and later downstream outcomes;
- identity, consent, privacy class, and retention policy.

Without versioned lineage, delayed feedback is nearly impossible to attribute. If a patch causes an incident
one week later, the learner must know which model and harness produced it, which tools were available, and
which later edits intervened.

The environment matters because experience is generated by a policy. Once an agent changes, it visits a
different distribution of states and receives different feedback. This is why the
[environment-scaling post](https://jxzhangjhu.github.io/blog/2026/environment-scaling-for-agentic-rl/)
and continual learning are two halves of one system: environments produce trajectories and verifiers;
continual learning decides what those trajectories should change.

### 2. Validate the signal and attribute the cause

Feedback is evidence, not ground truth.

- A user edit may encode a personal preference rather than a universal rule.
- A suggestion acceptance may mean “good enough to save time,” not “correct.”
- A failed tool call may be an expired credential, not a reasoning failure.
- A passing test may omit the behavior the user actually cares about.
- A model's self-critique is correlated with the blind spots of the model that made the error.

Validation therefore asks two questions:

1. **Can we trust the signal?** Check provenance, authority, privacy, adversarial risk, and agreement with
   independent evidence.
2. **What caused the outcome?** Attribute the failure to model knowledge, memory retrieval, context
   management, tool/interface, harness control flow, environment, or task ambiguity.

Different feedback types fail in different ways, and treating them uniformly is how learning loops go wrong:

| Feedback type | Example | Main weakness | Reasonable default |
|---|---|---|---|
| Explicit correction | "use the preregistered split" | may encode one person's preference | write at the narrowest scope that fits |
| Implicit behavioral | accept, reject, edit, retry | proxy for value, not correctness | aggregate before believing it |
| Executable verifier | tests, type checks, schema validation | may not cover the real objective | strong signal inside its coverage |
| Delayed outcome | an incident one week later | attribution is genuinely hard | quarantine until it recurs |
| Model self-critique | the agent grades itself | correlated with its own blind spots | never the sole gate for promotion |
| Adversarial input | injected instructions in a tool result | designed to look like evidence | treat untrusted content as data, never instruction |

The last row deserves emphasis, because memory turns a transient attack into a persistent one. If an agent
writes what it read in a web page or tool output into durable memory, an attacker who controls that content
controls part of the agent's future instructions. Provenance is therefore a security boundary, not only a
bookkeeping convenience.

[ARIA](https://arxiv.org/abs/2507.17131)'s expert clarification is one solution for explicit domain feedback. Executable verifiers are stronger
when they cover the real objective. Delayed business outcomes require causal analysis and often should remain
quarantined until repeated.

### 3. Route by surface, scope, and lifetime

Routing should prefer the smallest reversible write that can solve the problem:

- **context** for a tentative or one-session correction;
- **namespaced memory** for user/project facts and evolving rules;
- **skill or harness** for a verified procedure;
- **adapter or weights** for broadly transferable patterns;
- **updater/evaluator** only when evidence spans many learning cycles.

The same content can take different routes depending on scope. “Always use APA style” might be a user
preference, a journal-specific project rule, or a global product default. A router that predicts the right
surface but the wrong scope is still wrong.

The output should include uncertainty. Low confidence does not have to stop learning: the hot path can place
the item in temporary context or quarantined memory while requesting evidence. What uncertainty should stop
is *promotion*.

Six experiences from one week of the research agent make the decision concrete:

| Experience | Surface | Scope | Lifetime | Gate before it counts |
|---|---|---|---|---|
| "For this run, use the preregistered split." | context | session | this episode | none; repair current work |
| "I prefer figures in the lab's blue palette." | memory | user | until changed | confirm it is preference, not policy |
| "Dataset v3 renamed `label` to `target`." | memory | project | until schema changes again | one independent confirmation |
| "Check leakage before comparing retrievers." | skill | project, then org | versioned | passes on cases beyond the trigger |
| Recurring citation-format repair across 200 sessions | adapter | fleet | until next checkpoint | replay, transfer, and regression suites |
| "This reviewer always wants shorter abstracts." | quarantine | pending | 30-day review | recurrence across reviewers before promotion |

The same sentence can occupy different rows depending on who said it and how often it has been confirmed.
That is the point: routing is a function of evidence and scope, not of the content's surface form.

### 4. Replay, contrast, and consolidate

Raw trajectories are poor long-term memory. Consolidation should:

- deduplicate repeated experiences;
- preserve counterexamples and failures, not only successes;
- contrast cases to identify what actually mattered;
- interleave old representative behavior to detect forgetting;
- distill episodes into facts, procedures, or training examples;
- retain links from every abstraction back to source evidence.

[Contextual Experience Replay (CER)](https://arxiv.org/abs/2506.06698) distills environment dynamics and
decision patterns from web-agent trajectories into a retrievable buffer. It is a useful demonstration that
“replay” for language agents can mean *replaying abstractions in context*, not necessarily taking a gradient
on old examples. For shared weight updates, replay is one option: interleaving representative old data or
old-policy behavior can constrain forgetting, while regularization, distillation, and parameter isolation
offer alternatives.

### 5. Gate on transfer and non-regression

An update that fixes its triggering example has passed the easiest test. A release should require three
different kinds of evidence:

1. **Local repair:** does it fix the observed failure?
2. **Held-out transfer:** does it help related future cases that were not used to propose the change?
3. **No critical regression:** does it preserve old capabilities, safety rules, calibration, privacy, latency,
   and cost within defined bounds?

This is where many self-improvement claims become ordinary overfitting. If the same trace proposes,
selects, and explains the update, the loop has no independent evidence.

Gates should be surface-specific. A private memory does not need fleet-wide capability testing, but it needs
scope and privacy checks. A tool change needs integration and adversarial tests. A weight update needs
representative replay, broad safety/calibration evaluation, and a checkpoint rollback plan.

| Surface | Local repair | Transfer evidence | Regression bar | Rollback |
|---|---|---|---|---|
| Memory entry | the case it came from | none required | correct scope and privacy class; no contradiction with existing entries | delete the entry |
| Skill or tool | the failing trace | cases beyond the trigger | integration and adversarial tests pass | revert the commit |
| Adapter | held-out slice of the same domain | held-out task families | protected capabilities within bounds | unload the adapter |
| Shared weights | same | same, plus fleet distribution | capability, safety, calibration, latency, cost | previous checkpoint |
| Updater policy | the routing errors it fixes | later update cycles | promotion precision does not fall | restore prior policy |

The asymmetry across rows is deliberate. Requiring fleet-level evidence for a personal memory makes the fast
path useless; accepting trigger-only evidence for a weight update is how a plausible fix becomes a
population-wide regression.

### 6. Stage the release

Scope can expand gradually:

`session → user → project → organization → canary cohort → fleet`.

At each step, compare the candidate with a stable control and preserve a quick rollback. A useful operational
principle is:

> **Fast ingestion is compatible with slow promotion.**

The agent can react to a correction in seconds without rewriting shared weights. A local write gives immediate
value while the system gathers the evidence needed for a warm or cold update.

### 7. Monitor, roll back, and meta-learn

Release does not end learning. Monitor for distribution drift, stale memories, unexpected interactions, group
regressions, and changes in the relationship between feedback and true outcomes. Rollback is a normal action,
not an admission that the learning loop failed.

A few signals are worth watching continuously rather than at release time: the share of retrieved memories
that are never used in the final answer, the age distribution of entries that still fire, the rate at which
new writes contradict existing ones, the fraction of promotions later reverted, and the gap between the
proxy signal (acceptance, thumbs-up) and any ground-truth outcome you can measure later. Each of those tends
to degrade slowly and silently, which is exactly why it needs a dashboard rather than an incident.

Over many updates, the system can ask a higher-order question: which sources, routes, replay policies, and
gates predicted durable transfer? Updating those rules changes $$U_t$$—the start of learning to learn.

This gives three deployment paths:

- **Hot path:** context or isolated memory; seconds; local and easy to undo.
- **Warm path:** verified memory promotion, skill, prompt, tool, or routing change; versioned and eval-gated.
- **Cold path:** adapter, shared weights, or updater; representative replay, broad regression testing, staged
  checkpoint release, and rollback.

Real-time feedback belongs at the entrance of all three paths. It does not imply real-time access to every
parameter.

---

## What survives from classical continual learning

Classical continual learning usually pictures one neural network receiving a sequence of tasks. The modern
agent is a compound system, but the old mechanisms remain useful once we map them to the expanded set of
mutable surfaces.

![Classical continual-learning mechanisms mapped to agent update surfaces.](/assets/img/blog/continual-learning-for-next-generation-agents/fig3_classical_cl_to_agent_surfaces.png)
*Figure 3. Replay, regularization, isolation, dynamic architecture, distillation, and meta-learning still
matter. The change is that the protected or updated object may be a memory, skill, tool, adapter, model, or
updater—not only one monolithic parameter vector.*

| Classical mechanism | Original form | Agent-era form | New failure mode |
|---|---|---|---|
| Replay | interleave stored old examples | trajectory buffers, regression suites, in-context experience reuse | stale or irrelevant experience crowds out the task |
| Regularization | penalize movement of important weights | non-regression constraints on any surface | protects the measured behavior, not the intended one |
| Parameter isolation | per-task parameters or masks | memory namespaces, per-scope adapters, sandboxed tools | routing errors; blocked positive transfer |
| Dynamic architecture | grow capacity per task | add memories, skills, tools, subagents | unbounded growth without pruning |
| Distillation | teacher-to-student transfer | session-to-weight and fleet-to-base consolidation | quietly copies the teacher's mistakes |
| Meta-learning | learn the learning rule | learn the router, evaluator, or adaptation policy | overfits the update policy to one benchmark |

### Replay becomes experience and regression infrastructure

[Experience replay](https://arxiv.org/abs/1811.11682) approximates joint training by mixing old examples with
new ones, and [Gradient Episodic Memory](https://arxiv.org/abs/1706.08840) sharpens the idea by using stored
examples as constraints rather than merely as extra data—an update is allowed only if it does not increase
loss on retained memories. For agents, “old examples” expand to:

- trajectories and environment states;
- successful and failed procedures;
- user/project memories with scope;
- tool-interface contracts;
- regression tasks and safety cases;
- outputs from previous checkpoints.

Replay has two jobs. It protects stability by representing old behavior, and it reveals whether a new lesson
transfers beyond its trigger. More replay is not always better: stale or irrelevant experiences increase
context and compute, violate privacy, and can block adaptation. Selection and coverage are the research
problem.

### Regularization becomes a non-regression constraint

[Elastic Weight Consolidation (EWC)](https://arxiv.org/abs/1612.00796) protects parameters important to old
tasks.
[Learning without Forgetting](https://arxiv.org/abs/1606.09282) takes the complementary route: instead of
protecting parameters, it constrains the *function*, keeping the new model's outputs close to the old model's
on the same inputs. That distinction transfers directly to agents, where what you want to preserve is almost
always behavior rather than any particular parameter. In an agent, the same idea can constrain:

- adapter/weight movement on protected capabilities;
- divergence from a previous model on a replay set;
- changes to a prompt, routing policy, or tool contract;
- safety and calibration behavior;
- memory retrieval for protected user or organization rules.

The more heterogeneous the surfaces, the more useful it is to think in terms of **behavioral constraints**
rather than one penalty on weights.

### Isolation becomes namespacing, adapters, and experts

Parameter-isolation methods allocate different capacity to different tasks; [PackNet](https://arxiv.org/abs/1711.05769)
is the canonical version, packing several tasks into one network by iteratively pruning and freezing
task-specific subsets. Agents already have natural versions:

- per-user or per-organization memory namespaces;
- project-specific skills;
- LoRA adapters or task vectors;
- mixture-of-experts routing;
- sandboxed tools and subagents.

Isolation reduces interference and improves deletion, but it creates a routing problem and can prevent
positive transfer. The hard part is deciding which knowledge is genuinely shared.

### Dynamic architecture expands beyond neurons

[Progressive networks](https://arxiv.org/abs/1606.04671) and other expansion methods add capacity rather than
overwrite old capacity, keeping frozen columns for previous tasks and learning lateral connections into them.
An agent can add:

- a memory module;
- a specialist subagent;
- a new tool or validator;
- a new expert/adapter;
- a branch in its workflow.

This makes growth easy and governance hard. Without consolidation and pruning, the system accumulates
contradictory memories, overlapping skills, and brittle tools. Lifelong learning needs *forgetting on
purpose* as well as retention.

### Distillation connects the clocks

Distillation is the most important bridge in the stack:

- trajectory → episode summary;
- episodes → semantic or procedural memory;
- memory → verified skill;
- long-context “senior agent” → fresh-session student;
- specialized adapters or fleet learners → shared base model.

[Self-Distillation Fine-Tuning (SDFT)](https://arxiv.org/abs/2601.19897) uses a
demonstration-conditioned version of the current model as teacher and a query-only version as student,
creating a learner-relative on-policy signal. More generally, distillation is how fast, explicit learning can
become slow, implicit competence without preserving every raw experience forever.

### Meta-learning optimizes the updater

The classical “learning to learn” question becomes concrete in an agent. Can we optimize the policy that
turns trajectories into reflections, memories, prompt edits, training data, or parameter updates?

[Meta-TTL](https://arxiv.org/abs/2604.00830) treats a natural-language test-time adaptation policy as the
object of an outer evolutionary search. It learns across training tasks how an agent should use previous
episodes, then freezes that adaptation policy on unseen tasks. This is not RSI—the outer objective and search
remain designed and bounded—but it is exactly the updater surface $$U$$.

### What no longer survives as an assumption

The traditional experimental setting often supplies:

- clean task boundaries;
- a task ID;
- immediate labels;
- a fixed data-generating process;
- one mutable model;
- an offline test set independent of deployment.

Deployed agents get none of these for free. Tasks overlap. Feedback is delayed and implicit. The policy
changes what data it sees. Users and tools change. Memory, harness, and weights co-adapt. Privacy, poisoning,
and rollback are part of the learning problem.

That is why simply applying EWC or replay to an LLM is not the solution. Classical CL contributes mechanisms;
agentic CL changes the **system boundary, evidence model, and release process**.

---

## What works today

Among the public systems surveyed here, none demonstrates open-ended continual learning across context,
memory, harness, weights, and updater. But important slices already work. The clearest field map is not
“memory papers versus self-improvement papers.” It is **where each system writes experience and what gate
controls promotion**.

| System | Writes to | Promotion gate | Reported evidence | Main caveat |
|---|---|---|---|---|
| [CER](https://arxiv.org/abs/2506.06698) | context buffer | retrieval relevance | 36.7% WebArena success, +51.0% relative | buffer quality is unmanaged over long horizons |
| [ARIA](https://arxiv.org/abs/2507.17131) | timestamped memory | human expert answer | author-reported TikTok Pay deployment | no independent audit of scale or effect |
| [CASCADE](https://arxiv.org/abs/2605.06702) | case bank + retriever parameters | success-only retention; bandit retrieval | +20.9% over zero-shot on 16 tasks | guarantees cover case selection, not open worlds |
| [Replit](https://replit.com/blog/evaluating-and-improving-agent-at-scale) | agent product/harness | offline evals + A/B + human review | first-party engineering report | interested-party evidence |
| [OpenAI Tax AI](https://openai.com/index/building-self-improving-tax-agents-with-codex/) | scoped worktree, evals | targeted + regression evals, human merge | first-party case study | "self-improving" means loop automation |
| [Continual Harness](https://arxiv.org/abs/2605.09998) | prompt, skills, memory, weights | none independent of the run | reset-free Pokémon runs | bounded game domain |
| [Cursor Tab](https://cursor.com/blog/tab-rl) | shared weights | aggregated online RL, versioned release | 21% fewer suggestions, +28% acceptance | acceptance is a proxy for value |
| [SEAL](https://arxiv.org/abs/2506.10943) | weights, via self-edits | outer RL on post-update performance | SQuAD no-passage 39.7% → 47.0% | earlier tasks degrade across sequential edits |
| [Meta-TTL](https://arxiv.org/abs/2604.00830) | the adaptation policy | outer evolutionary search, then frozen | Jericho ID 50.4 → 110.8; τ²-bench OOD 0.33 → 0.37 | outer loop is offline and bounded |
| [SDFT](https://arxiv.org/abs/2601.19897) | weights | none independent of the method | controlled sequential-skill retention | not a deployment-scale system |
| [Darwin Gödel Machine (DGM)](https://arxiv.org/abs/2505.22954) | its own code archive | benchmark-selected archive | 20.0% → 50.0% on a 200-task SWE-bench Verified subset | frozen models, fixed benchmarks |

### Context and memory: useful experience without touching the base model

[CER](https://arxiv.org/abs/2506.06698) is a clean example of the fast path. A web agent completes a task,
distills environment dynamics and decision patterns from the trajectory, merges them into a dynamic buffer,
retrieves relevant experiences for the next task, and replays them in context. It supports offline, online,
and hybrid sources of trajectories. On the paper's WebArena setup, CER reports a 36.7% average success rate—a
51.0% relative improvement over its GPT-4o baseline—and 31.9% on VisualWebArena.

That is meaningful evidence that inference-time experience can improve later agent behavior without changing
weights. It is not evidence of indefinite accumulation. The result depends on distillation quality, retriever
coverage, environment similarity, and the buffer not filling with misleading experience.

[ARIA](https://arxiv.org/abs/2507.17131) addresses a different weakness: feedback does not arrive
automatically when domain rules change. The agent uses structured self-dialogue to identify uncertainty,
asks an expert a targeted question, and maintains timestamped knowledge with conflict detection. The paper
says ARIA is deployed within TikTok Pay, a service it describes as serving more than 150 million monthly
active users; it does not report how many users or decisions ARIA itself handled. Treat both deployment and
platform scale as author-reported, but the architecture is valuable: **uncertainty triggers a scoped human
query; the answer enters a conflict-aware memory, not global weights**.

[CASCADE](https://arxiv.org/abs/2605.06702) makes case reuse more formal. It treats deployment-time learning
as a third lifecycle stage after pretraining and fine-tuning, keeps the foundation model frozen, and learns
which episodic cases to retrieve using a contextual bandit. Only cases from reward-positive interactions are
retained. Worth stating precisely: "no parameter updates" refers to the *base model*. The auxiliary
retriever is trained online by gradient descent, so CASCADE is better described as frozen-LLM learning than
as gradient-free learning. Success-only retention is also its exposed surface—an answer rewarded for the
wrong reason becomes a trusted exemplar. Across the paper's 16-task suite, it reports a **20.9% relative improvement** in macro-averaged
success rate over zero-shot prompting. Its no-regret analysis applies to the formalized case-selection
process under stated assumptions—not to arbitrary open-world learning—but it shows what memory research
gains from an explicit long-run objective.

Together these systems clarify three points:

1. memory can produce real cross-episode adaptation while the base model stays frozen;
2. memory quality depends on **write, revise, retrieve, and forget**, not retrieval alone;
3. fast learning remains local and explicit, which is why it is easier to inspect and revoke.

### Harnesses: turn production failures into versioned changes

The warm path is already visible in production engineering.

Replit's official report,
[“Closing the loop: Evaluating and improving Replit Agent at scale”](https://replit.com/blog/evaluating-and-improving-agent-at-scale),
describes two measurement pillars and an optimization loop. Offline benchmarks test candidates before
release; A/B tests and production traces show what happens after release; failure clusters become hypotheses
for an agent to implement as a draft pull request. The candidate is compared with benchmarks, trajectory
data, A/B evidence, and recent baselines before a ship/iterate/drop recommendation. Engineers retain release
authority. This is persistent self-improvement, but the persistent object is the **agent product and harness**,
not an online-changing base model.

OpenAI's
[Tax AI case study](https://openai.com/index/building-self-improving-tax-agents-with-codex/) describes a
similar pattern with unusually clear boundaries. Reviewed production findings, source traces, expected
outputs, tax-engine documentation, and eval commands are packaged as a bounded Codex task. The candidate
can write only to a scoped worktree containing targeted and regression evals; production evidence is
read-only. Validation and human review stay part of the task environment. The label *self-improving* here means
“automation closes more of the engineering loop,” not “the model autonomously rewrites its weights.”

[Continual Harness](https://arxiv.org/abs/2605.09998) is a research counterpart in an embodied domain. A
refiner reads recent Pokémon trajectories and edits the system prompt, subagents, skills, and memory during
one reset-free run. A second experiment also updates an open model through process-reward relabeling and
soft supervised fine-tuning. The paper is useful precisely because it puts harness-only and
harness-plus-weight learning in the same topology. Its bounded game setting and paper-specific model results
should not be generalized to arbitrary deployment, but the architecture makes co-evolution concrete.

### Weights: narrow signals can support fast cold-path releases

The clearest production example of frequent weight updates in this source set is Cursor.

In [“Improving Cursor Tab with online RL”](https://cursor.com/blog/tab-rl), the action and feedback are
narrow: show or suppress a next-edit suggestion; observe accept or reject. The service handles enough
interactions to train on fresh policy data. Cursor reports that its resulting model displayed 21% fewer
suggestions while the suggestions it did show had a 28% higher acceptance rate. The newer
[Composer real-time RL report](https://cursor.com/blog/real-time-rl-for-composer) describes broader
production-interaction reward signals and a pipeline that can ship a new checkpoint as often as every five
hours.

This is an existence proof for a fast cold path—not a proof that arbitrary feedback should update weights.
The signal is frequent, the action is well-instrumented, and aggregation happens before a versioned
checkpoint release. Acceptance also remains a proxy: it does not fully measure correctness, maintainability,
or downstream incidents.

Research methods are attacking the update itself:

- [SEAL](https://arxiv.org/abs/2506.10943) lets a model generate a “self-edit”: synthetic training data and,
  optionally, optimization directives. Supervised fine-tuning applies the persistent update; an outer RL
  loop rewards self-edits by post-update performance. In its single-passage knowledge-incorporation
  experiment, no-passage SQuAD accuracy is 32.7% for the frozen base and 39.7% for self-edits from the
  untrained policy; learning to generate self-edits raises it to 47.0%. The paper also notes that earlier
  tasks degrade across sequential self-edits, so it demonstrates persistence, not non-regression.
- [SDFT](https://arxiv.org/abs/2601.19897) converts expert demonstrations into learner-relative on-policy
  distillation, reducing the off-policy mismatch of ordinary supervised fine-tuning and preserving old
  capabilities better in its controlled sequential-skill experiments.
- [Agent-Dice](https://arxiv.org/abs/2601.03641) assumes task-specific parameter updates already exist, then
  filters conflicting directions and amplifies shared directions before fusion. It is a useful cold-path
  answer to “which update components are common?” but does not solve feedback collection or causal
  attribution.

Each moves beyond “fine-tune on the newest data.” None yet combines production-grade provenance, delayed
feedback, personal/fleet scope, continual alignment, and long-run plasticity.

### The updater: the first pieces of learning to learn

[SEAL](https://arxiv.org/abs/2506.10943)'s outer loop learns what inner-loop training data to generate. [Meta-TTL](https://arxiv.org/abs/2604.00830) learns a natural-language policy
for adapting across test-time episodes. The
[DGM](https://arxiv.org/abs/2505.22954) goes in a different direction: coding agents
edit their own harness/code. Variants that compile and retain code-editing ability are benchmarked and
archived; performance- and underexploration-weighted parent selection lets later agents build from those
stepping stones.

These are important because the mutable object is partly the **improvement mechanism**. They are still
bounded:

- SEAL's reward and task setup are externally specified.
- Meta-TTL's outer evolutionary loop is trained offline and frozen at test time.
- DGM uses fixed coding benchmarks and frozen foundation models; its archive-maintenance and parent-selection
  process is fixed, foundation-model/training-script evolution is future work, and the experiments use
  sandboxing and human oversight.

The honest 2026 scorecard is therefore:

> We can make agents reuse experiences, maintain changing domain knowledge, improve versioned harnesses,
> frequently update weights under narrow production signals, and learn bounded adaptation policies. We do
> not yet have a trustworthy system that decides among all of these surfaces under delayed open-world
> feedback and improves that decision process indefinitely.

---

## Failure modes worth naming

Most continual-learning systems do not fail loudly. They degrade in ways that look like success for a while,
which is why the failures deserve names.

**Memory bloat.** Every experience is written, nothing expires, and retrieval quality falls as the store
grows. The system looks like it is learning—the memory count goes up—while task performance drifts down.
[LifelongAgentBench](https://arxiv.org/abs/2505.11942) shows the measurable version: adding more history can
reduce success. The fix is unglamorous: expiry dates, deduplication, and pruning treated as first-class
operations rather than maintenance chores.

**Preference laundering.** One user's stylistic preference is written at global scope, and later it is
indistinguishable from a verified rule because both are just entries in the same store. Scope errors are
harder to detect than factual errors because nothing about the content is wrong.

**Proxy capture.** The system optimizes the signal it can measure—acceptance, thumbs-up, tests passing—and
slowly diverges from the outcome anyone cares about. Acceptance rate is the clearest example: it is a
behavioral proxy that does not fully measure correctness, maintainability, or downstream incidents. The
defense is to keep at least one slower, independent outcome measure that the learning loop cannot optimize
directly.

**Self-confirming evidence.** The same model proposes the update, evaluates it, and writes the summary of
why it worked. The loop then has no independent evidence at all. This is the most common way a
"self-improvement" result turns out to be overfitting to the trigger case.

**Silent poisoning.** Untrusted content read during a task becomes a durable memory, and from then on the
attacker's text arrives as trusted context. The dangerous property is persistence: a one-time injection
becomes a standing instruction.

**Coupled Goodhart.** Memory, harness, and weights are all being updated against the same benchmark, so a
regression introduced by one surface is patched by another. Aggregate scores stay flat or improve while the
system becomes more brittle. Only per-surface ablations reveal it.

**Safety forgetting.** Capability updates preserve task accuracy while eroding refusal boundaries,
calibration, or abstention. Because most regression suites weight capability heavily, this can pass every
gate that is actually being run.

**Plasticity decay.** After many update cycles, the system retains old behavior but learns new behavior
progressively worse—the loss-of-plasticity failure
([Dohare et al., 2024](https://arxiv.org/abs/2306.13812)) in system form. It is invisible to any evaluation
that only measures retention.

> **Diagnostic.** For each failure above, ask what measurement would have caught it *before* the aggregate
> score moved. If no current dashboard answers that question, the learning loop is running without that
> particular safety net.

---

## From continual learning to self-improvement and RSI

Continual learning and RSI are often joined by one sentence: *if the model keeps learning, eventually it can
improve itself*. That skips the hard part. Compounding has levels.

![A ladder from bounded refinement to successor-building RSI.](/assets/img/blog/continual-learning-for-next-generation-agents/fig4_continual_learning_to_rsi.png)
*Figure 4. Continual learning can supply persistent state and compounding. Self-reference begins when the
update mechanism becomes an update target; successor-building RSI additionally requires open-ended
verification, direction setting, resources, authority, and governance.*

| Rung | What changes | What persists | Representative work | What is still missing |
|---|---|---|---|---|
| 1. Bounded refinement | one answer or artifact | nothing after reset | self-critique and repair loops | any durable state |
| 2. Persistent self-improvement | memory, skill, harness, weights | across sessions and releases | CER, ARIA, Replit, Cursor, SDFT | evidence that the update generalizes |
| 3. Learning to learn | the update policy itself | across learning cycles | SEAL, Meta-TTL | an objective the system did not receive |
| 4. Successor-building RSI | the research and build loop | inherited by successors | DGM's topology, in a bounded domain | open-ended verification, taste, authority |

### Rung 1 — bounded refinement

The system retries, reflects, critiques, or edits one artifact. [Self-Refine](https://arxiv.org/abs/2303.17651) and many coding-agent repair loops
live here. The result can be much better than the first attempt, but resetting the context resets the learner.

### Rung 2 — persistent self-improvement

An update survives and improves later tasks: a memory entry, skill, harness patch, adapter, or checkpoint.
[CER](https://arxiv.org/abs/2506.06698), [ARIA](https://arxiv.org/abs/2507.17131), [Replit](https://replit.com/blog/evaluating-and-improving-agent-at-scale)'s loop, Cursor's RL pipeline, and [SDFT](https://arxiv.org/abs/2601.19897) occupy different parts of this rung. This is where
continual learning begins to compound.

Persistence is necessary but not sufficient. The update must transfer and avoid regression. A growing memory
that makes unrelated tasks worse is persistent change, not useful continual learning.

### Rung 3 — learning to learn

The system improves how it turns experience into updates: retrieval policy, update router, evaluator,
curriculum, self-edit generator, or adaptation prompt. [Meta-TTL](https://arxiv.org/abs/2604.00830) and [SEAL](https://arxiv.org/abs/2506.10943) expose this level directly.

The improvement rate can now change. But the outer objective, task distribution, and authority are still
usually fixed by humans. A better optimizer of a fixed benchmark is meta-learning, not automatically RSI.

### Rung 4 — successor-building RSI

The system can improve the machinery that conducts research and builds its successor: formulate useful
questions, design experiments, modify training code and architectures, evaluate results, allocate resources,
and deploy a better learner that repeats the process.

[DGM](https://arxiv.org/abs/2505.22954) has a recursive *topology*: the coding agent edits code that helps it
edit future code. It reports large benchmark gains—20.0% to 50.0% on a 200-task SWE-bench Verified
evaluation and 14.2% to 30.7% on the full Polyglot benchmark—and an archive lets future variants reuse
stepping stones. Yet its scope is a benchmark-selected coding harness around frozen models. Calling that
“full RSI” would erase the unsolved conditions:

- **verification:** open-ended research rarely has a test suite that captures the true objective;
- **causal credit:** a research result may depend on months of interacting changes;
- **research taste:** choosing a consequential direction is different from optimizing a supplied one;
- **transfer:** benchmark improvement can overfit the evaluator;
- **resources and authority:** experiments require compute, data, deployment, and organizational decisions;
- **continual alignment:** the goals and safety properties of the learner must survive capability updates.

This sharpens the reported Liang Wenfeng claim. Continual learning could make AI research agents accumulate
organizational experience and accelerate development. That makes self-improvement more practical. It does
not logically guarantee a runaway loop, because the evaluator, direction setter, and deployment authority
can remain bottlenecks.

> **Distinction.** Continual learning is about the **time axis** of improvement. RSI is about the
> **self-reference and autonomy** of the improvement graph. Continual learning can exist without RSI; RSI
> needs persistent compounding, but that persistence may live in versioned external artifacts rather than a
> mechanism traditionally labeled continual learning.

---

## How to evaluate compounding learning

A one-shot score after placing the answer in memory does not measure continual learning. The minimum unit of
evaluation is a **stream plus learner state plus future transfer**.

At time $$t$$, evaluate the agent *before* showing feedback for the current instance. After it acts, provide the
allowed feedback and update its state. This prequential protocol prevents the learner from using the label it
is being scored on.

A simple gain isolates learning from raw model capability:

$$G_t = \mathrm{Score}(A_t, \mathcal{T}_t) - \mathrm{Score}(A_0, \mathcal{T}_t),$$

where $$A_0$$ is the reset agent and $$\mathcal{T}_t$$ is a future or held-out task set. Two classical
quantities complete the picture. Write $$s_{i,j}$$ for the score on task family $$j$$ after learning through
step $$i$$. **Backward transfer** compares a family's score at the end of the stream with its score right
after it was learned, $$\mathrm{BWT}_j = s_{T,j} - s_{j,j}$$, so negative values are forgetting. **Forward
transfer** compares performance on a family before it is ever trained on against the reset agent,
$$\mathrm{FWT}_j = s_{j-1,j} - s_{0,j}$$, capturing whether earlier experience helped in advance.

This uses the standard convention in which family $$j$$ is the one learned at step $$j$$. Reporting the pair
matters because they trade off: a system that never writes anything has zero forgetting and zero forward
transfer, while a system that writes everything usually buys forward transfer with regressions somewhere
else. One curve is not enough. A credible scorecard needs to answer:

### Does performance compound?

- cumulative reward or utility over the stream;
- online regret against an appropriate oracle;
- slope of improvement and whether it saturates;
- performance under a fixed context/memory/compute budget.

### How quickly does the agent adapt?

- interactions, tokens, human corrections, and wall-clock time to reach a target;
- zero-shot versus post-feedback gain;
- recovery time after concept drift.

### What happens to old and new capabilities?

- backward transfer and worst-case forgetting;
- forward transfer to related future tasks;
- transfer to held-out task families, not only new examples from the same family;
- loss of plasticity over long update sequences.

### Is the route itself trustworthy?

- accuracy and calibration of surface/scope decisions;
- false-promotion rate: uncertain or private experience sent too broadly;
- false-rejection rate: useful experience discarded;
- quarantine resolution and rollback rate.

### What did learning cost?

- inference, storage, training, and evaluation compute;
- memory growth and retrieval latency;
- human-review burden;
- release latency and operational rollback cost.

### Did alignment and reliability survive?

- capability, policy, and safety regressions reported separately;
- calibration and abstention after updates;
- privacy leakage, deletion compliance, and poisoning robustness;
- worst-group and tail-risk changes, not only average reward.

Recent benchmarks cover complementary pieces:

- [StreamBench](https://arxiv.org/abs/2406.08747) makes the input–feedback sequence explicit and allows
  prompts, retrievers, memories, or weights to update, though its public datasets do not tightly control
  cross-task reuse.
- [LifelongAgentBench](https://arxiv.org/abs/2505.11942) supplies 1,396 dependent, executable tasks across
  database, operating-system, and knowledge-graph environments. Its replay results show that more history can
  hurt.
- [MemoryBench](https://arxiv.org/abs/2510.17281) simulates explicit and implicit service-time feedback and
  evaluates whether systems construct declarative and procedural memory rather than merely retrieve a
  prefetched history.
- [Evo-Memory](https://arxiv.org/abs/2511.20857) compares memory modules on sequential streams and separates
  conversational recall from reusable experience.
- [CL-Bench](https://arxiv.org/abs/2606.05661) builds six expert-validated domains with hidden reusable
  structure and, in some cases, concept drift. In the reported experiments, naive in-context learning
  outperforms dedicated memory systems—a warning that adding memory plumbing is not the same as learning.
- [AgentCL](https://arxiv.org/abs/2606.02461) contrasts arbitrary streams with controlled compositional
  streams and estimates plasticity, stability, and generalization gain separately. Its central finding is
  methodological: naive streams can compress differences among memory designs, while held-out cases reveal
  memory-induced degradation.

The next benchmark should require **routing**, not assume the answer. Give the learner a private preference,
an expiring regulation, a reusable procedure, and a broadly transferable skill; let it choose context,
memory, harness, or weights; then score task utility, scope, deletion, transfer, and regression. Until
benchmarks evaluate that decision, every method competes on the surface its designer already chose.

> **Minimum credible protocol.** Predict before feedback; snapshot the full agent state; compare with a
> reset baseline and an unlimited-context oracle; hold out future task families; inject stale, conflicting,
> delayed, and poisoned feedback; and ablate the update surfaces that changed.

A results table for a continual-learning agent should therefore report, at minimum: cumulative utility over
the stream; gain against the reset baseline on held-out families; backward transfer on protected
capabilities; adaptation cost in interactions and human corrections; memory or parameter growth; and the
number of promotions later reverted. A paper that reports only the first of those has measured that
something changed, not that the agent learned.

---

## A research agenda

The multi-timescale view suggests projects that are concrete enough to implement now.

### 1. Calibrated update router

**Question:** Can a learner decide where an experience belongs and know when that decision is uncertain?

Build streams in which the correct route varies:

- a one-session constraint;
- a user preference;
- a project fact with an expiry date;
- a procedure that transfers across users;
- a fleet-wide skill;
- malicious or contradictory feedback.

The router outputs surface, scope, TTL, action, and confidence. Compare hand-written policies, LLM routers,
retrieval-based classifiers, and learned decision policies. Evaluate expected utility, routing calibration,
privacy violations, false promotion, transfer, and rollback.

This connects continual learning to uncertainty quantification in a way that accuracy alone cannot: an
uncertain router should choose a reversible write or ask for evidence, not pretend every experience has a
global label.

**Success criterion:** the learned router beats every fixed policy on cumulative utility *and* has a lower
false-promotion rate than the most aggressive fixed policy. Beating "always memory" on utility alone is not
enough, because the interesting claim is about knowing when not to write.

### 2. Session-to-weight consolidation

**Question:** How can a “senior” long-session agent teach a fresh-session version of itself?

Let an agent spend many episodes learning a new codebase, research area, or tool environment. At the end,
reset its context and compare:

1. full-history prompting;
2. retrieved episodic memory;
3. distilled semantic/procedural memory;
4. an executable skill;
5. a LoRA adapter;
6. [SDFT](https://arxiv.org/abs/2601.19897) or another learner-relative distillation method.

Measure future-task transfer, forgetting, calibration drift, inference cost, and the ability to delete one
source experience. The key ablation is not only which method wins, but **which experience should never have
been consolidated**.

**Success criterion:** a consolidated fresh session matches full-history prompting on future tasks at a
fraction of the context cost, without regressing protected capabilities. **Likely obstacle:** the
consolidation target is written by the same model whose errors it inherits, so teacher-error propagation
needs its own measurement.

### 3. Trustworthy online agent learning

**Question:** Can an agent learn when feedback is delayed, implicit, selected by its own policy, or
adversarial?

Create a versioned simulator with known causal structure:

- immediate user reactions that are noisy proxies;
- delayed objective outcomes;
- changing users and environment;
- poisoned feedback;
- corrections that apply only to one scope;
- revocation and right-to-forget requests.

Jointly infer credibility and causal attribution. Compare naive aggregation, inverse-propensity corrections,
causal models, and uncertainty-aware quarantine. Report time-to-recovery and false promotion, not only final
reward.

**Success criterion:** performance degrades gracefully as the poisoned fraction rises, and recovery time
after a drift event stays bounded. A method that wins on clean feedback and collapses at 5% poisoning has
not solved the deployment problem.

### 4. Co-evolving memory, harness, and weights

**Question:** When do multiple surfaces cooperate, and when do they hide one another's regressions?

Most work changes one object. Build a factorial study:

- memory only;
- harness only;
- adapter only;
- memory + harness;
- memory + adapter;
- harness + adapter;
- all three.

Use held-out tasks and Pareto gates for utility, cost, safety, calibration, and reversibility. Run causal
ablations or Shapley-style attribution on each accepted update. A likely result is that different clocks are
complementary—but joint optimization can produce coupled Goodhart effects, where one surface patches the
benchmark artifacts created by another.

**Success criterion:** the combined system beats the best single surface on held-out families, and each
surface's contribution remains positive when the others are ablated. If removing one surface *improves*
results, the combination was masking a regression rather than compounding gains.

### 5. Personal-to-fleet continual learning

**Question:** How can local agents share transferable lessons without leaking private or contradictory
experience?

Model a fleet with user, project, organization, and global namespaces. Some latent rules are shared; others
conflict. Study:

- selective/federated distillation;
- provenance-preserving aggregation;
- consensus and conflict resolution;
- per-scope adapters and expert routing;
- machine unlearning and propagation of deletions;
- incentives for users who supply valuable feedback.

The right output is not one universal memory. It is a hierarchy in which a lesson proves that it generalizes
before crossing an administrative boundary.

**Success criterion:** fleet-level learning improves new tenants who contributed nothing, while a deletion
request provably removes the contributor's influence from every derived artifact. The second half is what
makes this hard, and what makes naive "just fine-tune on everything" unusable.

### 6. Continual alignment

**Question:** Can safety, honesty, and calibrated abstention remain stable while capability grows?

Interleave capability tasks with changing policy constraints and adversarial feedback. Track backward
transfer separately for helpfulness, harmlessness, honesty, calibration, and policy compliance. Test whether
replay, regularization, isolation, and distillation that preserve task accuracy also preserve refusal
boundaries and uncertainty behavior.

Safety forgetting is not just another benchmark regression. A learner that becomes better at acting while
worse at knowing when not to act has moved in the wrong direction.

**Success criterion:** after a long update stream, refusal boundaries and calibration are statistically
indistinguishable from the pre-update model on a held-out safety suite, while capability has improved. Any
method that reports only the capability half of that sentence has left the important question unanswered.

### A starter experiment

A useful first project does not require a frontier model. Build a research or coding agent with:

- ordinary context;
- namespaced, provenance-rich memory;
- editable Markdown skills under version control;
- one LoRA adapter;
- an update router that can choose one surface, quarantine the experience, or make no update after each task;
- a hidden regression and future-transfer suite.

Stream related and unrelated tasks with explicit, implicit, stale, and conflicting feedback. Measure
cumulative utility, adaptation speed, transfer, forgetting, routing calibration, memory growth, human-review
cost, and rollback. The baselines are fixed routing policies—always context, always memory, always skill,
always fine-tune, and always no update.

That experiment directly tests the thesis of this post: **the learning algorithm is partly the policy that
chooses what kind of learner to become next**.

---

## Summary

Today's agents generate rich experience but rarely compound it. Classical continual learning correctly
identified the stability–plasticity dilemma, replay, regularization, isolation, expansion, distillation, and
meta-learning. The agent era changes the unit of analysis. The model is now one component of a system with
several mutable surfaces and several clocks.

The practical architecture is:

1. capture versioned trajectories and outcomes;
2. validate feedback and attribute the cause;
3. route the experience to context, memory, harness, weights, or updater at the right scope;
4. replay, contrast, and consolidate;
5. require local repair, held-out transfer, and non-regression;
6. stage the release and preserve rollback;
7. monitor the result and, eventually, improve the updater itself.

This also clarifies the path from continual learning to self-improvement. Persistent memory or weights can
compound competence. Learning the updater changes the improvement rate. Successor-building RSI requires
still more: honest open-ended verification, causal credit, research taste, direction setting, resources,
authority, and alignment.

Return to the new research assistant. We do not expect a human employee to rewrite their entire brain after
every comment. They keep working notes, form memories, adopt procedures, internalize repeated lessons, and
become better at deciding how to learn. Next-generation agents need the computational equivalent—each layer
operating on the right clock.

> **Takeaway.** The next generation of agents will be defined less by whether they *can* learn after
> deployment than by whether they can decide **what should be learned, where it should be written, and when
> it is safe to promote**.

---

*Source note: Liang Wenfeng's reported May 2026 remarks are cited through a July 2026 *Daily Economic News*
report on a circulating 42-page AI-organized transcript. The outlet said an investment institution
participating in DeepSeek's financing confirmed that a May meeting took place and regarded the content as
credible, but no original audio, speaker-labeled official transcript, or DeepSeek confirmation is public.
The article therefore treats the remarks as a
media-reported, institution-corroborated unofficial record and as motivation—not technical evidence.
Company engineering metrics elsewhere are first-party reports; recent 2026 preprints are described as
paper-specific evidence rather than settled results. All figures are original.*

---

## How to cite

> Zhang, Jiaxin. (Jul 2026). Continual Learning for Next-Generation Agents. *Jiaxin Zhang's Blog.*
> https://jxzhangjhu.github.io/blog/2026/continual-learning-for-next-generation-agents/

```bibtex
@article{zhang2026continualagents,
  title   = "Continual Learning for Next-Generation Agents",
  author  = "Zhang, Jiaxin",
  journal = "Jiaxin Zhang's Blog",
  year    = "2026",
  month   = "Jul",
  url     = "https://jxzhangjhu.github.io/blog/2026/continual-learning-for-next-generation-agents/"
}
```

---

## References

[1] Qingyao Ai, et al. ["MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems."](https://arxiv.org/abs/2510.17281) *ICML*, 2026.

[2] Parth Asawa, et al. ["Continual Learning Bench: Evaluating Frontier AI Systems in Real-World Stateful Environments."](https://arxiv.org/abs/2606.05661) arXiv:2606.05661, 2026.

[3] Cursor. ["Improving Cursor Tab with online RL."](https://cursor.com/blog/tab-rl) Engineering blog, 2025.

[4] Cursor. ["Improving Composer through real-time RL."](https://cursor.com/blog/real-time-rl-for-composer) Engineering blog, 2026.

[5] DeepSeek. ["DeepSeek 招聘."](https://talent.deepseek.com/) Official recruiting site, accessed July 2026.

[6] Shibhansh Dohare, et al. ["Loss of Plasticity in Deep Continual Learning."](https://www.nature.com/articles/s41586-024-07711-7) *Nature*, 2024. Preprint: arXiv:2306.13812.

[7] Vaggelis Dorovatas, et al. ["Position: Modular Memory is the Key to Continual Learning Agents."](https://arxiv.org/abs/2603.01761) *ICML*, 2026.

[8] Siyuan Guo, Yali Du, Hechang Chen, Yi Chang, and Jun Wang. ["CASCADE: Case-Based Continual Adaptation for Large Language Models During Deployment."](https://arxiv.org/abs/2605.06702) arXiv:2605.06702, 2026.

[9] Yufei He, et al. ["Enabling Self-Improving Agents to Learn at Test Time With Human-In-The-Loop Guidance."](https://arxiv.org/abs/2507.17131) arXiv:2507.17131, 2025.

[10] Hao Jiang, et al. ["LLM Evolution as an Industry-Scale Ecosystem: A Lifecycle Perspective on Continual Learning."](https://arxiv.org/abs/2606.24901) arXiv:2606.24901, 2026.

[11] Seth Karten, et al. ["Continual Harness: Online Adaptation for Self-Improving Foundation Agents."](https://arxiv.org/abs/2605.09998) arXiv:2605.09998, 2026.

[12] James Kirkpatrick, et al. ["Overcoming Catastrophic Forgetting in Neural Networks."](https://arxiv.org/abs/1612.00796) *PNAS*, 2017.

[13] Zhizhong Li and Derek Hoiem. ["Learning without Forgetting."](https://arxiv.org/abs/1606.09282) *ECCV*, 2016.

[14] Yitao Liu, Chenglei Si, Karthik Narasimhan, and Shunyu Yao. ["Contextual Experience Replay for Self-Improvement of Language Agents."](https://arxiv.org/abs/2506.06698) arXiv:2506.06698, 2025.

[15] David Lopez-Paz and Marc'Aurelio Ranzato. ["Gradient Episodic Memory for Continual Learning."](https://arxiv.org/abs/1706.08840) *NeurIPS*, 2017.

[16] Zhanzhi Lou, Hui Chen, Yibo Li, Qian Wang, and Bryan Hooi. ["Learning to Learn-at-Test-Time: Language Agents with Learnable Adaptation Policies."](https://arxiv.org/abs/2604.00830) arXiv:2604.00830, 2026.

[17] Aman Madaan, et al. ["Self-Refine: Iterative Refinement with Self-Feedback."](https://arxiv.org/abs/2303.17651) *NeurIPS*, 2023.

[18] Arun Mallya and Svetlana Lazebnik. ["PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning."](https://arxiv.org/abs/1711.05769) *CVPR*, 2018.

[19] James L. McClelland, Bruce L. McNaughton, and Randall C. O'Reilly. ["Why There Are Complementary Learning Systems in the Hippocampus and Neocortex: Insights From the Successes and Failures of Connectionist Models of Learning and Memory."](https://doi.org/10.1037/0033-295X.102.3.419) *Psychological Review*, 1995.

[20] OpenAI. ["Building self-improving tax agents with Codex."](https://openai.com/index/building-self-improving-tax-agents-with-codex/) 2026.

[21] Dwarkesh Patel. ["Andrej Karpathy — AGI is still a decade away."](https://www.dwarkesh.com/p/andrej-karpathy) Dwarkesh Podcast transcript, 2025.

[22] Dwarkesh Patel. ["Dario Amodei — 'We are near the end of the exponential'."](https://www.dwarkesh.com/p/dario-amodei-2) Dwarkesh Podcast transcript, 2026.

[23] Replit. ["Closing the loop: Evaluating and improving Replit Agent at scale."](https://replit.com/blog/evaluating-and-improving-agent-at-scale) 2026.

[24] David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, and Greg Wayne. ["Experience Replay for Continual Learning."](https://arxiv.org/abs/1811.11682) *NeurIPS*, 2019.

[25] Andrei A. Rusu, et al. ["Progressive Neural Networks."](https://arxiv.org/abs/1606.04671) arXiv:1606.04671, 2016.

[26] Idan Shenfeld, Mehul Damani, Jonas Hübotter, and Pulkit Agrawal. ["Self-Distillation Enables Continual Learning."](https://arxiv.org/abs/2601.19897) arXiv:2601.19897, 2026.

[27] Yiheng Shu, et al. ["AgentCL: Toward Rigorous Evaluation of Continual Learning in Language Agents."](https://arxiv.org/abs/2606.02461) arXiv:2606.02461, 2026.

[28] David Silver and Richard S. Sutton. ["Welcome to the Era of Experience."](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) Preprint of a chapter in *Designing an Intelligence*, MIT Press, 2025.

[29] Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. ["A Comprehensive Survey of Continual Learning: Theory, Method and Application."](https://arxiv.org/abs/2302.00487) *IEEE TPAMI*, 2024.

[30] Tianxin Wei, et al. ["Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory."](https://arxiv.org/abs/2511.20857) arXiv:2511.20857, updated 2026.

[31] Cheng-Kuang Wu, Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Chen, and Hung-yi Lee. ["StreamBench: Towards Benchmarking Continuous Improvement of Language Agents."](https://arxiv.org/abs/2406.08747) *NeurIPS Datasets and Benchmarks*, 2024.

[32] Zheng Wu, et al. ["Agent-Dice: Disentangling Knowledge Updates via Geometric Consensus for Agent Continual Learning."](https://arxiv.org/abs/2601.03641) arXiv:2601.03641, 2026.

[33] Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. ["Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents."](https://arxiv.org/abs/2505.22954) *ICLR*, 2026.

[34] Junhao Zheng, et al. ["LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners."](https://arxiv.org/abs/2505.11942) arXiv:2505.11942, 2025.

[35] Junhao Zheng, et al. ["Lifelong Learning of Large Language Model based Agents: A Roadmap."](https://doi.org/10.1109/TPAMI.2025.3650546) *IEEE TPAMI*, 2026.

[36] Adam Zweiger, et al. ["Self-Adapting Language Models."](https://arxiv.org/abs/2506.10943) *NeurIPS*, 2025.

[37] 每日经济新闻. ["梁文锋3小时44分钟闭门会聊了什么？全干货来了：十大核心主题+六大焦点问答."](https://www.nbd.com.cn/articles/2026-07-23/4504670.html) 2026.
