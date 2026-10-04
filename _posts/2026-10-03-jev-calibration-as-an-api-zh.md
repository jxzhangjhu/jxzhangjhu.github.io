---
layout: post
published: true
title: "Jev 与决策模型的回归：当 Calibration 成为 API"
date: 2026-10-03 16:00:00
author: Jiaxin Zhang
description: "一篇持续更新的 Jev 与开放 System One 决策模型综述：calibration、RLCD、证据质量、开放模型与方法、benchmark、应用、失效模式，以及 bounded decision 与 generative reasoning 的边界。"
tags: jev calibration uncertainty decision-models agents rlcd rlcr caopd 中文
categories: research-notes
giscus_comments: true
related_posts: false
ai_assisted: true
read_time: 82
og_image: https://jxzhangjhu.github.io/assets/img/blog/jev-calibration/og_card.png
---

<div class="lang-switch"><a href="/blog/2026/jev-calibration-as-an-api/">English</a> · <strong>中文</strong></div>

### 目录 {#table-of-contents}

- [一种不同的 AI 接口](#a-different-kind-of-ai-interface)
- [Jev 是什么，又不是什么](#what-jev-isand-what-it-is-not)
- [Jev 为什么会迅速走红？](#why-did-jev-become-popular-so-quickly)
- [一个 typed decision 的解剖](#the-anatomy-of-a-typed-decision)
- [Decision model 为什么可以这么快？](#why-can-a-decision-model-be-so-fast)
- [Calibration：从评测指标到运行契约](#calibration-from-a-metric-to-an-operating-contract)
  - [Probability、confidence 与 correctness 是不同对象](#probability-confidence-and-correctness-are-different-objects)
  - [如何测量 calibration](#how-calibration-is-measured)
  - [Calibration 如何变成 control flow](#how-calibration-becomes-control-flow)
- [RLCD：TypeSafe 公开了什么，又没有公开什么](#rlcd-what-typesafe-disclosed-and-what-it-did-not)
- [周边研究版图](#the-surrounding-research-landscape)
- [现有证据究竟说明了什么？](#what-does-the-evidence-actually-show)
- [开放的 Jev 生态](#the-open-jev-ecosystem)
- [护城河可能在哪里？](#where-might-the-moat-be)
- [应用：Jev 在 agent stack 中的位置](#applications-where-jev-fits-in-an-agent-stack)
- [硬边界：哪些问题仍然应该交给 LLM？](#the-hard-limits-where-an-llm-remains-the-right-abstraction)
- [我会如何评估一个 decision model](#how-i-would-evaluate-a-decision-model)
- [从 calibrated decision 到 calibrated agent](#from-calibrated-decisions-to-calibrated-agents)
- [开放研究问题](#open-research-questions)
- [结论](#conclusion)
- [如何引用](#how-to-cite)
- [参考文献](#references)

---

> **V2 · 文献快照：2026 年 10 月 4 日。** 第一版主要解释 Jev 的接口与 calibration 命题。这次修订加入了
> 快速增长的独立研究、按证据质量加权的研究表、开放模型与方法地图、benchmark 与 leaderboard 的阅读指南，以及
> 更完整的应用目录。Jev 问世仅数周，几乎所有直接研究都还是未经同行评审的 preprint。本文中的每项排名与数值结果，
> 都应被视为带日期的快照，而不是永久排序。

## 一种不同的 AI 接口 {#a-different-kind-of-ai-interface}

一个 agent 来到了决策岔路口：应该调用数据库，还是搜索网页？刚检索到的 memory 是否足够相关，值得重新放回
context？工具返回的结果是否真的支持那个 claim？计划中的 action 是否可逆？系统应该继续执行、询问用户、调用一个
更昂贵的 verifier，还是把案例交给人类？

大语言模型可以用自然语言回答以上每一个问题。但自然语言往往并不是正确的接口。应用程序不需要一段文字说，某个
action “看起来比较安全，不过仍然存在一些不确定性”；它需要的是一个来自预定义集合的值，以及一个可以直接用于代码
逻辑的 probability（概率）。例如：

```json
{
  "action": "ask_for_approval",
  "probabilities": {
    "proceed": 0.08,
    "ask_for_approval": 0.84,
    "stop": 0.08
  }
}
```

这个区别解释了为什么 TypeSafe AI 的第一个托管式 “System One” model——**Jev**——在 2026 年 9 月 15 日发布后
迅速获得大量关注。Jev 并不把自己定位成又一个 assistant。它接收文本状态，在有限输出空间内回答 typed question
（有类型约束的问题），并返回 probabilities。真正把这些 probabilities 转化成 action 的，是 model 外围的软件，而
不是 model 生成的文字
（[TypeSafe AI, 2026](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）。

大小写也有含义：产品名称是 **Jev**，而不是 “JEV”，官方也没有把它解释成一个缩写。TypeSafe 表示，这个名字是向
经济学家 William Stanley Jevons 致敬；“System One” 则借用了常见的 System 1/System 2 区分中快速、直觉式的一侧。
公司用它来对照那些通过 sequential decoding 生成字符串的 generative reasoning model。Jev 面向的是软件和 agent loop
内部数量巨大、不断重复的小型 semantic judgment
（[TypeSafe AI, 2026](https://docs.typesafe.ai/concepts/system-one)）。

![Decision model 把 semantic state 转化为软件可以执行的 typed probability](/assets/img/blog/jev-calibration/fig1_jev_decision_loop.svg)
*图 1. 这里的核心变化，是从 prompt → generated text → parsing，转为 state + typed question → probability
distribution → explicit policy。这是一张接口层面的示意图，并不是对 Jev 未公开内部架构的断言。*

这与我近期多项工作的核心观点非常接近：**uncertainty（不确定性）应该改变 AI system 下一步做什么**。在普通评测中，
confidence（置信度）只是与 accuracy 并列打印的一个指标；在实际部署的 agent 中，它可以成为 control signal。它可以触发
verification、分配更多 test-time compute、再检索一个来源、请求人工审核、更新 memory，或停止一项不安全的 action。
我关于 holistic trajectory calibration、Agentic Uncertainty Quantification，以及 calibration-aware on-policy
distillation 的研究，都是从 long-horizon generative agent 的角度探索这一思想。Jev 则把同一个原则封装到了快速、原子级
decision 上
（[Zhang et al., 2026a](https://proceedings.mlr.press/v306/zhang26gy.html)；
[Zhang et al., 2026b](https://arxiv.org/abs/2601.15703)；
[Zhang et al., 2026c](https://arxiv.org/abs/2604.16830)）。

本文也扩展了我最近一篇
[LinkedIn 短文](https://www.linkedin.com/posts/jiaxin-zhang-1425289b_ai-aiagents-calibration-activity-7507890333754580992-kSFq)
中最初提出的观点：calibration 最值得关注的时刻，是它不再只是一个 passive metric，而开始改变 system 的 next action。

这使 Jev 值得重视——但并不意味着围绕它的每个说法都成立。有限 output type 可以防止 malformed output，却不能防止
model 在错误选项上表现得非常自信。在公开 benchmark 上较低的 expected calibration error，并不能保证其 probability
面对某位客户已经 shift 的真实流量时依然可靠。一个速度很快的托管 endpoint，本身也不能说明优势究竟来自新架构、更小的
model、specialized data、training、serving，还是这些因素的组合。TypeSafe 虽然把 **reinforcement learning for
calibrated decisions（RLCD，面向校准决策的强化学习）**列为系统组成部分，但并没有公开 Jev 的 RLCD 算法细节。

因此，本文有三个目标。第一个目标是解释：把 Jev 的接口、calibration、潜在 use case，以及它与 LLM 的关系讲清楚。
第二个目标是划清认识边界：全文会持续区分四类证据：

1. TypeSafe 官方 API 与 model documentation 中的**已记录行为（documented behavior）**。
2. 来自 TypeSafe 自有 launch evaluation 的**厂商报告结果（vendor-reported measurements）**。
3. **早期独立证据（independent early evidence）**——其中大部分是近期 preprint，而不是已经沉淀下来的同行评议结论。
4. **我的判断（my interpretation）**——包括对产品护城河、研究意义和未来方向的分析。

第三个目标是做文献筛选。这个生态已经大到仅列出链接并没有太大帮助，因此 V2 使用一条简单的纳入规则：优先选择
matched comparison、held-out 或 sealed evaluation、uncertainty interval，以及可运行的 artifact；明确记录 label 是否来自
另一个 model；没有读清 scoring rule 之前，绝不把 community leaderboard 的分数升级成科学结论。本文收录 repository
与 leaderboard，是为了让领域更容易被检查，而不是因为 popularity 可以证明 quality。

核心结论很简单：

> **Calibration 可以成为一份软件契约，但它永远不会自动成为一张安全证书。**

---

## Jev 是什么，又不是什么 {#what-jev-isand-what-it-is-not}

TypeSafe 把 Jev 描述为 decision model，而不是 language model。从运行方式看，调用者提供一段文本形式的 **state**，
以及一个或多个 typed **question**。Jev 返回有限答案和相应的 probability distribution。它不返回 essay、chain of
thought、source code，也不产生 open-ended plan
（[TypeSafe AI, 2026](https://docs.typesafe.ai/introduction)）。

目前共有三种 primitive：

| Primitive | 输出空间 | 典型问题 | 返回信号 |
|---|---|---|---|
| **Choice（类别选择）** | 无序有限选项 | “下一步应该运行哪个 tool？” | 选中项、完整 distribution、派生 confidence |
| **Score（有序评分）** | 2–10 级有序刻度 | “这项 action 的风险有多高？” | score、rubric legend、distribution、派生 confidence |
| **Noul（二元判断）** | yes/no | “这条 evidence 是否支持该 claim？” | “yes” 的 probability |

**Noul** 是 TypeSafe 为二元 primitive 起的特殊名字。Choice 可用于 categorical routing；Score 可用于由明确 rubric
约束的 ordinal judgment；Noul 则可以表达一个 semantic predicate。声明类型非常重要，因为它消除了一大类 integration
problem。当程序只接受 `search`、`database` 或 `ask_user` 时，Choice question 不会突然返回一段话；在十分制下，
Score question 也不会凭空发明第十一级。输出在结构上天然有效
（[TypeSafe AI, 2026](https://docs.typesafe.ai/primitives)）。

但 structural validity 与 semantic validity 是两回事。假设 Choice 只有以下两个选项：

```text
[approve, reject]
```

如果正确 action 应该是“abstain and request more evidence”，type 本身就迫使 model 给出一个坏答案。即使 schema 中包含了
正确 action，Jev 也仍然可能给错误选项分配 0.99 probability。因此，准确的说法不是“Jev 不会 hallucinate”，而是：

> Jev 无法生成调用者定义输出空间之外的答案；但它仍然可能在合法选项中选错。

一篇早期独立 preprint 让这个区别变得非常具体。作者保持 state、question、rubric 和 option set 不变，却把 `yes`、`no`
这类具有明显语义的 option name 重新分配给不同 rubric。在 hosted model 上，交换后 AUC 从 0.8146 降至 0.5806，answer
flip 数量达到 test–retest floor 的 24 倍，而 type-error rate 始终严格为零。Typed head 可以完全遵守 schema，却误解 schema
designer 真正想表达的含义
（[Sun et al., 2026](https://arxiv.org/abs/2609.26758)）。

这一差异应该直接影响 schema design。真实 deployment 往往需要 `other`、`none_of_the_above`、`unknown` 或
`escalate`。调用方还需要一套独立 policy，规定当所有 probabilities 都很分散、输入彼此矛盾，或者 state 超出 model
经验范围时应该怎么办。

TypeSafe 表示，针对同一个 state 的不同问题会相互独立、并行评估。增加问题据称只会增加很少 latency，而且回答不会进入
一个共享的 conversational transcript。对 agent system 来说，这一点很有吸引力，因为同一个 state 可能同时需要判断
relevance、risk、completion、authorization 和 next action。独立 question 可以避免一个 generated answer 成为下一个
问题 context 所带来的 “context rot”；不过，因为这些问题共享 state 和 model，statistical error 依然可能相关
（[TypeSafe AI, 2026](https://docs.typesafe.ai/introduction)）。

![Typed decision model 与 generative LLM 的 execution path 对比](/assets/img/blog/jev-calibration/fig2_jev_vs_llm.svg)
*图 2. Generative LLM 需要 decode 并 parse 一个字符串；Jev 则把共享 state 与 typed question 映射成程序可以直接消费的
bounded distribution。同一个 state 可以并行支持 Choice、Score 与 Noul question。这是产品层面的比较，并不是对 Jev
未公开内部架构的断言。*

根据 2026 年 10 月的公开文档，Jev 仅支持 text，英文效果最好，Choice 最多支持 255 个选项；它提供 64k 的总 context
window，同时还对 state 和最长 question 有额外限制。当前托管版本为 `jev-1.13.0`，`jev-latest` 是其 alias。
TypeSafe 标出的价格是每百万 input token \$0.042，output 免费；动态 service limit 则可能随时间变化
（[TypeSafe AI, 2026](https://docs.typesafe.ai/models)）。这些是当前 service fact，而不是这类 model 永久不变的性质。

同样重要的是明确 Jev 不是什么。它不是 planner、memory system、browser、tool runtime，也不是完整 agent。它天然没有
执行答案的 authority；不会决定 agent 可以运行多久、可以获得哪些 credential，或某项高影响 action 是否必须经过人类
批准。这些都属于 harness。Jev 可以为系统内部提供一个有用判断，但真正定义 control loop 的是 harness。

“Not an LLM” 也不应该被理解成对隐藏实现已经过验证的技术陈述。TypeSafe 用这句话区分 Jev 的用途和 output semantics，
以及 generative language model 的用途。公司并没有公开 architecture、parameter count、tokenizer、training corpus 或
weights。我们能够站得住脚的说法是：**Jev 暴露的是一个 non-generative typed decision interface**。关于其内部 substrate
的说法仍然只是推测。

---

## Jev 为什么会迅速走红？ {#why-did-jev-become-popular-so-quickly}

与其把 Jev 的早期热度归因于某个孤立算法结果，不如把它理解成 timing、interface 与 economics 的共同作用。

第一，它的 framing 异常清晰。Generative AI 让开发者习惯于把所有问题都改写成 prompt、请求 model 生成文字、再解析
这些文字，并祈祷它遵循了 schema。TypeSafe 提出了一个带有挑衅性的问题：这些调用中，到底有多少真的是 generation
problem？Classification、routing、ranking、checking 与 gating 都是老问题，但 agent system 如今执行它们的频率前所
未有。“Decisions, not strings” 把一个熟悉的技术差异包装成了一个新的产品类别。

而且已经有直接证据表明，这次 launch 的热度迅速转化成了实际试用。Vercel 报告，Jev 接入 AI Gateway 后 24 小时内，
接近 13% 的付费团队已经使用过它——超过该 gateway 以往任何 model launch 首日覆盖率的两倍。这个数字衡量的是单一
distribution channel 中的快速试用，而不是长期 production retention；但它解释了 Jev 为什么会突然显得无处不在
（[Vercel, 2026](https://vercel.com/blog/ai-gateway-jev-model-launch)）。

第二，latency 会在 agent 内部累积。一次 1.5 秒的调用在聊天中可能还能接受；但如果一个 loop 要做 100 个小判断，而且
其中多个位于 critical path 上，这个延迟就会非常痛苦。一个能在数百毫秒内返回结果的 specialized call，会改变整体
architecture：开发者可以检查每一次 tool result、路由每一条 memory，或者同时运行多个独立 safeguard，而不必刻意节省
model call。

第三，probability distribution 是比脆弱的 parsed answer 更好的 building block。它允许软件做 cost-sensitive choice。
系统可以直接接受简单 case，把中等 confidence 的 case 发给更大 model，把少量高风险 case 转给人类。这比一句“我比较
有信心”更容易落到实际操作。

第四，这次发布把打磨得很好的 API、容易记住的 demo 和很大的 quantitative claim 放在了一起。TypeSafe 报告了大约
70–500 ms 的 end-to-end latency、在部分比较中 40–200 倍的 speedup，以及很低的 token cost。公司还展示了 game 和
workflow demo，使接口变得直观。这些 claim 仍然需要针对具体 workload 做验证，但它们容易理解，也很容易传播
（[TypeSafe AI, 2026](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）。

第五，产品出现的时机正好。这个领域正在从 single-turn assistant 迁移到 long-running agent。在这种 setting 中，
confidence 不再只是一个 evaluation concern；它会决定什么时候花更多 compute、什么时候 branch，以及什么时候停止。
一篇早期 literature review 统计，Jev 发布后的前九天就出现了 28 篇相关论文——这说明关注度极高，但并不等于形成了
scientific consensus
（[Tang and Zheng, 2026](https://arxiv.org/html/2609.32160)）。

最后，团队叙事也有作用。TypeSafe 的公开介绍称，CEO Diogo Almeida 之前参与过 RLHF 和 InstructGPT 相关工作；
founding team 还包括 Sasha Sheng 与 Erik Gafni。履历不是 evaluation result，但它会提高研究者和开发者认真调查一个
新系统，而不是直接把它当作一层浅 API wrapper 的概率
（[TypeSafe AI, 2026](https://typesafe.ai/team)）。

这里还有一个更广泛的文化原因。现代 AI 的大部分努力都在追求 generality：一个 model 解决所有任务。Jev 重新唤起了
machine learning 中互补的另一条传统：model 有一个狭窄而明确的 output contract，人们用 decision quality、probability
quality、latency 与 cost 来评估它。“Decision model 的回归”之所以让人感到新鲜，恰恰因为过去几年里，产业几乎把所有
semantic operation 都塞进了 free-form generation。

---

## 一个 typed decision 的解剖 {#the-anatomy-of-a-typed-decision}

设想一个 agent 正在判断是否执行某条 shell command。State 可以包括 user request、准备执行的 command、repository
policy、当前 working directory、diff summary，以及这项 action 是否可逆。系统可以询问：

```text
Choice: What should the runtime do next?
Options: [execute, request_approval, block]

Score: How consequential is this action?
Rubric: 1 = local and reversible; 5 = external or difficult to recover

Noul: Is the proposed action explicitly authorized by the user?
```

Jev 返回三个 distribution，再由 policy——而不是 Jev——把它们组合起来：

```python
if authorization_p < 0.90:
    request_approval()
elif risk_score >= 4:
    request_approval()
elif next_action == "execute" and next_action_p >= 0.95:
    execute_in_sandbox()
else:
    escalate_to_stronger_model()
```

这个例子揭示了五个比语法本身更重要的 design decision。

**第一，state 是 evidence boundary。** 如果 policy document、user approval 或 tool provenance 没有进入 state，model
就无法可靠地推断它。更快的 decision 不能弥补缺失的 state。

**第二，options 编码了 ontology。** 即使 probabilities 在给定选项集合内完美准确，一个糟糕的 option set 仍然会制造
糟糕的 decision。当 uncertainty 应该触发信息搜集时，“execute or block” 就不如 “execute, ask, block” 有用。

**第三，rubric 固定了 meaning。** 如果调用方没有定义每一级，一个 Score 4 就没有稳定含义。好的 rubric 应该使用可观察
的判断标准，而不是“有点危险”这种形容词。

**第四，probability 与 policy 应该保持分离。** Model 负责估计，application 负责决策。同一个 0.82 probability，可能足以
自动分类一份低成本 document，但在 financial transfer 之前仍然必须进入人工审核。

**第五，fallback 是 system 的一部分。** 只有当 fallback 具有互补优势时，abstention 才有价值。如果把所有 uncertain case
都发给一个恰好会在同类 case 上失败的 LLM，只会增加成本，并不会降低风险。

这就是为什么该接口不只是 “structured output”。LLM 的 JSON mode 解决的是 formatting；typed decision model 想解决的是
另一个问题：在声明好的 action space 上，生成一个有用、低 latency 的 distribution。Jev 能否在某个具体 workload 中实现
这个目标，仍然是一个 empirical question。

---

## Decision model 为什么可以这么快？ {#why-can-a-decision-model-be-so-fast}

Autoregressive generation 具有 sequential cost。为了生成 $T$ 个 output token，language model 必须反复运行 decoder、
sample 或选择下一个 token、扩展 key-value cache，再继续下一步。即使有 speculative decoding 和优化过的 serving，一段长
explanation 或 chain of thought 的耗时仍然与生成序列的长度相关。输出之后往往还需要 parsing 与 validation。

有限 decision endpoint 可以绕过 output-generation loop 的大部分成本：它直接返回 bounded answer space 上的 score，而不是
逐 token decode 一段 rationale。TypeSafe 文档明确说明，Jev 只 ingest 一次 state，然后并行评估不同 question；但它没有
披露 representation、head 或其他内部 computation 是否共享、以及如何共享。产品 output 不需要暴露 generated rationale，
但我们也不能仅凭这个 interface 反推 model 未公开的内部 computation。

因此，潜在速度优势至少来自三个不同来源：

1. **更少的 sequential output step。** 一个 distribution 取代了一段 generated response。
2. **跨 question 的 parallel fan-out。** TypeSafe 表示，一次 ingest 的 state 可以并行支持多个 question output，避免为每个
   answer 各自运行一段 sequential generation。
3. **可能的 specialization 或 internal amortization。** 为 bounded decision 优化的 model，可能比必须同时完成写作、coding、
   reasoning、translation 与 conversation 的 frontier generative model 使用更小容量，或以更高效率共享内部计算。

前两点是 TypeSafe 公开记录的 product-level behavior；第三点则是假设：model size、architecture、representation sharing 与
sampler 都没有披露。因此，我们无法把官方报告的速度归因于某一个内部 mechanism。Hardware、batching、quantization、
geographical distance、concurrency 与 service load 也都会影响 end-to-end latency。

这在与 LLM 比较时尤其重要。公平的 benchmark 应该比较**实现同一个 decision 的完整系统**。如果 LLM 被要求生成长篇
explanation，而 Jev 只需输出一个 distribution，那么测试结果有一部分衡量的是 output protocol；如果 LLM 只通过 option
logit 评估、完全不允许 reasoning，测试又可能压制了它平时会使用的能力。两种比较都有信息价值，但回答的是不同问题。

真正合理的经济指标通常不是孤立的 “milliseconds per call”，而更接近：

$$
\text{cost per useful automated decision at target risk}
=
\frac{\text{model cost} + \text{fallback cost} + \text{error cost}}
{\text{correct cases completed without escalation}}.
$$

即使一个便宜的 first-stage model 准确率稍低，只要它经过 target traffic 验证的 score 能识别出自己能够处理的 case，
而且 fallback 与它的 error 并非完全相关，它就可能非常有价值。相反，一个速度很快、但 confidence 不可靠的 model 可能让
cascade 变得更差，因为它恰恰会自动处理那些最不应该自动处理的 case。

---

## Calibration：从评测指标到运行契约 {#calibration-from-a-metric-to-an-operating-contract}

Calibration（校准）是 Jev 叙事的核心，但这个词经常被当作“confidence 很好”的同义词。这太模糊了。我们必须区分
probability、confidence、correctness、discrimination、calibration、sharpness 与 decision value。

### Probability、confidence 与 correctness 是不同对象 {#probability-confidence-and-correctness-are-different-objects}

对一个有 $K$ 个类别的 decision，设 model 输出

$$
p(x)=(p_1,\ldots,p_K), \qquad \sum_{k=1}^{K} p_k=1.
$$

预测类别与 top probability 分别是

$$
\hat y(x)=\arg\max_k p_k(x), \qquad c(x)=\max_k p_k(x).
$$

观察到 true label $y$ 之后，把 correctness 定义为

$$
z(x)=\mathbf{1}[\hat y(x)=y].
$$

如果 model 满足 top-label calibration，那么直观地说：

$$
P(z=1\mid c=q)=q.
$$

也就是说，在所有 confidence 接近 0.8 的预测中，应该约有 80% 是正确的。这是关于一组可比较 case 的 population-level
陈述，并不是说某一个特定答案在形而上意义上“有 80% 是正确的”。

Accuracy 和 calibration 是彼此独立的维度。一个 model 可以很准确但过度自信：它答对 90% 的 example，却几乎给所有
答案都分配 0.999。一个较弱的 model 也可以校准良好但没有帮助：它对每个 example 都只预测 base rate。
**Discrimination** 问的是 model 能否把容易的 positive 与 negative 分开；**sharpness** 问的是它能否给出偏离 base rate、
真正有信息量的预测；**calibration** 问的是这些 probability 是否与实际观察频率匹配。

Jev 又引入了一个术语陷阱。文档中的 `confidence` 并不总是 raw top probability。对一个有 $n$ 个选项的 Choice question，
TypeSafe 定义：

$$
\operatorname{confidence}_{\text{Choice}}
=
\frac{p_{\max}-1/n}{1-1/n}.
$$

它把 **declared option 上的均匀分布**——不是任务的 empirical base rate——重新缩放为零，把 point mass 缩放为一。当选项数
为 2 时，0.75 的 top probability 对应的官方 confidence 是 0.5。这个 concentration statistic 并不会自动成为跨 question
或跨 task 可比较的 correctness probability。Score confidence 基于 modal level 周围的 normalized concentration；Noul 则
直接暴露 $P(\text{yes})$，没有单独的 confidence field
（[TypeSafe AI, 2026](https://docs.typesafe.ai/confidence)）。

如果 application 使用 gate 来控制 action，就必须明确 threshold 的是哪一个量：raw option probability、margin、entropy、
文档定义的 confidence statistic、经过校准的 correctness predictor，还是它们的某种组合。“把 confidence threshold 设为
0.9”这句话，在没有明确对象之前是不完整的。

### 如何测量 calibration {#how-calibration-is-measured}

Binary Brier loss 是

$$
\operatorname{Brier}(p,y)=(p-y)^2,
$$

它的 multiclass 形式是

$$
\operatorname{Brier}(p,y)=\sum_{k=1}^{K}
\left(p_k-\mathbf{1}[y=k]\right)^2.
$$

Brier score（布里尔分数）是一种 **strictly proper scoring rule（严格适当评分规则）**：在期望意义下，forecaster 只有
如实报告真实 conditional distribution 才能把它最小化。Negative log-likelihood 也是 proper 的，但当 model 给实际发生
的 outcome 分配接近零的 probability 时，它的惩罚会急剧增大。

Expected calibration error（ECE）按 confidence 对 prediction 分箱：

$$
\operatorname{ECE}
=
\sum_{b=1}^{B}\frac{|S_b|}{n}
\left|
\operatorname{acc}(S_b)-\operatorname{conf}(S_b)
\right|.
$$

ECE 很直观，但也很脆弱。它的值会随 bin boundary 和 bin count 改变。较小的 global ECE 可能掩盖某个 rare class、
language、customer 或 safety-critical subgroup 上的严重 miscalibration。因此，adaptive bin、classwise calibration、
reliability diagram、Brier decomposition、negative log-likelihood 与 bootstrap confidence interval 应该与它一起报告。
单独一个小数并不是 calibration audit。

这里还要区分 **marginal coherence** 与 **joint coherence**。Model 可能在“statement A 是否成立？”和“not-A 是否成立？”
两个问题上分别校准良好，但为两者分配的 probabilities 却不能保持一致。早期测试 Jev 的工作发现，相比某些 LLM
probability-readout 方法，它的一致性更高，但在 negation、等价 question format 与 mutually exclusive label 之间仍然存在
不可忽略的 inconsistency
（[Li et al., 2026](https://arxiv.org/html/2609.33209)）。Calibration 并不蕴含 logical consistency。

最后，calibration 总是相对于某个 distribution。令 $P_{\text{cal}}$ 表示拟合或验证 probabilities 时使用的 held-out data，
$P_{\text{deploy}}$ 表示真实在线 workload。即便 model 在 $P_{\text{cal}}$ 上完美校准，一旦发生 domain、language、
temporal、prompt 或 base-rate shift，也不代表 calibration 仍然成立：

$$
\operatorname{Calibrated}_{P_{\text{cal}}}(p)
\not\Rightarrow
\operatorname{Calibrated}_{P_{\text{deploy}}}(p)
\quad\text{when }P_{\text{cal}}\ne P_{\text{deploy}}.
$$

### Calibration 如何变成 control flow {#how-calibration-becomes-control-flow}

Calibrated probability 的运行价值，首先体现在 **selective prediction（选择性预测）**。设系统只有在某个 score 超过 threshold
$\tau$ 时才接受 model 的答案：

$$
a_\tau(x)=\mathbf{1}[c(x)\ge\tau].
$$

Coverage 与 selective risk 分别是

$$
\operatorname{coverage}(\tau)=P(a_\tau=1),
$$

$$
\operatorname{risk}(\tau)
=P(\hat y\neq y\mid a_\tau=1).
$$

提高 $\tau$ 通常会降低 coverage；如果 score 有用，也会降低 risk。完整的 risk–coverage（风险—覆盖率）曲线及其面积，
比某个任意 threshold 下的 accuracy 更有信息量。在产品中，我们也可以反过来问：给定 error budget $\epsilon$，能够达到的
最大 coverage 是多少？

$$
\max_\tau \operatorname{coverage}(\tau)
\quad \text{subject to} \quad
\operatorname{risk}(\tau)\le\epsilon.
$$

![从 probability 到 monitored decision 的 calibration stack](/assets/img/blog/jev-calibration/fig3_calibration_stack.svg)
*图 3. Probability 只有经过完整 calibration stack，才会产生运行价值：proper evaluation、经过验证的 threshold、
cost-sensitive action policy、fallback 与 drift monitoring。不同 action 需要不同 operating point，因为 false acceptance、
false rejection、delay 与 escalation 的成本不同。*

对 asymmetric decision，threshold 应该来自 cost，而不是习惯。令
$p=P(\text{the proposed action is correct}\mid x)$。在一个简化 binary case 中，自动执行的 expected loss 是

$$
L_{\text{act}}(p)=pC_{\text{correct}}+(1-p)C_{\text{error}},
$$

而 escalation 的成本是 $C_{\text{esc}}$。只有当 $L_{\text{act}}(p)<C_{\text{esc}}$ 时才应该行动。真实 policy 还可能取决于
calibration estimate 本身的 uncertainty、human queue capacity、latency budget，以及 action 是否可逆。

因此，production threshold 应该：

- 在与目标 workload 匹配的 held-out data 上拟合；
- 按 action 和 consequence level 分别选择，而不是全局共用；
- 按 subgroup、domain 与 time slice 做评估；
- 与 model version、question wording、option order 和 rubric 一起固定；
- 持续监控 drift，并定期 recalibrate；
- 配套一个经过 joint error 测量的 fallback；
- 一旦 accepted-set risk 超出契约范围，就能够 rollback。

这就是 calibration 成为 API 的含义。API 不只是返回一个看似合理的数字，而是支持一套可以审计的明确 policy：*高于这个
已经验证的 threshold，就自动处理；低于它，就花更多 compute 或寻求帮助。*

---

## RLCD：TypeSafe 公开了什么，又没有公开什么 {#rlcd-what-typesafe-disclosed-and-what-it-did-not}

TypeSafe 把 RLCD 展开为 **reinforcement learning for calibrated decisions（面向校准决策的强化学习）**。它的公开
材料表示，Jev 结合了一种新的 model architecture、parallel sampling，以及专门用于生成 calibrated probability
distribution 的 RL post-training。公司并没有公开 model size、architecture、training data、reward formula、optimization
algorithm、rollout construction、ablation 或 calibration procedure
（[TypeSafe AI, 2026](https://docs.typesafe.ai/introduction/machine-learning-primer)）。

这里有一个容易造成大量混淆的命名提醒：TypeSafe 的 RLCD，与 2023 年提出的
**Reinforcement Learning from Contrastive Distillation** 完全无关。后者在 preference-based language model alignment 中
恰好使用了同一个 acronym
（[Yang et al., 2024](https://arxiv.org/abs/2307.12950)）。

这划出了一条非常重要的 evidence boundary：

| 状态 | 我们能够说什么 |
|---|---|
| **Documented** | Jev 暴露 typed finite distribution；TypeSafe 提到 RLCD、新 architecture 与 parallel sampler。 |
| **Vendor-reported** | 在 TypeSafe 自己的 workflow evaluation 中，Jev 速度快、价格低且 calibration 良好。 |
| **Unknown** | 精确 RLCD objective、architecture、data、model scale、reward，以及每个 component 的因果贡献。 |
| **Independent hypothesis** | OpenJev-RLCD 展示了一种用 proper scoring rule 优化 reasoning policy 的可行办法；它不是 TypeSafe 已公开的实现。 |

![Jev 与 RLCD 的公开信息边界](/assets/img/blog/jev-calibration/fig4_rlcd_disclosure.svg)
*图 4. 内层方框表示已经记录的产品行为与 vendor claim；内部 training recipe 仍然封闭。OpenJev-RLCD 是独立研究提案，
不能被画进 Jev 的 proprietary pipeline。*

这一点尤其重要，因为产品发布后不久就出现了一篇名为 **OpenJev-RLCD** 的独立论文。根据名字反向推测 TypeSafe 的方法
很有诱惑力，但这篇论文并没有声称自己揭示了 Jev 的实现。

OpenJev-RLCD 从一个 generative reasoning model 出发。对于 input $x$，它采样 rationale
$r\sim\pi_\theta(\cdot\mid x)$，再读取 answer-option token 上的 distribution $u_\theta(x,r)$。对实际答案 $y$，它提出一个
shifted negative Brier reward：

$$
J(u,y)=2u_y-\|u\|_2^2
=1-\|u-e_y\|_2^2.
$$

和 negative Brier score 一样，它是 strictly proper reward：在期望意义上，如实报告真实 conditional distribution 才能
得到最高 reward。论文强调了一个细微的 variance identity。把 reward 扩展到 target distribution $q$，其中
$J(u,q)=2u^\top q-\lVert u\rVert_2^2$，有：

$$
J(\mathbb{E}_r[u],q)
=
\mathbb{E}_r[J(u,q)]
+
\mathbb{E}_r\|u-\mathbb{E}_r[u]\|_2^2.
$$

如果我们对 sampled rationale 的**混合分布**打分，最后一项会奖励 disagreement。对 one-hot outcome 而言，它可能表现为
ordinary correctness reward 加一个 diversity bonus。但 diversity 并不总是 epistemic uncertainty；它也可能只是 reasoning
不稳定。OpenJev 因而对每一条 rationale 自己的 distribution 打分，而不是奖励聚合后的 mixture。

作者报告，naive end-to-end optimization 有两个 failure mode。第一，除非用 base reasoning policy 作为 anchor，否则 policy
可能 collapse 到空或极短的 rationale——一种 “System-One collapse”。第二，通过 sampled reasoning 传播的 score-function
gradient，比通过 probability readout 传播的 pathwise gradient 噪声大得多。因此，它的 two-stage recipe 是
**calibrate, then reinforce**：

1. 采样 on-policy rationale，但先用 proper score 只更新 probability readout。
2. 从这个 calibrated checkpoint 出发，用 proper-score reward、leave-one-out baseline、KL anchoring 和降低权重的
   score-function term 来强化 rationale policy。

在 Qwen3-1.7B 上，论文报告其 GSM8K-Verify accuracy 与 GRPO 类似，但在 5% selective-risk target 下能够保留
明显更高的 coverage；论文也报告了 MMLU-Pro 上的 gain。这提供了有用证据：correctness-only RL 可能无法产生良好的
uncertainty ranking，而 proper-score objective 可以改善 selective behavior。但它还不是一个 general result：实验只使用
一个小型 model family、有限任务与 training step，以及三个 random seed
（[Gao and Wang, 2026](https://arxiv.org/html/2609.38850)；
[code](https://github.com/ZimmyGao/openjev-rlcd)）。

这里还有一个更深的问题：对一个直接输出 finite score 的 model，为什么一定要用 reinforcement learning？如果不存在
latent trajectory 或 delayed interaction，直接在 labeled data 上最小化 supervised Brier loss 或 cross-entropy，能够以
更低 variance 得到相同的 expected optimum。只有当 distribution 依赖 sampled reasoning、action 会改变后续 observation、
label 要等完整 trajectory 后才出现，或者 system 必须学习哪一种 information-gathering action 能改善最终 decision 时，RL
才更有吸引力。OpenJev 论文自己也发现，direct proper-score RL 并不会自动优于普通 supervised training。因此，“RLCD”
并不是 calibration 的魔法同义词；收益取决于 stochastic policy 与 feedback 在哪里进入系统。

---

## 周边研究版图 {#the-surrounding-research-landscape}

Jev 并不是在思想真空中出现的。它位于 calibrated classification、constrained generation、selective prediction、
reinforcement learning with verifiable rewards、uncertainty-aware agent 与 model distillation 的交叉点。真正值得问的不是
每个 ingredient 是否史无前例，而是这些 ingredient 如何被组装成开发者可以直接使用的 interface。

### RLVR：优化 correctness，而不是 confidence {#rlvr-optimize-correctness-not-confidence}

在 reinforcement learning with verifiable rewards（RLVR）中，reasoning policy 采样 answer 或 trajectory，并收到 unit-test
success、数学题 exact correctness 或 task completion 等 reward。GRPO 一类 group-relative method 会提高 successful sample
的 probability。这可以改善 pass@1 与 reasoning behavior，但 binary correctness reward 并不强迫 model 报告的 confidence
与 empirical success rate 匹配。Policy 完全可能同时变得更准确、也更过度自信。

更根本地说，pass@1 与 calibration 回答的是不同问题。Pass@1 估计一次 sample 成功的频率；calibration 问的是 model 是否
知道哪些 sample 更可能成功。后者才让 selective automation 与 compute allocation 成为可能。两个 model 可以有同样的
pass@1，但如果其中一个能准确地区分 easy 与 hard case，它在 cascade 中的价值会完全不同。

### RLCR：让 reasoning model 说出 confidence {#rlcr-teach-a-reasoning-model-to-verbalize-confidence}

Reinforcement Learning with Calibration Rewards（RLCR，带校准奖励的强化学习）保留了 generative reasoning model。
Model 同时输出 answer $\hat y$ 与 scalar confidence $q$。它的一个简化 reward 形式是：

$$
R_{\text{RLCR}}
=
\mathbf{1}[\hat y=y]
-
\left(q-\mathbf{1}[\hat y=y]\right)^2.
$$

第一项继续推动 correctness，第二项是 Brier-style calibration term。Accuracy term 很重要：如果 system 只优化 calibration
error，一种 degenerate policy 可以故意答错，同时报告零 confidence。RLCR 论文报告，在不损失平均 accuracy 的情况下，
HotpotQA 与数学任务上的 ECE 大幅下降，也观察到部分 out-of-distribution improvement；但自相矛盾的 answer 仍可能保持
很高 confidence
（[Damani et al., 2025](https://arxiv.org/html/2507.16806)；
[project](https://rl-calibration.github.io/)）。

因此，RLCR 与 Jev 处理的是相关但不同的 interface problem。RLCR 教会 generative model 在 reasoning 旁边报告 self-assessment；
Jev 直接返回 finite distribution，并不暴露 generated reasoning。前者适合 output 本身必须 open-ended 的场景；后者针对的
是交由软件消费的 bounded decision。

### CaOPD：在 distillation 中修正 privileged-teacher certainty {#caopd-correct-privileged-teacher-certainty-during-distillation}

Calibration-aware On-Policy Distillation（CaOPD，校准感知的 on-policy 蒸馏）解决的是另一种 mismatch。Strong teacher
可能在 training 时拿到 reference solution、privileged context 或其他部署阶段 student 无法获得的信息。如果 student 原样
distill teacher 的 confidence，它学到的将是适合 teacher information 的 certainty，而不是适合自己的 certainty。即使
reasoning transfer 很有价值，这仍会导致 optimism 与 entropy collapse。

CaOPD 用 $K$ 次 on-policy student rollout 估计 student 在 deployment information 下的 success：

$$
\hat\mu(x)=\frac{1}{K}\sum_{k=1}^{K}R(x,a_k),
$$

其中 $R$ 是 verifier，$a_k$ 是 student sample。它保留 teacher 的 trajectory 或 reasoning target，但用 $\hat\mu(x)$ 替换
privileged confidence target，然后执行标准 on-policy distillation。简言之，它把**应该怎么做**与**部署后的 student 应该
多么确信**解耦，而且不要求再增加一个单独的 calibration-reward RL stage
（[Zhang et al., 2026c](https://arxiv.org/html/2604.16830)；
[code](https://github.com/SalesforceAIResearch/CaOPD)）。

它的代价是额外 training compute 与对 verifier 的依赖。较小的 $K$ 会产生离散且 noisy 的 target，较大的 $K$ 又很昂贵。
Verbalized confidence 还可能格式错误，或被 model 策略性生成。Jev 的 typed distribution 消除了 parsing failure，但没有消除
那个更深的问题：confidence 必须相对于 deployment 阶段真正可用的信息和 distribution 做校准。

### ACC 与 AUQ：从原子答案走向长 trajectory {#acc-and-auq-move-from-atomic-answers-to-long-trajectories}

Agentic Confidence Calibration（ACC）提出的问题，是估计完整 agent trajectory 最终成功的 probability。论文进一步提出
**Holistic Trajectory Calibration（HTC）** 方法，利用 process-level signal——而不只看 final answer——估计 trajectory
success。这一点非常重要，因为一个 long-horizon agent 可能在每一步都局部自信，但整套 plan 的 success probability 会因
乘法效应或 correlated hidden error 持续下降
（[Zhang et al., 2026a](https://proceedings.mlr.press/v306/zhang26gy.html)）。

Agentic Uncertainty Quantification（AUQ）进一步让这个估计产生 action。Uncertainty 可以决定检索哪条 memory、agent 是否
应该 reflect、在哪里重新审视 decision，以及何时请求帮助。AUQ 不是在完成后附加一个 confidence number，而是在 trajectory
仍然可恢复时就改变它
（[Zhang et al., 2026b](https://arxiv.org/abs/2601.15703)）。这与近期 survey 中从 passive uncertainty
measurement 转向 active uncertainty-guided computation 的整体趋势一致
（[Zhang et al., 2026d](https://aclanthology.org/2026.findings-acl.2064/)）。

![按 output type、training signal 与 decision horizon 划分的 calibration method 地图](/assets/img/blog/jev-calibration/fig5_method_landscape.svg)
*图 5. RLVR 优化 success；RLCR 为 generative reasoning 加入 calibration reward；OpenJev-RLCD 优化以 sampled rationale
为条件的 option distribution；CaOPD 在 distillation 中修正 certainty；Jev 原生暴露 typed distribution；ACC 与 AUQ 则
从 atomic answer 走向 trajectory-level control。*

最容易理解的对比如下：

| Method | 主要 output | Calibration target | 主要 training mechanism | Horizon |
|---|---|---|---|---|
| RLVR / GRPO | answer 或 trajectory | 无显式 target | verifiable correctness reward | answer 到 trajectory |
| RLCR | reasoning + verbal confidence | generated answer 的 correctness | accuracy + proper-score RL reward | answer |
| OpenJev-RLCD | rationale-conditioned option distribution | option correctness | two-stage proper-score training 与 RL | answer |
| CaOPD | distilled reasoning + confidence | student on-policy success rate | reverse-KL on-policy distillation | answer / rollout set |
| Jev | typed finite distribution | proprietary / 未公开 | TypeSafe 称之为 RLCD；recipe 未知 | atomic decision |
| ACC | trajectory-success probability | final trajectory outcome | trajectory-level calibrator | full trajectory |
| AUQ | adaptive agent behavior | uncertainty-conditioned improvement | memory 与 reflection control | full trajectory |

关键的概念演进并不是“哪一个 acronym 获胜”，而是：

```text
measure uncertainty
    → calibrate it
    → use it to abstain
    → use it to allocate computation
    → use it to change the trajectory
    → learn from the intervention outcome
```

Jev 为这条链的中间部分提供了一个异常干净的 primitive；完整 agent 仍然必须实现其余环节。

---

## 现有证据究竟说明了什么？ {#what-does-the-evidence-actually-show}

由于 Jev 很新、而且是 closed model，相关证据对时间异常敏感。一份 evidence-aware 文献索引在 10 月 3 日最后核验时，
记录了 77 篇 Jev-specific 或 Jev-style academic preprint，而首个九日审计只有 28 篇
（[Awesome JEV Papers, 2026](https://github.com/Oscar-dzy/Awesome-jev-papers)）。数量不等于成熟度：几乎所有研究都只覆盖
同一个 hosted release window，很多重复使用公开 benchmark；该索引也把这些 academic item 归为 preprint，而不是已经通过
同行评审的 publication。当前有用的证据层包括 TypeSafe 的 launch evaluation、覆盖面较广的独立 benchmark、针对性的
controlled audit，以及 system-level application。没有任何一层支持“typed decision model 普遍优于 LLM”这一结论。

### 厂商证据 {#vendor-evidence}

TypeSafe 报告的 end-to-end latency 约为 70–500 ms，并称在选定的 LLM workflow baseline 上具有很大的速度与价格优势。
其中一个 headline comparison 报告最高可达 193.6 倍速度和 444.6 倍成本优势。事实上，公司自己的 launch material 比
headline 更谨慎：它指出 compact、dense state 可能更有利于 Jev；部分 reference label 来自 LLM judgment，而不是 human
gold；workflow 是由公司自己的 capabilities team 构建的；强制 general LLM 使用同一个 adapter，也未必代表那些 LLM
最合适的使用方式。公开价格同样无法告诉我们当前 economics 是否受到补贴，或者能否长期维持
（[TypeSafe AI, 2026](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）。

这些结果说明，当前 service 在官方选择的 workload 上确实可以很快、很便宜；但它们没有分离 accuracy 或 calibration 究竟
来自 architecture、training data、RLCD、model scale、task selection，还是 serving stack。

### 一项大规模独立 benchmark {#a-large-independent-benchmark}

一项早期独立研究使用每个 dataset 一个固定 template，在 37 个公开 dataset 上总共发出了 346,009 次 request。论文报告
共处理 2.179 亿 input token，API 总费用为 \$9.15；在 concurrency 32 下，client-side mean latency 为 0.36 秒。Jev 在
IMDB、SST-2、HellaSwag 与 ARC 上达到 95–99% accuracy，在覆盖 122 种语言的 Belebele benchmark 上达到 86.7%。在基于
exact option probability 的 single-forward comparison 中，它在 37 个 dataset 中有 27 个优于 Qwen3.8-27B，并在全部
37 个 dataset 上优于 Gemma-4-E4B
（[Deußer et al., 2026](https://arxiv.org/html/2609.37647)）。

Calibration 结果比 aggregate accuracy 更有信息量。论文在 22 个 dataset 上报告 pooled Choice ECE 0.028、per-dataset
median ECE 0.028，以及 per-dataset mean ECE 0.061。这些 summary 掩盖了明显失败：Emotion 的 ECE 为 0.279，AfriXNLI
为 0.165。Noul 的 ECE 为 0.052，而且常常需要 task-specific threshold；在 training data 上调整 threshold 后，
UNFAIR-ToS micro-F1 据称从 0.50 提升到 0.75。

Selective prediction 的结果很有希望。在 50% coverage 下，accepted-set accuracy 据称在 Banking77 上从 79.7% 提升到
96.3%，在 SIB-200 上从 81.5% 提升到 98.2%，在 ANLI 上从 73.9% 提升到 86.9%，在 Emotion 上从 58.5% 提升到
73.3%。这正是 decision API 所需要的性质——前提是 threshold 能够迁移到 target traffic：即使 absolute task performance
并不完美，score 仍然能够识别更容易的 case。

不过，实验设计只支持更狭窄的解释：

- comparator LLM 没有使用 generated reasoning；
- 每个 dataset 只使用一个 template 和一次主要 run，因此 prompt 与 run variance 未知；
- 对任何 closed model，都无法排除 benchmark contamination；
- threshold tuning 使用了 label，而且可能无法在 shift 后 transfer；
- 在一些 low-resource、fine-grained、noisy、legal 与 rubric-heavy task 上，performance 仍然较弱；
- 旧公开 benchmark 上的强结果，并不能证明 private operational data 上的效果。

最公平的结论是：对许多 bounded semantic decision，Jev 似乎同时提供了很强的 speed–cost–accuracy operating point，以及
有用但不均匀的 confidence quality。它并不是“不需要 reasoning”的证据。

### 早期证据综合与 domain study {#early-evidence-synthesis-and-domain-studies}

一篇覆盖发布后最初九天文献的 evidence review 得出了相似而谨慎的结论。它发现，median single-call latency 常见于
0.15–0.45 秒，在部分研究中成本优势巨大；但没有 controlled evidence 能证明 typed interface 本身比 matched
label-logit readout 更准确。Raw calibration 随 workload 明显变化：post-hoc recalibration 能显著改善部分任务；high
confidence 可以与接近 base rate 的 accuracy 共存；错误 tool call 也可能保持高度自信
（[Tang and Zheng, 2026](https://arxiv.org/html/2609.32160)）。

另一篇早期 matched preprint 同时比较了 trained classifier、decision-model checkpoint 与 generative comparator，进一步说明
结果高度取决于具体条件。在部分有 label 的 intent task 上，小型 supervised classifier 最好；一个更大的 generative
comparator 在 workflow 和 intent 上与 Jev 相当，而且在 5% risk 下接受了更多 workflow case。一个用 held-out data、以 5%
in-scope risk 为目标的 gate，仍然接受了 31% 的 out-of-scope request。但在论文的 utilization assumption 下，先用便宜
classifier、再把 uncertain case escalate 给 Jev 的 cascade，只需 Jev 约 43% 的估计成本，就能匹配 Jev accuracy。实际教训
并不是某一类 model 永远获胜，而是 label、output space、risk target 与 cascade design 共同决定 frontier
（[Rafe and Das, 2026](https://arxiv.org/abs/2610.00346)）。

一篇 medical preprint 在 MetaMedQA 上报告 Jev 的 ECE 为 0.063，而 comparator 为 0.146；对分配至少 0.9 probability 的
example，accuracy 为 93.4%。但在 DiagnosisArena 上，Jev 报告的 AUROC 只有 0.645，而 comparator 为 0.768。Median
latency 约为 0.27–0.31 秒，2,823 个 item 的总费用据称只有 \$0.08。这是值得关注的 evaluation evidence，但不能支持
autonomous clinical use
（[Madrid-García and Merino-Barbancho, 2026](https://arxiv.org/abs/2609.34024)）。

前面提到的 probability-coherence study 又给出了一项警告。对 480 组 negation pair，它报告 Jev 的平均

$$
|P(X)+P(\neg X)-1|=0.064
$$

在该 setup 下优于两个 Qwen readout，但距离精确 coherence 仍然很远。对三个 mutually exclusive label，Jev 的 yes
probability 平均加总为 1.14；同一个 label 在 Noul 与 Choice formulation 下相差约 0.09。实验规模很小、也未经过同行评议，
但它说明了为什么独立 question answering 不能被自动视为一个 globally consistent probability model
（[Li et al., 2026](https://arxiv.org/html/2609.33209)）。

### 一个 calibrated decision system 必须通过的四项测试 {#four-tests-that-a-calibrated-decision-system-must-pass}

10 月出现的一批新论文让一个概念区分变得无法回避：“这个 model 是否 calibrated？”并不是一个单一问题，而至少包含四层：

1. **Marginal calibration（边际校准）。** 在目标分布上，所有报告约 0.8 probability 的 decision，是否大约有 80% 的 selected
   option 是正确的？
2. **Self-knowledge（自我知识边界）。** 当必要信息缺失、已经过时，或超出 model 的 knowledge boundary 时，confidence
   是否会下降？
3. **Probabilistic coherence（概率一致性）。** 等价问题、互补事件、层级结构与 coarsening，是否用彼此兼容的 probability
   描述同一个 event？
4. **Decision utility（决策效用）。** Probability 经过 threshold、defer cost、cascade 与 authority rule 后，是否真的带来
   更好的 action？

通过前一层，并不意味着自动通过后一层。一个 model 可以平均 calibration 良好，却对自己不知道的 fact 依然自信；两个分别
calibrated 的 interface 可以对同一个 event 得出不一致结论；更好的 Brier score 可能完全不改变 thresholded action，也可能
把它推过错误边界。反过来，一个稍差的 forecaster，如果在某个 operating point 上更能识别 fallback 真正可以修复的 case，
反而可能更有用。

![可靠 probability 的四个逐层提高的含义](/assets/img/blog/jev-calibration/fig6_calibration_stack.svg)
*图 6. Calibration 只是第一层，而不是整架梯子。真正部署还要求：面对信息缺失时具备 self-knowledge、在等价表示之间保持
coherence，并在实际 policy 下产生 utility。图中论文按其直接检验的最强问题放置，而不是按作者更广泛的主张分类。*

这一框架可以解释多项看似矛盾的结果：

- **Confidence 并不是缺失知识的 detector。** 一项覆盖 15 个以上公开 dataset 和六组生成 task family 的 controlled audit
  报告：在没有任何 answer-relevant information 时，Jev 仍可能给某个显眼选项高达 0.80 probability；对于超过观察到的
  knowledge boundary 的新闻，confidence 比 accuracy 高 0.21–0.33，而且用更早月份做 recalibration 也没有消除差距。
  直接问“当前 evidence 是否充分？”比 answer confidence 本身更有诊断价值。但这并不能证明第二个问题总有效；同一研究也
  表明，“你知道吗？”可能只是在读取 surface cue
  （[Shankaranarayana et al., 2026](https://arxiv.org/html/2610.01006)）。
- **合法 distribution 仍可能违反 probability contract。** 在 1,000 个可以精确计算的 finite world 中，当 defer cost 为
  0.10 时，Jev 的 Event 与 Choice interface 在 32.8% 的有效 matched pair 上导出了不同 binary action。教训不是把所有
  分歧求平均，而是要在部署前明确 event、loss 与 interface，并测试从 report 到 action 的完整 mapping
  （[Chen and Li, 2026](https://arxiv.org/html/2609.37470)）。
- **Flat 与 hierarchical decision 未必能够“加总”。** 在 TREC、CLINC150 和 MASSIVE 上，hierarchical reconstruction
  让 Jev 的 category-level distribution 出现 0.219–0.349 的 total variation。在 CLINC150 上，reconstructed
  distribution 使 accuracy 下降 22.9 个百分点，尽管两条路径本意都是描述同一个 fine label。因此 taxonomy 是 model
  contract 的一部分，而不是无害的 UI 细节
  （[Joy, 2026](https://arxiv.org/html/2609.33971)）。
- **Evaluation 与 simulation 是不同能力。** 在一项研究中，Jev 能答对 99% 的 Cognitive Reflection Test lure question，
  但当一次调用既要预测隐藏 state、又要用预测结果评价 action 时表现明显较差。在 misleading game condition 中，直接提供
  simulated opponent action，可把 performance 从 33% 提高到 98%；用 code 做 lookahead，则可把 solved ALFWorld game
  从 31% 提高到 87%。更合理的抽象因此是：*code 负责 state transition 与 simulation，decision model 评价 bounded
  candidate*
  （[Yang et al., 2026](https://arxiv.org/html/2610.01834)）。
- **显式 fallback option 并不会被自动理解。** 在一项 controlled arithmetic task 中，答案存在时 accuracy 为 99%，但答案
  缺失时，正确 reject 率只有 7%。在独立 development set 上拟合并冻结 threshold 后，reject 率升至 79%，同时保留 97%
  的 answer-present accuracy。“None of the above”同样需要 data、policy 与 validation；仅增加一个 label 并不够
  （[Zhong et al., 2026](https://arxiv.org/abs/2609.39496)）。

### 按证据质量加权的直接研究地图 {#a-quality-weighted-map-of-the-direct-evidence}

下表有意保持选择性。**较强（Stronger）**表示：针对表中问题，研究覆盖面较广或控制较好，并尽可能提供 uncertainty
analysis 与可用 artifact。**有用（Useful）**表示：结果具有信息量，但范围更窄，或 label/comparator 存在重要限制。这些
标签是我对 study design 的判断；下表每一项直接 Jev 研究仍然都是近期 preprint。

| 研究 | 检验什么 | 规模 / 控制 | 主要信号 | 权重与边界 |
|---|---|---|---|---|
| [Broad Jev benchmark](https://arxiv.org/html/2609.37647) | accuracy、calibration、latency、cost | 37 个 dataset；346,009 次 request；exact Qwen/Gemma option-likelihood baseline | speed–cost operating point 很强；calibration 与 accuracy 随 task 剧烈变化 | **较强**：覆盖面广；但每个 task 仅一个 template/run，且可能存在 public-data contamination |
| [Automated decision gates](https://arxiv.org/abs/2610.00346) | matched gate 与 cascade | 6 个 family 的 8 个 checkpoint，加 classifier 与 LLM；held-out threshold；bootstrap | 有 label 时 classifier 可胜出；LLM readout 可匹配 Jev；便宜 cascade 可到达 frontier | **较强**：matched comparison；但若干 label 是 proxy，而不是 production gold |
| [JevAdvBench](https://arxiv.org/html/2609.31142) | adversarial state sensitivity | 812 个 question、66 个 scenario、9,744 次 edit；含 control 与公开 output | 无关或操纵性 context 可以重定向 confident decision，或压低其 confidence | **较强**：robustness design；但多数 label 来自 Jev 自己的 clean output |
| [LLM2Jev](https://arxiv.org/html/2610.02076) | 是否必须有特殊 architecture | Qwen3.5-4B 与 Qwen3-0.6B；training-free 和 fine-tuned readout；含 ablation | 强 4B LLM 已经很有竞争力；tuning 能帮助弱项或 targeted case，也会产生 negative transfer | **有用–较强**：architecture evidence；使用 public JevBench diagnostic，且没有 proprietary matched backbone |
| [Beyond Answer Confidence](https://arxiv.org/html/2610.01006) | missing knowledge 与 information sufficiency | 15+ dataset、6 个 controlled task family、paired intervention | answer confidence 不能可靠暴露 ignorance；针对性 evidence question 更有帮助 | **较强**：black-box audit；但 targeted probe 同样需要 cue control |
| [Probability Contracts](https://arxiv.org/html/2609.37470) | exact posterior、coherence、decision loss | 1,000 个 finite world；4 种 interface configuration | interface disagreement 经常改变 downstream action | **较强**：internal validity；synthetic finite world 限制 external validity |
| [Do Decisions Add Up?](https://arxiv.org/html/2609.33971) | flat 与 hierarchical consistency | 3 个 dataset，每个 system 2,500 个 matched example；72K question | 等价 decomposition 可产生明显不同的 distribution 与 accuracy | **较强**：coherence evidence；但只覆盖 3 个 classification dataset |
| [Agent security decisions](https://arxiv.org/html/2609.33401) | attack shift 下的 allow/block/review | Jev、Laya、Decider、Nimble、classifier 与 LLM judge，覆盖多个 security task | aggregate calibration 掩盖 subgroup failure；严格 risk limit 下可自动化比例很低；judge 共享 confident error | **有用–较强**：policy analysis；benchmark security 不等于 live threat model |
| [OmniMed-Jev](https://arxiv.org/html/2610.00381) | decision-native 与 generative multimodal training | 同一个 4B backbone、23,558 个 state、3,685 step；726 个 held-out state | point accuracy 相似，报告的 calibration 显著更好；generative arm 在 counting 上仍胜出 | **有用**：matched schedule；typed arm 获得 155,729 个 factorized target，存在 supervision-density confound |
| [Code Owns the Simulation](https://arxiv.org/html/2610.01834) | direct evaluation 与 hidden-state simulation | CRT、matrix game、ALFWorld、robot control | 显式 simulation 或 lookahead 可把许多失败转化为强 bounded decision | **有用–较强**：capability boundary；只有一个 model version，且 setting 有人为构造成分 |
| [OpenJev-RLCD](https://arxiv.org/html/2609.38850) | 一种开放的 proper-score training recipe | Qwen3-1.7B、两个主要 reasoning task、3 个 seed、多种 baseline | calibrate-then-reinforce 改善 selective prediction | **有用**：method artifact；不是 TypeSafe proprietary RLCD 的 reconstruction |

其中两项发现尤其值得强调。第一，[LLM2Jev](https://arxiv.org/html/2610.02076) 削弱了“必须发明一种全新的 decision
architecture”这一说法。它直接读取 causal LM 对 bracketed numeric candidate suffix 的分数，在 candidate 间共享 prompt
computation；只有在确实需要 adaptation 时，才加入 tree-factorized listwise objective 与 KL anchor。4B backbone 本身已经是
一个有能力的 decision model；不加区分地继续使用短输入训练，可能损伤 long item，而加入 long-input data 后可大体恢复。
这比一句简单的“fine-tuning 获胜”更加细致。

第二，[OmniMed-Jev](https://arxiv.org/html/2610.00381) 提供了最清晰的 interface-controlled comparison 之一，也展示了“相同
data”多么容易掩盖不同监督量。两条 arm 都看到同样的 23,558 张 image，但 typed arm 把 multi-label annotation 分解成
155,729 个 decision target，而 generative arm 每个 state 只得到一个 answer target。它的 calibration 结果值得重视，但实验
并没有把 interface 与 supervision density 分离。今后的研究应该把这类 accounting 作为标准报告项。

因此，当前 evidence ledger 是：

| Claim | 当前支持 | Confidence |
|---|---|---|
| Jev 价格低、latency 小 | official pricing 加多项早期测量 | 对当前 hosted service 相对较强 |
| Jev 在许多 finite-choice task 上有竞争力 | broad early benchmark、vendor workflow | 有希望，但取决于 workload |
| Raw probability 普遍 calibrated | 被 per-task variation 反驳 | 不支持 |
| Answer confidence 能可靠识别 missing knowledge | controlled information 与 temporal-boundary audit 反驳这一点 | 如果没有单独的 evidence-sufficiency test，则不支持 |
| 等价 interface 定义了同一个 coherent belief state | complement、hierarchy、coarsening 与 Event/Choice study 均提出反例 | 不支持 |
| 必须使用专门的 typed head | frozen LLM readout 与 trained classifier 经常能够匹配或胜过 typed model | 作为一般命题不支持 |
| RLCD 导致了观察到的 calibration | 没有公开 ablation 或 recipe | 未知 |
| Jev 可以取代 generative LLM | output contract 无法覆盖 open-ended work | 作为一般命题是错误的 |
| Confidence 可以改善 cascade | selective prediction result 与成熟理论 | 当 threshold 可 transfer 且 fallback 互补时，有支持 |
| 应由 code 负责 simulation，Jev 评价给定 alternative | 一项跨 domain controlled study 加 system intuition | 有希望，但还不是普遍规律 |

---

## 开放的 Jev 生态 {#the-open-jev-ecosystem}

Jev 已经不只是一个产品名。它现在也描述了一类**接口家族**：读取 state，评价调用者定义的 finite set，返回 normalized
distribution，并由 code 决定 action。这个 interface 并不唯一决定 architecture。开放项目已经用 frozen LLM logit、
LoRA-tuned decoder、encoder classifier、custom pointer head、joint-embedding model、multimodal prefix sharing 与
rationale-conditioned RL 实现了它。

![从 evaluation resource 到 deployment 的开放 decision-model 生态](/assets/img/blog/jev-calibration/fig7_ecosystem_map.svg)
*图 7. 公开生态是一整套 stack，而不是单一 model family。Dataset 与 evaluation contract 位于 readout 和 training method
之下；model 与 runtime 暴露 typed interface；cascade 与 application 决定 probability 是否真正创造价值。一个项目可能同时
占据多个模块。*

### 四类实现路径 {#four-implementation-families}

理解这些项目最简单的方式，是看它们的 decision distribution 从哪里产生：

1. **Frozen decoder readout。** 把 option 格式化为字母或数字标识符，再从普通 causal LM 读取相应 next-token probability。
   SemIf、Cygnet 和 LLM2Jev 的 training-free 部分说明了这条路线可以走多远。它很容易迁移到新 backbone，并保留 generation
   能力，但仍然对 option token、顺序、prompt format 与 tokenizer structure 敏感。
2. **Adapted decoder readout。** 在 decision data、hard negative、teacher question、replay 与 calibration objective 上，
   fine-tune 同一个 backbone——常见方式是 LoRA。JevK5、Plumb、Decider 与 tuned LLM2Jev 属于这一类。Adaptation 可以修复
   已知弱项，也可能学到 benchmark 结构、损伤 long-input behavior，或 overfit public suite。
3. **Encoder 或 custom decision head。** 先编码 state 与 question，再用 classifier、pointer、joint embedding 或
   value-of-information head 给 option 打分。Laya、Kev、PACT、Bongard、Chinese-Jev 与 LAVOIR 探索了这一空间。这类模型
   可以很小、很快，但 open-domain transfer 高度依赖 training coverage。
4. **System composition。** 把 simulation、retrieval、hard policy 与昂贵 reasoning 放在 decision model 之外。AnyJev、
   JEVDB、Mnemon、confidence cascade 与 security gate 都说明，surrounding algorithm 与 checkpoint 同样重要。

### 值得阅读的开放模型与方法 {#open-models-and-methods-worth-reading}

“Open”可能指 code、weight、training recipe、evaluation harness，或四者全部。下表明确列出实际开放了什么，也避免把作者
自己跑出的 public score 当作 independent reproduction。Adoption 数字是带日期的 **2026 年 10 月 4 日**快照；star、like
与 rolling download 衡量的是关注度，而不是技术质量。

| 项目 | Backbone / artifact | Mechanism | 开放状态 | 最强公开信号 | 重要边界 |
|---|---|---|---|---|---|
| [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev)，原名 OpenJev | 运行在 frozen causal LM 上；默认 Qwen3.5-4B | option-token logit、shared state prefill、parallel question suffix | MIT code；遵循上游 model license；约 4.7K star | 作者在 37×21 prefix-reuse workload 上报告 20.03 decisions/s | 没有项目 checkpoint；BF16 fast path 有轻微 argmax drift；不是 Jev training reproduction |
| [OpenJev 27B](https://huggingface.co/openjev/openjev) | tuned Qwen3.5-derived 27B weight | first-position decision readout 加 fixed calibration | CC BY-NC 4.0 weight；Apache helper/server；约 5K monthly download | 作者在 10K text question 上报告 84.0%，hosted Jev 为 85.4% | 3,078 个 item 影响过 development；training recipe/data 未公开；noncommercial license |
| [open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) | 运行在 frozen LM 上；报告 Qwen3.6-27B test | next-token option probability；packed question 或 prefix-cached request | Apache-2.0 library；约 62 star | RACE-H 1K：packed 92.9%，逐题 92.6%；速度收益来自 batching | 记录了 packing interference、position bias 与 sub-4B 弱项；没有新 weight |
| [Cygnet](https://github.com/blockbrain-ai/cygnet-recipe) | frozen Gemma-4-12B-it | one-pass letter logit 加一个 fitted temperature | MIT recipe/shim/calibration asset；**没有新 weight 或 adapter** | author-run public JevBench 为 203/231（87.9%）；A6000 p50 66 ms | near tie 可能随 hardware/batching 改变；>20-option multi-pass calibration 测试不足 |
| [JevK5](https://github.com/allebee/jevk5) | Qwen3.5-4B/9B merged LoRA weight | teacher-distilled question、double-checking、replay、letter logit | Apache-2.0 code/weight；约 134 star | v0.3 public hard 78.4%，ECE .054 | 报告的 paired test 中 v0.3 并未显著优于 v0.2；开发时已查看 public suite |
| [Plumb-4B](https://github.com/crh225/plumb) | JevK5 v0.2 + 独立 4B LoRA | teacher generation、consistency filtering、hard mining、long document、public replay | Apache-2.0 recipe/weight | v5 public hard 80.2%；相对起点 8 个修复、1 个 regression，exact McNemar p=.039 | 持续针对 public suite 优化；v5 accuracy 提升，但 ECE 相比 v4 变差 |
| [reflex](https://github.com/kshetrajna12/reflex) | stable path 使用 frozen Qwen3.5-4B；browser demo 为 0.8B | evidence/criterion prompt；对 yes/no 两种顺序取平均 | MIT runtime/experiment | stable 4B public hard 68.5%，ECE .081；27B hard 76.6% | 有价值的 negative result：generic adapter 损伤 open judgment；双顺序平均不是一次 read |
| [Decider](https://github.com/Mapika/decider) | Qwen3.5 0.8B/2B/4B/35B-A3B，加 Gemma-4 12B family | family-specific CE/SFT/LoRA recipe 与 prefix caching | Apache-2.0 code、weight 与 server | 报告的 held-out accuracy 随 scale 从 4B 的 .784 升到 35B 的 .810 | 不能用一套 recipe 概括整个 family；所述 mixture 约 60% 可复现，teacher bias 仍存在 |
| [AnyJev](https://github.com/nokia-applied-research/AnyJev) | 覆盖多个 Qwen3 size 的 adapter | L0 rotation/prior correction；L1 temperature；L2 per-question closed-form hidden-state head | Apache-2.0 code/runtime；约 1K star | BANKING77 20-way Qwen3-8B：raw acc/ECE .747/.240 → L1 .807/.095 | L2 针对每个 question/schema 单独拟合，不是一个通用 converted checkpoint |
| [Kev](https://github.com/jaredpalmer/kev) | Qwen-derived 0.8B、4B、9B、27B family | LoRA 或 full tuning 加 pointer head、shared state cache、fitted temperature | Apache-2.0 weight、完整 recipe、runtime/server；约 8.4K star | 作者报告 scale 增长时 held-out accuracy 为 .697/.838/.852/.889 | 与 hosted Jev 的比较不受控；各 size 的 context 与 post-training coverage 不同 |
| [Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B) | Gemma-4-12B-it 加 merged rank-32 LoRA | shared prefix 与 branched readout；保留 chat 和 vision mode | Apache-2.0 weight/runtime；约 27.8K monthly download | author-run public JevBench 85.71%，ECE .0742 | private training mixture 与 benchmark-aware refinement；“clean” scorer label 不能证明没有 contamination |
| [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | Gemma-4-12B-it multimodal weight | 面向 text、image、short audio、16-frame video 的 typed readout；每次调用一个 question | Apache-2.0 weight/loader；约 362 like | self-reported JevBench 87.45%、MMAU 63.10%、MVBench 53.10% | data、overlap audit 与 recipe 未披露；latency 不含 preprocessing/network |

Architecture 的多样性本身就是结论。SemIf 的旧名与另一个 OpenJev 27B checkpoint 尤其容易混淆；二者是无关项目。这些
system 都不是 Jev 的 reconstruction；它们说明 public API 无法唯一决定内部实现，也说明很多 observable behavior 可以用
熟悉的 modeling ingredient 构建。对 training research，另外几项最有用的 artifact 是
[OpenJev-RLCD](https://github.com/ZimmyGao/openjev-rlcd)、
[PACT](https://github.com/BennyLinntu/PACT-Pairwise-Anchored-Calibrated-Tuning-for-Single-Token-Typed-Decisions)、
[LAVOIR](https://github.com/moganai/lavoir)、[Visual Jev](https://github.com/guanxuyu-sv/Visual-Jev)，以及
[Bongard-mini](https://huggingface.co/AgentBull/bongard-mini) weight。

### 一个 leaderboard 截面，而不是通用排名 {#a-leaderboard-snapshot-not-a-universal-ranking}

公开的 [JevBench repository](https://github.com/fstandhartinger/jevbench) 很有价值，因为它发布 adapter、scoring code、public
item、aggregate artifact，以及不断演化的 sealed component；它也恰好说明 leaderboard 为什么必须带 version。其 score
组合了 chance-corrected intelligence、calibration、latency 与 estimated cost。Self-hosted latency/cost 必须依赖假设；
public item 可以被用于 training；sealed family 即使不暴露 exact item，也可能影响 data generation；改变各 axis 的权重，
排名也会改变。

本文冻结的当前快照是 **2026 年 10 月 4 日的 JevBench v1.5.7**：每个完整 system 共有 1,624 个 decision——904 个 open、
720 个 sealed；117 个 system 中有 111 个进入排名。其 equal-axis view 的领先条目如下：

| System | Equal-axis composite | Access | 分数由什么驱动 | 阅读时要警惕… |
|---|---:|---|---|---|
| Cygnet | 73.70 | frozen Gemma-4-12B-it；open recipe | public/sealed capability 与 calibration 较强，并具有有利的 self-hosted operating point | 没有新 weight；只有一个 temperature，near tie 依赖 hardware |
| Winnow-12B Q8 | 73.23 | open merged Gemma adapter | 较广的 decision performance 加 local serving | private training mixture；benchmark-aware refinement；cost/latency 是 deployment assumption |
| Jev 1.13.0 | 72.13 | proprietary hosted API | capability 与 calibration 较强，hosted economics 简单 | architecture/data 未知；network path 与 tariff 不同于 self-hosting |
| JevK5 v0.3 | 约 71.9 | open Qwen3.5-4B LoRA | compact teacher-distilled model 加 replay | 开发时使用过 public suite；其 paired test 中 v0.3 相比 v0.2 的提升并不显著 |

更早的 v1.4.2.2 snapshot 仍有科学价值，因为 LLM2Jev 在该冻结 protocol 下记录了每个 system 的 public/sealed accuracy：
例如 Plumb 为 89.6% / 38.0%，Cygnet 为 87.9% / 33.8%，Jev 为 86.6% / 36.7%。这些 gap 说明 public score 与 sealed
score 回答的是不同问题。

以上并不意味着 Cygnet 或 Winnow 在一般意义上“优于 Jev”。它只描述一个 composite、一个版本、一套 hardware/pricing
model，以及一种 task mixture。[Live board](https://benchmarkheaven.com/jev-models) 变化很快；V2 冻结 retrieval date，
是为了避免后续变动悄悄改写本文论点。

### Dataset、benchmark 与 evaluation resource {#datasets-benchmarks-and-evaluation-resources}

目前没有任何一个 suite 能同时测量 accuracy、calibration、self-knowledge、probability coherence、option semantics、
open-set rejection、latency 与 downstream utility。因此，表面上的 frontier 会随着 benchmark contract 改变。

| Resource | 规模 | 测量内容 | Artifact | 不能证明什么 |
|---|---:|---|---|---|
| [JevBench](https://github.com/fstandhartinger/jevbench) | v1.5.7 snapshot：每个完整 system 904 个 open + 720 个 sealed decision；111 个进入排名 | 在大型 community roster 上组合 intelligence、calibration、latency 与 estimated cost | item、adapter、scoring code、open outcome；sealed aggregate | 单一 deployment optimum；composite 内含 value 与 cross-hardware assumption |
| [Jev Benchmarking Suite](https://arxiv.org/html/2609.37647) | 37 个 dataset；346,009 次 request；2.179 亿 input token | broad task accuracy、ECE、selective prediction、multilingual behavior、exact LLM option readout | [code 与 raw response](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking) | prompt/run variance，或无 contamination 的 private performance |
| [DecisionBench evaluation](https://arxiv.org/html/2609.39111) | 23,900 个 decision；Bongard paper 比较 61 个 system | general bounded-decision accuracy、probability quality、latency | [Bongard weight](https://huggingface.co/AgentBull/bongard-mini)；paper 引用 leaderboard artifact | 分离 architecture、data 与 serving effect |
| [JevAdvBench](https://jevadvbench.github.io/JevAdvBench/) | 812 个 question、66 个 scenario、9,744 个 single-edit variant | 对 rewording、unverified opinion、context injection 与 command 的 sensitivity | request、output、control 与 analysis | attacked-task gold accuracy：多数 target 是 model 自己的 clean decision |
| [RLCDAlignBench](https://github.com/sumleo/RLCDAlignBench) | 10 类 failure、44 个 benchmark、5 个 target model | 对 alignment failure 做廉价 zero-shot screening | code、data 与 cached result | 大多数 setting 下独立 human ground truth |
| [Probability Contracts](https://arxiv.org/html/2609.37470) | 1,000 个 exact finite world、4 种 model–interface configuration | posterior accuracy、coherence 与 downstream decision loss | paper 的 exact construction 与 accounting | natural-language 和 production-distribution validity |
| [Automated Decision Gates](https://arxiv.org/abs/2610.00346) | 6 个 family 的 8 个 checkpoint，加 LLM 与 trained/zero-shot classifier | matched accuracy、option robustness、OOS acceptance、calibration transfer、cascade cost | paper 描述的 frozen request、hash、answer 与 analysis | production label 与 unconstrained reasoning comparator |
| [SemBench / Shelob](https://arxiv.org/html/2610.02046) | 21 个 semantic query；join 多达 540K candidate pair | filter、join、rank、pruning 与 end-to-end database utility | 论文与 benchmark specification | decision model 本身相对于 pruning、caching 与 cascade 的独立贡献 |

如果由我设计实验，我会使用**三套 suite，而不是一套**：一套 broad public regression suite；一套冻结 threshold 的 private
time-split workload；再加一套 controlled stress suite，专门测试 option order、missing answer、irrelevant context、hierarchy
与 knowledge boundary。之后还要公开完整 policy——model version、schema、calibration set、threshold、fallback、latency
distribution 与 review budget——因为脱离 controller 的 probability，并不是 deployment result。

---

## 护城河可能在哪里？ {#where-might-the-moat-be}

当一个新系统走红时，人们自然会问：护城河究竟是 **data、model、evaluation，还是 idea**？对 Jev 来说，
“calibration”是最显眼的答案，但新证据要求我们作出更严格的区分：

> **Calibration 是 Jev 最重要的产品承诺，但现有证据并不支持把它当作稳定的 architecture moat。** 它仍然取决于
> workload、interface、threshold 与 distribution。更站得住脚的护城河假说，是那套能够生产、验证、serving、monitor，
> 并持续重新校准这些 probability 的 operating stack。

Idea 本身并不是耐久护城河。Probabilistic classifier、proper scoring rule、reject option、selective classification、
constrained decoding、label-logit readout、semantic router 与 confidence cascade 都早于 Jev。只要组织拥有 target domain 的
labeled data，往往就可以用 temperature scaling、Platt scaling、isotonic regression 或 learned calibrator 改善 calibration，
而不必重新训练 base model
（[Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)）。

Calibration 本身是一份强大的**契约**，却不是静态 intellectual property。它必须针对每个 workload 和 version 重新建立或
验证。如果另一个 model 也能提供同样有用的 risk ranking，而客户可以低成本地校准它，数值优势就可能缩小。

开放生态把这一点变得很具体：frozen 4B causal LM 已经能通过 numeric-suffix 或 option-logit readout 给出很有竞争力的
probability；当 target label 充足时，普通 trained classifier 也可以胜出；在窄 domain 上，open encoder 还可以小得多。
因此，最强的 proprietary claim 不能只是“我们输出 categorical distribution”，而必须是在新 task、distribution shift 与
operating cost 上，持续达到 matched public alternative 无法达到的 frontier。

更可能形成护城河的是一个 vertically integrated stack：

1. **Data。** 一个覆盖广泛、精心构建的 decision task、rubric、hard negative、distribution shift 与 confidence label
   mixture，可能带来难以复现的 transfer。TypeSafe 没有公开这些 data。
2. **Training。** Proprietary architecture、parallel sampler 与 RLCD recipe 可能共同改善 accuracy、calibration 与 latency
   的 Pareto frontier；每一项的独立贡献仍然未知。
3. **Serving。** 在多个 question 之间高效共享 state computation，并以高 throughput serving，可能和 model objective
   一样重要。
4. **Interface。** Choice、Score 与 Noul 为开发者提供了一套很小的 vocabulary，促使系统采用更好的 design。Product
   simplicity 未必有科学新颖性，却可以形成商业防御力。
5. **Evaluation flywheel。** 每次 deployment 都会暴露新的 slice、rubric ambiguity、drift 与 fallback outcome。如果被
   负责任地使用，这些 feedback 可以改善 model 与 threshold tooling。
6. **Distribution 与 ecosystem。** Adapter、observability、example 和 agent runtime integration 会提高 switching cost。

![Jev 可能的护城河是一套 stack，而不是单个算法](/assets/img/blog/jev-calibration/fig8_moat_stack.svg)
*图 8. Calibration 是对外 contract，serving economics 是切入口，proprietary data 与 training 是潜在技术护城河，
API/evaluation ecosystem 则可能成为 adoption moat。这个拆解是作者推断；TypeSafe 没有公开 causal ablation。*

这里的“可能”非常重要。如果没有 matched model 每次只改变 architecture、scale、data、RLCD 或 serving 中的一项，外部
观察者就无法合理分配 credit。目前最诚实的答案是：

> Jev 的核心 insight 并非不可复制；它的一体化 implementation 可能是。

---

## 应用：Jev 在 agent stack 中的位置 {#applications-where-jev-fits-in-an-agent-stack}

最适合 Jev 的 use case 共有四个特征：output space 有限、decision 高频发生、latency 或 cost 很重要，而且 uncertainty 可以
触发一个有意义的 fallback。

现有 application literature 已经足以区分*理论上合理的 use case*与*实际测量过的 system*。下表只列后者。大多数收益属于
完整 pipeline，而不是某一次 model call，因此正确的 evaluation unit 应该是 end-to-end policy。

| 应用 | Decision role | 规模 | 报告的 operating point | 主要 caveat | 来源 |
|---|---|---:|---|---|---|
| Semantic database | filter、join、classify、rank，并决定何时 escalate | 21 个 SemBench query；join 多达 540K pair | 87.4% pair pruning；reasoning escalation 减少 55.2%；mean F1 95.7–97.5% | 把 model decision 与 relational pruning、caching、cascade 组合在一起 | [JEVDB](https://arxiv.org/abs/2610.02046) |
| Long-term agent memory | 在 LLM 规划 search、撰写答案时，对 raw retrieved record 做 judgment | LoCoMo、LongMemEval-S；100K–10M-token history | LoCoMo 91.7–92.2%；跨该 history range，每题 cost 只增长 1.11 倍 | 分数取决于 answer model 与整套 memory pipeline | [Mnemon](https://arxiv.org/abs/2609.36059) |
| Memory write policy | 选择 raw turn，而不是 LLM-extracted memory | 预注册的 LoCoMo 与 LongMemEval study | 在紧 budget 下 non-inferior；报告 write cost 降低 3,061 倍 | context budget 宽松时，extraction 仍更强 | [Engram study](https://arxiv.org/abs/2609.34227) |
| Water-network triage | 四分类 incident attribution、rule confirmation，可选 LLM review | 4 轮 sealed preregistered evaluation；transfer 到 2 个 network | macro-F1 0.62–0.64；LLM review 减少 35–38% | prior 与 rule 也有贡献；公开 simulated network 并不等于 live utility traffic | [HydroJEV](https://arxiv.org/abs/2610.02048) 与 [artifact](https://github.com/mutianwei521/hydrojev) |
| Mobile GUI execution | 在偶尔调用 VLM plan 的前提下，反复选择 low-level action | 完整 AndroidWorld suite | success 79%，step-wise VLM 为 84%；成功 run 快 32.7%、便宜 73.4% | delegation policy 与更少 VLM call 和 Jev 作用相互混杂 | [Jev-Mobile](https://arxiv.org/abs/2609.30186) |
| Alignment screening | 对一个 model response 并行执行 10 个 typed check | 44 个 benchmark、10 类 failure、5 个 target model | 通用 question 的 median AUROC 0.886；报告 cost 比 LLM judge 低 63 倍 | 大多数 label 来自 benchmark scorer；部分 context field 会泄露 label | [Just Ask Jev](https://arxiv.org/abs/2609.29429) |
| Population-scale narrative coding | 把 free text 映射成 27 个 probabilistic variable，并分配 human review | screening 499,500 条 narrative；2,416 条 blind human label | 对 human label 的 F1 0.908；recalibration 使 calibration error 降低 3.3 倍 | specialized screening 与 sampling design | [Crash narratives](https://arxiv.org/abs/2609.24052) |
| Multimodal medical decision | classification、finding detection、bounded regression 与 counting | 15 个 dataset；23,558 个 state；726 个 held-out state | point accuracy 相似，但报告的 calibration error 低很多；generative arm 在 counting 上获胜 | decision arm 获得更密集的 factorized supervision | [OmniMed-Jev](https://arxiv.org/abs/2610.00381) |
| Agent security gate | 把 injection、harmful request 或 trace 映射为 allow/block/review | 多个 security benchmark 与 judge family | subgroup error 与 shared confident miss 严重限制 safe automation | benchmark threat model 不能替代 live red-team program | [Security evaluation](https://arxiv.org/abs/2609.33401) |
| Speech-neuroprosthesis rescoring | 从候选 decoded sentence 中选择 | 来自一位 participant 的 978 个 held-out sentence | WER 7.5%；重新调 fusion 后为 6.9% | 只有一位 participant；network latency 并不快于 local 7B inference | [Rescoring study](https://arxiv.org/abs/2609.33538) |

反复出现的共同模式是：**state-complete evidence + bounded alternative + 高频调用 + 有意义的 fallback + 可测量的 downstream
utility**。缺少其中任一项时，单次 decision 的低价格通常都不是主要瓶颈。

### Routing 与 tool selection {#routing-and-tool-selection}

Agent 可以在 search、database、code execution、calculator、memory 和询问用户之间做选择。Option set 必须对应真实
capability，并包含一个安全 fallback。Probability distribution 可以决定只运行一个 tool、并行运行多个 tool，还是 defer
给 planner。

### Retrieval、reranking 与 memory {#retrieval-reranking-and-memory}

对于每一份 candidate document 或 memory，Noul 可以估计 relevance，Score 可以估计 support quality，Choice 可以分配
semantic category。Two-stage system 可以先低成本过滤数千个 candidate，只让 borderline case 进入 generative model。
由于 false negative 可能永久移除关键 evidence，这里通常应该优先考虑 recall-sensitive threshold，而不是 top-1 accuracy。

### Verification 与 guardrail {#verification-and-guardrails}

Decision model 可以检查 citation 是否支持 claim、准备执行的 command 是否符合 user authorization、tool result 是否内部一致，
或 output 是否违反 policy。它应该是 defense in depth 的一层，而不是 irreversible action 的唯一 authority。Independent
model 与 deterministic check 很有价值，因为单一 shared model 可能产生 correlated mistake。

第一项覆盖面较广的 alignment-detector study 为这个 use case 提供了一些 quantitative support。在 44 个 benchmark 和
10 类 alignment failure 上，一个通用 Jev question 达到 0.886 的 median AUROC；论文报告，得到的 judge 比 LLM-based
scorer 便宜 63 倍。同一研究也发现，*Jev 能看到哪些 context field*，比 wording 的小幅变化更重要；而某些有帮助的 field
事实上编码了 benchmark label。大多数 label 来自现有 benchmark scorer，只有两个 setting 包含 human label。因此，这个
结果是 cheap zero-shot screening 的有希望证据，却不是 production false-negative guarantee，更不是把 Jev 当作独立
ground-truth oracle 的许可
（[Guo et al., 2026](https://arxiv.org/abs/2609.29429)）。

### Data labeling 与 semantic map/reduce {#data-labeling-and-semantic-mapreduce}

许多 labeling task 都是伪装成 prompt 的 finite decision：topic、sentiment、policy category、error class、quality tier 或
duplicate status。当需要数百万个短 decision 时，Jev 的 cost 与 parallelism 很有吸引力。Calibration 还能支持 active
learning：把 uncertain example 发给人类，再用新 label 改善 task-specific system。

### 已知 candidate 上的 recommendation {#recommendation-among-known-candidates}

当 candidate 已经由 retrieval 或 business logic 生成，Choice 可以从中选择或排序，而不必生成新内容。这适用于
next-best action、UI adaptation、workflow assignment 与 personalized recommendation。如果真正的难点在于发现一个全新
candidate，它就没有那么合适。

### 为 reasoning agent 分配 compute {#compute-allocation-for-reasoning-agents}

一个 lightweight decision 可以判断任务应该直接回答、进行 short reasoning 或 long reasoning、增加 rollout、调用 search，
还是寻求人类帮助。这让 uncertainty 变成 budget controller。为了避免 circular failure，必须评估昂贵 compute 是否真的修复了
被 controller 拒绝的 case。

### Confidence cascade {#a-confidence-cascade}

实用 architecture 并不是“Jev 对抗 LLM”，而是一套 cascade：

1. deterministic rule 拒绝不可能或未授权的 action；
2. Jev 处理 high-confidence bounded decision；
3. uncertain case 进入 reasoning LLM 或 verifier；
4. 具有严重后果的 disagreement 交给人类；
5. outcome 回流到 calibration monitoring 与 threshold update。

![展示 Jev、LLM 或 hybrid system 各自适用位置的 application matrix](/assets/img/blog/jev-calibration/fig9_application_matrix.svg)
*图 9. Jev 适合高频 bounded judgment；LLM 适合 open-ended generation 与 planning；许多 production workflow 最适合
hybrid cascade。只有当 fallback 的 error 互补、threshold 经过验证、且高影响 action 保留独立 control 时，hybrid 才会成功。*

Deferral 有两个乘数：**confidence 能否识别可修复的 case，以及 fallback 是否会犯不同的错？** 其中任何一项接近零，
escalation 都只会增加成本。一项 rubric-judge study 报告，在其 setup 中，大约 96% 的 confident Jev error 会被 LLM judge
重复；agent-security study 同样发现 shared high-confidence miss。相反，automated-gates study 发现，classifier-first 的廉价
cascade 能以约 43% 的估计成本匹配 Jev。“低于 0.8 就调用更强 model”并不是完整 cascade design；设计者必须测量 joint
error、给定 deferral 后的 repair rate，以及 realized cost
（[Rao and Callison-Burch, 2026](https://arxiv.org/abs/2609.29769)；
[Rafe and Das, 2026](https://arxiv.org/abs/2610.00346)）。

例如，设想一个 coding agent 准备执行 500 个 file operation。大部分只是 local read 或 reversible edit。Decision model 可以
在数百毫秒内分类其 authorization 与 risk。High-confidence low-risk action 在 sandbox 中继续；medium-confidence case 由带有
repository context 的 LLM review；network access、secret handling、permission change、publication 或 destructive operation
则无论 model confidence 多高，都必须经过 hard policy 或 human approval。Calibrated model 提高的是 throughput，而不是重新
定义 authority。

TypeSafe 自己的 use-case map 包括 routing、classification、scoring、filtering、recommendation、safety check 与
extraction-like pattern。这些是有用的起点，但每位客户仍然需要 private evaluation set 与 deployment-specific cost
（[TypeSafe AI, 2026](https://docs.typesafe.ai/concepts/use-case-map)；
[patterns](https://docs.typesafe.ai/patterns)）。

---

## 硬边界：哪些问题仍然应该交给 LLM？ {#the-hard-limits-where-an-llm-remains-the-right-abstraction}

Decision model 有意识地用 generality 换取更清晰的 contract。这个 trade-off 带来了真实边界。

逐渐清晰的边界并不只是**小 model 与大 model**，而是**评价已经给出的 evidence，与模拟尚未出现的 consequence**。当决定性
事实已经写进 state 时，Jev 可以很强；但如果一次调用必须推断 unseen opponent action、prerequisite subgoal、physical
transition 或 intermediate computation，它就可能失败。此时应由 planner、search procedure、environment copy、calculator
或 simulator 生成这些 consequence，再由 decision model 评价。这既是经验结果，也是一条良好的 software boundary
（[Yang et al., 2026](https://arxiv.org/html/2610.01834)）。

**它无法创造缺失选项。** 如果正确答案不在 schema 中，closed choice distribution 仍然必须把 probability 分配给某个已有
选项。Open-world task 必须包含 `other`、`unknown`，或使用 hierarchical candidate-generation stage。即使有这个 label，
也仍然需要 training 与 policy。一项 arithmetic study 中，答案存在时 Jev 的 accuracy 为 99%，答案缺失时却只有 7% 能正确
reject；在独立 development set 上冻结 threshold 后，reject 率升至 79%，同时保留 97% 的 answer-present accuracy
（[Zhong et al., 2026](https://arxiv.org/abs/2609.39496)）。

**它无法解释或综合。** 开发者有时不仅需要“reject”，还需要 grounded rationale、经过修改的 artifact 或 plan。当 language
本身就是 deliverable 时，general LLM 仍然更合适。

**它不会规划。** 选择一个 next action，不等于构建并持续修订 long-horizon plan。Harness 仍然必须保存 objective、管理
memory、从 error 中恢复，并执行 stop condition。

**它继承 state 的质量。** Prompt injection、stale memory、contradictory evidence、缺失 provenance 与 malformed rubric
都可能导致 bad decision。Typed output 并不会自动清洗 input。

这并不只是理论担忧。JevAdvBench 在 812 个 typed question 上评估了 9,744 个 single-edit variant。它用 Jev 自己的 clean
decision——而不是外部 correctness label——给每个 attacked answer 打分，因此 flip rate 衡量的是 behavioral robustness，
不是 attacked accuracy。对 `jev-1.13.0`，只是在 state 末尾附加一条未经验证的 opinion，就让 12.1% 的 decision 发生 flip，
并使 38% 原本 confident 的 answer 跌破 0.8 的 review threshold。这些具体比例只属于该 benchmark，不能代表所有
deployment；但背后的 mechanism 是普遍的：state 是一份经过组织、甚至可能带有 adversarial intent 的输入，而不是天然
可信的 ground truth
（[Hu et al., 2026](https://arxiv.org/abs/2609.31142)）。

**Hosted product 当前以文本为主，每一种新 modality 都需要单独证据。** 开放项目已经展示 visual shared-prefix readout 与
multimodal medical decision model，但它们是独立 system——不能被当作 hosted Jev 的证据。Belebele 上的强结果值得鼓励，
却不能证明 multilingual reliability 均匀一致；Chinese-Jev 的 specialization 结果同样需要独立复现。

**公开记录的 jaggedness 很重要。** TypeSafe 指出，Jev 在 counting 与 arithmetic、date comparison、multi-hop 或 indirect
formulation、冗长的 irrelevant context、contradiction 和 option ordering 等方面存在弱点
（[TypeSafe AI, 2026](https://docs.typesafe.ai/model-jaggedness/jev-1.13)）。这些并不是 agent 内部罕见的 edge
case；tool trace 与 policy state 本来就经常很长、很嘈杂，并且需要 compositional reasoning。

**Calibration 不是 local truth。** 一个 model 可以在 global 上校准良好，却在某个 rare subgroup 上持续且自信地失败；它也
可以 accuracy 很高，但在相关 question 之间缺乏 coherence。敏感 deployment 需要 stratified calibration、stress test，
以及对 probability estimate 本身 uncertainty 的明确表达。

**Closed model 限制 auditability。** 用户无法检查 training corpus、排除 benchmark contamination、复现 training，或独立
判断哪个 component 产生了某项 behavior。Hosted privacy、retention、versioning 与 rollback 都成为 risk model 的一部分。
TypeSafe 表示 customer request 与 response 不会被用于 training，并记录了 enterprise zero-data-retention option；但客户仍然
必须确认自己 account 所适用的 retention tier、regional processing 与具体条款
（[TypeSafe AI, 2026](https://docs.typesafe.ai/models)）。

**Jev 不支持 per-customer fine-tuning。** TypeSafe 表示，Jev 不会使用 customer data 做 fine-tuning 或 LoRA adaptation，
而且每个 account 使用同一套 weights。通过 state、question、option 与 rubric 做 prompting 可以调整 interface，但某些 domain
需要学习本地 representation。External calibration 可以修正 probability，却无法修复糟糕的 ranking 或缺失的 domain
capability
（[TypeSafe AI, 2026](https://docs.typesafe.ai/models)）。

**High confidence 会放大 automation bias。** 一个精确数字看起来非常权威。即使 0.97 已经 shift、未经充分验证，或来自错误
statistic，operator 也可能比面对一句模糊陈述时更愿意相信它。好的 UI 应该展示 evidence class、threshold policy 与历史
performance，而不只展示 probability。

这些限制暗示了如下分工：

| 在以下情况使用 decision model… | 在以下情况使用 generative LLM… |
|---|---|
| valid answer space 已知 | 发现 answer space 本身就是任务的一部分 |
| latency 与 volume 占主导 | deep synthesis 或 explanation 占主导 |
| probability 驱动一项 policy | language 或 code 就是产品 |
| 存在安全 fallback | 任务无法拆成 bounded judgment |
| local semantic judgment 已经足够 | multi-step world modeling 与 planning 是核心 |

两类 system 是互补关系。能力强的 LLM 可以生成 candidate 与 explanation；Jev 可以负责 routing、checking、scoring，并决定
何时应该调用昂贵 model。

---

## 我会如何评估一个 decision model {#how-i-would-evaluate-a-decision-model}

只围绕 accuracy 设计的 evaluation，会错过使用 Jev 的真正理由。严肃的 evaluation 至少应有七个层次。

<h3 id="1-task-quality">1. Task quality</h3>

根据任务报告 accuracy、macro/micro F1、AUROC 或 AUPRC，以及 confusion matrix。不只报告 pooled average，还要报告
per-class 与 per-slice result。如果 production 中存在 `none_of_the_above` 与 ambiguous case，evaluation 也必须包含它们。

<h3 id="2-probability-quality">2. Probability quality</h3>

报告 Brier score、negative log-likelihood、reliability diagram、adaptive 和 classwise calibration error，以及 bootstrap
interval。测试 equivalent phrasing、option permutation、negation 与 logically related question。不要悄悄用 TypeSafe 的派生
`confidence` 代替 raw probability。

<h3 id="3-selective-performance">3. Selective performance</h3>

绘制 risk 对 coverage 的曲线，报告 risk–coverage curve 下的面积，并给出多个 operational risk limit 下的 coverage。
只允许在 validation data 上确定 threshold，然后把它冻结，在未来时间段或 geographically separated test data 上只评一次。
在同一个 dataset 上调整并报告 threshold，并不是 deployment evidence。

<h3 id="4-shift-and-stress">4. Shift 与 stress</h3>

测试 time shift、base-rate shift、domain shift、long/irrelevant context、multilingual input、contradictory state、missing
evidence、prompt injection、option-order change 与 adversarially plausible distractor。记录当 calibration intercept 改变时，
ranking 是否仍然保留：有时 ranking 可以被 recalibrate，而一旦 ordering collapse，单纯重新校准就无法修复。

<h3 id="5-cascade-value">5. Cascade value</h3>

评估完整 policy，而不只评估第一层 model。测量 fallback rate、fallback accuracy、error overlap、end-to-end latency、human
workload，以及每一个 correct automated decision 的成本。最关键的 counterfactual 是：Jev 拒绝的 case 是否真的在下游得到
修复。

<h3 id="6-agent-level-consequences">6. Agent-level consequence</h3>

对 long-running agent，在加入或移除 decision layer 的情况下 replay 完整 trajectory。测量 task success、unsafe action rate、
unnecessary escalation、error recovery、token 与 wall-clock cost，以及 tail latency。小的 per-step error 可能彼此相关并持续
累积；反过来，一个并不完美的 decision model，也可能因为捕捉到少数 high-leverage mistake 而改善 total performance。

<h3 id="7-operations">7. Operations</h3>

固定 model version，并记录 state hash、question template、option order、returned distribution、policy decision、fallback
outcome 与 eventual label。检测 drift、定义 rollback trigger，并且只在 leakage-safe data 上 recalibrate。如果 provider 修改了
`jev-latest`，system 不应该在无声无息中继承一条新的 operating curve。

一张精简 evaluation card 可以写成：

```text
Model/version: jev-1.13.0
Task and population: production support-routing, English, Sep 2026
Primary metric: coverage at <=2% routing error
Threshold: fitted on Aug holdout, frozen before Sep test
Slices: customer tier, issue class, input length, new products
Fallback: reasoning LLM + human for consequential cases
Shift tests: option permutation, paraphrase, base-rate, injection
Ops: version pin, daily drift alert, weekly delayed-label audit
```

它不如 benchmark leaderboard 华丽，却回答了真正重要的问题：*这项 probability 能否安全地承载我软件中的一个 branch？*

---

## 从 calibrated decision 到 calibrated agent {#from-calibrated-decisions-to-calibrated-agents}

Jev 最深层的研究价值，也许是它迫使我们把 **estimation** 与 **control** 分开。Generative agent 往往把两者纠缠在同一串
token 中：它 reasoning、陈述 confidence、选择 next action，再解释原因。Typed decision endpoint 让边界显现出来：model
估计一个 distribution，policy 再依据成本与 constraint 使用它。

但 agent 并不是一次 decision，而是一条 sequence：

$$
\tau=(s_0,a_0,o_1,s_1,a_1,o_2,\ldots,s_T),
$$

其中 action 会改变后续 observation 与 future state。完整 trajectory 的 success probability 一般并不等于各个 local
confidence 相互独立时的乘积。Error 彼此相关；agent 可能从一次 mistake 中恢复；看似安全的 action 可能关闭未来 option；
uncertainty 也可能来自 epistemic knowledge、environment，或不充分定义的 objective。

因此，robust agent 需要多个层级的 calibration：

- **atomic：**这次 classification、tool call 或 claim 是否正确？
- **state：**Agent 是否已经拥有足够信息来做决定？
- **transition：**该 action 是否会让 system 朝 objective 前进，同时不违反 constraint？
- **plan：**当前 strategy 是否可能成功？
- **trajectory：**从现在开始，最终成功的 probability 是多少？
- **institutional：**Model、harness、tool、human 与其他 agent 的组合，是否仍然处于授权边界内？

这会形成一个 active loop：

1. 估计 uncertainty；
2. 识别 uncertainty 的来源；
3. 选择 intervention——retrieve、reason、simulate、verify、ask 或 stop；
4. 观察 intervention 是否消除了 uncertainty；
5. 更新 memory 与未来 calibration；
6. 无论 confidence 多高，都保留 hard authorization boundary。

![结合 deterministic policy、Jev、generative reasoning、trajectory calibration 与 human control 的 hybrid future](/assets/img/blog/jev-calibration/fig10_hybrid_future.svg)
*图 10. 在未来的 heterogeneous stack 中，Jev 位于快速 atomic-decision layer；ACC 估计 trajectory success；AUQ 让
uncertainty 改变 memory 与 reflection；generative model 负责 planning 与 repair；harness 把 estimate 转成 bounded action。
尚未解决的问题，是贯穿完整 loop 的 compositional calibration。*

最有希望的 architecture 也许是 heterogeneous 的。Deterministic code 处理 invariant；decision model 处理高频 bounded
semantic judgment；generative reasoner 处理 open-ended planning 与 repair；trajectory calibrator 估计 cumulative risk；
harness 选择 intervention；human 则保留对 irreversible high-impact action 的 authority。没有任何一个 model 必须在所有方面
都是最好。

这也解释了为什么更快的 decision 除了 latency 之外还有更深价值。如果一次 check 便宜到可以在每一步运行，system 就能为
过去不可见的 behavior 增加 instrumentation：询问某条 memory 是否仍然 relevant、某个 claim 是否有 evidence、objective 是否
已经 drift，以及再运行一个 rollout 是否具有 expected value。收益不仅来自替换一次 LLM call，也来自启用过去因成本而被
省略的 check。

危险则是 over-instrumentation。数百个各自带噪声的 gate 可能产生脆弱 workflow、correlated false alarm 与隐藏 failure
mode。每个 monitor 都会改变 system behavior，每个 threshold 都会制造一个新的 optimization target。Agent training 最终也
可能学会绕过可预测 gate。因此，decision model 应该作为 adaptive control system 的 component 来评估，而不是作为彼此独立
的 oracle 被随意撒进 harness。

---

## 开放研究问题 {#open-research-questions}

<h3 id="1-what-is-inside-jev">1. Jev 的内部究竟是什么？</h3>

一份真正的 model card 应该披露 parameter scale、architecture class、training stage、data governance、language、RLCD
objective、contamination control、known limitation 与 causal ablation。商业保密可以理解，但如果缺少这些基本信息，就很难
评估关于一个新 model category 的科学 claim。

<h3 id="2-which-component-creates-the-frontier">2. 哪一个 component 创造了 frontier？</h3>

观察到的优势有多少来自 model specialization、data、RLCD、parallel state processing、hardware、batching 或 API design？
一个 matched ablation 应该在固定 architecture 与 compute 下，对比 cross-entropy、Brier supervision、post-hoc calibration、
proper-score RL 与 proprietary recipe。

<h3 id="3-does-calibration-transfer">3. Calibration 能否 transfer？</h3>

我们需要针对 base rate、customer population、language 和 model version 变化做 longitudinal evaluation。一个小型 labeled
calibration set 能否在不 retrain 的情况下修复 probability？什么时候 recalibration 会失败，因为 representation 已经无法正确
rank example？

<h3 id="4-can-marginal-probabilities-become-coherent-beliefs">4. Marginal probability 能否变成 coherent belief？</h3>

Independent question 可能违反 exclusivity、implication 与 negation。Projection layer 能否在不损害 empirical calibration 的
情况下强制 logical constraint？相关 question 是否应该 jointly evaluate？System 应该如何表达 genuine ambiguity，而不是强行
制造表面 consistency？

<h3 id="5-how-should-out-of-set-cases-work">5. Out-of-set case 应该如何处理？</h3>

如果 model 无法识别 novelty，单独一个 `other` 并不够。我们需要 open-set detection、abstention、hierarchical candidate
generation，以及“true answer 不在候选集合中”的 calibrated probability。这对 tool 尤其重要：选择一个最不错误的 tool，
可能比完全不行动更危险。

<h3 id="6-can-guarantees-sit-above-learned-probabilities">6. 能否在 learned probability 之上建立 guarantee？</h3>

Conformal prediction、risk-controlling prediction 与 sequential testing，可能在明确 assumption 下，把 held-out data 转化成
有限样本的 error 或 coverage bound。它们应该如何与快速迭代版本的 hosted model，以及 non-exchangeable agent traffic 协同？

<h3 id="7-how-do-probabilities-compose-over-time">7. Probability 如何跨时间组合？</h3>

Local calibration 并不蕴含 trajectory calibration。我们需要为 correlated error、recovery、option value 与 state-dependent
hazard 建模。Atomic Jev judgment 能否输入 ACC 这样的 trajectory model，又不 double count evidence？AUQ 能否学习哪一种
intervention 具有最高 expected value of information？

<h3 id="8-when-is-reasoning-worth-its-cost">8. Reasoning 何时值得它的成本？</h3>

有些任务完全不需要 chain of thought，另一些任务没有 multi-step inference 就会失败。一个 calibrated router 不只要估计 cheap
answer 是否正确，还要估计增加 search、reasoning 或 sampling 是否可能以有益方式改变答案。这是关于 compute 的 causal
question，而不是普通 confidence。

<h3 id="9-what-happens-under-strategic-pressure">9. 面对战略性压力会发生什么？</h3>

如果 decision model 变成 monitor 或 gate，上游 agent 可能学会构造 input 来 exploit 它。Evaluation 应该包括 adaptive attack、
prompt injection、state obfuscation、option manipulation，以及由 agent 自己诱发的 distribution shift。在 passive data 上训练的
monitor，一旦成为另一个 optimizer 的 target，就可能失败。

<h3 id="10-how-do-we-prevent-self-confirming-calibration-loops">10. 如何防止 self-confirming calibration loop？</h3>

Selective system 主要只能为 accepted 或 escalated case 观察到 label。它的 action 会改变之后用于 recalibration 的 data，造成
bandit feedback、censoring 与 selection bias。即使看起来低效，random audit 与 exploration 也可能不可或缺。

<h3 id="11-how-should-uncertainty-be-shown-to-humans">11. 应该怎样向人类展示 uncertainty？</h3>

用户往往误解 probability，也容易过度相信精确数字。Interface 应该传达 reference class、calibration history、uncertainty
interval、shift warning 与采取 action 的后果。我们需要研究 probability display 是真的改善了 decision，还是只让人类更
自动地服从 model。

<h3 id="12-can-the-interface-expand-without-losing-its-advantage">12. Interface 能否扩展而不丢失优势？</h3>

真实 system 需要 vision、audio、continuous value、set、ranking、structured object 与 combinatorial action space。扩展 type
system 可能需要新的 head、loss 与 serving path，并可能削弱一部分 simplicity 或 latency advantage；但它不必然需要重新引入
autoregressive generation。Decision model 与 general structured generator 的边界本身就是一个研究问题。

<h3 id="13-what-should-the-benchmark-optimize">13. Benchmark 应该优化什么？</h3>

静态公开 classification accuracy 很容易被污染，而且距离 deployment 很远。更强的 benchmark 应该使用 private、temporally
held-out task；报告 risk–coverage 与 cost；在 evaluation 前冻结 threshold；测试 prompt 和 option perturbation；加入
fallback；并评估完整 decision policy，而不是孤立 model。

<h3 id="14-can-we-make-the-system-auditable-and-private">14. 如何让 system 可审计且保护隐私？</h3>

Hosted inference 带来 data retention、version drift、regional processing 与 reproducibility 问题。Open weight、on-premises
deployment、signed version manifest 与 standardized decision log，可以扩大 calibrated decision model 值得被信任的领域。

> **证据纪律。** 对 2026 年相关文献的正确理解，既不是“Jev 已经解决了 calibrated decision”，也不是“Jev 只是一个
> classifier”。它让一个长期被忽视的 system interface——带有 operational probability 的 bounded decision——变得易于购买、
> 组合和部署。这个 interface 是否会成为持久的 model class，仍取决于它能否经受 matched readout baseline、distribution shift、
> open-set rejection、probability-coherence test 与真实 downstream cost 的共同检验。

---

## 结论 {#conclusion}

Jev 不是缩小版 chatbot，也不是一个 acronym。它是一项 typed decision service：文本 state 与 bounded question 进入，有限
probability distribution 输出。听起来很克制，但在 agent system 中可能影响深远，因为 control loop 的很大一部分并不是
写作，而是做 decision——route、verify、retrieve、accept、retry、ask 或 stop。

它的热度来自几个同时发生的因素：“decisions, not strings” 这个容易记住的 framing、低 latency serving、当前极低价格、
干净的 API，以及一个迫切需要 machine-usable uncertainty 的 agent ecosystem。早期证据支持真实的速度与成本优势，也表明
它在许多 bounded task 上具备有希望的 accuracy 与 selective prediction；同样的证据也显示 calibration 并不均匀、存在
domain weakness 与 coherence failure，而且不能把刚出现的 preprint 当作已成定论的事实。

技术中心是 calibration，但 V2 最强的结论是：calibration 只是第一项测试。Probability 不等于 TypeSafe 派生的 confidence
statistic；frequency calibration 不等于 self-knowledge；self-knowledge 不等于 coherence；coherence 也不等于 decision
utility。较低 pooled ECE 可能掩盖 subgroup failure；一个看似 confident 的 model 可能在越过 knowledge boundary 时毫无
察觉；等价 interface 还可能把同一个 event 推到不同 action threshold 两侧。真正的 operational question 是：面对未来
target traffic，一套冻结的 policy——而不只是一个 score——能否给出可接受的 risk、coverage 与 cost。

RLCD 是 TypeSafe 叙事的一部分，但 Jev 的实际实现仍然没有公开。独立 OpenJev-RLCD 论文给出了一种合理的 proper-score
training method，也分析了 rationale variance；它不是 proprietary recipe。RLCR、CaOPD、ACC 与 AUQ 则从互补角度照亮了
从 correctness 到 calibrated self-assessment、deployment-aware distillation，再到 trajectory-level intervention 的道路。

那么护城河在哪里？大概率不在“calibrated classification”这个抽象 idea 本身，也大概率不在 typed head 本身。Frozen LM
readout、open pointer head、specialized encoder 与 trained classifier 已经覆盖了大量可观察 behavior。更强的假设是一套
stack：proprietary data 与 training、parallel low-latency serving、typed API、evaluation 与 monitoring，以及 integration。
Calibration 是产品承诺，serving economics 是切入口，model/data recipe 是潜在技术护城河，harness 与 recalibration loop
则可能成为 adoption moat。

正确的 deployment 同样不是“Jev 取代 LLM”，而是 heterogeneous cascade。Deterministic code 执行 invariant 并模拟 state
transition；Jev 对已经给出的 consequence 做快速 bounded judgment；generative LLM 负责 planning、explanation 与 repair；
trajectory calibrator 追踪 long-horizon risk；harness 决定 authority、fallback 与 stopping；human 则留在 irreversible
decision 的 loop 中。只有当 confidence 能识别可修复 case、且 fallback 会犯互补 error 时，escalation 才值得它的成本。

这最终指向一个我认为最有吸引力的 research frontier：calibration 不应该停留在测量 confidence；它应该决定 system 下一步
做什么——act、verify、branch、reflect、retrieve、投入更多 compute，还是寻求帮助。Jev 让这幅愿景在单个 decision 层面变得
异常具体。尚未解决的问题，是如何让它贯穿完整 agent trajectory。

> **Calibration 是一份契约，而不是一张证书。** 只有当 probabilities 在 deployment shift 下得到验证，并连接到一套仍然
> 可以说出 “I do not know” 的 policy 时，它的价值才真正实现。

---

*来源说明：本文区分了 TypeSafe 的公开文档和 vendor evaluation、独立 preprint，以及我自己的分析。截至 2026 年 10 月
4 日，Jev 的 architecture、parameter count、training data 与 RLCD recipe 均未公开。OpenJev-RLCD 是独立提案，并不是经
TypeSafe 确认的 reconstruction。全部十张图均为原创 schematic；它们描述的是 interface 与 research concept，而不是未公开
的实现细节。*

---

## 如何引用 {#how-to-cite}

> Zhang, Jiaxin. (Oct 2026). Jev and the Return of the Decision Model: When Calibration Becomes an API. *Jiaxin
> Zhang's Blog.* https://jxzhangjhu.github.io/blog/2026/jev-calibration-as-an-api/

```bibtex
@article{zhang2026jevcalibration,
  title   = "Jev and the Return of the Decision Model: When Calibration Becomes an API",
  author  = "Zhang, Jiaxin",
  journal = "Jiaxin Zhang's Blog",
  year    = "2026",
  month   = "Oct",
  url     = "https://jxzhangjhu.github.io/blog/2026/jev-calibration-as-an-api/"
}
```

---

## 参考文献 {#references}

[1] TypeSafe AI. ["Introducing System One Models & Jev."](https://typesafe.ai/blog/introducing-system-one-models-and-jev) September 15, 2026.

[2] TypeSafe AI. ["Introduction to Jev."](https://docs.typesafe.ai/introduction) Documentation, accessed October 3, 2026.

[3] TypeSafe AI. ["System One."](https://docs.typesafe.ai/concepts/system-one) Documentation, accessed October 3, 2026.

[4] TypeSafe AI. ["Primitives."](https://docs.typesafe.ai/primitives) Documentation, accessed October 3, 2026.

[5] TypeSafe AI. ["Confidence."](https://docs.typesafe.ai/confidence) Documentation, accessed October 3, 2026.

[6] TypeSafe AI. ["Models and Pricing."](https://docs.typesafe.ai/models) Documentation, accessed October 3, 2026.

[7] TypeSafe AI. ["Machine Learning Primer."](https://docs.typesafe.ai/introduction/machine-learning-primer) Documentation, accessed October 3, 2026.

[8] TypeSafe AI. ["Jev 1.13 Model Jaggedness."](https://docs.typesafe.ai/model-jaggedness/jev-1.13) Documentation, accessed October 3, 2026.

[9] TypeSafe AI. ["Use Case Map."](https://docs.typesafe.ai/concepts/use-case-map) Documentation, accessed October 3, 2026.

[10] TypeSafe AI. ["Patterns."](https://docs.typesafe.ai/patterns) Documentation, accessed October 3, 2026.

[11] TypeSafe AI. ["Team."](https://typesafe.ai/team) Accessed October 3, 2026.

[12] TypeSafe AI. ["System One Adapter for Python."](https://github.com/typesafe-ai/system-one-adapter-python) GitHub repository, 2026.

[13] Zhimin Gao and Pichao Wang. ["OpenJev-RLCD: A Working RLCD Implementation."](https://arxiv.org/html/2609.38850) arXiv:2609.38850, 2026. [Code.](https://github.com/ZimmyGao/openjev-rlcd)

[14] Mehul Damani, et al. ["Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty."](https://arxiv.org/html/2507.16806) arXiv:2507.16806, 2025; revised 2026. [Project page.](https://rl-calibration.github.io/)

[15] Jiaxin Zhang, et al. ["The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation."](https://arxiv.org/html/2604.16830) arXiv:2604.16830, 2026. [Code.](https://github.com/SalesforceAIResearch/CaOPD)

[16] Jiaxin Zhang, Caiming Xiong, and Chien-Sheng Wu. ["Agentic Confidence Calibration."](https://proceedings.mlr.press/v306/zhang26gy.html) *ICML*, 2026.

[17] Jiaxin Zhang, et al. ["Agentic Uncertainty Quantification."](https://arxiv.org/abs/2601.15703) arXiv:2601.15703, 2026.

[18] Jiaxin Zhang, et al. ["From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models."](https://aclanthology.org/2026.findings-acl.2064/) *Findings of ACL*, 2026.

[19] Tobias Deußer, Lorenz Sparrenberg, and Rafet Sifa. ["Evaluating and Benchmarking the System One Model Jev."](https://arxiv.org/html/2609.37647) arXiv:2609.37647, 2026.

[20] Lijuan Tang and Yuemeng Zheng. ["Typed Decision Models: An Early Evidence Audit and Evaluation Checklist."](https://arxiv.org/html/2609.32160) arXiv:2609.32160, 2026.

[21] Keyi Li, Yihao He, and Quanyi Li. ["Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms?"](https://arxiv.org/html/2609.33209) arXiv:2609.33209, 2026.

[22] Alfredo Madrid-García and Beatriz Merino-Barbancho. ["Jev in Medicine: A Benchmark Evaluation."](https://arxiv.org/abs/2609.34024) arXiv:2609.34024, 2026.

[23] Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. ["On Calibration of Modern Neural Networks."](https://proceedings.mlr.press/v70/guo17a.html) *ICML*, 2017.

[24] Brier, Glenn W. ["Verification of Forecasts Expressed in Terms of Probability."](https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2) *Monthly Weather Review*, 1950.

[25] Geifman, Yonatan, and Ran El-Yaniv. ["Selective Classification for Deep Neural Networks."](https://arxiv.org/abs/1705.08500) *NeurIPS*, 2017.

[26] Angelopoulos, Anastasios N., and Stephen Bates. ["A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification."](https://arxiv.org/abs/2107.07511) arXiv:2107.07511, 2021.

[27] Vercel. ["Jev Is the Fastest-Adopted Model in AI Gateway History."](https://vercel.com/blog/ai-gateway-jev-model-launch) September 18, 2026.

[28] Yu Sun, Junhao Xu, Jiajia Shi, and Zijin Yang. ["Type-Safe Is Not Error-Free: A Constrained Decision Head Follows the Option Name, Not the Rubric Bound to It."](https://arxiv.org/abs/2609.26758) arXiv:2609.26758, 2026.

[29] Jianyi Hu, Hangtao Zhang, Yi Liu, et al. ["JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models."](https://arxiv.org/abs/2609.31142) arXiv:2609.31142, 2026.

[30] Ruoqi Guo, Yi Liu, Gelei Deng, et al. ["Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures."](https://arxiv.org/abs/2609.29429) arXiv:2609.29429, 2026.

[31] Amir Rafe and Subasish Das. ["Benchmarking System One Decision Models against Trained Classifiers and Language Models for Automated Decision Gates."](https://arxiv.org/abs/2610.00346) arXiv:2610.00346, 2026.

[32] Kevin Yang, Dan Klein, Asli Celikyilmaz, Nanyun Peng, and Yuandong Tian. ["RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment."](https://arxiv.org/abs/2307.12950) *ICLR*, 2024.

[33] Yinheng Li and Justin Wagle. ["LLM2Jev: LLMs Are Already Jev-Style Decision Models—When and How to Fine-Tune Them."](https://arxiv.org/html/2610.02076) arXiv:2610.02076, 2026.

[34] Sharath M. Shankaranarayana, Davor Runje, and Jan Jannink. ["Beyond Answer Confidence: A Controlled Audit of Self-Knowledge in a Black-Box Decision Model."](https://arxiv.org/html/2610.01006) arXiv:2610.01006, 2026. [Code.](https://github.com/Syntheme/beyond-answer-confidence)

[35] Han Chen and Yingrui Li. ["Probability Contracts: Accuracy, Coherence, and Decisions Across LLM Interfaces."](https://arxiv.org/html/2609.37470) arXiv:2609.37470, 2026.

[36] Saman Sarker Joy. ["Do System One Decisions Add Up? A Study of Probabilistic Coherence."](https://arxiv.org/html/2609.33971) arXiv:2609.33971, 2026. [Code.](https://github.com/samanjoy2/system-one-coherence)

[37] Yaodong Yang, Hongyao Tang, Yi Ma, et al. ["Code Owns the Simulation, Jev Owns the Evaluation."](https://arxiv.org/html/2610.01834) arXiv:2610.01834, 2026.

[38] Jike Zhong, Ming Li, and Yuxiang Lai. ["When the Right Answer Is Missing: An Arithmetic-Dependent Rejection Bottleneck in Jev."](https://arxiv.org/html/2609.39496) arXiv:2609.39496, 2026.

[39] Yixuan Liu. ["Evaluating System One Models for Agent Security Decisions: Reliability, Calibration, and Selective Automation."](https://arxiv.org/html/2609.33401) arXiv:2609.33401, 2026. [Artifacts.](https://github.com/yxsec/system-one-security-eval)

[40] Luyao Tang and Cheng Chen. ["OmniMed-Jev: Calibrating LVLM Confidence for Trustworthy Medical Multimodal Decisions via System One."](https://arxiv.org/html/2610.00381) arXiv:2610.00381, 2026. [Code.](https://github.com/lytang63/OmniMed-Jev)

[41] Li Ding, Haidi Jin, and Chen Ji. ["Bongard: Training Machine Intuition—An Open Encoder–Decoder Model for Probabilistic Judgment."](https://arxiv.org/html/2609.39111) arXiv:2609.39111, 2026. [Weights.](https://huggingface.co/AgentBull/bongard-mini)

[42] Benchmark Heaven and contributors. ["JevBench: A Benchmark for Jev-Class Typed Decision Models."](https://github.com/fstandhartinger/jevbench) Versioned repository and [live leaderboard](https://benchmarkheaven.com/jev-models), accessed October 4, 2026.

[43] Rishabh Sharma and Rishika Lall. ["When Does Selection Replace Extraction? A Pre-Registered Test of Agent Memory with a Typed Decision Model."](https://arxiv.org/html/2609.34227) arXiv:2609.34227, 2026. [Code and preregistration.](https://github.com/ris3abh/Engram)

[44] Guangren Wang. ["Mnemon: Raw Records, Fast Judgments, Slow Thoughts."](https://arxiv.org/html/2609.36059) arXiv:2609.36059, 2026. [Code and run records.](https://github.com/Grivn/mnemon-memory-agent)

[45] Zhengle Wang, Hanxu Yan, Fuheng Zhao, and Chunwei Liu. ["Prune First, Decide Fast: Scalable Semantic Query Processing with JEVDB."](https://arxiv.org/html/2610.02046) arXiv:2610.02046, 2026.

[46] Tianwei Mu, Shengyan Jiang, Mingzhe Yuan, et al. ["HydroJEV: A One-Second, Training-Free Screen for Cyber-Attack and Fault Attribution in Water Distribution Networks."](https://arxiv.org/html/2610.02048) arXiv:2610.02048, 2026. [Code and data.](https://github.com/mutianwei521/hydrojev)

[47] Amir Rafe and Subasish Das. ["Calibrated Decisions at Scale: Converting Police Crash Narratives into Probabilistic Crash Variables with a System One Model (Jev)."](https://arxiv.org/html/2609.24052) arXiv:2609.24052, 2026.

[48] Linghua Zhang. ["Jev-Mobile: Jev as an Executor for Mobile GUI Agents."](https://arxiv.org/html/2609.30186) arXiv:2609.30186, 2026.

[49] Gabriele Cinà. ["Jev Matches 7B Language Models for Speech-Neuroprosthesis Rescoring."](https://arxiv.org/html/2609.33538) arXiv:2609.33538, 2026. [Code and data.](https://github.com/gabrycina/how-much-language-model)

[50] Tianxiang Gao, Jinzhe Li, Zhiyuan Li, Yi Chang, and Yuan Wu. ["More Choices, Fewer Decisions: Ordinal-Scale Bias in JEV-like Direct-Decision Models."](https://arxiv.org/abs/2609.38827) arXiv:2609.38827, 2026. [Code and data.](https://github.com/Glax147/jev_ordinal_scale_bia)

[51] Yida Lin. ["PACT: Pairwise-Anchored Calibrated Tuning for Single-Token Typed Decisions."](https://arxiv.org/abs/2609.35865) arXiv:2609.35865, 2026. [Code and runs.](https://github.com/BennyLinntu/PACT-Pairwise-Anchored-Calibrated-Tuning-for-Single-Token-Typed-Decisions)

[52] Guanxu Yu and Yuhang Yao. ["Visual Jev: Accurate and Efficient Decisions from Shared Visual Context."](https://arxiv.org/abs/2609.25845) arXiv:2609.25845, 2026. [Code.](https://github.com/guanxuyu-sv/Visual-Jev)

[53] Furkan Yilmaz, Habibe Aleyna Tasdemir, and Muhammed Faruk Gozay. ["LAVOIR: Teaching a Single-Pass Decision Encoder When and What to Ask with Amortized Value of Information."](https://arxiv.org/abs/2609.30706) arXiv:2609.30706, 2026. [Code.](https://github.com/moganai/lavoir) [Model.](https://huggingface.co/moganai/lavoir)

[54] Zexiao Wang, Zihao Zhang, Xudong Wang, et al. ["Chinese-Jev: Bringing System One Model to Chinese-Language Tasks."](https://arxiv.org/abs/2609.36965) arXiv:2609.36965, 2026. [Project.](https://gulucaptain.github.io/Chinese-Jev/)

[55] Theo Lee. ["SemIf-OpenJev: Semantic Ifs from Frozen Open Models."](https://github.com/TheoLeeCJ/SemIf-OpenJev) GitHub repository, accessed October 4, 2026.

[56] OpenJev contributors. ["OpenJev 27B."](https://huggingface.co/openjev/openjev) Hugging Face model card, accessed October 4, 2026.

[57] Iker Moel. ["open-alternative-jev."](https://github.com/ikermoel/open-alternative-jev) GitHub repository, accessed October 4, 2026.

[58] BlockBrain AI. ["Cygnet Recipe."](https://github.com/blockbrain-ai/cygnet-recipe) GitHub repository, accessed October 4, 2026.

[59] Alibek Serikbay. ["JevK5."](https://github.com/allebee/jevk5) GitHub repository and model family, accessed October 4, 2026.

[60] Chris H. ["Plumb."](https://github.com/crh225/plumb) GitHub repository and model recipe, accessed October 4, 2026.

[61] Kshetrajna. ["reflex."](https://github.com/kshetrajna12/reflex) GitHub repository, accessed October 4, 2026.

[62] Mapika. ["Decider: A Family of System One-Style Models."](https://github.com/Mapika/decider) GitHub repository and model family, accessed October 4, 2026.

[63] Jiamu Zhang, Tianze Yang, Yucheng Shi, and Liang Wu. ["AnyJev: Turn Any LLM into a Jev-Style Decision Model."](https://github.com/nokia-applied-research/AnyJev) GitHub repository; see also arXiv:2610.00831, 2026.

[64] Jared Palmer. ["Kev: Jev-Like Open Decision Models."](https://github.com/jaredpalmer/kev) GitHub repository and model family, accessed October 4, 2026.

[65] Eldan Ring. ["Winnow-12B."](https://huggingface.co/EldanRing/Winnow-12B) Hugging Face model card and benchmark document, accessed October 4, 2026.

[66] Akhilaaa3. ["Jev-Omni."](https://huggingface.co/akhilaaa3/Jev-Omni) Hugging Face model card, accessed October 4, 2026.

[67] Delip Rao and Chris Callison-Burch. ["JEV vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the Same Places."](https://arxiv.org/abs/2609.29769) arXiv:2609.29769, 2026.

[68] Oscar-dzy. ["Awesome JEV Papers."](https://github.com/Oscar-dzy/Awesome-jev-papers) GitHub literature index, last verified October 3, 2026.
