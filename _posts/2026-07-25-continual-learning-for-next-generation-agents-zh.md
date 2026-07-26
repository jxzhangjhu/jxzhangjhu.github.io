---
layout: post
title: "Continual Learning for Next-Generation Agents (中文版)"
date: 2026-07-25 10:00:00
author: Jiaxin Zhang
description: "下一代智能体应如何把部署期经验路由到 context、memory、harness、weights 与学习机制，并在多时间尺度上安全地持续学习。"
tags: agents continual-learning llm self-improvement
categories: research-notes
giscus_comments: true
related_posts: false
ai_assisted: true
og_image: https://jxzhangjhu.github.io/assets/img/blog/continual-learning-for-next-generation-agents/fig1_multitimescale_stack.png
---

<div class="lang-switch"><a href="/blog/2026/continual-learning-for-next-generation-agents/">English</a> · <strong>中文</strong></div>

### 目录

- [为什么 agent 必须在部署后继续学习](#为什么-agent-必须在部署后继续学习)
- [一套不会把万物混为一谈的词汇](#一套不会把万物混为一谈的词汇)
- [多时间尺度学习栈](#多时间尺度学习栈)
  - [Context：即时学习，安全遗忘](#context即时学习安全遗忘)
  - [Memory：持久化并保留来源链](#memory持久化并保留来源链)
  - [Harness 与 skill：把经验教训化为流程](#harness-与-skill把经验教训化为流程)
  - [Adapter 与 weights：巩固可迁移的模式](#adapter-与-weights巩固可迁移的模式)
  - [Updater：学会如何学习](#updater学会如何学习)
- [部署学习循环](#部署学习循环)
- [经典持续学习中仍然有效的东西](#经典持续学习中仍然有效的东西)
- [当下哪些方法有效](#当下哪些方法有效)
- [值得命名的失效模式](#值得命名的失效模式)
- [从持续学习到自我改进与 RSI](#从持续学习到自我改进与-rsi)
- [如何评估复利式学习](#如何评估复利式学习)
- [研究议程](#研究议程)
- [总结](#总结)
- [如何引用](#如何引用)
- [参考文献](#参考文献)

---

## 为什么 agent 必须在部署后继续学习

想象你招来了一位才华横溢的新研究助理。入职第一天，这位助理懂机器学习，会写代码，读过的论文比实验室里的任何人都多，却还不知道：你的团队用 bootstrap 区间报告不确定性；某个内部数据集上个月刚改过 schema；某个 benchmark（基准测试）藏着数据泄漏；或者，“把图做得更干净”其实意味着保留过去三篇论文沿用的配色。这位助理之所以会越来越顺手，不是因为通用智能每天都在变强，而是因为**情境化经验在不断累积**：纠正、结果、例外、惯例，以及一套关于这个组织究竟如何运作的模型。

再看今天的 agent（智能体）。它可以花一个小时搜索、写代码、调用工具、失败、恢复，最终解决任务——然后在下一次 session 开始时，仿佛这一切从未发生。我们投入巨量 inference-time compute 生成经验，最后却把其中绝大部分扔掉。

这种浪费很容易被低估，因为每一次损失单独看都很小。一次长 session 会弄清楚：五个看似合理的 API 里究竟哪一个真实存在；哪个测试是 flaky 的；哪份内部文档已经过期；遇到某个特定错误后，哪三步恢复流程有效。这些内容不在训练数据里，也不在最终答案里，默认情况下更不会活过这次 session。下一次运行会以同样的代价重新发现同样的事实。而用户昨天刚看着 agent 学会一件事，今天当然不希望再教一遍。

这个缺口正同时出现在一些看似非常不同的讨论中。2026 年 7 月，[《每日经济新闻》的一篇报道](https://www.nbd.com.cn/articles/2026-07-23/4504670.html)援引一份关于当年 5 月投资人会议的非官方 AI 整理版记录，把“持续学习是 agent 之后应当解决的问题”这一观点归于梁文锋：模型应当能够像进入组织的员工一样，在较长时间里继续学习。该媒体称，一家参与投资 DeepSeek 的机构确认了 5 月闭门会的存在，并认为内容真实可信；但目前没有公开录音，也没有 DeepSeek 官方逐字稿，因此本文把它视为一份**媒体报道、投资机构佐证的非官方记录**，而不是逐字路线图。不过，DeepSeek 的[官方招聘页面](https://talent.deepseek.com/)也提供了独立证据，表明这确实是组织内部正在推进的研究：页面列出了 `Frontier（持续学习/自进化/新范式）研究员` 岗位。

David Silver 与 Richard Sutton 从强化学习的角度切入同一个缺口。他们的 [**Era of Experience**](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) 主张，下一代 agent 应主要从长期、以环境为根基的交互流中学习，而不能只依赖固定的人类生产语料。Andrej Karpathy 对时间表更谨慎：在他的 [Dwarkesh 访谈](https://www.dwarkesh.com/p/andrej-karpathy)中，“你不能只是告诉它们一件事，就指望它们记住”这一事实，正是 agent 的“一年”更像是十年工程周期的原因之一。Dario Amodei 在[另一场访谈](https://www.dwarkesh.com/p/dario-amodei-2)中给出了一个很有价值的补充视角：也许在持久化 weight 更新得到解决之前，更长的 context 与更强的 in-context learning 就足以覆盖大量在岗学习。毕竟，100 万 token 足以装下数天的阅读内容。

这些观点并没有看起来那么矛盾，因为它们讨论的是**不同的时钟**：

- context（上下文）可以在一次 session 内调整行为；
- memory（记忆）可以把筛选过的经验带到下一次 session；
- skill 与 harness（围绕模型的提示、工具和控制软件）可以编码经过验证的流程；
- adapter 与 weights（权重）可以压缩能够广泛迁移的模式；
- 外层学习机制则可以改进上述所有更新的选择方式。

所以，真正的问题不是：“agent 应当在 context 中学习，还是在 weights 中学习？”真正的问题是：

> **面对一条部署经验，应当把它写到哪里、作用于多大范围、保留多久，以及在它获准影响更多用户之前，需要什么证据？**

这就是本文的核心论点：

> **论点。** 面向 agent 的持续学习，已经不再只是“更新参数而不发生灾难性遗忘”。它要解决的是：**把经验路由到正确的更新表面，并跨多个时间尺度加以巩固**——同时守住迁移、隐私、校准、安全与回滚能力。

![一个 agent、五个学习表面与多个时钟。](/assets/img/blog/continual-learning-for-next-generation-agents/fig1_multitimescale_stack.png)
*图 1. 一个 agent 拥有多个学习表面。快表面局部且可逆；慢表面共享、强大，却需要更严格的验证与治理。“实时反馈”应当快速进入系统，但未必应当快速进入共享 weights。图中的时钟与 scope 是典型设计选择，而不是内在限制：memory 可以是 fleet 级的，adapter 可以是个人级的，router 也可以运行在热路径。*

下文会贯穿一个例子：**一名刚加入 AI 实验室的研究 agent**。

- 第一次 session 中，用户纠正了一条项目专属的引用规则。
- 第一周里，它学会了反复出现的项目事实、人员、数据集与偏好。
- 第一个月里，反复成功的工作流变成了一项经过验证的文献检索或消融分析 skill。
- 横跨许多项目后，某种模式通过了 held-out 评测，成为 adapter 或 weight 训练材料。
- 横跨多个 release 后，系统学会了哪些反馈来源与更新路径真正能够预测迁移；此时，连 *updater 本身*也发生了变化。

把每一条纠正都写进共享 weights，既不安全又缓慢，而且常常是错的。把所有东西都留在当前 context，则会让每次新 session 都变成 agent 上班的第一天。真正的设计问题，就落在这两种极端失败之间。

---

## 一套不会把万物混为一谈的词汇

“Continual learning”、“online learning”、“lifelong learning”、“test-time learning”、“self-improvement”和“RSI（递归自我改进）”经常被说成同一种能力逐级升级后的不同名字。它们不是。更有用的词汇体系，会对每个术语分别追问一个不同的问题。

**Online learning（在线学习）**问的是*节奏*：学习器是否按顺序处理样本，并随数据流抵达而更新？在线算法可以优化累计奖励或 regret，却不保留每一项旧技能。反过来，一个系统也可以通过周期性 batch 做持续学习，而不是每看到一个样本就更新一次。

**Continual learning（持续学习，CL）**问的是*在变化数据流上的能力*：系统能否在资源受限的情况下，获得有用的新行为、保留仍然重要的旧能力，并把经验向前迁移？经典目标同时包含**稳定性—可塑性权衡**、任务内与任务间泛化，以及效率（[Wang et al., 2024](https://arxiv.org/abs/2302.00487)）。灾难性遗忘是稳定性一侧的失败。还有一个同样重要的镜像问题：**可塑性丧失（loss of plasticity）**——反复训练的网络保住了旧行为，却逐渐丧失学习新行为的能力（[Dohare et al., 2024](https://arxiv.org/abs/2306.13812)）。

**Continuous learning（连续学习）**与 **continuous training（连续训练）**通常是工程术语，指一条周期性流水线：收集数据、重新训练、验证、发布。一个每周运行的 batch 在运营意义上可以是“continuous”的，即便它没有使用任何持续学习算法，甚至每次都从头训练。从产业视角看，这其实是带版本的模型生态中的“更新与发布”问题——[Jiang 等人（2026）](https://arxiv.org/abs/2606.24901)正是用这个框架论证：可塑性余量、能力继承与问责，属于生命周期层面的问题，而不只是算法问题。

**Lifelong learning（终身学习）**是更宽泛的愿景：一个开放式学习器在整个运行生命周期中不断积累知识与技能。大量文献会把 *lifelong*、*continual* 与 *incremental* learning 混用；[Zheng 等人（2026）面向 agent 的 roadmap](https://doi.org/10.1109/TPAMI.2025.3650546) 很好地梳理了这批文献，也梳理了在变化环境中感知、memory 与行动如何相互耦合。本文把 *lifelong* 作为目的地，把 *continual learning* 作为需要解决的技术问题。

**In-context learning、test-time adaptation 与 test-time learning**问的是行为是否会在部署期间改变。这种变化可能只存在于 context、外部 memory、prompt 或临时参数里。它们都算学习机制，却未必能够持久化。一个系统如果第五次尝试比第一次做得更好，说明它已经适应；但这并不意味着，面对明天另一项不同的任务时，它也会更好。

**Self-improvement（自我改进）**问的是谁产生了候选改进。一个模型批评并重写自己的单次回答，做的是有边界的自我精炼。一个 agent 编辑一项可复用 skill，并保留更好的版本，做的是持久化自我改进。不能只因为循环两侧出现的是同一个模型，就把二者称为递归。

**Self-evolution（自演化）**没有唯一且公认的定义。它通常指由外循环选择的持久结构变化——变化对象可以是 memory、skill、工作流、代码、种群或环境。在姊妹篇[自演化 agentic harness](https://jxzhangjhu.github.io/blog/2026/self-evolving-agentic-harnesses-zh/)中，可变对象是冻结模型周围的软件。

**Recursive self-improvement（递归自我改进，RSI）**问的是*改进循环的拓扑*：系统是否会改进那个产生下一轮改进的 updater、evaluator、curriculum、研究流程，或继任者构建机制？持续学习描述能力如何随时间变化；RSI 描述改进过程本身是否也成为了改进对象。

两个坐标轴足以避免大多数概念混淆：

1. **持久性：** 单条 trajectory 内的短暂变化 → 跨 session 持久存在 → 被未来版本继承。
2. **变化对象：** 当前知识/行为 → memory/harness/weights → 学习机制本身。

把这些术语放到这两条轴上，区别就从修辞变成了机制：

| 术语 | 它回答的问题 | 典型持久性 | 变化对象 |
|---|---|---|---|
| Online learning | 更新的节奏 | 视情况而定 | 通常是 weights |
| Continual learning | 变化数据流上的能力 | 跨 session 或跨版本 | 任意持久表面 |
| Continuous training | 流水线的排期 | 跨 release | 通过重训改变 weights |
| Lifelong learning | 长期愿景 | 整个运行生命周期 | 任意表面 |
| In-context / test-time learning | 部署期行为是否改变 | 常常只在单个 episode 内 | context 或临时参数 |
| Self-improvement | 谁提出改进 | 有边界或持久 | 工件、memory、harness、weights |
| Self-evolution | 外循环在什么之上做选择 | 持久 | 结构：memory、skill、代码、种群 |
| RSI | 改进循环的拓扑 | 被继任者继承 | updater 与研究流程本身 |

Retrieval、reflection、fine-tuning 与 self-revision 都可以参与持续学习，但没有哪一种单独就能保证累积式学习。持续学习是 RSI 的合理基础设施，却不是 RSI 已经发生的证据。

> **一个实用检验。** 在一次看似成功的改进之后，重置 context，再给出一项全新的未来任务。究竟是哪部分状态把改进带到了未来？它能迁移吗？它遗忘了什么？如果答案是“没有任何状态”，那么你观察到的是 refinement，而不是持续学习。

---

## 多时间尺度学习栈

把时刻 $$t$$ 的已部署 agent 表示为

$$A_t = (C_t, M_t, H_t, \theta_t, U_t),$$

其中：

- $$C_t$$ 是 context 与工作状态；
- $$M_t$$ 是 episodic、semantic 与 procedural memory；
- $$H_t$$ 是 harness——prompt、工具、控制流、skill 与 validator；
- $$\theta_t$$ 是 adapter 与共享模型 weights；
- $$U_t$$ 是更新机制：router、evaluator、curriculum，以及训练/搜索配方。

这并不是说每个产品都需要五个独立数据库。这个表示只是为了把**学习发生的位置**显式化。每个表面都有不同的写入延迟、作用范围、容量、回滚机制与失败模式。

| 表面 | 典型写入延迟 | 自然作用范围 | 如何撤销 | 特征性失败 |
|---|---|---|---|---|
| Context $$C_t$$ | 即时 | 单次 session | 结束 session | 稀释；早期错误变成前提 |
| Memory $$M_t$$ | 秒到分钟 | 用户、项目、组织 | 删除或用新条目取代 | 陈旧、矛盾或分散注意力的召回 |
| Harness $$H_t$$ | 小时到天 | 产品层面 | 回滚提交 | 过拟合的流程；脆弱的工具契约 |
| Weights $$\theta_t$$ | 天到周；窄信号下可到小时级 | 使用该 checkpoint 的所有人 | 回滚 checkpoint | 遗忘；校准与安全漂移 |
| Updater $$U_t$$ | 周到月 | 未来每一次更新 | 恢复旧策略 | 系统性的错误晋升 |

这张表描述的是一条成本梯度，而不是一架等级阶梯。越往上，写入越便宜、也越容易被论证为合理；越往下，写入越有力、越危险，需要的证据也越多。生产中关于“这件事该不该让 agent 学”的争论，大多其实是在争论：这次提议要写入的，究竟是表中的哪一行。

后文会反复用三个简称来指代这条梯度：**热路径（hot path）**在数秒内写入 context 或隔离 memory；**暖路径（warm path）**把已验证的经验晋升为持久 memory、skill 或 harness 代码；**冷路径（cold path）**改动 adapter、共享 weights，或 updater 本身。后面会分别给出它们各自的发布要求。

### Context：即时学习，安全遗忘

Context 是最快的表面。一条纠正无需启动训练任务，也无需持久写入，就能改变 agent 的下一个动作：

> “这个项目请使用预注册划分，而不是公开排行榜划分。”

对研究 agent 来说，正确的即时反应是把这条指令放进工作状态，重新审视当前计划，并在必要时修复已经完成的工作。最后这一步恰恰是大多数系统会跳过的：一条只改变未来动作的纠正，会留下一个前后不一致的产物——结果表里有一半是用错误划分算出来的。快速学习必须包含本 episode 内的回溯修复。

Context 便宜、可逆，因此很适合承载试探性信念、局部工具状态与一次性约束。实践中有四类内容属于工作状态：当前计划及其未决问题、用户已经声明的约束、本次 session 中发现的环境事实（哪条命令可用、哪个路径存在），以及已经被排除的失败路径。这几类内容，要么重新发现的代价很低，要么尚未被充分验证，还不适合晋升。

它的局限同样重要。Context 会结束；过长的历史会稀释相关证据；而一个早期错误可能成为此后每一步的前提。更长的 context 增加了模型*可以*据以调整输出的信息量，却不会替系统判断什么值得持久化，也不会消解矛盾，更不能保证新的 session 会带着这条经验开始。这就是长 context 与持续学习互为补充、而非彼此替代的原因。

经济账也指向同一个结论。每一轮都重新读一遍庞大历史，是一笔按 session、按用户反复支付的成本；而 consolidation 只需支付一次。当同样 40 行来之不易的项目惯例要在每次 session 中被重新推导时，更划算的设计是把它们写下来；而当某条约束只适用于一次对话时，为 context 付费才是正确的。“这条经验该放在哪里”在某种程度上是一个带准确率代价的缓存问题。

### Memory：持久化并保留来源链

Memory 跨越了 session 边界。它不应该只是一袋旧消息。一套有用的 agent memory 至少承担三种角色：

- **episodic（情景记忆）：** 某次运行发生了什么，包括模型/harness 版本与结果；
- **semantic（语义记忆）：** 从多个 episode 中抽象出的、相对稳定的事实与偏好；
- **procedural（程序性记忆）：** 关于如何行动的一条可复用经验。

Recall 与 learning 之间的区别至关重要。[Evo-Memory](https://arxiv.org/abs/2511.20857) 说得很清楚：对话回忆检索的是*说过什么*；经验复用检索的是*学到了什么*——例如检索一元二次方程求根公式，而不是上一道方程的根。后者需要抽象、筛选与修订。

每一条持久 memory 都应携带 metadata：来源、时间戳、作用范围、置信度、证据、失效时间，以及指向矛盾条目或明确取代它的新条目的链接。[ARIA](https://arxiv.org/abs/2507.17131) 给出了一个具体且有用的模式：已部署 agent 识别知识缺口，向人类专家提出一个有针对性的问题，把指导写进带时间戳的知识库，与相关条目对照，并在规则冲突时请求澄清。

写出来看，一条持久条目不像聊天记录，更像一条带归属人和失效日期的小型记录：

```yaml
id: mem_4193
kind: procedural
claim: "在内部 QA 数据集上比较 retrieval 方法之前，先运行泄漏检查。"
scope: {project: qa-retrieval, users: all}
source: {type: human_correction, actor: reviewer, session: s_2291}
evidence: [run_8842_failed_review, run_8871_passed_after_check]
confidence: 0.82
created: 2026-05-02
review_by: 2026-11-02
supersedes: mem_3907
status: active
```

这些 metadata 不是官僚主义，每个字段都回答了学习循环迟早要问的一个问题。`scope` 决定影响范围；`source` 与 `evidence` 决定当它与另一条冲突时应被信任到什么程度；`review_by` 迫使陈旧规则过期而不是不断堆积；`supersedes` 则让规则变更后的历史仍然可审计。缺少这些字段，“解决冲突”就会退化成“retriever 恰好返回了哪一条”。

由此，memory 质量由四种操作决定，而其中只有一种是检索：

- **write（写入）：** 判断某件事是否值得持久化，以及应写在什么作用范围；
- **revise（修订）：** 新证据到来时，合并、细化或推翻已有条目；
- **retrieve（检索）：** 只呈现与当前步骤相关的少数条目，而不是所有相似内容；
- **forget（遗忘）：** 过期、剪枝或按请求删除，并把删除传播到由它派生的一切。

对我们的实验室 agent 而言，“Jiaxin 喜欢蓝色图表”可能只作用于某个用户；“Dataset v3 把 `label` 改名为 `target`”只作用于某个项目和时间段；“比较 retrieval 方法之前先做泄漏检查”则可能成为一条 procedural memory。Scope（作用范围）本身就是正确性的一部分。

Memory 很有吸引力，因为它可编辑、可删除。它也很危险，因为 retrieval 很容易给人一种“免费改进”的错觉。[LifelongAgentBench](https://arxiv.org/abs/2505.11942) 发现，当无关历史占据 context 或分散 agent 注意力时，加入更多经验反而会*降低*性能。Agent memory 需要垃圾回收、冲突消解与评测，而不只是向量搜索。

### Harness 与 skill：把经验教训化为流程

有些经验教训最适合表达成可检查的工件：

- 一份论文审阅清单；
- 一段验证参考文献链接的脚本；
- 一个返回更短、更结构化输出的工具 wrapper；
- 一条在反复失败后调用第二个模型的路由规则；
- 一项在声称任务完成前先运行测试的控制流改动。

这些都属于 **harness 与 skill 层**。与 weights 相比，harness 改动便宜、可读、可 diff，也容易回滚。与原始 memory 相比，它能同时编码一段可执行流程及其 verifier（验证器）。这是“记住这件事”与“重新训练模型”之间的暖路径。

姊妹篇 [harness 演化文章](https://jxzhangjhu.github.io/blog/2026/self-evolving-agentic-harnesses-zh/) 专门研究了这一层。本文更关心它在整个栈中的位置：trace 可以先变成有明确 scope 的 memory；反复出现且经过验证的模式，再晋升为 skill 或工具。晋升所需的证据必须表明，这套流程修复的不只是最初触发它的那个 episode。

一条可操作的“memory → skill”晋升规则包含四个条件，而通常被漏掉的正是最后一条：

1. **重复出现：** 这类情形出现的次数已经多到，用流程比用查表更划算；
2. **确定性：** 正确反应是一串步骤，而不是依具体情况而定的判断；
3. **可检验性：** 该流程可以由提出它的模型之外的东西来验证；
4. **无干扰性：** 在不适用的情形下运行它无害，或者这项 skill 知道自己的前置条件。

违反条件 4，正是 skill 库逐渐劣化的方式。一项为修复某个失败而写的 skill，常常会在从未被测试过的相邻情形中被触发，代价则落在无关任务上。在 skill 内部显式写明前置条件，并在“它不应触发”的用例上测试它，才能避免不断增长的库变成一个又慢又自相矛盾的第二模型。

### Adapter 与 weights：巩固可迁移的模式

Weights 的价值在于压缩行为。被内化到 adapter 或共享模型中的经验无需 retrieval，泛化能力也可能超出逐字存储的 memory。但与 memory 或 harness 改动相比，weight 更新的影响范围要大得多，局部可逆性也最弱。

只有当一种模式满足以下条件时，才适合写入这个表面：

1. 得到许多可追溯经验的支持；
2. 在单个用户或项目之外仍然有用；
3. 足够稳定，不会下周就过期；
4. 在法律与隐私层面允许用于训练；
5. 能够通过 replay（回放）、held-out 迁移、安全与校准检查。

生物学上的类比是[**互补学习系统（complementary learning systems）**](https://doi.org/10.1037/0033-295X.102.3.419)：快速的 episodic 存储保护单次经验，较慢、交错进行的 consolidation（巩固）则把结构提炼进 semantic memory。这个类比并没有规定具体算法，却给出了正确的系统原则。[Dorovatas et al. (2026)](https://arxiv.org/abs/2603.01761) 的 memory-first 立场论文把 agent 版本说得很明确：把快速的 in-context adaptation 与低频的 in-weight learning 结合起来，不要强迫任一表面同时承担两项工作。

对研究 agent 来说，一次成功的消融表格排版方式应当先留在 skill 或 memory 中。如果同一流程在数百个项目与模型之间都能迁移，它才可能已经适合整合进 adapter 或 weights。

这里也值得把“一次 consolidation 任务到底包含什么”讲清楚，因为“更新 weights”这句话掩盖了绝大部分工作量。一个现实的冷路径任务会：从已接受的经验及其经过验证的结果中组装训练目标；混入代表“不允许移动的能力”的 replay 数据；选择写入表面（按 scope 划分的 LoRA、合并后的 adapter，还是共享 checkpoint）；训练；然后在任何流量切换之前，跑完局部修复、held-out 迁移、回归、安全与校准评测。其中训练这一步往往是最便宜、不确定性最小的一环。

作用范围在这里同样重要。按用户或按组织划分的 adapter 能把干扰限制在局部，也让删除变得可行，代价是服务复杂度上升、共享变弱；单一共享 checkpoint 则获得最大迁移与最小隔离。这个选择并不纯粹是技术问题：它决定了某个客户的特异惯例是否可能拖累另一个客户的结果。

### Updater：学会如何学习

最后一个表面很容易被忽略，因为它不是通常意义上的“知识”。系统必须决定：

- 哪些反馈可信；
- 一次失败应归因于模型、memory、harness、工具还是环境；
- 哪些经验值得 replay；
- 应该用哪些未来任务评测哪一项候选更新；
- 何时应当回滚更新；
- 如何分配探索与训练资源。

把这个机制记为 $$U_t$$。一条手写规则——“把每一条用户纠正都存进 memory”——就是 updater。一个学习得到、负责路由反馈的模型也是 updater。一个跨任务优化 test-time adaptation prompt 的演化过程同样是 updater。

一个**经过校准的更新 router**可以写成

$$R(e_t, q_t) \rightarrow (s, \sigma, \tau, a, c),$$

其中，$$e_t$$ 是一条经验；$$q_t$$ 汇总可信度、因果归因、新颖性、重复出现程度、可迁移性、隐私、回归风险与成本；$$s$$ 是目标表面；$$\sigma$$ 是 scope；$$\tau$$ 是 time-to-live 或复核周期；$$a$$ 是写入、隔离、请求证据、暂存、合并或丢弃；$$c$$ 则是对该路由决定的置信度。

$$q_t$$ 里的特征值得展开，因为它们正是 router 可以被训练和评估的对象：

- **可信度：** 这个信号由谁或什么产生，是否有独立证据与之一致？
- **归因：** 观察到的结果由模型、memory、harness、工具还是环境导致？
- **新颖性与重复出现：** 这是新信息吗？它出现的次数是否已足以支撑泛化？
- **可迁移性：** 这条经验教训是否有理由适用于产生它的那个案例之外？
- **隐私与合规：** 这段内容是否允许持久化，以及可以持久化到什么范围？
- **回归风险：** 这次改动可能破坏哪些既有行为？
- **成本：** 在其整个生命周期里，写入、存储、检索或训练它各要花多少？

这样看，router 是一个带显式预算的决策策略，而不是一个分类器。每条路由都带有期望收益（未来任务的改进）、期望成本（计算、延迟、人工复核时间）与期望风险（回归、泄漏、投毒）。在能拿到大部分收益的前提下选择最小可逆写入，本质上是一个带延迟、部分可观测奖励的 bandit 问题——这也正是路由值得被学习和度量，而不是一次性写死的原因。

这与[校准文章](https://jxzhangjhu.github.io/blog/2026/calibrating-long-horizon-agents/)直接相连：不确定性不应只决定 agent 是否行动，还应决定由此得到的经验是否适合记住，以及它可以传播多远。

Updater 表面也是通往 meta-learning 与 RSI 的桥。学会一个新事实，会改变 agent 知道什么；学会哪种更新策略能够带来未来收益，则会改变 agent 如何学习。

> **要点。** 这个栈并不是一架最终会把每条 memory 都送进 weight 的梯子。绝大多数经验都应在局部过期。要晋升到更慢、更广、可逆性更弱的表面，就必须拿出更强的证据。

---

## 部署学习循环

学习栈说明学习可以发生在*哪里*；学习循环则说明一条经验要经过什么考验，才有资格被持久写入。

![带门控的部署学习循环。](/assets/img/blog/continual-learning-for-next-generation-agents/fig2_deployment_learning_loop.png)
*图 2. 捕获不等于学习，反馈也不会自动成为真相。一次部署经验只有经过验证、路由、巩固，以及回归/安全门，才能变成正式 release。被拒的证据可以先行隔离，而不必阻塞快速的局部适应。*

纵观 memory 系统、生产工程报告与 weight 更新论文，同一副骨架反复出现。

### 1. 捕获完整经验

不要只保存最终答案。一条有用的记录应当包括：

- 任务及相关环境状态；
- 完整的动作/工具 trajectory；
- 模型、prompt、memory、工具与 harness 版本；
- 最终结果与中间结果；
- 显式反馈及此后出现的下游结果；
- 身份、同意、隐私类别与保留政策。

如果没有带版本的 lineage（谱系），延迟反馈几乎无法归因。如果一个 patch 在一周后引发事故，学习器必须知道它由哪个模型和 harness 产生、当时有哪些工具可用，以及此后有哪些编辑介入其中。

环境之所以重要，是因为经验由策略产生。一旦 agent 改变，它就会访问不同的状态分布，得到不同的反馈。这就是[环境扩展文章](https://jxzhangjhu.github.io/blog/2026/environment-scaling-for-agentic-rl-zh/)与持续学习构成同一系统两半的原因：环境产生 trajectory 与 verifier；持续学习决定这些 trajectory 应当改变什么。

### 2. 验证信号并归因

反馈是证据，不是 ground truth。

- 用户编辑可能表达的是个人偏好，而不是普适规则。
- 接受一条建议可能只是“足够好，能省时间”，而不是“正确”。
- 一次工具调用失败可能源于 credential 过期，而不是推理失败。
- 一项通过的测试可能根本没有覆盖用户真正关心的行为。
- 模型的 self-critique 与犯错模型自身的盲点高度相关。

所以，验证需要回答两个问题：

1. **这个信号可信吗？** 检查来源、权威性、隐私、对抗风险，以及它是否与独立证据一致。
2. **是什么导致了这个结果？** 把失败归因到模型知识、memory retrieval、context 管理、工具/接口、harness 控制流、环境，或任务歧义。

不同类型的反馈会以不同方式失效，用统一方式对待它们，正是学习循环出问题的原因：

| 反馈类型 | 例子 | 主要弱点 | 合理的默认处理 |
|---|---|---|---|
| 显式纠正 | “请使用预注册划分” | 可能只编码了某个人的偏好 | 写入能容纳它的最小 scope |
| 隐式行为 | 接受、拒绝、编辑、重试 | 是价值的代理，不是正确性 | 先聚合，再采信 |
| 可执行 verifier | 测试、类型检查、schema 校验 | 可能覆盖不到真实目标 | 在其覆盖范围内是强信号 |
| 延迟结果 | 一周后发生的事故 | 归因确实困难 | 在反复出现前先隔离 |
| 模型自评 | agent 给自己打分 | 与自身盲点相关 | 绝不作为晋升的唯一门 |
| 对抗输入 | 工具结果中注入的指令 | 被刻意伪装成证据 | 不可信内容只当数据，绝不当指令 |

最后一行值得强调，因为 memory 会把一次性攻击变成持久攻击。如果 agent 把从网页或工具输出中读到的内容写进持久 memory，那么控制这段内容的攻击者，就控制了 agent 未来的一部分指令。因此，provenance 是一条安全边界，而不只是记账上的便利。

[ARIA](https://arxiv.org/abs/2507.17131) 的专家澄清机制，是处理显式领域反馈的一种方案。可执行 verifier 若真正覆盖目标，则提供更强的证据。延迟出现的业务结果需要因果分析，而且在反复出现之前，往往应当继续隔离。

### 3. 按表面、scope 与生命周期路由

路由时应优先选择能够解决问题的、最小且可逆的写入：

- **context**：试探性纠正或仅限一次 session 的纠正；
- **带 namespace 的 memory**：用户/项目事实与不断变化的规则；
- **skill 或 harness**：经过验证的流程；
- **adapter 或 weights**：可广泛迁移的模式；
- **updater/evaluator**：只在证据横跨多个学习周期时才更新。

同样的内容会因 scope 不同而走上不同路径。“始终使用 APA 格式”可能是一项用户偏好、一条某期刊项目专属的规则，也可能是全产品默认值。一个 router 即使猜对了表面，只要猜错 scope，仍然是错的。

路由输出还应包含不确定性。低置信度不必让学习完全停止：热路径可以先把条目放进临时 context 或隔离 memory，同时请求更多证据。不确定性真正应当阻止的是*晋升*。

用研究 agent 一周里的六条经验，可以把这个决策讲得很具体：

| 经验 | 表面 | Scope | 生命周期 | 生效前需要的门 |
|---|---|---|---|---|
| “本次运行请使用预注册划分。” | context | 单次 session | 本 episode | 无；修复当前工作即可 |
| “我喜欢用实验室的蓝色配色画图。” | memory | 用户 | 直到被修改 | 确认这是偏好而非规范 |
| “Dataset v3 把 `label` 改名为 `target`。” | memory | 项目 | 直到 schema 再次变化 | 一次独立确认 |
| “比较 retriever 之前先做泄漏检查。” | skill | 先项目、后组织 | 受版本管理 | 在触发案例之外也能通过 |
| 200 次 session 中反复出现的引用格式修复 | adapter | fleet | 直到下个 checkpoint | replay、迁移与回归评测 |
| “这位审稿人总是想要更短的摘要。” | 隔离区 | 待定 | 30 天复核 | 需在多位审稿人处复现才可晋升 |

同一句话会因为“谁说的”和“被确认过多少次”而落到不同行。这正是重点：路由是证据与 scope 的函数，而不是内容表面形式的函数。

### 4. Replay、对照与巩固

原始 trajectory 并不是理想的长期 memory。Consolidation（巩固）应当：

- 对重复经验去重；
- 不只保留成功，也保留反例与失败；
- 对照不同案例，找出真正起作用的因素；
- 交错放入有代表性的旧行为，以检测遗忘；
- 把 episode 蒸馏成事实、流程或训练样本；
- 保留从每一项抽象结果返回源证据的链接。

[Contextual Experience Replay（CER）](https://arxiv.org/abs/2506.06698) 把 web agent trajectory 中的环境动态与决策模式蒸馏进一个可检索 buffer。它很好地说明，对语言 agent 而言，“replay”可以指*在 context 中回放抽象后的经验*，不一定意味着在旧样本上计算梯度。对于共享 weight 更新，replay 只是选项之一：交错加入有代表性的旧数据或旧策略行为可以约束遗忘，regularization、distillation 与参数隔离也能提供替代路径。

### 5. 用迁移与非回归把关

一项更新如果修好了触发它的那个样本，只是通过了最容易的一道测试。要正式 release，应当要求三类彼此不同的证据：

1. **局部修复：** 它修复了观察到的失败吗？
2. **Held-out 迁移：** 它能帮助那些未被用来提出改动、但与之相关的未来案例吗？
3. **无关键回归：** 它能否在规定边界内保住旧能力、安全规则、校准、隐私、延迟与成本？

许多 self-improvement 的主张，正是在这里退化成普通的过拟合。如果同一条 trace 同时负责提出、选择并解释更新，那么整个循环就没有独立证据。

门控应当针对不同表面分别设计。私有 memory 不需要全 fleet 能力测试，却需要 scope 与隐私检查；工具改动需要集成测试与对抗测试；weight 更新需要有代表性的 replay、广泛的安全/校准评测，以及 checkpoint 回滚计划。

| 表面 | 局部修复 | 迁移证据 | 回归底线 | 回滚方式 |
|---|---|---|---|---|
| Memory 条目 | 它来自的那个案例 | 不作要求 | scope 与隐私类别正确；与已有条目不矛盾 | 删除该条目 |
| Skill 或工具 | 失败的那条 trace | 触发案例之外的用例 | 通过集成与对抗测试 | 回滚提交 |
| Adapter | 同领域的 held-out 切片 | held-out 任务族 | 受保护能力在边界内 | 卸载 adapter |
| 共享 weights | 同上 | 同上，外加 fleet 分布 | 能力、安全、校准、延迟、成本 | 上一个 checkpoint |
| Updater 策略 | 它修复的那些路由错误 | 后续更新周期 | 晋升精确率不下降 | 恢复旧策略 |

各行之间的不对称是刻意的。要求一条个人 memory 提供 fleet 级证据，会让快路径失去意义；而只凭触发案例的证据就更新 weights，正是一个看似合理的修复演变成全体回归的方式。

### 6. 分阶段 release

Scope 可以逐步扩大：

`session → user → project → organization → canary cohort → fleet`.

每向前一步，都应当把候选版本与稳定 control 对比，并保留快速回滚能力。一条有用的运维原则是：

> **快速摄取与缓慢晋升并不冲突。**

Agent 可以在几秒内响应一条纠正，而无需重写共享 weights。局部写入会立即产生价值；与此同时，系统继续收集暖更新或冷更新所需的证据。

### 7. 监控、回滚与 meta-learn

Release 并不是学习的终点。系统需要监控分布漂移、过时 memory、意外交互、群体回归，以及反馈与真实结果之间关系的变化。回滚是一项正常操作，不代表学习循环已经失败。

有几个信号值得持续监控，而不是等到 release 时才看：被检索出来却从未用于最终答案的 memory 占比、仍在被触发的条目的年龄分布、新写入与既有条目发生矛盾的比率、事后被撤销的晋升比例，以及代理信号（接受率、点赞）与你事后能测到的任何真实结果之间的差距。这些指标往往缓慢而无声地劣化，因此它们需要的是一块看板，而不是一次事故。

经过许多轮更新，系统可以提出一个更高阶的问题：哪些来源、路由、replay 策略与门控，曾经预测到持久迁移？更新这些规则，就改变了 $$U_t$$——这正是 learning to learn 的起点。

由此得到三条部署路径：

- **热路径（hot path）：** context 或隔离 memory；耗时数秒；局部且容易撤销。
- **暖路径（warm path）：** 经过验证的 memory 晋升，或 skill、prompt、工具、路由改动；带版本，且必须通过 eval 门。
- **冷路径（cold path）：** adapter、共享 weights 或 updater；需要有代表性的 replay、广泛的回归测试、分阶段 checkpoint release 与回滚。

实时反馈位于三条路径的入口。它并不意味着实时访问每一个参数。

---

## 经典持续学习中仍然有效的东西

经典持续学习通常设想一个神经网络依次接收一系列任务。现代 agent 是一套复合系统，但只要把旧机制映射到扩展后的可变表面上，它们依然有用。

![经典持续学习机制到 agent 更新表面的映射。](/assets/img/blog/continual-learning-for-next-generation-agents/fig3_classical_cl_to_agent_surfaces.png)
*图 3. Replay、regularization、isolation、dynamic architecture、distillation 与 meta-learning 依然重要。变化在于，被保护或被更新的对象可以是 memory、skill、工具、adapter、模型或 updater，而不再只是一个单体参数向量。*

| 经典机制 | 原始形态 | Agent 时代的形态 | 新的失败模式 |
|---|---|---|---|
| Replay | 交错回放存储的旧样本 | trajectory buffer、回归套件、context 中的经验复用 | 陈旧或无关经验挤占任务空间 |
| Regularization | 惩罚重要权重的移动 | 对任意表面施加非回归约束 | 保住了被测行为，而非真正意图 |
| 参数隔离 | 按任务分配参数或掩码 | memory namespace、按 scope 的 adapter、沙箱工具 | 路由错误；正迁移被阻断 |
| 动态架构 | 按任务扩容 | 增加 memory、skill、工具、子 agent | 只增不减，缺乏剪枝 |
| Distillation | 教师到学生的迁移 | session 到 weight、fleet 到 base 的巩固 | 悄悄复制了教师的错误 |
| Meta-learning | 学习学习规则 | 学习 router、evaluator 或适应策略 | 更新策略过拟合到单一 benchmark |

### Replay 变成经验与回归基础设施

[Experience replay](https://arxiv.org/abs/1811.11682) 通过混合旧样本与新样本，近似联合训练；[Gradient Episodic Memory](https://arxiv.org/abs/1706.08840) 把这个想法进一步收紧：把存储样本当作约束而不仅是额外数据——只有在不增大保留样本损失的前提下，一次更新才被允许。对 agent 而言，“旧样本”扩展为：

- trajectory 与环境状态；
- 成功和失败的流程；
- 带 scope 的用户/项目 memory；
- 工具接口 contract；
- 回归任务与安全案例；
- 旧 checkpoint 的输出。

Replay 有两项工作：用旧行为的代表性样本守住稳定性，并揭示一条新经验能否迁移到触发案例之外。Replay 并非越多越好：陈旧或无关的经验会增加 context 与计算负担，侵犯隐私，也可能阻碍适应。如何筛选、如何覆盖，才是真正的研究问题。

### Regularization 变成非回归约束

[Elastic Weight Consolidation（EWC）](https://arxiv.org/abs/1612.00796) 会保护那些对旧任务重要的参数；[Learning without Forgetting](https://arxiv.org/abs/1606.09282) 则走互补路线：它不保护参数，而是约束*函数*，让新模型在相同输入上的输出贴近旧模型。这个区分可以直接迁移到 agent 上——你想保住的几乎总是行为，而不是某个具体参数。在 agent 中，同一个思想可以约束：

- adapter/weight 在受保护能力上的变化；
- 新模型相对旧模型在 replay set 上的偏离；
- prompt、路由策略或工具 contract 的变化；
- 安全与校准行为；
- 面向受保护用户或组织规则的 memory retrieval。

表面越异质，就越应该从**行为约束**出发思考，而不是只在 weights 上加一项 penalty。

### Isolation 变成 namespace、adapter 与 expert

参数隔离方法为不同任务分配不同容量；[PackNet](https://arxiv.org/abs/1711.05769) 是其中的代表：通过反复剪枝并冻结任务专属子集，把多个任务打包进同一个网络。Agent 已经天然拥有这些版本：

- 每个用户或组织独立的 memory namespace；
- 项目专属 skill；
- LoRA adapter 或 task vector；
- mixture-of-experts 路由；
- 沙箱化工具与 subagent。

Isolation 能减少干扰、改善删除能力，却也制造了路由问题，并可能阻碍正迁移。最难的是判断哪些知识真正应该共享。

### Dynamic architecture 扩展到神经元之外

[Progressive network](https://arxiv.org/abs/1606.04671) 与其他扩展方法通过增加容量、而不是覆盖旧容量来学习：它们为既有任务保留冻结的列，并学习通向这些列的横向连接。一个 agent 可以增加：

- memory 模块；
- 专家 subagent；
- 新工具或 validator；
- 新 expert/adapter；
- 工作流中的一条新分支。

这样做让增长变得容易，却让治理变得困难。如果没有 consolidation（巩固）与 pruning，系统就会积累互相矛盾的 memory、彼此重叠的 skill 与脆弱的工具。终身学习不仅要学会保留，也要学会*有意遗忘*。

### Distillation 连接不同的时钟

Distillation（蒸馏）是整个栈中最重要的桥：

- trajectory → episode 摘要；
- 多个 episode → semantic 或 procedural memory；
- memory → 经过验证的 skill；
- 长 context 的“资深 agent” → 新 session 中的学生；
- 专用 adapter 或 fleet learner → 共享基座模型。

[Self-Distillation Fine-Tuning（SDFT）](https://arxiv.org/abs/2601.19897) 把当前模型中以 demonstration 为条件的版本作为 teacher，把只看到 query 的版本作为 student，由此构造相对于学习器的 on-policy 信号。更一般地说，distillation 能把快速、显式的学习变成缓慢、隐式的能力，而无需永远保留每一条原始经验。

### Meta-learning 优化 updater

经典的“learning to learn”问题，在 agent 中变得非常具体：我们能否优化这样一套策略——它把 trajectory 转成 reflection、memory、prompt 编辑、训练数据或参数更新？

[Meta-TTL](https://arxiv.org/abs/2604.00830) 把一项自然语言 test-time adaptation 策略，作为外层演化搜索的对象。它先跨训练任务学习 agent 应该如何使用先前 episode，再把这项适应策略冻结到未见任务上。这不是 RSI——外层目标与搜索仍由人设计且边界明确——但它恰好就是 updater 表面 $$U$$。

### 哪些经典假设不再成立

传统实验设定通常会直接提供：

- 清晰的任务边界；
- task ID；
- 即时 label；
- 固定的数据生成过程；
- 一个可变模型；
- 一套独立于部署的离线测试集。

已部署 agent 不会免费得到其中任何一项。任务彼此重叠，反馈延迟且隐式；策略会改变自己看到的数据；用户与工具会变化；memory、harness 与 weights 会共同适应；隐私、投毒与回滚本身就是学习问题的一部分。

因此，简单地把 EWC 或 replay 套在 LLM 上并不是答案。经典 CL 提供机制；agentic CL 改变的是**系统边界、证据模型与 release 流程**。

---

## 当下哪些方法有效

在本文调查的公开系统中，还没有一个能够同时跨越 context、memory、harness、weights 与 updater，展示开放式持续学习。但一些重要切片已经可以工作。描绘这个领域最清楚的方式，不是把论文分成“memory 论文”和“self-improvement 论文”，而是看**每个系统把经验写到哪里，以及哪一道门控制它的晋升**。

| 系统 | 写入对象 | 晋升门 | 报告证据 | 主要 caveat |
|---|---|---|---|---|
| [CER](https://arxiv.org/abs/2506.06698) | context buffer | 检索相关性 | WebArena 成功率 36.7%，相对提升 51.0% | 长期来看 buffer 质量缺乏治理 |
| [ARIA](https://arxiv.org/abs/2507.17131) | 带时间戳的 memory | 人类专家回答 | 作者报告的 TikTok Pay 部署 | 规模与效果均无独立审计 |
| [CASCADE](https://arxiv.org/abs/2605.06702) | case bank + retriever 参数 | 仅保留成功案例；bandit 负责检索 | 16 项任务上相对 zero-shot +20.9% | 保证覆盖案例选择，不覆盖开放世界 |
| [Replit](https://replit.com/blog/evaluating-and-improving-agent-at-scale) | agent 产品/harness | 离线 eval + A/B + 人工复核 | 第一方工程报告 | 属利益相关方证据 |
| [OpenAI Tax AI](https://openai.com/index/building-self-improving-tax-agents-with-codex/) | 受限 worktree 与 eval | 定向 + 回归 eval，人工合入 | 第一方案例研究 | “自我改进”指循环自动化 |
| [Continual Harness](https://arxiv.org/abs/2605.09998) | prompt、skill、memory、weights | 无独立于该运行的门 | 无重置的宝可梦运行 | 局限于有界游戏设定 |
| [Cursor Tab](https://cursor.com/blog/tab-rl) | 共享 weights | 聚合后的 online RL，带版本发布 | 建议少 21%，接受率高 28% | 接受率只是价值的代理 |
| [SEAL](https://arxiv.org/abs/2506.10943) | 通过 self-edit 改 weights | 外层 RL 以更新后表现为奖励 | SQuAD 无原文 39.7% → 47.0% | 连续 self-edit 下早期任务会退化 |
| [Meta-TTL](https://arxiv.org/abs/2604.00830) | 适应策略本身 | 外层演化搜索，随后冻结 | Jericho ID 50.4 → 110.8；τ²-bench OOD 0.33 → 0.37 | 外循环离线且有界 |
| [SDFT](https://arxiv.org/abs/2601.19897) | weights | 无独立于该方法的门 | 受控顺序技能保持实验 | 尚非部署规模系统 |
| [DGM](https://arxiv.org/abs/2505.22954) | 自身代码 archive | 由 benchmark 选择的 archive | 在 200 个任务的 SWE-bench Verified 子集上 20.0% → 50.0% | 冻结模型、固定 benchmark |

### Context 与 memory：不动基座模型也能复用经验

[CER](https://arxiv.org/abs/2506.06698) 是快路径的一个干净例子。Web agent 完成任务后，从 trajectory 中蒸馏环境动态与决策模式，将其合并进动态 buffer，为下一项任务检索相关经验，再在 context 中 replay。它支持离线、在线与混合来源的 trajectory。在论文的 WebArena 设定上，CER 报告了 36.7% 的平均成功率——相对 GPT-4o baseline 提升 51.0%——以及 VisualWebArena 上的 31.9%。

这是有意义的证据，说明 inference-time 经验无需改变 weights，也可以改善 agent 此后的行为。但它没有证明经验能够无限累积。结果取决于 distillation 质量、retriever 覆盖率、环境相似性，以及 buffer 不被误导性经验填满。

[ARIA](https://arxiv.org/abs/2507.17131) 处理的是另一个弱点：领域规则变化时，反馈不会自动到来。Agent 通过结构化 self-dialogue 识别不确定性，向专家提出有针对性的问题，并维护带时间戳、支持冲突检测的知识。论文称 ARIA 已部署在 TikTok Pay 内部，并称 TikTok Pay 服务超过 1.5 亿月活用户；但它没有报告 ARIA 自身处理了多少用户或决策。部署与平台规模都应视为作者报告；不过其架构很有价值：**不确定性触发一次有明确 scope 的人类询问；答案进入能够感知冲突的 memory，而不是全局 weights**。

[CASCADE](https://arxiv.org/abs/2605.06702) 把案例复用做得更形式化。它把部署时学习视为 pretraining 与 fine-tuning 之后的第三个生命周期阶段，冻结 foundation model，再用 contextual bandit 学习应检索哪些 episodic case。系统只保留来自正奖励交互的 case。这里需要说准确：所谓“不更新参数”指的是*基座模型*；辅助 retriever 是在线用梯度下降训练的，因此 CASCADE 更适合被描述为“冻结 LLM 的学习”，而不是“无梯度学习”。仅保留成功案例也是它暴露的软肋——一个因为错误原因而获得奖励的答案，会变成被信任的范例。在论文的 16 项任务套件上，它报告的宏平均成功率相对 zero-shot prompting 提高 **20.9%**。它的 no-regret 分析只适用于所形式化的案例选择过程及论文明确列出的假设，并不适用于任意开放世界学习；但它说明了，当 memory 研究拥有一个明确的长期目标后，可以得到什么。

这些系统合在一起，澄清了三点：

1. memory 可以在基座模型保持冻结的情况下，产生真实的跨 episode 适应；
2. memory 的质量取决于**写入、修订、检索与遗忘**，而不只是 retrieval；
3. 快速学习仍然是局部且显式的，因此也更容易检查与撤销。

### Harness：把生产故障变成带版本的改动

暖路径已经出现在生产工程中。

Replit 的官方报告 [“Closing the loop: Evaluating and improving Replit Agent at scale”](https://replit.com/blog/evaluating-and-improving-agent-at-scale) 描述了两个度量支柱与一个优化循环。离线 benchmark 在发布前测试候选版本；A/B test 与生产 trace 告诉团队发布后究竟发生了什么；失败簇变成假设，再由 agent 将其实现为 pull request 草稿。候选版本会与 benchmark、trajectory 数据、A/B 证据和近期 baseline 对比，然后系统给出 ship/iterate/drop 建议。工程师保留 release 权限。这是持久化 self-improvement，但被持久改进的是**agent 产品与 harness**，不是在线变化的基座模型。

OpenAI 的 [Tax AI 案例研究](https://openai.com/index/building-self-improving-tax-agents-with-codex/) 描述了一种相似模式，而且边界划得格外清楚。经过人工审阅的生产发现、源 trace、预期输出、税务引擎文档与 eval 命令，被打包成一项有边界的 Codex 任务。候选 agent 只能写入一个有明确 scope、包含定向 eval 与回归 eval 的 worktree；生产证据保持只读。验证与人工复核始终是这个任务环境的一部分。这里的 *self-improving* 是说“自动化闭合了工程循环中的更多环节”，而不是“模型自主重写自身 weights”。

[Continual Harness](https://arxiv.org/abs/2605.09998) 是具身领域里的研究对应物。一个 refiner 读取近期 Pokémon trajectory，在一次不重置的运行中编辑 system prompt、subagent、skill 与 memory。第二项实验还通过 process-reward relabeling 与 soft supervised fine-tuning 更新一个开放模型。这篇论文的价值恰恰在于，它把仅更新 harness 与同时更新 harness、weight 的学习放进同一套拓扑。它的有界游戏设定与论文专属模型结果，不应被泛化到任意部署场景；但这个架构让共同演化变得具体。

### Weights：窄信号可以支撑快速冷路径 release

在本文的来源集合中，Cursor 是高频 weight 更新最清晰的生产案例。

在[“用 online RL（在线强化学习）改进 Cursor Tab”](https://cursor.com/blog/tab-rl)中，动作与反馈都非常窄：展示或抑制下一处编辑建议；观察用户接受还是拒绝。服务处理的交互量足以在新鲜的 policy data 上训练。Cursor 报告称，最终模型少展示了 21% 的建议，但展示出来的建议接受率提高了 28%。较新的 [Composer 实时 RL 报告](https://cursor.com/blog/real-time-rl-for-composer)描述了更广泛的生产交互奖励信号，以及一条最快每五小时就能发布新 checkpoint 的流水线。

这是快速冷路径存在性的证明，却不能证明任意反馈都应该更新 weights。这里的信号频繁，动作有完善的 instrumentation，而且会先聚合，再发布带版本的 checkpoint。接受仍然只是一项 proxy：它无法完整衡量正确性、可维护性或下游事故。

研究方法则正在直接攻克更新本身：

- [SEAL](https://arxiv.org/abs/2506.10943) 让模型生成一次“self-edit”：合成训练数据，以及可选的优化指令。Supervised fine-tuning 负责施加持久更新；外层 RL 循环则根据更新后的性能奖励 self-edit。在单 passage 的知识纳入实验中，不提供 passage 时的 SQuAD 准确率为：冻结基座 32.7%，用未经训练的策略生成 self-edit 为 39.7%，而学会生成 self-edit 之后提升到 47.0%。论文同时指出，随着 self-edit 顺序累加，早期任务会退化——因此它证明的是持久性，而不是非回归。
- [SDFT](https://arxiv.org/abs/2601.19897) 把专家 demonstration 转成相对于学习器的 on-policy distillation，减轻普通 supervised fine-tuning 的 off-policy mismatch，并在受控的连续 skill 实验中更好地保留旧能力。
- [Agent-Dice](https://arxiv.org/abs/2601.03641) 假设任务专属参数更新已经存在，然后在融合前过滤相互冲突的方向、放大共享方向。它为“哪些更新分量是共通的？”给出了一个有用的冷路径答案，却没有解决反馈收集或因果归因。

它们都超越了“在最新数据上 fine-tune”。但还没有一种方法同时具备生产级 provenance、延迟反馈、个人/fleet scope、持续对齐与长期可塑性。

### Updater：learning to learn 的最初拼图

[SEAL](https://arxiv.org/abs/2506.10943) 的外循环学习内循环应当生成什么训练数据。[Meta-TTL](https://arxiv.org/abs/2604.00830) 学习一套跨 test-time episode 进行适应的自然语言策略。[Darwin Gödel Machine（DGM）](https://arxiv.org/abs/2505.22954) 则走了另一条路：coding agent 编辑自己的 harness/代码。能够编译并保留代码编辑能力的变体会接受 benchmark 评测并进入 archive；结合性能与探索不足程度的 parent selection，让后续 agent 能从这些 stepping stone 出发继续构建。

这些方法很重要，因为可变对象已经部分变成了**改进机制**。但它们仍然边界明确：

- SEAL 的 reward 与任务设定由外部指定。
- Meta-TTL 的外层演化循环在离线阶段训练，并在 test time 冻结。
- DGM 使用固定的 coding benchmark 与冻结的 foundation model；其 archive 维护与 parent-selection 过程本身也是固定的，foundation-model/训练脚本演化仍属未来工作，实验则采用 sandbox 与人工监督。

所以，诚实的 2026 年成绩单是：

> 我们已经能让 agent 复用经验、维护不断变化的领域知识、改进带版本的 harness、在狭窄生产信号下频繁更新 weights，并学习有边界的适应策略。但我们还没有一个可信系统，能够在延迟、开放世界反馈下，在所有这些表面之间做决定，并无限期地改进这项决策过程本身。

---

## 值得命名的失效模式

大多数持续学习系统不会以显眼的方式崩溃。它们会以“一度看起来像成功”的方式缓慢劣化——正因如此，这些失效模式值得被命名。

**Memory 膨胀。** 每条经验都被写入，没有任何东西过期，检索质量随存储增长而下降。系统看起来在学习——memory 条数在上升——任务表现却在向下漂移。[LifelongAgentBench](https://arxiv.org/abs/2505.11942) 给出了可测版本：加入更多历史反而降低成功率。解决办法并不光鲜：把失效日期、去重与剪枝当作一等操作，而不是维护杂务。

**偏好越权扩散（preference laundering）。** 某一位用户的风格偏好被写到了全局 scope，此后它与一条经过验证的规范再也无法区分——因为两者都只是同一个存储里的条目。Scope 错误比事实错误更难发现，因为内容本身没有任何毛病。

**代理指标俘获。** 系统去优化它能测到的信号——接受率、点赞、测试通过——并逐渐偏离所有人真正在意的结果。最清楚的例子就是接受率：它是行为代理，并不能完整衡量正确性、可维护性或下游事故。防御办法是保留至少一个更慢、独立、且学习循环无法直接优化的结果指标。

**自我确认的证据。** 同一个模型提出更新、评估更新，并撰写“它为什么有效”的说明，于是整个循环根本没有独立证据。这是“self-improvement”结果最终被证明只是对触发案例过拟合的最常见方式。

**无声投毒。** 任务过程中读到的不可信内容变成了持久 memory，此后攻击者的文本就会以可信 context 的身份到达。危险之处在于持久性：一次性注入变成了长期指令。

**耦合 Goodhart。** memory、harness 与 weights 都在针对同一套 benchmark 更新，于是一个表面引入的回归被另一个表面打上补丁。总分保持平稳甚至上升，系统却变得更脆。只有逐表面消融才能看出来。

**安全遗忘。** 能力更新保住了任务准确率，却侵蚀了拒答边界、校准或弃权行为。由于多数回归套件对能力赋予更高权重，这类劣化可以通过所有实际在跑的门。

**可塑性衰减。** 经过多轮更新后，系统保住了旧行为，学习新行为的能力却越来越差——这正是系统层面的 loss-of-plasticity 失败（[Dohare et al., 2024](https://arxiv.org/abs/2306.13812)）。任何只度量保持能力的评测都看不见它。

> **诊断方式。** 对上面每一种失效，问一句：在总分发生变化*之前*，什么度量本可以抓住它？如果现有看板回答不了这个问题，那么学习循环就是在缺少这道安全网的情况下运行的。

---

## 从持续学习到自我改进与 RSI

持续学习与 RSI 经常被一句话连在一起：*如果模型不断学习，最终它就能改进自身*。这句话跳过了最难的部分。复利式改进有不同层级。

![从有边界的 refinement 通向继任者构建型 RSI 的阶梯。](/assets/img/blog/continual-learning-for-next-generation-agents/fig4_continual_learning_to_rsi.png)
*图 4. 持续学习可以提供持久状态与复利效应。当更新机制本身成为更新对象时，自指性已经开始；而继任者构建型 RSI 还需要开放式验证、方向设定、资源、权限与治理。*

| 阶梯 | 改变什么 | 什么被保留 | 代表工作 | 仍然缺什么 |
|---|---|---|---|---|
| 1. 有边界的 refinement | 单个答案或工件 | 重置后一无所留 | 自我批评与修复循环 | 任何持久状态 |
| 2. 持久的 self-improvement | memory、skill、harness、weights | 跨 session 与 release | CER、ARIA、Replit、Cursor、SDFT | 更新能够泛化的证据 |
| 3. 学会如何学习 | 更新策略本身 | 跨学习周期 | SEAL、Meta-TTL | 一个并非外部给定的目标 |
| 4. 继任者构建型 RSI | 研究与构建循环 | 被继任者继承 | DGM 的拓扑（限于有界领域） | 开放式验证、品味、权限 |

### 第 1 阶——有边界的 refinement

系统重试、reflection、critique，或编辑一项工件。[Self-Refine](https://arxiv.org/abs/2303.17651) 与许多 coding-agent 修复循环都位于这一层。结果可能远好于第一次尝试，但只要重置 context，学习器也随之重置。

### 第 2 阶——持久化 self-improvement

一项更新得以保留，并改善此后的任务：它可以是一条 memory、一个 skill、一项 harness patch、一个 adapter 或一个 checkpoint。[CER](https://arxiv.org/abs/2506.06698)、[ARIA](https://arxiv.org/abs/2507.17131)、[Replit](https://replit.com/blog/evaluating-and-improving-agent-at-scale) 的循环、Cursor 的 RL 流水线与 [SDFT](https://arxiv.org/abs/2601.19897)，分别位于这一阶的不同位置。持续学习从这里开始产生复利。

持久化是必要条件，却不是充分条件。更新必须能够迁移，且避免回归。一个不断增长、却让无关任务变差的 memory，只是持久变化，而不是有用的持续学习。

### 第 3 阶——learning to learn

系统改进自己把经验转成更新的方式：retrieval policy、update router、evaluator、curriculum、self-edit generator，或 adaptation prompt。[Meta-TTL](https://arxiv.org/abs/2604.00830) 与 [SEAL](https://arxiv.org/abs/2506.10943) 直接呈现了这一层。

此时，改进速度本身可以改变。但外层目标、任务分布与权限通常仍由人类固定。对一个固定 benchmark 做出更好的 optimizer，属于 meta-learning，并不会自动成为 RSI。

### 第 4 阶——继任者构建型 RSI

系统能够改进那套开展研究并构建继任者的机器：提出有价值的问题、设计实验、修改训练代码与架构、评测结果、分配资源，再部署一个能够重复此过程的更好学习器。

[DGM](https://arxiv.org/abs/2505.22954) 具有递归*拓扑*：coding agent 编辑一份代码，而这份代码会帮助它在未来继续编辑代码。论文报告了大幅 benchmark 增益——在一项含 200 个任务的 SWE-bench Verified 评测上从 20.0% 提升到 50.0%，在完整 Polyglot benchmark 上从 14.2% 提升到 30.7%——并用 archive 让未来变体复用 stepping stone。但它的范围仍是围绕冻结模型、由 benchmark 选择的 coding harness。把它称作“完整 RSI”，会抹掉以下尚未解决的条件：

- **验证：** 开放式研究很少拥有真正覆盖目标的测试套件；
- **因果归因：** 一项研究成果可能依赖持续数月、彼此作用的改动；
- **研究品味：** 选择一个后果重大的方向，不同于优化别人给出的方向；
- **迁移：** benchmark 改进可能过拟合 evaluator；
- **资源与权限：** 实验需要计算、数据、部署与组织决策；
- **持续对齐：** 学习器的目标与安全属性必须在能力更新后继续存在。

这也让前文转述的梁文锋主张更加清晰。持续学习可以让 AI 研究 agent 积累组织经验、加快研发，从而让 self-improvement 更实际。但它并不在逻辑上保证一个失控加速的循环，因为 evaluator、方向设定者与部署权限仍可能成为瓶颈。

> **区别。** 持续学习讨论的是改进的**时间轴**；RSI 讨论的是改进图的**自指性与自主性**。持续学习可以在没有 RSI 的情况下存在；RSI 需要持久复利，但这种持久性也可能存在于带版本的外部工件中，而不是传统上被称为持续学习的机制里。

---

## 如何评估复利式学习

把答案放进 memory 后只测一次得分，并不能衡量持续学习。最小评测单元必须是**一条数据流，加上学习器状态，再加上向未来的迁移**。

在时刻 $$t$$，先让 agent 作答，之后才提供规则允许的反馈并更新其状态。这种 prequential protocol（预测式序贯协议）可以防止学习器使用正在被评分的 label。

一个简单的增益指标，可以把学习效果从模型原始能力中分离出来：

$$G_t = \mathrm{Score}(A_t, \mathcal{T}_t) - \mathrm{Score}(A_0, \mathcal{T}_t),$$

其中，$$A_0$$ 是重置后的 agent，$$\mathcal{T}_t$$ 是未来或 held-out 任务集。还有两个经典量能把画面补完整。记 $$s_{i,j}$$ 为学习到第 $$i$$ 步之后、在任务族 $$j$$ 上的得分。**Backward transfer（后向迁移）**比较一个任务族在整条流末尾的得分与它刚被学会时的得分，$$\mathrm{BWT}_j = s_{T,j} - s_{j,j}$$，负值即遗忘；**Forward transfer（前向迁移）**比较“尚未在该任务族上训练时”的表现与重置 agent 的表现，$$\mathrm{FWT}_j = s_{j-1,j} - s_{0,j}$$，刻画早期经验是否提前带来了帮助。

这里沿用标准约定：任务族 $$j$$ 就是在第 $$j$$ 步被学到的那一族。同时报告这两个量之所以重要，是因为它们彼此权衡：什么都不写的系统，遗忘为零，前向迁移也为零；什么都写的系统，通常要用别处的回归来换前向迁移。只有一条曲线还不够。一份可信的 scorecard 至少需要回答：

### 性能能否持续复利？

- 数据流上的累计 reward 或 utility；
- 相对于合适 oracle 的 online regret；
- 改进斜率，以及它是否饱和；
- 在固定 context/memory/compute 预算下的性能。

### Agent 的适应速度有多快？

- 达到目标所需的交互次数、token、人类纠正次数与 wall-clock time；
- zero-shot 与得到反馈后的增益之差；
- concept drift 后的恢复时间。

### 新旧能力发生了什么？

- backward transfer 与最坏情况遗忘；
- 面向相关未来任务的 forward transfer；
- 面向 held-out 任务族的迁移，而不只是同一家族里的新样本；
- 长更新序列中的可塑性丧失。

### 路由本身可信吗？

- 表面/scope 决策的准确率与校准；
- 错误晋升率：把不确定或私有经验发送到过宽范围；
- 错误拒绝率：丢弃有用经验；
- 隔离条目的解决率与回滚率。

### 学习付出了什么代价？

- inference、存储、训练与评测计算；
- memory 增长与 retrieval 延迟；
- 人工审阅负担；
- release 延迟与运行回滚成本。

### 对齐与可靠性保住了吗？

- 分开报告能力、policy 与安全回归；
- 更新后的校准与 abstention；
- 隐私泄漏、删除合规与抗投毒能力；
- 最差群体与尾部风险的变化，而不只看平均 reward。

近期 benchmark 覆盖了彼此互补的不同切面：

- [StreamBench](https://arxiv.org/abs/2406.08747) 把输入—反馈序列显式化，并允许更新 prompt、retriever、memory 或 weights；不过，其公开数据集并没有严格控制跨任务复用。
- [LifelongAgentBench](https://arxiv.org/abs/2505.11942) 提供了 1,396 项彼此依赖、可执行的任务，横跨数据库、操作系统与知识图谱环境。它的 replay 结果表明，历史越多，表现可能越差。
- [MemoryBench](https://arxiv.org/abs/2510.17281) 模拟服务期间的显式与隐式反馈，评测系统是否构建 declarative 与 procedural memory，而不只是检索预先装载的历史。
- [Evo-Memory](https://arxiv.org/abs/2511.20857) 在顺序数据流上比较不同 memory 模块，并把对话回忆与可复用经验区分开。
- [CL-Bench](https://arxiv.org/abs/2606.05661) 构建了六个经专家验证的领域，其中包含隐藏的可复用结构，部分领域还存在 concept drift。在论文报告的实验里，朴素 in-context learning 胜过专门的 memory 系统——这提醒我们，加装 memory plumbing 不等于学习。
- [AgentCL](https://arxiv.org/abs/2606.02461) 对比任意数据流与受控的组合式数据流，并分别估计 plasticity、stability 与 generalization gain。它的核心发现是方法论层面的：朴素数据流会压缩不同 memory 设计之间的差异，而 held-out case 则会揭示 memory 引发的性能退化。

下一代 benchmark 应当要求学习器做**路由**，而不是预先替它假定答案。给学习器一项私有偏好、一条即将过期的规定、一套可复用流程，以及一项能够广泛迁移的 skill；让它自己选择 context、memory、harness 或 weights；然后从任务 utility、scope、删除、迁移与回归几个方面评分。在 benchmark 开始评测这个决定之前，每种方法都只是在自己设计者预先选定的表面上竞争。

> **最低可信协议。** 在反馈前作出预测；快照保存完整 agent 状态；与 reset baseline 和 unlimited-context oracle 对比；留出未来任务族；注入陈旧、冲突、延迟与被投毒的反馈；并对发生变化的更新表面做消融。

因此，一篇持续学习 agent 的结果表至少应当报告：整条流上的累计效用；相对 reset baseline 在 held-out 任务族上的增益；受保护能力上的后向迁移；以交互次数与人工纠正衡量的适应成本；memory 或参数的增长；以及事后被撤销的晋升次数。只报告其中第一项的工作，测到的是“有东西发生了变化”，而不是“agent 学会了”。

---

## 研究议程

多时间尺度视角自然引出一批现在就足够具体、可以动手实现的项目。

### 1. 校准更新 router

**问题：** 学习器能否判断一条经验属于哪里，并在自己不确定时意识到这一点？

构建一批正确路由各不相同的数据流：

- 一项只适用于一次 session 的约束；
- 一项用户偏好；
- 一条带失效日期的项目事实；
- 一套能够跨用户迁移的流程；
- 一项 fleet-wide skill；
- 恶意或自相矛盾的反馈。

Router 输出表面、scope、TTL、action 与 confidence。比较手写 policy、LLM router、基于 retrieval 的分类器与学习得到的决策 policy。评测期望 utility、路由校准、隐私违规、错误晋升、迁移与回滚。

这以一种单看准确率无法做到的方式，把持续学习与 uncertainty quantification 连了起来：不确定的 router 应当选择可逆写入或请求证据，而不是假装每条经验都有一个全局 label。

**成功判据：** 学到的 router 在累计效用上胜过所有固定策略，*并且*错误晋升率低于最激进的那条固定策略。只在效用上赢过“永远写 memory”还不够，因为真正有意思的主张是“知道什么时候不该写”。

### 2. 从 session 到 weight 的巩固

**问题：** 一名经历过长 session 的“资深” agent，如何教会刚重置 session 的自己？

让 agent 用许多个 episode 学习一个新 codebase、研究领域或工具环境。最后重置它的 context，并比较：

1. full-history prompting；
2. 检索到的 episodic memory；
3. 蒸馏后的 semantic/procedural memory；
4. 一项可执行 skill；
5. 一个 LoRA adapter；
6. [SDFT](https://arxiv.org/abs/2601.19897) 或另一种 learner-relative distillation 方法。

衡量未来任务迁移、遗忘、校准漂移、inference 成本，以及删除某一条源经验的能力。最关键的消融不只是比较哪种方法胜出，而是找出**哪些经验原本就绝不该被巩固**。

**成功判据：** 经过巩固的新 session，在未来任务上追平全历史 prompting，而 context 成本只有其一小部分，且不回退受保护能力。**可预见的障碍：** 巩固目标由那个自身带有错误的模型写出，因此“教师错误传播”需要单独度量。

### 3. 可信的 online agent learning

**问题：** 当反馈延迟、隐式、由 agent 自身策略筛选，或带有对抗性时，agent 还能学习吗？

构建一个带版本、因果结构已知的模拟器，其中包含：

- 作为 noisy proxy 的即时用户反应；
- 延迟出现的客观结果；
- 不断变化的用户与环境；
- 被投毒的反馈；
- 只适用于某个 scope 的纠正；
- 撤销与 right-to-forget 请求。

联合推断可信度与因果归因。比较朴素 aggregation、inverse-propensity correction、causal model 与 uncertainty-aware quarantine。不要只报告最终 reward，还要报告恢复时间与错误晋升。

**成功判据：** 随着被投毒比例上升，性能优雅退化，且漂移事件后的恢复时间保持有界。一个在干净反馈下取胜、在 5% 投毒时崩溃的方法，并没有解决部署问题。

### 4. Memory、harness 与 weights 的共同演化

**问题：** 多个表面何时彼此协作，又会在何时掩盖彼此的回归？

现有工作大多只改变一个对象。可以构建一项 factorial study：

- 仅 memory；
- 仅 harness；
- 仅 adapter；
- memory + harness；
- memory + adapter；
- harness + adapter；
- 三者全部。

使用 held-out 任务，并为 utility、成本、安全、校准与可逆性设置 Pareto 门。对每次接受的更新运行因果消融或 Shapley-style attribution。一个很可能出现的结果是，不同时钟彼此互补；但联合优化也会产生耦合的 Goodhart effect：一个表面负责修补另一个表面制造的 benchmark artifact。

**成功判据：** 组合系统在 held-out 任务族上胜过最好的单一表面，且在消融其他表面时，每个表面的贡献仍为正。如果移除某个表面反而让结果*变好*，那么这套组合掩盖的是回归，而不是在积累复利。

### 5. 从个人到 fleet 的持续学习

**问题：** 本地 agent 如何分享可迁移的经验教训，同时又不泄漏私有或相互矛盾的经验？

建模一个拥有用户、项目、组织与全局 namespace 的 fleet。有些潜在规则能够共享，另一些则彼此冲突。可以研究：

- selective/federated distillation；
- 保留 provenance 的 aggregation；
- consensus 与 conflict resolution；
- 各 scope 独立的 adapter 与 expert routing；
- machine unlearning 与删除操作的传播；
- 面向提供有价值反馈的用户的 incentive。

正确的产物不是一套全局通用 memory，而是一个层级：一条经验只有先证明自己能够泛化，才可以跨越行政边界。

**成功判据：** fleet 级学习能让毫无贡献的新租户也获得提升，同时一次删除请求可被证明地从所有派生工件中移除该贡献者的影响。后半句才是真正的难点，也正是朴素的“把所有数据拿去 fine-tune”不可用的原因。

### 6. 持续对齐

**问题：** 当能力增长时，安全、诚实与经过校准的 abstention 能否保持稳定？

把能力任务与不断变化的 policy 约束、对抗性反馈交错在一起。分别追踪 helpfulness、harmlessness、honesty、calibration 与 policy compliance 的 backward transfer。检验那些能够保住任务准确率的 replay、regularization、isolation 与 distillation，是否同样能保住拒答边界与不确定性行为。

安全遗忘不只是又一次 benchmark 回归。一个学习器如果越来越善于行动，却越来越不知道何时不该行动，那么它走错了方向。

**成功判据：** 在一长串更新之后，拒答边界与校准在 held-out 安全套件上与更新前模型统计上不可区分，同时能力有所提升。任何只报告这句话中“能力”那一半的方法，都没有回答真正重要的问题。

### 一个起步实验

一个有用的起步项目不需要 frontier model。构建一个研究或 coding agent，配备：

- 普通 context；
- 按 namespace 隔离、provenance 丰富的 memory；
- 置于版本控制下、可编辑的 Markdown skill；
- 一个 LoRA adapter；
- 一个每次任务结束后可以选择某个表面、隔离该经验，或不做更新的 update router；
- 一套隐藏的回归与未来迁移测试。

把相关与不相关任务组成数据流，并混入显式、隐式、陈旧与矛盾反馈。衡量累计 utility、适应速度、迁移、遗忘、路由校准、memory 增长、人工审阅成本与回滚。Baseline 是一组固定路由 policy：始终写入 context、始终写入 memory、始终写入 skill、始终 fine-tune，以及始终不更新。

这项实验会直接检验本文的论点：**学习算法的一部分，正是那套决定下一步要成为哪种学习器的 policy**。

---

## 总结

今天的 agent 会产生丰富经验，却很少让经验形成复利。经典持续学习准确地识别了稳定性—可塑性困境，以及 replay、regularization、isolation、expansion、distillation 与 meta-learning。Agent 时代改变了分析单元：模型如今只是一套系统中的一个组件，而系统拥有多个可变表面与多个时钟。

一套实用架构是：

1. 捕获带版本的 trajectory 与结果；
2. 验证反馈并归因；
3. 按正确 scope，把经验路由到 context、memory、harness、weights 或 updater；
4. replay、对照并巩固；
5. 要求局部修复、held-out 迁移与非回归；
6. 分阶段 release，并保留回滚能力；
7. 监控结果，并最终改进 updater 本身。

这也澄清了从持续学习通往 self-improvement 的道路。持久 memory 或 weights 可以让能力形成复利；学习 updater 会改变改进速度；继任者构建型 RSI 还需要更多条件：诚实的开放式验证、因果归因、研究品味、方向设定、资源、权限与对齐。

回到那位新研究助理。我们不会期待一名人类员工在每次听到意见后，都重写自己的整个大脑。他们会记工作笔记、形成 memory、采用流程、内化反复出现的经验教训，并越来越善于决定如何学习。下一代 agent 需要与之对应的计算机制——每一层都运行在适合自己的时钟上。

> **要点。** 下一代 agent 的关键，与其说是它们部署后*能否*学习，不如说是它们能否判断：**什么应该学习、应该写到哪里，以及何时可以安全晋升**。

---

*来源说明：梁文锋据报道在 2026 年 5 月发表的言论，来自 2026 年 7 月《每日经济新闻》的一篇报道；该报道所依据的是一份流传中的 42 页 AI 整理版记录。该媒体称，一家参与投资 DeepSeek 的机构确认了 5 月闭门会的存在，并认为内容真实可信；但目前没有公开的原始录音、带说话人标注的官方记录，或 DeepSeek 的确认。因此，本文把它视为一份媒体报道、投资机构佐证的非官方记录，只将这些言论用作动机，而不是技术证据。文中其他公司工程指标均来自第一方报告；近期 2026 年预印本被表述为各论文自身的证据，而非已经尘埃落定的结果。所有图片均为原创。*

---

## 如何引用

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

## 参考文献

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
