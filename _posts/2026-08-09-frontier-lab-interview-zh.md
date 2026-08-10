---
layout: post
title: "Three Gates and Seven Buckets: What Frontier Labs Actually Test (中文版)"
date: 2026-08-09 10:00:00
author: Jiaxin Zhang
description: "Frontier lab 的面试是三场考试共用一个名字：研究履历让你被看见，技术熟练度让你通过，而第三类没人告诉你要准备的功课决定你最终拿到什么。七个技术方向，附可运行代码。"
tags: interviews careers llm rl systems research
categories: research-notes
giscus_comments: true
related_posts: false
ai_assisted: true
og_image: https://jxzhangjhu.github.io/assets/img/blog/frontier-lab-interview/fig1_three_gates.png
---

<div class="lang-switch"><a href="/blog/2026/frontier-lab-interview/">English</a> · <strong>中文</strong></div>

### 目录

- [让你进门的东西，进门之后就不管用了](#the-thing-that-gets-you-in-stops-working-once-you-are-in)
  - [材料从哪来](#where-this-comes-from)
  - [两个贯穿全文的例子](#two-running-examples)
- [第一道门：被看见](#gate-1-getting-seen)
  - [到底是什么带来了第一次面试](#what-actually-produces-a-first-interview)
  - [信号，以及那些会扣分的信号](#signals-and-the-ones-that-count-against-you)
  - [Cold email](#cold-emails)
- [一轮面试长什么样](#the-shape-of-the-loop)
- [方向一：渐进式系统构建](#bucket-1-progressive-system-building)
  - [四级阶梯](#the-four-level-ladder)
  - [完整例子：把一个 KV store 写四遍](#worked-example-a-key-value-store-four-times)
  - [为什么很强的 ML 候选人会挂在这一轮](#why-strong-ml-candidates-fail-this-round)
  - [题库](#question-bank-system-building)
- [方向二：ML Coding](#bucket-2-ml-coding)
  - [定义这一轮的那个约束](#the-constraint-that-defines-the-round)
  - [Attention，以及藏在里面的四个坑](#attention-and-the-four-traps-inside-it)
  - [手推反向传播](#the-backward-pass-by-hand)
  - [KV cache，以及怎么证明它是对的](#the-kv-cache-and-proving-it-correct)
  - [位置信息：RoPE](#positions-rope)
  - [Online softmax，也就是 FlashAttention](#online-softmax-which-is-flashattention)
  - [采样、归一化、损失](#sampling-normalization-loss)
  - [LoRA](#lora)
  - [Mixture of Experts](#mixture-of-experts)
  - [分词](#tokenization)
  - [Autograd](#autograd)
  - [题库](#question-bank-ml-coding)
- [方向三：AI Fundamentals](#bucket-3-ai-fundamentals)
  - [Attention 与架构](#attention-and-architecture)
  - [归一化、残差与深度](#normalization-residuals-and-depth)
  - [优化](#optimization)
  - [Scaling 与评测](#scaling-and-evaluation)
- [方向四：训练机制与训练方案](#bucket-4-training-schemes-and-mechanisms)
  - [阶梯，以及每一级能修什么](#the-ladder-and-what-each-rung-can-fix)
  - [Reward model](#reward-models)
  - [PPO、GRPO、DPO：KL 到底放在哪](#ppo-grpo-dpo-where-the-kl-lives)
  - [可验证奖励与 reward hacking](#verifiable-rewards-and-reward-hacking)
  - [分布式训练：每种策略切的是什么](#distributed-training-what-each-strategy-shards)
  - [数值精度](#numerics)
  - [调试一次 loss spike](#debugging-a-loss-spike)
  - [题库](#question-bank-training)
- [方向五：ML 系统设计](#bucket-5-ml-systems-design)
  - [Prefill 和 decode 是两台不同的机器](#prefill-and-decode-are-two-different-machines)
  - [完整设计：服务一整个模型家族](#worked-design-serve-a-family-of-models)
  - [题库](#question-bank-systems-design)
- [方向六：Research taste、深挖与价值观](#bucket-6-research-taste-deep-dives-and-values)
  - [Research presentation](#the-research-presentation)
  - [论文轮](#the-paper-round)
  - [项目深挖](#the-project-deep-dive)
  - [价值观轮](#the-values-round)
- [方向七：数学](#bucket-7-math)
- [第三道门：时机、work trial 与谈判](#gate-3-timing-work-trials-and-negotiation)
  - [给你的公司排序](#sequencing-your-companies)
  - [Work trial](#work-trials)
  - [谈判](#negotiation)
- [没人会教你准备的那部分](#the-part-nobody-prepares-for)
- [准备顺序](#sequencing-your-preparation)
- [我可能仍然是错的地方](#what-i-would-still-get-wrong)
- [参考文献](#references)

---

<a id="the-thing-that-gets-you-in-stops-working-once-you-are-in"></a>
## 让你进门的东西，进门之后就不管用了

2026 年有三个人在差不多一个月之内，各自写了自己求职 frontier lab 的复盘。他们做的方向不同、面的公司不同、行文风格也完全不同。但他们说的是同一件事。

Alisa Liu 在华盛顿大学读完六年 NLP 博士，现在是 OpenAI 的 Research Scientist，她说得最精确：

> *"总体来说，技术能力和知识的权重远高于研究经历——尽管后者大概率是让你拿到面试的原因。"* [30]

Silvia Sapora 从 ML 博士进了 Google DeepMind 做 Research Scientist，说得更不客气：

> *"如果你已经在拿面试了：再多发论文对你现在没有任何帮助。你需要的是通过面试，而面你的人往往根本不会看你的简历。所以，别再折腾你的研究和论文了，开始准备面试！"* [42]

而 Yong Zheng-Xin，一位在求职中途从多语言 NLP 转向 AI safety 的 Brown 博士，发现真正起作用的论文只有一两篇——*"有时候一篇都不算，我就是在被评估能不能当场解决这个团队的问题"* [55]。

这就是整件事的结构。它不是一场考试，而是**三场，依次排列**，并且**让你通过前一道门的通货，在下一道门是废纸。**

![三道门，三种通货](/assets/img/blog/frontier-lab-interview/fig1_three_gates.png)
*图 1. 你的研究履历打开门，然后就不再重要。技术熟练度让你通过 loop，而它和你发什么论文没有关系。至于你最终真正拿到什么，由第三样东西决定——而没有人会告诉你要去准备它。*

第二道门——技术 loop——是大多数准备精力的去处，本文的主体也是它，因为它是最有可学内容的部分。但开篇就该说清楚：它是问题的中间三分之一，不是全部。

第二道门内部还有一个结构性事实。当每一轮都在探你在技术栈不同部位的动手深度时，**每一轮都变成了一票否决**。这个 loop 打的是你的**下限**，不是上限。

![两个候选人总分相同，但只有一个拿到 offer](/assets/img/blog/frontier-lab-interview/fig2_soft_spot.png)
*图 2. 合取式评分。四项极强、一项偏弱，通常比七项都还行要糟，哪怕总分完全一样。*

而那个短板很少是随机的。它几乎总是**你自己判定"这个方向对我不重要"的那一个**。Yuan Meng 是一位几乎在所有 onsite 都拿到 offer 的 ML 工程师，她说自己最差的一轮是面向对象的系统构建——也就是 ML 出身的人普遍跳过的那个"纯 SDE coding"方向 [34]。Mimansa Jaiswal 在准备研究岗时，天花板出现在 RLHF——而这是一个研究者会默认自己已经拿下的方向 [21]。Sapora 的版本最锋利：

> *"我知道有些极其出色的研究者，就是因为没准备而在面试里被拒。每天做 ML，和能够从零手写 attention、推导反向传播、写出 flash attention，是两码事。"* [42]

所以，本文的主张是：

> **Frontier lab 的 loop 是三场共用一个名字的考试。你的研究履历让你被看见。技术熟练度——那种在计时器下凭记忆敲得出来的熟练度——让你通过。而第三类不体面的功课，关于时机、谈判和你自己的神经系统，决定你最终拿到什么。只优化中间那一场，是标准错误。**

在技术这一段内部，常见的计划有四块：general coding（通常被打折）、ML coding、AI fundamentals、训练机制。这个计划漏掉了实际被打分的七项里的三项，还把第四项的性质搞错了。

![七个方向，以及四方向计划漏掉了什么](/assets/img/blog/frontier-lab-interview/fig3_seven_buckets.png)
*图 3. 方向一不是 LeetCode。方向五取代了经典系统设计。方向六临时抱佛脚抱不动。而方向七——一轮专门的数学面试——是真实存在的，而且在 Liu 把它列出来之前，我从没在任何地方见人写过。*

下面每一道门一节，每一个方向一节。文章故意写得很长：它的定位是可以做完的东西，不是可以扫一眼的东西。

<a id="where-this-comes-from"></a>
### 材料从哪来

先说一句警告，因为这个话题的污染很严重。

你去搜任何相关关键词，都会淹没在内容农场里——那些卖课或卖工具的站点，每一个都报出同样一批听起来很具体的数字。这些数字彼此吻合得可疑，而这通常是"互相生成"的证据，不是"互相印证"的证据。**我一个都不复述。**如果你见过"CodeSignal 600 分要拿到 520+"、"录取率低于 1%"、或某个精确的薪酬上限，我都没能追溯到任何一手来源，所以下文不会把它们当事实。

我也要承认一个影响了本文早期版本的检索失误。我一开始找的是面试**题目**，于是找到了第二梯队：Meng [34] 和 Jaiswal [21]，两者都确实有用。而本文真正依托的四份材料是**求职复盘**——这是另一个文体，在搜索里的排序完全不同，我第一轮没找到。它们是：

- **Alisa Liu** [30] —— 博士 → OpenAI。**11 家公司、57 场面试、46 通 recruiter 电话、16 次 offer 后沟通。**轮次分类是所有材料里最完整的，谈判那一节也写得最好。
- **Silvia Sapora** [42] —— 博士 → DeepMind。凡是走完流程的公司全部给了 offer。"该练到什么程度"的清单最具体，对情绪代价也异常诚实。
- **Yong Zheng-Xin** [55] —— 博士，求职中途转向 safety。明确是作为前两篇的补充来写的；他的贡献是六个让他意外的地方，其中大部分都在拆"整洁版本"的台。
- **Nathan Lambert** [26][27] —— 唯一一份从**招人那一侧**写的材料，外加他自己 2022 年的博士求职复盘，这套公开时间线的做法似乎就是从他这里开始的。

实验室官方文档只有一份，值得完整读一遍：Anthropic 关于候选人使用 AI 的指引 [2]。核心指令毫不含糊——准备阶段随便用 Claude，但在 live interview 中，*"This is all you–no AI assistance unless we indicate otherwise."*

结构和样本量方面我还用了 Alexey Grigorev 的 AI Engineering Field Guide [13]，这是我找到的唯一一份数据驱动的资源：4,894 份职位描述、51 家公司的面试流程记录，题库里每一道题都标注了它是从哪里被报告出来的。

**关于 X。**这个话题上确实流传着几条转发量很高的 thread；X 现在不登录读不了，所以我没有引用任何我没能亲眼读到的文字。凡是 thread 指向了公开产物的——Gauri Gupta 的优化笔记 [14]、Grigorev 的仓库——我都直接去找了那个产物并引用它。

最后是一条适用于全文的提醒，来自 Yong：**这些轮次远没有任何指南暗示的那么标准化。**他被问过系统设计、`asyncio` 并发，还有专门评估他怎么驱动 AI agent 的轮次。*"永远要预期会有 wildcard 题目和五花八门的轮次"* [55]。把下面的一切当作先验，然后用你的 recruiter 实际告诉你的信息去替换它。

<a id="two-running-examples"></a>
### 两个贯穿全文的例子

全文会反复出现两个对象。

> **E1 —— causal self-attention。**它先以一道 25 分钟的 coding 题出现，然后作为 fundamentals 题、机制题、系统设计题、研究题回来。每次都是同一个对象——这正是重点。
>
> **E2 —— 一个 100B 参数的预训练在第 42,000 步 loss 突然飙了。**它以调试题的形式进入，之后凡是涉及训练基础设施的地方都会再出现。

![同一个对象被问出六种问法](/assets/img/blog/frontier-lab-interview/fig4_one_artifact.png)
*图 4. 准备对象，而不是准备题目。把 self-attention 吃透到六层深，你就同时为六个不同的轮次准备了一部分。*

**Takeaway.** 是三道门，不是一道。中间那道门要按方向的**并集**准备；而第一道和第三道，不要因为没人把它们叫做"考试"就放着不管。

---

<a id="gate-1-getting-seen"></a>
## 第一道门：被看见

在有人能评估你之前，得先有人决定和你聊。这个阶段漏斗最宽、周期最长——而按这四份材料看，也是大多数候选人投入刻意努力最少的地方。

![真实的漏斗长什么样](/assets/img/blog/frontier-lab-interview/fig5_funnel.png)
*图 5. 三份具名材料，各自的真实数字。注意形状：每一个案例里，recruiter 电话和 networking 对话的数量都超过技术轮。漏斗的大部分根本不是技术性的，而它最宽的那一段发生在任何人考你之前。*

从这张图里要拿走两件事。**规模比人们预期的大**——57 场面试不是异常值，而是一次彻底的求职本来的样子，Liu 明确说 *"求职是一份全职工作"* [30]。以及**成分和你猜的不一样**：Liu 在 57 场面试之外还记录了 46 通 recruiter 电话；Lambert 是 46 通 networking 电话对约 53 场面试 [26]。

<a id="what-actually-produces-a-first-interview"></a>
### 到底是什么带来了第一次面试

从候选人这一侧看，诚实的答案是：一个人。Liu：

> *"说点显而易见的：博士期间好好做研究、多交朋友、多合作！要拿到第一次面试，有时候你需要公司内部有人替你背书。你可以早早为此做准备——去会议上社交、广泛合作、参加 networking 活动（当然这对有些人来说并不轻松——对我肯定不是——所以也照顾好自己的精力和舒适度）。"* [30]

最后那个括号很重要。她和 Lambert 都没有把这件事描述成自然或舒服的。Liu 对"重新联系"这件事给了一个宽厚的框架：*"求职很大一部分是在重新联系那些你可能好几年没说过话的人——这没关系，这是预期之内的，而且结果证明这是这个过程一个很美好的副作用"* [30]。

Sapora 给出了我找到的唯一一个明确的简历门槛，然后立刻削弱了它的重要性：*"3 篇以上一作论文，加上至少一段实习或工业界经历，看起来是在顶级实验室稳定拿到回复的门槛"*——紧接着就是那句"一旦开始拿面试就别再折腾论文了" [42]。她自己的经历说明这个过程噪声很大：她申请的地方几乎都给了面试，而有三家（Waymo、Wayve、SpaceXAI）完全沉默，看不出任何理由。

内推值得拿，但不值得纠结。Sapora 在 DeepMind 有两个岗位有内推、第三个没有，结果 *"拿到面试的是其中一个有内推的岗位，和那个没内推的"* [42]。在 Anthropic 她一直没有音讯，直到一位前同事帮她内推。Meng 的看法完全相反，她跳过了内推，理由是强的简历本来就能过，而不匹配的岗位有了内推只会更快被拒 [34]。两种说法可以同时成立；成本很低。

<a id="signals-and-the-ones-that-count-against-you"></a>
### 信号，以及那些会扣分的信号

这是 Lambert 的招人视角比任何候选人复盘都值钱的地方，因为他描述的是他**实际上**怎么处理一份申请。

> *"有人说读一条推文就能看出一个人是不是天才，我同意。书面文字仍然是一种极其有效而且被严重低估的沟通形式。一篇优秀的博客文章可以体现出真实且稀有的理解力。反过来对 AI slop 同样成立。**一篇 AI 生成的垃圾博客会直接毁掉你的申请。**"* [27]

这是一个值得理解的**不对称赌注**：公开写作上限很高，但确实有下界。发一篇有想法的东西是你能做的杠杆最高的事之一；发一篇模型生成的注水内容是净负值。

他还点出了一个**我在别处从没见人写过的负面信号**：*"一个很小但很明确的负面信号，是一个 junior researcher 在太多论文里挂中间作者。学会说不，这对你有好处"* [27]。背后的原则是**先深后广**——*"太多早期研究者急着建立影响力的广度（比如在很多项目里攒贡献），却还没有向自己和导师清楚地证明深度。"*

关于开源，他态度正面但现实。按他的经验，开源比开放研究小组更容易转化，但正在变难：*"要在 AI slop 的 PR 和 Issue 海里冒头会很难。这需要格调、创造力、人味和耐心"* [27]。Gordić 更早的"作品集胜过履历"的论点仍然成立，而且说得更暖：比起测试一个人五小时，他更愿意雇一个自己在开源项目里观察了几个月的人 [11]。

还有两个容易被忽略的渠道，Lambert 点名了：*"有些公司大量从 Twitter 招人，有些从 GPU Mode 或 NanoGPT speedrunning 这样的社区招人"* [27]。

以及他最后给出的那个框架，在你忍不住要过度推销自己的时候最该记住：*"第一个要问的问题是'这个人行不行？'第二个问题是'这个人在这里能不能活得好？'"* [27]

<a id="cold-emails"></a>
### Cold email

桌子两侧都认可这件事，而 Lambert 精确地说出了失败模式：

> *"你仰慕的那些 AI 从业者，很多人是会读邮件的；你没收到回复的原因是你的邮件格式不对。最好的 cold email 会让收信人觉得自己从中学到了东西，或者明显因为收到它而受益。恭维和客套当然让人愉快，但最好的 cold email 是能激发行动的。"* [27]

他最近的两位 hire 就是从这扇侧门进来的，不是从招聘页。Sapora 给 DeepMind 的 hiring manager 发了邮件并得到回复；她的建议不是复述简历，而是 *"解释你为什么适合这个具体的团队，以及他们的工作里什么让你真正兴奋"* [42]。她最大的遗憾就是没多做这件事：*"对那些无视我的公司，我本该更主动……如果你真的很想去某个地方而一直没有回音，就去做点什么"* [42]。

关于 cover letter，Sapora 有一句必须转述，因为诱惑太明显：*"求求了，别直接让 Claude / Gemini / ChatGPT 帮你写。你完全可以自己写完再让它们润色，那没问题"* [42]。这也正是 Anthropic 自己的候选人指引所要求的——先自己起草，再润色 [2]。

**Takeaway.** 这道门的时钟以年计，不以周计。一个好的公开作品、几段真实的关系、一封写给具体某个人的好邮件，作用大过再发一篇论文。

---

<a id="the-shape-of-the-loop"></a>
## 一轮面试长什么样

在规划任何一个小时的复习之前，先把轮次清单拿到手。Meng 的规则：

> *"我只在已经约好几场 phone screen 之后才开始准备，因为不同公司的 loop 差别很大——在 recruiter 解释清楚都有哪些轮、每一轮考什么之前，我根本不知道该从哪儿开始。"* [34]

显而易见的反驳是那时候来不及了。她的回答是：recruiter call 之后，loop 什么时候开始是你说了算的，而且 *"如果一家公司接下来两三个月都不招人，你为什么要去？"* [34] Lambert 从另一侧补了一个具体做法——直接问 recruiter 要准备什么，*"有时候 recruiter 真的会给你一个方向，比如 threading 或者面向对象编程"* [26]。

Liu 的分类是最完整的，下面用她的类别而不是那些指南的类别 [30]：

| 轮次 | 是什么 | 按 Liu 的频率 |
|---|---|---|
| ML coding | 实现一个架构、一种解码策略、一个 ML 算法，"有时候是更有创意的东西" | **"到目前为止最常见"** |
| General coding | LeetCode，有时带点花样 | 常见 |
| Technical discussion（深挖型） | 设计实验来回答一个研究问题；为你的选择辩护；解释假设结果 | 常见 |
| Technical discussion（快问快答型） | 广度检查。她的原例：*"编码位置信息有哪些不同做法？什么是 5D parallelism？PPO 和 GRPO 的区别是什么？"* | 常见 |
| Research discussion | 从一个过去的项目开始，然后自由发散 | 研究岗常见 |
| Behavioral | 教科书式，偶尔夹一道 AI safety 题 | 普遍 |
| **数学** | *"从有趣的逻辑谜题，到需要纸笔的严肃数学推导"* | 部分公司 |
| Job talk | 比学术版短，聚焦一篇论文或一个方向 | 研究岗 |

注意 technical discussion 内部的那个分裂，因为它改变你的准备方式：*"前一种考的是你怎么思考，后一种检查的是你在这个领域的知识广度"* [30]。前者临时抱不了佛脚，后者也不可能靠现场推理蒙过去。

Sapora 给出的技术轮数量：**3 到 8 场**，视公司而定 [42]。

三个比任何具体格子都更具普适性的结构性要点：

**这些轮次远没有看上去那么标准化。**这是 Yong 的核心修正，而且正因为它不方便，才更值得当真。他被问过系统设计、用 `asyncio` 写并发、以及评估他怎么使用 AI agent 的轮次 [55]。Meta 有一轮 AI-enabled coding，你要在有 LLM 可用的情况下调试一个陌生的多文件代码库 [34]。而 Anthropic 的 live 轮明确禁止 AI [2]。搞清楚你在哪一种里。

**其中很多和你的专长毫无关系。**Yong 转向 AI safety，预期会有大量 safety 相关的面试；结果 *"感觉仍然是在被评估作为一个 AI 研究者有多全面"* [55]。

**Reference 和 team match 是末尾两道独立的关。**Meng 提到她接触的所有 frontier lab 都会在发 offer 前要两到三位 reference [34]——这是一个提前量以年计的准备项。而过了技术门槛不等于有工作：当只有少数团队有 headcount 时，*"你的年限、项目复杂度和面试表现会被重新评估一遍"* [34]。Yong 补充说，研究岗的 **return offer 很少**，不像 SWE；他即便当过 OpenAI 的 Astra Fellow，仍然走了完整的 loop [55]。

**Takeaway.** 先问 recruiter 要准确的轮次清单和通过标准，然后针对你**实际拥有**的 loop 准备——同时给"至少有一轮没人提前告诉你"留出预算。

---

<a id="bucket-1-progressive-system-building"></a>
## 方向一：渐进式系统构建

这就是人们说"coding 轮不是 LeetCode"时指的那一轮，也是 ML 候选人最常低估的一轮。

形式是一道题分成若干层层递进的关卡，通常 45 到 90 分钟。你先实现一个小系统，然后随着新需求进来扩展三次。Field Guide 的说法我认为最准确：*"代码必须是可扩展的，因为每一关都建立在上一关的代码之上"* [13]。被报告过的题目包括：从 `SET`/`GET`/`DELETE` 开始搭一个内存 KV 数据库、实现一个网络爬虫、一个规则不断加码的额度管理系统、一个支持类 SQL 操作的内存数据库，以及——很有画面感的——重构 100 到 120 行嵌套极深的混乱代码 [13]。

Meng 列出的同类题：time-based key-value store、in-memory database、类 C 的内存分配器、类型推断引擎、circuit breaker、带限流的 API gateway、LRU/LFU cache、线程池 / 任务调度器、事务或额度系统 [34]。

注意这些题的共同点。它们全都是真实后端基础设施的微缩版。没有一道有巧妙技巧。每一道都有一个**对象模型**，而它要么对，要么错。

<a id="the-four-level-ladder"></a>
### 四级阶梯

递进方式的可预测程度足以让你直接针对它准备：

| 关卡 | 新增什么 | 实际在考什么 |
|---|---|---|
| 1 | 核心功能、单线程、最小 API | 你能不能选对数据结构并快速跑起来？ |
| 2 | 约束、边界情况、更多 API | 你第一关的设计留了余地，还是必须重写？ |
| 3 | **并发**、性能保证、竞争下的正确性 | 你懂不懂共享可变状态？ |
| 4 | 可扩展性、可插拔、配置、故障处理 | 这东西上生产能活下来吗？ |

第三关是 ML 候选人掉队的地方，因为它是唯一一个跟算法完全无关的关卡。在一份被报告的经历里，recruiter 甚至提前打了招呼：你需要熟悉如何处理并行。（这条来自论坛汇编而非具名作者，所以当成一个形状可信的传闻看待就好——但照着做的成本很低。）

<a id="worked-example-a-key-value-store-four-times"></a>
### 完整例子：把一个 KV store 写四遍

我把被报告最多的那道题从头走到尾，因为内容在**决策**里，不在代码里。

**第一关 —— `set`、`get`、`delete`。**

朴素答案是一个 dict。朴素答案是对的。不要在第一关过度设计，你后面需要时间。但要把选择的理由说出来，因为面试官已经在判断你是不是按接口思考。

```python
class KVStore:
    def __init__(self):
        self._data: dict[str, bytes] = {}

    def set(self, key: str, value: bytes) -> None:
        self._data[key] = value

    def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None
```

唯一值得口头点出的决策：`delete` 返回"是否删掉了东西"而不是抛异常。幂等的删除才是真实客户端想要的，说这一句只花四秒。

**第二关 —— 加 TTL。**

从这里开始分人。显然的实现是给每个 value 存一个过期时间，读的时候检查。而这个显然的实现有一个面试官专门盯着的 bug：**急着往第四关冲的候选人会忘记在读的时候检查 TTL**，于是过期的 key 会一直可见，直到别的操作碰到它。

更微妙的设计问题是**惰性过期 vs 主动过期**。惰性（访问时检查）实现起来很简单，但对再也不会被读到的 key 会漏内存。主动（后台清扫线程）能回收内存，但要一个线程，还会引入自己的并发问题。面试里通常的正确答案是"默认惰性，可选加清扫线程"，而说这句话的**理由**是——Redis 就是这么做的。

```python
import time
from dataclasses import dataclass


@dataclass(slots=True)
class _Entry:
    value: bytes
    expires_at: float | None  # 单调时钟秒数；None 表示不过期


class KVStore:
    def __init__(self, clock=time.monotonic):
        self._data: dict[str, _Entry] = {}
        self._clock = clock  # 注入进来，这样测试不必真的 sleep

    def _live(self, key: str) -> _Entry | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and entry.expires_at <= self._clock():
            del self._data[key]  # 惰性过期：由发现它的那次读来回收
            return None
        return entry

    def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        expires_at = None if ttl is None else self._clock() + ttl
        self._data[key] = _Entry(value, expires_at)

    def get(self, key: str) -> bytes | None:
        entry = self._live(key)
        return None if entry is None else entry.value

    def delete(self, key: str) -> bool:
        existed = self._live(key) is not None
        self._data.pop(key, None)
        return existed
```

这里有两件事比代码本身更值钱。第一，**把时钟注入进来**——它说明你不用被提醒就会考虑可测试性，而且面试官接下来那句"你怎么测过期？"已经被提前回答了。第二，让所有读操作都走 `_live`，意味着 TTL 在后面任何一关都不可能被漏掉。这才是"可扩展"在实践中的含义：**不变量只住在一个地方**。

**第三关 —— 做成线程安全。**

第一反应是拿一把大锁把所有东西包起来。这是对的，而且你应该先这么做，因为一把正确的粗粒度锁胜过一个微妙地错了的细粒度锁。做完之后，再说你下一步会怎么做。

```python
import threading


class KVStore:
    def __init__(self, clock=time.monotonic, shards: int = 16):
        # 分片：互不相干的 key 用各自的锁，避免争用
        self._shards = [({}, threading.RLock()) for _ in range(shards)]
        self._clock = clock

    def _shard(self, key: str):
        return self._shards[hash(key) % len(self._shards)]

    def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        data, lock = self._shard(key)
        expires_at = None if ttl is None else self._clock() + ttl
        with lock:
            data[key] = _Entry(value, expires_at)

    def get(self, key: str) -> bytes | None:
        data, lock = self._shard(key)
        with lock:
            entry = data.get(key)
            if entry is None:
                return None
            if entry.expires_at is not None and entry.expires_at <= self._clock():
                del data[key]
                return None
            return entry.value

    def compare_and_set(self, key: str, expected: bytes | None, value: bytes) -> bool:
        """让并发客户端能够组合起来的那个原语。"""
        data, lock = self._shard(key)
        with lock:
            current = data.get(key)
            if (current.value if current else None) != expected:
                return False
            data[key] = _Entry(value, None)
            return True
```

这一关能拿分的东西，大致按顺序：

- **在修之前先把竞态说出来。**"在检查过期的那次 `get` 和执行删除的 `del` 之间，另一个线程可能写入了新值，我们会把有效数据删掉。"然后把两者都挪进锁里。
- **用分片而不是一把全局锁**，并明确说出代价：吞吐上去了，但任何跨多个 key 的操作现在要么需要全局锁，要么需要按固定顺序获取多把锁来避免死锁。
- **主动提供 `compare_and_set`。**客户端侧的 read-modify-write 无论你内部锁做得多好都不安全；客户端需要一个原子原语。主动提这个是很强的信号。
- **知道 GIL 管什么、不管什么。**在 CPython 里 GIL 让单个 dict 操作是原子的，但**不会**让你的"先检查再动作"序列变成原子的。能把这一点讲对，可以区分出"调试过并发 Python 的人"和"读过并发 Python 的人"。

**第四关 —— 持久化、快照、淘汰。**

第四关通常是开放的，面试官这时评估的是判断力多过代码。正确的动作是把维度列出来、给一个默认选择、然后问他想深入哪一个：

- **持久性。**变更的 append-only 日志 + 周期性快照压缩（这就是 Redis 的 AOF + RDB，把这一点说出来是白送的分）。讨论一下写入是在 fsync 之前还是之后确认——那才是真正的持久性/延迟旋钮。
- **内存上限下的淘汰。**LRU 需要哈希表 + 双向链表做到 O(1)；LFU 需要频率桶。说清楚哪个更贴合访问模式，而不是条件反射地默认 LRU。
- **快照隔离。**一个读者在写者持续修改时遍历整个 store，需要 copy-on-write 或者每条目一个版本号。这是第四关最可能以"我想不停写地做备份，怎么办？"形式出现的问题。
- **故障处理。**日志写到一半崩了怎么办？你需要每条记录一个校验和，以及一条"截断到最后一条有效记录"的恢复路径。

<a id="why-strong-ml-candidates-fail-this-round"></a>
### 为什么很强的 ML 候选人会挂在这一轮

Meng 的自我诊断值得完整引用，因为它是我见过关于这一轮写得最有用的一段：

> *"我原来以为是'快'这件事害了我。后来在一次和后端工程师的模拟面试里，我看到了真正的问题：我的代码能过测试、看着也干净，但我会做出一些任何一个好的后端工程师都不会做的设计选择。比如设计购物车时，我把价格、数量和其他属性直接存在了一个 `Item` 数据类里，而后端工程师会用一个唯一的 `product_id`，需要时再去关联外部元数据。"* [34]

整个失败模式就在这里。代码能跑。测试能过。但那个对象模型说明：你从没维护过一个"下单之后价格会变"的系统。

三个习惯能解决大部分问题：

1. **把身份和状态分开建模。**实体拥有稳定的 ID；可变属性住在一个能被一次性更新的地方，而不是被复制到每一处引用里。
2. **把不变量放在一个地方。**如果 TTL 在三个方法里各检查一次，其中一个迟早会写错。让所有读走同一个访问器。
3. **假定下一个需求马上就来。**它确实会来。这就是这轮的形式。

剩下一半靠暴露量。Meng 建议学原则而不是攒题目——*"有些人执着于收集别的候选人遇到过的题——这既累又冒险……万一你碰到一道新题，或者老题的新变体呢？你就慌了，然后挂了"* [34]。Field Guide 同意，并补上了真正管用的那个练法：做一个可以逐层加东西的项目，然后 *"在时间压力下练习扩展它。如果你最初的设计不重写就撑不住新需求，那就是该重新设计的信号"* [13]。

<a id="question-bank-system-building"></a>
### 题库

答案给得简短；重点是回答的**形状**，不是字数。

**设计一个限流器。**先问语义：固定窗口（便宜，但在边界允许 2 倍突发）、滑动窗口日志（精确，内存随请求数增长）、滑动窗口计数（在两个固定窗口间插值，折中不错）、还是令牌桶（允许受控突发，而这通常正是 API 想要的）。默认选令牌桶并说明理由：每客户端 O(1) 内存，是唯一能自然表达"持续速率 + 突发额度"的方案，也是大多数生产网关的实现。然后是分布式追问：计数器要挪到 Redis，check-and-decrement 必须原子（Lua 脚本或带过期的 `INCR`），而且你得决定在网络分区期间是否容忍轻微超发。

**实现 O(1) 的 LRU cache。**哈希表从 key 映射到节点，加一个带哨兵头尾的双向链表。哨兵是大家最容易搞砸的部分——它让所有拼接逻辑都不需要判空。`get` 时把节点移到头部。`put` 超容量时从尾部淘汰。追问：做成线程安全的（一把锁就行，解释为什么无锁在这里很难），然后做成 LFU（频率桶，每个桶是一个 LRU 链表，再加一个指向最小频率的指针）。

**实现一个 circuit breaker。**三个状态：closed、open、half-open。Closed 在滑动窗口里数失败；越过阈值就跳到 open。Open 立即拒绝并启动冷却计时。冷却结束后 half-open 放少量试探请求进去；成功则关闭，失败则重新打开并（理想情况下）延长冷却。值得主动提的微妙点：在 half-open 状态下你必须**限制并发**，否则一堆排队的请求会一次性涌进去，把一个还没好的服务再打垮一次。

**对大到放不进内存的文件做去重。**候选人经历中报告过，而这其实不是一道哈希表题。按固定大小分块流式读入；对每块做哈希；比较哈希而不是内容。然后才是真正的讨论：固定大小分块在插入面前会崩（后面全部错位），所以用滚动哈希做内容定义分块，能得到抗位移的边界。再往下：哈希索引本身超内存怎么办？按哈希前缀分片并溢写到磁盘。再往下：碰撞策略是什么——哈希命中时是否逐字节校验，还是接受 256 位哈希的生日界风险？

**设计一个网络爬虫。**在 Anthropic 被报告过 [13]。先单线程：一个 frontier 队列、一个 visited 集合、一个抓取-解析-入队循环。然后是礼貌性（按域名限速和 `robots.txt`），这会迫使 frontier 从一个全局队列变成按域名分的多个队列。然后是并发（工作线程池共享 frontier，引出 visited 集合的竞态）。然后是分布式（按域名哈希切分 URL 空间，让礼貌性只需在单个 worker 内保证；visited 集合超内存后换 Bloom filter）。最后是故障情况：爬虫陷阱、重定向环、同一内容的不同 URL。

**Time-based key-value store。**`set(key, value, timestamp)` 和 `get(key, timestamp)` 返回不大于查询时间戳的最大时间戳对应的值。每个 key 一个 (timestamp, value) 列表，读时二分。追问几乎总是：时间戳乱序到达怎么办？那你需要插入到有序结构里，并且应该讨论是在写时付代价（保持有序）还是在读时付（惰性排序）。

**Takeaway.** 三关干净胜过四关半残。给你打分的人是后端工程师，他读的是你的对象模型，不是你的算法。

---

<a id="bucket-2-ml-coding"></a>
## 方向二：ML Coding

这是过去两年变化最大的一轮，也是"我懂这个"和"我能写出这个"之间差距最大的一轮。

<a id="the-constraint-that-defines-the-round"></a>
### 定义这一轮的那个约束

关于该怎么准备，一切都由一个数字推导出来，Jaiswal 说得很直白：

> *"这些概念只要给够时间和调试条件都能写出来，但面试环境有它独特的挑战。你通常只有 **25 到 35 分钟**，需要一次写对，而且全程都要把矩阵维度维持正确。"* [21]

25 分钟，没有调试器，维度必须对。这个预算排除了"现场理解"。Meng 从另一面讲了同一件事：

> *"在 ML coding 面试里，你被期待写 PyTorch 像写普通 Python 一样流畅。就算可以 Google，只要你需要查得多一点，就一定会超时。"* [34]

她对目标状态的描述是：流畅到能**凭记忆**写出 *"MLP、CNN、RNN、Transformer encoder/decoder"* 以及各种构件——*"线性层、投影、残差连接、layer norm、batch norm、causal self-attention、双向 self-attention、激活函数、优化器"* [34]。

有一个习惯的回报率高于其他所有，来自 Jaiswal：

> *"一定要把维度写进变量名或注释里。这对调试有帮助，而且面试官经常通过查维度来考你的理解——不只是你的记忆力。"* [21]

她后来从写注释改成了用 Noam Shazeer 的 **shape suffix** 约定，并说它 *"也许还更好"* [21]。做法是把形状编进名字：`(batch, time, channel)` 的张量叫 `x_BTC`，切完头之后叫 `q_BHTD`。在一个失败模式是"一次静默的转置"的轮次里，把形状写进标识符意味着 bug 在调用处就暴露，而不是三行之后。

下面所有内容都出自同一个文件，它作为测试套件跑在 PyTorch 的 ground truth 上，所以没有一段是"看起来对"的代码。

```
  pass   causal MHA matches F.scaled_dot_product_attention
  pass   causal mask actually blocks the future
  pass   KV-cached incremental decode == full recompute
  pass   RoPE attention logits depend only on relative offset
  pass   online (streaming) softmax == naive softmax
  pass   LoRA is identity at init, and merging preserves outputs
  pass   cross entropy matches F.cross_entropy, including ignore_index
  pass   MoE routing respects capacity; aux loss is minimised by a uniform router
  pass   GRPO: zero advantage spread -> ~zero loss; k3 KL is non-negative
  pass   micrograd-style autograd matches torch.autograd, incl. reused nodes
  ...
16/16 checks passed
```

<a id="attention-and-the-four-traps-inside-it"></a>
### Attention，以及藏在里面的四个坑

这是 E1 —— 出自 Vaswani 等人 [52] 的机制，写成你应该能不假思索敲出来的样子。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))
```

大约二十行。其中四行是人们丢分的地方。

**坑 1：`.contiguous()`。**`transpose(1, 2)` 之后张量是一个 stride 不连续的 view。`.view()` 要求内存连续，会直接报错。你也可以改用 `.reshape()`，它在需要时会静默复制——但如果面试官问为什么，"reshape 必要时会复制，view 不会"就是他想听的答案，它说明你分得清张量和它的存储。

**坑 2：除以 `sqrt(d_head)` 而不是 `sqrt(d_model)`。**点积是在 head 维度上做的，所以那才是你要校正方差的维度。搞错了模型照样能训，只是差一点——这正是它成为好面试题的原因。

**坑 3：在 softmax 之前用 `-inf` 做 mask，而不是在之后乘 0。**在 softmax 之后把概率乘以 0/1 掩码，会让被屏蔽的位置仍然贡献到分母里，于是存活下来的权重不再和为 1。在 softmax 之前做加性 `-inf`，才能让它们精确为零。

**坑 4：把 QKV 融进一次投影。**三个独立的 `nn.Linear` 在数学上完全等价，但明显更慢——同样的总 FLOPs 下一次 GEMM 胜过三次，因为 kernel 启动开销和访存。用快的那种写法，并说明原因。

验证 mask 真的有效的那个检查值得记住，因为它同时也是调试方法：扰动最后一个 token，确认它之前的一切都不动。

```python
y1 = model(x)
x2 = x.clone(); x2[:, -1, :] += 10.0
y2 = model(x2)
assert torch.allclose(y1[:, :-1], y2[:, :-1])   # 过去看不到未来
assert not torch.allclose(y1[:, -1], y2[:, -1])  # 但当下确实变了
```

如果一个 causal mask 的 bug 混进了训练，这三行就是把它找出来的方法。把这一点说出来，和写出实现一样值钱。

<a id="the-backward-pass-by-hand"></a>
### 手推反向传播

Sapora 在她的 baseline 里明确列了这一项，而这也是大多数人会跳过的一项，因为大家默认 `autograd` 已经让它过时了 [42]。它没有：实验室问这个，恰恰是因为它能分开"知道框架在干什么"和"知道怎么调用框架"两种人。Liu 从另一侧印证了同一件事——NumPy 在她的 loop 里出现，主要就是 *"从零写反向传播的时候"* [30]。

推导只有四行。先把它们写在纸上，再写代码，代码就只是誊抄：

$$O = PV \;\Rightarrow\; dV = P^\top dO,\quad dP = dO\,V^\top$$

$$P = \mathrm{softmax}(S) \;\Rightarrow\; dS = P \odot \big(dP - \textstyle\sum_j (dP \odot P)_j\big)$$

$$S = \tfrac{1}{\sqrt{d}} QK^\top \;\Rightarrow\; dQ = \tfrac{1}{\sqrt{d}}\, dS\,K,\quad dK = \tfrac{1}{\sqrt{d}}\, dS^\top Q$$

```python
def attention_forward(q, k, v, causal=True):
    """q,k,v: (B, H, T, D)。返回 (out, cache) 供手写反向使用。"""
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
    q, k, v, p, scale = cache
    d_v = p.transpose(-2, -1) @ d_out
    d_p = d_out @ v.transpose(-2, -1)
    # softmax 的 Jacobian，逐行作用，不物化 (T, T, T) 张量
    d_s = p * (d_p - (d_p * p).sum(dim=-1, keepdim=True))
    d_q = (d_s @ k) * scale
    d_k = (d_s.transpose(-2, -1) @ q) * scale
    return d_q, d_k, d_v
```

唯一值得你能讲清楚的是 softmax 的 Jacobian 那一行。对单独一行来说，
$$\partial p_i / \partial s_j = p_i(\delta_{ij} - p_j)$$，所以完整的 Jacobian 是
$$\mathrm{diag}(p) - pp^\top$$——每一行一个稠密的 $$T \times T$$ 矩阵，物化出来就是 $$T^3$$。
`p * (dP - rowsum(dP * p))` 这个表达式就是在不构造它的前提下完成那次矩阵-向量乘法。面试官会专门问这一步。

还有两点能加分。causal mask 在反向里不需要任何特殊处理，因为 `P` 在被屏蔽的位置已经精确为零，梯度会继承这一点。以及，FlashAttention 之所以在反向里重算 attention 而不是把它存下来，原因就摆在这里：反向需要 `P`，而 `P` 正是那个你千方百计不想保留的 $$O(N^2)$$ 对象。

MLP 的反向是同一个练习，符号更少，也是 Sapora baseline 的另一半：

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
    d_h = d_a * (h > 0)                       # ReLU'(h)；0 处的次梯度取 0
    d_W1 = x.transpose(-2, -1) @ d_h
    d_b1 = d_h.sum(dim=0)
    d_x = d_h @ W1.transpose(-2, -1)
    return d_x, d_W1, d_b1, d_W2, d_b2
```

两者都在参考实现里用 float64 对照 `torch.autograd` 验证过——用 float64 是为了让比较是精确相等而不是近似相等。有两个约定要随时能说出来：**对权重矩阵的梯度形状与权重本身相同**，这是能抓住绝大多数转置错误的检查；以及 bias 的梯度要在 batch 维上求和，因为 bias 是沿着这一维广播出去的。

<a id="the-kv-cache-and-proving-it-correct"></a>
### KV cache，以及怎么证明它是对的

万能的追问是"现在让生成变快"。答案是：自回归解码时你在每一步都重新计算之前所有 token 的 K 和 V，纯属浪费，所以缓存它们。而一旦开始缓存，缓存大小就变成了瓶颈约束——这正是整个领域转向 grouped-query attention 的原因。两个改动应该写在同一份实现里。

```python
class GroupedQueryAttention(nn.Module):
    """n_kv_heads < n_heads。n_kv_heads == 1 就是 MQA；== n_heads 就是普通 MHA。"""

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
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        T_full = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # 绝对位置 (T_full - T + i) 上的 query 可以看到 keys 0..(T_full - T + i)
        causal = torch.ones(T, T_full, dtype=torch.bool, device=x.device).tril(
            diagonal=T_full - T
        )
        att = att.masked_fill(~causal, float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))
```

区分"真实现"和"背下来的实现"的那一行是 mask。有了 cache，你的 query 块不是从位置 0 开始的，而是从 `T_full - T` 开始。这里用普通的 `tril` 是错的。`diagonal=T_full - T` 这个偏移，才让单 token 的 decode 步骤合法（它可以看到全部历史），同时让多 token 的 prefill 步骤保持正确的因果性。

以及一个你应该在面试官问之前就主动说出来的正确性性质：**带 cache 的增量解码必须与完整重算在数值上完全一致。**

```python
full = model(x)                       # teacher forcing，一次算完

cache, outs = {}, []
for t in range(T):                    # 逐 token，带 cache
    outs.append(model(x[:, t : t + 1, :], cache=cache))
step = torch.cat(outs, dim=1)

assert torch.allclose(full, step, atol=1e-5)
```

这个测试在参考实现里是通过的。如果你没法让它通过，就是 mask 的偏移错了——而这恰恰是那种会以"模型 eval 时没问题、一生成就变差"的形式流到生产环境的 bug。

现在看催生 GQA 的那笔账。KV cache 的大小是

$$\text{bytes per token} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{bytes per element}$$

这个 2 是 K 和 V。公式里没有任何微妙的东西，这正是它是道好白板题的原因：纯粹是记账，你要么算过，要么没算过。

![MHA、GQA、MQA、MLA 的每 token KV cache](/assets/img/blog/frontier-lab-interview/fig6_kv_cache_math.png)
*图 6. 对一个 70B 级别的 decoder（80 层、64 头、d_head 128、bf16），普通 MHA 每 token 要 2.5 MB——一段 128k context 的对话在加载任何权重之前就需要 312 GB 缓存。GQA 用 8 个 KV 头把它砍到 1/8。就是这一个改动让长上下文在经济上变得可行。*

各个变体以及它们各自的取舍：

| 变体 | KV 头数 | 每 token 字节（上述配置） | 取舍 |
|---|---|---|---|
| MHA | 64 | 2,560 KB | 质量最好，缓存承担不起 |
| GQA [1] | 8 | 320 KB | 省 8 倍，报告的质量损失可忽略 |
| MQA [47] | 1 | 40 KB | 省 64 倍，质量损失可测量 |
| MLA [7] | latent | 约 90 KB | 把 K/V 压成低秩 latent；DeepSeek 报告它在质量上还胜过 MHA |

值得提前装好的追问：**为什么是 GQA 赢了 MQA？**因为 MQA 只用一个共享 KV 头，瓶颈收得太狠——质量下降，训练也不够稳；而 GQA 用一个可调的旋钮换来了绝大部分的显存收益。以及**为什么 DeepSeek 选 MLA 而不是 GQA？**他们的消融显示 GQA 在建模质量上比 MHA 略**差**，而 MLA 略**好**，这让 MLA 成为少见的"不是取舍"的优化 [7]。

<a id="positions-rope"></a>
### 位置信息：RoPE

旋转位置编码 [51] 现在被问得非常频繁，原因是它有一个可以一句话说清、三行代码证明的性质。

```python
def rope_cache(seq_len, d_head, base=10000.0):
    assert d_head % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)          # (T, d_head/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, d_head)。把坐标对 (2i, 2i+1) 旋转 m * theta_i 的角度。"""
    T = x.shape[-2]
    cos, sin = cos[:T], sin[:T]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)
```

那个性质是：RoPE 对 Q 和 K 各自施加一个**绝对**旋转，但因为两个被旋转向量之间的点积只依赖于旋转角之**差**，最终的 attention logit 就只是相对偏移的函数。具体地，位置 (5, 2) 和 (20, 17) 给出相同的 logit；(5, 2) 和 (5, 4) 不会。参考实现里断言的就是这一条。

三个会立刻跟上来的追问：

- **RoPE 加在哪里？**只加在 Q 和 K 上，在切头之后、点积之前。绝不加在 V 上——V 携带的是内容，不是位置。
- **为什么 RoPE 外推到训练长度之外效果差？**低频分量在训练中连一整圈都没转完，所以模型从没见过那些角度，也就没有插值的依据。这正是 position interpolation 和 NTK-aware / YaRN 缩放要打的补丁。
- **RoPE 和 KV cache 怎么配合？**缓存**旋转之后**的 key。缓存旋转前的 key、读的时候再转也能работать，但每步都在浪费计算。

<a id="online-softmax-which-is-flashattention"></a>
### Online softmax，也就是 FlashAttention

面试官很少让你写 FlashAttention 的 kernel——那是 CUDA 练习。他们问的是你懂不懂那个**想法**，而那个想法完全可以用二十行 Python 表达。

问题在于：朴素的 attention 会在 HBM 里物化一个 $$N \times N$$ 的分数矩阵。在长上下文下这既是显存天花板，也是速度天花板（因为 attention 是访存受限的）。FlashAttention [5] 通过永不物化这个矩阵来绕开它，而这要求你在没看到全部输入的情况下完成 softmax 归约。这是可能的，因为 softmax 有一个重缩放递推：

```python
def online_softmax_weighted_sum(scores, values, block=4):
    """流式计算 softmax(scores) @ values，不物化概率向量。"""
    n = scores.shape[0]
    m = torch.tensor(float("-inf"))   # running max
    l = torch.tensor(0.0)             # running 分母
    acc = torch.zeros(values.shape[1])  # running 分子

    for start in range(0, n, block):
        s = scores[start : start + block]
        v = values[start : start + block]
        m_new = torch.maximum(m, s.max())
        correction = torch.exp(m - m_new)   # 把到目前为止的累积量重缩放
        p = torch.exp(s - m_new)
        l = l * correction + p.sum()
        acc = acc * correction + p @ v
        m = m_new
    return acc / l
```

每当一个 block 揭示出更大的最大值，你就把累积的分子和分母乘以 `exp(m_old - m_new)` 重缩放，然后继续。结果与完整 softmax 可比到数值精度——参考实现用故意放大的 scores 验证了这一点，那正是朴素实现会溢出的地方。

Sapora 的 baseline 里写的是"implement flash attention"，而这在面试里的含义就是上面那段的**分块版本**：外层循环遍历 query 块，内层循环遍历 key 块，为每一行 query 携带各自的 running 统计量。

```python
def flash_attention_forward(q, k, v, block_q=16, block_kv=16, causal=True):
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(q)

    for i in range(0, T, block_q):
        qi = q[:, :, i : i + block_q]
        bq = qi.shape[2]
        m = torch.full((B, H, bq), float("-inf"), dtype=q.dtype, device=q.device)
        l = torch.zeros(B, H, bq, dtype=q.dtype, device=q.device)
        acc = torch.zeros(B, H, bq, D, dtype=q.dtype, device=q.device)

        for j in range(0, T, block_kv):
            if causal and j > i + bq - 1:
                break                                   # 整块都在未来
            kj, vj = k[:, :, j : j + block_kv], v[:, :, j : j + block_kv]
            s = (qi @ kj.transpose(-2, -1)) * scale
            if causal:
                rows = torch.arange(i, i + bq, device=q.device).unsqueeze(1)
                cols = torch.arange(j, j + kj.shape[2], device=q.device).unsqueeze(0)
                s = s.masked_fill(cols > rows, float("-inf"))

            m_new = torch.maximum(m, s.amax(dim=-1))
            correction = torch.exp(m - m_new)
            p = torch.exp(s - m_new.unsqueeze(-1))
            l = l * correction + p.sum(dim=-1)
            acc = acc * correction.unsqueeze(-1) + p @ vj
            m = m_new

        out[:, :, i : i + bq] = acc / l.unsqueeze(-1)
    return out
```

参考实现里还有一个稍长的版本，额外处理了"整块被完全屏蔽"的情况：此时 `m` 停留在 `-inf`，朴素的 `exp(m - m_new)` 会产生 `nan`。这个保护值得提一句，哪怕你不写出来——因为一旦你用一个不能整除序列长度的 block size 去跑，立刻就会撞上它。测试确认输出与 `F.scaled_dot_product_attention` 完全一致，而且**与 block size 无关**——如果面试官问你怎么测，这就是你应该主动提出去验证的性质。

把这三句说出来，这一轮就覆盖到位了：

1. 它是**精确**的，不是近似。这一点常常让人意外，也是它被全行业采用的关键。
2. 显存从 $$O(N^2)$$ 降到 $$O(N)$$；FLOPs 其实还略微**上升**，因为反向传播是在片上重算 attention 而不是读回存好的矩阵。它依然更快，因为这个操作原本是被 HBM 带宽卡住的，不是被算力卡住的。
3. 这个技巧很老——流式归一化的递推早于那篇论文 [36]。FlashAttention 的贡献是让它在真实硬件上赢的那套 IO-aware 分块和 kernel 融合。

<a id="sampling-normalization-loss"></a>
### 采样、归一化、损失

三个小东西，会作为热身题或"把生成循环补完"的追问出现。

**采样。**坑在顺序。温度必须最先（它改变了截断所作用的分布），然后 top-k，然后 top-p——按原始术语叫 nucleus sampling [17]。

```python
def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if temperature == 0:                       # 贪心；挡住除零
        return int(logits.argmax())
    logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        srt, idx = torch.sort(logits, descending=True)
        probs = F.softmax(srt, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        drop = cum - probs >= top_p            # 平移一位，让越过阈值的那个 token 活下来
        srt = srt.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, srt)

    return int(torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator))
```

`cum - probs >= top_p` 这一行必须写对：你保留的是累积质量**超过** p 的最短前缀，也就是说越过阈值的那个 token 要被包含进来，而不是被丢掉。这里差一位会静默改变采样分布。而 `temperature == 0` 需要一个显式分支，否则你就在除以零——这是真实推理服务里真实出现过的 bug。

**交叉熵**，本质是一道 log-sum-exp 题：

```python
def cross_entropy(logits, targets, ignore_index=-100):
    keep = targets != ignore_index
    logits, targets = logits[keep], targets[keep]
    m = logits.max(dim=-1, keepdim=True).values
    logsumexp = m.squeeze(-1) + (logits - m).exp().sum(-1).log()
    picked = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (logsumexp - picked).mean()
```

先减最大值再取指数就是全部要点；不这么做，20 左右的 logit 在 fp32 里就会溢出。`ignore_index` 的处理不是装饰——它正是 SFT 时屏蔽 prompt token 的方式，所以问到 instruction tuning 的面试官很可能会绕回这个函数。

**RMSNorm**，也就是现代模型实际在用的：

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).type_as(x) * self.weight
```

不减均值，没有 bias。追问：**为什么整个领域抛弃了 LayerNorm 的重新中心化？**因为消融显示起作用的是重新缩放，重新中心化基本没用 [57]，而去掉它就少了一次在特征维上的归约——当它在 80 层里每层跑两次时，这是有意义的。另外注意那个 `.type_as(x)`：在 bf16 下你希望归约在 fp32 里做、结果再转回来，提到这一点说明你训过东西，而不只是读过训练。

<a id="lora"></a>
### LoRA

低秩适配 [18] 是所有人都被期待能写出来的那个参数高效方法。

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B 保持为零：适配器在第 0 步是精确的恒等
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.T @ self.B.T * self.scaling

    @torch.no_grad()
    def merge(self):
        self.base.weight += (self.B @ self.A) * self.scaling
        self.A.zero_(); self.B.zero_()
        return self.base
```

面试官在检查两个性质，参考实现里都验证了：

**初始化时它是恒等。**`B` 从零开始，所以 `BA = 0`，被适配的模型精确等于基座模型。如果你把两个矩阵都随机初始化，你就静默地污染了起点，最初几步训练都在往回救。把两个都随机初始化的候选人暴露的是：他用过 LoRA 的库，但没读过它。

**它可以无损合并。**$$W + \frac{\alpha}{r}BA$$ 就是一个权重矩阵，所以训练完之后推理开销为零——不像 adapter 层那样增加深度。这才是 LoRA 真正胜出的原因。

追问：$$\alpha/r$$ 的存在是为了让你改秩时不用重调学习率。通常挂在注意力投影上，更难的任务上加 MLP 矩阵会有帮助。QLoRA 额外把冻结的基座量化到 4-bit、适配器保持较高精度，用一点质量换很多显存。以及诚实的局限：LoRA 非常擅长风格、格式和任务适配，非常不擅长注入大量新知识——一个低秩更新根本没有那个容量。

<a id="mixture-of-experts"></a>
### Mixture of Experts

你不会被要求写一个生产级 MoE 层，但带容量限制的 top-1 路由是一道合理的 20 分钟题，而且它能暴露你知不知道 MoE 难在哪。

```python
def top1_route(logits, capacity):
    """logits: (T, E)。返回 (expert_idx, gate, kept_mask)，带每专家容量。"""
    gates = F.softmax(logits, dim=-1)
    gate, expert = gates.max(dim=-1)
    kept = torch.zeros_like(expert, dtype=torch.bool)
    for e in range(logits.shape[1]):
        idx = (expert == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        # 溢出时保留最有把握的 token，其余丢弃
        order = idx[torch.argsort(gate[idx], descending=True)][:capacity]
        kept[order] = True
    return expert, gate, kept


def load_balancing_loss(logits):
    """Switch Transformer 辅助损失：E * sum_e (路由到 e 的比例) * (e 的平均概率)。"""
    gates = F.softmax(logits, dim=-1)
    E = logits.shape[-1]
    expert = gates.argmax(dim=-1)
    frac = torch.bincount(expert, minlength=E).float() / expert.numel()
    prob = gates.mean(dim=0)
    return E * (frac * prob).sum()
```

代码是为了暴露那个概念：**token 会被丢掉。**容量是有限的，因为 all-to-all 通信需要固定大小的缓冲区，所以当一个热门专家溢出时，多出来的 token 会整层跳过、顺着残差流走过去。这就是"一个 batch 化的 MoE 模型的输出会因为同 batch 里还有什么而略有不同"这一现象背后的机制。

辅助损失是另一半。路由是一个没有梯度的离散 argmax，所以放任不管的话 router 会塌缩到少数几个专家上。Switch Transformer 的损失 [10] 把路由到每个专家的 token **比例**乘以该专家的**平均门控概率**，求和后再乘以专家数。它在均匀路由处取最小值，此时等于 1——参考实现同时验证了"均匀 router 得分约等于 1"和"倾斜 router 得分更高"。值得一提的前沿进展：DeepSeek-V3 完全去掉了辅助损失，改用一个在训练中调整的偏置项来平衡负载，从而避免引入一个与语言建模目标相互拉扯的梯度 [8]。

<a id="tokenization"></a>
### 分词

字节对编码 [45] 出现的频率比人们以为的高，通常以"为什么模型数不清字母"的形式出现。

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
                merged.append(new_id); j += 2
            else:
                merged.append(ids[j]); j += 1
        ids = merged
    return merges
```

编码时要注意的细节：merge 要**按学到的顺序**应用，而不是按待编码字符串里的频率。搞反了会得到一个 round-trip 不一致的分词器，这是真正难查的生产 bug。

从原始 UTF-8 字节而不是字符出发，是另一个值得解释的决策：字节级词表可以表示任何输入，所以永远不存在 OOV。代价是非拉丁文字每个字符消耗更多 token，这是一个真实的公平性与成本问题，也是一个值得主动提出来的点。

<a id="autograd"></a>
### Autograd

Meng 提到这个时带着明显的怵——*"怎么从头实现 autograd 而不是调 `loss.backward()`？……即便每天用 PyTorch 的人，也很少知道计算图是怎么建起来的、autograd 底层怎么工作"* [34]。这是个合理的问题，而且大约四十行。

```python
class Value:
    """标量反向模式自动微分，仿 Karpathy 的 micrograd。"""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad      # 是 += 不是 = ：一个节点可能被用很多次
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
        for v in reversed(topo):       # 逆拓扑序
            v._backward()
```

两个想法撑起了全部，而如果你能把它们讲清楚，剩下的都能在压力下重建出来：

**每个操作都存一个闭包，它知道怎么把梯度推给自己的输入。**图是在前向过程中隐式建起来的；每个节点捕获它的父节点和自己的局部导数规则。

**梯度是累加的，遍历顺序是逆拓扑序。**那个 `+=` 是整个文件里最重要的一个字符——一个被用在两处的节点会从两条路径收到梯度，写成 `=` 就会静默丢掉一份。拓扑排序保证当你调用某个节点的 `_backward` 时，它的所有消费者都已经贡献完了。参考实现用一个 `a` 和 `b` 各自参与多条路径的表达式来对照 `torch.autograd`，那正是朴素实现会算错的情形。

<a id="question-bank-ml-coding"></a>
### 题库

除了上面已经实现的，还要能冷启动写出下面这些。清单来自 Meng [34]、Jaiswal [21] 和 Field Guide [13] 各自独立报告的内容。

**从零写（NumPy 或 PyTorch）：**带手写反向的 MLP；RNN cell，然后 LSTM 和 GRU cell（Jaiswal 特别指出这是大多数刷题平台的缺口 [21]）；带 stride 和 padding 的二维卷积；带 running statistics 的 BatchNorm，以及 train/eval 的差别；layer normalization；带 inverted scaling 的 dropout；从更新公式写 Adam；一个能把十个样本过拟合的完整训练循环。

**经典 ML，仍然会考：**带 SGD、L2 和 early stopping 的逻辑回归（在 Mistral 被报告过 [13]）；k-means；k 近邻；按信息增益做的决策树分裂；用 SVD 实现 PCA；以及各种指标——precision、recall、ROC-AUC、NDCG、MRR——从标签和分数手写出来，而不是 import。

**LLM 相关：**带长度归一化的 beam search；带 KV cache 的贪心解码；speculative decoding 的接受/拒绝步骤；label smoothing；DPO 和 GRPO 的损失（都在下面的方向四里）；sliding-window attention；encoder-decoder 的 cross-attention。

**调试变体，越来越常见：**给你一份看起来没问题但训不好的模型代码。手里要有清单。loss 在 padding 上 mask 对了吗？causal mask 差了一位吗？label 相对 input 移位了吗？warmup 之后的学习率合理吗？模型在 `train()` 模式吗？调了 `optimizer.zero_grad()` 吗？eval 时 dropout 还开着吗？有什么东西 detach 了计算图吗？数据真的 shuffle 了吗？Meng 对此的建议是：先用一个极小 batch 过拟合——如果模型连十个样本都记不住，bug 在代码里，不在超参里 [34]。

**在哪里刷。**第一手记录共同指向的四个资源，以及各自的用途：Karpathy 的 nanoGPT 和 micrograd [23]，用来端到端搭一个 GPT 和一个 autograd 引擎；Raschka 的 *Build a Large Language Model (From Scratch)* [41]，它的仓库里有 MHA、GQA、MLA、sliding-window attention、KV cache 和 MoE 各自独立的参考实现——它是这个方向最接近标准答案的东西；Deep-ML [6]，ML 版的 LeetCode，Meng 和 Jaiswal 都用它刷量；以及 Huyen 的面试书 [20]，覆盖 fundamentals 那一侧。Jaiswal 处理书籍仓库的做法值得偷师：她把 notebook 改造成了填空练习，而不是读它们 [21]。

**Takeaway.** 交付物是**流畅度**。读这些实现不算准备；从一个空编辑器反复敲出来、直到维度第一次就写对，才算。

---

<a id="bucket-3-ai-fundamentals"></a>
## 方向三：AI Fundamentals

这个方向的危险程度与它占用的时间完全不成比例。它常常只是 phone screen 开头的五分钟，而 Meng 对后果说得很直白：

> *"很多候选人失败之后一头雾水，记得自己 coding 题答得很完美，却忘了自己在 ML fundamentals 上犯了基础错误。**一两个答错就足以被拒。**"* [34]

准备方式不是背定义。两位主要材料在这一点上说的是同一件事，值得当真。Meng：*"我花在面试准备上的每一分钟，都应该让我成为一个更好的工程师……背答案做不到这一点"* [34]。她推荐的路径是读 Prince 的 *Understanding Deep Learning* [38]——时间紧的话读第 1–9、11、12 章——并且把准备当成一次找知识盲点的练习。

背诵之所以失败是结构性的：**追问永远是"那它什么时候会失效？"**定义对这个问题无话可说。所以下面的格式是**为什么是这样**，而且每个答案都自带它的失效模式。

<a id="attention-and-architecture"></a>
### Attention 与架构

**为什么要除以 $$\sqrt{d_k}$$？**设 $$q$$ 和 $$k$$ 的各分量独立、方差为 1。它们的点积是 $$d_k$$ 个这样的乘积之和，所以方差是 $$d_k$$、标准差是 $$\sqrt{d_k}$$。当 $$d_k = 128$$ 时，在训练还没做任何事之前，logits 的分布宽度就已经在 ±11 左右。在这么宽的 logits 上做 softmax 几乎就是 one-hot，而饱和的 softmax 梯度消失——attention pattern 在初始化时就被冻住，学不动。除以它就恢复了单位方差。*什么时候会失效：*这个论证假设了它所描述的初始化；如果输入尺度不对，或者权重漂移之后，logits 仍然可能爆掉——这正是超大模型里引入 QK-normalization 要处理的问题。

**有没有办法把 attention 看成某种熟悉的东西？**两个框架都值得有。Murphy 的（经 Jaiswal [21]）：*"我们可以把 attention 看成一次软字典查找：把 query $$q$$ 和每个 key $$k_i$$ 比较，然后取回对应的 value $$v_i$$。"* 硬字典在精确匹配时返回一个值；attention 返回的是按相似度加权的凸组合。第二个框架是 Meng 指出大家常常想不到的：attention 是**软的、可学的 k 近邻**。它不是按固定距离度量取 top-k，而是在一个可学的度量下对所有点做 softmax 加权平均。她恰恰把这类联系当作区分候选人的标志：*"我们可能分别很懂 KNN 和 attention，却意识不到后者是前者的软化版本"* [34]。

**softmax 的温度在做什么？**在 softmax 之前把 logits 除以 $$\tau$$，是在两个极限之间插值：$$\tau \to 0$$ 得到 argmax，$$\tau \to \infty$$ 得到均匀分布。它不改变排序，只改变锐度。*什么时候要紧：*$$\tau$$ 明显小于 1 时采样会退化成复读循环；大于 1 则语无伦次。这和 attention 的缩放因子是同一个旋钮，这一点值得主动指出来。

**Encoder-only、decoder-only 还是 encoder-decoder？**Encoder-only（BERT）用双向 attention，训练目标是 masked language modeling——适合分类和检索这类"把固定输入嵌入成向量"的任务，不能生成。Decoder-only（GPT）用因果 attention 和 next-token prediction，这意味着每个位置都是训练信号，而且同一个模型就能生成。Encoder-decoder（T5）用 cross-attention 把两者分开，适合真正的序列到序列任务，比如翻译。*为什么 decoder-only 赢了？*训练效率（每个 token 都提供监督）、架构简单，以及 in-context learning 把几乎所有任务都变成了生成任务。

**什么是 multi-token prediction，为什么要做？**像 DeepSeek-V3 那样在每个位置预测多个未来 token [8]，会让训练信号更稠密，并给模型一个前瞻目标。它同时还免费送你一个用于 speculative decoding 的 draft 模型。

<a id="normalization-residuals-and-depth"></a>
### 归一化、残差与深度

**Pre-LN 还是 post-LN，整个领域为什么迁移了？**原始 Transformer 把 LayerNorm 放在残差相加之后。这就把一个归一化放在了残差通路上，于是梯度在每一层都被重新缩放，深模型不精调 warmup 根本训不动。Pre-LN 归一化的是每个子层的**输入**，让残差流保持一条从 embedding 到输出的干净恒等通路。Xiong 等人 [53] 证明了这正是去掉 warmup 需求的原因。*代价：*残差流的幅度随深度增长，所以输出头之前需要一个 final norm；而且非常深的 pre-LN 模型在后段层可能出现表示塌缩——这就是 sandwich norm 之类变体存在的理由。

**Transformer 里为什么用 LayerNorm 不用 BatchNorm？**三个理由，面试官想听到不止一个。序列长度可变，所以 batch 统计量是在一组参差不齐、前后不一致的位置上算出来的。Batch 统计量把一个 batch 内的样本耦合起来，这会破坏 batch size 为 1 的自回归生成——你得靠 running statistics，而它和训练时永远对不上。以及在分布式训练里，BatchNorm 每次前向都需要跨设备同步。LayerNorm 在单个 token 的特征向量内部归一化，完全不依赖 batch 的组成。

**为什么要残差连接？**常见答案"它们解决梯度消失"只说对了一半。$$y = x + f(x)$$ 对 $$x$$ 的梯度是 $$1 + f'(x)$$，所以永远存在一条让梯度流过去的恒等通路。更好的框架是**残差流**视角：每一层从一条共享总线上读、也往上写，于是一个 100 层的网络可以表现得像一群更短路径的集成，而任何没用的层可以学会"几乎什么都不写"，而不必去学恒等映射 [15]。

**在一次 LLM 训练里，梯度爆炸的真实原因是什么，你怎么办？**通常不是深度——pre-LN 加残差已经处理了。实践中是一批坏数据、对当前曲率而言过高的学习率，或者 fp16 溢出。标准缓解手段：全局梯度范数裁剪（而且你应该**记录裁剪前的范数**，因为它的尖峰是你最早的预警）、用 bf16 而不是 fp16、以及 warmup。如果梯度范数经常尖峰，裁剪只是在掩盖问题而不是解决它。

<a id="optimization"></a>
### 优化

**Adam 和 AdamW——解耦到底修好了什么？**在 Adam 里，L2 正则是加到梯度上的，于是它会和其他东西一起经过同一套 per-parameter 自适应缩放。结果是有效的 weight decay 变得与梯度近期幅度**成反比**：梯度小的参数被狠狠衰减，梯度大的参数几乎不被衰减。这不是任何人说 weight decay 时的本意。AdamW [33] 把衰减直接作用在权重上、放在自适应步骤之外，恢复了本来想要的那种一致的向零拉力。这就是所有现代 LLM 都用 AdamW 的原因。

**Adam 为什么这么吃显存，能怎么办？**每个参数两份状态——一阶矩和二阶矩——都是 fp32。对一个 bf16 的 $$P$$ 参数模型，你大约要背 $$2P$$ 字节权重、$$2P$$ 梯度、$$4P + 4P$$ 优化器状态，再加上如果保留 fp32 master weights 的 $$4P$$：**每个参数约 16 字节，还没算任何激活。**这笔账就是 ZeRO 存在的全部理由，能随口算出来是很好的信号。

**为什么要 warmup？**Adam 的二阶矩估计在最初几百步是不可靠的，于是自适应的分母噪声很大，有效步长可能大到离谱。Warmup 让你在估计稳定之前保持小步。注意这里的交互：pre-LN 降低了对 warmup 的**需求**，但并没有消除来自优化器状态的那个理由。

**muP 是什么，为什么该关心？**标准参数化下，最优学习率会随宽度变化，所以每换一个尺寸都要重调——在前沿规模上这是承担不起的。Maximal update parameterization [54] 按层重新缩放初始化和学习率，使得最优超参对宽度不变。你在一个小的 proxy 模型上调，然后迁移到大模型。对"一次只能训一遍的 run，你怎么选超参"这个问题，这是个好答案。

**有比 AdamW 更新的东西吗？**Muon 确实有实质进展——它通过 Newton-Schulz 迭代把二维参数的动量更新正交化，已有报告称它能扩展到大规模 LLM 训练并带来有意义的效率收益 [32]。知道它的存在说明你在读当下的工作；对它有强烈观点大概不说明。

<a id="scaling-and-evaluation"></a>
### Scaling 与评测

**Kaplan 和 Chinchilla 的分歧是什么？**Kaplan 等人 [22] 发现了参数、数据、算力上的幂律，而他们的分析暗示在固定算力预算下应该把增量主要花在参数上。Hoffmann 等人 [16] 用更妥当的学习率调度处理重做了一遍，发现计算最优前沿大致是**等比例**放大——大约每个参数 20 个 token。这重塑了整个领域：模型变小了，数据集大了很多。

**Chinchilla 之后又变了什么？**Chinchilla 优化的是**训练**算力。如果你要把模型服务给几百万用户，推理成本会主导总成本，那么把一个更小的模型训到远超它计算最优 token 数就是理性的——Llama 3 就是这么做的，8B 模型用了约 15T token，比 Chinchilla 点高出几个数量级 [12]。对"最优模型大小是多少"的正确回答是一个反问：**对训练成本最优，还是对全生命周期成本最优？**

**Test-time compute 算第三个 scaling 轴吗？**算，而且形状最有意思。Snell 等人 [49] 表明在某些条件下，把算力花在推理上胜过花在参数上。对评测的实际含义是，单个 benchmark 数字现在是欠定义的——同一份权重，贪心解码和大搜索预算下，实际上是两个不同的系统。

**Perplexity 是什么，什么时候会误导？**每 token 负对数似然均值的指数——有效分支因子。它在**跨分词器**时会误导（不同词表不可比），它被简单 token 主导，而且在 RLHF 之后它通常会**变差**而模型变得更有用。永远不要跨不同分词器的模型比较 perplexity。

**当你没法检查答案时，怎么评测一个模型？**把阶梯说出来：有 verifier 的地方用精确匹配；没有的地方用人类偏好；用 LLM-as-judge 作为人类的可扩展代理，并说出它已知的失效模式（位置偏好、长度偏好、self-preference）；以及用成对比较而不是绝对打分，因为人和 judge 在**排序**上都比在**评分**上可靠得多。然后是污染问题，也就是实验室真正担心的那个：你怎么知道 benchmark 不在训练集里？n-gram overlap 检查、用训练截止日之后构建的 held-out 集，以及 canary string。

**Takeaway.** 先答**为什么**，再主动交出**什么时候会崩**。在面试官问之前就走到失效模式，是"正确答案"和"好答案"的分界线。

---

<a id="bucket-4-training-schemes-and-mechanisms"></a>
## 方向四：训练机制与训练方案

这就是你说的"training scheme 与 mechanism"，也是深度梯度最陡的一个方向。几乎每个候选人都能说出 PPO、GRPO、DPO 的名字。而能说清楚每一个里面 KL 项到底放在哪的人非常少——而那恰恰就是问题。

<a id="the-ladder-and-what-each-rung-can-fix"></a>
### 阶梯，以及每一级能修什么

把 post-training 想成一串分布迁移，每一级有明确的职责和明确做不到的事。

| 阶段 | 数据 | 能修什么 | 修不了什么 |
|---|---|---|---|
| 预训练 | 网络规模的无标注文本 | 知识、句法、世界模型 | 指令遵循、格式 |
| Midtraining | 精选的高质量、长上下文、代码、数学 | 领域能力、上下文长度 | 偏好、风格 |
| SFT | 示范数据 | 格式、指令遵循、工具语法 | 没被示范过的东西；它只能模仿 |
| Reward modeling | 偏好对 | 一个"更好"的标量代理 | 它自身的错误设定 |
| RL | prompt 加奖励或 verifier | 优化奖励，包括钻它的空子 | 基座模型不具备的知识 |
| 蒸馏 | teacher 输出 | 成本、延迟 | 一般来说超不过 teacher |

最有用的一句框架，随时要能说出来：**SFT 教模型"一个好答案长什么样"；RL 教它"它自己产出的答案里哪个更好"。**这就是为什么 SFT 饱和之后 RL 还能继续起作用——SFT 只能往示范的方向推，而 RL 能对模型自己的采样排序，把它推进没有任何人示范过的区域。这也是为什么 RL 装不进缺失的知识：它只能重新加权基座模型已经能产出的东西。

<a id="reward-models"></a>
### Reward model

标准的 reward model 是 Bradley-Terry。给定偏好对，训练一个标量头使得

$$\mathcal{L} = -\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)$$

其中 $$y_w$$ 优于 $$y_l$$。关于它有三件事要能说：

**奖励只在相差一个平移的意义下被确定。**Bradley-Terry 约束的是差值而不是绝对值，所以尺度是任意的，跨训练 run 比较原始 reward 数值毫无意义。这就是大家要对 reward 做 per-batch 归一化的原因。

**Reward model 是薄弱环节。**它是在一个很窄的回答分布上训练的，然后随着 policy 移动被在远离该分布的地方查询。对着它过度优化就是教科书级的 Goodhart 案例——而这正是 KL 惩罚存在的全部理由。

**能拿到 verifier 的地方，verifier 胜过学出来的奖励。**单元测试或数学检查器是一个函数，不是神经网络，没法用同样的方式被钻空子。从"这个答案是对的"到一个梯度，因果链短得多。

<a id="ppo-grpo-dpo-where-the-kl-lives"></a>
### PPO、GRPO、DPO：KL 到底放在哪

![PPO vs GRPO vs DPO](/assets/img/blog/frontier-lab-interview/fig7_posttraining.png)
*图 7. 面试官真正会探的那个对比。四个模型常驻 vs 三个 vs 两个；per-token 的 GAE vs 每条完成一个标量 vs 隐式的 log-ratio；以及关于 KL 约束加在哪里的三种不同答案。*

**PPO** [44] 在显存里放四个模型：policy、冻结的 reference、reward model，以及一个学出来的 critic。Critic 的存在是为了降方差——没有 baseline 的话，一个样本的 advantage 就是它的原始 reward，而梯度会被"某些 prompt 本来就更简单"这件事主导。Advantage 来自 GAE [43]，它在高偏差的一步 TD 和高方差的 Monte Carlo 之间插值：

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    T = len(rewards)
    adv = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + values
```

要能说出的性质：$$\lambda = 1$$ 退化为 Monte Carlo（无偏、高方差），$$\lambda = 0$$ 退化为一步 TD（有偏、低方差）。参考实现断言了这两个极限，这也是让自己相信实现正确的最便宜的方法。

Clipped surrogate objective 提供了一个软信赖域——它阻止任何单次更新把 policy 推得太远，而这正是朴素 policy gradient 所缺的。**在 PPO 里，KL 惩罚按惯例是从 reward 里减掉的**，然后再算 advantage。

**GRPO** [46] 去掉了 critic。核心洞察是：value function **只**在充当 baseline，而你可以对每个 prompt 采一组 $$G$$ 条完成、用它们的平均 reward 免费得到一个 baseline。

```python
def grpo_loss(logp, logp_old, logp_ref, rewards, mask,
              clip_eps=0.2, beta=0.04, group_size=None):
    """logp/logp_old/logp_ref: (B, L)。rewards: (B,)。mask: (B, L) 覆盖完成部分的 token。"""
    B = rewards.shape[0]
    g = group_size or B
    r = rewards.view(-1, g)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-4)
    adv = adv.reshape(B, 1)                       # 每条完成一个标量，沿 L 广播

    ratio = (logp - logp_old).exp()
    policy = -torch.min(ratio * adv,
                        ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)

    # k3 估计量：无偏，且恒 >= 0，不像原始 log-ratio
    log_ratio = logp_ref - logp
    kl = log_ratio.exp() - log_ratio - 1.0

    return ((policy + beta * kl) * mask).sum() / mask.sum().clamp(min=1.0)
```

三个细节才是真正的面试内容：

1. **KL 挪进了 loss 里**，作为 per-token 项，而不是被折进 reward。而且通常用的是 **k3 估计量** $$e^{-x} + x - 1$$ 而不是原始 log-ratio，因为 k3 既无偏**又**非负——朴素的 log-ratio 差值在单个样本上可能为负，那是一个没有意义的 KL 估计。
2. **Advantage 是 bandit 式的**：每条完成一个标量，广播到每一个 token。**完全没有 per-token 的信用分配。**这是一个真实的局限，也是值得主动点出来的。
3. **如果一组里每条完成拿到的 reward 都一样，advantage 精确为零**，这一组不贡献任何梯度。参考实现验证了这一点。在一个大多数 prompt 要么总是被解出、要么从不被解出的数据集上，你的大部分算力什么都没产出——而这正是 DAPO 的 dynamic sampling [56] 通过重采样直到组内 reward 有方差来修复的东西。

**DPO** [39] 完全跳过 RL。它的结论是：对于带 KL 约束的 RLHF 目标，最优 policy 和 reward 函数之间存在闭式关系，于是 reward 可以用 policy 自身表达出来。代入之后，偏好学习变成一个分类损失：

```python
def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """所有参数都是序列 log-prob 之和，形状 (B,)。"""
    pi_logratio = pi_chosen - pi_rejected
    ref_logratio = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()
```

没有 reward model、没有 critic、训练循环里不生成：对固定文本做四次前向，跑在和 SFT 相同的基础设施上、约两倍显存。在 reference policy 处 margin 为零、loss 恰好是 $$\log 2$$——一个不错的 sanity check，参考实现也断言了它。

要说出的取舍：DPO 是**离线**的，从一个固定的偏好数据集上学。随着 policy 偏离这些偏好被采集时的分布，信号会变陈旧。PPO 和 GRPO 持续从当前 policy 采样，更贵也更能适应。还有一个值得主动提的微妙点：DPO 的目标可以通过把 rejected 回答的似然**压下去**（而不是把 chosen 的抬上来）来增大 margin，这有时会导致两者的概率都下降。

**DAPO** [56] 是"最近有什么新东西"的自然答案。它报告了对 GRPO 的四项修正：*Clip-Higher*（非对称裁剪区间，让低概率 token 仍能被提升，防止熵塌缩）、*dynamic sampling*（丢掉所有 rollout 都打平的组）、*token-level policy loss*（而不是按序列平均，后者会低估长回答的权重）、以及 *overlong reward shaping*。

| | PPO | GRPO | DPO |
|---|---|---|---|
| 常驻模型数 | 4 | 3 | 2 |
| 训练循环内需要生成 | 是 | 是 | 否 |
| Advantage | GAE，per-token | 组均值，每条完成一个 | 隐式 log-ratio |
| KL 位置 | 在 reward 里 | 在 loss 里（k3） | 隐式，经由 reference |
| On-policy | 是 | 是 | 否 |
| 关键超参 | clip ε, γ, λ, value coef | G, clip ε, β | β |
| 主要失效模式 | critic 拟合、GAE 调参 | 全打平的组、组归一化不稳 | reference 漂移、β 调参 |

<a id="verifiable-rewards-and-reward-hacking"></a>
### 可验证奖励与 reward hacking

GRPO 之所以站住脚，真正的原因不是省显存，而是它天然配合**可验证**奖励。在数学和代码里你可以用一个函数检查答案。DeepSeek-R1 [9] 表明，从一个足够强的基座模型出发、对着 verifier 做 RL，能在无人示范的情况下长出长链推理。

预期这个追问：**当奖励可检查、但任务本身不可检查时，会崩在哪？**Reward hacking。值得点名的具体形态：模型针对测试套件写特例而不是解决问题；它找到评分器里的格式漏洞；它通过无效的推理得到正确的最终答案，而基于 verifier 的奖励检测不到，因为它只看答案。缓解手段：留出模型永远不会在上面训练的测试、验证过程而不只是输出、把 KL 缰绳勒短，以及监控推理轨迹的分布迁移而不只是 reward 曲线。

<a id="distributed-training-what-each-strategy-shards"></a>
### 分布式训练：每种策略切的是什么

组织性的问题不是"我该用哪种并行"，而是**"我到底是哪一项内存不够？"**

$$\text{memory} = \underbrace{P}_{\text{params}} + \underbrace{P}_{\text{grads}} + \underbrace{2P\text{–}4P}_{\text{optimizer}} + \underbrace{\text{activations}}_{\text{随 batch} \times \text{seq 增长}}$$

![每种并行切的是什么](/assets/img/blog/frontier-lab-interview/fig8_parallelism.png)
*图 8. 每种策略攻击的是不同的那一项。在说出策略名字之前先说出是哪一项，才是面试官在等的答案。*

- **DDP** 复制一切，对梯度做 all-reduce。简单，而且对显存毫无帮助。
- **ZeRO** [40] 依次切分优化器状态（stage 1）、梯度（stage 2）、参数（stage 3），并在需要时即时 gather 每一层的参数。**FSDP** [58] 是 PyTorch 的原生实现。Stage 3 用通信量换显存。
- **Tensor parallelism** [48] 把单个 matmul 切到多个设备上。它需要在**每一层内部**做 all-reduce，所以要 NVLink 级别的带宽——保持在节点内。
- **Pipeline parallelism** [19] 按层切。代价是 bubble：朴素调度下，设备在等前一段。Micro-batching 和 1F1B 调度能缩小它；$$p$$ 段、$$m$$ 个 micro-batch 时 bubble 比例大约是 $$(p-1)/m$$。
- **Context / ring attention** [31] 切序列，这是当单条序列的激活都放不下时唯一有用的东西。
- **Expert parallelism** 分布 MoE 的专家，配 all-to-all 路由。All-to-all 是瓶颈，也是 capacity factor 存在的原因。
- **激活重算** [24] 根本不是并行——它丢掉激活、在反向时重算，用大约多 30% 的算力换很大一块显存。

**3D parallelism** 就是 DP × TP × PP 的组合。标准布局：TP 在节点内最内层，PP 跨节点，DP 在最外层。

<a id="numerics"></a>
### 数值精度

混合精度 [35] 是让一次大 run 塞得下的另一半，之所以被问，是因为它的失效模式既具体又好记。

方案是：保留一份 fp32 的**master copy** 权重，用低精度跑前向和反向，把优化器更新作用在 master copy 上。之所以这么做，是因为更新量通常比权重小好几个数量级，在 fp16 里相加会直接被舍入成零——你的模型就这样停止学习，而 loss 曲线看上去还挺合理。

**bf16 为什么赢了 fp16。**两者位宽相同，但切分方式不同：fp16 是 5 位指数 + 10 位尾数，bf16 是 8 + 7。八位指数让 bf16 拥有**和 fp32 相同的动态范围**，也就是 attention logits 不会溢出，而且完全不需要 loss scaling 那套机制。代价是尾数精度，而事实证明训练并不太在乎。用 fp16 时你需要动态 loss scaling——在反向之前把 loss 乘大，把小梯度抬进可表示范围，再在优化器步之前还原，一看到 inf 就回退 scale。这套机制是"训练悄悄不再进步"类 bug 的常见来源，也是整个领域迁移的原因。

**什么必须留在 fp32。**归约。Softmax 分母、layer-norm 和 RMSNorm 的统计量、loss 累加，以及梯度 all-reduce。这就是前面 `RMSNorm` 实现里用 `.type_as(x)` 转回来的原因——归约在更高精度里做，只有结果被压窄。

**FP8** 是当下的前沿：DeepSeek-V3 用一套 FP8 混合精度框架完成了大规模训练 [8]，采用 per-tile 和 per-block 的缩放而不是一个全局 scale，因为 FP8 的范围窄到单个 scale factor 覆盖不了整个张量。如果被问到，诚实的答案是收益是真的，而数值工程确实很难。

**MFU** 是概括这一切是否奏效的那个数字：实际达到的 FLOP/s 除以硬件峰值。分子大约按 $$6 \times P \times D$$ 估算（$$P$$ 参数、$$D$$ token；一次前向-反向大约每参数每 token 6 FLOPs，如果做激活重算再加约 2）。大规模训练里 35–50% 是健康的。数字低的时候，原因几乎总是：通信没和计算重叠、pipeline bubble、data loader 喂不上、或者每设备 batch 太小。

<a id="debugging-a-loss-spike"></a>
### 调试一次 loss spike

这是完整版的 E2，因为它是被报告最多的"硬"机制题，而且它有一个你真的可以排练的结构。

*"你在训一个 100B 模型。第 42,000 步 loss 飙了。你怎么办？"*

错误答案是立刻提议降学习率。**正确答案从给这个 spike 分类开始**，因为三种形状原因不同、处理也不同。

![三种 loss spike 形状](/assets/img/blog/frontier-lab-interview/fig9_loss_spike.png)
*图 9. Bekman 的分类 [4]。关键的微妙之处：可见尖峰之前的那个 batch 几乎总是看上去人畜无害，因为问题在几百步之前就开始酝酿了。*

Bekman 那本关于 ML engineering 的开放书 [4]——写自训练 BLOOM-176B 和 IDEFICS-80B 的经历，是这个话题最好的公开参考——给了三种类型：**快速恢复**、**缓慢恢复**、**不完全恢复**。他对常见原因的判断：

> *"这些尖峰通常源自一段坏数据，要么是数据 shuffle 得不好，要么是没从网上爬来的垃圾里清理干净。"* [4]

以及那个让它成为好面试题的微妙点：

> *"人们会怀疑是尖峰前的那个 batch 触发的，但如果你去研究那个 batch 的内容，很可能什么异常都找不到——问题往往在很多步之前就开始发展，然后突然之间就发生了。"* [4]

一个有结构的答案会从最便宜的往最贵的走：

1. **它是真的，还是日志问题？**检查尖峰是否同时出现在梯度范数和验证 loss 里，还是只出现在某一个 rank 上被平滑过的训练曲线里。
2. **它是 resume 造成的假象吗？**这是价值最高、而几乎没人会提的一条检查。如果 run 重启过而 data sampler 没有恢复位置，模型就在重读它已经见过的 token。Bekman 的警告很刺眼：你可能事后才发现自己 *"把原计划各看一次的 300B token，变成了同样的 50B token 训了 6 遍"* [4]。这不是尖峰的病理，而是一次被悄悄作废的训练。
3. **是硬件吗？**单张有问题的 GPU 产出坏梯度就会毒化 all-reduce。检查 per-rank loss、找 ECC 错误、跑一次集合通信 benchmark。
4. **是数值吗？**fp16 下 attention logits 溢出，或者 loss scaler 崩了。在裁剪之前检查梯度里的 inf/NaN。这也是 bf16 取代 fp16 的很大一部分原因——动态范围和 fp32 相同，不需要 loss scaling。
5. **是数据吗？**现在再去看尖峰**之前那个窗口**里的 batch，而不是尖峰那一刻的 batch。长串重复 token、损坏的 shard、或者语言切换都是典型。
6. **最后才轮到优化。**学习率对当前曲率是不是太高？调度变更之后二阶矩估计是不是陈旧了？某次 warmup 重启是不是被跳过了？

对应的处理，与分类一一对应：快速恢复的，记一笔然后继续。缓慢恢复的，考虑降学习率或跳过那段数据。不恢复的，回滚到上一个好的 checkpoint 并用不同的数据顺序重启——注意这时你应该**已经**对 checkpoint 频率有一套策略，因为这本质上是"你愿意损失多少算力"的问题。

值得说出口的元层面观点：**去查公开的 training logbook。**Bekman 专门维护了一份合集，正是因为你遇到的不稳定大概率已经被记录下来、连缓解手段一起 [4]。说一句"我会先确认这是不是一个已知失效模式，再去构造假设"，比任何具体假设都更像一个更好的工程师。

<a id="question-bank-training"></a>
### 题库

**为什么 LLM 的 value function 难训？**奖励稀疏（整条回答一个标量）、目标分布随 policy 改进而迁移所以 critic 永远滞后、以及它是显存里的又一个全尺寸模型。这三条论证都指向 GRPO。

**什么时候 GRPO 是个坏选择？**当奖励稠密或 per-token 时；当你负担不起每个 prompt 采 $$G$$ 条时；以及当组内 reward 方差很低时——那会让大多数组什么都不贡献。对没有 verifier 的偏好类奖励，GRPO 能用，但它相对 PPO 的优势就不那么明显了。

**KL 系数怎么选？**别凭感觉挑——**盯住一个目标 KL 值**。监控相对 reference 的实际 KL 散度，自适应调整 β 让它维持在目标附近。KL 接近零说明模型没在学；KL 无界上涨则会带来 reward hacking 和能力损失。

**PPO 的 KL 和 GRPO 的 KL 有什么区别？**位置（reward 里 vs loss 里）、估计量（原始 vs k3）、粒度。这是最可靠的"这个人到底有没有实现过"的问题。

**为什么需要 reference model？**为了限制相对 SFT policy 的漂移。没有它，优化一个学出来的 reward model 会找到它的失效模式，reward 数字上涨而 policy 丢掉通用能力。

**讲一遍完整的 RLHF pipeline。**预训练、在示范数据上 SFT、在 policy 采样上收集偏好比较、训练一个 Bradley-Terry reward model，然后用 PPO 对着这个 reward 优化并加上到 SFT policy 的 KL 惩罚 [37]。然后说清楚之后变了什么：能用 verifier 的地方用可验证奖励、推理工作里用 GRPO 取代 PPO、有静态偏好数据且想简单时用 DPO，以及用多轮迭代而不是一遍过 [28]。

**Pipeline parallelism 的 bubble 是什么，怎么缩小？**某一段在等待时的空转时间。用更多 micro-batch、交错式 1F1B 调度，或者把反向拆成"输入梯度"和"权重梯度"两半的 zero-bubble 调度。

**为什么用 bf16 而不是 fp16？**指数范围和 fp32 相同，所以不需要 loss scaling、溢出故障少得多，代价是尾数精度——而这被证明对训练影响不大。保留 fp32 master weights，归约在 fp32 里做。

**Takeaway.** 谁都能报出这些算法的名字。信号在于 KL 项放在哪、advantage 是什么形状，以及你面对 loss spike 的第一个动作是不是去查 data sampler。

---

<a id="bucket-5-ml-systems-design"></a>
## 方向五：ML 系统设计

经典系统设计在这类 loop 里基本被取代了。有一份候选人记述说得再直白不过：没有人让他设计 URL shortener，没有人问聊天服务；题目是**给数百万请求、多种模型尺寸做推理服务基础设施，同时保持 GPU 利用率不掉**——batching 怎么做、KV cache 内存怎么管、请求路由到哪个实例、延迟又是怎么在 transformer pipeline 里越积越多。

从这里能得出两件事。第一，**面试官搭过真的那一套。**被报告的经历里反复出现"聊不到两分钟就被看出是拿通用材料准备的"。第二，题目高度聚集：批量 GPU 请求、推理系统、model downloader、prompt playground、评测 harness。

<a id="prefill-and-decode-are-two-different-machines"></a>
### Prefill 和 decode 是两台不同的机器

如果这一节你只带走一件事，就带走这个。几乎所有服务侧的决策都从一个区分里推出来，而在头两分钟把它说出来，会重新定义整场对话。

![Prefill 计算受限；decode 访存带宽受限](/assets/img/blog/frontier-lab-interview/fig10_prefill_decode.png)
*图 10. Prefill 并行处理整个 prompt，把 GPU 的算术单元喂满。Decode 一次产出一个 token，一生都在等内存。它们瓶颈不同、SLO 不同、修法也不同。*

**Prefill** 处理 prompt。每个 token 都要和其他所有 token 做 attention，所以"每读入一字节权重能做的并行工作量"很大。算术强度高，你位于 roofline 拐点的右侧，是**计算受限**的。成本随 prompt 长度的平方增长。

**Decode** 生成一个 token。你要读完整个权重矩阵——几十 GB——只为算出一个 token 的算术量。算术强度极差，你远在拐点左侧，是**访存带宽受限**的。GPU 的 FLOPs 几乎完全闲置。

这一个事实基本解释了整个服务栈：

- **Batching 是 decode 的主要杠杆**，因为它把权重的读取摊到很多条序列上，把你在 roofline 上往右推。它对 prefill 几乎没用，因为 prefill 本来就饱和了。
- **Continuous batching** 之所以存在，是因为静态 batching 浪费了尾部：固定 batch 下，短序列要空等最长的那条结束。Continuous batching 在每一步驱逐已完成的序列、放入新的。
- **KV cache 分页** [25] 之所以存在，是因为 cache 才是限制你能 batch 多少条序列的东西，而朴素的连续分配碎片化很严重——你必须按最大可能长度预留。把 cache 分页成固定大小的块（完全就是虚拟内存）消除了碎片，还让 copy-on-write 的前缀共享成为可能。
- **Chunked prefill** 之所以存在，是因为一个超长 prompt 会独占 GPU、拖住所有人的 decode 步，毁掉每一个并发用户的 inter-token latency。
- **前缀缓存**之所以存在，是因为否则共享的 system prompt 会在每个请求里被重算一遍。
- **Prefill/decode 分离部署**之所以存在，是因为两个阶段想要的硬件配比不同，共享设备时会互相干扰。
- **Speculative decoding** [29] 之所以存在，是因为 decode 有闲置的 FLOPs。一个小的 draft 模型提出若干 token，大模型在一次并行前向里验证它们，你用富余的算术换到 token。关键的注意事项，也是他们会追问的：**它的收益随 batch size 增大而缩小**，因为在高 batch 下你已经不再是带宽饥饿的，也就没有富余 FLOPs 可用。

你要负责的两个 SLO 是 **TTFT**（首 token 时间，由 prefill 和排队决定）和 **TPOT**（每输出 token 时间，由 decode 决定）。它们互相权衡，而吞吐又和两者权衡。搞清楚产品在乎哪一个，是你该问的第一个问题。

<a id="worked-design-serve-a-family-of-models"></a>
### 完整设计：服务一整个模型家族

下面是那道被报告过的 Anthropic 风格题目，按我会用的方式走一遍。

*"设计一个推理服务，给数百万日请求提供多种尺寸的模型，同时保持 GPU 利用率很高。"*

**第一步：澄清，然后把算术**大声**算出来。**这是候选人会跳过的一步，也是最能体现经验的一步。先问：几个模型、多大？请求速率和峰均比？prompt 长度和输出长度的分布？交互式还是批处理？延迟 SLO 是多少，卡的是 TTFT 还是端到端？有没有共享的 system prompt？

然后给定数字并算下去。假设每天 1 亿请求——平均约 1,150 QPS，峰值算 3,500。假设 bf16 的 70B 模型：140 GB 权重，一张 80 GB 卡放不下，需要至少 2 张、实际上 4 张做 tensor parallel。GQA 用 8 个 KV 头时 cache 是每 token 320 KB，所以 10k token 的上下文每条序列约 3.2 GB。4×H100 共 320 GB，减去 140 GB 权重，扣掉开销后大约剩 160 GB 可用 KV 空间——在那个上下文长度下约 50 条并发序列。如果一个请求占用一个槽位 5 秒，那么一个 4 卡副本大约服务 10 QPS，峰值需求就需要约 350 个副本、也就是 1,400 张 GPU。**到这时**你才有了谈架构的依据，而且你已经展示了你能给系统做容量估算。

**第二步：请求路径。**Gateway（鉴权、限流、配额）→ router → 各模型的副本池 → scheduler → engine。Router 负责选模型和感知负载的放置；scheduler 负责准入和批处理。

**第三步：scheduler，真正的内容在这里。**Continuous batching 配分页 KV。一个 running batch，只要有 cache block 释放就放入新请求。Chunked prefill，让长 prompt 被切块、与 decode 步交错而不是阻塞它们。如果交互式和批处理流量共享硬件，就要有优先级分类。以及一条明确的**抢占策略**，用于 cache 打满时：要么之后重算被驱逐序列的 prefill，要么把它的 block 换出到主机内存。知道这个决策存在本身就是很强的信号。

**第四步：多模型。**朴素做法是按模型独占 GPU，但流量倾斜时会浪费容量。要提出并评估的选项：独立池 + 自动扩缩（简单，适应慢）；在一个设备上复用多个模型（只对小模型可行，KV cache 让它很难）；LoRA 式多租户，让很多 adapter 共享一份基座权重（当变体都是微调时非常好用）；以及冷启动缓解，因为加载 140 GB 权重要几分钟——你需要预热池，以及一个 model downloader / cache 服务，而这本身就是被报告过的题目之一。

**第五步：什么会崩。**一个超长请求把 cache block 占住几分钟。一次流量尖峰打满 cache 并触发抢占风暴。一个坏节点拖垮整个 tensor-parallel 组——TP 是同步的，最慢的设备决定节奏。队列里的 head-of-line blocking。以及成本问题：你的每百万 token 成本是多少，哪个旋钮影响最大？（通常是 batch size，然后是量化，然后是低负载下的 speculative decoding。）

**第六步：你会测什么。**TTFT 和 TPOT 的 p50/p95/p99，不是均值。每 GPU 每秒 token 数。KV cache 利用率和抢占率。Batch size 分布。队列深度。以及 **goodput**——**在 SLO 内**被服务的请求数，这才是真正重要的指标，也是候选人很少说出来的那个。

<a id="question-bank-systems-design"></a>
### 题库

**为一个 100B 模型设计训练集群。**先算内存方程：AdamW 加 fp32 master weights 时每参数约 16 字节，意味着不算激活就已经 1.6 TB，所以你需要模型分片而不只是数据并行。然后是布局：节点内 TP=8，跨节点 PP，其余交给 ZeRO/FSDP。然后是人们会忘的部分——checkpoint 频率与重启成本的权衡、数据管线的吞吐（它必须喂得动几千张 GPU）、故障检测与自动重启，以及把 MFU 当作健康指标来监控。

**为一个 coding agent 设计评测 harness。**沙箱执行是全部难点：无网络的容器、资源限制、超时。然后是确定性——锁定依赖、固定随机种子，以及对 flaky 测试的处理策略。然后是评测延迟问题：如果一个任务要跑一小时，你就没法迭代，所以你需要跨任务并行和一个快速冒烟子集。然后是污染：从公开仓库构建的 benchmark 可能已经在训练数据里了，所以你需要在截止日之后创建的 held-out 任务。

**设计一个 RAG 系统，然后把它拆坏。**设计本身是标准的；有意思的一半是失效分类。检索没命中（修法：BM25 + 稠密混合、query 改写）、检索命中了但生成器无视上下文（修法：指令微调、强制引用）、分块边界把答案切断（修法：重叠分块，或 parent-document 检索）、以及索引陈旧。然后是评测：检索质量和生成质量必须分开测量，否则你根本判断不出是哪一半坏了。

**你会怎么把推理成本砍一半？**按投入产出比排序：把 batch size 抬到延迟 SLO 卡住为止；如果有共享 prompt 就开前缀缓存；把权重量化到 INT8 或 FP8；把简单 query 路由到小模型；如果负载低且是交互式的就加 speculative decoding；最后才考虑蒸馏。说出你会先试哪一个，以及为什么。

**Takeaway.** 先算术，再架构，并且尽早把 prefill/decode 的区分说出口。坐在对面的人运营过这套系统，两分钟内就能判断你有没有。

---

<a id="bucket-6-research-taste-deep-dives-and-values"></a>
## 方向六：Research taste、深挖与价值观

这是决定 loop 的那个方向，也是临时抱佛脚抱不动的那个。Meng 的判断值得坐下来想一想：

> *"Coding 面试筛掉弱的候选人，但恰恰没有人是因为 coding 表现好而被录用的。设计面试考的是第一性原理思考，但你可能在这个领域还从没成功过。项目深挖则不同，它精确地展示了为什么你是这个团队要的那个人。"* [34]

<a id="the-research-presentation"></a>
### Research presentation

研究方向的岗位可能要求你做一场关于自己工作的 job talk：大约十页，像答辩一样被质询。Meng 的描述是 *"一种 job talk 式的、关于你过往工作的报告……你要像博士候选人那样'defend'你的整个工作体系"* [34]。Liu 的版本更短——比学术版短，聚焦单篇论文或单个方向，而且 *"有时这种报告只有 20 分钟"* [30]。

一个管用的结构：

1. **问题，以及它为什么还没被解决。**一页。如果听众听完这一页还不在意，后面什么都落不下来。
2. **你具体的贡献**，说成一个你能辩护的**主张**，而不是一串活动清单。
3. **方法**，把那个真正重要的设计决策单独给一页。
4. **结果**，包括那个你差点输给的 baseline。
5. **什么没成功**，以及你学到了什么。这一页是把研究者和"汇报结果的人"分开的那一页。
6. **你接下来会做什么**，以及你怎么知道自己是对的。

最常见的失败是讲成了编年史而不是论证。第二常见的是答不上"这里面最弱的一环是什么"——**准备一个真实的答案**，因为你能发出的最强信号，就是你已经当过自己最严厉的评审。

Lambert 从招人一侧补了一句，对内容与形式的权衡说得很准：*"在 job talk 上多花时间。它不是关于技术内容，而是关于传达一个愿景和一个故事。说实话，我收到的反馈越多，就越觉得自己该少放几张图。"* [26]

<a id="the-paper-round"></a>
### 论文轮

有些实验室会提前把论文发给你，有些当场给你一篇。无论哪种，标准都不是"它讲了什么"，而是**你接下来会做什么**。

一个可复用的框架，在你没读过那篇论文时同样管用：

- **主张。**它到底断言了什么？把经验性主张和作者附加的解释分开。
- **证据。**什么实验支持它？什么能证伪它？baseline 公平吗——同样的算力、同样的数据、同样的调参投入？
- **机制。**为什么这会有效？如果作者给了解释，这个解释能不能预测他们没测过的别的东西？
- **适用范围。**放大 10 倍会怎样？换个模态呢？换个更强的基座呢？
- **下一步。**你会跑的那一个实验，以及每种结果会告诉你什么。

Jaiswal 保持前沿感知的系统是我找到的最实用的做法，值得整套照搬。她维护了一个她**特意**命名为 **"I know of these papers"** 的板块——用 "know" 这个词是因为它 *"涵盖了多个层次的熟悉度：深读过、扫过、从社交媒体上看到过、从别人的报告或讨论里了解过、以及亲手实现过"* [21]。配上一条我认为是很好的面试建议的诚实原则：

> *"当面试官问开放性问题时，我会特意注明来源，比如说'我是从一篇博客里学到的'或者'我在 Twitter 上看到很多人讨论这个'。我维护着一个很广的参考集合，并且始终对自己在每篇论文上的理解深度保持透明。"* [21]

**校准过的自信读起来是强，不是弱。**"这篇我只扫过，但我的理解是 X——对吗？"是一个远好于含糊其辞的答案，而且它把面试官请进一场对话，而不是一场考试。

<a id="the-project-deep-dive"></a>
### 项目深挖

每个 loop 都有一轮，而且对大多数人来说这是杠杆最高的一轮，因为它是唯一一轮以你的真实经历为素材的。

被反复报告的模式是不留情面的追问：面试官会一直问**为什么**，直到碰到基岩或者你的边界。你具体做了什么（相对于你的团队）？为什么是这个设计而不是那个显而易见的替代方案？取舍是什么？你怎么衡量的？出了什么问题？停留在高层会被读作没有真正深入。

管用的准备方式：把项目用手写的方式全部铺开，所有发生过的事，然后组织成章节——设计、开发、上线、教训、下一步——并且把技术复杂度和协作复杂度分开，因为面试官两者都会探。然后找出那两三个你在岔路口选了一边的决策，并且能为两边都辩护。

这一轮最难的问题通常是：*"你会做哪些不一样？"* 一个实质上是变相自夸的答案会挂。一个指出真实误判、解释你当时判断错了什么、以及你现在相信什么的答案，会过。

<a id="the-values-round"></a>
### 价值观轮

没人把这一轮当技术轮来准备。而它被反复报告为淘汰技术很强的候选人比任何一轮 coding 都多。

它让人失败有三个原因。它不是文化闲聊，所以关于团队冲突的 STAR 故事落不了地。它探的是推理而不是结论，所以无论你说什么，追问都是"为什么？"，然后是"什么会让你改变看法？"。而且它能检测出背诵——像"一个你改变过的信念"、"一个你对这家公司的真实批评"、"哪种 LLM 失效模式最让你担心，给 agent 加上工具使用之后又有什么变化"这类问题，是没法照稿子答的。

真正管用的准备：

**读一手材料，并且对某个具体的点持不同意见。**如果你面 Anthropic 或 OpenAI，就分别是 Constitutional AI [3] 和 InstructGPT [37]；再加上 model card、system card、公开的安全框架。Anthropic 自己的候选人指引甚至明确建议用 Claude 建一份学习提纲，涵盖 *"我该复习的关键主题，包括 AI safety 概念、[公司的]研究重点"* [2]——所以这是被期待的，不是可选的。但**读只是地板**。有一个具体的、论证扎实的不同意见，才是信号。

**对具体的东西形成立场，而不是对笼统的东西。**"我关心 AI safety"是噪声。"我认为目前反对把 RLHF 当作安全机制的最强论证是：它训练模型产出人类**会赞同**的输出，而这和**正确**的输出是两个不同的目标——随着任务超出人类的评估能力，这个缺口是在扩大而不是收窄"，这才是立场。它可以被反驳，而这正是重点。

**要能承住不确定性。**这一轮考的是你能不能在真实的不确定下推理，而不塌缩成虚假的自信或拒绝表态。做法是：说出你的看法、说出你的置信度、说出什么证据会让你改变。

**要有真实的故事，诚实地讲。**一次你的价值观被考验的时刻。一次你在重要的事情上想错了。一个你会做不同选择的决定。这些必须是真的，因为追问会往下挖三层，编的故事会没有细节可给。

**Takeaway.** 这个方向的提前量以月计，压缩不了。它也是准备过程和"单纯做一个对自己领域有想法的人"几乎无法区分的那个方向——所以最先开始它，成本为零。

---

<a id="bucket-7-math"></a>
## 方向七：数学

在 Liu 把它单列出来之前，我没在任何地方见人写过这一轮：

> *"有些公司有一轮数学面试，从有趣的逻辑谜题，到需要纸笔的严肃数学推导都有。我建议把概率、线性代数和微积分复习一下。"* [30]

她为此专门写了一整份笔记，*"全都是为了那一场决定性的面试"* [30]——这既说明它是真的，也说明它罕见到足以成为一个难受的意外。Sapora 的主题清单里独立地带着一整块线性代数：半正定、Jacobian、Hessian、特征值与特征向量、零空间与像空间、正交性、线性无关、奇异矩阵、秩与张成、行列式 [42]。Lambert 在 2022 年也见过同样的东西，他描述有些公司跑 *"'ML Background' 面试，主要是数学技巧和基本的 ML 取舍"*，并且承认 *"为这个做准备很难，学点课程内容会有帮助。我没学"* [26]。

范围比资格考窄得多。真正反复出现的是：

**概率。**期望和方差，特别是把**期望的线性性**当作证明工具用得很顺。条件概率和 Bayes。常见分布以及各自在什么情况下出现。几个集中不等式要能叫出名字（Markov、Chebyshev、Hoeffding）并说清各自买到了什么。马尔可夫链和平稳分布。以及那一类经典谜题——看到某个模式所需的期望抛硬币次数、选票问题、生日碰撞论证——这通常就是"有趣的逻辑谜题"在实践中的意思。

**带 ML 口音的线性代数。**特征分解和 SVD，关键是**它们意味着什么**而不是怎么算：SVD 给你最优低秩逼近，这就是 LoRA 为什么说得通、PCA 为什么有效。半正定性，以及为什么协方差矩阵和极小点处的 Hessian 是半正定的。秩、零空间，以及为什么一个秩为 $$r$$ 的更新有 $$r(m+n)$$ 个参数。矩阵微积分：Jacobian、矩阵形式的链式法则，以及那一小撮让你不用查资料就能推出反向传播的求导恒等式。

**微积分与优化。**梯度和 Hessian，凸性以及它为什么重要（还有为什么没人真有它），把 Taylor 展开当作梯度下降的正当性来源，Lagrange 乘子，以及 Jensen 不等式——它出现得极其频繁，因为它是 ELBO 背后的引擎。

**信息论。**熵、交叉熵、KL 散度。要准备好回答 Sapora 说自己答错、事后哭了一场的那个问题，因为它是**那个**经典题：**为什么 forward KL 是 mean-covering 而 reverse KL 是 mode-seeking？**答案在于那个无穷惩罚坐在哪一边。Forward KL，$$D_{KL}(p \| q)$$，按 $$p$$ 加权，所以凡是 $$p$$ 有质量而 $$q$$ 没有的地方你都要付出巨大代价——拟合出的 $$q$$ 因此必须覆盖 $$p$$ 的全部支撑集，在多个模态之间摊开。Reverse KL，$$D_{KL}(q \| p)$$，按 $$q$$ 加权，所以 $$q$$ 会因为把质量放在 $$p$$ 没有质量的地方而受罚，却完全不因为整个忽略掉一个模态而付出代价——于是它塌到一个模态上。极大似然是 forward KL；变分推断和 RLHF 的 KL 惩罚是 reverse。她在"两篇论文里都处理过 forward vs reverse KL"之后仍然答错了 [42]，这件事本身就是"要排练你已经会的东西"的全部论证。

**Takeaway.** 面积小，方差大。一个周末的概率和线性代数复习，是对一轮可能没人提前告诉你的面试的廉价保险。

---

<a id="gate-3-timing-work-trials-and-negotiation"></a>
## 第三道门：时机、work trial 与谈判

到这里为止的一切都是人们会准备的部分。而按所有近期亲历者的说法，接下来这部分决定了出乎意料大的一块结果——却几乎没有人排练它。

<a id="sequencing-your-companies"></a>
### 给你的公司排序

标准建议是拿几家公司练手，然后把其余流程的时间调整到 offer 一起落地。Liu 说这在精神上大致正确，然后补了三条比那条规则本身更重要的修正 [30]：

**你的耐力是有限的。***"练手面试有帮助，但也要认识到你的耐力是有限的——别等到了你真正在意的地方，人已经烧干了！"* Sapora 用同一套打法但更细：从 *"更小的创业公司、你不太想去的地点、有意思但不是首选的岗位"* 开始——你在还不重要的时候校准了自信，也摸清了薪酬的样子 [42]。

**外部因素可能盖过你的准备。***"时机上有一些外部因素值得考虑，比如公司有没有 headcount、哪些团队在积极招人，而这可能比你的准备更重要"* [30]。Yong 独立得出同一个结论，并说要直接问 recruiter 关于 headcount 的情况 [55]。Lambert 从内部确认了：公司会在周期中途遇到 *"总 headcount 冻结"*，而你应该和同领域正在找工作的人聊聊，好确认不是你的问题 [26]。

**Deadline 的弹性比看上去大。**Liu：*"Deadline 有很大的灵活性……recruiter 明白你还有别的流程要走完，而且有各种技巧可以推迟 offer 和决定"*——但要专门去调查所谓的 exploding offer [30]。Sapora 见过的 deadline 从一周、两周到"你花个合理的时间"都有，而且她遇到的公司在延期上不灵活，尽管她有朋友争取到了延期 [42]。所以：把灵活性当作大概率但不保证的事，并且尽早问清楚。

还有两条操作性规则。**把你手上其他流程告诉每一家公司**——Sapora：*"我知道这对有些人来说不舒服，但这完全正常而且是被预期的。它让时间线保持清晰、推动流程顺畅往前走，而且如果他们真的有兴趣，往往会让他们加快。"* [42] 以及**尽可能一天只面一场**：*"面试很耗人，你在当天的第三场里自然会发挥失常。"* [42]

Yong 关于流程被压缩的警告是另一侧的配重：*"别惊讶于你可能要在同一天连着面三场，而且只有不到一天时间准备它们"* [55]。把一个流程的**开始**往后推一两个月是正常的；一旦开始，间隔就很短了。

还有一件候选人会忘记自己有权做的事，来自 Lambert：*"你可以拒绝公司。如果他们在做一些离谱的事，比如不沟通日程、或者说只有一个面试时间段，就要求他们给你腾出空间。"* [26]

<a id="work-trials"></a>
### Work trial

我原本把这个当成内容农场编出来的东西，因为只有内容农场提到它。它是真的，而 Yong 把它描述为整个求职过程里最大的意外 [55]：

> *"Work trial 和 onsite 完全不同——你不是被飞到公司去连着面好几轮；而是和团队一起解决一个任务。有时候任务是开放式的。这些 work trial 通常是带薪的，但让我意外的是，有些线下的 work trial 可以长达一周。"*

真正要内化的是它的实际后果，因为它会摧毁你的排程：

> *"做 work trial 让我很难为其他公司的面试做准备，因为我必须把全部精力投进当前这个任务，完全没有余力去准备别家的面试。"* [55]

如果你的 pipeline 里有 trial，就把它当成一段**封锁期**，其他一切围绕它排。另外要注意，trial 是一次**双向**评估——在团队里待一周，你对"想不想要这份工作"的了解，超过任何数量的 team match 通话。

<a id="negotiation"></a>
### 谈判

这是整个流程里每小时价值最高的工作，也是准备得最少的。Liu 关于它的那一段是所有这些材料里最有用的一段：

> *"事实是谈判很难。博士生涯里没有任何东西为此做过准备，而且与面试不同，这一部分没法靠学习攻克。相比 recruiter，你在市场知识和谈判技巧上都处于劣势，而且你接触的每个人都想从你这里得到不同的东西。你可能在想，'我对我的 offer 已经很满意了，我会独立于薪酬做决定！'——知道自己的价值观当然很好！但如果你不谈判，你是在亏待自己。初始 offer 在设计上就留了谈判空间；recruiter 经常明确邀请你入局，比如说'我不指望你接受我们的第一份 offer'。**在这里投入几周的精力，字面意义上可以等价于在初始 offer 上工作好几年。**"* [30]

她的方法才是可迁移的那部分，而它其实就是把面试准备指向了另一个目标：

> *"每次 recruiter 通话之前，我都会写下我愿意和不愿意透露什么，以及一些我可以逐字背出来的话。在 offer 之后的阶段，我会预判他们可能问的问题和可能提的点，并仔细构造我能自如说出、同时仍然在为自己争取的回应。"* [30]

也要注意 offer 后阶段本身就是一大块工作——她在其他一切之外还记录了 **16 次 offer 后沟通**，并描述那是 *"在应付多到压不住的沟通量"* [30]。

Sapora 的经历给通常的建议加了三条经验性修正 [42]：

**"盲拍"策略没能扛住现实。**标准建议是绝不透露竞争 offer。*"这对我没用：好几家公司在提价之前明确要求看到其他 offer 的证明，还有一家质疑我的截图。"*

**公司手上有你没有的数据。***"如果你告诉 Anthropic 你在认真考虑 Peppers Burgers 的 offer，他们有数据知道同时拿到这两个 offer 的候选人有多大比例真的选了后者。如果答案是'几乎从不'，你的虚张声势就不成立。"* 推论是：竞争 offer 只有来自真正的同级对手才有分量。

**你在被持续解读。***"Recruiter 读人的能力出奇地强……即使很小的信号也重要：你提到某家公司的频率、你谈论他们的方式，全都会被记下来。如果 recruiter 知道他们公司已经是你的首选，谈判就会更难。"*

两人都毫无保留地说的一件事：**如果公司想要你，他们的数字能动很多，而且永远值得开口问。**

还有两件事属于这里。依靠朋友——Liu：*"在这个阶段，非常关键的是依靠你的朋友，获取和 recruiter 打交道的门道，以及更多用来校准你要价的数据点"* [30]。至于在多个 offer 之间做选择，Sapora 诚实地记录了她去问遍两家公司所有人的结果：*"DeepMind 的每一个人都告诉我他们会选 DeepMind，Isomorphic 的每一个人都告诉我他们会选 Isomorphic。非常有帮助。"* 真正解决问题的是和**了解她本人**的人谈 [42]。

**Takeaway.** 给这件事留出两周，并且像准备技术轮一样排练它。这是整个流程里唯一有直接、即时且永久财务回报的部分。

---

<a id="the-part-nobody-prepares-for"></a>
## 没人会教你准备的那部分

Liu 和 Sapora 都用了不小的篇幅写心理代价，而且都异常直接。我把它写进来，是因为它是被报告得最一致、却被写得最少的一部分，而且**提前读到它本身就是一种准备**。

Liu：

> *"这篇文章里我聚焦在求职的具体环节上，但实际上我个人体验中很大一部分，是在处理身处求职市场所带来的全部情绪。有很多社会感知需要应付：把自己和同辈比较不是好受的事，每个人都对你该去哪不该去哪有意见，而且人们会异常投入地关心你的人生过得怎么样……坦白说，有好几个月我压力很大、很痛苦，生活的其他部分都运转不下去。希望你能找到更多快乐；如果没有，至少知道你不是一个人。"* [30]

Sapora 对其中的机制写得更细。她面试前一晚睡不着，而这在 *"你一周有 10 场面试"* 时会变成结构性问题。她吃不下，看到食物就恶心。她的应对是运动（*"面试前跑步对我帮助很大，它烧掉了紧张的能量，也让脑子重置"*）、固定的晚间流程、一套固定的面试前仪式，以及一条规则：只要第二天早上没有面试，当晚就和朋友吃饭 [42]。然后是更难的部分：

> *"到某个时刻，拖住我的已经是我的焦虑而不是我的准备了。我的大脑偶尔会在面试中途一片空白……事后想想，那种反思更该在你**开始之前**做（了解你的触发点、你和失败的关系、你的价值感究竟系在什么上面），这样你就不必像我那样在炮火下才发现它们。"* [42]

有三件事值得作为真正的建议提取出来。

**睡觉胜过临时抱佛脚，而且差距不小。**Liu 在只睡了两小时、临时把 LLM 推理的细节塞进脑子之后去面了她的第一场技术面：*"临时背的东西一个都没考到，而我花了 10 分钟卡在一个 off-by-one 错误上，因为我的齿轮几乎转不动。"* [30] 半夜多复习一小时的期望价值是负的。

**这个过程是随机的，它的判决不是关于你的信息。**Sapora 在答错了一道她本人发过论文的 forward-vs-reverse KL 题之后写道：*"你作为一个人的价值不会由这些面试决定……你会搞砸，甚至会在你本来会的东西上搞砸，而这没关系。"* [42]

**在开始之前做那场自省，而不是在过程之中。**这是她最清晰、也最不花成本的一条建议。提前知道自己面对失败会怎么反应，比再刷二十道 LeetCode 值钱。

我还想从这些材料的来源本身补一个观察。Liu、Sapora 和 Yong 都是在**成功之后**才写下这些复盘的——分别拿到了 OpenAI、DeepMind，以及他自己满意的实验室。**如果连毫无疑问做得很好的人都把这个过程描述为痛苦，那么你觉得它痛苦，并不能作为它进展不顺的证据。**

**Takeaway.** 按"一个即使顺利也会不好受的、以月计的过程"来做计划，并且在需要之前就把睡眠、运动和社交的支架搭好，而不是之后。

---

<a id="sequencing-your-preparation"></a>
## 准备顺序

各个方向的提前量差别很大，而应该驱动顺序的是提前量，不是重要性。

![按提前量排序](/assets/img/blog/frontier-lab-interview/fig11_plan.png)
*图 11. 可见度以年为单位积累，急不来。谈判是两周的工作，却是整张图上每小时回报最高的一项。Fundamentals 可以缓存，属于最后阶段。大多数人把这件事做反了。*

从四份材料里能提炼出的规则：

**先拿到轮次清单。**下游一切都依赖它，而泛泛地准备意味着在你根本不会遇到的轮次上过度投资。

**把 ML coding 做成肌肉记忆，而且要早开始。**这是所有材料里被重复最多的技术建议。Liu 的具体建议最可操作：看斯坦福 CS336, Language Modeling from Scratch [50]，并把 **Assignment 1 当作单项投入产出比最高的东西**——*"实现 / 调试一个 transformer 在面试里出现得太频繁了，把它变成肌肉记忆会有巨大回报，实在不值得在这上面丢分"* [30]。Sapora 的六项 baseline [42] 是"练到什么算完成"的清单：端到端的 transformer、causal/cross/self attention、flash attention、attention 反向传播、MLP 前向和反向、以及一个训练循环。

**练习时把 AI 完全关掉。**Liu 说得很重：*"练习写代码的时候一定要把 AI 辅助完全关掉，来模拟面试环境（否则你会低估自己的依赖程度）！"* [30] 这与 Anthropic 对 live 轮的明文政策一致 [2]，而且在 2026 年，这大概是被违反得最多的一条准备建议。

**按场次准备，而不是泛泛准备。**Sapora：*"我几乎没做通用准备。几乎所有东西都是针对下一场具体面试或具体公司的。这让我保持专注，也意味着被问到的材料在我脑子里是新鲜的"* [42]。Liu 描述的节奏是一样的：*"每一场面试都是一门略微不同的数学课或计算机课，你一节课都没去过，而现在你有大约 3 天时间为期中考试突击"* [30]。而两人接着都指出，到最后你反正也覆盖了大部分材料。

**用 LLM 做模拟面试。**Sapora 在每一轮之前把岗位、公司和轮次描述贴给 Claude，让它面自己：*"那些练习题和面试官实际问的问题重合得出奇地频繁"* [42]。注意它和上一条的区别——用 LLM **模拟面试**是鼓励的；在**写代码练习里**用 LLM 则毁掉了练习本身的意义。

**维护一份诚实的题目日志。**Jaiswal 的四档评分——*"Aced it"、"Took time"、"Didn't get it"、"Just saw it somewhere"* [21]——逼你把"认得出"和"想得起"分开，而这恰恰是 25 分钟一轮所考的那个区分。Liu 在每场面试之后记笔记，出于同样的理由 [30]。

**做一个表格。**Sapora 最大的遗憾：*"我当时确信自己脑子里能记住所有事。技术上确实可以，但一个简单的表格（要申请的公司、每个流程走到哪、deadline、联系人）本可以让我不至于忘记去投那些我其实很感兴趣的地方"* [42]。

**至少预留一个月，并且预期它感觉像一份全职工作。**Sapora：*"至少分配一个月的固定学习时间"* [42]。Liu：*"对我和我聊过的大多数人来说，求职就是一份全职工作"* [30]。

最后用 Liu 的一段来结束这一节，因为它把整件事重新框定成了别的东西，而不只是一笔税：

> *"学习给我带来了巨大的额外收益。更宽的知识面直接提升了我作为研究者的自信……更神奇的是，我发现学习让我在手头正在做的项目上效率高了非常多。我能产生一些以前根本触及不到的技术想法、做更多技术性的工作，这让人兴奋。"* [30]

Meng 从另一个方向到达同一条原则：*"我花在面试准备上的每一分钟，都应该让我成为一个更好的工程师……背答案做不到这一点"* [34]。**如果你的准备计划过不了这个测试，那它就是错的计划。**

---

<a id="what-i-would-still-get-wrong"></a>
## 我可能仍然是错的地方

直说，因为这样一篇文章不该假装比它实际拥有的更确定。

**记录很薄、很新，而且自我指涉。**真正详细的第一手复盘大概只有五份，其中三份发表在 2026 年年中相隔一个月之内，而且互相引用。Liu 的时间线图是照着 Lambert 的做的；Yong 的文章明确是作为 Liu 和 Sapora 的补充写的。作为一门文献这是健康的引用图，但它仍然是一个非常小的样本，而且几乎全部来自强校博士项目里**成功了**的人。**所有求职不顺的人都不在样本里**，这是对上面一切最重要的限制。

**我第一轮检索没找到这些材料，这应该降低而不是提高你对我检索能力的信心。**我找到了第二梯队，并且在被指出主要材料之前已经基于它写完了一整稿。很可能还有我至今没找到的第三层。

**这些轮次是真的不标准化。**Yong 的核心论点切中了任何方向分类法的整洁性，包括我这个。七个方向是一个有用的组织工具，不是一份规格说明。给 wildcard 留预算。

**没人知道 AI 辅助的轮次是怎么打分的。**Meta 那一版有记录，rubric 没有，而且此刻同一家公司内不同面试官几乎肯定在用不同标准打分。与此同时 Anthropic 的 live 轮禁用 AI [2]。两种政策指向相反的准备方式，而行业还没有收敛。

**Fit 可能压过这一切。**Meng 唯一挂掉的那场 onsite 是她发挥最好的一场——岗位方向不匹配 [34]。Yong 发现自己转方向之后，那些拿过最佳论文奖的工作完全不起作用 [55]。如果真正的约束是 *"为什么是你？为什么不是别人？"*，那么很大一部分准备精力瞄准的并不是决定因素。Meng 的结论——*"找一个你打心底热爱的领域，然后申请那个领域的岗位"* [34]——要么是这里最深的建议，要么是幸存者偏差的合理化，我分不出来是哪一个。

**这篇会很快过时。**GRPO 完全晚于 Jaiswal 的那一轮 [21]。Work trial 和 AI 辅助轮次在两年前几乎不存在。十八个月后的对应物是什么，这些材料里没有一份能告诉你。

---

<a id="references"></a>
## 参考文献

1. Ainslie, J., Lee-Thorp, J., de Jong, M., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
2. Anthropic. *Guidance on Candidates' AI Usage.* [anthropic.com/candidate-ai-guidance](https://www.anthropic.com/candidate-ai-guidance)
3. Bai, Y., Kadavath, S., Kundu, S., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
4. Bekman, S. (2023–2026). *Machine Learning Engineering Open Book.* [github.com/stas00/ml-engineering](https://github.com/stas00/ml-engineering)
5. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
6. Deep-ML. *Practice Machine Learning.* [deep-ml.com](https://www.deep-ml.com/)
7. DeepSeek-AI. (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
8. DeepSeek-AI. (2024). *DeepSeek-V3 Technical Report.* [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
9. DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
10. Fedus, W., Zoph, B., & Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
11. Gordić, A. (2021). *How I Got a Job at DeepMind as a Research Engineer (without a Machine Learning Degree!).* [gordicaleksa.medium.com](https://gordicaleksa.medium.com/)
12. Grattafiori, A., et al. (2024). *The Llama 3 Herd of Models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
13. Grigorev, A. (2026). *AI Engineering Field Guide.* [github.com/alexeygrigorev/ai-engineering-field-guide](https://github.com/alexeygrigorev/ai-engineering-field-guide)
14. Gupta, G. (2025). *Notes on large-scale ML design and optimization for efficient training and inference.* Shared via X and LinkedIn; see [linkedin.com/in/gauri19](https://www.linkedin.com/in/gauri19)
15. He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
16. Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). *Training Compute-Optimal Large Language Models.* [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
17. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019). *The Curious Case of Neural Text Degeneration.* [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
18. Hu, E. J., Shen, Y., Wallis, P., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
19. Huang, Y., Cheng, Y., Bapna, A., et al. (2018). *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.* [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
20. Huyen, C. (2021). *Introduction to Machine Learning Interviews Book.* [huyenchip.com/ml-interviews-book](https://huyenchip.com/ml-interviews-book/)
21. Jaiswal, M. (2024). *LLM (ML) Job Interviews — Resources.* [mimansajaiswal.github.io](https://mimansajaiswal.github.io/posts/llm-ml-job-interviews-resources/)
22. Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). *Scaling Laws for Neural Language Models.* [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
23. Karpathy, A. *nanoGPT, micrograd, and Neural Networks: Zero to Hero.* [github.com/karpathy](https://github.com/karpathy)
24. Korthikanti, V., Casper, J., Lym, S., et al. (2022). *Reducing Activation Recomputation in Large Transformer Models.* [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
25. Kwon, W., Li, Z., Zhuang, S., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
26. Lambert, N. (2022). *Job Hunt as a PhD in AI / ML / RL: How it Actually Happens.* [natolambert.com](https://natolambert.com/writing/ai-phd-job-hunt)
27. Lambert, N. (2026). *Thoughts on the job market in the age of LLMs.* Interconnects. [interconnects.ai](https://www.interconnects.ai/p/thoughts-on-the-hiring-market-in)
28. Lambert, N., Morrison, J., Pyatkin, V., et al. (2024). *Tülu 3: Pushing Frontiers in Open Language Model Post-Training.* [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)
29. Leviathan, Y., Kalman, M., & Matias, Y. (2022). *Fast Inference from Transformers via Speculative Decoding.* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
30. Liu, A. (2026). *Notes on the Industry Job Search.* [alisawuffles.github.io](https://alisawuffles.github.io/blog/job-search/)
31. Liu, H., Zaharia, M., & Abbeel, P. (2023). *Ring Attention with Blockwise Transformers for Near-Infinite Context.* [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
32. Liu, J., Su, J., Yao, X., et al. (2025). *Muon is Scalable for LLM Training.* [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
33. Loshchilov, I., & Hutter, F. (2017). *Decoupled Weight Decay Regularization.* [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
34. Meng, Y. (2026). *MLE Interview 2.0: Research Engineering and Scary Rounds.* [yuan-meng.com](https://www.yuan-meng.com/posts/mle_interviews_2.0/)
35. Micikevicius, P., Narang, S., Alben, J., et al. (2017). *Mixed Precision Training.* [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
36. Milakov, M., & Gimelshein, N. (2018). *Online normalizer calculation for softmax.* [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
37. Ouyang, L., Wu, J., Jiang, X., et al. (2022). *Training language models to follow instructions with human feedback.* [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
38. Prince, S. J. D. (2023). *Understanding Deep Learning.* MIT Press. [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook/)
39. Rafailov, R., Sharma, A., Mitchell, E., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
40. Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2019). *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
41. Raschka, S. (2024). *Build a Large Language Model (From Scratch).* Manning. [github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
42. Sapora, S. (2026). *ML Job Interviews: The Ultimate Guide.* [silviasapora.github.io](https://silviasapora.github.io/blog/ml-interviews.html)
43. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
44. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
45. Sennrich, R., Haddow, B., & Birch, A. (2015). *Neural Machine Translation of Rare Words with Subword Units.* [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
46. Shao, Z., Wang, P., Zhu, Q., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
47. Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All You Need.* [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
48. Shoeybi, M., Patwary, M., Puri, R., et al. (2019). *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
49. Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
50. Stanford University. *CS336: Language Modeling from Scratch.* [cs336.stanford.edu](https://cs336.stanford.edu/)
51. Su, J., Lu, Y., Pan, S., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
52. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
53. Xiong, R., Yang, Y., He, D., et al. (2020). *On Layer Normalization in the Transformer Architecture.* [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
54. Yang, G., Hu, E. J., Babuschkin, I., et al. (2022). *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
55. Yong, Z.-X. (2026). *Surprising lessons from my research scientist job search.* [yongzx.github.io](https://yongzx.github.io/blog/2026/06/24/job-search/)
56. Yu, Q., Zhang, Z., Zhu, R., et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
57. Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
58. Zhao, Y., Gu, A., Varma, R., et al. (2023). *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.* [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)

---

## 引用方式

```bibtex
@article{zhang2026threegates,
  title   = {Three Gates and Seven Buckets: What Frontier Labs Actually Test},
  author  = {Zhang, Jiaxin},
  journal = {jxzhangjhu.github.io},
  year    = {2026},
  url     = {https://jxzhangjhu.github.io/blog/2026/frontier-lab-interview-zh/}
}
```
