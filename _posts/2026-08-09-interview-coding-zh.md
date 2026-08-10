---
layout: post
title: "面试题库 II · Coding + Math：要写，不要只读（中文版）"
date: 2026-08-09 12:00:00
author: Jiaxin Zhang
description: "Frontier lab coding 轮会考的组件的完整实现，全部带测试——attention、KV cache、RoPE、采样、GRPO、BPE，加上配套的概率与线性代数，以及一套限时练习工具。"
tags: interviews llm coding math pytorch qbank
categories: research-notes
giscus_comments: true
related_posts: false
ai_assisted: true
---

<div class="lang-switch"><a href="/blog/2026/interview-coding/">English</a> · <strong>中文</strong></div>

<div class="lang-switch"><a href="/blog/2026/interview-knowledge-zh/">I · 知识</a> · <strong>II · 代码 + 数学</strong> · <span class="text-muted">III · 讨论 + BQ</span></div>

第一篇问的是你**想不想得起来**，这一篇问的是你能不能在**空白文件里、有钟在走的情况下写出来**。

这是两种能力，而它们之间的差距就是这个页面要配一个仓库的全部理由。把一段 attention
实现读到觉得「显然」，对你二十分钟内写出一版几乎没有帮助。所以下面的代码是讲解层，
真正练手在 [`interview-practice/`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/tree/master/interview-practice)：27 道带 stub 和测试的题、10 个 debug drill，
外加一个计时的 runner。

> **每节的结构。**先简述概念，给出完整带注释的实现，列出每次都会有人踩的坑，
> 说明面试官在看什么，然后指向对应的练习题。

---

### 目录

- **[B0 · 这部分该怎么练](#section-b0)**
  - [B0.1 两层结构](#b0-1)
  - [B0.2 让练习真正有效的三件事](#b0-2)
  - [B0.3 题目清单](#b0-3)
- **[B9 · 调试](#section-b9)**
  - [B9.1 在钟走着的时候管用的方法](#b9-1)
  - [B9.2 miniGPT 练习](#b9-2)
  - [B9.3 GRPO 循环练习](#b9-3)
  - [B9.4 微练习](#b9-4)
- **[B1 · NumPy 与 PyTorch 基础](#section-b1)** — 3 题
  - [B1.1 向量化：唯一值得背下来的技巧](#b1-1)
  - [B1.2 BatchNorm，以及它为什么有两种模式](#b1-2)
  - [B1.3 会造成静默 bug 的张量语义](#b1-3)
- **[B2 · Transformer 组件](#section-b2)** — 7 题
  - [B2.1 因果多头注意力](#b2-1)
  - [B2.2 KV cache 与增量解码](#b2-2)
  - [B2.3 分组查询注意力](#b2-3)
  - [B2.4 旋转位置编码](#b2-4)
  - [B2.5 RMSNorm](#b2-5)
  - [B2.6 SwiGLU](#b2-6)
  - [B2.7 拼成一个 block](#b2-7)
- **[B3 · 训练循环](#section-b3)** — 4 题
  - [B3.1 Cross entropy，以及它为什么收 logits](#b3-1)
  - [B3.2 Loss masking 与 packing](#b3-2)
  - [B3.3 干别的之前，先过拟合十条样本](#b3-3)
  - [B3.4 过滤坏标注](#b3-4)
- **[B4 · 手推 backward](#section-b4)** — 2 题
  - [B4.1 能生成所有 backward 的那条规则](#b4-1)
  - [B4.2 四十行的 autograd](#b4-2)
  - [B4.3 Attention 的 backward](#b4-3)
- **[B5 · 推理与采样](#section-b5)** — 2 题
  - [B5.1 Temperature、top-k、top-p](#b5-1)
  - [B5.2 Speculative decoding](#b5-2)
- **[B6 · 高效实现](#section-b6)** — 2 题
  - [B6.1 流式 softmax](#b6-1)
  - [B6.2 分块的 FlashAttention forward](#b6-2)
- **[B7 · 后训练算法](#section-b7)** — 4 题
  - [B7.1 LoRA](#b7-1)
  - [B7.2 GRPO 目标函数](#b7-2)
  - [B7.3 DPO](#b7-3)
  - [B7.4 GAE](#b7-4)
- **[B8 · 数据与 tokenization](#section-b8)** — 2 题
  - [B8.1 Byte-pair encoding](#b8-1)
  - [B8.2 带 capacity 的 top-1 MoE routing](#b8-2)
- **[C1 · 概率：覆盖大部分题目的四种套路](#section-c1)**
  - [C1.1 首步分析](#c1-1)
  - [C1.2 指示变量加期望的线性性](#c1-2)
  - [C1.3 $$n$$ 个变量的最大值与最小值——走 CDF](#c1-3)
  - [C1.4 把对称性当证明工具](#c1-4)
  - [C1.5 该抓哪个不等式](#c1-5)
- **[C2 · 先模拟，再验证](#section-c2)** — 1 题
  - [C2.1 旋转的光源](#c2-1)
  - [C2.2 通用套路](#c2-2)
- **[C3 · 线性代数](#section-c3)**
  - [C3.1 其余一切都从中推出的四条事实](#c3-1)
  - [C3.2 半正定，以及它为什么反复出现](#c3-2)
  - [C3.3 范数、条件性，以及那些会炸的东西](#c3-3)
  - [C3.4 你真正用得上的矩阵微积分](#c3-4)

---
<a id="section-b0"></a>

## B0 · 这部分该怎么练

大多数 coding 复习材料都以同一种方式失效：它给你看一份正确实现，你点点头，
然后以为自己会了。你不会。你只是确认了自己**读得懂**。在有钟、没有补全、有人盯着的编辑器里，
垮掉的不是你的理解——是你在不停顿的情况下产出二十行正确代码的能力。

所以这个页面**故意只是材料的一半**，另一半是一个仓库。

---

<a id="b0-1"></a>
### B0.1 两层结构

**这个页面是讲解层。**每一节给出完整的带注释实现，加上反复有人踩的坑，
以及面试官真正在看什么。读一遍建立模型，练不下去的时候回来查。

**`practice/` 是训练层。**同样这些题以 stub 形式存在——只有签名和 docstring，函数体挖空——
配一套测试，把你的实现和 PyTorch 及参考解法做数值比对。

```bash
git clone https://github.com/jxzhangjhu/jxzhangjhu.github.io
cd jxzhangjhu.github.io/interview-practice

python run.py                 # 题目列表、时间预算、你的历史成绩
python run.py p01             # 开始计时，跑这道题的测试
python run.py --cold          # 冷启动那一组，按顺序过
python run.py --reset p01     # 清空重做
```

下面每一节末尾的**练习**行直接链到那道题的 stub 和分级提示，不 clone 也能在 GitHub 上看。
`pNN` 这个编号就是 `run.py` 的参数。

测试是按「诊断」写的，不是单纯报错。因果 mask 漏了未来时，它告诉你
*"perturbing the last token changed earlier outputs: the mask leaks the future"*，
而不是甩给你一堆张量。

> **关于参考解法。**它们在 `interview-practice/reference.py`，每一个都对着 PyTorch 做了验证——
> 28 项自检，全绿，否则说明这个页面是错的。**练的时候不要打开那个文件**，那是三级提示存在的意义。

---

<a id="b0-2"></a>
### B0.2 让练习真正有效的三件事

**加钟。**每道题都有时间预算——多头注意力 20 分钟，RMSNorm 5 分钟，小型 autograd 30 分钟。
不限时地练会建立一种进屋就蒸发的信心，因为压力下的失败模式**不是概念性的**：
是忘了 `.contiguous()`、是 mask 取反了、是除错了那个平方根。

**保留一个小的冷启动集。**26 道题里有 11 道标了 `cold`：这些要能从空文件写出来，值得每周重做。
其余的标准低一些——想得起解法的形状，看一眼提示能重建出来就够了。
想把 26 道全部练到肌肉记忆，是面试前一周把自己耗垮的标准路径。

**把 bug 单独拿出来练，而且要先练。**公开面经显示 debug **才是最高频的** ML coding 形式，
不是从零手写——所以 `drills/` 不是配菜。其中两道（`d09`、`d10`）是 OpenAI 和 Anthropic
最高频那两题的完整复刻，另外八道是只错一行的微型练习，每道约三分钟。
B9 那一节讲的就是这些，它排在这个页面最前面是有原因的。

---

<a id="b0-3"></a>
### B0.3 题目清单

`cold` 是要求能从空文件写出来的那一组。时间预算是面试的预算，不是「读懂答案要多久」。

**「面经来源」那一列是最有用的。**它记录公开面经里这道题被哪家考过、大概多少次。
按那个顺序练，而不是按我排的顺序。空白只表示这题是标准题但没有具体归属，不代表它不重要。

<!-- TABLE -->
| | 题目 | 预算 | 冷启动 | 面经来源 |
|---|---|---|---|---|
| **B1 · NumPy 与 PyTorch** | | | | |
| p24 | 纯 NumPy 的 1-NN，不许循环 | 15 分钟 | ● | OpenAI 3+ |
| p25 | BatchNorm 前向、反向、eval 模式 | 20 分钟 |  | Datadog |
| **B2 · 组件** | | | | |
| p01 | 因果多头注意力 | 20 分钟 | ● | universal |
| p02 | KV cache 与增量解码 | 15 分钟 | ● | OpenAI 7+ (as follow-up) |
| p03 | 分组查询注意力 | 10 分钟 |  | Datadog |
| p04 | 旋转位置编码 | 15 分钟 | ● |  |
| p05 | RMSNorm | 5 分钟 | ● |  |
| p06 | SwiGLU 前馈层 | 5 分钟 |  |  |
| p07 | 完整的 pre-norm block | 15 分钟 |  |  |
| **B3 · 训练** | | | | |
| p08 | 交叉熵与 log-sum-exp | 10 分钟 | ● |  |
| p09 | SFT loss masking 与 packing | 10 分钟 |  |  |
| p10 | 把一个小 batch 过拟合 | 20 分钟 | ● |  |
| p26 | 过滤劣质人工标注 | 20 分钟 |  | OpenAI 2+ |
| **B4 · 反向** | | | | |
| p11 | 40 行 autograd | 30 分钟 |  | OpenAI 2+ |
| p12 | 手写 attention 反向 | 25 分钟 |  | OpenAI |
| p13 | 手写 MLP 反向 | 15 分钟 |  |  |
| **B5 · 推理** | | | | |
| p14 | temperature / top-k / top-p | 15 分钟 | ● |  |
| p15 | 投机解码的接受/拒绝 | 20 分钟 |  |  |
| **B6 · 效率** | | | | |
| p16 | 流式 softmax | 15 分钟 |  |  |
| p17 | 分块 FlashAttention 前向 | 25 分钟 |  |  |
| **B7 · 后训练** | | | | |
| p18 | LoRA 与无损合并 | 10 分钟 | ● |  |
| p19 | GRPO 目标 | 20 分钟 | ● | OpenAI + Anthropic 4+ |
| p20 | DPO 损失 | 15 分钟 |  |  |
| p21 | GAE | 15 分钟 |  |  |
| **B8 · 数据** | | | | |
| p22 | Byte-pair encoding | 20 分钟 | ● |  |
| p23 | 带容量的 top-1 MoE 路由 | 20 分钟 |  |  |
| **C2 · 模拟** | | | | |
| p27 | 旋转光源 → Cauchy 分布 | 20 分钟 |  | OpenAI |

| | Drill | 预算 | 错在哪一处 | 面经来源 |
|---|---|---|---|---|
| d09 | minigpt | 35 分钟 | nanoGPT 里的四个 bug，然后加 KV cache | OpenAI 7+ — the single most reported ML-coding question |
| d10 | grpo_loop | 30 分钟 | GRPO 训练脚本里的三个 bug | Anthropic 3+, also OpenAI |
| d01 | mask_inverted | 3 分钟 | masked_fill 填的是 mask 为 True 的位置，而 mask 反了 |  |
| d02 | missing_contiguous | 3 分钟 | 在非连续张量上 transpose 之后直接 .view() |  |
| d03 | top_p_off_by_one | 4 分钟 | top-p 把越过阈值的那个 token 丢了 |  |
| d04 | cache_mask_offset | 5 分钟 | 缓存解码用了 tril 但没写 diagonal=T_full-T |  |
| d05 | lora_both_random | 3 分钟 | LoRA 的 A 和 B 都随机初始化 |  |
| d06 | softmax_overflow | 3 分钟 | softmax 没有减去行最大值 |  |
| d07 | wrong_scale | 3 分钟 | 除以 sqrt(d_model) 而不是 sqrt(d_head) |  |
| d08 | prompt_not_masked | 4 分钟 | SFT 的 loss 把 prompt token 也算进去了 |  |
<!-- TABLE -->

**一个能塞进上班日程的四周轮转。**第一周先做两道旗舰 drill（`d09`、`d10`），
再把冷启动那组各做一遍，允许看提示——目标是覆盖，不是速度。第二周做剩下 15 道，规则相同。
第三周冷启动那组再来一遍，不看提示、严格计时，没过的记进一张短名单。
第四周做短名单加八道微型 drill，面试前一天把冷启动组完整过一遍。

> **这类计划最诚实的失败模式**是：你会去反复刷那些你**已经会**的题，因为做起来舒服。
> 你的历史记录在 `practice/attempts.local.json` 里，按「最近一次失败」排序，从那里开始。

---

<a id="section-b9"></a>

## B9 · 调试

时间不够就先看这一节。

公开面经把排序摆得很清楚：**最常见的 ML coding 形式是调试**，不是从零写。
在 OpenAI，「debug 这个 transformer」出现在七份以上互相独立的面经里，比任何其他题都多。
在 Anthropic，排前两位的也都是调试——一个 GRPO 训练循环，和一次 NumPy 找 bug。
这一页里其余那些从零实现的题目都是真题，但重心不在那儿。

这种题的形式固定到可以专门去准备：给你一段**能跑起来、不抛异常**、但结果是错的代码。
bug 是逻辑上的，不是语法上的。通常会告诉你个数（「大概四个」），
而且通常带一个追问——追问是让你扩展这段代码，不是继续修它。

---

<a id="b9-1"></a>
### B9.1 在钟走着的时候管用的方法

通过的人写的面经收敛到同一套做法，而那套做法不是「再仔细读一遍代码」。

**先做到确定性复现。**所有随机源都固定种子，模型切到 `eval()`，用贪心解码。
输出自己会动的话，你根本判断不了某个改动有没有用。

**用断言定位，不要靠读。**每一步都把形状打出来。把你知道必须成立的不变量断言出来：
注意力每行和为 1、带 cache 的解码等于完整重算、关掉位置信息后置换输入就得到置换输出。
每一条断言把搜索空间砍一半，重读一遍不会。

**一次只修一个 bug，然后重跑。**bug 之间会互相掩盖。下面那道练习里，
把 head 和 time 搞乱的那次 reshape 掩盖了「模型看不见位置」这件事，因为搞乱本身也破坏了置换等变性。
一口气改三处再跑，你就不知道到底是哪一处起了作用。

**文件里有注释就信它。**好几份面经提到，出 bug 的区域是被标出来的，通过的人也是同一句话：
时间紧的时候，别去重审没被标记的代码。没有标记就问一句——这是个合理的问题。

**说出你找到的是哪一类 bug。**「mask 加在了 softmax 之后，所以每行不再和为 1」
比「修好了」是强得多的信号。面试官打分打的是你对这个系统的建模，不是你的 diff。

> **为这种题型做的最好的准备**，是至少完整地写过一次 nanoGPT 那种规模的模型，
> 从 embedding 表一路写到训练循环。多份面经说，光是熟悉 nanoGPT 就足够了。

---

<a id="b9-2"></a>
### B9.2 miniGPT 练习

这道题直接复刻了 OpenAI 的原题：一个小的 decoder-only LM，能跑、能训、生成的是垃圾，
埋了四个 bug，外加一个 KV cache 追问。

```
python -m pytest tests/test_d09_minigpt.py -q      # 35 分钟预算
```

四类 bug，和面经点名的完全一致：

| bug | 你能测出来的症状 |
|---|---|
| 位置 embedding 用一个常数下标去取 | 置换输入只会得到置换后的输出 |
| 因果 mask 加在了 softmax *之后* | 注意力每行不再和为 1 |
| 合并多头时没把时间维 transpose 回来 | 输出静默地错，不抛异常 |
| 训练循环从来没让优化器 step | 走一步之后参数没有变化 |

**第三个值得多看两眼**，因为只有它能教会你点东西。注意力的输出是 `(B, n_heads, T, d_head)`，
你要的是 `(B, T, C)`。直接 reshape 是*能跑*的——元素个数对得上——然后把 head 和 time 交错在了一起。
什么都不报。模型训出一个更差的 loss，而你手上没有任何报错可查。
这就是「为什么 `.transpose(1, 2).contiguous().view(...)` 要写成这个样子」、
以及「为什么带形状后缀的变量名（`y_BHTD`）值回票价」的标准例子。

**追问是 KV cache**，其中有一个细节几乎所有人都会漏：新 token 的**位置下标是 cache 的长度**，
不是零。第 $$t$$ 步 decode 必须 embed 位置 $$t$$。写错了，生成就会以一种 teacher-forced 评测
永远给你看不到的方式退化。把不变量说出口——带 cache 的解码必须和完整重算数值完全一致——然后去测它。

---

<a id="b9-3"></a>
### B9.3 GRPO 循环练习

Anthropic 的版本：一个完整的 GRPO 训练脚本，大概 150 行，端到端跑得通。
两个 bug 是数值上的，一个是算法上的。

```
python -m pytest tests/test_d10_grpo_loop.py -q    # 30 分钟预算
```

**原始 logits 直接喂给了 `torch.multinomial`。**这个函数收的是未归一化的*权重*，不是 logits，
于是你静默地从一个错误的分布里采样——负的 logits 让情况更糟。改法：先做 softmax。

**advantage 除了一个裸的标准差。**当一组里每条 completion 拿到的 reward 都一样时，
标准差是零、advantage 是 NaN，下一步就传染到每一个参数上。这不是边角情况：
一组全对或者全错，在训练早期和后期都是*常见*情况，
而且正是同一个零方差状况让这些组对学习毫无价值（第一篇 A9.5）。改法：`+ 1e-5`。

**ratio 被算成了对数差。**importance ratio 是
$$\exp(\log \pi_\theta - \log \pi_{\text{old}})$$。拿这个差本身当 ratio 不是 ratio，
而且判据很锐利：on-policy 时新旧 log-prob 相等，ratio 必须正好是 1，
未截断的 surrogate 必须等于 advantage。对数差在那里给出零，
于是目标函数恰好在训练起点上没有梯度。

**然后才是真正的那道题**，而且它是道讨论题，不是代码题：

> 这个循环严格是 on-policy 的。那为什么平均 importance ratio 不是正好 1？

好的回答会点出好几个原因，并且对每一个都说清楚你会去查什么：

- **每个 rollout batch 做了不止一次优化器 step。**第一步之后策略已经动了，
  剩下的 mini-epoch 在构造上就是 off-policy 的。去查内层 epoch 数。
- **采样引擎和训练引擎不是同一个。**vLLM 出的 rollout 和在 HF 里重算的 log-prob
  不会逐位相同——kernel 不同、attention 实现不同、精度不同。
  查法是对同一批 token 在两边各算一遍 log-prob，然后做差。
- **采样参数只在生成时生效、打分时没生效。**temperature、top-p 和 logit bias
  改变了你实际采样的那个分布。如果你用原始分布去打分，你的「old」log-prob 属于另一个策略。
- **精度与非确定性。**log-prob 在 fp32 还是 bf16 里累加、attention 用 fused 还是 eager，
  即便权重完全相同也会让 ratio 轻微偏移。

能把**设计上就该有的漂移**和**真正的 bug** 分开，才是这道题在考的东西。

---

<a id="b9-4"></a>
### B9.4 微练习

八段实现，每段正好有一行是错的，一段大概三分钟。便宜到可以在等编译的间隙做完，
而且和上面那些 bug 类别一一对应。

| 练习 | 错的那一行 |
|---|---|
| `d01` | `masked_fill` 填的是 mask 为 **True** 的位置，而这里 mask 反了 |
| `d02` | transpose 之后在非连续张量上调 `.view()` |
| `d03` | top-p 把跨过阈值的那个 token 丢掉了 |
| `d04` | 带 cache 的解码用了 `tril`，却没写 `diagonal=T_full - T` |
| `d05` | LoRA 把 `A` 和 `B` 都做了随机初始化 |
| `d06` | softmax 没有减去每行的最大值 |
| `d07` | 用 $$\sqrt{d_\text{model}}$$ 而不是 $$\sqrt{d_\text{head}}$$ 做缩放 |
| `d08` | SFT loss 把 prompt token 也算了进去，不只是 response |

> **练这项能力，为什么微练习胜过重写一遍。**从零写 attention 练的是产出，
> 找出一个反了的 mask 练的是识别。调试轮考的是识别，
> 而在一次从零实现所花的时间里，你能把识别练二十遍。

---

<a id="section-b1"></a>

## B1 · NumPy 与 PyTorch 基础

这一节里有两道题在面经里被点名——OpenAI 的纯 NumPy 1-NN，和 Datadog 的 BatchNorm——
但它排在最前面的真正理由是：后面的一切都建立在它上面。从零实现类题目里的失败大多不是概念问题，
就是一次 transpose、一次 broadcast，或者一个 dtype。

---

<a id="b1-1"></a>
### B1.1 向量化：唯一值得背下来的技巧

这道题在 OpenAI 出现过三次以上：用 NumPy 写 1-最近邻，不许用循环。考的不是分类器，
是你知不知道怎么把一次成对距离计算变成一次矩阵乘。

$$\|a - b\|^2 = \|a\|^2 - 2\,a \cdot b + \|b\|^2$$

把平方展开，交叉项就变成一次 matmul，而 BLAS 跑它比你用循环写的任何东西都快得多。

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

**边写边要说出来的三件事。**全程不开根号，因为 `argmin` 对单调变换不变，
而 `sqrt` 要在整个矩阵上白走一遍。broadcast 是 `(n_test, 1)` 对 `(1, n_train)`，
用 `[:, None]` 显式写出来、而不是依赖隐式对齐，是这段代码能读的原因。
还有，两个点几乎重合时，展开式会因浮点相消而略微为负——对 `argmin` 无害，
但开根号之前要先 `np.maximum(d2, 0)`。

> **面经里的追问是内存。**这会实打实地开出一个 `n_test × n_train` 的矩阵。
> 十万乘十万就是 80 GB，所以你要按测试行分块、维护一个当前最优。
> 这句话要在被问之前说。

**练习** —— [`p24`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p24_nn_vectorized.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p24_nn_vectorized.md) · 15 分钟 · 冷启动集 · *OpenAI 3+*

---

<a id="b1-2"></a>
### B1.2 BatchNorm，以及它为什么有两种模式

Datadog 考过。它看着像热身题，其实不是——有意思的部分是状态，不是公式。

```python
class BatchNorm1dScratch(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.eps, self.momentum = eps, momentum
        # buffer 而不是 parameter：靠滑动平均更新，不靠梯度
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))

    def forward(self, x):
        if self.training:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)          # 归一化用有偏方差……
            with torch.no_grad():
                n = x.shape[0]
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                # ……但滑动估计用的是无偏方差
                self.running_var.mul_(1 - self.momentum).add_(
                    self.momentum * var * n / (n - 1))
        else:
            mean, var = self.running_mean, self.running_var
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
```

**拉开答案差距的四个细节。**

**训练和评估算的是两个不同的函数。**训练时统计量来自当前 batch，评估时来自滑动估计。
这是唯一一个忘了 `.eval()` 就会改变输出的常用层，也正因如此，
BatchNorm 会在 batch size 为 1 的推理和分布式训练里出问题，而 LayerNorm 不会。

**用 `register_buffer`，不是 `nn.Parameter`。**滑动统计量跟着 `.to(device)` 走、会存进 state dict，
但不接收梯度。把它们写成 parameter 是一个常见的、而且不声不响就错了的答案。

**归一化用有偏，滑动估计用无偏。**PyTorch 用有偏方差（$$/n$$）做归一化，
累积的却是无偏的那个（$$/(n-1)$$）。不对齐这一点，你的实现就只在 eval 模式下和
`nn.BatchNorm1d` 对不上——这是「测试必须覆盖两种模式」的绝佳例子。

**epsilon 在根号里面**，不是外面。放外面就完全起不到防零方差的作用。

> **值得提前备好的追问：**transformer 为什么改用 LayerNorm？序列长度可变、
> 一个 batch 内样本之间被耦合（这会让 batch 为 1 的自回归生成失效）、
> 以及分布式训练下每次前向都要做一次跨设备同步。完整版在第一篇 A1.7。

**练习** —— [`p25`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p25_batchnorm.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p25_batchnorm.md) · 20 分钟 · *Datadog*

---

<a id="b1-3"></a>
### B1.3 会造成静默 bug 的张量语义

大部分伤害来自四件事。

**`view` 与 `reshape`。**`view` 要求内存连续，否则直接拒绝；`reshape` 会退化成复制。
`transpose` 之后你是非连续的，所以 `view` 会报错——这是*好*情况，因为它告诉了你。
坏情况是元素个数恰好对得上，一次 reshape 就悄悄把轴交错在一起，
这正是 miniGPT 练习里的第三个 bug（B9.2）。

**broadcast 从右往左对齐。**`(B, T, C) * (C,)` 可以，`(B, T, C) * (B,)` 不行。
想做按 batch 的缩放，就必须写成 `(B, 1, 1)`。用 `None` 索引显式写出来、
而不是依赖对齐规则，是防住这类错误的习惯。

**原地操作与 autograd。**对反向要用到的张量做 `x += 1` 会触发 version counter 报错，
`x = x + 1` 不会。原地操作在优化器状态和滑动统计量上是划算的，其他地方基本不值。

**dtype 提升是静默的。**bf16 乘 fp32 得到 fp32。归一化层就是这样悄悄返回了错误的 dtype（B2.5），
一次「bf16」训练也是这样在你没打算的地方留下了 fp32 激活。

**练习** —— B9.4 里的微练习 `d02`、`d06`、`d07` 打的正是这几个点。

---

<a id="section-b2"></a>

## B2 · Transformer 组件

最基础的一轮。每家实验室都会考这里面的一部分，而门槛不是「你知不知道 attention 是什么」——
是那二十行能不能在钟走着的时候干净地写出来，以及你能不能在面试官指出之前自己发现错误。

这里所有代码都在 `interview-practice/reference.py` 里，全部对着 PyTorch 验证过。

---

<a id="b2-1"></a>
### B2.1 因果多头注意力

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

缩放让 logits 在初始化时保持单位方差，softmax 不至于饱和；mask 是在 softmax **之前**
以加性 $$-\infty$$ 的形式加入的，这样被屏蔽的位置对分母毫无贡献。两条论证都在第一篇（A2.3）。

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
        # buffer 而不是 parameter：跟着 .to(device) 走、会存进 state dict，但不接收梯度
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x):
        B, T, C = x.shape
        # q、k、v 融成一次矩阵乘，比分三次便宜
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

**四个高频错误，按出现频率排序。**

1. **`.contiguous()`。**`transpose(1, 2)` 之后张量是 stride 不连续的 view，`.view()` 会报错。
   想用 `.reshape()` 也行，但要能说出区别：`view` 从不复制、所以直接拒绝；`reshape` 必要时退化成复制。
2. **除以 $$\sqrt{d_\text{model}}$$ 而不是 $$\sqrt{d_\text{head}}$$。**点积是在头的维度上做的，
   那才是你要修正方差的那一维。写错了照样能训，只是更差——这正是它成为考点的原因。
3. **在 softmax 之后做 mask。**事后把被屏蔽位置置零，它们仍然留在分母里，
   剩下的权重不再和为 1，而且每一行的误差还不一样。
4. **三个独立的投影。**数学等价，但可测地更慢：同样 FLOPs 下一次大 GEMM 胜过三次小的。

**在被问之前主动给出因果性检查。**三行，而且它是这道题里最强的信号：

```python
y1 = model(x)
x2 = x.clone(); x2[:, -1, :] += 10.0
assert torch.allclose(y1[:, :-1], model(x2)[:, :-1])   # 过去看不到未来
```

> **面试官在看什么。**形状纪律、你会不会在没人提醒时自己想到连续性问题、
> 以及你验不验自己的东西。主动写出这个检查的候选人，和需要被问「你怎么验证」的候选人，
> 是两个类别。

**练习** —— [`p01`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p01_mha.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p01_mha.md) · 20 分钟 · 冷启动集

---

<a id="b2-2"></a>
### B2.2 KV cache 与增量解码

在 decode 第 $$t$$ 步你只有一个 query，但需要全部历史的 key 和 value。
Q 是瞬时的，K/V 是累积的。没有 cache 就每步重算整段历史，即 $$O(T^2)$$ 的无谓开销。

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

    k = k.repeat_interleave(self.n_rep, dim=1)        # 把 kv 头扩展到和 q 头一样多
    v = v.repeat_interleave(self.n_rep, dim=1)

    T_full = k.shape[2]
    att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
    # 第 i 个 query 的绝对位置是 T_full - T + i，这就是对角线偏移量
    causal = torch.ones(T, T_full, dtype=torch.bool, device=x.device).tril(
        diagonal=T_full - T
    )
    att = att.masked_fill(~causal, float("-inf"))
    y = F.softmax(att, dim=-1) @ v
    return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))
```

**mask 的偏移量就是这道题的全部。**Prefill 时 $$T = T_\text{full}$$，普通 `tril` 是对的。
但缓存解码时你的 query 块是从矩阵中间某一行开始的，所以需要 `diagonal=T_full - T`。
写错的后果是：teacher-forced 评测里一切正常，生成时悄悄退化——
这是线上很难查的一类 bug，因为你的评测永远看不到它。

**要主动说出的正确性性质：**带 cache 的增量解码必须和完整重算**数值完全一致**。
这是可测的，那就去测。

> **面试官在看什么。**你有没有意识到 mask 变了。大多数候选人 cache 拼接写得对，
> 然后原样复用了 prefill 的 mask。

**练习** —— [`p02`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p02_kv_cache.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p02_kv_cache.md) · 15 分钟 · 冷启动集

---

<a id="b2-3"></a>
### B2.3 分组查询注意力

GQA 和 MHA 之间只差一行：query 头分组，组内共享一个 K/V 头。

```python
k = k.repeat_interleave(self.n_rep, dim=1)   # n_rep = n_heads // n_kv_heads
v = v.repeat_interleave(self.n_rep, dim=1)
```

$$n_\text{kv} = 1$$ 是 MQA，$$n_\text{kv} = n_\text{heads}$$ 是普通 MHA，
中间是一个作用在 KV cache 上的可调旋钮——GQA 存在的唯一理由就是这个。

**会绊倒人的追问：它并不减少注意力的 FLOPs。**K 和 V 在矩阵乘之前被扩展回完整头数，
所以 $$QK^\top$$ 和 $$AV$$ 一点没变。变小的是 cache 以及读它所需的带宽，
而 decode 是带宽受限的，加速就来自那里。（被追问时要说准：K/V 的**投影**确实变小了，
每层从 $$2D^2$$ 降到 $$2DKH$$。）

**练习** —— [`p03`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p03_gqa.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p03_gqa.md) · 10 分钟

---

<a id="b2-4"></a>
### B2.4 旋转位置编码

RoPE 把每个坐标对按与位置成正比的角度旋转，使注意力 logit 只依赖相对偏移。
三行证明在第一篇（A2.6），这里重要的是实现。

```python
def rope_cache(seq_len, d_head, base=10000.0):
    theta = base ** (-torch.arange(0, d_head, 2).float() / d_head)   # (d_head/2,)
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, theta)                                  # (T, d_head/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (..., T, d_head)。把 (0,1)、(2,3)…… 每一对独立旋转
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)
```

**三个细节。**它只加在 **Q 和 K** 上，在拆头之后、点积之前——绝不加在 V 上，
V 携带的是内容不是位置。配 KV cache 时缓存的是**旋转之后**的 key。
还有配对约定（`0::2, 1::2` 还是前后半分）必须在建表和应用两处一致，
否则你得到的是一个训练 loss 更差、但不报任何错的模型。

**练习** —— [`p04`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p04_rope.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p04_rope.md) · 15 分钟 · 冷启动集

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
        # 归约必须在 fp32 里做：在 bf16 下把 d 个平方值加起来，
        # 累积的舍入误差足以让 norm 偏移。最后再转回去，让这一层对 dtype 透明。
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * rms).type_as(x) * self.weight.type_as(x)
```

五行，而有意思的那一行是 `float()`。没有减均值也没有 bias——
消融显示起作用的是重新缩放、重新中心化基本没用，去掉它省一次归约，
在八十层、每层两次的场景下这是有意义的。

> **这是真陷阱，不是风格问题。**一个在 bf16 里做归约的实现，能通过你写的每一个 fp32 测试，
> 然后在 bf16 训练里悄悄劣化。练习题特意喂它量级 $$10^4$$ 的 bf16 输入，就是为了逼出这个。

**练习** —— [`p05`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p05_rmsnorm.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p05_rmsnorm.md) · 5 分钟 · 冷启动集

---

<a id="b2-6"></a>
### B2.6 SwiGLU

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        # 8/3 让参数量和 4 倍宽的 ReLU FFN 持平：3*d*F == 2*d*4d
        d_ff = d_ff or int(8 * d_model / 3)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

**三个矩阵，不是两个**——这就是这道题的全部，外加知道为什么 $$F = \tfrac83 D$$：
它让参数量和普通的 $$4D$$ FFN 持平，这样架构对比才是在同等参数下做的，才有意义。

**练习** —— [`p06`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p06_swiglu.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p06_swiglu.md) · 5 分钟

---

<a id="b2-7"></a>
### B2.7 拼成一个 block

```python
class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))     # pre-norm：残差流保持一条恒等通路
        x = x + self.mlp(self.norm2(x))
        return x
```

forward 只有两行。Pre-norm 归一化的是子层的**输入**，从 embedding 到输出留下一条干净的恒等通路，
这正是它去掉 warmup 架构性需求的原因。代价是残差流的幅度随深度增长，
所以**完整模型在 `lm_head` 之前需要一个 final norm**——从零写模型时最常被忘掉的一行。

**练习** —— [`p07`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p07_transformer_block.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p07_transformer_block.md) · 15 分钟

---

<a id="section-b3"></a>

## B3 · 训练循环

没有人会拿「写一个训练循环」当主问题问你。它出现的方式是：调试轮里要你修的那段东西，
或者你刚实现完的组件外面那层壳——而四个经典 miniGPT bug 里，有一个就住在这儿。

---

<a id="b3-1"></a>
### B3.1 Cross entropy，以及它为什么收 logits

```python
def cross_entropy(logits, targets, ignore_index=-100):
    keep = targets != ignore_index
    logits, targets = logits[keep], targets[keep]
    # log_softmax(x)[t] = x[t] - logsumexp(x)；绝不要先造出概率再取对数
    logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -logprobs.gather(1, targets[:, None]).mean()
```

**这个 API 为什么收 logits 而不是概率。**`log(softmax(x))` 先取指数、再归一化、再取对数——
三次丢精度的机会，而且大 logit 的 `exp` 在你走到对数那一步之前就已经溢出成 `inf` 了。
融合写法直接减掉 `logsumexp`，而 `logsumexp` 自己会先减掉每行的最大值。
logits 量级在 $$10^4$$ 附近时，朴素版本返回 `nan`，融合版本仍然精确。

**`ignore_index` 里最常被漏掉的一点：**分母必须是*留下来*的 token 数，不是 $$N$$。
先 mask、再对全部求平均，等于悄悄把 loss 乘上了保留比例，而这个比例又会和你的学习率纠缠在一起。

**还有一种全部被 mask 掉的情况。**packing 之后的某个 microbatch 可能整批都被 mask，
这时这段代码返回 NaN（`F.cross_entropy` 也一样），下一步就传染到每个参数上。
应该返回一个仍然挂在计算图上的零。

**练习** —— [`p08`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p08_cross_entropy.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p08_cross_entropy.md) · 10 分钟 · 冷启动集

---

<a id="b3-2"></a>
### B3.2 Loss masking 与 packing

两件看着像管道活、其实是正确性的事。

```python
labels = input_ids.clone()                    # input_ids: (B, T)
for i, n in enumerate(prompt_lens):
    labels[i, :n] = -100                      # 每条样本用自己的 prompt 长度
labels[attention_mask == 0] = -100            # padding 也要屏蔽
```

**白板上要躲开的那个手滑：**在 `(B, T)` 张量上写 `labels[:len(prompt_ids)] = -100`，
切的是 **batch** 维，整条整条地抹掉前几个样本，而不是屏蔽掉每条样本的 prompt。
它能跑，能训，而且是错的。

**Packing** 把多条短样本拼成一条定长序列，避开 padding 的浪费——那份浪费常常是你算力的一半。
代价是 token 现在能跨文档边界做 attention。要么用 varlen kernel（配 `cu_seqlens` 的
FlashAttention），要么用块对角 mask；另外 `position_ids` 要按文档重置，
否则第二篇文档是从位置 512 开始的。

**练习** —— [`p09`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p09_loss_masking.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p09_loss_masking.md) · 10 分钟

---

<a id="b3-3"></a>
### B3.3 干别的之前，先过拟合十条样本

```python
def overfit_tiny(steps=2000, lr=0.5):
    torch.manual_seed(0)
    x, y = torch.randn(10, 4), torch.randint(0, 3, (10,))
    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3))
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(steps):
        opt.zero_grad()          # 忘掉这一行是最常见的单个 bug
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(x)
        return F.cross_entropy(logits, y).item(), (logits.argmax(-1) == y).float().mean().item()
```

**一个模型如果连十条样本都背不下来，bug 在你的代码里，不在你的超参里。**
这是机器学习里最便宜的诊断，也是最多人跳过的那个。面试时要把它说出口——
它说明你真的调过训练，而不只是读过。

**这三行的顺序有讲究，而且真的会被问。**`zero_grad` → `backward` → `step`。
梯度默认是*累加*的，所以跳过 `zero_grad` 就是把每一步的梯度全叠在一起；
跳过 `step` 就什么都不更新，loss 曲线一条平线；在 `backward` 之前调 `step`，
更新用的是上一轮的陈旧梯度。

> **梯度为什么要默认累加**——它明明制造了这么多 bug：因为跨 micro-batch 的梯度累积正是靠它，
> 而那是你在有限显存下拿到大 effective batch 的办法。默认值服务的是更难的那种情况。

**练习** —— [`p10`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p10_training_loop.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p10_training_loop.md) · 20 分钟 · 冷启动集

---

<a id="b3-4"></a>
### B3.4 过滤坏标注

OpenAI 面经里出现过两次。它不是一道建模题——它考的是你能不能在不过度设计的前提下，
把标签噪声想清楚。

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

**这道题真正在考的判断力落在 `min_items` 上。**一个只标了两条、两条都和多数不一致的标注员，
看起来糟透了，但两个样本是噪声，不是证据。没有这个下限，你会把每一个标得少的标注员都标出来，
顺手扔掉好数据。这话要在被问之前说出来；它是「一个过滤器」和「一个吃掉你数据集的启发式」之间的分界。

**三个值得先备好的追问。**当坏标注员在某一条上恰好占多数时，多数投票是循环论证——
真实的流水线会掺进已知答案的 gold-standard 条目做独立校验。不一致不等于错误：
真正有歧义的条目上人人都不一致，于是条目难度和标注员质量是混淆在一起的，
而 Dawid-Skene 那一类模型做的正是把两者联合估计出来。还有，系统性有偏的标注员比随机出错的更危险，
因为他们的错误是相关的，平均不掉。

**练习** —— [`p26`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p26_data_filtering.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p26_data_filtering.md) · 20 分钟 · *OpenAI 2+*

---

<a id="section-b4"></a>

## B4 · 手推 backward

OpenAI 面经里 autograd 出现过两次，attention backward 至少一次。这道题问的从来不是
「你还记不记得 chain rule」——它问的是你在用它的时候能不能把形状跟住，
以及你知不知道框架替你做了什么。

---

<a id="b4-1"></a>
### B4.1 能生成所有 backward 的那条规则

你不需要背每一层的公式。一条规则就能把它们全部重新生成出来：

> **对任何张量的梯度都和那个张量同形**，而它是用传进来的梯度和其余操作数拼出来的，
> 拼法就是让形状对上。通常有且只有一种缩并方式能对上。

对 $$Z = XW + b$$，其中 $$X: (m, n_\text{in})$$、$$W: (n_\text{in}, n_\text{out})$$：

$$\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Z}W^\top,\qquad
\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Z},\qquad
\frac{\partial L}{\partial b}=\sum_i \frac{\partial L}{\partial z_{i}}$$

用形状验一遍：$$(m, n_\text{out}) \times (n_\text{out}, n_\text{in})$$ 给出 $$X$$ 的形状，
$$(n_\text{in}, m) \times (m, n_\text{out})$$ 给出 $$W$$ 的。bias 的梯度要在 batch 维上求和，
因为 forward 里的广播对应 backward 里的求和——这一对关系值得明确说出口，
因为它对你以后写的每一次广播都成立。

**backward 为什么大约是 forward 的两倍。**每层你要算两个乘积而不是一个：对输入的梯度，
用来继续往回传；对权重的梯度，用来更新。所以每 token 合计 $$2N + 4N = 6N$$ FLOPs（第一篇 A10.0）。

---

<a id="b4-2"></a>
### B4.2 四十行的 autograd

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
            self.grad += other.data * out.grad      # 是 +=，不是 =
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
        for v in reversed(topo):                    # 逆拓扑序
            v._backward()
```

**两件事撑起整个答案。**

**用 `+=` 而不是 `=`。**一个被用在两处的节点会从两条路径各收到一份梯度，
多元 chain rule 说它们要相加。赋值会悄悄只留下最后那一份——而在每个节点只被用一次的图上，
这个 bug 完全看不见，偏偏那正是你会拿来测试的那种图。

**逆拓扑序。**一个节点的 backward 只有在它所有的消费者都贡献完之后才能跑。
朴素的 DFS 或 BFS 在任何菱形图上都会给出错误的顺序。

> **追问：PyTorch 为什么动态建图？**因为那张图就是「实际跑过的那些算子」，边跑边记下来——
> 这正是控制流、循环和依赖数据的形状不需要任何编译步骤就能工作的原因。
> 代价是每次迭代都要重建一遍，而 `torch.compile` 抢回来的就是这部分。

**练习** —— [`p11`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p11_autograd.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p11_autograd.md) · 30 分钟 · *OpenAI 2+*

---

<a id="b4-3"></a>
### B4.3 Attention 的 backward

手推这个的意义在 softmax 的 Jacobian，那是唯一一处不显然的地方。

对逐行的 $$P = \text{softmax}(S)$$，Jacobian 是
$$\partial p_i/\partial s_j = p_i(\delta_{ij} - p_j)$$，于是 vector-Jacobian product
塌缩成一行就写得完的东西，全程不需要把那个 $$T \times T \times T$$ 的 Jacobian 显式造出来：

$$\frac{\partial L}{\partial S} = P \odot \left(\frac{\partial L}{\partial P}
- \Big(\sum_j \frac{\partial L}{\partial p_j} p_j\Big)\right)$$

```python
def attention_backward(d_out, cache):
    q, k, v, p, scale = cache
    d_v = p.transpose(-2, -1) @ d_out
    d_p = d_out @ v.transpose(-2, -1)
    # softmax 的 VJP：逐元素，再减掉那个行和项
    d_s = p * (d_p - (d_p * p).sum(-1, keepdim=True))
    d_s = d_s * scale
    d_q = d_s @ k
    d_k = d_s.transpose(-2, -1) @ q
    return d_q, d_k, d_v
```

**被 mask 的位置会怎么样。**它们的 $$p = 0$$，所以 `p * (...)` 自动把它们的梯度清零。
backward 里不需要再 mask 一次——这个细节会让人意外，值得主动提一句。

**为什么它在面试之外也重要：**FlashAttention 在片上重算、而不是从 HBM 读回来的，
正是这个 backward；它之所以能这么干，是因为 $$P$$ 从 $$Q$$ 和 $$K$$ 重新生成很便宜、
存下来却很贵（B6.2）。

**练习** —— [`p12`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p12_attention_backward.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p12_attention_backward.md) · 25 分钟 · *OpenAI* · MLP 版本是 `p13`

---

<a id="section-b5"></a>

## B5 · 推理与采样

短的一节，两道题，而且两道都比看上去微妙。

---

<a id="b5-1"></a>
### B5.1 Temperature、top-k、top-p

```python
def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0:                       # greedy；顺带挡住这次除法
        return int(logits.argmax())
    logits = logits / temperature

    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        srt, idx = torch.sort(logits, descending=True)
        probs = F.softmax(srt, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        drop = cum - probs >= top_p            # 独占前缀和：跨过阈值的那个要留下
        drop[0] = False                        # 概率最大的 token 永远保留，p=0 也不会清空
        srt = srt.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, srt)

    return int(torch.multinomial(F.softmax(logits, dim=-1), 1))
```

**顺序有讲究：先 temperature，再 top-k，最后 top-p。**temperature 改变的是截断所作用的那个分布，
所以把它放到最后，等于是在一个错误的分布上挑 nucleus。

**那个会悄悄改掉你采样分布的 off-by-one。**你要的是*累计质量达到 p 的最短前缀*，
也就是说跨过阈值的那个 token 要**留下**。`cum - probs` 是独占前缀和——
严格排在这个 token 之前的那部分质量——在它已经超过 `p` 的地方丢弃，才是对的。
写成 `cum >= top_p` 就把跨阈值的那个丢掉了：在 `[0.5, 0.3, 0.15, 0.05]` 这样的分布上取
`p = 0.9`，你会不声不响地从两个 token 里采样，而不是三个。

**`temperature == 0` 需要一个显式分支**，否则你在除零。这个 bug 真的在生产推理服务里上线过。

**追问：**top-p 为什么通常比 top-k 好？因为候选集的大小会跟着模型的置信度自适应。
模型有把握时，nucleus 就是一两个 token；没把握时它自己变宽。
固定的 $$k$$ 要么在有把握的那些步上放得太松，要么在不确定的那些步上卡得太死。

**练习** —— [`p14`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p14_sampling.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p14_sampling.md) · 15 分钟 · 冷启动集

---

<a id="b5-2"></a>
### B5.2 Speculative decoding

有意思的性质是它**精确**——它不是在近似目标模型的分布，它就是在复现那个分布。

```python
def speculative_accept(p_target, q_draft, token, u):
    if u <= min(1.0, (p_target[token] / q_draft[token]).item()):
        return int(token), True
    resid = torch.clamp(p_target - q_draft, min=0)
    return int(torch.multinomial(resid / resid.sum(), 1)), False
```

以概率 $$\min(1, p(x)/q(x))$$ 接受草稿模型给的 token；拒绝时，
从归一化后的残差 $$\propto \max(0, p - q)$$ 里采一个。这就是拒绝采样，
得到的样本可证明服从 $$p$$。

**被问到就一行证给他看。**吐出 $$x$$ 的概率是
$$q(x)\min(1, p/q) + P(\text{reject})\cdot\frac{\max(0, p-q)}{\sum_y \max(0, p-q)}$$。
第一项是 $$\min(q, p)$$，第二项恰好补上缺掉的那份 $$\max(0, p-q)$$，加起来正是 $$p(x)$$。

> **值得写的那个测试。**采 20 万次，把经验分布和目标分布比一比。
> 「精确」是一个你能验的断言，那就去验——参考实现就是这么做的。

**加速从哪来，又从哪没的。**decode 是带宽受限的，FLOPs 闲着，
所以在一次并行 forward 里验证 $$k$$ 个草稿 token，墙钟时间和生成一个差不多。
batch 变大之后你不再缺带宽了，验证反过来要和此刻稀缺的算力抢资源，
收益缩到零，然后转负。它是给交互式服务用的延迟优化，不是吞吐优化。

**练习** —— [`p15`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p15_speculative.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p15_speculative.md) · 20 分钟

---

<a id="section-b6"></a>

## B6 · 高效实现

两道题，其实是同一个想法看了两遍：softmax 可以在从不同时持有全部输入的情况下算出来，
而长上下文 attention 之所以可行，靠的就是这一个事实。

---

<a id="b6-1"></a>
### B6.1 流式 softmax

softmax 看上去必须先扫完一整遍才能归一化任何东西——你需要最大值来保稳定，需要和来当分母。
其实不必。维护一个滚动的最大值 $$m$$、一个滚动的分母 $$\ell$$、一个滚动的分子，
每当新的一块暴露出更大的最大值时，重新缩放一次。

```python
m = -inf; l = 0.0; acc = 0.0
for s_block, v_block in blocks:
    m_new = max(m, s_block.max())
    correction = exp(m - m_new)          # 把此前累积的一切重新缩放
    l = l * correction + exp(s_block - m_new).sum()
    acc = acc * correction + exp(s_block - m_new) @ v_block
    m = m_new
out = acc / l
```

**修正因子就是这个算法的全部。**此前累积的一切都是相对旧的最大值算出来的；
乘上 $$e^{m_\text{old}-m_\text{new}}$$ 就把它改写成相对新最大值的表达。
$$\ell$$ 和累加器两者都要乘，漏掉累加器是经典 bug——分母对了、分子没对，
产出一个看起来很合理但错的结果。

**这是精确的**，不是近似。拿它跟朴素 softmax 断言比对；参考实现就是这么写的。

> **知道了能加分：**这个递推是 Milakov & Gimelshein（2018），早于 FlashAttention。
> FlashAttention 的贡献不是这个递推，而是搭在它上面的 IO 感知 tiling。

**练习** —— [`p16`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p16_online_softmax.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p16_online_softmax.md) · 15 分钟

---

<a id="b6-2"></a>
### B6.2 分块的 FlashAttention forward

现在把 $$V$$ 也放进循环里跑这个递推，query 块和 key 块两个方向都做 tiling，
你就得到了 FlashAttention 的 forward。

```python
for i, q_block in enumerate(query_blocks):
    m_i = full(-inf); l_i = zeros(); acc_i = zeros()
    for j, (k_block, v_block) in enumerate(kv_blocks):
        if causal and j_start > i_end:        # 整块都在未来
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

**关于它要说三件事。**

**显存从 $$O(N^2)$$ 降到 $$O(N)$$**，因为分数矩阵从来没有被显式造出来——
同一时刻只有一块存在，而且在 SRAM 里。

**FLOPs 是*涨*的，不是降的。**backward 在片上重算注意力矩阵，而不是去读存好的那一份。
说「FlashAttention 减少了计算量」，这个回答会直接标明你读的是摘要而不是论文。

**它照样更快，因为这个算子受限的是 HBM 流量，不是算术。**拿 FLOPs 换访存量，
在 roofline 的访存受限那一侧是赚的，而知道自己在哪一侧才是真本事。

**值得一提的因果优化：**有因果 mask 时，完全落在对角线以上的块可以整块跳过，
只有对角线上的块需要逐元素 mask。这接近 2× 的节省，是 tiling 白送的。

**练习** —— [`p17`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p17_flash_attention.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p17_flash_attention.md) · 25 分钟

---

<a id="section-b7"></a>

## B7 · 后训练算法

整页里出现频率仅次于 transformer 调试的就是 GRPO：OpenAI 和 Anthropic 都有，
四份以上面经，而且通常是*调试*题而不是从零写（B9.3）。但你还是要从零写一遍——
你没亲手拼过的目标函数，你调不了。

---

<a id="b7-1"></a>
### B7.1 LoRA

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False              # base 冻住：赚的就是这里
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B 保持零，于是 B @ A == 0，adapter 在第 0 步是精确的恒等

    def forward(self, x):
        return self.base(x) + x @ self.A.T @ self.B.T * self.scaling

    def merged_weight(self):
        return self.base.weight + (self.B @ self.A) * self.scaling
```

**它在验两条性质。**初始化时是恒等，这要求 `B = 0`——两个都随机初始化会毁掉你的起点，
而这正是「用过库里的 LoRA、却从没读过它」的破绽。以及**无损合并**：
改造后的层就是一个权重矩阵，训完之后推理开销为零，不像 adapter 层那样加深度。
这才是 LoRA 真正胜出的原因。

**显存省在哪**——不是省在权重上。base 还是得常驻。省的是优化器状态和梯度：
全参 AdamW 微调大约是每参数 16 字节，base 冻住之后它只占 bf16 权重那 2 字节，
剩下的 14 字节只作用在 adapter 上。70B 模型上就是从 1,120 GB 的状态降到大约 140 GB。

**练习** —— [`p18`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p18_lora.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p18_lora.md) · 10 分钟 · 冷启动集

---

<a id="b7-2"></a>
### B7.2 GRPO 目标函数

```python
r = rewards.view(-1, G)
adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
adv = adv.reshape(B, 1)                    # 每条 completion 一个标量，广播到各个 token

ratio  = (logp - logp_old).exp()
policy = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)

log_ratio = logp_ref - logp
kl = log_ratio.exp() - log_ratio - 1.0     # k3：无偏，而且非负

loss = ((policy + beta * kl) * mask).sum() / mask.sum()
```

**四个细节，每一个都是 bug 的藏身处**——和 Anthropic 那道调试题埋的正是同样四个（B9.3）。

**分母里那个 epsilon 不是装饰。**一组里每条 completion 拿到同样的 reward 时，标准差是零，
而这是*常见*情况：策略总是做对、或者总是做错的那些 prompt。没有 epsilon 你会拿到 NaN，
下一步就传染到每一个参数上。但 epsilon 救不了 $$G = 1$$：单个样本的无偏 `std` 在加
epsilon 之前就已经是 NaN 了，这种情况要单独判掉，或者传 `correction=0`。

**ratio 是对数差的 `exp`。**on-policy 时新旧 log-prob 相等，ratio 必须正好是 1，
未截断的 surrogate 必须等于 advantage。裸的对数差在那里给出零——
恰好在训练起点上没有梯度。

**KL 是 loss 里的逐 token 项**，不像 PPO 那样折进 reward 里，而且用的是 Schulman 的 k3 估计量：
取 $$r = \pi_\text{ref}/\pi_\theta$$、样本来自 $$\pi_\theta$$，则
$$\widehat{\mathrm{KL}} = r - \log r - 1$$。它逐样本既无偏*又*非负，
而朴素的 $$-\log r$$ 在单个样本上可能是负的，那是个没有意义的 KL 估计。

**advantage 是 bandit 形状的。**每条 completion 一个标量，广播到每一个 token——
根本不存在任何逐 token 的 credit assignment。这个局限要在被问之前主动说。

**练习** —— [`p19`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p19_grpo_loss.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p19_grpo_loss.md) · 20 分钟 · 冷启动集 · *OpenAI + Anthropic 4+*

---

<a id="b7-3"></a>
### B7.3 DPO

```python
def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()
```

四个 log-prob，一个 sigmoid。没有 reward model，没有 critic，训练循环里没有生成——
它跑在 SFT 的基础设施上，显存大约 2 倍。

**要说出口的那个 sanity check：**在参考策略处 margin 是零，loss 正好是
$$\log 2 \approx 0.693$$。第一步的 loss 不是这个数，就是你的参考 log-prob 算错了。

**它换掉了什么。**它是 off-policy 的——学的是在一个策略正在漂离的分布上收集来的偏好——
而且它挡不住 *likelihood displacement*：margin 变大是因为被拒答案的概率在掉，
而不是被选答案的概率在涨，有时候两个一起往下走。

**练习** —— [`p20`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p20_dpo_loss.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p20_dpo_loss.md) · 15 分钟

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

$$\lambda = 0$$ 退化成单步 TD（低方差，被 critic 的误差带偏）；
$$\lambda = 1$$ 退化成 Monte Carlo（无偏，高方差）。实现完就**把这两个极限断言出来**——
这是能拿到的最便宜的正确性检查，而不用别人提醒就主动给出它，正是这道练习的意义所在。

**循环是倒着跑的**，因为 $$\hat A_t$$ 依赖 $$\hat A_{t+1}$$。
正着写是常见的手滑，而且会产出一堆看起来很合理的数。

**练习** —— [`p21`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p21_gae.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p21_gae.md) · 15 分钟

---

<a id="section-b8"></a>

## B8 · 数据与 tokenization

两个实现。真正会被问的是 BPE；MoE routing 放在这儿，
是因为从零写的题目里只有这一处会碰到稀疏层。

---

<a id="b8-1"></a>
### B8.1 Byte-pair encoding

```python
def bpe_train(text, num_merges):
    ids = list(text.encode("utf-8"))          # 字节：永远不会有 out-of-vocabulary
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
    for pair, new_id in merges.items():       # 学到的顺序，不是频率顺序
        ids = replace_pair(ids, pair, new_id)
    return ids
```

**真正会成为 bug 源头的只有一件事：**编码时施加 merge 的顺序是它们被**学到**的顺序，
不是它们在待编码字符串里的频率顺序。Python 的 dict 保序，所以直接遍历 `merges` 是对的——
但只要你把它们塞进 `set`、排一遍序、或者重建一次 dict，你就得到一个 round-trip 不自洽的
tokenizer。这是很难缠的线上 bug，因为它依赖数据。

**为什么用字节而不是字符。**字节级词表能表示任何输入，于是永远不存在 out-of-vocabulary。
代价是非拉丁文字每个字符要吃掉更多 token，这是实打实的成本问题、也是公平性问题，
值得不等人问就主动提。

**解码表值得顺手写出来**，哪怕没人要求，因为你就是靠它测 round-trip 的：

```python
table = {i: bytes([i]) for i in range(256)}
for (a, b), new in merges.items():
    table[new] = table[a] + table[b]
assert b"".join(table[i] for i in ids).decode("utf-8") == text
```

**必然会来的那个追问：**模型为什么数不清「strawberry」里有几个 r？因为它从来看不见字符。
这个词可能是三个 token，而表示里没有任何东西把 token 内部的字母暴露出来。
这是输入表示的产物，不是推理能力的问题。

**练习** —— [`p22`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p22_bpe.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p22_bpe.md) · 20 分钟 · 冷启动集

---

<a id="b8-2"></a>
### B8.2 带 capacity 的 top-1 MoE routing

```python
def top1_route(logits, capacity):
    gates = F.softmax(logits, dim=-1)          # (T, E)
    gate, expert = gates.max(dim=-1)
    # 每个 expert 按顺序收 token 直到装满；剩下的溢出
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
    f = F.one_hot(gates.argmax(-1), E).float().mean(0)   # 路由到各 expert 的比例
    p = gates.mean(0)                                     # 平均 gate 概率
    return E * (f * p).sum()                              # 均匀时取最小，= 1
```

**token dropping 是大多数人没想过的那部分。**all-to-all 需要定长缓冲区，
所以每个 expert 有一个 capacity 上限；溢出的 token **整层直接跳过**，顺着残差流穿过去。
要主动说出来的后果是：同一个输入，会因为 batch 里还有些什么别的东西，而产出不同的输出。

**auxiliary loss 存在的理由不是 router 没有梯度**——它有梯度。gate 概率乘在所选 expert
的输出上，所以 LM loss 会反传进 router；不可导的只有 top-$$k$$ 这个*选择*动作。
真正的问题是这个梯度会自我强化：拿到更多 token 的 expert 训得更快，于是 router 更偏爱它们，
路由随之坍缩。这一点答对是真正的区分度所在，因为「router 没有梯度」那个版本到处都在被复读。

**这个 loss 为什么是 $$E\sum_e f_e p_e$$：**$$f$$ 不可导（它在数分配次数），$$p$$ 可导，
所以梯度沿 $$p$$ 流动、并以实际观测到的负载加权。它在均匀路由处取到最小，此时等于 1。

**练习** —— [`p23`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p23_moe_routing.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p23_moe_routing.md) · 20 分钟

---

<a id="section-c1"></a>

## C1 · 概率：覆盖大部分题目的四种套路

公式部分直接看 Alisa Liu 公开的数学笔记——分布、期望、不等式、极限定理，全都带证明。
在这里重抄一遍没有意义。那份笔记缺的是**自测这一层**，所以这一节做的是套路清单：
四种技巧，合起来能解掉这些面试轮里绝大多数的概率题，
每一种都配上那句告诉你「就用它」的识别特征。

---

<a id="c1-1"></a>
### C1.1 首步分析

**识别特征：**一个会不断重复的过程，问你某件事发生之前的期望时间。

对第一步做条件，把未知的期望用它们自己写出来。连续掷出两次正面所需的期望次数：

$$E_0=\underbrace{\tfrac 12 (1+E_0)}_\text{tails, no progress}+\underbrace{\tfrac 12(1+E_1)}_\text{heads},
\qquad E_1=\underbrace{\tfrac 12(1+E_0)}_\text{tails, start over}+\underbrace{\tfrac 12\cdot 0}_\text{done}$$

两个方程、两个未知数，$$E_0 = 6$$。

**真正让它成立的是状态选对了。**这里的状态是「我朝 HH 走了多远」，只有三个取值
（还没有、已有一个 H、完成）。状态选错，方程就闭不上。全部技巧都在这个选择上；
代数部分毫无难度。

**同一套路，换更难的外壳：**赌徒破产（状态 = 当前赌本）、随机游走的返回时间、
马尔可夫链击中某个集合的期望步数。

---

<a id="c1-2"></a>
### C1.2 指示变量加期望的线性性

**识别特征：**「期望有多少个 X 会……」——问的是一个计数。

定义 $$X_i = \mathbb 1[\text{item } i \text{ has the property}]$$，求和，然后用
$$\mathbb E[\sum X_i] = \sum \mathbb E[X_i] = \sum P(X_i = 1)$$。

**指示变量独不独立，线性性都成立**，这就是这个技巧强到离谱的全部原因。
相依性通常会让联合分布彻底没法算，却不影响每一个边缘分布依旧简单。

随机置换中不动点个数的期望：由对称性 $$P(X_i = 1) = 1/n$$，所以答案是
$$n \cdot 1/n = 1$$，对每一个 $$n$$ 都成立。从不动点个数的分布出发去算这件事是真的难；
用指示变量一行就完。

**优惠券收集**是同一个想法倒过来用：把总时间拆成两张新券之间的等待时间，
每一段都是几何分布、均值 $$n/(n-k)$$，加起来得到 $$n H_n \approx n \ln n$$。

---

<a id="c1-3"></a>
### C1.3 $$n$$ 个变量的最大值与最小值——走 CDF

**识别特征：**任何关于若干次抽样中最大或最小者的问题。

不要直接去啃密度。最大值的 CDF 简单到近乎平凡，因为「最大值不超过 $$x$$」
就是「它们全都不超过 $$x$$」：

$$F_M(x) = P(\max_i X_i \le x) = [F_X(x)]^n$$

最小值取补：$$P(\min > x) = [1 - F_X(x)]^n$$。真的需要密度，最后再求导。

**值得背下来：**$$[0,1]$$ 上 $$n$$ 个 iid 均匀分布，$$\mathbb E[\max] = n/(n+1)$$、
$$\mathbb E[\min] = 1/(n+1)$$——两者之间的对称性，是这类题任何答案的一个好自检。

---

<a id="c1-4"></a>
### C1.4 把对称性当证明工具

**识别特征：**答案感觉上就不该依赖某个东西。

会碰到的两个例子。随机置换里，元素 $$i$$ 落在位置 $$j$$ 的概率对每一对都是 $$1/n$$
——C1.2 能成立靠的就是这条。秘书问题里，最好的候选人出现在任意给定位置上的概率是均匀的，
所以整个分析只跟最大值*落在哪里*有关。

**Monty Hall 是反例**，之所以被问，正因为它专治对称性直觉。主持人的选择*不*对称
——他从不打开有奖的那扇门——而 2/3 恰恰就是从这个不对称里来的。你要搬对称性，
就说清楚问题在哪个变换下不变；说不出那个变换，你就是在猜。

---

<a id="c1-5"></a>
### C1.5 该抓哪个不等式

| 你手上有 | 用 | 给出 |
|---|---|---|
| 只有均值，$$X \ge 0$$ | 马尔可夫 | $$P(X \ge a) \le \mathbb E[X]/a$$ |
| 均值和方差 | 切比雪夫 | $$P(\|X-\mu\| \ge k\sigma) \le 1/k^2$$ |
| 有界的独立求和 | Hoeffding / Chernoff | 指数尾 |
| 期望的凸函数 | Jensen | $$f(\mathbb E[X]) \le \mathbb E[f(X)]$$ |

**马尔可夫最弱，也最有用**，因为它几乎什么都不要求——而切比雪夫不过是把马尔可夫
用在 $$(X-\mu)^2$$ 上，这句话值得能随口说出来。

**Jensen 是真正出现在 ML 里、而不是出现在智力题里的那一个。**ELBO 之所以是下界、
$$\log \mathbb E[\cdot] \ge \mathbb E[\log \cdot]$$ 之所以在重要性采样里要紧、
KL 散度之所以非负，都是它。

---

<a id="section-c2"></a>

## C2 · 先模拟，再验证

一类自成一体的题型，OpenAI 面经里报过：给你一个物理场景，让你模拟它，
再让你验证采出来的样本和你解析推导出的分布一致。它正好卡在 coding 轮和数学轮之间，
于是两边的准备都把它漏掉了。

---

<a id="c2-1"></a>
### C2.1 旋转的光源

> 一盏灯距离一堵无限长的墙 1 个单位，朝均匀随机的方向照。模拟它打在墙上的位置。
> 那是什么分布？验证它。

**推导。**取 $$\theta \sim \text{Uniform}(-\pi/2, \pi/2)$$，落点就是 $$x = \tan\theta$$。
做密度变换：

$$f_X(x) = f_\Theta(\theta)\left|\frac{d\theta}{dx}\right|
= \frac{1}{\pi}\cdot\frac{1}{1+x^2}$$

这就是标准 **Cauchy** 分布。选它来出题，是因为它的病态既容易演示，又很难糊弄过去。

```python
def light_source_samples(n, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    return np.tan(theta)
```

**这道题的题眼是均值不存在。**$$\int |x| f(x)\,dx$$ 发散，所以大数定律不适用，
样本均值永远不会稳定下来——它一直在游走，时不时因为某个样本落到很远的尾巴上而猛跳一次。
别只是断言，要演示出来：分别在 $$10^4$$、$$10^5$$ 和 $$4\times10^5$$ 个样本处算滑动均值，
指出它并没有在收缩。换成任何一个均值有限的分布，误差都会按 $$1/\sqrt n$$ 掉下去。

**中位数是良态的**，改用它来估位置参数，才是实践上正确的应对。

**验证里埋着一个陷阱，而且是个好陷阱。**在 $$[-5, 5]$$ 这种窗口上拿直方图跟解析 PDF 比，
粗心就会*对不上*，而原因不是代码写错了。NumPy 的 `density=True` 是在你画出来的那些 bin 上
归一化的，但真正的 Cauchy 只有 $$\tfrac{2}{\pi}\arctan 5 = 0.874$$ 的质量落在那个窗口里。
尾巴重到这个程度，忽略截断会把你的直方图整体抬高 14%，让一次正确的模拟看上去是错的。
要跟*条件*密度比：

```python
in_range = 2 * np.arctan(L) / np.pi
assert np.abs(hist - cauchy_pdf(centres) / in_range).max() < 0.01
```

> **这道题好在哪里。**推导干净，模拟是任何一个合格候选人都写得出来的，
> 然后那一步验证，把认真核对自己工作的人和大概核对一下的人分开了。
> 截断修正不是什么小聪明——真实实验能不能复现，靠的就是这类东西。

**练习** —— [`p27`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p27_cauchy_simulation.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p27_cauchy_simulation.md) · 20 分钟 · *OpenAI*

---

<a id="c2-2"></a>
### C2.2 通用套路

这个设定是可以推广的，手上有一套固定流程，就不用在时间压力下临场发挥。

**先推导，再模拟。**先模拟你就没有可对照的东西，而「直方图看着差不多」不算答案。
单调的 $$g$$ 的变换公式是 $$f_Y(y) = f_X(g^{-1}(y))\,|dg^{-1}/dy|$$；
$$g$$ 不单调，就在各个分支上求和。

**分三个层次验证，从最便宜的开始。**矩（均值、方差）在存在的时候一行就能算。
用 Kolmogorov–Smirnov 统计量去比 CDF，比直方图更灵敏，而且不用挑分箱。
直方图看着最有说服力，也最容易出错，原因就是上面那个截断。

**主动说你打算拿方差怎么办。**样本量决定了你的分辨率：$$n$$ 个样本下，
一个概率为 $$p$$ 的 bin 相对误差大约是 $$1/\sqrt{np}$$，
所以尾部的 bin 很吵，不该在所有 bin 上一律用同样收紧的容差。

**把标准变换一一点名**，因为出题人想听的通常就是其中之一：
凡是有闭式分位函数的都用逆 CDF 采样，高斯用 Box–Muller，指数分布用 $$-\log U/\lambda$$，
Cauchy 用两个标准正态之比——最后这个正好是这道题的一个漂亮交叉验证，
因为它应该给出和正切构造完全相同的分布。

---

<a id="section-c3"></a>

## C3 · 线性代数

大多数人准备时的缺口，包括 Alisa 那份其余部分都很好的笔记——它覆盖了概率和微积分，
唯独没有这块。它会被问，是因为你一整天摸到的每一个对象都是矩阵，
也因为这些问题顺带就在检验你到底懂不懂自己天天在用的方法。

---

<a id="c3-1"></a>
### C3.1 其余一切都从中推出的四条事实

**矩阵是一个线性映射，形状告诉你它连接的是哪两个空间。**形状为 $$(m, n)$$ 的 $$W$$
把 $$\mathbb R^n \to \mathbb R^m$$。你要是这样读矩阵、而不是把它读成一格一格的数字，
绝大多数形状 bug 自己就化掉了。

**秩是输出空间里真正被覆盖到的维数。**低秩矩阵把输入压进一个子空间，
LoRA 吃的正是这一点：内维为 $$r$$ 的 $$BA$$ 只能在一个 $$r$$ 维子空间里挪动权重
——这既是它便宜的原因，也是它装不进大量新知识的原因。

**特征向量是映射只做缩放的那些方向**，$$Av = \lambda v$$。方阵才有；
对称矩阵的特征向量还构成一组正交基、特征值全是实数——Hessian 和协方差矩阵之所以这么好处理，
就是因为这个。

**SVD 对每一个矩阵都成立**，方阵不方阵都行：$$A = U\Sigma V^\top$$，
一次正交旋转、沿坐标轴的一次缩放、再一次旋转。奇异值就是那些缩放因子，
把小的截掉，就是 Frobenius 范数意义下最好的秩 $$k$$ 近似。
仅这一条事实就撑起了 PCA、低秩压缩，以及人们判断一次权重更新是不是「真的」低秩时的那套推理。

---

<a id="c3-2"></a>
### C3.2 半正定，以及它为什么反复出现

当 $$x^\top M x \ge 0$$ 对所有 $$x$$ 成立时，$$M$$ 是半正定的；等价地说，就是特征值全部非负。

**它在 ML 里真正决定了什么的三个地方。**协方差矩阵在构造上就是半正定的，
因为 $$x^\top \Sigma x$$ 是某个投影的方差，而方差非负。Hessian 处处半正定意味着目标是凸的，
于是任何驻点都是全局最小——神经网络的 loss 明摆着不是这样，所以我们改去谈鞍点。
还有核矩阵必须半正定，核技巧才对应到某个空间里的一个内积。

**值得提前备好的追问：高维下为什么典型的临界点是鞍点？**在一个随机的临界点上，
Hessian 的每个特征值大致以对半的概率取正或取负，所以 $$d$$ 个符号全部一致的概率
量级是 $$2^{-d}$$。$$d$$ 上到百万量级时，局部极小相对鞍点少到可以忽略
——这就是为什么从二维图像来的二阶直觉会把人带偏。

---

<a id="c3-3"></a>
### C3.3 范数、条件性，以及那些会炸的东西

**用哪个范数，以及为什么这件事重要。**梯度裁剪量的是 $$\ell_2$$ 范数；
Frobenius 范数就是把矩阵拉平之后的 $$\ell_2$$ 范数；谱范数是最大的那个奇异值，
也就是一个矩阵最多能把某个向量拉长多少。管稳定性的是谱范数，
因为它给出了一个扰动穿过一层之后能放大到多少的上界。

**条件数** $$\kappa = \sigma_\max/\sigma_\min$$ 告诉你相对误差会被放大多少倍。
我们要归一化输入是因为它，病态问题需要小学习率是因为它，Adam 的逐参数缩放有用也是因为它
——那个缩放就是在近似一个对角预条件子。

**由此得出两条数值规则。**能避开就绝不显式构造 $$X^\top X$$ 去解最小二乘：
那会把条件数平方，于是你损失掉两倍的有效位数。以及绝不为了解 $$Ax = b$$ 去求矩阵的逆
——用分解（`np.linalg.solve`，不是 `inv(A) @ b`），更快，条件性也更好。

---

<a id="c3-4"></a>
### C3.4 你真正用得上的矩阵微积分

四个恒等式几乎能把其余全部重新生成出来，而且每一个都能用形状验一遍。

$$\frac{\partial}{\partial x}(a^\top x) = a,\qquad
\frac{\partial}{\partial x}(x^\top A x) = (A + A^\top)x,\qquad
\frac{\partial}{\partial X}\operatorname{tr}(AX) = A^\top,\qquad
\frac{\partial}{\partial X}\|X\|_F^2 = 2X$$

**二次型在 $$A$$ 对称时给出 $$2Ax$$**，你遇到的就是这种情形，
但把一般形式说出来，能表明你不是只背了那个特例。

**比背公式更管用的规则**是 B4.1 里那条：对某个张量的梯度和那个张量同形，
而在手上的操作数里，通常有且只有一种缩并方式能产出它。你要是能只靠形状推出
$$\partial L/\partial W = X^\top \partial L/\partial Z$$，就不需要什么表。

> **它会在没有预告的地方冒出来：**推线性层的 backward（B4.1）、softmax 的 Jacobian（B4.3）、
> 正规方程，以及任何一个「为什么某条更新规则长成这个样子」的问题。
