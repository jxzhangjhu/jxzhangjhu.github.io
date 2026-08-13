---
layout: post
title: "面试题库 II · Coding + Math：要写，不要只读（中文版）"
date: 2026-08-09 12:00:00
author: Jiaxin Zhang
description: "用于 frontier-lab 风格 coding 练习的完整实现，全部带测试——attention、KV cache、RoPE、采样、GRPO、BPE，加上配套的概率与线性代数，以及一套限时练习工具。"
tags: interviews llm coding math pytorch qbank
categories: research-notes
giscus_comments: true
related_posts: false
ai_assisted: true
---

<div class="lang-switch"><a href="/blog/2026/interview-coding/">English</a> · <strong>中文</strong></div>

<div class="lang-switch"><a href="/blog/2026/interview-knowledge-zh/">I · 知识</a> · <strong>II · 代码 + 数学</strong> · <a href="/blog/2026/interview-discussion-zh/">III · 讨论 + BQ</a></div>

第一篇问的是你**想不想得起来**，这一篇问的是你能不能在**空白文件里、有钟在走的情况下写出来**。

这是两种能力，而它们之间的差距就是这个页面要配一个仓库的全部理由。把一段 attention
实现读到觉得「显然」，对你二十分钟内写出一版几乎没有帮助。所以下面的代码是讲解层，
真正练手在 [`interview-practice/`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/tree/master/interview-practice)：28 道带 stub 和测试的题、
10 个 debug drill，外加一个计时的 runner。11 道冷启动题和本文清单
共用同一份 manifest，数字不会各自漂移。

> **每节的结构。**先简述概念，给出完整带注释的实现，列出常见坑，
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
- **[B1 · NumPy 与 PyTorch 基础](#section-b1)** — 2 题
  - [B1.1 向量化：唯一值得背下来的技巧](#b1-1)
  - [B1.2 BatchNorm，以及它为什么有两种模式](#b1-2)
  - [B1.3 会造成静默 bug 的张量语义](#b1-3)
- **[B2 · Transformer 组件](#section-b2)** — 8 题
  - [B2.1 因果多头注意力](#b2-1)
  - [B2.2 KV cache 与增量解码](#b2-2)
  - [B2.3 MHA、MQA、GQA 与 MLA](#b2-3)
  - [B2.4 旋转位置编码](#b2-4)
  - [B2.5 RMSNorm](#b2-5)
  - [B2.6 SwiGLU](#b2-6)
  - [B2.7 拼成一个 block](#b2-7)
- **[B3 · 训练循环](#section-b3)** — 4 题
  - [B3.1 Cross entropy，以及它为什么收 logits](#b3-1)
  - [B3.2 Loss masking 与 packing](#b3-2)
  - [B3.3 干别的之前，先过拟合十条样本](#b3-3)
  - [B3.4 过滤坏标注](#b3-4)
- **[B4 · 手推 backward](#section-b4)** — 3 题
  - [B4.1 检查每个 backward 的约束](#b4-1)
  - [B4.2 最小标量 autograd](#b4-2)
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
- **[C1 · 概率：五种可复用套路](#section-c1)**
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
- **[C4 · 计数](#section-c4)**
  - [C4.1 决定公式的那一个判断](#c4-1)
  - [C4.2 先重复计数，再除掉](#c4-2)
  - [C4.3 容斥原理](#c4-3)
  - [C4.4 计数在哪儿撞上 ML](#c4-4)
- **[C5 · 马尔可夫链与随机游走](#section-c5)**
  - [C5.1 马尔可夫性到底给你换来了什么](#c5-1)
  - [C5.2 赌徒破产](#c5-2)
  - [C5.3 随机游走，以及维数带来的意外](#c5-3)
  - [C5.4 可迁移的例题](#c5-4)
- **[C6 · 统计与估计](#section-c6)**
  - [C6.1 最大似然，以及你的损失函数究竟是什么](#c6-1)
  - [C6.2 偏差、方差，以及有偏估计量何时有用](#c6-2)
  - [C6.3 集中不等式：你到底需要多少样本](#c6-3)
  - [C6.4 假设检验，简单说说，以及它在 ML 里的失效模式](#c6-4)
- **[参考文献](#section-refs)**

---
<a id="section-b0"></a>

## B0 · 这部分该怎么练

阅读正确实现测的是识别，不是回忆。限时练习多了一层要求：
不看答案，自己产出实现和验证它的测试。

所以这个页面**故意只是材料的一半**，另一半是一个仓库。

---

<a id="b0-1"></a>
### B0.1 两层结构

**这个页面是讲解层。**Coding 各节给出经过测试的实现、常见坑和练习所检查的不变量；
数学各节补上推导和闭卷自测。先读一遍建立模型，练不下去时再回来查。

**[`interview-practice/`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/tree/master/interview-practice)
是训练层。**同样这些题以 stub 形式存在——只有签名和 docstring，函数体挖空——
配有按行为性质编写的测试和经过验证的 reference。可点击入口包括
[`run.py`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py)、
[`README.md`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/README.md)
以及 B0.3 自动生成的题目表。

从 `reference.py` 摘出的 Python 代码块默认共用下面的 prelude：

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
```

使用 NumPy 的函数会在内部 import。完整 reference 文件可直接执行；
构建时会从它同步博客代码块及中英文副本。

```bash
git clone https://github.com/jxzhangjhu/jxzhangjhu.github.io.git
cd jxzhangjhu.github.io/interview-practice

python run.py                 # 题目列表、时间预算、你的历史成绩
python run.py p01             # 开始计时，跑这道题的测试
python run.py --cold          # 冷启动那一组，按顺序过
python run.py --reset p01     # 恢复原始 stub 后重做
```

每一行**练习**都同时链接 stub、三级提示、症状导向的测试和 runner，
并保留可复制的 shell 命令。`pNN` 这个编号就是 `run.py` 的参数。

测试是按「诊断」写的，不是单纯报错。因果 mask 漏了未来时，它告诉你
*"perturbing the last token changed earlier outputs: the mask leaks the future"*，
而不是甩给你一堆张量。

> **关于参考解法。**它们在 `interview-practice/reference.py`。
> 它自己的数值自检和 `_validate.py` 都必须通过，页面才可以构建。
> **练的时候不要打开那个文件**，那是三级提示存在的意义。

---

<a id="b0-2"></a>
### B0.2 让练习真正有效的三件事

**加钟。**每道题都有时间预算——多头注意力 20 分钟，RMSNorm 5 分钟，小型 autograd 30 分钟。
限时练习会额外暴露检索和实现错误，例如忘记 `.contiguous()`、mask 取反或除错平方根。

**保留一个小的冷启动集。**只有 manifest 里标了 `cold` 的题需要从空文件写出，值得每周重做；
`python run.py list` 会打印当前集合和数量。其余的标准低一些——想得起解法的形状，
看一眼提示能重建出来就够了；不必把整个题库都维持在同样的背写强度。

**把 bug 单独拿出来练，而且要先练。**实名准备复盘明确提到了 ML debugging 轮，
所以这套题把构造能力和故障识别分开训练，并把 B9 紧接在本节之后。
`d09` 和 `d10` 是根据公开轶事性面经综合出来的 **OpenAI 风格**与
**Anthropic 风格**练习，并非官方题或逐字复刻；其余条目是只错一行的微型练习。
B9 会在你尝试之后解释所有 planted bug。

---

<a id="b0-3"></a>
### B0.3 题目清单

`cold` 是要求能从空文件写出来的那一组。时间预算是面试的预算，不是「读懂答案要多久」。

**把「面经来源」理解成来源标签，不要理解成频率。**它只表示某份公开、轶事性的面经
把相似题目和某家联系起来，不是官方题库，也不能据此估计命中概率。
空白表示「标准练习，但这里没有具体归属」，不表示不重要。

<!-- TABLE -->

| | 题目 | 预算 | 冷启动 | 面经来源 |
|---|---|---|---|---|
| **B1 · NumPy 与 PyTorch** | | | | |
| [p24](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p24_nn_vectorized.py) | [纯 NumPy 的 1-NN，不许循环](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p24_nn_vectorized.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p24_nn_vectorized.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p24_nn_vectorized.py) · [`python run.py p24`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 分钟 | ● |  |
| [p25](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p25_batchnorm.py) | [BatchNorm 前向、梯度与 eval 模式](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p25_batchnorm.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p25_batchnorm.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p25_batchnorm.py) · [`python run.py p25`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 分钟 |  | 作者个人轶事性面经：Datadog |
| **B2 · 组件** | | | | |
| [p01](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p01_mha.py) | [因果多头注意力](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p01_mha.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p01_mha.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p01_mha.py) · [`python run.py p01`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 分钟 | ● |  |
| [p02](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p02_kv_cache.py) | [KV cache 与增量解码](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p02_kv_cache.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p02_kv_cache.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p02_kv_cache.py) · [`python run.py p02`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 分钟 | ● |  |
| [p03](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p03_gqa.py) | [分组查询注意力](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p03_gqa.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p03_gqa.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p03_gqa.py) · [`python run.py p03`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 10 分钟 |  | 作者个人轶事性面经：Datadog |
| [p28](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p28_mla.py) | [多头潜在注意力与压缩缓存](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p28_mla.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p28_mla.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p28_mla.py) · [`python run.py p28`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 25 分钟 |  |  |
| [p04](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p04_rope.py) | [旋转位置编码](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p04_rope.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p04_rope.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p04_rope.py) · [`python run.py p04`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 分钟 | ● |  |
| [p05](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p05_rmsnorm.py) | [RMSNorm](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p05_rmsnorm.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p05_rmsnorm.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p05_rmsnorm.py) · [`python run.py p05`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 5 分钟 | ● |  |
| [p06](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p06_swiglu.py) | [SwiGLU 前馈层](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p06_swiglu.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p06_swiglu.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p06_swiglu.py) · [`python run.py p06`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 5 分钟 |  |  |
| [p07](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p07_transformer_block.py) | [完整的 pre-norm block](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p07_transformer_block.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p07_transformer_block.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p07_transformer_block.py) · [`python run.py p07`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 分钟 |  |  |
| **B3 · 训练** | | | | |
| [p08](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p08_cross_entropy.py) | [交叉熵与 log-sum-exp](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p08_cross_entropy.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p08_cross_entropy.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p08_cross_entropy.py) · [`python run.py p08`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 10 分钟 | ● |  |
| [p09](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p09_loss_masking.py) | [SFT loss masking 与 packing](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p09_loss_masking.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p09_loss_masking.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p09_loss_masking.py) · [`python run.py p09`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 分钟 |  |  |
| [p10](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p10_training_loop.py) | [把一个小 batch 过拟合](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p10_training_loop.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p10_training_loop.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p10_training_loop.py) · [`python run.py p10`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 分钟 | ● |  |
| [p26](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p26_data_filtering.py) | [过滤劣质人工标注](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p26_data_filtering.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p26_data_filtering.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p26_data_filtering.py) · [`python run.py p26`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 分钟 |  |  |
| **B4 · 反向** | | | | |
| [p11](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p11_autograd.py) | [最小标量 autograd](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p11_autograd.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p11_autograd.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p11_autograd.py) · [`python run.py p11`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 30 分钟 |  |  |
| [p12](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p12_attention_backward.py) | [手写 attention 反向](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p12_attention_backward.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p12_attention_backward.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p12_attention_backward.py) · [`python run.py p12`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 25 分钟 |  |  |
| [p13](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p13_mlp_backward.py) | [手写 MLP 反向](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p13_mlp_backward.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p13_mlp_backward.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p13_mlp_backward.py) · [`python run.py p13`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 分钟 |  |  |
| **B5 · 推理** | | | | |
| [p14](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p14_sampling.py) | [temperature / top-k / top-p](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p14_sampling.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p14_sampling.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p14_sampling.py) · [`python run.py p14`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 分钟 | ● |  |
| [p15](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p15_speculative.py) | [投机解码的接受/拒绝](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p15_speculative.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p15_speculative.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p15_speculative.py) · [`python run.py p15`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 分钟 |  |  |
| **B6 · 效率** | | | | |
| [p16](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p16_online_softmax.py) | [流式 softmax](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p16_online_softmax.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p16_online_softmax.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p16_online_softmax.py) · [`python run.py p16`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 分钟 |  |  |
| [p17](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p17_flash_attention.py) | [分块 FlashAttention 前向](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p17_flash_attention.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p17_flash_attention.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p17_flash_attention.py) · [`python run.py p17`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 25 分钟 |  |  |
| **B7 · 后训练** | | | | |
| [p18](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p18_lora.py) | [LoRA 与无损合并](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p18_lora.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p18_lora.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p18_lora.py) · [`python run.py p18`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 10 分钟 | ● |  |
| [p19](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p19_grpo_loss.py) | [GRPO 目标](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p19_grpo_loss.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p19_grpo_loss.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p19_grpo_loss.py) · [`python run.py p19`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 分钟 | ● |  |
| [p20](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p20_dpo_loss.py) | [DPO 损失](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p20_dpo_loss.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p20_dpo_loss.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p20_dpo_loss.py) · [`python run.py p20`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 分钟 |  |  |
| [p21](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p21_gae.py) | [广义优势估计（GAE）](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p21_gae.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p21_gae.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p21_gae.py) · [`python run.py p21`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 15 分钟 |  |  |
| **B8 · 数据** | | | | |
| [p22](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p22_bpe.py) | [Byte-pair encoding](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p22_bpe.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p22_bpe.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p22_bpe.py) · [`python run.py p22`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 分钟 | ● |  |
| [p23](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p23_moe_routing.py) | [带容量的 top-1 MoE 路由](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p23_moe_routing.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p23_moe_routing.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p23_moe_routing.py) · [`python run.py p23`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 分钟 |  |  |
| **C2 · 模拟** | | | | |
| [p27](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p27_cauchy_simulation.py) | [旋转光源 → Cauchy 分布](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p27_cauchy_simulation.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p27_cauchy_simulation.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p27_cauchy_simulation.py) · [`python run.py p27`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 20 分钟 |  |  |

| | Drill | 预算 | 症状 | 面经来源 |
|---|---|---|---|---|
| [d09](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d09_minigpt.py) | [调试 miniGPT](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d09_minigpt.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d09_minigpt.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d09_minigpt.py) · [`python run.py --drill d09`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 35 分钟 | 四条模型不变量失败；随后实现 KV cache | OpenAI 风格；基于轶事性面经 |
| [d10](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d10_grpo_loop.py) | [调试 GRPO 循环](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d10_grpo_loop.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d10_grpo_loop.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d10_grpo_loop.py) · [`python run.py --drill d10`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 30 分钟 | 采样、优势值和策略比率违反不变量 | Anthropic 风格；基于轶事性面经 |
| [d01](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d01_mask_inverted.py) | [因果 mask 故障](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d01_mask_inverted.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d01_mask_inverted.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d01_mask_inverted.py) · [`python run.py --drill d01`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 3 分钟 | attention 看到了未来 |  |
| [d02](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d02_missing_contiguous.py) | [合并注意力头故障](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d02_missing_contiguous.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d02_missing_contiguous.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d02_missing_contiguous.py) · [`python run.py --drill d02`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 3 分钟 | 合并注意力头时报错或数值交错 |  |
| [d03](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d03_top_p_off_by_one.py) | [nucleus 支持集故障](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d03_top_p_off_by_one.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d03_top_p_off_by_one.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d03_top_p_off_by_one.py) · [`python run.py --drill d03`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 4 分钟 | nucleus 的支持集不正确 |  |
| [d04](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d04_cache_mask_offset.py) | [缓存 mask 故障](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d04_cache_mask_offset.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d04_cache_mask_offset.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d04_cache_mask_offset.py) · [`python run.py --drill d04`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 5 分钟 | 缓存解码看不到本应可见的历史 |  |
| [d05](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d05_lora_both_random.py) | [LoRA 初始化故障](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d05_lora_both_random.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d05_lora_both_random.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d05_lora_both_random.py) · [`python run.py --drill d05`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 3 分钟 | 新建 adapter 改变了基础模型 |  |
| [d06](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d06_softmax_overflow.py) | [softmax 稳定性故障](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d06_softmax_overflow.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d06_softmax_overflow.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d06_softmax_overflow.py) · [`python run.py --drill d06`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 3 分钟 | 有限的大 logits 产生非有限概率 |  |
| [d07](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d07_wrong_scale.py) | [attention 缩放故障](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d07_wrong_scale.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d07_wrong_scale.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d07_wrong_scale.py) · [`python run.py --drill d07`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 3 分钟 | attention 与缩放点积定义不一致 |  |
| [d08](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d08_prompt_not_masked.py) | [SFT 目标故障](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d08_prompt_not_masked.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d08_prompt_not_masked.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d08_prompt_not_masked.py) · [`python run.py --drill d08`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) | 4 分钟 | 改动 prompt token 会改变 completion loss |  |

<!-- TABLE -->

**一个能塞进上班日程的四周轮转。**第一周先做两道旗舰 drill（`d09`、`d10`），
再把冷启动那组各做一遍，允许看提示——目标是覆盖，不是速度。第二周做冷启动集之外的题，规则相同。
第三周冷启动那组再来一遍，不看提示、严格计时，没过的记进一张短名单。
第四周做短名单加微型 drill，面试前一天把冷启动组完整过一遍。

> **避免只练已经通过的题。**runner 会把本地结果与耗时记录在
> `interview-practice/attempts.local.json`；优先回看失败项和慢通过项。

---

<a id="section-b9"></a>

## B9 · 调试

时间不够就先看这一节。

实名准备复盘明确提到了 ML debugging 轮：
[Alisa Liu](https://alisawuffles.github.io/blog/job-search/) 点名了实现和调试 transformer，
[Silvia Sapora](https://silviasapora.github.io/blog/ml-interviews.html) 列出了训练循环找 bug。
这些公开证据是轶事性的，**不足以支持可靠的逐公司频率排名**。但实践结论仍然成立：
调试是一项独立能力，只练从零实现不会自动把它练出来。

这里的练习采用可复现的形式：代码能跑，但违反行为不变量；bug 是逻辑上的，不是语法上的；
长练习修完后还有扩展任务。`d09`、`d10` 是综合练习，不是泄露题或官方题。

---

<a id="b9-1"></a>
### B9.1 在钟走着的时候管用的方法

下面这套方法在钟走着的时候很稳，而「再仔细读一遍代码」不算方法。

**先做到确定性复现。**所有随机源都固定种子，并明确 train/eval 模式。
排查推理不一致时切到 `eval()` 并用贪心解码；排查训练 bug 时保留 `train()`，
但固定数据顺序和随机性。输出自己会动的话，你根本判断不了某个改动有没有用。

**用断言定位，不要靠读。**每一步都把形状打出来。把你知道必须成立的不变量断言出来：
注意力每行和为 1、带 cache 的解码等于完整重算、完整 forward 的位置下标等于
`arange(T)`。每一条断言都缩小搜索空间，并留下可复现的故障边界。

**一次只修一个 bug，然后重跑。**bug 之间会互相掩盖。下面那道练习里，
端到端 loss 无法告诉你故障来自位置下标还是 head/time 布局；聚焦断言才能把两者分开。
一口气改三处再跑，你就不知道到底是哪一处起了作用。

**先问清搜索边界。**如果文件标了可疑区域，就问未标记代码是否也在范围内。
时间紧时不要擅自假定「所有注释都可信」，也不要擅自把整个仓库都重审一遍。

**说出你找到的是哪一类 bug。**「mask 加在了 softmax 之后，所以每行不再和为 1」
说明了机制和被破坏的不变量；「修好了」只描述了 diff。

> **一种扎实的准备方式**，是至少完整写过一次 nanoGPT 规模的模型，
> 从 embedding 表一路写到训练循环。Liu 明确建议把 transformer 实现与调试练成肌肉记忆；
> Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 是紧凑的参考。

---

<a id="b9-2"></a>
### B9.2 miniGPT 练习

这是根据轶事性面经综合出的 **OpenAI 风格**练习：一个小型 decoder-only LM，
埋了四个 bug，外加一个 KV cache 扩展。它不是公司的逐字原题。

[`练习文件`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d09_minigpt.py)
· [`三级提示`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d09_minigpt.md)
· [`测试`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d09_minigpt.py)
· [`runner`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py)

```bash
python run.py --drill d09                          # 35 分钟预算
```

植入的四类 bug 分别检查不同 invariant：

| bug | 你能测出来的症状 |
|---|---|
| 位置 embedding 用一个常数下标去取 | 每个 token 都拿到了位置下标零 |
| 因果 mask 加在了 softmax *之后* | 注意力每行不再和为 1 |
| 合并多头时没把时间维 transpose 回来 | 输出静默地错，不抛异常 |
| 训练循环从来没让优化器 step | 走一步之后参数没有变化 |

**第三个值得多看两眼**，因为 shape 检查抓不到它。注意力的输出是 `(B, n_heads, T, d_head)`，
你要的是 `(B, T, C)`。直接 reshape 是*能跑*的——元素个数对得上——然后把 head 和 time 交错在了一起。
什么都不报。模型会带着错误连接继续训练，而你手上没有任何异常可查。
这解释了为什么 `.transpose(1, 2).contiguous().view(...)` 要按这个顺序写，
也解释了带形状后缀的变量名（`y_BHTD`）为什么有用。

**追问是 KV cache。**新 token 的**位置下标是 cache 的长度**，不是零。
第 $$t$$ 步 decode 必须 embed 位置 $$t$$。写错后，生成会退化，而只做 teacher forcing
的评测仍可能正常。把不变量说出口——带 cache 的解码必须和完整重算数值吻合——然后去测它。

---

<a id="b9-3"></a>
### B9.3 GRPO 循环练习

这是根据轶事性面经综合出的 **Anthropic 风格**练习：一个完整的玩具 GRPO 循环，
两个故障是数值上的，一个是算法上的；同样不是官方题。

[`练习文件`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/drills/d10_grpo_loop.py)
· [`三级提示`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/d10_grpo_loop.md)
· [`测试`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_d10_grpo_loop.py)
· [`runner`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py)

```bash
python run.py --drill d10                          # 30 分钟预算
```

**平移后的 logits 被当作权重喂给 `torch.multinomial`。**植入代码先减最小值，
所以权重非负、脚本能跑，但这并不会把 logits 变成策略概率。这个函数收的是非负*权重*，
不是 logits。改法是先做 softmax。

**advantage 除了一个裸的标准差。**当一组里每条 completion 拿到的 reward 都一样时，
标准差是零、advantage 是 NaN，下一次更新就可能传播非有限梯度。只要一组全对或全错就会如此；
这个组也没有组内相对 reward 信号（第一篇 A9.5）。应使用
`std(correction=0) + 1e-5`：population correction 还能让 singleton group 保持有限。

**ratio 被算成了对数差。**importance ratio 是
$$\exp(\log \pi_\theta - \log \pi_{\text{old}})$$。拿这个差本身当 ratio 不是 ratio，
而且判据很锐利：on-policy 时新旧 log-prob 相等，ratio 必须正好是 1，
未截断的 surrogate 必须等于 advantage。对数差在那里给出零，
使第一次更新时的 surrogate 数值与 clipping 区域都处在错误状态。

**一个有用的追问**是讨论题，不是代码题：

> on-policy rollout 的第一次更新中，importance ratio 何时应严格等于 1？
> 为什么之后日志里的 minibatch 平均值可能偏离 1？

好的回答会点出好几个原因，并且对每一个都说清楚你会去查什么：

- **每个 rollout batch 做了不止一次优化器 step。**第一步之后策略已经动了，
  剩下的 mini-epoch 在构造上就是 off-policy 的。去查内层 epoch 数。
  即使合法 importance ratio 在完整分布下的期望为 1，有限 minibatch 的平均值也不必为 1。
- **采样引擎和训练引擎不是同一个。**vLLM 出的 rollout 和在 HF 里重算的 log-prob
  可能不会逐位相同——kernel 不同、attention 实现不同、精度不同。
  查法是对同一批 token 在两边各算一遍 log-prob，然后做差。
- **采样参数只在生成时生效、打分时没生效。**temperature、top-p 和 logit bias
  改变了你实际采样的那个分布。如果你用原始分布去打分，你的「old」log-prob 属于另一个策略。
- **精度与非确定性。**log-prob 在 fp32 还是 bf16 里累加、attention 用 fused 还是 eager，
  即便权重完全相同也会让 ratio 轻微偏移。

诊断目标是把**设计上就该有的漂移**和**真正的 bug** 分开。

---

<a id="b9-4"></a>
### B9.4 微练习

这些微型实现每段正好有一行是错的，只需几分钟，便宜到可以在等编译的间隙做完，
而且能直接复习上面的 bug 类别。

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

> **为什么微练习能补充从零实现。**从零写 attention 练的是构造，
> 找出一个反了的 mask 练的是故障识别。短练习提供更多重复，
> 长练习则检查这些局部判断能否组成完整调试策略。

---

<a id="section-b1"></a>

## B1 · NumPy 与 PyTorch 基础

本节从向量化 1-NN 和带状态的 BatchNorm 实现开始。
Datadog 标签记录的是作者本人的轶事性面试经历，不是官方题库或频率估计。
它排在前面的真正理由是后面的一切都建立在 transpose、broadcast 和 dtype 纪律上。

---

<a id="b1-1"></a>
### B1.1 向量化：唯一值得背下来的技巧

用 NumPy 写 1-最近邻，不许用循环。重点不是分类器，
而是能不能把所有成对距离表达成数组操作。

$$\|a - b\|^2 = \|a\|^2 - 2\,a \cdot b + \|b\|^2$$

平方展开能把交叉项变成一次快速 matmul，但在 float32 下，大而接近的项相减
可能把真正要比较的距离消掉。因此教学参考实现直接 broadcast 差值，再做平方。

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

**边写边要说出来的三件事。**全程不开根号，因为 `argmin` 对单调变换不变。
直接差值的 broadcast 形状是 `(n_test, n_train, d)`，再沿 feature 轴求和，
留下距离矩阵。最后，这个选择是数值上的而不只是代数上的：float32 坐标
`100.02`、`100.001` 和 query `100.0` 会让平方范数展开式相消到选错邻居。

> **扩展性检查是内存。**直接 broadcast 还会开出
> `n_test × n_train × d` 的临时量，所以实用的精确实现会按测试行分块；
> `scipy.spatial.distance.cdist` 也是经过验证的选择。展开式 matmul 的临时内存更少、
> 也可能更快，但对大而接近的 float32 坐标要明确采用 float64 累积等精度策略。

<!-- EXERCISE p24 -->
**练习** — [`p24` · 纯 NumPy 的 1-NN，不许循环](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p24_nn_vectorized.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p24_nn_vectorized.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p24_nn_vectorized.py) · [`python run.py p24`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 分钟 · 冷启动集
<!-- /EXERCISE -->

---

<a id="b1-2"></a>
### B1.2 BatchNorm，以及它为什么有两种模式

作者本人在一次 Datadog 面试中遇到过这道题；这只是个人轶事经历，
不是 Datadog 官方题库。它看着像热身题，其实不是——有意思的部分是状态，不是公式。
技术来源是 [Ioffe 与 Szegedy（2015）](https://arxiv.org/abs/1502.03167)。

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

这份面试规模实现约定 `x` 为 `(N, D)`。PyTorch 的 `BatchNorm1d` 还支持
`(N, C, L)`，此时统计量沿 `N` 与 `L` 聚合。

**四个语义细节。**

**训练和评估算的是两个不同的函数。**训练时统计量来自当前 batch，评估时来自滑动估计。
每个 channel 只有一个训练值时无法估计方差，评估模式则使用已经存下的统计量。
分布式训练还必须明确哪些 worker 共享统计量；LayerNorm 没有这两种依赖。

**用 `register_buffer`，不是 `nn.Parameter`。**滑动统计量跟着 `.to(device)` 走、会存进 state dict，
但不接收梯度。把它们写成 parameter 是语义错误，即使初始 forward 看起来正常。

**归一化用有偏，滑动估计用无偏。**PyTorch 用有偏方差（$$/n$$）做归一化，
累积的却是无偏的那个（$$/(n-1)$$）。不对齐这一点时，训练输出仍可能匹配，
但滑动状态、进而评估输出会偏离。测试必须覆盖两种模式。

**PyTorch 的定义把 epsilon 放在根号里面。**放在外面仍可避免除零，
但实现了不同的缩放，尤其在小方差时明显，也无法匹配参考层。

> **架构对比：**transformer 为什么通常用 LayerNorm 而不是 BatchNorm？
> LayerNorm 逐 token 计算，train/eval 统计一致，不耦合 batch 内样本，
> 也不需要同步跨设备 batch 统计量。
> 完整版在第一篇 A1.7。

<!-- EXERCISE p25 -->
**练习** — [`p25` · BatchNorm 前向、梯度与 eval 模式](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p25_batchnorm.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p25_batchnorm.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p25_batchnorm.py) · [`python run.py p25`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 分钟 · *作者个人轶事性面经：Datadog*
<!-- /EXERCISE -->

---

<a id="b1-3"></a>
### B1.3 会造成静默 bug 的张量语义

四种反复出现的语义值得显式测试。

**`view` 与 `reshape`。**`view` 要求内存连续，否则直接拒绝；`reshape` 会退化成复制。
`transpose` 之后你是非连续的，所以 `view` 会报错——这是*好*情况，因为它告诉了你。
坏情况是元素个数恰好对得上，一次 reshape 就悄悄把轴交错在一起，
这正是 miniGPT 练习里的第三个 bug（B9.2）。

**broadcast 从右往左对齐。**`(B, T, C) * (C,)` 可以，`(B, T, C) * (B,)`
通常不行；若恰好 `B == C`，还可能静默对齐到 `C`。想做按 batch 的缩放，
就写成 `(B, 1, 1)`；显式 singleton 维度让目标轴可以被检查。

**原地操作与 autograd。**对反向要用到的值做 `x += 1`，可能触发 leaf 或 version-counter
错误；`x = x + 1` 会创建新张量。对刻意不求导的状态（如优化器 buffer 与滑动统计量），
在正确的 gradient context 下才适合原地更新。

**dtype 提升是静默的。**bf16 乘 fp32 得到 fp32。归一化层就是这样悄悄返回了错误的 dtype（B2.5），
一次「bf16」训练也是这样在你没打算的地方留下了 fp32 激活。

**调试练习** —— B9.4 里的微练习 `d02`、`d06`、`d07` 打的正是这几个点。

---

<a id="section-b2"></a>

## B2 · Transformer 组件

这是可复用的基础部分：限时实现每个组件，再验证行为 invariant，
而不是只依赖 shape 检查。

这里所有代码都在 `interview-practice/reference.py` 里，并对照 PyTorch 原语
或明确的行为 invariant 做过检查。

---

<a id="b2-1"></a>
### B2.1 因果多头注意力

原始来源：[Vaswani 等，2017](https://arxiv.org/abs/1706.03762)。

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

在分量独立、单位方差的常用模型下，缩放让 logit 方差保持常数量级，
避免 softmax 仅因 head width 增长而饱和；mask 在 softmax **之前**
以加性 $$-\infty$$ 加入，使被屏蔽位置对分母毫无贡献。两条论证都在第一篇（A2.3）。

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

**四种故障模式与设计选择。**

1. **`.contiguous()`。**`transpose(1, 2)` 之后张量是 stride 不连续的 view，`.view()` 会报错。
   想用 `.reshape()` 也行，但要能说出区别：`view` 从不复制、所以直接拒绝；`reshape` 必要时退化成复制。
2. **除以 $$\sqrt{d_\text{model}}$$ 而不是 $$\sqrt{d_\text{head}}$$。**点积是在头的维度上做的，
   那才是你要修正方差的那一维。分母写错会静默改变 softmax 的有效温度。
3. **在 softmax 之后做 mask。**事后把被屏蔽位置置零，它们仍然留在分母里，
   剩下的权重不再和为 1，而且每一行的误差还不一样。
4. **一次融合投影还是三次独立投影。**二者数学等价；一次大投影通常更高效，
   但实测差异取决于 compiler 与硬件。

**写一个因果性检查。**先切到 eval mode，避免 dropout 让测试带随机性：

```python
model.eval()
y1 = model(x)
x2 = x.clone(); x2[:, -1, :] += 10.0
assert torch.allclose(y1[:, :-1], model(x2)[:, :-1])   # the past cannot see the future
```

> **要展示什么。**说明预期 shape，解释 transpose 后 stride 的变化，并数值测试因果性。

<!-- EXERCISE p01 -->
**练习** — [`p01` · 因果多头注意力](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p01_mha.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p01_mha.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p01_mha.py) · [`python run.py p01`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 分钟 · 冷启动集
<!-- /EXERCISE -->

---

<a id="b2-2"></a>
### B2.2 KV cache 与增量解码

decode 第 $$t$$ 步只有一个新 query，却需要全部历史 key/value。Q 是瞬时的，K/V 是累积的。
对单层生成长度为 $$T$$ 的序列时，每一步重算完整 prefix attention matrix，
累计 attention 工作是三次方量级；cache 把它降成二次方量级，也避免重复历史 K/V 投影。

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

这里的 `torch.cat` 让 contract 易读，却会重新分配并复制 cache。生产 serving 会预分配存储
或使用 paged block，而 grouped kernel 也会避免真的复制 K/V 头。

**mask 的偏移量就是这道题的全部。**Prefill 时 $$T = T_\text{full}$$，普通 `tril` 是对的。
但缓存解码时你的 query 块是从矩阵中间某一行开始的，所以需要 `diagonal=T_full - T`。
写错的后果是：teacher-forced 评测里一切正常，生成时悄悄退化——
如果评测只做 teacher forcing，就看不到它。

**要主动说出的正确性性质：**带 cache 的增量解码必须和完整重算**数值接近**；
浮点 kernel 的运算顺序可能阻止 bitwise identity。这是可测的，那就去测。

> **要展示什么。**把 cache 拼接和带偏移的 causal mask 当作两条独立 invariant；
> 只测前者会漏掉这个故障。

<!-- EXERCISE p02 -->
**练习** — [`p02` · KV cache 与增量解码](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p02_kv_cache.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p02_kv_cache.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p02_kv_cache.py) · [`python run.py p02`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 分钟 · 冷启动集
<!-- /EXERCISE -->

---

<a id="b2-3"></a>
### B2.3 MHA、MQA、GQA 与 MLA

作者本人在一次 Datadog 面试中遇到过 GQA。这只是个人轶事来源标签，
不是 Datadog 官方题库或频率断言。

四种变体由同一条设计轴连接：**每个 token 到底在 KV cache 里保留什么？**
普通 MHA 为每个 query 头保留独立的 key 和 value；MQA 让全部 query 头共享一组 K/V；
GQA 把 query 头分组，每组共享一组 K/V。若有 $$H$$ 个 query 头、$$H_{kv}$$ 个 KV 头，
头宽为 $$d_h$$，则每层每 token 的 cache 是 $$2H_{kv}d_h$$ 个数值。

```python
k = k.repeat_interleave(self.n_rep, dim=1)   # n_rep = n_heads // n_kv_heads
v = v.repeat_interleave(self.n_rep, dim=1)
```

$$H_{kv}=1$$ 是 MQA，$$H_{kv}=H$$ 是普通 MHA，中间就是
[GQA](https://arxiv.org/abs/2305.13245)。它是一个可调的质量—缓存旋钮。

**会绊倒人的追问：它并不减少注意力的 FLOPs。**K 和 V 在矩阵乘之前被扩展回完整头数，
所以 $$QK^\top$$ 和 $$AV$$ 一点没变。变小的是 cache 以及读它所需的带宽，
而 decode 往往受带宽限制，加速就来自那里。（被追问时要说准：
K/V 的**投影**确实变小了，每层从 $$2D^2$$ 降到
$$2D H_{kv} d_h$$ 个参数。）

<!-- EXERCISE p03 -->
**练习** — [`p03` · 分组查询注意力](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p03_gqa.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p03_gqa.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p03_gqa.py) · [`python run.py p03`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 10 分钟 · *作者个人轶事性面经：Datadog*
<!-- /EXERCISE -->

**[MLA](https://arxiv.org/abs/2405.04434) 改变的是被缓存的对象。**
它不保存展开后的内容 K/V，而是保存一个低秩潜变量
$$c_t=W^{DKV}x_t\in\mathbb R^r$$，需要时再从中重建内容 key 和 value。
RoPE 一般不能直接穿过这个低秩投影，所以 MLA 还单独保留一个很小的位置 key。
下面这份面试规模的实现只缓存 `c: (B,T,r)` 和 `k_rope: (B,1,T,d_rope)`：
每 token 是 $$r+d_\text{rope}$$ 个数值，而 MHA 是 $$2Hd_h$$。

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

它忠实实现了**缓存表示和 attention 语义**，但并不声称复刻 DeepSeek 的生产 kernel。
生产版 MLA 会在 decode 时把一部分上投影吸收到 query 路径里，从而避免显式构造展开后的 key；
这里为了可读性把它们重建出来，重点验证缓存压缩，把 kernel fusion 留在范围之外。

> **正确性不变量没有变化：**整段前向和逐 token 缓存解码必须数值一致。
> 如果压缩缓存改变了 logits，那是近似；MLA 本身不是近似。

<!-- EXERCISE p28 -->
**练习** — [`p28` · 多头潜在注意力与压缩缓存](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p28_mla.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p28_mla.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p28_mla.py) · [`python run.py p28`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 25 分钟
<!-- /EXERCISE -->

---

<a id="b2-4"></a>
### B2.4 旋转位置编码

原始来源：[Su 等，2021](https://arxiv.org/abs/2104.09864)。

RoPE 把每个坐标对按与位置成正比的角度旋转，使注意力 logit 只依赖相对偏移。
三行证明在第一篇（A2.6），这里重要的是实现。

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

**三个细节。**它只加在 **Q 和 K** 上，在拆头之后、点积之前，不加在 V 上。
配 KV cache 时缓存的是**旋转之后**的 key。
还有配对约定（`0::2, 1::2` 还是前后半分）必须在建表和应用两处一致，
否则 logits 会静默实现另一种位置变换。

<!-- EXERCISE p04 -->
**练习** — [`p04` · 旋转位置编码](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p04_rope.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p04_rope.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p04_rope.py) · [`python run.py p04`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 分钟 · 冷启动集
<!-- /EXERCISE -->

---

<a id="b2-5"></a>
### B2.5 RMSNorm

原始来源：[Zhang 与 Sennrich，2019](https://arxiv.org/abs/1910.07467)。

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

值得注意的是归约 dtype：低精度输入提升到 fp32，而 float64 保持 float64。
没有减均值也没有 bias——RMSNorm 论文报告其试验中只做重新缩放可以匹配
LayerNorm，而去掉重新中心化能省一次归约。

> **这是数值选择，不是风格问题。**在 bf16 里做归约的实现可以通过只含 fp32
> 的测试，却在 mixed precision 下累积大得多的舍入误差；反过来，无条件调用
> `x.float()` 会降级 float64。练习会同时检查这两个方向，包括平方后量级约为
> $$10^4$$ 的 bf16 输入。

<!-- EXERCISE p05 -->
**练习** — [`p05` · RMSNorm](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p05_rmsnorm.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p05_rmsnorm.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p05_rmsnorm.py) · [`python run.py p05`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 5 分钟 · 冷启动集
<!-- /EXERCISE -->

---

<a id="b2-6"></a>
### B2.6 SwiGLU

原始来源：[Shazeer，2020](https://arxiv.org/abs/2002.05202)。

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

**三个矩阵，不是两个。**取 $$F \approx \tfrac83 D$$，会让矩阵参数量近似匹配
无 bias 的普通 $$4D$$ FFN。这个参考实现有整数截断，所以只是近似；
生产实现通常还会把宽度取整到硬件友好的倍数。

<!-- EXERCISE p06 -->
**练习** — [`p06` · SwiGLU 前馈层](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p06_swiglu.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p06_swiglu.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p06_swiglu.py) · [`python run.py p06`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 5 分钟
<!-- /EXERCISE -->

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
        x = x + self.attn(self.norm1(x))     # pre-norm: residual stays an identity path
        x = x + self.mlp(self.norm2(x))
        return x
```

forward 只有两行。Pre-norm 归一化的是子层的**输入**，留下干净的恒等通路，
从而改善深层堆叠中的梯度流。残差流的幅度仍可能随深度增长，
所以标准完整模型会在 `lm_head` 前放一个 **final norm**。
Pre-norm 能降低优化脆弱性，但本身不保证可以不用 warmup。

<!-- EXERCISE p07 -->
**练习** — [`p07` · 完整的 pre-norm block](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p07_transformer_block.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p07_transformer_block.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p07_transformer_block.py) · [`python run.py p07`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 分钟
<!-- /EXERCISE -->

---

<a id="section-b3"></a>

## B3 · 训练循环

训练循环经常是某个组件外面的 harness，也可能直接成为调试对象。
这套题的四个 miniGPT bug 里有一个就住在这里，所以即使它不是主实现题，也值得练到会审计。

---

<a id="b3-1"></a>
### B3.1 Cross entropy，以及它为什么收 logits

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

**这个 API 为什么收 logits 而不是概率。**手写
`log(exp(x) / exp(x).sum())` 会在大 logit 上溢出；即使 softmax 本身稳定，
小概率也可能在后续取 log 前下溢。在 log 空间里先减每行最大值，就不必物化概率。
logits 量级在 $$10^4$$ 附近时，参考实现仍保持有限，并在浮点容差内匹配 `F.cross_entropy`。

**`ignore_index` 的归约：**分母必须是*留下来*的 token 数，不是 $$N$$。
先 mask、再对全部求平均，等于悄悄把 loss 乘上了保留比例，而这个比例又会和你的学习率纠缠在一起。

**还有一种全部被 mask 掉的情况。**packing 之后的某个 microbatch 可能整批都被 mask，
若没有开头的 guard，mean reduction 会返回 NaN（`F.cross_entropy` 也一样），
并可能通过非有限梯度污染下一次更新。应该返回一个仍然挂在计算图上的零。

<!-- EXERCISE p08 -->
**练习** — [`p08` · 交叉熵与 log-sum-exp](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p08_cross_entropy.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p08_cross_entropy.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p08_cross_entropy.py) · [`python run.py p08`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 10 分钟 · 冷启动集
<!-- /EXERCISE -->

---

<a id="b3-2"></a>
### B3.2 Loss masking 与 packing

两件看着像管道活、其实是正确性的事。

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

**白板上要躲开的那个手滑：**在 `(B, T)` 张量上写 `labels[:len(prompt_ids)] = -100`，
切的是 **batch** 维，整条整条地抹掉前几个样本，而不是屏蔽掉每条样本的 prompt。
它能跑，能训，而且是错的。

**Packing** 把多条短样本拼成一条定长序列，避开在长度差异很大的 batch 里占主导的 padding 浪费。
代价是 token 现在能跨文档边界做 attention。要么用 varlen kernel（配 `cu_seqlens` 的
FlashAttention），要么用块对角 mask；另外 `position_ids` 要按文档重置，
否则第二篇文档是从位置 512 开始的。

Reference 把这份契约写成了可执行代码。`response_mask` 由预处理生成，只标 target token；
`segment_ids` 标出每个位置所属的连续文档 run；`build_packed_attention`
返回重置后的 position 和一个 `(B, T, T)` 布尔 mask，它同时满足 causal 与 block diagonal。
Padding 既没有可见的行，也没有可见的列。生产 kernel 会紧凑地编码同一组边界，
而不是显式物化这张教学用 mask。后面再次使用同一个数值 segment ID，仍会开始一个新 run。
这些 label helper 还假设 causal LM 在内部移动 logits 与 labels；手写 loss 时，
要显式让 token $$t$$ 的 logits 对齐 token $$t+1$$ 的 label。

<!-- EXERCISE p09 -->
**练习** — [`p09` · SFT loss masking 与 packing](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p09_loss_masking.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p09_loss_masking.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p09_loss_masking.py) · [`python run.py p09`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 分钟
<!-- /EXERCISE -->

---

<a id="b3-3"></a>
### B3.3 干别的之前，先过拟合十条样本

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

**把记忆小样本当成受控诊断。**模型容量足够、batch 固定且优化器设置合理时，
loss 仍无法逼近零，就应检查数据、目标函数、梯度路径、更新步骤与模型状态。
它排除了大部分数据和容量的不确定性，适合在完整训练前做便宜而高信号的检查。

**这三行的顺序有语义。**`zero_grad` → `backward` → `step`。
梯度默认是*累加*的，所以跳过 `zero_grad` 就是把每一步的梯度全叠在一起；
跳过 `step` 就什么都不更新，loss 曲线一条平线；在 `backward` 之前调 `step`，
要么没有新梯度可用，要么消费上一轮残留的梯度。

> **梯度为什么会累加：**同一参数可能沿计算图的多条路径收到贡献，跨 micro-batch
> 的梯度累积也依赖这个语义。代价是训练循环必须显式决定何时清零。

<!-- EXERCISE p10 -->
**练习** — [`p10` · 把一个小 batch 过拟合](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p10_training_loop.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p10_training_loop.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p10_training_loop.py) · [`python run.py p10`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 分钟 · 冷启动集
<!-- /EXERCISE -->

---

<a id="b3-4"></a>
### B3.4 过滤坏标注

它主要不是一道建模题——它考的是能不能在不过度设计的前提下，把标签噪声想清楚。

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

**关键的统计 guard 是 `min_items`。**一个只标了两条、两条都和多数不一致的标注员，
看起来糟透了，但两个样本是噪声，不是证据。没有这个下限，你会把每一个标得少的标注员都标出来，
而证据并不充分。

**这是教学用 heuristic，不是生产 estimator。**拿包含标注员自己投票的多数结果给他打分，
存在循环；leave-one-annotator-out agreement 或 gold item 能减少这种泄漏。
如果坏标注员形成多数，consensus 本身就是错的。不一致也不等于错误：
有歧义条目会把条目难度和标注员质量混淆，Dawid–Skene 类模型会联合估计两者。
一组错误系统性相关的标注员也可能穿过多数投票。

<!-- EXERCISE p26 -->
**练习** — [`p26` · 过滤劣质人工标注](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p26_data_filtering.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p26_data_filtering.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p26_data_filtering.py) · [`python run.py p26`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 分钟
<!-- /EXERCISE -->

---

<a id="section-b4"></a>

## B4 · 手推 backward

真正有用的挑战不是「你还记不记得 chain rule」——而是应用它时能不能跟住形状，
以及能不能解释框架替你做了什么。

---

<a id="b4-1"></a>
### B4.1 检查每个 backward 的约束

先用局部导数和 chain rule，再施加一条不可省略的检查：

> **对任何张量的梯度都和那个张量同形**，并由传入梯度与局部操作数构造。

形状本身不能证明导数，却能排除许多错误转置与归约。微分给出数值，
形状约束合法的缩并方式。

对 $$Z = XW + b$$，其中 $$X: (m, n_\text{in})$$、$$W: (n_\text{in}, n_\text{out})$$：

$$\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Z}W^\top,\qquad
\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Z},\qquad
\frac{\partial L}{\partial b}=\sum_i \frac{\partial L}{\partial z_{i}}$$

用形状验一遍：$$(m, n_\text{out}) \times (n_\text{out}, n_\text{in})$$ 给出 $$X$$ 的形状，
$$(n_\text{in}, m) \times (m, n_\text{out})$$ 给出 $$W$$ 的。bias 的梯度要在 batch 维上求和，
因为 forward 里的广播对应 backward 沿被广播轴求和。

**dense backward 为什么大约是 dense forward 的两倍。**线性层 backward 要算两个相近大小的乘积：
输入梯度和权重梯度。把一次 multiply-add 记作两个 FLOP，
常用的 core-model 估算就是每 token 的 forward 加 backward 共
$$2N + 4N = 6N$$ FLOPs（第一篇 A10.0）；这还没算 optimizer、重计算、稀疏性和非 matmul 操作。

---

<a id="b4-2"></a>
### B4.2 最小标量 autograd

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

这是刻意缩小的标量教学引擎：没有 tensor broadcasting、梯度清零 API、
graph retention 或高阶导数。

**两件事撑起整个答案。**

**用 `+=` 而不是 `=`。**一个被用在两处的节点会从两条路径各收到一份梯度，
多元 chain rule 说它们要相加。赋值会悄悄只留下最后那一份——而在每个节点只被用一次的图上，
这个 bug 完全看不见，偏偏那正是你会拿来测试的那种图。

**逆拓扑序。**一个节点的 backward 只有在它所有的消费者都贡献完之后才能跑。
遍历时一遇到节点就执行 callback，会在菱形图上失败；代码只用 DFS 构造 postorder，
再把这个列表反转。

> **追问：PyTorch 为什么动态建图？**因为那张图就是「实际跑过的那些算子」，边跑边记下来——
> 所以控制流、循环和依赖数据的形状能 eager 执行。`torch.compile` 会捕获并优化带 guard
> 的执行区域；guard 改变时可能 graph break 或重新编译。

<!-- EXERCISE p11 -->
**练习** — [`p11` · 最小标量 autograd](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p11_autograd.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p11_autograd.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p11_autograd.py) · [`python run.py p11`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 30 分钟
<!-- /EXERCISE -->

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

**被 mask 的位置会怎么样。**它们的 $$p = 0$$，所以 `p * (...)` 自动把它们的梯度清零。
只要每个 query 至少有一个有效 key，这个 backward 就不需要再施加一次 mask。

**为什么它在面试之外也重要：**FlashAttention 在 tile 上使用同一套 backward 代数，
从 $$Q$$ 与 $$K$$ 在片上重新生成 $$P$$，而不是把完整 attention matrix 存进 HBM（B6.2）。

<!-- EXERCISE p12 -->
**练习** — [`p12` · 手写 attention 反向](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p12_attention_backward.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p12_attention_backward.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p12_attention_backward.py) · [`python run.py p12`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 25 分钟
<!-- /EXERCISE -->

**MLP 是同一种方法，只是轴更少。**要缓存激活前的 `h`，因为 ReLU 的导数取决于它的符号；
bias 在 forward 中沿 batch 广播，所以它的梯度要沿 batch 求和；每个返回梯度都应与原张量同形。
这份紧凑实现约定 `x` 与 `d_y` 是二维 `(N, D)` 张量。

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

真正有用的自检不是手算一个例子，而是生成随机 float64 张量，给两种实现喂同一个随机上游梯度，
再把五个梯度全部与 `torch.autograd` 比较。转置写错或漏掉 bias 归约都会被抓出来。

<!-- EXERCISE p13 -->
**练习** — [`p13` · 手写 MLP 反向](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p13_mlp_backward.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p13_mlp_backward.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p13_mlp_backward.py) · [`python run.py p13`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 分钟
<!-- /EXERCISE -->

---

<a id="section-b5"></a>

## B5 · 推理与采样

短的一节，两道题，而且两道都比看上去微妙。

---

<a id="b5-1"></a>
### B5.1 Temperature、top-k、top-p

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

**顺序有讲究：先 temperature，再 top-k，最后 top-p。**temperature 改变的是截断所作用的那个分布，
所以把它放到最后，等于是在一个错误的分布上挑 nucleus。

**top-p 的 off-by-one。**你要的是*累计质量达到 p 的最短前缀*，
也就是说跨过阈值的那个 token 要**留下**。`cum - probs` 是独占前缀和——
严格排在这个 token 之前的那部分质量——在它已经超过 `p` 的地方丢弃，才是对的。
写成 `cum >= top_p` 就把跨阈值的那个丢掉了：在 `[0.5, 0.3, 0.15, 0.05]` 这样的分布上取
`p = 0.9`，你会不声不响地从两个 token 里采样，而不是三个。

**`top_p >= 1` 是 no-op。**此时应直接跳过 nucleus filtering，而不是依赖累计和：
面对极端但有限的 logits，浮点累计质量可能提前舍入到一，误删原本有限 logit 的 support。

**`temperature == 0` 需要一个显式分支**，否则你在除零。

**Top-k 应当严格保留 k 个下标。**如果只是把小于第 k 大数值的项 mask 掉，
边界上有相同 logit 时会留下超过 k 个 token。用 `topk` 取下标再 `scatter`
可以显式处理 tie，并守住 support 大小的契约。

**top-p 的自适应 support：**概率质量集中时 nucleus 较小，分散时就变宽；
top-k 则固定 support 大小。二者没有普遍支配关系，应该比较应用能接受哪种失效模式。

<!-- EXERCISE p14 -->
**练习** — [`p14` · temperature / top-k / top-p](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p14_sampling.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p14_sampling.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p14_sampling.py) · [`python run.py p14`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 分钟 · 冷启动集
<!-- /EXERCISE -->

---

<a id="b5-2"></a>
### B5.2 Speculative decoding

原始来源：[Leviathan 等（2022）](https://arxiv.org/abs/2211.17192)和
[Chen 等（2023）](https://arxiv.org/abs/2302.01318)。

有意思的性质是它**精确**——在下面 acceptance-and-correction 假设下，
它不近似目标模型分布，而是复现那个分布。

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

这个函数只是单个位置的 correction kernel。完整算法让 draft model 自回归提出
$$K$$ 个 token，target 用一次 forward 给这些位置打分，再从左到右走到第一次拒绝。
在位置 $$i$$，$$p_i$$ 与 $$q_i$$ 都以此前已接受 prefix 为条件。
一旦拒绝，就从 residual 发出一个 token 并丢掉剩余 draft suffix；
若全部接受，target 还会多发一个 token。

这个 kernel 假设 `p_target` 与 `q_draft` 已归一化、草稿 token 的
`q_draft[token]` 为正，且 `u` 在 `[0, 1)` 上均匀。实现会在除法前检查 draft mass
为正，并使用严格事件 `u < p/q`；若 `p[token] = 0` 且 `u = 0`，写成 `<=`
会错误地发出一个 target 概率为零的 token。

以概率 $$\min(1, p(x)/q(x))$$ 接受草稿模型给的 token；拒绝时，
从归一化后的残差 $$\propto \max(0, p - q)$$ 里采一个。这就是拒绝采样，
得到的样本可证明服从 $$p$$。

**被问到就一行证给他看。**吐出 $$x$$ 的概率是
$$q(x)\min(1, p/q) + P(\text{reject})\cdot\frac{\max(0, p-q)}{\sum_y \max(0, p-q)}$$。
第一项是 $$\min(q, p)$$，第二项恰好补上缺掉的那份 $$\max(0, p-q)$$，加起来正是 $$p(x)$$。
normalizer 成立是因为归一化的 $$p$$ 与 $$q$$ 满足
$$\sum (p-q)_+ = \sum (q-p)_+ = P(\text{reject})$$。

> **值得写的那个测试。**采 20 万次，把经验分布和目标分布比一比。
> 有限 Monte Carlo 不能证明精确性，但可以对照推导出的目标检查实现；参考实现做了这一步。

**加速从哪来，又从哪没的。**小 batch decode 常受带宽限制、FLOPs 闲着，
所以一次并行 forward 验证 $$k$$ 个草稿 token，可能比串行跑 $$k$$ 次 target 便宜得多。
batch 变大后，target 验证更可能受算力限制，计入接受率后的加速可能缩小甚至转负。
延迟和吞吐收益都取决于 draft、接受率、batching 策略、序列长度与硬件；
应在实际 serving regime 下测量。

<!-- EXERCISE p15 -->
**练习** — [`p15` · 投机解码的接受/拒绝](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p15_speculative.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p15_speculative.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p15_speculative.py) · [`python run.py p15`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 分钟
<!-- /EXERCISE -->

---

<a id="section-b6"></a>

## B6 · 高效实现

两道题共享同一个递推：softmax normalization 可以逐块更新。
IO-aware kernel 再把它与 tiling、fusion 结合，避免存下完整 attention matrix。

---

<a id="b6-1"></a>
### B6.1 流式 softmax

softmax 看上去必须先扫完一整遍才能归一化任何东西——你需要最大值来保稳定，需要和来当分母。
其实不必。维护一个滚动的最大值 $$m$$、一个滚动的分母 $$\ell$$、一个滚动的分子，
每当新的一块暴露出更大的最大值时，重新缩放一次。

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

**修正因子就是这个算法的全部。**此前累积的一切都是相对旧的最大值算出来的；
乘上 $$e^{m_\text{old}-m_\text{new}}$$ 就把它改写成相对新最大值的表达。
$$\ell$$ 和累加器两者都要乘；漏掉累加器很适合做 failure test——分母对了、分子没对，
产出一个看起来很合理但错的结果。

**它在代数上精确**，不是近似。浮点分块顺序仍可能改变末尾 bit，
所以应按 dtype 容差与朴素 softmax 比对；参考实现就是这么做的。

> **来源：**这个递推来自
> [Milakov & Gimelshein（2018）](https://arxiv.org/abs/1805.02867)，早于 FlashAttention。
> FlashAttention 的贡献不是这个递推，而是搭在它上面的 IO 感知 tiling。

<!-- EXERCISE p16 -->
**练习** — [`p16` · 流式 softmax](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p16_online_softmax.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p16_online_softmax.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p16_online_softmax.py) · [`python run.py p16`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 分钟
<!-- /EXERCISE -->

---

<a id="b6-2"></a>
### B6.2 分块的 FlashAttention forward

现在把 $$V$$ 也放进循环里跑这个递推，query 块和 key 块两个方向都做 tiling，
就得到了 [FlashAttention](https://arxiv.org/abs/2205.14135) forward 的结构。

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

这份可读 contract 覆盖 `q`、`k`、`v` 形状相同且没有 padding mask 的 dense self-attention。
输入为 fp16 或 bf16 时，它用 fp32 计算 block logits 并维护 `m`、`l` 与 `acc`，
最后把 attention 输出 cast 回输入 dtype；逐行 log-sum-exp 保持 fp32。
生产 kernel 会在融合操作内部采用同样的累积策略。

**关于它要说三件事。**

**固定 head width 时，辅助显存从 $$O(N^2)$$ 降到 $$O(N)$$**，因为分数矩阵没有被显式造出。
这里的 Python 实现表达了 tiling，却不能控制内存层级；融合 GPU kernel 才会把活跃 tile 留在片上。

**FLOPs 是*涨*的，不是降的。**backward 在片上重算注意力矩阵，而不是去读存好的
$$N\times N$$ 矩阵。
FlashAttention 减少的是内存流量，不是精确 dense attention 的算术量。

**当算子受 HBM 流量而非算术限制时，它仍可能更快。**拿 FLOPs 换访存量，
在 roofline 的访存受限一侧是赚的；实际加速取决于 shape、精度、kernel 和硬件。

**值得一提的因果优化：**有因果 mask 时，完全落在对角线以上的块可以整块跳过，
完全落在下方的块不需要 mask，只有跨过因果边界的块需要逐元素 mask。
长方阵序列的 score-tile 工作量因此接近减半，但端到端加速更小，也依赖 kernel。

<!-- EXERCISE p17 -->
**练习** — [`p17` · 分块 FlashAttention 前向](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p17_flash_attention.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p17_flash_attention.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p17_flash_attention.py) · [`python run.py p17`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 25 分钟
<!-- /EXERCISE -->

---

<a id="section-b7"></a>

## B7 · 后训练算法

B9.3 提供了明确标为 Anthropic 风格的 GRPO 调试练习。
目标函数仍要先从零写一遍——没亲手拼过的目标函数，很难可靠地调。

---

<a id="b7-1"></a>
### B7.1 LoRA

原始来源：[Hu 等，2021](https://arxiv.org/abs/2106.09685)。

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

**它在验两条性质。**在这个参数化下，初始化时保持原函数要求 `B = 0`；
两个 factor 都随机初始化会改变起始函数。以及**无损合并**：
改造后的层就是一个权重矩阵，合并后不再有额外的 LoRA 矩阵乘，不像 adapter 层那样加深度。
如果 LoRA dropout 非零，这个等价关系指 eval-mode 路径；训练路径仍有随机性。

**显存省在哪**——不是省在 base 权重上，它还是得常驻。在一种常见 mixed-precision AdamW
核算里，bf16 权重与梯度、fp32 master weight、两个 fp32 moment 合计每参数约 16 字节。
base 冻住后只剩 2 字节 bf16 权重，其余状态只作用在 adapter 上。70B base 相关 footprint
就从约 1,120 GB 降到 140 GB，再加 adapter 状态。
这个粗算不包括 activation、临时 buffer、量化和分布式 sharding。

<!-- EXERCISE p18 -->
**练习** — [`p18` · LoRA 与无损合并](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p18_lora.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p18_lora.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p18_lora.py) · [`python run.py p18`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 10 分钟 · 冷启动集
<!-- /EXERCISE -->

---

<a id="b7-2"></a>
### B7.2 GRPO 目标函数

原始来源：[DeepSeekMath](https://arxiv.org/abs/2402.03300)。

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

**实现细节。**B9.3 的 Anthropic 风格练习覆盖这里的前两项，
再加一项从 logits 采样；KL 与 credit assignment 是概念追问。

**分母里那个 epsilon 不是装饰。**一组里每条 completion 拿到同样 reward 时，标准差为零；
策略把一个 prompt 的样本全做对或全做错时就会发生。没有 epsilon 会得到 NaN。
这里用总体 `std(correction=0)`：singleton group 的 spread 与相对 advantage 都为零，
而不是在加 epsilon 之前先产生 NaN。

**ratio 是对数差的 `exp`。**on-policy 时新旧 log-prob 相等，ratio 必须正好是 1，
未截断的 surrogate 必须等于 advantage。裸的对数差在那里给出零——
让第一次更新时的 surrogate 数值与 clipping 区域都错掉。

**这个公式把 KL 写成 loss 里的逐 token 项**，而不是折进 shaped reward，
并使用 Schulman 的 k3 估计量：
取 $$r = \pi_\text{ref}/\pi_\theta$$、样本来自 $$\pi_\theta$$，则
$$\widehat{\mathrm{KL}} = r - \log r - 1$$。它在期望上无偏，而且逐样本非负，
而朴素的 $$-\log r$$ 在单个样本上可能是负的，尽管它的期望仍是 KL。

**advantage 是 bandit 形状的。**每条 completion 一个标量，广播到每一个 token——
没有 reward 导出的逐 token credit assignment。

**保留原始 GRPO 的 sequence-level reduction。**参考实现先平均每条 completion 的有效
token，再平均至少有一个有效 token 的 completion。DAPO 的 global-token reduction
是另一种刻意选择的变体：它用整个 batch 的有效 response token 总数作分母。
这并非天然错误，但会改变长度权重，让长 completion 比在原始 sequence-level mean
下拥有更大影响。

<!-- EXERCISE p19 -->
**练习** — [`p19` · GRPO 目标](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p19_grpo_loss.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p19_grpo_loss.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p19_grpo_loss.py) · [`python run.py p19`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 分钟 · 冷启动集
<!-- /EXERCISE -->

---

<a id="b7-3"></a>
### B7.3 DPO

原始来源：[Rafailov 等，2023](https://arxiv.org/abs/2305.18290)。

```python
def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """All args are summed sequence log-probs, shape (B,). No rollouts, no reward model."""
    pi_logratio = pi_chosen - pi_rejected
    ref_logratio = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()
```

四个 log-prob，一个 sigmoid。训练循环不需要 reward model、critic 或在线生成。
若冻结的 reference model 常驻，它大约多占一份 model-weight footprint；
预计算 reference log-prob 可以用存储换掉这部分显存，但灵活性也更低。

**sanity check：**如果 policy 从完全相同的 reference 初始化，margin 为零，
loss 是 $$\log 2 \approx 0.693$$。若不匹配，应检查 masking、sequence log-prob 聚合，
以及 reference/policy 是否已漂移。只累加 response token；prompt 与 padding 不属于 preference margin。

**它换掉了什么。**它是 off-policy 的——学的是在一个策略正在漂离的分布上收集来的偏好——
而且它挡不住 *likelihood displacement*：margin 变大是因为被拒答案的概率在掉，
而不是被选答案的概率在涨，有时候两个一起往下走
（[Raz 等，2024](https://arxiv.org/abs/2410.08847)）。

<!-- EXERCISE p20 -->
**练习** — [`p20` · DPO 损失](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p20_dpo_loss.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p20_dpo_loss.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p20_dpo_loss.py) · [`python run.py p20`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 分钟
<!-- /EXERCISE -->

---

<a id="b7-4"></a>
### B7.4 GAE

原始来源：[Schulman 等，2015](https://arxiv.org/abs/1506.02438)。

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

$$\lambda = 0$$ 退化成单步 TD；$$\lambda = 1$$ 给出完整 discounted residual sum，
轨迹真正 terminal 时等价于 Monte Carlo 风格 return 减 baseline；
若只是 time-limit truncation，仍会从 `last_value` bootstrap。把两个极限都写成正确性检查。

**循环是倒着跑的**，因为 $$\hat A_t$$ 依赖 $$\hat A_{t+1}$$。
正着写会使用尚不可得的 future accumulator。这份紧凑函数只处理一条不间断 trajectory：
真正 terminal 时传 `last_value=0`，truncation 时传 critic bootstrap；batched episode 还需要 done mask。

<!-- EXERCISE p21 -->
**练习** — [`p21` · 广义优势估计（GAE）](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p21_gae.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p21_gae.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p21_gae.py) · [`python run.py p21`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 15 分钟
<!-- /EXERCISE -->

---

<a id="section-b8"></a>

## B8 · 数据与 tokenization

两个紧凑实现。BPE 练确定性 tokenization 和 merge replay；
MoE routing 练稀疏派发、容量控制与加权累加。

---

<a id="b8-1"></a>
### B8.1 Byte-pair encoding

原始来源：[Sennrich 等，2015](https://arxiv.org/abs/1508.07909)。

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

这是最小的 byte-stream BPE。生产 tokenizer 通常还会先做 pre-tokenization、
禁止跨指定边界 merge、保留 special token，并定义明确的 tie-breaking；
这些策略包在同一个「学习并重放 merge」核心之外。

**一个关键 bug 源头：**编码时施加 merge 的顺序是它们被**学到**的顺序，
不是它们在待编码字符串里的频率顺序。Python 的 dict 保序，所以直接遍历 `merges` 是对的——
但只要塞进 `set`、排一遍序或重建 dict，token ID 就不再匹配训练模型时的 tokenizer。
解码仍可能恢复原始 bytes，所以单测 round-trip 还抓不到；还要保留 golden token-ID 测试。

**为什么用字节而不是字符。**字节级词表能表示任意 byte sequence，所以没有字节级 OOV。
非拉丁字符从更多 UTF-8 字节起步；最终是否占更多 token，取决于有限的 merge 预算如何分配给各脚本。

**解码表值得顺手写出来**，哪怕没人要求，因为你就是靠它测 round-trip 的：

```python
table = {i: bytes([i]) for i in range(256)}
for (a, b), new in merges.items():
    table[new] = table[a] + table[b]
assert b"".join(table[i] for i in ids).decode("utf-8") == text
```

**一个有用的追问：**字符计数为什么容易不稳？模型收到的是 token ID，
不是显式字符序列，一个 token 可能包含多个字母。模型可以从统计中学到拼写信息，
但表示层没有内置精确的字符访问。

<!-- EXERCISE p22 -->
**练习** — [`p22` · Byte-pair encoding](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p22_bpe.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p22_bpe.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p22_bpe.py) · [`python run.py p22`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 分钟 · 冷启动集
<!-- /EXERCISE -->

---

<a id="b8-2"></a>
### B8.2 带 capacity 的 top-1 MoE routing

这种 routing 风格的原始来源：
[Switch Transformer](https://arxiv.org/abs/2101.03961)。

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

**token dropping 是容易漏掉的那部分。**在这个 Switch 风格策略里，每个 expert 有 capacity；
溢出的 token **跳过 expert 计算**，顺着残差流继续。其他 MoE 系统也可能重路由或完全不丢 token。
一旦采用 dropping，同一个输入会因为 batch 里还有什么而产出不同输出。

**auxiliary loss 存在的理由不是 router 没有梯度**——它有梯度。gate 概率乘在所选 expert
的输出上，所以 LM loss 会反传进 router；不可导的只有 top-$$k$$ 这个*选择*动作。
真正的问题是这个梯度会自我强化：拿到更多 token 的 expert 训得更快，于是 router 更偏爱它们，
路由可能随之坍缩。

**这个 loss 为什么是 $$E\sum_e f_e p_e$$：**$$f$$ 不可导（它在数分配次数），$$p$$ 可导，
所以梯度沿 $$p$$ 流动、并以实际观测到的负载加权。assignment 与平均概率都均衡时它等于 1，
而有置信度的坍缩 router 会被惩罚。它仍是有退化情形的 surrogate：
完全均匀 logits 也得 1，尽管 deterministic `argmax` 的 tie-breaking 会把 token 都送到同一 index。

<!-- EXERCISE p23 -->
**练习** — [`p23` · 带容量的 top-1 MoE 路由](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p23_moe_routing.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p23_moe_routing.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p23_moe_routing.py) · [`python run.py p23`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 分钟
<!-- /EXERCISE -->

---

<a id="section-c1"></a>

## C1 · 概率：五种可复用套路

Alisa Liu 公开的数学笔记为分布、期望、不等式和极限定理提供了来源地图。
本节挑出五种可复用套路，把每一种都写成自包含的「识别特征→推导→例题→陷阱→闭卷自测」。
这是一套复习脚手架，不是对面试频率的断言。

---

<a id="c1-1"></a>
### C1.1 首步分析

**识别特征：**一个会不断重复的过程，问你某件事发生之前的期望时间。

对第一步做条件，把未知的期望用它们自己写出来。连续掷出两次正面所需的期望次数：

$$E_0=\underbrace{\tfrac 12 (1+E_0)}_\text{tails, no progress}+\underbrace{\tfrac 12(1+E_1)}_\text{heads},
\qquad E_1=\underbrace{\tfrac 12(1+E_0)}_\text{tails, start over}+\underbrace{\tfrac 12\cdot 1}_\text{heads, done}$$

两个方程、两个未知数，$$E_0 = 6$$。

**真正让它成立的是状态选对了。**这里的状态是「我朝 HH 走了多远」，只有三个取值
（还没有、已有一个 H、完成）。状态选错，方程就闭不上。全部技巧都在这个选择上；
代数部分毫无难度。

**同一套路，换更难的外壳：**赌徒破产（状态 = 当前赌本）、随机游走的返回时间、
马尔可夫链击中某个集合的期望步数。

<details><summary><strong>自测 · 先推导再展开</strong></summary>
公平硬币第一次出现 `HTH` 之前，期望要掷多少次？用「已经匹配的最长前缀」作状态。
方程为 `E0 = 1 + (E0 + E1)/2`、`E1 = 1 + (E1 + E2)/2`、
`E2 = 1 + E0/2`；解得 `E0 = 10`。
</details>

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

<details><summary><strong>自测 · 允许相依</strong></summary>
长度为 `n` 的公平随机 bit 串里，相邻且相等的 pair 期望有多少个？
共有 `n - 1` 个指示变量，每个为真的概率是 `1/2`，所以答案是 `(n - 1) / 2`。
相邻指示变量不必独立。
</details>

---

<a id="c1-3"></a>
### C1.3 $$n$$ 个变量的最大值与最小值——走 CDF

**识别特征：**任何关于若干次抽样中最大或最小者的问题。

对 iid 变量先走 CDF。「最大值不超过 $$x$$」就是「它们全都不超过 $$x$$」，
独立性再把交集概率变成乘积：

$$F_M(x) = P(\max_i X_i \le x) = [F_X(x)]^n$$

最小值取补：$$P(\min > x) = [1 - F_X(x)]^n$$。真的需要密度，最后再求导。

**值得背下来：**$$[0,1]$$ 上 $$n$$ 个 iid 均匀分布，$$\mathbb E[\max] = n/(n+1)$$、
$$\mathbb E[\min] = 1/(n+1)$$——两者之间的对称性，是这类题任何答案的一个好自检。

<details><summary><strong>自测 · 先写 CDF</strong></summary>
三个独立 `Uniform(0, 1)` 样本的最大值，在 `[0, 1]` 上的 CDF 是 `x^3`。
对密度 `3x^2` 乘 `x` 积分，或对 `1 - x^3` 积分，都得到期望最大值 `3/4`。
</details>

---

<a id="c1-4"></a>
### C1.4 把对称性当证明工具

**识别特征：**答案感觉上就不该依赖某个东西。

会碰到的两个例子。随机置换里，元素 $$i$$ 落在位置 $$j$$ 的概率对每一对都是 $$1/n$$
——C1.2 能成立靠的就是这条。秘书问题里，最好的候选人出现在任意给定位置上的概率是均匀的，
所以整个分析只跟最大值*落在哪里*有关。

**Monty Hall 是反例。**主持人的选择*不*对称——他从不打开有奖的那扇门——
而 2/3 恰恰就是从这个不对称里来的。你要搬对称性，
就说清楚问题在哪个变换下不变；说不出那个变换，你就是在猜。

<details><summary><strong>自测 · 说出变换</strong></summary>
均匀随机置换里，元素 `a` 出现在 `b` 前面的概率是多少？答案是 `1/2`：
交换 `a`、`b` 的位置，会把一种顺序中的每个排列与另一种顺序中的唯一排列配对。
</details>

---

<a id="c1-5"></a>
### C1.5 该抓哪个不等式

| 你手上有 | 用 | 给出 |
|---|---|---|
| 只有均值，$$X \ge 0$$ | 马尔可夫 | $$P(X \ge a) \le \mathbb E[X]/a$$ |
| 均值和方差 | 切比雪夫 | $$P(\lvert X-\mu\rvert \ge k\sigma) \le 1/k^2$$ |
| 有界的独立求和 | Hoeffding | 指数尾 |
| MGF 可控的独立项 | Chernoff | 指数尾 |
| 期望的凸函数 | Jensen | $$f(\mathbb E[X]) \le \mathbb E[f(X)]$$ |

**这张表里马尔可夫的假设最少：**非负且均值有限。
切比雪夫就是把马尔可夫用在 $$(X-\mu)^2$$ 上。

**Jensen 是真正出现在 ML 里、而不是出现在智力题里的那一个。**ELBO 之所以是下界、
$$\log \mathbb E[\cdot] \ge \mathbb E[\log \cdot]$$ 之所以在重要性采样里要紧、
KL 散度之所以非负，都是它。

<details><summary><strong>自测 · 选够用的最弱工具</strong></summary>
若非负 `X` 的均值为 4，马尔可夫给出 `P(X >= 20) <= 1/5`。
若还知道均值 `mu`、标准差为 3，切比雪夫给出
`P(|X - mu| >= 6) <= 1/4`。两者都不假设具体分布。
</details>

---

<a id="section-c2"></a>

## C2 · 先模拟，再验证

这种混合题要求先模拟一个物理场景，解析推导它的分布，再验证样本与推导一致。
这道练习有意把 coding 和概率连在一起，
所以两半都必须检查。

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

**这道题的题眼是均值不存在。**$$\int |x| f(x)\,dx$$ 发散，所以通常的大数定律不适用。
更强的结论是：$$n$$ 个独立标准 Cauchy 样本的均值，对每个 $$n$$ 仍是标准 Cauchy；
它不会随着 $$n$$ 增加而集中。可以在 $$10^4$$、$$10^5$$ 和 $$4\times10^5$$
个样本处观察 running mean，但不要把一条样本路径误当成证明。有限方差分布的标准误会按
$$1/\sqrt n$$ 下降；Cauchy 既没有有限均值，也没有有限方差。

**中位数是良态的**，改用它来估位置参数，才是实践上正确的应对。

**验证里埋着一个陷阱，而且是个好陷阱。**在 $$[-5, 5]$$ 这种窗口上拿直方图跟解析 PDF 比，
粗心就会*对不上*，而原因不是代码写错了。NumPy 的 `density=True` 是在你画出来的那些 bin 上
归一化的，但真正的 Cauchy 只有 $$\tfrac{2}{\pi}\arctan 5 = 0.874$$ 的质量落在那个窗口里。
尾巴重到这个程度，忽略截断会把你的直方图整体抬高 14%，让一次正确的模拟看上去是错的。
要跟*条件*密度 $$f(x)/P(|X|\leq L)$$ 比，其中
$$P(|X|\leq L)=2\arctan(L)/\pi$$；容差应由样本数决定，而不是设一个与 $$n$$ 无关的硬阈值。

> **这个练习为什么有用。**实现很短，但验证时必须明确图中 bin 实际代表哪个分布。
> 截断修正正是那类足以让真实实验失效的条件化问题。

<!-- EXERCISE p27 -->
**练习** — [`p27` · 旋转光源 → Cauchy 分布](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/stubs/p27_cauchy_simulation.py) · [提示](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/hints/p27_cauchy_simulation.md) · [测试](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/tests/test_p27_cauchy_simulation.py) · [`python run.py p27`](https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice/run.py) · 20 分钟
<!-- /EXERCISE -->

---

<a id="c2-2"></a>
### C2.2 通用套路

这个设定是可以推广的，手上有一套固定流程，就不用在时间压力下临场发挥。

**先推导，再模拟。**先模拟你就没有可对照的东西，而「直方图看着差不多」不算答案。
单调的 $$g$$ 的变换公式是 $$f_Y(y) = f_X(g^{-1}(y))\,|dg^{-1}/dy|$$；
$$g$$ 不单调，就在各个分支上求和。

**分三个层次验证，从最便宜的开始。**矩（均值、方差）在存在的时候一行就能算。
Kolmogorov–Smirnov 统计量给出不依赖分箱的全局 CDF 差异。
直方图有助于目视检查，但对分箱、范围和上面的截断问题都很敏感。

**主动说你打算拿方差怎么办。**样本量决定了你的分辨率：$$n$$ 个样本下，
一个概率为 $$p$$ 的 bin 相对误差大约是 $$1/\sqrt{np}$$，
所以尾部的 bin 很吵，不该在所有 bin 上一律用同样收紧的容差。

**把标准变换一一点名**，因为出题人想听的通常就是其中之一：
凡是有闭式分位函数的都用逆 CDF 采样，高斯用 Box–Muller，指数分布用 $$-\log U/\lambda$$，
Cauchy 用两个标准正态之比——最后这个正好是这道题的一个漂亮交叉验证，
因为它应该给出和正切构造完全相同的分布。

<details><summary><strong>自测 · 变换并验证</strong></summary>
令 `U` 在 `(0, 1)` 上均匀，并定义 `Y = -log(1 - U) / lambda`。
那么 `P(Y <= y) = P(U <= 1 - exp(-lambda*y)) = 1 - exp(-lambda*y)`，
正是指数分布 CDF。模拟时用分位数或 KS 统计量验证，不要只看直方图。
</details>

---

<a id="section-c3"></a>

## C3 · 线性代数

这一节补上源笔记的线性代数侧：把矩阵看作映射、秩与 SVD、半正定、条件数和矩阵微积分。
每个概念都会接回 coding 部分已经用过的操作。

---

<a id="c3-1"></a>
### C3.1 其余一切都从中推出的四条事实

**矩阵是一个线性映射，形状告诉你它连接的是哪两个空间。**形状为 $$(m, n)$$ 的 $$W$$
把 $$\mathbb R^n \to \mathbb R^m$$。你要是这样读矩阵、而不是把它读成一格一格的数字，
绝大多数形状 bug 自己就化掉了。

**秩是输出空间里真正被覆盖到的维数。**低秩矩阵把输入压进一个子空间。
对 LoRA，内维为 $$r$$ 的 $$\Delta W=BA$$ 满足
$$\operatorname{rank}(\Delta W)\le r$$：每个输入先穿过 $$r$$ 维瓶颈，再扩回输出空间。

**特征向量是映射只做缩放的那些方向**，$$Av = \lambda v$$。这个定义要求方阵；
一般实矩阵可能需要复特征向量，也未必有完整特征基。实对称矩阵则有一组实正交归一特征基——
Hessian 和协方差矩阵之所以尤其好处理，就是因为这个。

**SVD 对每一个矩阵都成立**，方阵不方阵都行：$$A = U\Sigma V^\top$$，
一次正交坐标变换、沿坐标轴缩放、再做一次正交坐标变换。奇异值就是那些缩放因子，
把小的截掉，就是 Frobenius 范数意义下最好的秩 $$k$$ 近似。
仅这一条事实就撑起了 PCA、低秩压缩，以及人们判断一次权重更新是不是「真的」低秩时的那套推理。

<details><summary><strong>自测 · 连接形状、秩与 SVD</strong></summary>
一个 `6 x 4` 矩阵的秩为 3。由秩-零度定理，它的零空间是一维的；它恰好有三个非零奇异值，
列空间为三维。Frobenius 范数下最好的 rank-2 近似保留最大的两个奇异三元组。
</details>

---

<a id="c3-2"></a>
### C3.2 半正定，以及它为什么反复出现

对实对称矩阵 $$M$$，若 $$x^\top M x \ge 0$$ 对所有 $$x$$ 成立，它就是半正定的；
等价地说，它的特征值全部非负。

**它在 ML 里真正决定了什么的三个地方。**协方差矩阵在构造上就是半正定的，
因为 $$x^\top \Sigma x$$ 是某个投影的方差，而方差非负。对凸域上的二阶可微目标，
Hessian 处处半正定意味着凸性，于是任何驻点都是全局最小；神经网络目标通常不满足这个条件。
还有核矩阵必须半正定，核技巧才对应到某个空间里的一个内积。

**Hessian 如何分类临界点。**正定 Hessian 给出严格局部极小，负定给出严格局部极大；
不定 Hessian 同时有上升和下降方向，所以是鞍点。Hessian 特征值的符号不是独立抛硬币，
因此用 $$2^{-d}$$ 论证极小点或鞍点有多常见，并不是一般性的证明。

<details><summary><strong>自测 · 证明一个常用矩阵半正定</strong></summary>
对任意长方形 `A`，`A.T @ A` 都半正定，因为
`x.T @ A.T @ A @ x = ||A @ x||_2^2 >= 0`。它恰好在 `A` 的零空间平凡时正定，
也就是 `A` 满列秩时。
</details>

---

<a id="c3-3"></a>
### C3.3 范数、条件性，以及那些会炸的东西

**用哪个范数，以及为什么这件事重要。**global-norm 梯度裁剪通常量 $$\ell_2$$ 范数；
Frobenius 范数就是把矩阵拉平之后的 $$\ell_2$$ 范数；谱范数是最大的那个奇异值，
也就是一个矩阵最多能把某个向量拉长多少。谱范数给出扰动穿过线性层后的放大上界，
因此是稳定性与 Lipschitz 分析的一部分。

**条件数** $$\kappa = \sigma_\max/\sigma_\min$$ 给出解满秩线性系统时相对扰动可能被放大的上界；
当 $$\sigma_\min=0$$ 时条件数为无穷。
特征缩放和预条件可以减少各向异性；Adam 的逐坐标缩放可以理解成自适应对角预条件子，
但它并不是逆 Hessian。

**由此得出两条数值规则。**能避开就绝不显式构造 $$X^\top X$$ 去解最小二乘：
那会把条件数平方，于是你损失掉两倍的有效位数。以及绝不为了解 $$Ax = b$$ 去求矩阵的逆
——用分解（`np.linalg.solve`，不是 `inv(A) @ b`），更快、数值上也更稳定。
问题的条件数是 $$A$$ 的性质，不会仅仅因为换了更好的算法就改善。

<details><summary><strong>自测 · 量化损失</strong></summary>
若矩阵的奇异值为 10 和 0.1，它的 2-范数条件数是 100。
构造正规方程矩阵会把条件数平方为 10,000，所以最小二乘更适合用 QR 或 SVD。
</details>

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

<details><summary><strong>自测 · 用微分和形状推导</strong></summary>
对 `y = X @ w` 与 `L = 0.5 * ||y - t||^2`，令 `r = X @ w - t`。
于是 `dL = r.T @ X @ dw`，梯度是 `X.T @ r`，形状与 `w` 相同。
</details>

---

<a id="section-c4"></a>

## C4 · 计数

第一步是在动手算之前说清楚*你在数什么*。典型错误是同一个对象被数了两遍，
或者题目没问顺序，却把顺序也数了进去。

---

<a id="c4-1"></a>
### C4.1 决定公式的那一个判断

动笔之前先回答两个问题：**顺序算不算数**，以及**元素能不能重复**。
这就是一张 2×2 的表，公式随之定死。

| | 顺序算数 | 顺序不算数 |
|---|---|---|
| **不可重复** | $$\dfrac{n!}{(n-k)!}$$ | $$\dbinom{n}{k}$$ |
| **允许重复** | $$n^k$$ | $$\dbinom{n+k-1}{k}$$ |

右下角那一格来自**隔板法（stars and bars）**：
把 $$k$$ 个相同的物品分进 $$n$$ 个有标号的盒子，等价于把 $$k$$ 个星和 $$n-1$$ 个隔板
排成一行，于是你只要在 $$n+k-1$$ 个位置里选出哪 $$k$$ 个是星。

**要说的是那个对应关系，不是那个公式。**「星与隔板的每一种排列恰好对应一种分法」
是证明；直接报出 $$\binom{n+k-1}{k}$$ 只是在说你背过。

<details><summary><strong>自测 · 先分类再计算</strong></summary>
八位小写字符串允许重复、顺序算数，所以有 `26^8` 个。
无顺序地选八个不同字母则是 `C(26, 8)`。选表达式之前，先说清楚所数的对象。
</details>

---

<a id="c4-2"></a>
### C4.2 先重复计数，再除掉

一个可复用的技巧是：数一个好数、但按已知倍数重复计数的对象，然后把倍数除掉。

**多重集的排列。**MISSISSIPPI 的字母：先假装 11 个字母两两不同（$$11!$$），
再除掉每一组相同字母内部的排列数，S、I、P 三组给出 $$4!\,4!\,2!$$。

**圆排列。**若旋转视为相同、镜像仍视为不同，$$n$$ 个人围圆桌有 $$(n-1)!$$ 种坐法；
每种坐法此前被数了 $$n$$ 次，每个旋转各一次。

**先选后排。**$$\binom{n}{k}k! = n!/(n-k)!$$ 把排列公式还原了出来，
这是个好自检——它说明你脑子里装的是一套模型，而不是两个背下来的公式。

> **一个调试启发：**当带标签和不带标签的计数相差一个对称群时，
> 重复因子往往是阶乘。例如三个可互换对象会产生 `3!` 的因子。

<details><summary><strong>自测 · 找出每个对称因子</strong></summary>
把 12 个不同的人分成三个**没有标签**的四人组。先排列所有人，
再除以每组内部的 `4!` 和三组之间的 `3!`，得到 `12! / ((4!)^3 3!)`。
</details>

---

<a id="c4-3"></a>
### C4.3 容斥原理

**识别特征：**「至少有一个」，或者若干互相重叠的条件求并。

$$|A \cup B \cup C| = \sum|A_i| - \sum|A_i \cap A_j| + |A_1 \cap A_2 \cap A_3|$$

正负交替是因为：把单个的加起来，两两交被数了两遍；再减掉两两交，
三重交就多减了一次；以此类推。

**最经典的应用是错排**——没有任何不动点的置换：

$$D_n = n!\sum_{k=0}^{n}\frac{(-1)^k}{k!} \approx \frac{n!}{e}$$

于是一个随机置换没有不动点的概率趋于 $$1/e \approx 0.368$$。
有限 $$n$$ 时概率并非常数，但收敛很快。

**先检查补集。**很多「至少一个」问题会变成「1 − 一个都没有」，
其中「一个都没有」只是一个乘积。若改写后仍有重叠事件，再用容斥原理。

<details><summary><strong>自测 · 使用补集</strong></summary>
随机置换至少有一个不动点的概率是 `1 - D_n/n!`，极限为 `1 - 1/e`。
这是错排事件的补集，不是把各不动点概率当成独立事件相加。
</details>

---

<a id="c4-4"></a>
### C4.4 计数在哪儿撞上 ML

这类题披上 ML 外衣的两个地方。

**生日问题，用在哈希碰撞和重复检测上。**把 $$n\leq d$$ 个独立均匀物品扔进 $$d$$ 个桶，
不发生碰撞的精确概率是 $$\prod_{i=0}^{n-1}(1 - i/d)$$，近似为
$$\exp[-n(n-1)/(2d)]$$。所以在数量级意义上，一旦 $$n \sim \sqrt d$$，
碰撞概率就是常数量级。在 birthday regime $$n=O(\sqrt d)$$ 下，这个近似渐近准确；
更一般地说，忽略 log 展开的下一项要求 $$n^3/d^2 \ll 1$$。
精确乘积的条件 $$n\leq d$$ 是另一回事；若 $$n>d$$，抽屉原理让无碰撞概率直接为零。
这就是为什么 64 位哈希不足以给一万亿篇文档去重；
这个近似也是审查任何哈希去重设计时的第一步。

**数参数量。**推 transformer 的参数量是一道带形状论证的计数题（第一篇 A10）。
纪律完全一样：动手乘之前，先说清楚你在数什么——每层、每头、embedding 还是 unembedding。

<details><summary><strong>自测 · 精确相乘前先估算</strong></summary>
均匀 128 位哈希的碰撞尺度是桶数的平方根，约为 `2^64` 个物品。
一万亿个物品时，pair-collision 近似约为 `10^24 / 2^129`，即 `1.5e-15`。
</details>

---

<a id="section-c5"></a>

## C5 · 马尔可夫链与随机游走

首步分析（C1.1）的天然主场，也是好几道经典智力题的舞台。
MCMC 和扩散直接使用马尔可夫过程；MDP 是带控制的扩展，
固定 policy 后则会诱导出一条马尔可夫链。

---

<a id="c5-1"></a>
### C5.1 马尔可夫性到底给你换来了什么

$$P(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = P(X_{t+1} \mid X_t)$$

给定现在，未来与过去条件独立。对有限、时齐的链，实际后果是**一个转移矩阵**
$$P$$ 就能指定动力学，而 $$n$$ 步的行为就是 $$P^n$$——于是动力学问题变成了线性代数。

**建模的功夫在于把状态选对**，和首步分析里一模一样。几乎任何过程，
只要把状态扩大到装下真正要紧的那部分历史，都能变成马尔可夫的。
对目标 `HTH`，只看上一次结果并不够；「当前匹配了目标最长前缀的几位」
（0、1、2，或已经吸收）才是充分状态。

**平稳分布** $$\pi$$ 满足 $$\pi P = \pi$$——一个特征值为 1 的左特征向量。
有限不可约链的平稳分布唯一；再加上非周期，$$P^n$$ 才保证收敛到它。
对有限可逆链，非平凡特征值可以给出收敛的谱隙界。谱隙倒数是混合时间的尺度，
而不是混合时间本身；常数与最小平稳质量也有影响。

<details><summary><strong>自测 · 解左特征向量方程</strong></summary>
对转移矩阵 `[[0.9, 0.1], [0.2, 0.8]]`，联立 `pi @ P = pi` 与
`pi.sum() = 1`。流量平衡给出 `0.1*pi0 = 0.2*pi1`，所以 `pi = [2/3, 1/3]`。
</details>

---

<a id="c5-2"></a>
### C5.2 赌徒破产

你手上有 $$i$$ 块钱，每次押一块赌一枚公平硬币，到 $$0$$ 或 $$N$$ 停手。

**先摸到 $$N$$ 的概率是 $$i/N$$**，最快的推法是注意到你的赌本是一个鞅：
它的期望从不改变，所以 $$i = 0\cdot(1-p) + N\cdot p$$，立刻得到 $$p = i/N$$。
首步分析走 $$p_i = \tfrac12 p_{i-1} + \tfrac12 p_{i+1}$$ 得到同一个答案，
在给定的边界条件下，它的解关于 $$i$$ 是线性的。

**期望持续时长是 $$i(N-i)$$**，值得知道，因为它长得出人意料——
$$N = 100$$ 从一半起步，被吸收之前的期望下注次数是 2,500。

**直觉崩掉的地方是有偏的情形。**胜率 $$q \ne 1/2$$ 时，答案变成几何项之比，
哪怕 $$q$$ 只比 $$1/2$$ 低一点点，破产概率也会指数级地逼近 1。
该说出口的结论是：对手那一点点持续的优势，放到很多轮上并不是一点点劣势。

<details><summary><strong>自测 · 同时用两组边值解</strong></summary>
公平游走取 `i = 3`、`N = 10` 时，先到 10 的概率是 `3/10`，
期望吸收时间是 `3 * (10 - 3) = 21` 次下注。
</details>

---

<a id="c5-3"></a>
### C5.3 随机游走，以及维数带来的意外

**$$\mathbb Z$$ 上的对称游走。**走 $$n$$ 步之后 $$\mathbb E[X_n] = 0$$、
$$\operatorname{Var}(X_n) = n$$，所以典型位移按 $$\sqrt n$$ 增长。
标准误与布朗扩散里也出现同样的平方根缩放，因为独立增量相加的是方差，而不是标准差。

**Pólya 定理：**对称随机游走在一维和二维是常返的——以概率 1 回到原点——
而在**三维及以上是暂留的**。三维最终返回概率约为 0.34，更高维还会更小。

**期望返回时间。**对有平稳分布 $$\pi$$ 的不可约正常返链，Kac 公式给出回到状态
`i` 的期望时间 $$1/\pi_i$$。在 $$\mathbb Z$$ 上，游走以概率 1 返回，
却没有平稳概率分布，而且期望返回时间为无穷——常返，却是零常返。

<details><summary><strong>自测 · 分开均值与尺度</strong></summary>
一维对称游走 100 步后的平均位置为 0，标准差为 10。
恰好回到原点的概率是 `C(100, 50) / 2^100`，约为 `0.0796`；
均值为零不表示大概率待在零点。
</details>

---

<a id="c5-4"></a>
### C5.4 可迁移的例题

每个例子都隔离出一个可以迁移到陌生问题的建模动作。

**优惠券收集。**集齐全部 $$n$$ 张券要 $$n H_n \approx n\ln n$$ 次抽取，
做法是拆成一段段几何等待时间（C1.2）。方差是 $$O(n^2)$$，
所以波动仍有 $$n$$ 的量级——只要你是在靠采样覆盖一个空间，这条就有关系。

**生日问题。**碰撞变得很可能是在 $$n \sim \sqrt d$$，不是 $$n \sim d$$（C4.4）。

**Monty Hall。**换门赢 2/3。原因是主持人的动作和真相并不独立——他从不打开有奖的那扇门
——所以他的选择传递了信息。要把它讲成对主持人的*规则*做条件，而不是对那扇门做条件。

**秘书问题。**渐近地，先拒掉大约前 $$n/e$$ 个候选人，之后遇到第一个好过已见全部的就拿下。
成功概率趋于 $$1/e \approx 0.37$$；同一个 $$1/e$$ 也出现在错排里（C4.3），但成因无关。

**蓄水池抽样。**对容量为一的 reservoir，以概率 $$1/k$$ 用流中的第 $$k$$ 个元素替换当前元素。
一行归纳法能说明 $$n$$ 个流元素中的每一个最终都以
$$1/n$$ 的概率留下。这是在流长度事先未知时做均匀抽样的正确原语。

**连续两次正面。**期望 6 次（C1.1）。追问——HT 的期望次数——是 4，
而 HH 和 HT 会不一样，这才是有意思的部分：HT 尝试因再次出现 H 而失败时仍保留部分进度，
HH 尝试因 T 失败时则不会。真正改变答案的是目标模式的重叠结构。

<details><summary><strong>自测 · 证明 reservoir 均匀</strong></summary>
第 `j` 个元素以 `1/j` 的概率进入容量为一的 reservoir，随后从 `j+1` 活到 `n` 的概率为
`(j/(j+1)) * ((j+1)/(j+2)) * ... * ((n-1)/n) = j/n`。
两者相乘为 `1/n`，与 `j` 无关。
</details>

---

<a id="section-c6"></a>

## C6 · 统计与估计

这一节把上面的概率论连接到 ML 中实际使用的 loss、估计量与评测决策。

---

<a id="c6-1"></a>
### C6.1 最大似然，以及你的损失函数究竟是什么

MLE 挑的是让观测数据最可能出现的那组参数：

$$\hat\theta = \arg\max_\theta \prod_i p(x_i \mid \theta)
= \arg\max_\theta \sum_i \log p(x_i \mid \theta)$$

取 log，是因为一大堆小数连乘会下溢，也因为求和求导干净。

**值得随叫随写的三个推导**，因为每一个都表明某个眼熟的 loss *就是*某种噪声假设下的 MLE：

**方差已知的高斯给出最小二乘。**$$\log p \propto -(x-\mu)^2$$，
所以最大化似然就是最小化平方误差。在似然解释下，使用 MSE 对应方差固定的高斯观测模型。

**伯努利给出交叉熵。**$$\log p = y\log\hat y + (1-y)\log(1-\hat y)$$，
这是二元交叉熵的负数，因此最小化负对数似然就得到 BCE；
类别分布的似然给出多分类交叉熵。

**Laplace 给出绝对误差。**$$\log p \propto -|x - \mu|$$，
所以 L1 loss 对应尾部比高斯更厚的噪声——这是 L1 对大残差不那么敏感的概率解释。

> **三个都配套的那个追问：**MAP 估计加了一个先验，权重上的高斯先验给出 L2 正则，
> Laplace 先验给出 L1。经典 SGD 的 weight decay 等价于 L2 正则；
> 自适应优化器里的 decoupled weight decay 一般不等价。L1 在零点不可导，所以鼓励精确的零。

<details><summary><strong>自测 · 区分似然与先验</strong></summary>
观察到 8 次成功、2 次失败时，伯努利 MLE 为 `p = 0.8`。
加 `Beta(2, 2)` 先验后，MAP 众数为 `(8 + 2 - 1) / (10 + 2 + 2 - 2) = 0.75`。
</details>

---

<a id="c6-2"></a>
### C6.2 偏差、方差，以及有偏估计量何时有用

对一个估计量 $$\hat\theta$$：

$$\mathbb E[(\hat\theta - \theta)^2] = \underbrace{(\mathbb E[\hat\theta]-\theta)^2}_{\text{bias}^2}
+ \underbrace{\operatorname{Var}(\hat\theta)}_{\text{variance}}$$

**$$n-1$$ 那个问题。**用 $$1/n$$ 的样本方差偏小，
因为你量的是相对*样本*均值的偏离，而样本均值本身是拟合出来的，
因此它比真实均值更贴近这些点。除以 $$n-1$$ 就修正了——你花掉了一个自由度去估计均值。

**接下来是让它成为 ML 回答的那部分：**目标不是无偏，是均方误差，
一个方差更小的有偏估计量常常更划算。很多归一化层的 forward 使用等价于
`torch.var(unbiased=False)` 的计算——它们归一化当前激活集合，而不是估计总体方差。
BatchNorm 里的微妙之处也在这里：它用有偏方差做归一化，累积的却是无偏的那个（B1.2）。

**经典图像遗漏了什么。**对平方误差，分解本身始终成立；可能失效的是
「模型越大，就沿 U 形曲线单调地用偏差换方差」这个简单故事。Double descent 表明，
越过插值阈值后测试风险可以再次下降，具体取决于优化、数据与隐式正则（第一篇 A1.8）。

<details><summary><strong>自测 · 区分目标量与估计量</strong></summary>
对观测 `1, 2, 3`，相对样本均值的平方偏差为 `1, 0, 1`。
除以 3 得到经验方差 `2/3`；除以 `n - 1 = 2` 得到 iid 总体方差的无偏估计 1。
</details>

---

<a id="c6-3"></a>
### C6.3 集中不等式：你到底需要多少样本

这是 C1.5 里那些不等式的实用形态，也是有人问「多少条 eval 样本才够」时你真正会用的东西。

**对一个比例**，标准误是 $$\sqrt{p(1-p)/n}$$，最坏情况 $$1/(2\sqrt n)$$。
在 iid 正态近似下，最坏情况的 95% 置信区间大致是 $$\pm 1/\sqrt n$$：
100 条给 $$\pm 10\%$$，
1,000 条给 $$\pm 3\%$$，10,000 条给 $$\pm 1\%$$。**把这三个背下来**，
把它们当作最坏情况的尺度。500 条样本的 benchmark 上 2 个点的差距可能落在抽样噪声内；
要看配对置信区间，不能只看两个点估计。

**Hoeffding** 不用正态近似就给出同样的形状：对 `[0, 1]` 内的独立变量，
$$P(|\bar X - \mu| \ge t) \le 2\exp(-2nt^2)$$，于是 $$n \sim \log(1/\delta)/t^2$$。
贵的是那个 $$1/t^2$$——精度多要一位，样本量就要 100 倍。

**值得知道的配对比较技巧。**在同一批样本上比较两个模型时，
比的应该是逐样本的差值，而不是两个均值。如果样本难度对两个模型正相关，
差值的方差可能小得多。应报告这些配对差值的置信区间，而不是默认点估计是真的。

<details><summary><strong>自测 · 反解集中界</strong></summary>
Hoeffding 取误差 `t = 0.03`、失败概率 `delta = 0.05` 时，需要
`n >= log(2/delta) / (2t^2)`，所以 `n >= 2050` 个有界独立样本已经足够。
这里假设样本独立且取值在 `[0, 1]`。这个界可能保守；这里练的是代数。
</details>

---

<a id="c6-4"></a>
### C6.4 假设检验，简单说说，以及它在 ML 里的失效模式

实践问题是某个 benchmark 上的提升是不是真的，而不是能不能背出 t 检验的机械步骤。

**你需要的那套词汇：**p 值是 $$P(\text{data this extreme} \mid H_0)$$
——*不是*原假设为真的概率，把这个说反了是经典错误。
对应的置信区间还给出效应量范围，而不只是对原假设证据做阈值判断。

**多重比较是 ML 中一个主要失效模式。**二十个彼此独立且原假设为真的检验，
都用 $$p < 0.05$$ 时，假阳性的期望数为 1，至少出现一个的概率约为 64%。
如果用同一个 benchmark 从超参或多个变体中做选择，就会产生同一个问题；
应对包括只碰一次的留出集、校正，或事先登记测量目标。

> **这里要建立的联系**是：这跟在验证集上过拟合是同一种失败，
> 只不过换成了统计的视角。在一个 benchmark 上选模型，就让那个 benchmark 变成了
> 对它性能的乐观估计；最终估计应该来自没有驱动模型选择的数据。

<details><summary><strong>自测 · family-wise error</strong></summary>
20 个独立原假设各按 0.05 检验，至少一个假阳性的概率是 `1 - 0.95^20`，约 `0.642`。
Bonferroni 把每个阈值设成 `0.05/20 = 0.0025`，无需独立性也能让 family-wise error
至多为 0.05。
</details>

---

<a id="section-refs"></a>

## 参考文献

本文的技术论断以以下原始论文为准。关于面试题型的说法只标为「轶事性面经」或「轶事性报告」，
来源是列出的实名准备复盘；公司名不表示官方题库。

1. Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)。GQA 原始论文。
2. Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)。FlashAttention 原始论文。
3. DeepSeek-AI. (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)。MLA 技术来源。
4. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)。LoRA 原始论文。
5. Jaiswal, M. (2024–2025). *LLM (ML) Job Interviews — Resources.* [mimansajaiswal.github.io](https://mimansajaiswal.github.io/posts/llm-ml-job-interviews-resources/)。实名面试准备复盘。
6. Karpathy, A. *micrograd* and *nanoGPT.* [micrograd](https://github.com/karpathy/micrograd) · [nanoGPT](https://github.com/karpathy/nanoGPT)。端到端实现参考。
7. Leviathan, Y., Kalman, M., & Matias, Y. (2022). *Fast Inference from Transformers via Speculative Decoding.* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)。投机解码原始论文。
8. Liu, A. (2026). *Notes on the Industry Job Search.* [alisawuffles.github.io](https://alisawuffles.github.io/blog/job-search/)。实名求职与面试复盘。
9. Meng, Y. (2026). *MLE Interview 2.0: Research Engineering and Scary Rounds.* [yuan-meng.com](https://www.yuan-meng.com/posts/mle_interviews_2.0/)。实名面试复盘。
10. Milakov, M., & Gimelshein, N. (2018). *Online Normalizer Calculation for Softmax.* [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)。在线 softmax 来源。
11. Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)。DPO 原始论文。
12. Sapora, S. (2026). *ML Job Interviews: The Ultimate Guide.* [silviasapora.github.io](https://silviasapora.github.io/blog/ml-interviews.html)。实名面试准备复盘。
13. Schulman, J., et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)。GAE 原始论文。
14. Sennrich, R., Haddow, B., & Birch, A. (2015). *Neural Machine Translation of Rare Words with Subword Units.* [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)。BPE 论文。
15. Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)。GRPO 来源。
16. Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All You Need.* [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)。MQA 来源。
17. Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)。RoPE 来源。
18. Vaswani, A., et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)。Transformer 来源。
19. Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)。RMSNorm 来源。
20. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)。BatchNorm 原始论文。
21. Shazeer, N. (2020). *GLU Variants Improve Transformer.* [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)。SwiGLU 来源。
22. Raz, G., et al. (2024). *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization.* [arXiv:2410.08847](https://arxiv.org/abs/2410.08847)。DPO likelihood displacement 来源。
23. Chen, C., et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling.* [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)。投机采样原始论文。
24. Fedus, W., Zoph, B., & Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)。Switch 风格 MoE routing 来源。
