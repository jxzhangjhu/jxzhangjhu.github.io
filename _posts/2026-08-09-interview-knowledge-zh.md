---
layout: post
title: "面试题库 I · Knowledge：LLM 与 ML 知识自测（中文版）"
date: 2026-08-09 11:00:00
author: Jiaxin Zhang
description: "一个自测用的 LLM/ML 知识题库。每题带追问和常见错答。基于 Alisa Liu 公开的学习笔记，加上数据、agentic RL、校准三块自己的补充。"
tags: interviews llm ml knowledge qbank
categories: research-notes
giscus_comments: true
related_posts: false
ai_assisted: true
---

<div class="lang-switch"><a href="/blog/2026/interview-knowledge/">English</a> · <strong>中文</strong></div>

<div class="lang-switch"><strong>I · 知识</strong> · <a href="/blog/2026/interview-coding-zh/">II · 代码 + 数学</a> · <span class="text-muted">III · 讨论 + BQ</span></div>

这是一个**自测题库**，不是教程。它的存在理由只有一个：面试前我只想过一个地方。

> **怎么用。**看到问题**先在心里答出来**，再往下看。如果你直接读答案，这个页面对你没有任何
> 作用——所有一手材料都指向同一件事：瓶颈是**回忆**，不是**认得**。
>
> **每个概念的结构是**讲解 → `自测` → 题目。每道题的答案后面跟着**追问**（他们接下来一定会问的）
> 和**陷阱**（最常见的错答）。人是死在 follow-up 上的，不是主问题。

**底本。**A1–A6 主要基于 Alisa Liu 公开的 LLM 笔记（她 2026 年从博士进入 OpenAI，
把整个求职过程和学习材料都公开了），加上我自己补的量化、MoE、MFU、长上下文。
A9、A12、A13 压缩自我自己写过的长文：数据管线、环境扩展与 agentic RL、校准与持续学习。

**这一篇的范围**是知识层——「你想得起来吗」。手写代码和数学在第二篇，
系统设计对话和 BQ 在第三篇。

---

### 目录

- **[A1 · ML / DL 基础](#section-a1)** — 24 题
  - [A1.1 线性层与矩阵形式](#a1-1)
  - [A1.2 激活函数](#a1-2)
  - [A1.3 梯度、Jacobian、Hessian](#a1-3)
  - [A1.4 反向传播与计算图](#a1-4)
  - [A1.5 优化器](#a1-5)
  - [A1.6 学习率调度](#a1-6)
  - [A1.7 归一化](#a1-7)
  - [A1.8 泛化、正则化与 double descent](#a1-8)
  - [A1.9 损失函数与信息论](#a1-9)
  - [A1.10 数值稳定性](#a1-10)
  - [A1.11 训练循环与调试](#a1-11)
  - [A1.12 基础统计](#a1-12)
  - [A1.13 采样中的梯度流](#a1-13)
  - [A1.14 理论 CS 的几个常用件](#a1-14)
- **[A2 · Transformer 架构与实现](#section-a2)** — 18 题
  - [A2.1 三种架构范式](#a2-1)
  - [A2.2 一个 block 的解剖：残差流](#a2-2)
  - [A2.3 Self-attention 与 $$\sqrt{d_k}$$](#a2-3)
  - [A2.4 手写实现](#a2-4)
  - [A2.5 注意力变体：MHA → MQA → GQA → MLA](#a2-5)
  - [A2.6 位置编码：RoPE](#a2-6)
  - [A2.7 FFN 与 SwiGLU](#a2-7)
  - [A2.8 ★ Mixture of Experts](#a2-8)
  - [A2.9 ★ 分词](#a2-9)
  - [A2.10 参数都在哪里](#a2-10)
  - [A2.11 长上下文的架构手段](#a2-11)
  - [A2.12 ★ 多模态怎么接进来](#a2-12)
  - [A2.13 ★ 注意力的替代品](#a2-13)
- **[A3 · 常见模型](#section-a3)** — 6 题
  - [A3.1 一张对照表](#a3-1)
  - [A3.2 Llama 3：把 Chinchilla 扔掉](#a3-2)
  - [A3.3 DeepSeek-V3 / R1：三个值得学的选择](#a3-3)
  - [A3.4 Qwen3 与 hybrid thinking](#a3-4)
  - [A3.5 Mixtral 与 MoE 的主流化](#a3-5)
- **[A4 · 预训练](#section-a4)** — 9 题
  - [A4.1 训练目标：为什么是 next-token prediction](#a4-1)
  - [A4.2 从零训一个模型的顺序](#a4-2)
  - [A4.3 架构与超参的选择](#a4-3)
  - [A4.4 训练动态：曲线该长什么样](#a4-4)
  - [A4.5 Checkpoint 与容错](#a4-5)
  - [A4.6 预训练里的评测](#a4-6)
- **[A5 · 训练基础设施](#section-a5)** — 6 题
  - [A5.1 显存都花在哪](#a5-1)
  - [A5.2 并行策略：每种切什么](#a5-2)
  - [A5.3 混合精度](#a5-3)
  - [A5.4 MFU](#a5-4)
  - [A5.5 训练不稳定的诊断](#a5-5)
- **[A6 · Post-training 与 RL](#section-a6)** — 12 题
  - [A6.1 后训练阶梯](#a6-1)
  - [A6.2 SFT：细节比想象中多](#a6-2)
  - [A6.3 Reward model 与 Bradley-Terry](#a6-3)
  - [A6.4 Policy gradient 的推导](#a6-4)
  - [A6.5 Baseline 为什么无偏](#a6-5)
  - [A6.6 PPO](#a6-6)
  - [A6.7 GRPO](#a6-7)
  - [A6.8 DPO](#a6-8)
  - [A6.9 Reward hacking 与 KL 控制](#a6-9)
  - [A6.10 ★ 蒸馏](#a6-10)
  - [A6.11 LoRA 与 PEFT](#a6-11)
- **[A7 · 推理模型与 test-time compute](#section-a7)** — 6 题
  - [A7.1 第三条扩展轴](#a7-1)
  - [A7.2 推理模型是怎么训出来的](#a7-2)
  - [A7.3 推理模型的代价](#a7-3)
  - [A7.4 训练算力 vs 推理算力：怎么分配](#a7-4)
- **[A8 · 推理与服务](#section-a8)** — 12 题
  - [A8.1 Prefill 与 decode 是两台机器](#a8-1)
  - [A8.2 服务指标：先问要优化哪个](#a8-2)
  - [A8.3 KV cache](#a8-3)
  - [A8.4 Continuous batching 与 PagedAttention](#a8-4)
  - [A8.5 Prefix caching](#a8-5)
  - [A8.6 投机解码](#a8-6)
  - [A8.7 采样](#a8-7)
  - [A8.8 FlashAttention](#a8-8)
  - [A8.9 ★ 量化](#a8-9)
  - [A8.10 ★ 长上下文扩展](#a8-10)
  - [A8.11 Batching、packing 与 padding](#a8-11)
- **[A9 · 数据](#section-a9)** — 9 题
  - [A9.1 监督信号的三个来源](#a9-1)
  - [A9.2 预训练数据：过滤才是产品](#a9-2)
  - [A9.3 Midtraining：没人写下来的那一阶段](#a9-3)
  - [A9.4 SFT 数据：一道就绪门，不是能力来源](#a9-4)
  - [A9.5 RL 数据是题目，不是答案](#a9-5)
  - [A9.6 验证阶梯](#a9-6)
  - [A9.7 Agent 级数据](#a9-7)
  - [A9.8 合成数据什么时候坍塌](#a9-8)
  - [A9.9 污染](#a9-9)
- **[A10 · 估算题](#section-a10)** — 13 题
  - [A10.0 四个锚点数字与三条公式](#a10-0)
- **[A11 · Scaling 与评测](#section-a11)** — 7 题
  - [A11.1 Kaplan 与 Chinchilla](#a11-1)
  - [A11.2 muP](#a11-2)
  - [A11.3 Test-time compute 对评测的影响](#a11-3)
  - [A11.4 困惑度](#a11-4)
  - [A11.5 无法验证答案时怎么评测](#a11-5)
  - [A11.6 涌现是真的吗](#a11-6)
  - [A11.7 设计一个评测](#a11-7)
- **[A12 · Agentic RL 与环境](#section-a12)** — 8 题
  - [A12.1 从 chat 到 agent：形式上变了什么](#a12-1)
  - [A12.2 环境的解剖](#a12-2)
  - [A12.3 难度 ≠ 可训练性](#a12-3)
  - [A12.4 长时程的信用分配](#a12-4)
  - [A12.5 环境扩展的管线](#a12-5)
  - [A12.6 工具设计与失效模式](#a12-6)
  - [A12.7 Agent 评测](#a12-7)
  - [A12.8 为什么要 RL，而不是在好轨迹上做 SFT](#a12-8)
- **[A13 · 对齐、校准与持续学习](#section-a13)** — 9 题
  - [A13.1 完整的 RLHF 流程](#a13-1)
  - [A13.2 Constitutional AI 与 RLAIF](#a13-2)
  - [A13.3 校准的定义与度量](#a13-3)
  - [A13.4 为什么后训练会破坏校准](#a13-4)
  - [A13.5 Agent 的校准有什么不同](#a13-5)
  - [A13.6 灾难性遗忘](#a13-6)
  - [A13.7 部署之后的学习](#a13-7)
  - [A13.8 监控，以及为什么不要在 CoT 上做训练](#a13-8)
  - [A13.9 越狱与对抗鲁棒性](#a13-9)
- **[参考文献](#section-refs)**

---
<a id="section-a1"></a>

## A1 · ML / DL 基础

这一节是 rapid-fire 轮的主战场。Meng 的原话：*"一两个答错就足以被拒。"*

**读法：**先顺着概念读一遍建立骨架，再做每个概念下面的自测题。概念部分是给你**建立系统认知**的，
题目是用来**检验回忆**的——两件事不能互相替代。

---

<a id="a1-1"></a>
### A1.1 线性层与矩阵形式

**单个神经元**做的事：对输入加权求和、加偏置、过激活函数。

$$y=f\Big(\sum_{i=1}^n w_i x_i+b\Big)=f(\mathbf w^\top\mathbf x +b)$$

**一层**就是把很多神经元的权重向量堆成矩阵。$$n_\text{in}$$ 输入、$$n_\text{out}$$ 个神经元：

$$\mathbf h=f(W\mathbf x+\mathbf b),\qquad W\in\mathbb R^{n_\text{out}\times n_\text{in}}$$

**批处理**时把 $$m$$ 个样本按行排成矩阵，$$W$$ 的形状按惯例转过来：

$$H=f(XW+\mathbf b),\qquad X\in\mathbb R^{m\times n_\text{in}},\ W\in\mathbb R^{n_\text{in}\times n_\text{out}}$$

其中 $$\mathbf b$$ 被广播成 $$m\times n_\text{out}$$。

> **实现注记。**数学记号里 $$W$$ 是 $$(n_\text{in}, n_\text{out})$$，但 **PyTorch 把它存成
> $$(n_\text{out}, n_\text{in})$$**，前向算的是 `X @ W.T`。转置是免费的（只改 stride，不搬数据）。
>
> **别把理由说成「让梯度形状对上」**——两种存法梯度形状都自动对得上（存 $$(out,in)$$ 时
> $$\partial L/\partial W = (\partial L/\partial Z)^\top X$$ 就是 $$(out,in)$$）。真正的理由是
> **行主序下每个输出单元的权重是连续的**，符合 GEMM 的访存模式；另一半是历史包袱，
> 从 Torch7 的 `nn.Linear` 继承下来。

**先约定记号，因为反向传播引入了两个新符号。**记 $$Z = XW + \mathbf b$$ 为**激活前**的输出，
于是这一层就是 $$H = f(Z)$$。而 $$L$$ 是最终的**标量损失**——整个 batch 算出来的一个数，
来自网络最末端的损失函数。

反向传播要算的是 $$L$$ 对每个张量的偏导，而**每个梯度的形状永远等于它所对应张量的形状**。
这正是这些公式可以自检的原因：能把手头的算子收缩成正确形状的方式，通常只有一种。

**反向传播。**对 $$Z=XW+\mathbf b$$：

$$\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Z}W^\top,\qquad
\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Z},\qquad
\frac{\partial L}{\partial b_j}=\sum_{i=1}^m\frac{\partial L}{\partial z_{ij}}$$

> **一条能重建全部三个公式的规则。**先对**单个样本**推 Jacobian（干净、二维），然后：
> 张量若在 batch 间**共享**（如 $$W,b$$）→ batch 维被**求和掉**（contract）；
> 张量若**不共享**（如激活 $$X$$）→ batch 维被**保留**（stack）。
> 这条规则可以推广到 attention 和任何被问到的层，比背公式可靠。

#### 自测 · A1.1

**Q A1.1.1** — 推导 $$\partial L/\partial X$$、$$\partial L/\partial W$$ 和 $$\partial L/\partial b$$
（对 $$Z = XW + b$$），并说明你会怎么在不重推一遍的前提下检查它们。

三个公式在上面。真正要说的是检查方法：

$$\frac{\partial L}{\partial X}:\ (m,n_\text{out})\times(n_\text{out},n_\text{in}) = (m,n_\text{in})\ \checkmark$$
$$\frac{\partial L}{\partial W}:\ (n_\text{in},m)\times(m,n_\text{out}) = (n_\text{in},n_\text{out})\ \checkmark$$

**对某个权重求出的梯度，形状永远和这个权重一样。**如果你写的表达式形状对不上，那它就是错的
——不需要再推一遍。

> **追问**
> - *为什么 bias 的梯度要在 batch 维上求和？* → 同一个 $$b$$ 被加到每一行上，
>   所以每一行都产生自己的一份梯度，它们累加起来。
>
> **陷阱**
> - $$\partial L/\partial W$$ 写成 $$\frac{\partial L}{\partial Z}X^\top$$ —— 形状对不上。


**Q A1.1.2** — PyTorch 为什么把权重按数学惯例的转置来存？

小心，最顺口的那个答案是错的。「让梯度形状和存储形状对上」区分不了两种存法，因为**两种都对得上**：
存成 $$(n_\text{out}, n_\text{in})$$ 时 $$\partial L/\partial W = (\partial L/\partial Z)^\top X$$
就是 $$(n_\text{out}, n_\text{in})$$；存成 $$(n_\text{in}, n_\text{out})$$ 时
$$X^\top(\partial L/\partial Z)$$ 就是 $$(n_\text{in}, n_\text{out})$$。

真正的理由是内存布局和历史。行主序下**每个输出单元的权重是连续的一行**，正好是 GEMM 想要的访存模式；
另一半是从 Torch7 的 `nn.Linear` 继承下来的惯例。前向那次转置免费，因为它只交换 stride，不搬数据。

> **追问**
> - *什么是 stride？* → 沿每个维度走一格在内存里要跨过的步长。转置只是交换 stride 而不复制，
>   这也是为什么 `.view()` 在转置过的张量上会失败、必须先 `.contiguous()`。


---

<a id="a1-2"></a>
### A1.2 激活函数

没有非线性，多层会**塌缩成一层**：$$W_2(W_1x)=(W_2W_1)x=Wx$$。加了非线性之后网络才是
universal approximator。

| 函数 | 形式 | 导数特性 | 主要问题 |
|---|---|---|---|
| sigmoid | $$\frac{1}{1+e^{-x}}$$ | $$\sigma(1-\sigma)\le 0.25$$ | 梯度消失；非零中心 |
| tanh | $$2\sigma(2x)-1$$ | $$1-\tanh^2\in(0,1]$$ | 仍然 ≤1，深了照样消失 |
| ReLU | $$\max(x,0)$$ | 正半轴恒为 1 | dying ReLU |
| Leaky ReLU | $$x$$ / $$\alpha x$$ | 负半轴 $$\alpha$$ | 修好了 dying |
| Swish | $$x\cdot\sigma(x)$$ | 平滑、非单调 | — |
| GLU | $$xW_1\odot\sigma(xW_2)$$ | 门控 | 参数翻倍 |
| SwiGLU | $$(xW_1)\odot\text{Swish}(xW_2)$$ | 现代 LLM 默认 | 三个矩阵 |

![激活函数与它们的导数](/assets/img/blog/interview-knowledge/qa1_activations.png)
*右图是这一节的全部内容：sigmoid 的导数上界 0.25，每层最多乘 1/4；tanh 峰值 1.0 但仍 ≤1；
只有 ReLU 在正半轴恒为 1，不缩放梯度。*

**关键直觉：**sigmoid 和 tanh 的导数**永远只会缩小**梯度，所以深度一上去必然消失。ReLU 的导数
在正半轴恒为 1，这是它能训深网络的全部原因。代价是**负半轴恒为 0**——一个预激活对所有输入都为负
的神经元会永久收不到梯度，死掉。

#### 自测 · A1.2

**Q A1.2.1** — 比较 sigmoid、tanh、ReLU 作为隐层激活。各自的失效模式是什么？

**Sigmoid。**导数 $$\le 0.25$$，所以每层最多把梯度乘 $$1/4$$ → 梯度消失随深度复利式累积。
它还不是零中心的，进入同一个权重的梯度符号相同 → 优化路径来回折。
输出层要产出概率时它仍然是对的选择。

**tanh。**零中心，修好了符号问题，导数峰值为 1。但这个因子仍然只会缩小——消失被推迟了，
没有被解决。

**ReLU。**正半轴导数恰好是 1，梯度流得过去，而且便宜。失效模式是 **dying ReLU**：
一个预激活对所有输入都为负的单元，永远拿不到梯度。

> **追问**
> - *为什么最后大家都用门控变体？* → 纯经验。Shazeer 自己的论文里写它们
>   "把成功归于神的恩典"。没有干净的理论。
>
> **陷阱**
> - 把 tanh 说成"解决了梯度消失"。它只是把上界从 0.25 抬到 1。


**Q A1.2.2** — SwiGLU 为什么是 $$F = \tfrac{8}{3}D$$ 而不是 $$4D$$？

SwiGLU 有**三个**矩阵（$$3DF$$ 个参数），而普通 FFN 只有两个（$$2\cdot4D^2=8D^2$$）。
令 $$3DF = 8D^2$$ 得到 $$F=\tfrac83 D$$——参数量被固定住，比较才是公平的。

> **陷阱**
> - FFN 写成两个矩阵。


---

<a id="a1-3"></a>
### A1.3 梯度、Jacobian、Hessian

**导数**说的是敏感度：$$\partial f/\partial x = 3$$ 意味着 $$x$$ 变动 $$h$$，$$f$$ 变动约 $$3h$$。

**梯度** $$\nabla f$$ 是偏导组成的向量（标量输出）。

**Jacobian**（$$f:\mathbb R^n\to\mathbb R^m$$）是 $$m\times n$$ 矩阵，形状是 **输出 × 输入**：

$$\frac{\partial f}{\partial x}=\begin{bmatrix}
\partial f_1/\partial x_1 & \cdots & \partial f_1/\partial x_n\\
\vdots & \ddots & \vdots\\
\partial f_m/\partial x_1 & \cdots & \partial f_m/\partial x_n\end{bmatrix}$$

**Hessian** 是二阶偏导矩阵 $$H_{ij}=\partial^2 f/\partial x_i\partial x_j$$，描述损失曲面的**曲率**；
在极小点处半正定。

**链式法则**：标量乘导数，向量乘 Jacobian。

$$\frac{\partial \mathbf h}{\partial \mathbf x}=\frac{\partial \mathbf h}{\partial \mathbf z}\frac{\partial \mathbf z}{\partial \mathbf x}$$

> **为什么 LLM 里没人物化 Hessian。**它是 $$P\times P$$。70B 参数就是 $$5\times10^{21}$$ 个元素。
> 二阶方法改用 Hessian-vector product（二次反传）或对角近似——Adam 的 $$v$$ 就是一个粗糙的对角代理。

#### 自测 · A1.3

**Q A1.3.1** — softmax 的 Jacobian 是什么，为什么从来不把它物化出来？

对单独一行，$$\partial p_i/\partial s_j = p_i(\delta_{ij}-p_j)$$，所以 Jacobian 是
$$\mathrm{diag}(p) - pp^\top$$ ——**每一行**都是一个稠密的 $$T\times T$$ 矩阵，
整条序列物化出来就是 $$T^3$$。

反向传播直接算那个矩阵-向量乘积：

$$dS = P \odot \big(dP - \mathrm{rowsum}(dP \odot P)\big)$$

> **追问**
> - *这一步会出现在哪？* → 它就是 attention 反向传播的中间那一行，
>   面试官会专门问这一步。


**Q A1.3.2** — Hessian 告诉你什么，半正定为什么重要？

损失曲面的曲率。在局部极小点处它是半正定的（所有特征值 $$\ge 0$$），意思是每个方向都往上弯。
条件数（最大特征值与最小特征值之比）说明这个问题有多病态——条件数大，
正是朴素梯度下降会来回折、而自适应方法有用的原因。

> **追问**
> - *和 Adam 的联系？* → Adam 的 $$\sqrt{v}$$ 是曲率的对角近似：每个坐标除以它自己近期的
>   梯度幅度，等于一个廉价的逐坐标预条件子。


---

<a id="a1-4"></a>
### A1.4 反向传播与计算图

**两个核心思想**，能讲清这两个就能从零重建 autograd：

1. **每个操作存一个闭包**，它知道怎么把梯度推给自己的输入。图是在前向过程中**隐式**建起来的，
   每个节点捕获它的父节点和局部导数规则。
2. **梯度是累加的，遍历顺序是逆拓扑序。**一个被用在多处的节点会从多条路径收到梯度——所以是
   `+=` 而不是 `=`。拓扑排序保证调用某节点的 backward 时，它的所有消费者都已经贡献完毕。

```python
def backward(self):
    topo, seen = [], set()
    def build(v):
        if v in seen: return
        seen.add(v)
        for c in v._prev: build(c)
        topo.append(v)
    build(self)
    self.grad = 1.0
    for v in reversed(topo):     # 逆拓扑序
        v._backward()
```

**为什么反向大约是前向的 2 倍算力。**每层要算两个矩阵乘：
$$\partial L/\partial X$$（传给上一层）和 $$\partial L/\partial W$$（更新本层）。
所以前向+反向 ≈ 3× 前向；开了梯度检查点则是 4×。

#### 自测 · A1.4

**Q A1.4.1** — 手写 autograd 时，为什么是 `self.grad += ...` 而不是 `=`？

因为一个节点在图里可能**被用不止一次**。它会从每个消费者那里收到梯度，
而 `=` 会静默地丢掉除最后一份之外的所有梯度。这一个字符是凭记忆手写 micrograd 时最常见的 bug，
而且它只在有复用子项的表达式上才暴露——所以测试用例里必须有一个这样的表达式。

> **追问**
> - *为什么是逆拓扑序？* → 它保证你调用某个节点的 `_backward` 时，
>   这个节点的所有消费者都已经把自己那份贡献完了。
> - *PyTorch 为什么默认往 `.grad` 里累加？* → 同样的理由，另外它让跨 micro-batch 的
>   梯度累积变成免费的。这也是你必须调 `zero_grad()` 的原因。


**Q A1.4.2** — 为什么反向传播的 FLOPs 大约是前向的 2 倍？

每一层都要做两次和前向同样规模的矩阵乘：

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z}W^\top \quad\text{(pass upstream)}$$
$$\frac{\partial L}{\partial W} = X^\top\frac{\partial L}{\partial Z}\quad\text{(update this layer)}$$

所以前向 + 反向 $$\approx 3\times$$ 前向。$$6ND$$ 就是这么来的：每 token 前向 $$2N$$，
反向 $$4N$$。


---

<a id="a1-5"></a>
### A1.5 优化器

**SGD** $$\theta \leftarrow \theta - \alpha g$$。简单，但对病态曲率（condition number 大）会
来回震荡。

**Momentum** 累积历史梯度方向，抑制震荡、加速一致方向：

$$v_t=\beta v_{t-1}+g_t,\qquad \theta_t=\theta_{t-1}-\alpha v_t$$

**Adam** 同时维护一阶矩（方向）和二阶矩（每坐标步长）：

$$m_t = \beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad v_t = \beta_2 v_{t-1}+(1-\beta_2)g_t^2$$

$$\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_t=\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

偏差修正是因为 $$m_0=v_0=0$$，早期估计偏向零。

**AdamW** 把 weight decay 从梯度里拿出来，直接作用在权重上：

$$\theta_t=\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}-\alpha\lambda\theta_{t-1}$$

> **显存账。**Adam 每参数两份 fp32 状态 = 8 字节，占混合精度训练 16 字节/参数预算的**一半**。
> 这就是 ZeRO 存在的全部理由。
>
> **LLM 常用超参：**$$\beta_1=0.9$$，$$\beta_2=0.95$$（低于 0.999 默认值，因为长程二阶矩会陈旧），
> weight decay 0.1。

#### 自测 · A1.5

**Q A1.5.1** — 相对 Adam，AdamW 究竟改了什么？这为什么重要？

在 Adam 里，L2 正则是加进梯度的，于是它和别的东西一样要过**同一套自适应缩放**。
最终的有效衰减量与梯度近期幅度成*反*比——梯度小的参数被狠狠衰减，梯度大的几乎不衰减。
这跟任何人说 weight decay 时想表达的意思都不一样。

AdamW 把衰减直接作用在权重上，放在自适应那一步**之外**，
从而恢复了原本想要的、一致地把参数往零拉的效果。

> **追问**
> - *有更新的东西吗？* → Muon 用 Newton-Schulz 迭代把二维参数的动量更新正交化，
>   在 LLM 规模上有收益报告。
>
> **陷阱**
> - 说 AdamW"就是 Adam 加 weight decay"。Adam 也能加，区别在于**加在哪里**。


**Q A1.5.2** — Adam 为什么需要偏差修正？

$$m_0 = v_0 = 0$$，所以早期估计偏向零——第 1 步时 $$m_1 = (1-\beta_1)g_1$$，
在 $$\beta_1=0.9$$ 下只有真实梯度的 10%。除以 $$1-\beta_1^t$$ 就修正了这一点，
而且这个修正会随 $$t$$ 增大衰减到 1。

> **追问**
> - *不修正会怎样？* → 最初几百次迭代步长极小。它和 warmup 有交互：
>   两者都在处理训练早期的不稳定，只是从不同方向下手。


---

<a id="a1-6"></a>
### A1.6 学习率调度

**Warmup。**Adam 的 $$\hat v$$ 在最初几百步样本太少、估计噪声大，自适应分母不可靠，有效步长可能
极大。Warmup 让你在估计稳定前保持小步。典型取总步数的 1–2%。

**Cosine decay。**早期大步快速穿过差区域，后期小步收敛。用 cosine 而不是线性或阶梯，主要是经验结论。

> **一个会咬人的约束。**Cosine 是**对固定总步数**定义的。如果训到一半决定多训，你没法简单地延长
> ——学习率已经衰减下去了。这正是 **WSD**（warmup-stable-decay）流行的原因：常数段上可以随时
> 岔出一个衰减段，这让 midtraining 和继续预训练成为可重复的操作，而不是开跑前就焊死的决定。

#### 自测 · A1.6

**Q A1.6.1** — 为什么要 warmup？pre-LN 是不是让它变得不必要了？

Warmup 存在的理由是 Adam 的二阶矩估计在早期不可靠，有效步长可能大得离谱。

Pre-LN 削掉的是**架构层面**的需求（post-LN 把归一化放在残差路径上，梯度每层都被重新缩放，
深模型必须精调 warmup）。但**优化器状态**那条理由依然成立，所以实际的 run 还是都用 warmup。


**Q A1.6.2** — 为什么 WSD 取代 cosine 流行了起来？

Cosine 是对着一个固定总步数定义的，所以在第 0 步你就把整条 schedule 焊死了。
WSD 保留一个恒定的"stable"段，随时可以从任意一点岔出一个衰减段——这让 midtraining、
继续预训练、以及"还在涨就多训一会儿"全都变得可操作。

> **追问**
> - *为什么衰减段这么关键？* → 最后那段衰减里看到的数据，对最终权重的影响大得不成比例。
>   所以最好的数据要留到最后。


---

<a id="a1-7"></a>
### A1.7 归一化

**为什么不用 BatchNorm**（三个理由，面试要给不止一个）：

1. 序列长度可变，batch 统计量算在参差不齐的位置集合上；
2. batch 统计量**耦合了同批样本**，破坏 batch=1 的自回归生成；
3. 分布式训练下每次前向都要跨设备同步。

**LayerNorm** 在单个 token 的特征向量内部归一化，与 batch 组成无关：

$$\text{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$$

**RMSNorm** 去掉减均值和 bias：

$$\text{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\tfrac1D\sum_i x_i^2+\epsilon}}$$

消融显示**重新缩放**在起作用、**重新中心化**基本没用，而去掉它少了一次特征维归约——在 80 层
每层两次的场景下有意义。

**$$\gamma$$ 是干什么的。**归一化强制单位 RMS，这销毁了学到的尺度信息。$$\gamma$$ 把逐维度的
幅度控制还回来：$$\gamma_i>1$$ 放大，$$\gamma_i\approx 0$$ 杀掉该维度。

**Pre-LN vs post-LN。**Pre-LN 归一化子层**输入**，残差流保持一条干净的恒等通路，去掉了 warmup 的
架构性需求。代价是残差流幅度随深度增长，所以输出头前需要一个 final norm。

#### 自测 · A1.7

**Q A1.7.1** — 给出三个 Transformer 用 LayerNorm 而不用 BatchNorm 的理由。

序列长度可变；batch 内样本被互相耦合（破坏 batch=1 的生成）；分布式训练下要跨设备同步。
完整形式见上文。

> **陷阱**
> - 只给一个理由。给三个。


**Q A1.7.2** — bf16 下，RMSNorm 的哪一部分必须留在 fp32，为什么？

是**归约**那一步——特征维上的均方。在 bf16 里把 $$D$$ 个平方值加起来，舍入误差累积得很难看。
用 fp32 算完再把结果转回去，这就是为什么实现的最后都跟着一句 `.type_as(x)`。

> **追问**
> - *还有什么必须留在 fp32？* → softmax 的分母、loss 累加、梯度 all-reduce。
>   规则就一条：归约用 fp32，逐元素运算用低精度。


---

<a id="a1-8"></a>
### A1.8 泛化、正则化与 double descent

**Bias-variance 分解**（平方损失）：

$$\mathbb E[(y-\hat f)^2]=\underbrace{(\mathbb E[\hat f]-f)^2}_{\text{bias}^2}+\underbrace{\operatorname{Var}[\hat f]}_{\text{variance}}+\sigma^2$$

经典图景：容量↑ → bias↓、variance↑ → 测试误差 U 形，取谷底。

**为什么它对 LLM 不再是全部。**现代网络训练到远超插值阈值（训练误差为零），测试误差**继续下降**
——**double descent**。经典 U 形只是第一段下降，插值点之后还有第二段。所以"模型越大越过拟合"
在 LLM 规模上根本不是观察到的现象。

![double descent](/assets/img/blog/interview-knowledge/qa5_double_descent.png)
*经典 U 形只是第一段下降。插值阈值之后测试误差再次下降，而 LLM 全都活在右半边。*

诚实的说法是：这个分解仍然**正确**，但不再**有预测力**，因为 SGD 和架构带来的隐式正则化在做
这个框架没有建模的事。

**LLM 预训练里实际的正则化**主要是数据规模和 weight decay。Dropout 基本从预训练里消失了
（数据充足时它有害）。

#### 自测 · A1.8

**Q A1.8.1** — bias-variance 能解释为什么 LLM 越大越好吗？

不能——经典 U 形曲线预测的恰恰相反。实际观察到的是 **double descent**：越过插值阈值之后，
测试误差再次下降。这个分解仍然正确，但没有预测力，因为来自 SGD 和架构的隐式正则化
在做这个框架没有建模的事。

> **追问**
> - *LLM 到底会不会过拟合？* → 会，在重复数据上会。预训练很少过拟合，是因为它接近单 epoch，
>   而且语料大到模型背不下来。


**Q A1.8.2** — dropout 为什么基本从 LLM 预训练里消失了？

Dropout 是给数据稀缺场景准备的正则化手段。预训练数据充裕、接近单 epoch，
没有多少过拟合需要防——而 dropout 要付出容量和吞吐的代价。
在小数据集上做微调时它仍然会出现。

> **追问**
> - *什么是 inverted dropout？* → 在**训练**时就按 $$1/(1-p)$$ 缩放，这样推理时不必再缩放。
>   所有框架都这么做，也正因如此 `model.eval()` 直接把它关掉就行。


---

<a id="a1-9"></a>
### A1.9 损失函数与信息论

$$\operatorname{CE}(p,q)=-\sum_x p(x)\log q(x),\qquad
\operatorname{KL}(p\,\|\,q)=\sum_x p(x)\log\frac{p(x)}{q(x)},\qquad
H(p)=-\sum_x p(x)\log p(x)$$

**三者的关系**（两行可证）：

$$\operatorname{CE}(p,q)=\operatorname{KL}(p\,\|\,q)+H(p)$$

**对 LM 训练的意义。**目标是 one-hot，所以 $$H(p)=0$$，交叉熵**就是** KL 散度；而且它退化成
下一个 token 的负对数似然：

$$\mathcal L=-\sum_{t=1}^{T}\log p(x_t\mid x_{<t})$$

**forward vs reverse KL** —— 这是最经典的一道题，区别在于**无穷惩罚坐在哪一边**：

| | 权重 | 行为 | 用在哪 |
|---|---|---|---|
| Forward $$\operatorname{KL}(p\|q)$$ | 按 $$p$$ | **mean-covering**：$$q$$ 必须覆盖 $$p$$ 的全部支撑集，在多模态间摊开 | 极大似然 |
| Reverse $$\operatorname{KL}(q\|p)$$ | 按 $$q$$ | **mode-seeking**：忽略某个模态不受惩罚，塌到一个模态 | 变分推断、RLHF 的 KL 惩罚 |

#### 自测 · A1.9

**Q A1.9.1** — 证明 $$\operatorname{CE}(p,q) = \operatorname{KL}(p\|q) + H(p)$$。

$$\operatorname{KL}(p\,\|\,q) = \sum_x p(x)\log p(x) - \sum_x p(x)\log q(x) = -H(p) + \operatorname{CE}(p,q)$$

移项即得。两行。

> **追问**
> - *KL 是距离吗？* → 不是。不对称，也不满足三角不等式。


**Q A1.9.2** — 为什么 forward KL 是 mean-covering，reverse KL 是 mode-seeking？

Forward KL 按 $$p$$ 加权，所以只要哪里 $$p$$ 有质量而 $$q$$ 没有，$$\log(p/q)\to\infty$$，
代价大到无法承受——$$q$$ 被迫覆盖 $$p$$ 的全部支撑集，在多个模态之间摊开。

Reverse KL 按 $$q$$ 加权，所以 $$q$$ 会因为把质量放到 $$p$$ 没有质量的地方而受罚，
但整个忽略掉一个模态**一分不罚**（那些区域 $$q\approx0$$，对求和的贡献 $$\approx 0$$）。
于是它塌到一个模态上，并把这一个做好。

> **陷阱**
> - 答反。这是 Sapora 说自己答错、事后哭了一场的那道题——而她在两篇论文里都处理过它。
>   **已经会的东西也要排练。**


---

<a id="a1-10"></a>
### A1.10 数值稳定性

三件要盯的事：$$e^x$$ 对大 $$x$$ 溢出（fp32 约在 $$x\approx89$$）；$$\log x$$ 对接近 0 的 $$x$$
下溢；$$\log x$$ 对接近 1 的 $$x$$ 掉精度。

**softmax** 利用平移不变性：

$$\text{softmax}(x)_i=\frac{e^{x_i-c}e^c}{\sum_j e^{x_j-c}e^c}=\text{softmax}(x-c)_i$$

取 $$c=x_\max$$，最大的指数变成 $$e^0=1$$。

**log-softmax** 不要写成 `log(softmax(x))`（小概率取对数不稳），用

$$\log\text{softmax}(x)_i=x_i-\text{logsumexp}(x)$$

**logsumexp** 同样的技巧：$$\log\sum_i e^{x_i}=x_\max+\log\sum_i e^{x_i-x_\max}$$

> **同一个递推会再出现一次。**FlashAttention 的 online softmax 携带一个 running max，
> 每个 block 按 $$e^{m_\text{old}-m_\text{new}}$$ 重缩放——就是这个平移不变性，增量地用。

#### 自测 · A1.10

**Q A1.10.1** — `F.cross_entropy` 为什么收 logits 而不是概率？

这样它内部才能走数值稳定的路径：logits → logsumexp → gather，全程不出现
"先显式 softmax 再取 log"。传概率进去就逼着它对小 $$p$$ 算不稳定的 `log(p)`，
同时还丢掉了减最大值那个技巧。

> **陷阱**
> - 写 `torch.log(torch.softmax(x))`。永远用 `log_softmax`。


---

<a id="a1-11"></a>
### A1.11 训练循环与调试

```python
for batch in loader:
    optimizer.zero_grad()                                     # 1
    logits = model(batch.input_ids)                           # 2
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, V),                        # 3  shift
        batch.labels[:, 1:].reshape(-1),
        ignore_index=-100)                                    # 4  mask
    loss.backward()                                           # 5
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 6
    optimizer.step()                                          # 7
    scheduler.step()                                          # 8
```

**三个真实会发生的 bug：**

1. **忘了 `zero_grad()`。**PyTorch 默认**累加**梯度。缺了它相当于在一个不断变大的 batch 上用
   陈旧梯度训练。loss 会慢慢变怪，很难发现。
2. **shift 差一位。**位置 $$t$$ 要预测 token $$t+1$$。错了要么模型看见答案（loss 快得可疑），
   要么被要求凭空预测。
3. **忘了 loss mask。**SFT 时在 prompt token 上算了 loss，或者在 padding 上。这个不会崩，
   只会静默降低质量。

> **值得主动说出来的调试动作：先用十个样本过拟合。**如果模型连十个样本都记不住，bug 在代码里，
> 不在超参里。这一个测试能隔离上面绝大部分问题。

**梯度爆炸在实践中的真实原因**（不是深度——pre-LN 加残差已经处理了）：一批坏数据、
对当前曲率过高的学习率、fp16 溢出。缓解：全局梯度范数裁剪（而且要**记录裁剪前的范数**，
它的尖峰是最早的预警）、bf16、warmup。**如果经常裁剪，说明裁剪在掩盖问题。**

#### 自测 · A1.11

**Q A1.11.1** — 讲一遍一个训练 step，并说出最常见的三个 bug。

循环见上。三个：漏了 `zero_grad()`；label 移位差一位；prompt 或 padding token 上漏了 loss mask。

> **追问**
> - *梯度累积加在哪里？* → 连续 $$k$$ 个 micro-batch 跳过 `zero_grad`/`step`，
>   并把 loss 除以 $$k$$。
> - *为什么裁剪要放在 `step()` 之前？* → 优化器消费的是 `.grad`；裁在后面等于什么都没做。


**Q A1.11.2** — loss 不降。你按什么顺序排查？

1. **先用十个样本过拟合。**如果这都做不到，bug 就在代码里。这是单项价值最高的测试，
   能隔离掉绝大多数原因。
2. `zero_grad()` 调了吗？模型在 `train()` 模式吗？
3. label 移位对不对？mask 对不对？
4. **warmup 之后**的学习率是否正常——把实际值打出来，不要看配置。
5. 梯度有没有到达所有参数？检查有没有 `None` 梯度，以及有没有什么地方把图 detach 了。
6. 数据真的打乱了吗？loader 返回的是不是你以为的东西？

> **追问**
> - *loss 是 NaN 而不是平的，清单一样吗？* → 不一样：检查输入里有没有 inf、有没有除零、
>   有没有 `log(0)`、fp16 有没有溢出，以及学习率是不是单纯就太高了。


---

<a id="a1-12"></a>
### A1.12 基础统计

面试里真正反复出现的只有一小撮。

**期望与方差**

$$\mathbb E[aX+b]=a\mathbb E[X]+b,\qquad \operatorname{Var}[aX+b]=a^2\operatorname{Var}[X]$$

$$\operatorname{Var}[X]=\mathbb E[X^2]-\mathbb E[X]^2$$

**期望的线性性**对**任意**随机变量都成立，**不需要独立**——这是它作为证明工具最有用的地方。
方差的可加性则**需要**独立（或至少不相关）。

$$\mathbb E\Big[\sum_i X_i\Big]=\sum_i\mathbb E[X_i]\quad\text{（总成立）}$$

$$\operatorname{Var}\Big[\sum_i X_i\Big]=\sum_i\operatorname{Var}[X_i]\quad\text{（需独立）}$$

**常用分布**

| 分布 | $$\mathbb E$$ | $$\operatorname{Var}$$ | 出现在哪 |
|---|---|---|---|
| Bernoulli($$p$$) | $$p$$ | $$p(1-p)$$ | 二元奖励、pass/fail |
| Binomial($$n,p$$) | $$np$$ | $$np(1-p)$$ | $$n$$ 次采样里成功几次 |
| Geometric($$p$$) | $$1/p$$ | $$(1-p)/p^2$$ | 第一次成功要几次（best-of-N） |
| Gaussian($$\mu,\sigma^2$$) | $$\mu$$ | $$\sigma^2$$ | 初始化、噪声 |

**一个反复用到的技巧：指示变量平方。**对二元变量 $$X\in\{0,1\}$$，$$X^2=X$$，
所以 $$\mathbb E[X^2]=\mathbb E[X]=p$$，于是 $$\operatorname{Var}=p-p^2=p(1-p)$$。

> **为什么这条在 RL 里重要。**二元奖励的方差是 $$\hat p(1-\hat p)$$，在 $$\hat p=0.5$$ 处最大、
> 在两端为零。这就是"总对"和"总错"的任务都不产生梯度的数学来源（见 A12.3）。

#### 自测 · A1.12

**Q A1.12.1** — 一个模型以概率 $$p$$ 解出某个任务。你采样 $$n$$ 次取 best-of-$$n$$。
至少成功一次的概率是多少？第一次成功之前的期望采样次数是多少？

至少成功一次：$$1-(1-p)^n$$。第一次成功前的期望采样次数：$$1/p$$（几何分布）。

取 $$p = 0.1$$：best-of-10 的成功概率是 $$1-0.9^{10} = 65\%$$，而第一次成功预期要采 10 次。
这就是 test-time scaling 背后的全部算术——同时它也把收益递减摆在了台面上：
从 $$n=10$$ 走到 $$n=100$$，也不过是从 65% 到 99.997%。

> **追问**
> - *为什么 best-of-$$n$$ 需要一个 verifier？* → 没有它你根本不知道 $$n$$ 个采样里哪个成功了，
>   那个概率也就没有意义。test-time scaling 的瓶颈几乎总是在排序，而不是在采样。

---

<a id="a1-13"></a>
### A1.13 采样中的梯度流

**问题。**你想对一个**离散采样**的结果做反向传播，但 $$z\sim\text{Categorical}(p_\theta)$$
这一步不可导——采样把梯度切断了。这在 MoE 路由、离散 latent、以及任何"选一个"的场景里都会遇到。

**三种解法：**

**1. REINFORCE / score function estimator**（就是 A4.2 的 policy gradient）

$$\nabla_\theta\mathbb E_{z\sim p_\theta}[f(z)]=\mathbb E_{z\sim p_\theta}\big[f(z)\nabla_\theta\log p_\theta(z)\big]$$

无偏，但**方差很大**。适用于 $$f$$ 不可导甚至是黑盒的情况（比如 $$f$$ 是一个 verifier）。

**2. 重参数化（reparameterization trick）**

把随机性挪到一个与参数无关的噪声源上。对高斯：

$$z=\mu_\theta+\sigma_\theta\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I)$$

现在 $$z$$ 对 $$\theta$$ 可导。方差低得多，但**只适用于连续分布**——这就是 VAE 能这么训的原因。

**3. Gumbel-Softmax / Concrete**

对离散变量的连续松弛。用 Gumbel 噪声加到 logits 上再取 softmax（温度 $$\tau$$）：

$$y_i=\frac{\exp((\log p_i+g_i)/\tau)}{\sum_j\exp((\log p_j+g_j)/\tau)},\qquad g_i\sim\text{Gumbel}(0,1)$$

$$\tau\to0$$ 时逼近 one-hot 采样，$$\tau$$ 大时平滑可导。**有偏但低方差**。

**Straight-through estimator (STE)。**前向用硬的 argmax（保证离散），反向假装它是恒等函数
（或用 softmax 的梯度）。这就是量化训练（QAT）里穿过 round 操作的做法。

> **一句话选择指南。**$$f$$ 不可导或是黑盒 → REINFORCE；分布连续 → 重参数化；
> 分布离散但你能松弛它 → Gumbel-Softmax 或 STE。

#### 自测 · A1.13

**Q A1.13.1** — 为什么重参数化技巧不能用在类别分布上？

重参数化要求把采样结果写成参数和一个与参数无关的噪声源的**可导**函数。对类别分布，
采样结果是一个离散下标——任何这样的表达式都逃不掉 argmax 或阶跃函数，
而它们的导数几乎处处为零、在跳变点无定义。从 $$\theta$$ 到采样下标之间没有一条光滑通路。

Gumbel-Softmax 的绕法是**松弛**输出：它返回的不是下标，而是一个接近 one-hot 的连续向量，
因此可导。代价是引入了偏差。

> **追问**
> - *STE 在 LLM 这边出现在哪？* → 量化感知训练：前向 round 到 INT8，
>   反向就当它是恒等函数把梯度放过去。一些 MoE router 里也有。
> - *方差那么大，REINFORCE 有值得优先用的时候吗？* → 有，只要奖励是个黑盒——
>   一个单元测试、一个编译器、一个人。这正是 RLHF/RLVR 的场景。

---

<a id="a1-14"></a>
### A1.14 理论 CS 的几个常用件

在 ML 面试里出现的理论 CS 很窄，基本就这几样。

**复杂度记号。**$$O$$ 是上界、$$\Omega$$ 是下界、$$\Theta$$ 是紧界。注意 attention 是
$$O(n^2 d)$$ —— 说 $$O(n^2)$$ 时把 $$d$$ 藏起来了，长上下文下这没问题，但比较不同 $$d$$ 时会误导。

**摊还分析。**动态数组 push 的最坏单次是 $$O(n)$$，但**摊还** $$O(1)$$。KV cache 的分页分配是同一个
道理：单次扩容要分配新 block，摊还下来是常数。

**动态规划 = 记忆化 + 最优子结构。**beam search 不是 DP（它是贪心的近似），而 Viterbi 是。

**分治与主定理。**$$T(n)=aT(n/b)+f(n)$$。在集合通信里它适用于**递归折半/倍增**和树形
all-reduce（每步问题规模减半）。

**注意 ring all-reduce 不属于这一类**——它是线性流水线，不是分治，主定理套不上去。
直接数就行：$$2(p-1)$$ 步、每步传 $$N/p$$，所以每张卡收发 $$2N(p-1)/p\approx 2N$$，
**与设备数几乎无关**，这是它能扩展的全部原因。

#### 自测 · A1.14

**Q A1.14.1** — 为什么 ring all-reduce 在带宽上是最优的？

每张卡总共收发 $$2N(p-1)/p$$ 字节，$$p$$ 增大时趋于 $$2N$$ ——**与设备数无关**。
all-reduce 的下界正是 $$2N(p-1)/p$$，ring 达到了它。

代价是**延迟**：它要走 $$2(p-1)$$ 个串行步骤，所以张量又多又小时，每步的固定开销占主导。
这就是为什么框架在 all-reduce 之前要把梯度打包进大的扁平缓冲区，
也是为什么小消息改用基于树的算法。

> **追问**
> - *这对 ZeRO 有什么影响？* → all-reduce = reduce-scatter + all-gather，各 $$N(p-1)/p$$。
>   所以 ZeRO-2 的总带宽和 DDP 一样，却只存 $$1/p$$ 的状态。
>   ZeRO-3 多一次 all-gather，通信量约为 DDP 的 1.5 倍。

---

> **待补概念：**MLE 与 MAP、权重初始化（Xavier / Kaiming / 为什么 LLM 用 $$\mathcal N(0,0.02)$$）、
> 梯度检查点、经典模型（逻辑回归 / 决策树 / k-means / SVM）。

---

<a id="section-a2"></a>

## A2 · Transformer 架构与实现

这一节是**手写轮的主场**：causal self-attention 会被问出六种问法。Alisa 的书在这一块最深，
但她完全没写 MoE、分词、多模态和 SSM——那几块是新增的（标 ★）。

**读法：**A2.1–A2.4 是骨架，必须能闭卷重建；A2.5–A2.8 是每个现代模型都会做的选择；
A2.9–A2.13 是被追问时用来展示深度的。

---

<a id="a2-1"></a>
### A2.1 三种架构范式

在讲 attention 之前先把地图铺开，否则后面所有"为什么"都没有参照系。

| | 注意力 | 训练目标 | 擅长 |
|---|---|---|---|
| Encoder-only (BERT) | 双向 | Masked LM | 分类、检索、embedding |
| Decoder-only (GPT) | 因果 | Next-token | 生成，以及通过 prompt 覆盖的其他一切 |
| Encoder-decoder (T5) | Enc 双向，Dec 因果+交叉 | Seq2seq | 翻译、真正的序列到序列 |

**Decoder-only 胜出的三个理由：**

1. **训练效率。**每个位置都是一次监督。MLM 只 mask 约 15%，同样数据量下信号少约 6 倍。
2. **架构简单。**单栈、无交叉注意力，更容易扩展和切分。
3. **In-context learning。**Prompt 把几乎所有任务变成生成，不需要任务专用头。

> **双向注意力仍然有主场：**embedding 和检索。你编码的是一段固定输入，希望每个 token 都能
> 看到全文。现代 embedding 模型常常从 decoder-only 出发**去掉因果 mask** 再继续训练。

#### 自测 · A2.1

**Q A2.1.1** — decoder-only 为什么赢了？它在哪里仍然是错的选择？

赢的三个理由：信号密度（每个位置都被监督，而 MLM 只有约 15%）、架构简单（单栈、
无交叉注意力、更容易切分）、以及 in-context learning（prompting 让任务头变得不必要）。

不该用它的地方：embedding 和检索——你编码的是一段固定输入，要的是双向上下文。
还有源端又长又固定的真正 seq2seq 任务，encoder-decoder 可以把源编码一次，然后反复交叉注意它。

> **追问**
> - *什么是交叉注意力？* → Q 来自 decoder，K/V 来自 encoder 的输出。decoder-only 模型没有它
>   ——它的"上下文"就是同一条序列里更早的位置。
>
> **陷阱**
> - 只答"decoder-only 更简单"。训练信号密度那条更有说服力。

---

<a id="a2-2"></a>
### A2.2 一个 block 的解剖：残差流

一个 Transformer block 只有两行：

```python
x = x + self.attn(self.norm1(x))     # pre-norm, residual
x = x + self.mlp(self.norm2(x))
```

![Transformer block 的数据流](/assets/img/blog/interview-knowledge/qa2_block.png)

**残差流（residual stream）的视角**是理解整个架构最有用的一个心智模型：把 $$x$$ 看成一条
**共享总线**，从 embedding 一直通到输出。每一层从总线上**读**、算一点东西、再**写回**总线。

这个视角一下解释了好几件事：

- 一个没用的层学会写入约等于零，**而不需要学会恒等映射**——这是残差连接真正的价值。
- 100 层网络的行为更像"许多条短路径的集成"，而不是一条 100 层的深路径。
- 层与层之间通过总线通信，所以可以谈"某一层把某个特征写进了流里"。

**Pre-LN vs post-LN。**

- **Post-LN**（原始论文）：$$x \leftarrow \text{LN}(x + \text{sublayer}(x))$$。归一化**在残差路径上**，
  梯度每层都被重新缩放，深模型不精调 warmup 就训不起来。
- **Pre-LN**：$$x \leftarrow x + \text{sublayer}(\text{LN}(x))$$。残差流是从 embedding 到输出的
  一条干净恒等通路。Xiong 等人证明这正是 warmup 可以去掉的原因。

**Pre-LN 的代价**（一定要能说出来）：残差流的量级**随深度增长**，所以输出头前必须补一个
final norm；极深的 pre-LN 模型后段还可能出现表示坍塌——sandwich norm 这类变体就是为此存在的。

#### 自测 · A2.2

**Q A2.2.1** — 这个领域为什么从 post-LN 换成了 pre-LN？换来的代价是什么？

Post-LN 把归一化放在残差路径上，于是梯度每层都被重新缩放，深模型不精调 warmup 就根本训不起来。
Pre-LN 留下一条从 embedding 到输出的干净恒等通路，正是这一点让 warmup 不再是必需品。

代价是：残差流的量级随深度增长，因为每层都往里加，而没有任何东西把它缩回去。
你必须在 `lm_head` 之前补一个 final norm。极深的 pre-LN 模型还可能在后段出现表示坍塌，
sandwich norm（子层前后各归一化一次）就是冲着这个来的。

> **陷阱**
> - 只夸 pre-LN 不说代价，一定会被追问。
> - 从零写模型时忘掉最后那个 final norm。

---

<a id="a2-3"></a>
### A2.3 Self-attention 与 $$\sqrt{d_k}$$

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

**缩放因子的论证**（三步，必须能背）：取分量独立、方差为 1 的 $$q,k$$。点积是 $$d_k$$ 个
这样的乘积之和，所以

$$\operatorname{Var}[q\cdot k] = d_k,\qquad \text{std} = \sqrt{d_k}$$

$$d_k = 128$$ 时，**训练还没开始**，logits 就已经散布在约 $$\pm 11$$。这么宽的 softmax
接近 one-hot，而饱和的 softmax 梯度消失——注意力模式被冻结在初始化状态，学不动。
除以 $$\sqrt{d_k}$$ 把方差拉回 1。

**为什么是 $$d_k$$ 而不是 $$d_\text{model}$$。**点积是在**头**的维度上做的，那才是你要修正
方差的那一维。写错了模型照样能训，只是更差——所以它是个好考题。

**因果 mask** 在 softmax **之前**以加性 $$-\infty$$ 的形式加入。乘 0 是错的：被屏蔽的位置
仍然进入分母，剩下的权重不再和为 1。

#### 自测 · A2.3

**Q A2.3.1** — 解释一下那个缩放因子。没有它会坏在哪里？

初始化时 logits 的标准差是 $$\sqrt{d_k}$$，所以 $$d_k=128$$ 时，训练还没开始它们就散在
大约 $$\pm 11$$。在这么宽的 logits 上做 softmax 几乎就是 one-hot，而饱和的 softmax
梯度消失——注意力模式被冻结在初始化状态，学不动。

除以 $$\sqrt{d_k}$$ 把方差拉回 1。它和 softmax 温度是同一个旋钮：$$1/\sqrt{d_k}$$ 就是一个
让初始化时注意力熵保持在合理范围的温度。

> **追问**
> - *这个论证什么时候不再成立？* → 它假设的就是自己描述的那个初始化。权重一漂移，logits
>   又会涨起来——这正是 **QK-norm**（点积之前对 Q 和 K 做 RMSNorm）在大规模下要处理的事。
>
> **陷阱**
> - 用 $$\sqrt{d_\text{model}}$$。
> - 只说"归一化"，说不出方差量级和 softmax 饱和这两步。

**Q A2.3.2** — 因果 mask 为什么是 softmax 之前的加性 mask，而不是之后的乘性 mask？

因为 softmax 是在整行上归一化的。如果你在 softmax **之后**把被屏蔽的位置清零，
那些位置仍然进了分母，剩下的权重就不再和为 1——每一行都被按被屏蔽的比例静默地缩小，
而且缩放系数还随位置变化。

在 softmax 之前加 $$-\infty$$，$$e^{-\infty}=0$$ 对分母毫无贡献，
剩下的权重才是一个正经的分布。

---

<a id="a2-4"></a>
### A2.4 手写实现

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop, self.resid_drop = nn.Dropout(dropout), nn.Dropout(dropout)
        self.register_buffer("mask",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)   # (B, nh, T, hd)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))
```

**四个坑，全在这二十行里：**

1. **`.contiguous()`** — `transpose` 之后是 stride 不连续的 view，`.view()` 会报错。
   用 `.reshape()` 也行（必要时复制），但要能说出区别。
2. **除以 `sqrt(d_head)`** 而不是 `sqrt(d_model)`。
3. **`-inf` 加性 mask 在 softmax 之前**（理由见 A2.3）。
4. **QKV 融成一次投影**。三次 `nn.Linear` 数学等价但更慢——同样 FLOPs 下一次 GEMM 胜过三次。

**形状纪律。**把形状编进变量名（Shazeer 的 shape-suffix 约定）：`x_BTC`、`q_BHTD`。
在一个失败模式是"静默转置"的场合里，bug 会在调用处就暴露，而不是三行之后。

**Mask 约定。**`tensor.masked_fill(mask, value)` 填的是 mask 为 **True** 的位置。
所以如果你的 mask 用 True 表示**允许**，需要写 `masked_fill(~mask, -inf)`。
一半的 mask bug 都是这个取反。

**三行验证因果性**（也是调试方法）：

```python
y1 = model(x)
x2 = x.clone(); x2[:, -1, :] += 10.0
assert torch.allclose(y1[:, :-1], model(x2)[:, :-1])   # 过去看不到未来
```

#### 自测 · A2.4

**Q A2.4.1** — 从零手写多头因果自注意力。不许用 `nn.MultiheadAttention`。

（代码见上。）被考的四件事：transpose 之后的 `.contiguous()`、用
$$\sqrt{d_\text{head}}$$ 而不是 $$\sqrt{d_\text{model}}$$ 缩放、softmax 之前的加性 $$-\infty$$
mask、以及把 QKV 融成一次投影。

然后把因果性检查主动说出来——扰动最后一个 token，断言更早的输出不变。
不等人问就先给出测试，是一个很强的信号。

> **追问**
> - *`register_buffer` 和 `nn.Parameter` 的区别？* → buffer 会跟着 `.to(device)` 走，
>   也会存进 state dict，但不接收梯度。
> - *dropout 放在哪？* → attention softmax 之后，以及残差分支的输出上。现代预训练里
>   几乎总是 0。
> - *view 和 reshape 的区别？* → `view` 要求内存连续，从不复制；`reshape` 在不得不复制时
>   会退化成复制。

---

<a id="a2-5"></a>
### A2.5 注意力变体：MHA → MQA → GQA → MLA

唯一的驱动力是 **KV cache 大小**：每 token $$2LKH$$ 个**元素**，再乘每元素字节数。

| 变体 | KV 头数 | Cache (70B, bf16) | 权衡 |
|---|---|---|---|
| MHA | $$N$$ = 64 | 2,560 KiB/token | 质量最好，cache 承担不起 |
| MQA | 1 | 40 KiB/token | 省 64×，质量有可测损失 |
| GQA | 8 | 320 KiB/token | 省 8×，损失可忽略 |
| MLA | latent 512+64 | 90 KiB/token | DeepSeek 报告**优于** MHA |

**GQA** 把 query 头分组，组内共享一个 K/V 头。实现是一行：`k.repeat_interleave(n_rep, dim=1)`。

**为什么 GQA 赢过 MQA。**MQA 只留一个共享 KV 头，瓶颈过窄，质量下降且训练不够稳。
GQA 给了一个可调旋钮，拿到大部分收益。

**为什么 DeepSeek 选 MLA。**他们的消融里 GQA 略**差**于 MHA，而 MLA 略**优**——
这是少见的"不是权衡"的优化。MLA 把 K/V 压成低秩 latent 再缓存，外加一个小的解耦 RoPE key。

#### 自测 · A2.5

**Q A2.5.1** — 把注意力变体过一遍。各自在换什么，为什么最后是 GQA 赢了？

围绕一个驱动力来讲：**KV cache 大小**，每 token $$2LKH$$ 个*元素*——bf16 下就是
$$2LKH\times2$$ 字节——它决定了并发数和上下文长度的上限。其余一切都是它的推论。

**MHA** 给每个 query 头自己的 K/V——质量最好，但 70B 模型每 token 要 2,560 KiB，
长上下文下承担不起。**MQA** 塌成一个共享 KV 头，省 64 倍，但瓶颈掐得太紧：
质量有可测的损失，训练也不够稳。**GQA** 把 query 头分组，每组共享一个 K/V 头——
省 8 倍而损失可忽略，更关键的是它是一个**可调的旋钮**，而不是全有或全无的选择。
正是这份可调性让它赢了。

**MLA** 走的是另一个轴：不共享 K/V，而是把它们投影进一个低秩 latent、再按头重建，
外加一个小的解耦 RoPE key。每个头仍然有自己的 K/V，只是都从一份共享的压缩表示里导出。
DeepSeek 的消融显示它比 MHA 略**好**，而不只是更便宜——这是少见的、不构成权衡的优化。

**Q A2.5.2** — GQA 能省 FLOPs 吗？

**注意力计算那部分不省。**K/V 在矩阵乘之前会被扩展回 $$N$$ 个头，所以 $$QK^\top$$ 和
$$AV$$ 的 FLOPs 完全一样。（被追问时要说准：K/V 的*投影*确实变小了，每层从 $$2D^2$$
降到 $$2DKH$$——这是实打实的，但在总量里占比很小。）GQA 买到的是**显存和带宽**——
KV cache 按分组倍数缩小，而解码是访存带宽受限的，这就直接变成吞吐。

这个区分反复绊倒人，说准了就是一个明确的信号：你真的想过解码时间到底花在哪里。

> **追问**
> - *怎么把一个 MHA 的 checkpoint 转成 GQA？* → "Uptraining"：把每组内的 K/V 头做均值池化
>   来初始化，然后用原预算的一小部分继续训练。
> - *MLA 为什么需要解耦的 RoPE key？* → RoPE 是位置相关的，而 latent 只缓存一次，
>   所以那个旋转折不进压缩里。你得单独留一个携带位置的小 key。
>
> **陷阱**
> - 说 GQA 省 FLOPs。
> - 算 cache 时用 query 头数而不是 KV 头数。

---

<a id="a2-6"></a>
### A2.6 位置编码：RoPE

**要求。**找一个 $$f$$，使得变换后的 query 和 key 的内积只依赖相对位置：

$$\langle f(\mathbf q, m), f(\mathbf k, n)\rangle = g(\mathbf q,\mathbf k, n-m)$$

**RoPE 的解**是旋转：$$f(\mathbf x, m) = R_{m\theta}\mathbf x$$，二维块上

$$R_\theta = \begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}$$

**证明**——三行，值得背，因为它短：

$$\langle R_{m\theta}\mathbf q, R_{n\theta}\mathbf k\rangle
= \mathbf q^\top R_{m\theta}^\top R_{n\theta}\mathbf k
= \mathbf q^\top R_{-m\theta} R_{n\theta}\mathbf k
= \mathbf q^\top R_{(n-m)\theta}\mathbf k$$

用到 $$R_\alpha^\top = R_{-\alpha}$$ 和 $$R_\alpha R_\beta = R_{\alpha+\beta}$$。

每个坐标对 $$(2i, 2i+1)$$ 以自己的频率 $$\theta_i = \text{base}^{-2i/d}$$ 旋转，
所以不同的对编码不同波长。

**实现**（实际形式，不做 2×2 矩阵乘）：

```python
x1, x2 = x[..., 0::2], x[..., 1::2]
rx1 = x1 * cos - x2 * sin
rx2 = x1 * sin + x2 * cos
out = torch.stack([rx1, rx2], dim=-1).flatten(-2)
```

#### 自测 · A2.6

**Q A2.6.1** — attention 为什么需要位置信息？

因为 attention 是**置换等变**的。某个位置的输出是对 value 的加权求和，而权重来自
根本不涉及位置的点积——打乱输入 token，输出就以完全相同的方式被打乱。
没有位置信息，"狗咬人"和"人咬狗"会得到同一组表示。

因果 mask 确实注入了*一部分*顺序信息（token $$t$$ 看到的前缀和 token $$t+1$$ 不一样），
这也是为什么 decoder-only 模型在缺少显式位置编码时比 encoder 模型退化得更温和——
但它远弱于"知道两个 token 之间的实际距离"，所以每个生产模型都会显式编码位置。

> **追问**
> - *为什么不直接把下标拼进去？* → 泛化不了：模型得在原始下标上学算术，
>   而没见过的更大下标属于分布外。正弦和旋转给的是一个光滑、有界、有结构的表示。

**Q A2.6.2** — 证明 RoPE 让 attention logits 只依赖相对位置。

（三行证明见上。）关键事实是 $$R_\alpha^\top = R_{-\alpha}$$ 和
$$R_\alpha R_\beta = R_{\alpha+\beta}$$——旋转矩阵的转置就是它的逆，而旋转是加性复合的。
剩下的都顺着推出来。

> **追问**
> - *加在什么上面？* → **只有 Q 和 K**，在分头之后、点积之前。绝不加在 V 上——V 携带的是
>   内容，不是位置。
> - *和 KV cache 怎么配合？* → 缓存**旋转之后**的 key。

**Q A2.6.3** — RoPE 超出训练上下文之后为什么外推很差，怎么修？

低频分量在训练里连一整圈都转不完，所以推理时序列一变长，模型就被要求解释
**它从没见过的角度**。高频分量没问题（它们绕了很多圈）；携带长程位置信息的是那些长波长分量，
坏掉的恰恰就是它们。

各种修法的共同点都是把角度留在训练过的范围里：**位置插值**把位置按比例压小，
让长度 $$2L$$ 映射进 $$[0,L]$$；**NTK-aware 缩放**改 RoPE 的 base，让高频几乎不受影响、
低频被压得更多；**YaRN** 把频率相关的插值和一个注意力温度修正结合起来。
所有这些都需要在目标长度上做一次短微调。

> **陷阱**
> - 说 RoPE 也加在 V 上。
> - 说"RoPE 天然可以外推"。它天然是**相对**的，不等于可外推。

---

<a id="a2-7"></a>
### A2.7 FFN 与 SwiGLU

$$\text{FFN}(x) = \big(\text{Swish}(xW_\text{gate}) \odot xW_\text{up}\big)W_\text{down}$$

其中 $$\text{Swish}(x) = x\cdot\sigma(x)$$。

**三个矩阵，不是两个。**普通 FFN 是 $$\text{ReLU}(xW_1)W_2$$，$$F=4D$$，参数 $$2\cdot 4D^2=8D^2$$。
SwiGLU 多一个 gate 投影，参数是 $$3DF$$。要保持参数量不变：

$$3DF = 8D^2 \implies F = \tfrac{8}{3}D$$

**gate 在做什么。**$$W_\text{up}$$ 产生内容，$$W_\text{gate}$$ 产生一个自己算出来的门。
两条路都贡献内容——模型得到的是一个被自身"置信度"调制过的表示。

#### 自测 · A2.7

**Q A2.7.1** — SwiGLU 的中间维为什么是 $$\tfrac83 D$$ 而不是 $$4D$$？

为了让参数量和两矩阵的 baseline 持平。门控 FFN 有三个矩阵（$$3DF$$ 个参数），
经典版是两个（$$F=4D$$ 时 $$8D^2$$），配平就得到 $$F = \tfrac83 D$$。
这样架构之间的比较才是等参数的，消融也才有意义。

> **追问**
> - *为什么参数量是 FFN 主导？* → 在 $$F/D = 8/3$$ 下 FFN 每层是 $$8D^2$$，
>   而 attention 是 $$4D^2$$，GQA 还会把 attention 再压小。Llama-3-70B 里 FFN 占每层的 **82%**。
> - *门控为什么有用，有理论吗？* → 基本没有。Shazeer 自己的论文说这些架构
>   "把成功归于神的恩典"。这是纯经验的。
>
> **陷阱**
> - FFN 写成两个矩阵。SwiGLU 是三个。

---

<a id="a2-8"></a>
### A2.8 ★ Mixture of Experts

**想法。**把 FFN 换成 $$E$$ 个专家 FFN 加一个 router。每个 token 送到 top-$$k$$ 个专家
（$$k$$ 从 1 到 8 都有：Switch 和 Mixtral 是 1–2，DeepSeek-V3 和 Qwen3-235B 激活 8 个路由专家），
于是**参数随 $$E$$ 增长而每 token 的 FLOPs 基本不变**。
这就是全部意义：容量与算力解耦。

```python
gates = F.softmax(x @ W_router, dim=-1)     # (T, E)
gate, expert = gates.max(dim=-1)            # top-1
```

**容量与 token dropping。**All-to-all 通信需要固定大小的缓冲区，所以每个专家有**容量上限**。
热门专家溢出时，多出来的 token **整层跳过**，直接沿残差流通过。这就是为什么同一个 MoE 模型
在不同 batch 组成下输出会略有不同。

**Auxiliary loss。**先纠正一个流传很广的说法：**router 不是没有梯度**。门概率 $$p_e$$ 乘在
被选专家的输出上，所以语言建模损失会经由它回传到 $$W_\text{router}$$——router 正是这样学会
「哪个专家好」的。不可微的只有 top-$$k$$ 这个**选择**动作。

问题在于这个梯度会自我强化：拿到更多 token 的专家训得更快，于是 router 更偏向它，
形成富者愈富的**路由坍缩**。再加上专家容量和 expert 并行都要求负载均衡，才需要一个额外的均衡项。
Switch Transformer 的损失把"路由到每个专家的 token 比例"$$f_e$$ 乘以"该专家的平均门概率"$$p_e$$：

$$\mathcal L_\text{aux} = E\sum_{e=1}^{E} f_e \cdot p_e$$

在均匀路由处取最小值 1。

**前沿。**DeepSeek-V3 把**批级**负载均衡从损失里拿掉，改用**训练中动态调整的偏置项**，
理由是辅助损失引入的梯度在和语言建模目标对抗（见 A3.3）。注意它并非一点辅助损失都不留——
还保留了一个系数极小（$$\alpha=10^{-4}$$）的**序列级**均衡损失，防止单条序列内的极端不均衡。他们还用**共享专家**，
让公共知识不必在每个专家里重复一遍。

#### 自测 · A2.8

**Q A2.8.1** — MoE 层是怎么工作的？auxiliary loss 是干什么用的？什么是 token dropping？

一个 router 给每个 token 对 $$E$$ 个专家打分，把它送到 top-$$k$$ 个（通常 1 或 2），
于是参数随 $$E$$ 增长而每 token 的 FLOPs 基本不变。router 的输入是这个 token 在该层的隐状态，
所以路由是**上下文相关**的，不是按词表决定的。

**auxiliary loss 的存在理由不是 router 缺梯度**——它有梯度。门概率乘在被选中专家的输出上，
所以语言建模损失会回传进 router；不可微的只有 top-$$k$$ 这个选择动作。真正的问题是
这个梯度会**自我强化**：拿到更多 token 的专家训得更快，router 就更偏向它们，
路由塌到少数几个上。在这之上，专家容量和 expert 并行也都要求负载均衡，
所以才需要一个显式的均衡项。Switch 的损失
$$\mathcal L_\text{aux} = E\sum_e f_e p_e$$ 把路由到每个专家的 token 比例乘以
该专家的平均门概率，在均匀路由处取最小。

**Token dropping** 来自 all-to-all 需要固定大小的缓冲区，所以每个专家有容量上限。
热门专家溢出时，多出来的 token **整层跳过**，沿残差流直接通过。有一个值得不等人问
就主动说的后果：同一个输入，会因为同一批里还有什么而产生不同的输出。

**Q A2.8.2** — MoE 的显存和算力该怎么记账？

显存按**总**参数算——每个专家都必须驻留，因为你没法事先知道哪些 token 会路由到哪。
算力按**激活**参数算。

DeepSeek-V3 是 671B 总参数、37B 激活：按 671B 去配 GPU 规模，用 $$6ND$$ 估训练 FLOPs 时
取 $$N=37$$B。把这两个记反，是 MoE 上最常见的一个错误。

> **追问**
> - *MoE 为什么难服务？* → expert 并行意味着每个 MoE 层都要 all-to-all，而负载是数据相关的，
>   推理时很难均衡。
> - *router 的输入是什么？* → 这个 token 在该层的隐状态，所以路由是上下文相关的，
>   而不是按词表决定的。
>
> **陷阱**
> - 说 MoE"省显存"。它省的是**算力**，显存反而更大。

---

<a id="a2-9"></a>
### A2.9 ★ 分词

**BPE 训练循环。**从字节序列开始，反复：统计所有相邻对，把最频繁的一对合并成新 token，
记录这次合并。到目标词表大小停止。

```python
for i in range(num_merges):
    counts = Counter(zip(ids, ids[1:]))
    best = max(counts, key=counts.get)
    merges[best] = 256 + i
    ids = replace_pair(ids, best, 256 + i)
```

**编码时按合并的学习顺序应用**，而不是按待编码字符串里的频率。搞反了会得到一个
round-trip 不一致的 tokenizer——一个相当难查的线上 bug。

**为什么用字节而不是字符。**字节级词表能表示**任何**输入，永远不存在 OOV。代价是
非拉丁文字的每字符 token 数更多，这是一个真实的公平性和成本问题，值得主动提。

**为什么数不清 strawberry 里有几个 r。**模型从来没见过字符。`strawberry` 可能是三个 token，
而表示里没有任何东西暴露它们内部的字母。这是输入表示的产物，不是推理能力的问题。

#### 自测 · A2.9

**Q A2.9.1** — 词表大小为什么重要？把它调大要付出什么？

词表更大意味着序列更短，这在 attention 上是平方级地便宜，在其他地方是线性地便宜。
Llama 3 从 32k 提到 128k，主要就是为了多语言的 token 效率。

代价：embedding 和 unembedding 一共 $$2VD$$ 个参数；输出 softmax 更贵；
罕见 token 拿到的更新极少——**glitch token** 就是这么来的（词表里有它，
训练数据里几乎没有，于是那个 embedding 基本没被训过，拿它去 prompt 会得到离奇行为）。

> **追问**
> - *BPE、WordPiece、Unigram 的区别？* → WordPiece 按似然增益而不是原始频率来合并；
>   Unigram（SentencePiece）从一个大词表出发做*剪枝*，还能给出切分上的概率。
>   BPE 是 LLM 的默认选择。
>
> **陷阱**
> - 说 encode 时按频率应用 merge。是按**学习顺序**。

---

<a id="a2-10"></a>
### A2.10 参数都在哪里

**Llama-3-70B 的分布**：embedding 3%，attention 17%，**FFN 80%**。
值得记，因为它告诉你量化和 MoE 的收益在哪。

**Weight tying** 把 embedding 矩阵和 unembedding（`lm_head`）共享，省 $$VD$$ 参数。
理由：两者都在 token 身份和残差流之间映射，只是方向相反。

**什么时候值得。**小模型配大词表时占比很大：$$V=128256, D=2048$$ 就是 $$2.6\times10^8$$ 参数，
可能超过模型的 15%。而 70B 模型 $$D=8192$$ 时同样的 $$VD$$ 只有约 1.5%，
所以大多数大模型**不做** tying。

#### 自测 · A2.10

**Q A2.10.1** — weight tying 什么时候值得做？

取决于 $$VD$$ 在总参数里的占比。小模型配大词表时，这个共享矩阵可能超过模型的 15%，
tying 既省显存又起正则作用——通常是划算的。70B 模型上它只有约 1.5%，正则也没必要，
所以大多数大模型不做 tying。

理论上的反对意见是这两个矩阵想要的东西不一样：输入 embedding 希望**上下文**相似的 token
挨得近，输出希望**预测分布**相似的 token 挨得近。小规模下正则的收益盖过了这一点，
大规模下盖不过。

---

<a id="a2-11"></a>
### A2.11 长上下文的架构手段

**滑动窗口。**每个 token 只关注最近 $$W$$ 个。计算从 $$O(n^2)$$ 降到 $$O(nW)$$，
而更大的收益是——**KV cache 变得与序列长度无关**：token 离开窗口就可以丢弃。

**代价。**信息仍能通过层间跳跃传得更远：$$L$$ 层的感受野是 $$L\times W$$。
但那是一条**有损**通路，不是直接注意力。

**交错。**标准修法是局部层和全局层交替——大多数层用窗口，少数层用全注意力。
Gemma 和 Mistral 都做了各自的版本。你拿到大部分显存收益，同时保住真正的长程检索。

#### 自测 · A2.11

**Q A2.11.1** — 滑动窗口和 FlashAttention 是同一个思路吗？

不是，而且把两者混为一谈是个常见错误。FlashAttention 是**精确**的——它只改访存模式
（分块和重算），算出来的输出一模一样。滑动窗口**改的是模型本身**：它是一个近似，
函数类都不同。

实践上的区别：FlashAttention 随时打开都安全；滑动窗口是一个架构决策，
你必须带着它训练、并且评测它。

> **追问**
> - *那学出来的稀疏呢？* → DeepSeek Sparse Attention 学的是"该关注哪些 key"，
>   而不是套一个固定的几何模式——更灵活，但更难做快。

---

<a id="a2-12"></a>
### A2.12 ★ 多模态怎么接进来

两条路线，值得知道分界线在哪。

**1. 后接式（LLaVA 式 projector）。**冻结的视觉编码器（CLIP/SigLIP）出 patch embedding，
过一个小 projector（线性层或 MLP）映射到 LLM 的 $$D$$ 维，当成 token 拼进序列。

- 便宜——只训 projector，甚至可以只用几十万样本。
- 视觉编码器是为对比学习训的，不是为生成训的，细粒度信息（文字、计数、空间关系）容易丢。

**2. 原生多模态。**从预训练开始就混合模态，图像 token 和文本 token 一起做 next-token
预测（Gemini、GPT-4o 路线）。贵得多，但模态之间的对齐深得多，而且能**生成**图像/音频。

**关键工程问题：token 预算。**一张 $$336\times336$$ 的图在 $$14\times14$$ patch 下是 576 个 token；
高分辨率切片可以到几千。**图像很快就主导了上下文**，所以 token 压缩（Q-Former、
pooling、可变分辨率）是这一块的主要研究方向。

#### 自测 · A2.12

**Q A2.12.1** — 预算很小，你会怎么给一个纯文本 LLM 加上视觉？

冻住一个预训练视觉编码器（SigLIP 或 CLIP），加一个小 projector（线性层或两层 MLP）
把 patch embedding 映射到 LLM 的隐藏维，再把它们当成 token 插进序列。

分两阶段训：先只训 projector，用图文对建立对齐；再训 projector 加 LLM（通常上 LoRA），
用指令数据。视觉编码器一般保持冻结——它本来就够好，在小数据集上解冻只会把它训坏。

我会主要盯着 **token 预算**：一张图 576 个 token 就已经比大多数文本 prompt 还长，
高分辨率切片还要在此之上翻几倍。要不要做压缩，取决于你预期每轮对话里有多少张图。

> **追问**
> - *这条路线为什么在 OCR 和计数上很吃力？* → 冻结的编码器是用对比学习针对图像级语义训的，
>   细粒度的空间和符号细节保存得不好。常见修法是提高输入分辨率，
>   以及加大 OCR 类训练数据的比重。
> - *那直接上原生多模态呢？* → 对齐更好，而且能跨模态生成，但那是预训练量级的投入，
>   不是一次微调。

---

<a id="a2-13"></a>
### A2.13 ★ 注意力的替代品

值得知道，因为"你觉得 Transformer 会被取代吗"是常见的开放题。

![RNN / Transformer / SSM 的取舍](/assets/img/blog/interview-knowledge/qa6_architectures.png)

**核心权衡是"状态大小"。**

| | 状态 | 训练 | 推理每 token | 弱点 |
|---|---|---|---|---|
| Attention | $$O(n)$$（KV cache） | 并行 | $$O(n)$$ | 长上下文内存爆炸 |
| RNN/LSTM | $$O(1)$$ 固定 | **串行** | $$O(1)$$ | 无法并行训练、长程遗忘 |
| SSM / Mamba | $$O(1)$$ 固定 | 并行（扫描） | $$O(1)$$ | 精确回忆能力弱 |
| 线性注意力 | $$O(1)$$ 固定 | 并行 | $$O(1)$$ | 质量通常不及 softmax 注意力 |

**Mamba 解决了什么。**RNN 的致命伤是训练不能并行。SSM 用**并行扫描**恢复了训练并行性，
同时保留 $$O(1)$$ 的推理状态。Mamba 的 selective 机制让状态转移**依赖输入**，
这样模型可以决定记住什么、忘掉什么。

**为什么它没有取代 Transformer。**固定大小的状态意味着**精确回忆**必然有损。
"从 100k 上下文里找出那个电话号码"这类任务上，注意力可以直接看回去，SSM 只能依赖
它压缩进状态里的东西。所以现在主流是**混合架构**：大部分层用 Mamba，少数层用注意力，
兼顾效率和精确回忆。

#### 自测 · A2.13

**Q A2.13.1** — 状态空间模型会取代注意力吗？

权衡在状态大小，而这是根本性的，不是实现细节。注意力维持一个 $$O(n)$$ 的状态（KV cache），
因此可以精确回看任何一个 token。SSM 维持 $$O(1)$$ 的状态，所以回忆必然有损——
没被压进状态里的东西就是没了。

这恰好预测了实际观察到的现象：SSM 在语言建模损失上有竞争力，长上下文下便宜得多，
但在需要从很远处精确检索的任务上更弱——而大多数 agentic 和长文档负载要的正是这个。

所以答案大概不是取代，而是**混合**——大部分层用 SSM 做便宜的序列混合，
少数几层用注意力做精确回忆。最近已经有好几个模型是这么发的。

> **追问**
> - *Mamba 修好了 RNN 的什么？* → 训练并行性，靠的是并行扫描，同时保住 $$O(1)$$ 的推理状态。
>   selective 机制让状态转移依赖输入，于是模型自己决定留什么。
> - *线性注意力为什么更弱？* → 去掉 softmax 之后注意力矩阵是低秩的，
>   表示不了尖锐、有选择性的注意力模式。

---

> **待补概念：**cross-attention 的实现细节、ALiBi 与相对位置偏置、
> nGPT / 归一化的其他变体、diffusion language model、
> 架构搜索与"为什么这些超参是这些值"的历史。

---

<a id="section-a3"></a>

## A3 · 常见模型

★ 全新一节。它的价值不在于罗列，而在于**强迫你把架构选择和约束连起来**：
为什么 Llama 3 用 GQA 而 DeepSeek 用 MLA？为什么 DeepSeek-V3 敢去掉 auxiliary loss？

**这一节也是最高频的"你最近在关注什么"的弹药库。**被问到时，你需要能说出某个模型
**做了什么不一样的选择、以及为什么**，而不是复述参数量。

---

<a id="a3-1"></a>
### A3.1 一张对照表

| | Llama 3 70B | DeepSeek-V3 | Qwen3 | Mixtral 8×7B |
|---|---|---|---|---|
| 类型 | Dense | MoE | Dense + MoE 双线 | MoE |
| 参数 | 70B | 671B 总 / 37B 激活 | 0.6B–235B | 47B 总 / 13B 激活 |
| 注意力 | GQA (8 KV 头) | **MLA** | GQA | GQA |
| FFN | SwiGLU | DeepSeekMoE + 共享专家 | SwiGLU / MoE | 8 个专家取 top-2 |
| Norm | RMSNorm, pre-LN | RMSNorm, pre-LN | RMSNorm + **QK-norm** | RMSNorm |
| 位置 | RoPE | RoPE（解耦） | RoPE | RoPE |
| 词表 | 128k | 129k | 151k | 32k |
| 训练 token | ~15T | 14.8T | ~36T | — |
| 精度 | bf16 | **FP8 混合精度** | bf16 | bf16 |
| 值得记的一件事 | 8B 也训 15T，远超 Chinchilla | 无 aux-loss 负载均衡 + MTP | hybrid thinking 模式 | 让 MoE 进入主流 |

> **怎么用这张表。**不要背。挑三列，能就每一列说出"它做了什么不一样的选择、解决什么约束"
> 就够了。面试官想看的是你能把选择映射到约束，不是记忆力。

#### 自测 · A3.1

**Q A3.1.1** — 挑一个最近的模型，说说它有意思在哪。

好答案会挑**一个设计决策**，然后把它追溯到某个约束上。以 DeepSeek-V3 为例：

*"有意思的地方在于他们一次同时打了三种不同的成本。MLA 把 KV cache 压成一个低秩 latent——
而且少见的是，他们的消融显示它在质量上是超过 MHA 的，不只是持平，所以这不是一个权衡。
他们把 MoE 的 auxiliary loss 换成了一个在训练中调整的偏置项，因为 auxiliary loss 会引入
一个和语言建模目标对抗的梯度。还有他们用 per-tile 缩放做 FP8 训练，这是 FP8 预训练可以
稳定跑通的第一个大规模证据。"*

三个具体选择，每个都挂在一个约束上，30 秒讲完。

> **追问**
> - *为什么 FP8 要 per-tile 缩放？* → FP8 的动态范围太窄，一个全局 scale 盖不住整个张量，
>   结果要么离群值饱和，要么正常值的分辨率被压没。
>
> **陷阱**
> - 背参数表却说不出任何一个设计选择要解决什么约束。表是索引，不是答案。

---

<a id="a3-2"></a>
### A3.2 Llama 3：把 Chinchilla 扔掉

**关键决策：8B 模型训了约 15T token**，是 Chinchilla 最优点（约 160B token）的 **90 倍**。

**为什么。**Chinchilla 优化的是**训练**算力。如果模型要服务几亿次请求，**推理成本主导总成本**，
那么把一个更小的模型训得更久是理性的——它永远更便宜地服务，而额外训练成本是一次性的。

**其他值得说的：**

- **GQA 全系列使用**，包括 8B。KV cache 是长上下文可行性的前提。
- **词表 128k**（从 Llama 2 的 32k 大幅提升），主要为多语言 token 效率。
- 405B 版本用 bf16 而**没有**用 FP8——他们明确选择了保守，把稳定性放在效率之前。

#### 自测 · A3.2

**Q A3.2.1** — Llama 3 8B 训了约 15T token。这是个错误吗？

不是——这是有意换掉了"在优化什么"这个前提。Chinchilla 给的是**训练**算力最优点。
一旦把模型整个生命周期里的推理成本算进来，最优点就会强烈地往"更小、训更久"偏移：
更小的模型在每一次请求上都更便宜，而且是永远，多出来的训练成本只付一次。

这个区间有时被叫作"inference-optimal"。它的上限在数据：高质量 token 总会用完，
接下来只能重复，而大约 4 个 epoch 之后收益就塌了。

> **追问**
> - *那 Chinchilla 是错的吗？* → 不是，它是在正确地回答另一个问题。永远先问一句：
>   "最优是对训练成本而言，还是对终身总成本而言？"
>
> **陷阱**
> - 说 Llama 3「违反了 scaling law」。它没有违反，它优化的是另一个目标——终身总成本而不是训练成本。

---

<a id="a3-3"></a>
### A3.3 DeepSeek-V3 / R1：三个值得学的选择

**1. MLA（Multi-head Latent Attention）.** 把 K/V 压成一个低秩 latent（512 维）再缓存，
外加一个 64 维的**解耦 RoPE key**（因为 RoPE 是位置相关的，没法吸收进压缩）。
放到 A2.5 那张表的同一个 70B 配置下比较（$$L=80$$），每 token 约 90 KiB，
对 MHA 的 2,560 KiB 小一个多数量级。DeepSeek-V3 自己是 61 层，实际数更小。
**而且他们的消融显示 MLA 的建模质量优于 MHA**——GQA 是略差于 MHA 的，MLA 不是。

**2. 批级负载均衡不再靠辅助损失。** 传统 MoE 用辅助损失逼 router 均衡，但那个损失会
和语言建模目标抢梯度。DeepSeek 改成给每个专家一个**在训练中动态调整的偏置项**：
过载就调低、欠载就调高。均衡通过改变路由**决策**实现，而不是通过加一个对抗性的梯度。
准确说是**批级**——他们仍保留一个 $$\alpha=10^{-4}$$ 的序列级均衡损失。

**3. FP8 混合精度训练。** 用 per-tile / per-block 缩放而不是全局 scale。

**R1：RLVR 让长推理涌现。** R1-Zero 从基座模型直接做可验证奖励的 RL，**没有 SFT 冷启动**，
长链推理自己长了出来——包括"等一下，让我重新检查"这种回溯行为。这是很强的证据：
推理能力可以从奖励中被**激发**，而不必被示范。发布的 R1 仍加了冷启动 SFT，主要是为了可读性。

#### 自测 · A3.3

**Q A3.3.1** — MLA 和 GQA 都在压 KV cache，为什么 MLA 更好？

两者压缩的轴不同。**GQA 是共享**：一组 query 头共用一份 K/V 头——丢掉的是头之间的多样性，
消融里对 MHA 有一点可测的质量损失。

**MLA 是投影**：把 K/V 压进一个低秩 latent，再按头重建回来。每个头仍然有自己的 K/V，
只是都从一份共享的压缩表示里导出。因为这个投影是学出来的，它能留住真正要紧的方向——
同时它还带一点温和的正则效果，这是通常用来解释"为什么它在消融里比 MHA 还略**好**"的说法。

代价是复杂度：你需要那个解耦的 RoPE key，实现难度也明显高于一行 `repeat_interleave`。

> **追问**
> - *为什么 RoPE 不能被吸收进压缩里？* → RoPE 做的是位置相关的旋转。压缩后的 latent
>   只缓存一次、跨位置复用，位置相关的变换折不进去。所以要单独留一个携带位置的小 key。

**Q A3.3.2** — MoE 的 auxiliary loss 有什么问题？

它是**一个与主目标竞争的第二目标**。均衡损失的梯度会把 router 往均匀推，
而不管均匀路由对语言建模损失是不是好事——于是你在拿质量换均衡，
还得靠调那个系数来管理这笔交换。

DeepSeek 的做法是把**批级**均衡从梯度里拿出去：给每个专家一个偏置，按观测到的负载
在步与步之间调整，它只改变路由决策，自己不贡献任何梯度。均衡从一个优化项变成了一个控制问题。

被追问时要把范围说准——他们并没有把辅助损失全部去掉。一个 $$\alpha = 10^{-4}$$ 的
序列级均衡损失仍然保留着，用来防住单条序列内部的极端不均衡。

> **追问**
> - *什么是共享专家？* → 在路由专家之外、每个 token 都会经过的专家。公共知识放在那里，
>   就不用在每个专才里复制一遍，路由专家才能真的去做专才。
>
> **陷阱**
> - 说 MLA「就是 GQA 的一种」。GQA 共享 K/V 头，MLA 压低秩再重建，压缩的轴不同。
> - 说去掉 aux loss 是为了省算力。是为了消掉那个和语言建模对抗的梯度。

---

<a id="a3-4"></a>
### A3.4 Qwen3 与 hybrid thinking

**Hybrid thinking mode.** 同一个模型支持"思考"和"不思考"两种模式，由用户或模板切换。
思考模式产出长 CoT，非思考模式直接答。

**为什么这是个好设计。**推理是**昂贵的**，而大多数请求不需要它。把决定权交给调用方，
避免了"要么全都慢、要么全都笨"的二选一。它也顺带解决了一个产品问题：用户能看到
自己在为什么付费。

**其他：**训练 token 约 36T（比 Llama 3 还多）；dense 和 MoE 两条产品线；
**QK-norm**（对 Q 和 K 做 RMSNorm 再算点积）来稳定大模型的 attention logits。

#### 自测 · A3.4

**Q A3.4.1** — QK-norm 解决的是什么问题？

$$1/\sqrt{d_k}$$ 那套缩放论证假设 q 和 k 的分量方差为 1。这在初始化时成立，
权重一漂移就不再成立——规模一大，attention logits 会一路涨到 softmax 饱和，训练随之失稳。

QK-norm 在点积之前对 Q 和 K 各做一次 RMSNorm，直接把它们的幅度框住，
而不是依赖一个只在初始化时刻成立的论证。代价是每层多两次归一化，买到的是大规模下的稳定性。

> **追问**
> - *这和 attention entropy collapse 有关吗？* → 有。softmax 饱和就意味着注意力熵接近零，
>   这是大规模训练里有明确记录的一种失稳模式。
>
> **陷阱**
> - 把 QK-norm 和 pre-LN 的 norm 混为一谈。前者作用在 Q/K 上、专治 logits 增长，后者作用在子层输入上。

---

<a id="a3-5"></a>
### A3.5 Mixtral 与 MoE 的主流化

**Mixtral 8×7B** 是让 MoE 进入开源主流的模型：8 个专家、每 token 选 top-2，
47B 总参数但每 token 只激活约 13B。

**它教给我们的记账方式**（这一点比模型本身重要）：

- **显存按总参数算**（所有专家都要驻留）：47B
- **算力按激活参数算**（只有 2 个专家参与）：13B
- 所以它的**质量接近 47B 级别，速度接近 13B 级别，但显存需求是 47B 级别**

这个"显存贵、算力便宜"的特性决定了 MoE 的适用场景：**吞吐优先、显存充裕**的服务，
不适合边缘部署。

#### 自测 · A3.5

**Q A3.5.1** — 服务预算固定，你选 47B 的 MoE 还是 13B 的 dense？

先反问一句：这个预算是按什么计价的。

**显存受限**（单卡，或者 KV cache 占主导的长上下文）：选 13B dense，
因为不管实际激活多少，MoE 的 47B 权重都得全部驻留。

**吞吐受限、显存有富余**：选 MoE，因为你用接近 13B 的每 token 算力拿到接近 47B 的质量。

另外还有一笔服务复杂度的账：expert 并行意味着每个 MoE 层都要 all-to-all，
而负载是数据相关的，batching 和均衡都更难。

> **追问**
> - *MoE 的训练 FLOPs 怎么算？* → 还是 $$6ND$$，但 $$N$$ 取**激活**参数。
>   用总参数会按稀疏倍数高估。

---

> **待补概念：**GPT-OSS 与开源权重模型、Gemma 的滑窗交错、
> Kimi/Moonshot 的 Muon 优化器实践、闭源模型可推断的架构信息、
> 模型卡与 system card 的读法。
>
> **陷阱**
> - 用总参数量算 MoE 的训练 FLOPs。要用激活参数。

---

<a id="section-a4"></a>

## A4 · 预训练

★ 全新一节。之前预训练被拆散在三处（并行在 A5、数据在 A9、scaling law 在 A11），
但没有一节讲**预训练本身**——目标函数、完整流程、超参怎么定、训练动态该长什么样。

---

<a id="a4-1"></a>
### A4.1 训练目标：为什么是 next-token prediction

**目标函数**就是序列的负对数似然：

$$\mathcal L(\theta)=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$

因为目标分布是 one-hot，$$H(p)=0$$，所以**交叉熵就是 KL 散度**——最小化 CE 精确等于最小化到
数据分布的 KL（见 A1.9）。

**为什么它这么强。**三个理由，面试要给不止一个：

1. **信号密度最高。**每个位置都是一次监督。MLM 只 mask 约 15% 的 token，相同数据量下
   信号少约 6 倍。
2. **它是一个"压缩即理解"的目标。**要把下一个词预测好，你被迫学会语法、事实、推理、
   甚至说话人的意图——因为这些都是降低困惑度的手段。
3. **训练和使用是同一个操作。**不需要任务头，prompting 把几乎所有任务变成生成。

**变体：multi-token prediction (MTP)。**每个位置额外预测未来若干个 token。两种做法要分清：
Gloeckle 等（2024）用 4 个**独立并行头**；DeepSeek-V3 用的是深度 $$D=1$$ 的**串行**模块，
即只额外预测 1 个 token，且与主模型**共享 embedding 和输出头**、保留因果链。
两者都让信号更稠密，而且**顺手得到一个 draft 模型**用于 speculative decoding。

> **一个值得主动说的局限。**next-token 是**行为克隆**：它只学"人类写下的文本长什么样"，
> 不学"什么是对的"。模型对自己错误的恢复能力、对不确定性的表达，都不在这个目标里——
> 这正是 post-training 存在的理由。

#### 自测 · A4.1

**Q A4.1.1** — Masked language modeling 为什么输给了 next-token prediction？

主要论点是信号密度：MLM 只监督约 15% 的位置，next-token 监督 100%。
同样算力下，每 token 数据拿到的梯度信号大约多 6 倍。

还有两条：MLM 用到下游要接任务专用头，而 decoder-only 的语言模型一切都靠生成完成；
以及 MLM 的训练/使用不一致（训练时它从来不生成），这让它在真正重要的那类生成任务上很别扭。

BERT 那一类模型在 **embedding 和检索**上仍然是更对的选择——你编码的是一段固定输入，
希望每个 token 都能看到整条序列。

> **追问**
> - *两个目标能不能一起训？* → 有人做过（UL2、prefix-LM）。收益不大而复杂度是实打实的，
>   所以整个领域最后收敛到了 decoder-only。
>
> **陷阱**
> - 说 MLM「更差」。它在 embedding 和检索上仍然是更合适的目标。

**Q A4.1.2** — Multi-token prediction 买到了什么，代价是什么？

更稠密的训练信号（每个位置监督若干个未来 token）、对"这段话要往哪去"更好的内部表示，
以及一个白送的 draft 模型用于 speculative decoding。

代价取决于用哪种做法。并行头（Gloeckle 等）实打实地增加参数和算力。DeepSeek-V3 的串行模块
和主模型共享 embedding 与输出头，所以每多一层深度的边际成本只是一个 transformer block
加一次投影——而他们只用了深度 1。两种做法都要给辅助损失配权重，而且这套额外的预测结构
**通常在预训练之后就丢掉**，除非你留着做投机解码。

---

<a id="a4-2"></a>
### A4.2 从零训一个模型的顺序

一个可以背下来的清单。面试里被问"你会怎么从头训一个模型"时，按这个顺序讲。

1. **定预算。**多少 GPU、多少天 → 总 FLOPs $$C$$。这决定了后面一切。
2. **定模型和数据规模。**由 $$C$$ 和 Chinchilla（或你自己的推理成本考量）反推 $$N$$ 和 $$D$$。
3. **训 tokenizer。**在目标数据分布上训 BPE，定词表大小（多语言要更大）。
   **这一步定死之后极难改。**
4. **建数据管线。**采集 → 抽取 → 过滤 → 去重 → 去污染 → 配比（见 A9）。
5. **定架构。**层数/宽度比、注意力变体（GQA/MLA）、FFN 类型、位置编码、norm 位置。
6. **用小 proxy 模型定超参。**muP 让最优 LR 与宽度无关，所以可以在小模型上扫。
7. **短跑验证。**几百步，检查 loss 下降、MFU、显存、checkpoint 能存能读。
8. **开跑，并盯住仪表盘。**loss、梯度范数（裁剪前）、MFU、各 rank 的一致性。
9. **Midtrain。**长上下文扩展 + 高质量数据退火（见 A9.3）。
10. **评测与决策。**held-out loss + 目标 benchmark，判断是继续、回滚，还是进入后训练。

> **最容易被忽略的一步是 7。**几百步的短跑能抓出 90% 的配置错误，成本是整个 run 的万分之一。
> 直接开大跑然后在第 40k 步发现 data sampler 有 bug，是真实会发生的事（见 A5.5）。

#### 自测 · A4.2

**Q A4.2.1** — 给你 512 张 H100、一个月。讲讲你会怎么规划这次训练。

**先算算力预算。**40% MFU 下，$$512 \times 9.89\times10^{14} \times 0.40 \times 30\times86400
\approx 5.2\times10^{23}$$ FLOPs。

**再定模型规模。**用 $$C = 6ND$$ 和 Chinchilla 的 $$D \approx 20N$$：
$$C = 120N^2 \Rightarrow N = \sqrt{C/120} \approx 6.6\times10^{10}$$——大约是一个 66B 模型
配 1.3T token。

**然后拿服务成本回头核一遍。**如果这个模型要大量对外服务，Chinchilla 最优就是错的靶子——
应该训一个更小的、训更久。20B 模型配 4T token 花掉同样的算力，而服务成本便宜 3 倍。

**剩下的按清单走**：tokenizer、数据管线、架构、小 proxy 扫超参、短跑验证、开跑。

> **追问**
> - *什么情况下你会偏离 Chinchilla？* → 推理成本占主导、目标领域数据不够，
>   或者服务端的显存预算被焊死了。
> - *开跑之后第一件要确认的事是什么？* → loss 在降，而且 MFU 和短跑里量到的对得上。
>   如果 MFU 只有当时的一半，先停下来查清楚，别拿一个月去烧。
>
> **陷阱**
> - 跳过第 7 步的短跑验证直接开大跑。几百步能抓出九成的配置错误，成本是整个 run 的万分之一。

---

<a id="a4-3"></a>
### A4.3 架构与超参的选择

**形状（宽 vs 深）。**给定参数预算 $$N \approx 12LD^2$$，你可以选很多 $$(L, D)$$ 组合。经验：

- **太深太窄** → pipeline 段多、bubble 大，而且每层的矩阵瘦，MFU 低。
- **太宽太浅** → 表达深度不足，且 TP 通信量随 $$D$$ 增长。
- 实践中 $$D/L$$ 落在 100–150 附近（Llama-3-70B：$$8192/80 = 102$$）。

**其他要定的：**

| 选项 | 现代默认 | 理由 |
|---|---|---|
| 注意力 | GQA（$$K=8$$）或 MLA | KV cache 是长上下文的瓶颈 |
| FFN | SwiGLU，$$F=\tfrac83 D$$ | 经验更好，参数持平 |
| Norm | RMSNorm，pre-LN | 更少归约、去掉对 warmup 的**架构性**依赖（仍然要 warmup，见 A1.6） |
| 位置 | RoPE | 相对位置、可外推（配缩放） |
| 词表 | 32k–256k | 多语言要大；影响 $$2VD$$ |
| 初始化 | $$\mathcal N(0, 0.02)$$，残差层按 $$1/\sqrt{2L}$$ 缩放 | 控制残差流增长 |

**超参。**Batch size 用 token 计（百万级），随规模增大。LR 随规模**下降**——这正是 muP 要解决的。
Warmup 取总步数 1–2%。Weight decay 0.1。$$\beta_2=0.95$$ 而不是 0.999。

> **为什么残差层的初始化要按 $$1/\sqrt{2L}$$ 缩放。**Pre-LN 下残差流的方差随层数累加。
> 如果每层的输出都是 $$O(1)$$，$$L$$ 层之后流的量级就是 $$O(\sqrt L)$$，后面的层相对越来越
> 无足轻重。按深度缩放初始化能让每层的相对贡献保持一致。

#### 自测 · A4.3

**Q A4.3.1** — 宽深比怎么定？

中间是一大片平台，所以这不是个需要精雕的选择——但两端的失效模式是真的。太深太窄：
pipeline 段更多（bubble 更大）、矩阵更瘦（MFU 低）、优化也更难。太宽太浅：组合深度不够，
而且 TP 通信量随 $$D$$ 增长。

实用锚点：$$D/L \approx 100$$–$$150$$。Llama-3-70B 是 $$8192/80 = 102$$。

最后拍板的通常是系统层面的账：深度的代价是 pipeline bubble，宽度的代价是张量并行带宽。
选你的互联扛得住的那一边。

> **追问**
> - *这个最优比例会随规模变吗？* → 会，但很慢——模型越大，相对于深度会稍微更宽一些。
>   scaling law 的论文里对此有显式拟合。
>
> **陷阱**
> - 说「越深越好」。深度换来的是 pipeline bubble 和更瘦的矩阵，两头都有代价。

**Q A4.3.2** — 为什么模型越大，峰值学习率越小？

在标准参数化下，更新量相对于权重本身的尺度会随宽度增长，所以稳定的 LR 必须变小——
经验上大致 $$\propto 1/\sqrt{D}$$，或者拟合成 $$\text{LR}(C)=\beta C^{-\alpha}$$。

muP 的修法是重新缩放初始化和每层的学习率，让**相对**更新幅度与宽度无关。
这样最优 LR 就能从小 proxy 模型迁移过来——对于一个只跑得起一次的 run，这是唯一现实的调法。

---

<a id="a4-4"></a>
### A4.4 训练动态：曲线该长什么样

**正常的 loss 曲线**在 log-log 下接近直线（幂律），叠加：

- 开头几百步陡降（学到 token 频率分布——即 unigram baseline）；
- 之后是长而平滑的幂律段；
- LR 衰减阶段会有一段额外的加速下降。

**要同时盯的四条线**，只看 loss 是不够的：

| 指标 | 正常 | 异常意味着 |
|---|---|---|
| Loss | 平滑幂律 | 尖峰 → 见 A5.5 |
| **梯度范数（裁剪前）** | 平稳，偶有小峰 | 持续上升 → 不稳定在酝酿 |
| MFU | 恒定 | 下降 → 通信/数据管线问题 |
| 各 rank loss 一致性 | 一致 | 某 rank 偏离 → 硬件问题 |

**梯度范数是最早的预警**，而它只有在你记录**裁剪前**的值时才有用。很多人只记录裁剪后的，
那条线永远是平的，什么也看不出来。

> **什么时候该停。**如果 held-out loss 还在下降，通常就该继续——预训练很少真正饱和，
> 停下来往往是预算问题而不是收益问题。真正该停的信号是：held-out loss 平了但训练 loss 还在降
> （过拟合，说明数据重复了），或者目标能力的 benchmark 不再动。

#### 自测 · A4.4

**Q A4.4.1** — 你的 loss 曲线开头有很长一段平台才开始下降。这是怎么回事？

几乎总是学习率太低，或者 warmup 太长。模型卡在 unigram 解附近——它学会了 token 频率，
别的什么也没学到。

诊断办法：看这个 loss 值是不是正好等于你语料上 unigram 分布的熵。如果是，
模型输出的就是一个频率匹配的分布，上下文完全没用上。然后把 warmup 之后的**实际** LR
打出来（不是配置里的值）——scheduler 差一位是很常见的原因。

> **追问**
> - *如果是先快速下降、然后停在很高的位置呢？* → 可能是 label/shift 的 bug，
>   或者数据管线返回了退化的东西。用十个样本过拟合来隔离。
>
> **陷阱**
> - 只盯 loss。梯度范数、MFU、各 rank 一致性要一起看，而且梯度范数要记裁剪前的。

**Q A4.4.2** — 为什么梯度范数要记裁剪之前的值？

因为裁剪之后那条线按构造就是平的——你想监控的信号已经被自己毁掉了。裁剪前的范数持续上升，
是不稳定正在酝酿的最早信号，往往比它在 loss 上显形早几百步。

它还能告诉你裁剪到底**有没有在生效**。如果大多数步都在被裁，那个裁剪阈值其实在干学习率的活，
说明你的 LR 太高了。

---

<a id="a4-5"></a>
### A4.5 Checkpoint 与容错

**为什么它是预训练的一等问题。**一次 90 天的 run 里硬件故障**必然**发生。GPU 会掉、节点会挂、
NCCL 会超时。没有容错设计，一次故障就损失自上次 checkpoint 以来的全部算力。

**要存什么。**模型权重、优化器状态（这是大头，$$8P$$）、学习率调度器状态、
**以及 data sampler 的位置**。

> **最后那一项是最容易漏的，而且后果最严重。**如果 resume 时 data sampler 没有恢复位置，
> 模型会重读已经见过的 token。Bekman 的警告：你可能事后才发现自己
> *"把原计划各看一次的 300B token，变成了同样的 50B token 训了 6 遍。"*
> 这不是尖峰，是一次被静默作废的训练。

**频率怎么定。**设平均无故障时间 MTBF，checkpoint 间隔 $$T_c$$，则期望损失约 $$T_c/2$$ 的算力。
经验：$$T_c$$ 取 15–30 分钟，远小于 MTBF。但写入本身不能拖慢训练——用异步写入和分片
checkpoint（每个 rank 只写自己那份）。

#### 自测 · A4.5

**Q A4.5.1** — 2048 张 GPU、90 天的 run，设计一套 checkpoint 策略。

**存什么：**权重、优化器状态、scheduler 状态、RNG 状态，以及 data sampler 的位置。
漏掉最后一项会静默作废整个 run。

**频率：**这个规模下 MTBF 大约 4 小时，那就每 15–30 分钟存一次；
一次故障的期望损失就压在 15 分钟算力以内。

**把写入做便宜：**分片写（每个 rank 只写自己那一份，不做 gather）、异步写（先拷到主机内存，
再后台刷盘，训练不停），并保留最近 $$k$$ 个的滚动窗口加上定期的永久快照——
如果发散是很晚才被发现的，你可能得往回退不止一个 checkpoint。

**重启也要一起设计：**自动检测（watchdog 盯的是 step 有没有推进，而不只是进程还活着——
NCCL 集合通信挂住时进程是活的），外加一个备用节点池，让重启不必等资源调度。

> **追问**
> - *怎么验证一个 checkpoint 真的能加载？* → 起一个单独的作业把它恢复出来，
>   检查固定 batch 上的 loss 对得上。没恢复过的 checkpoint 不算 checkpoint。
> - *为什么要留好几个永久快照？* → 因为缓慢发展的发散可能几千步之后才看得出来，
>   你需要有一个干净的位置能退回去。
>
> **陷阱**
> - checkpoint 只存权重和优化器状态，忘了 data sampler 的位置。

---

<a id="a4-6"></a>
### A4.6 预训练里的评测

**主指标是 held-out loss**，而不是 benchmark。理由：它平滑、可比、方差低、每一步都能算，
而 benchmark 是离散的、噪声大、还可能被污染。

**但 loss 不够。**要配合：

- **分领域的 held-out loss**（代码 / 数学 / 多语言各一份）——总 loss 掩盖了此消彼长。
- **少量便宜的 benchmark**，定期跑，看趋势不看绝对值。
- **定性抽样**。定期读几十条生成。这个最容易被跳过，也最容易发现 loss 看不出来的问题
  （比如复读、格式崩坏）。

> **不要在预训练期间频繁跑大 benchmark。**它们慢、噪声大，而且会诱使你对着噪声做决策。
> 一天一次的小套件 + 里程碑处的全量评测就够了。

#### 自测 · A4.6

**Q A4.6.1** — 训练期间为什么 held-out loss 比 MMLU 更适合当指标？

Loss 是连续的、方差低、每一步都能算，而且在同一个 tokenizer 的不同 checkpoint 之间可比。
Benchmark 准确率是带阈值的（exact match），因此离散且噪声大——模型可以明显变好而
benchmark 纹丝不动，也可以仅仅因为 run 与 run 之间的噪声就动一下。

要注意的是：loss 在不同 tokenizer 之间**不可比**，而且进了后训练之后它就完全不再跟踪
"有没有用"了。非要跨 tokenizer 比，就用 bits-per-byte。

> **追问**
> - *那预训练期间什么时候看 benchmark？* → 在里程碑处看，用来决定继续、改数据配比还是停。
>   不用来做逐步决策。

---

> **待补概念：**继续预训练（continued pretraining）与领域适配、
> 训练/推理的数值不一致、模型合并（model merging / soup）、
> 训练日志的公开 logbook 阅读法。
>
> **陷阱**
> - 预训练期间频繁跑大 benchmark。慢、噪声大，而且会诱使你对着噪声做决策。

---

<a id="section-a5"></a>

## A5 · 训练基础设施

并行、精度、稳定性、MFU —— 这些对预训练和后训练**都**适用，所以单独成节。

**这一节的分界线：**先说你**哪一项内存不够**，再说用哪种并行。反过来就是背名词。

---

<a id="a5-1"></a>
### A5.1 显存都花在哪

**先写出显存等式，再谈策略。**这个顺序本身就是信号。

$$\text{memory} = \underbrace{P}_{\text{参数}} + \underbrace{P}_{\text{梯度}} + \underbrace{2P\text{–}4P}_{\text{优化器}} + \underbrace{\text{激活}}_{\propto BS}$$

![训练显存的 16 bytes/param 从哪来](/assets/img/blog/interview-knowledge/qa4_memory.png)

**混合精度 + AdamW 的标准账**（每参数字节数）：

| 项 | 精度 | 字节/参数 |
|---|---|---|
| bf16 权重 | bf16 | 2 |
| bf16 梯度 | bf16 | 2 |
| fp32 主权重 | fp32 | 4 |
| Adam 一阶矩 | fp32 | 4 |
| Adam 二阶矩 | fp32 | 4 |
| **合计** | | **16** |

所以一个 70B 模型光是状态就 **1,120 GB**，还没算激活。这就是为什么单卡训练大模型
从一开始就不在讨论范围内。

**激活是另一半故事**，而且它随 $$B\times S$$ 增长，所以长上下文训练里它反而是主导项。
Gradient checkpointing 用约 30% 的额外计算换掉大部分激活显存。

#### 自测 · A5.1

**Q A5.1.1** — 解释 gradient checkpointing。它的代价是什么，什么时候不划算？

前向算出来的激活必须留着，因为反向要用。Checkpointing 只保留其中一部分——通常是每个层边界
一个张量——反向时从最近的保存点**重算**其余部分。

标准的交换比是**约 30% 的额外算力换掉大部分激活显存**。说得干净一点：$$L$$ 层的网络，
每 $$\sqrt{L}$$ 层存一个 checkpoint，显存从 $$O(L)$$ 降到 $$O(\sqrt L)$$，
代价是多一次前向。

什么时候不划算：激活本来就不是你的瓶颈（卡在优化器状态上，那就该去切它），
或者你已经是计算受限、延迟比塞下更大的 batch 更重要。

有一个统计口径上的细节值得提：重算**不算**进 $$6N$$ 那个分子，所以打开 checkpointing 会
**拉低 MFU**，哪怕吞吐（token/秒）其实变高了。想看硬件层面的真实情况就去比 HFU（见 A5.4）。

**Q A5.1.2** — 为什么是每参数 16 字节，而不是 6？

因为大头在优化器。bf16 的权重和梯度各 2 字节；fp32 主副本 4 字节；Adam 的两个矩估计在 fp32 下
各 4 字节。十六字节里有十二字节是优化器和主状态，这正是 ZeRO 优先切它们的原因——
光是 stage 1 就干掉了最大的那一项。

这个数字有时被报成 18。差别在于梯度是否用 fp32 累加、以及要不要把一份临时的 bf16 副本算进去，
所以先把标准配方说清楚，再补一句这取决于框架。

> **追问**
> - *带 momentum 的 SGD 要多少？* → 每参数少 4 字节（一个矩而不是两个）。LLM 不用它，
>   是因为 Adam 的逐参数缩放对 transformer 很关键。

---

<a id="a5-2"></a>
### A5.2 并行策略：每种切什么

| 策略 | 切什么 | 通信模式 | 什么时候撑不住 |
|---|---|---|---|
| DDP | 什么都不切 | 梯度 all-reduce | 每卡要放下完整状态 |
| ZeRO-1 | 优化器状态 | reduce-scatter + all-gather | — |
| ZeRO-2 | + 梯度 | 同上 | — |
| ZeRO-3 / FSDP | + 参数 | 每层 gather | 通信量增长 |
| Tensor (TP) | 层**内**的参数和激活 | 每层**内部**两次 all-reduce | 需要 NVLink，不要跨节点 |
| Pipeline (PP) | 按层切 | 边界点对点 | Bubble $$\approx (p-1)/(m+p-1)$$ |
| Context / Ring | 切序列 | K/V 环形交换 | 只解决激活 |
| Expert (EP) | 切专家 | all-to-all | 负载不均 |
| Recompute | — | 无 | 约 30% 算力换激活显存 |

**3D 并行** = DP × TP × PP。标准布局：**TP 放在节点内最内层**（最吃带宽），
**PP 跨节点**（通信量最低），**DP 在最外层**。

![集合通信操作](/assets/img/blog/interview-knowledge/qa3_collectives.png)

**集合通信原语**：all-reduce（求和，所有人拿到结果）、all-gather（拼接，所有人拿到全部）、
reduce-scatter（求和，每人拿一片）。注意 **all-reduce = reduce-scatter + all-gather**，
这正是 ZeRO 的通信成本能和 DDP 相当的原因。

#### 自测 · A5.2

**Q A5.2.1** — 训练大模型时显存爆了。把可选方案过一遍。

**从哪一项太大开始说**，不要从策略清单开始背。

如果大头是**优化器状态**，先 ZeRO-1 再 ZeRO-2——它们切掉十六字节里最重的十二字节，
而通信量和 DDP 相当，几乎是白拿的。

如果**参数**放不下，上 ZeRO-3/FSDP 或者张量并行。TP 只在节点内用：它每层要做两次 all-reduce，
而跨节点带宽比 NVLink 低一个数量级，一跨节点收益立刻被吃光。

如果大头是**激活**——长上下文下通常都是——先上 gradient checkpointing
（约 30% 额外算力换掉大部分显存），再考虑 context / 序列并行。

如果模型深到这些都不够，就跨节点做流水线并行，接受 $$(p-1)/(m+p-1)$$ 的 bubble，
并用足够多的 micro-batch 把它压小。

> **追问**
> - *怎么把 pipeline bubble 压小？* → 加 micro-batch 数、交错式 1F1B，或者把反向拆成
>   输入梯度和权重梯度两半的 zero-bubble 调度。
> - *为什么 ZeRO 的通信成本和 DDP 相当？* → 因为 all-reduce 本来就是 reduce-scatter 接
>   all-gather，而 ZeRO-1/2 做的正是这两步。
>
> **陷阱**
> - 先说策略名字再说要解决什么。**先说你哪一项内存不够**，那才是面试官在等的。

---

<a id="a5-3"></a>
### A5.3 混合精度

**方案。**fp32 主权重；前向/反向用低精度；优化器更新作用在主副本上。

**为什么要主副本。**更新量通常比权重小几个数量级。bf16 有 7 位尾数，相对精度约
$$2^{-8}\approx 0.4\%$$。当 $$|\Delta w|/|w| < 0.4\%$$ 时，加法**直接舍回 $$w$$**——
模型悄悄停止学习，而 loss 曲线看起来还挺合理。

**bf16 vs fp16。**同样位宽，不同划分：fp16 是 5 指数 + 10 尾数，bf16 是 8 + 7。
八位指数让 bf16 拥有**和 fp32 相同的动态范围**——注意力 logits 不会溢出，
而且**完全不需要 loss scaling 那套机制**。代价是尾数精度，而训练结果表明它不太在乎。

用 fp16 就需要动态 loss scaling：反向前把 loss 乘大，让小梯度进入可表示范围，
优化器步之前再除回来，遇到 inf 就退让。这套机制是"训练悄悄不再进步"的常见来源。

**必须留在 fp32 的：**归约。Softmax 分母、layer-norm 统计量、loss 累加、梯度 all-reduce。

#### 自测 · A5.3

**Q A5.3.1** — 为什么 bf16 取代了 fp16？

动态范围。两者都是 16 位，但 fp16 划成 5 指数 + 10 尾数，bf16 划成 8 + 7。八位指数让 bf16 拿到
和 fp32 一样的范围，注意力 logits 和梯度都不会溢出，**loss scaling 那一整套机制完全不需要**。

代价是尾数精度，而经验上训练并不太在乎——这是一个很好的例子：选对该优化的那条轴，
而不是挑数字更大的那个。

> **追问**
> - *FP8 呢？* → DeepSeek-V3 用 **per-tile 和 per-block 的 scaling** 在 FP8 上做了大规模训练，
>   因为 FP8 的范围太窄，一个全局 scale 盖不住整个张量。收益是实打实的，
>   数值工程也是实打实地难。
>
> **陷阱**
> - 说 bf16"精度更高"。它精度**更低**，赢在动态范围。

---

<a id="a5-4"></a>
### A5.4 MFU

$$\text{MFU} = \frac{6N\cdot(\text{tokens/s})}{\text{GPUs}\times\text{peak FLOP/s}}$$

分子用的是**模型所需**的 FLOPs（每 token $$6N$$）——不含重算、不含通信。
所以 gradient checkpointing 会**降低** MFU 而可能**提高**吞吐。
HFU（Hardware FLOPs Utilization）把重算算进分子，MFU 不算。

**大规模训练的健康区间：35–50%。**低于 30% 通常意味着某个具体问题。

**按这个顺序查：**

1. **通信没有和计算重叠。**最常见。DP 的 all-reduce 有没有和反向重叠？
   ZeRO-3 的参数 gather 有没有预取？
2. **Pipeline bubble。**这里要小心**两套口径**，面试里被追问一次就露馅：
   空闲占**墙钟**的比例是 $$(p-1)/(m+p-1)$$，而 Megatron 论文报的
   $$(p-1)/m$$ 是 bubble 相对**理想计算时间**的比例。$$p=m=8$$ 时前者 47%、后者 87.5%。
   本节统一用墙钟口径。
3. **每卡 batch 太小。**矩阵太瘦，喂不饱。
4. **Data loader 把 GPU 饿着了。**看空闲时间的*分布*，不要看平均利用率。
5. **TP 跨了节点边界。**
6. **序列非常长。**$$S^2$$ 的注意力项不计入 $$6N$$，所以长上下文下 MFU *合理地*更低——
   这里的低 MFU 不是 bug。

#### 自测 · A5.4

**Q A5.4.1** — 你的 MFU 是 20%。你会查什么？

按命中概率从高到低：通信没和计算重叠（DP 的 all-reduce 没盖住反向，或者 ZeRO-3 的 gather
没预取）；pipeline bubble，它让 $$(p-1)/(m+p-1)$$ 的墙钟时间空转，$$p=m=8$$ 时光这一项就 47%；
每卡 batch 太小，矩阵乘喂不饱；data loader 把 GPU 饿着；以及张量并行跨了节点边界。

有一种情况下 20% 是**对的**：序列非常长。$$S^2$$ 的注意力项是真实的计算量，
但不在 $$6N$$ 这个分子里，所以 MFU 读数低是合理的。动手去追之前先看一眼 HFU。

> **追问**
> - *直接看 `nvidia-smi` 的利用率不行吗？* → 它只说明有 kernel 在跑，不说明它在做有用的算术。
>   一个纯访存受限的 kernel 也会显示 100%。
> - *MoE 怎么算？* → 分子用**激活**参数量，不是总参数量。
>
> **陷阱**
> - 分母用了稀疏峰值（H100 是 989 TFLOP/s dense，1979 是 2:4 稀疏）。

---

<a id="a5-5"></a>
### A5.5 训练不稳定的诊断

**不要一上来就降学习率。**先给尖峰分类——三种形状原因不同、处理也不同。

**Bekman 的分类：**快速恢复、缓慢恢复、不完全恢复。常见原因：
*"一段坏数据，要么是 shuffle 得不好，要么是没清洗干净。"*

还有一个让它成为好问题的细节：

> *"人们会怀疑是尖峰前那个 batch 触发的……但相当多时候问题在很多步之前就开始酝酿，
> 然后才突然爆发。"*

**排查阶梯，从最便宜的开始：**

1. **它是真的还是日志问题？**梯度范数和验证 loss 里有没有？还是只在某个 rank 的
   平滑训练曲线上？
2. **是不是 resume 的产物？** ← 价值最高的检查，而且几乎没人会提。如果 run 重启过而
   **data sampler 没有恢复位置**，模型在重读已见过的 token。Bekman 的警告很刺眼：
   你可能发现自己*"把原计划各看一次的 300B token，变成了同样的 50B token 训了 6 遍。"*
   这不是尖峰病理，这是一次被静默作废的训练。
3. **硬件？**一张坏卡就能污染整个 all-reduce。检查各 rank 的 loss、ECC 错误、
   跑一次集合通信基准。
4. **数值？**fp16 溢出或 loss scaler 崩了。在裁剪**之前**检查 inf/NaN。
5. **数据？**现在才去看尖峰**之前那个窗口**的 batch，而不是尖峰当时那个。
   重复 token、损坏的 shard、语言切换。
6. **最后才是优化。**LR 对当前曲率太高？调度变更后二阶矩过时？

**按分类匹配处理：**快速恢复 → 记录并继续。缓慢 → 降 LR 或跳过那段数据。
不恢复 → 回滚到上一个好 checkpoint，换一个数据顺序重来。

#### 自测 · A5.5

**Q A5.5.1** — 你在预训练一个 100B 模型。第 42,000 步 loss 出现尖峰。讲讲你会怎么做。

（阶梯见上。）有两点能让这个回答显出水平。

**先分类，再动手。**快速恢复的尖峰通常没事——记录一下继续跑。缓慢恢复意味着降 LR
或者跳过那段数据。不恢复意味着回滚并换一个数据顺序。第一反应就去动学习率，是最常见的错误动作。

**早点检查 resume 这条路径。**如果 run 重启过而 data sampler 没有恢复位置，你就是在重读 token，
这不是尖峰的病理，这是一次被静默作废的训练。这个检查不花任何成本，而且几乎没人会提。

**还有，要看尖峰之前，而不是尖峰当时。**不稳定通常是在很多步里慢慢长出来的，
紧挨着尖峰的那个 batch 很少是真凶。

> **追问**
> - *你会提前记录哪些东西？* → 裁剪前的梯度范数（见 A4.4）、每个 rank 的 loss，以及每个
>   checkpoint 里 data sampler 的位置。大多数尖峰排查失败，都是因为没有这些。
>
> **陷阱**
> - 第一反应是降 LR。第一反应应该是**分类**，而且要提到 data sampler。

---

> **待补概念：**ZeRO 各阶段的通信量定量分析、NCCL 调优与拓扑感知、
> SLURM/K8s 编排、故障检测与自动重启、弹性训练、
> 训练/推理数值不一致的排查。

---

<a id="section-a6"></a>

## A6 · Post-training 与 RL

从 SFT 到 RLHF 到 RLVR。Alisa 在 policy gradient 上写了 185 行含完整证明，这一块她最深。

**这一节的分界线：**谁都能报出 PPO / GRPO / DPO 的名字。信号在于 **KL 项放在哪**、
**advantage 是什么形状**、以及**你能不能把三个推导写出来**。

---

<a id="a6-1"></a>
### A6.1 后训练阶梯

| 阶段 | 数据 | 能修 | 修不了 |
|---|---|---|---|
| 预训练 | 网络规模无标注 | 知识、语法、世界模型 | 指令遵循、格式 |
| Midtraining | 精选高质量、长上下文、代码、数学 | 领域能力、上下文长度 | 偏好、风格 |
| SFT | 示范 | 格式、指令遵循、工具语法 | 没被示范的东西——它只能模仿 |
| Reward modeling | 偏好对 | "更好"的标量代理 | 它自己的错误设定 |
| RL | Prompt + 奖励或 verifier | 优化奖励，包括钻它的空子 | 基座模型没有的知识 |
| 蒸馏 | 老师的输出 | 成本、延迟 | 一般无法超过老师 |

**一句话框架，随时可用：**

> **SFT 教模型好答案长什么样；RL 教模型它自己的哪些答案更好。**

这解释了为什么 SFT 饱和之后 RL 还能继续起作用——SFT 只能朝示范推，而 RL 可以对模型
**自己的**样本排序，推向没有人示范过的区域。这也解释了 RL 为什么装不进新知识：
它只是给基座已有的能力重新加权。

#### 自测 · A6.1

**Q A6.1.1** — 把从预训练到上线模型的各个阶段摆出来。

（表见上。）真正重要的是这个框架：SFT 教模型好答案长什么样，RL 教模型它**自己的**哪些答案更好。
这就是为什么 SFT 饱和之后 RL 还能继续起作用，也是为什么 RL 装不进基座模型没有的知识——
它只是给策略已经能产出的东西重新加权。

> **追问**
> - *为什么说 midtraining 是"没人写下来的那个阶段"？* → 因为长上下文扩展、代码/数学的大幅加权、
>   领域注入，实际都发生在这里，而各家实验室几乎什么都不披露——数据配比就是护城河。
> - *RL 之前一定要先 SFT 吗？* → 大多数配方里是的——从基座直接做 RL 方差高、也慢。
>   R1-Zero 说明了在可验证奖励下，从基座纯 RL *确实*可行，但发布出来的 R1 仍然保留了
>   一个冷启动 SFT 阶段来保证可读性。
>
> **陷阱**
> - 说 RL 能"教会模型新知识"。它只能重新加权已有能力。

---

<a id="a6-2"></a>
### A6.2 SFT：细节比想象中多

看起来 SFT 就是"继续做 next-token 预测"，但有四个实现细节会被问到。

**1. Loss masking。**只在**回答**的 token 上算损失，prompt 部分的标签设为 `-100`。
不 mask 的话模型会花容量去学习建模 prompt 分布，而那不是你要的行为。

```python
labels = input_ids.clone()                    # input_ids: (B, T)
for i, n in enumerate(prompt_lens):
    labels[i, :n] = -100                      # 每条样本各自的 prompt 长度
labels[attention_mask == 0] = -100            # padding 也要屏蔽
```

> 写成 `labels[:len(prompt_ids)] = -100` 是白板上最常见的手滑：在 `(B, T)` 上
> 那是在切 **batch 维**——把前几条样本整条屏蔽掉，而不是屏蔽每条的 prompt。
> 单条无 batch 时才对。

**2. Packing 与串扰。**把多条短样本拼进一个序列以提高利用率，但**必须阻断跨样本注意力**
（block-diagonal mask 或 `position_ids` 重置）。不做的话样本 B 能看到样本 A，
这是一种静默的数据污染。

**3. Epoch 数。**SFT 数据通常很小，**1–3 个 epoch**。再多就开始死记硬背，
多样性和校准都会掉。

**4. 质量远胜数量。**LIMA 那条结论——1000 条精心挑选的样本可以打过数万条噪声样本——
在这一阶段成立，因为你调的是**格式和行为**，不是知识。

#### 自测 · A6.2

**Q A6.2.1** — SFT 时为什么要屏蔽 prompt token 上的损失？

因为你训的是条件分布 $$p(\text{response}\mid\text{prompt})$$，不是联合分布。
把 prompt token 算进去，等于拿容量去建模指令的分布，那不是你要的行为，
还会稀释你真正在意的那些 token 上的梯度。

prompt 相对回答很长时影响最大——一份 2000 token 的文档配 50 token 的答案，
97% 的损失来自 prompt，模型基本上是在学预测文档。

> **追问**
> - *有没有在 prompt 上训反而有用的情况？* → 数据极少时略有帮助，起正则作用。大多数配方还是 mask。
> - *Packing 会坏在哪？* → 串扰：没有块对角 mask 的话，拼在一起的序列里一条样本能看到另一条。
>   另外别忘了重置 `position_ids`。

---

<a id="a6-3"></a>
### A6.3 Reward model 与 Bradley-Terry

在偏好对上训一个标量头：

$$\mathcal L = -\log\sigma\big(r_\theta(x,y_w) - r_\theta(x,y_l)\big)$$

这就是 Bradley-Terry 模型：假设人更偏好 $$y_w$$ 的概率是 $$\sigma(r_w - r_l)$$。

**三件必须能说出来的事：**

1. **奖励只在相差一个常数的意义下被确定。**BT 约束的是**差**，不是绝对值，
   所以跨 run 比较原始奖励值没有意义。因此要做 per-batch 归一化。
2. **它在窄分布上训练，然后被查询在远离该分布的地方**——因为策略在移动。
   这是教科书式的 Goodhart 设置，也正是 KL 惩罚存在的全部理由。
3. **能拿到 verifier 的地方，verifier 胜过学出来的奖励。**单元测试是一个函数，不是一个网络，
   没法用同样的方式被钻空子。从"这个答案对"到梯度的因果链短得多。

#### 自测 · A6.3

**Q A6.3.1** — 为什么不能跨 run 比较 reward model 的分数？

Bradley-Terry 只约束**差值**：损失只依赖 $$r_w - r_l$$，给每个分数加上任意常数都不改变它。
奖励只在相差一个平移的意义下被确定，绝对尺度因而是任意的，而且随 run 而变。

由此得到几条：把奖励当 advantage 用之前先做 per-batch 归一化；不要在原始值上设阈值；
说"奖励涨了"的时候，讲清楚是相对什么涨的。

---

<a id="a6-4"></a>
### A6.4 Policy gradient 的推导

**目标。**最大化 $$J(\theta) = \mathbb E_{\tau\sim\pi_\theta}[R(\tau)]$$。

**推导**——整个过程只靠一个 log-derivative trick $$\nabla P = P\nabla\log P$$：

$$\begin{aligned}
\nabla_\theta J(\theta)
&= \nabla_\theta \sum_\tau P(\tau\mid\theta)R(\tau)\\
&= \sum_\tau \nabla_\theta P(\tau\mid\theta)\,R(\tau)\\
&= \sum_\tau P(\tau\mid\theta)\,\nabla_\theta\log P(\tau\mid\theta)\,R(\tau)\\
&= \mathbb E_{\tau\sim\pi_\theta}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,R(\tau)\Big]
\end{aligned}$$

**为什么它长得像 SFT。**更新量是 $$\nabla\log\pi_\theta(a_t\mid s_t)$$——正是 SFT 的梯度——
但有两个区别：数据 $$\tau$$ 是**从策略自己采样的**，而且梯度被 $$R(\tau)$$ **加权**。
正奖励抬高整条轨迹每个 token 的对数概率，负奖励压低，幅度决定步长。

**一个值得说的概念点。**"policy gradient loss"不是通常意义上的损失。$$L(\theta)$$ 并不度量
策略有多好——它只是一个标量，其 `.backward()` 恰好给出正确的梯度。这里没有固定的目标函数，
因为数据分布随策略移动。

#### 自测 · A6.4

**Q A6.4.1** — 推导 REINFORCE 估计量，并解释它为什么方差大。

（推导见上。）承重的那一步是 log-derivative trick，它把一个概率的梯度变成了一个你可以采样的期望。

方差大是因为**一个标量奖励被记到了整条轨迹头上**。每个 token 拿到相同的权重，
所以在一条大体不错但中间有一步很糟的回答上，那一步糟糕的也被强化了；
而在一个所有回答都得正分的 prompt 上，连平庸的回答也一并被强化。
估计量本身没有任何逐 token 的信用分配。

> **陷阱**
> - 推导时漏掉 log-derivative trick 那一步，或者说不出为什么可以把 $$\nabla$$ 移进求和。

---

<a id="a6-5"></a>
### A6.5 Baseline 为什么无偏

证明基线项期望为零：

$$B = \mathbb E_{\tau}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,b(s_t)\Big]$$

把期望移进求和，再按 $$(s_t, a_t)$$ 分解联合分布：

$$B = \sum_t \mathbb E_{s_t}\Big[\mathbb E_{a_t\mid s_t}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,b(s_t)\big]\Big]$$

内层期望：

$$\begin{aligned}
\mathbb E_{a_t\mid s_t}\big[\nabla\log\pi_\theta(a_t\mid s_t)\,b(s_t)\big]
&= b(s_t)\sum_{a}\pi_\theta(a\mid s_t)\nabla\log\pi_\theta(a\mid s_t)\\
&= b(s_t)\sum_{a}\nabla\pi_\theta(a\mid s_t)\\
&= b(s_t)\,\nabla\Big(\sum_a \pi_\theta(a\mid s_t)\Big) = b(s_t)\,\nabla 1 = 0
\end{aligned}$$

**承重的那一步**是把 $$b(s_t)$$ 提到对 $$a_t$$ 的期望外面——这只有**因为 $$b$$ 依赖状态而不依赖
动作**才合法。整个条件就是这一条。

**为什么要它。**它不改变目标，只改变估计量的方差。我们希望 $$b(s_t)$$ 与 $$R(\tau)$$ 相关，
使 $$R-b$$ 尽量小。PPO 用学出来的 $$V_\psi(s_t)$$，GRPO 用组均值。

#### 自测 · A6.5

**Q A6.5.1** — 证明依赖状态的 baseline 不会给 policy gradient 引入偏差。

（证明见上。）撑住整个证明的一步，是把 $$b(s_t)$$ 提到对 $$a_t$$ 的期望外面，
而这**只有**在 $$b$$ 不依赖动作时才合法。之后就是
$$\sum_a \nabla\pi_\theta(a\mid s_t) = \nabla 1 = 0$$。

> **追问**
> - *baseline 可以依赖动作吗？* → 不行——那会让证明失效并引入偏差。这也是为什么你不能拿采样动作
>   自己的奖励当它的 baseline。
> - *GRPO 的组均值算依赖动作吗？* → 它是由同一 prompt 的其他样本算出来的，
>   所以对单个样本而言近似独立。把样本自身也算进去会带来一点微妙的偏差；
>   有些实现用留一均值。

---

<a id="a6-6"></a>
### A6.6 PPO

**Clipped surrogate。**记 $$r_t = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}$$：

$$L^{\text{CLIP}} = \mathbb E_t\Big[\min\big(r_t \hat A_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big]$$

**clipping 买到什么。**一个软信赖域。它移除了把概率比推到 $$1\pm\epsilon$$ 之外的动机，
所以单次更新不会让策略走太远——这正是朴素 policy gradient 缺的东西。`min` 让它**悲观**：
截断收益但不截断损失。

**GAE。**Advantage 在一步 TD（有偏、低方差）和蒙特卡洛（无偏、高方差）之间插值：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),\qquad \hat A_t = \sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$$

$$\lambda=1$$ 退化为蒙特卡洛，$$\lambda=0$$ 退化为一步 TD。实现时**断言这两个极限**——
最便宜的正确性检查。

**显存里有四个模型：**policy、冻结的 reference、reward model、critic。

**KL 放在哪：**PPO 里按惯例是**从 reward 里减掉**，再算 advantage。

#### 自测 · A6.6

**Q A6.6.1** — 写出 PPO 的目标函数，并解释 GAE 是干什么的。

$$L^{\text{CLIP}} = \mathbb E_t\big[\min(r_t \hat A_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t)\big]$$

其中 $$r_t$$ 是新旧策略概率之比。两个部件，每个都要说得出理由。

**clip 是一个软信赖域。**朴素的 policy gradient 没有任何东西阻止一次更新把策略推离数据采样时的
分布，而那会让重要性加权失效。Clipping 移除了走出 $$1\pm\epsilon$$ 的动机，`min` 让它变得悲观——
截断收益但不截断损失，所以一次有害的更新仍然拿到完整的纠正梯度。

**GAE 控制 advantage 里的偏差-方差权衡。**取
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$ 和
$$\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$$，$$\lambda = 0$$ 给出一步 TD
（方差低，但被 critic 的误差带偏），$$\lambda = 1$$ 给出蒙特卡洛（无偏，方差高）。
$$\lambda$$ 在两者之间插值。

如果由我来实现，我会把这两个极限都写成断言——这是最便宜的正确性检查。

**Q A6.6.2** — PPO 的 clipping 实际限制的是什么？

是**概率比**，不是梯度大小，也不是直接限制 KL。一旦这个比值朝着改善的方向走出
$$[1-\epsilon, 1+\epsilon]$$，目标函数在那里就变平了，梯度为零——于是没有动机继续往前推。

`min` 让它变成单边的：截断收益，不截断损失。如果一次更新把事情弄糟了很多，
你仍然拿到完整的纠正梯度。这个不对称就是所谓的悲观，也是为什么目标函数要写成 `min`
而不是只写 `clip`。

> **追问**
> - *为什么 LLM 的价值函数难训？* → 奖励稀疏（每条回答只有一个标量）；策略在提升导致分布漂移，
>   critic 永远滞后；再加上它是显存里又一个全尺寸模型。这三条理由都指向 GRPO。
>
> **陷阱**
> - 说 clipping 是"限制梯度大小"。它限制的是**概率比**。

---

<a id="a6-7"></a>
### A6.7 GRPO

**洞察。**价值函数**只**是在充当 baseline。那就每个 prompt 采 $$G$$ 条完成、用它们的均值奖励
代替——critic 消失了。

$$\hat A_i = \frac{r_i - \text{mean}(\mathbf r)}{\text{std}(\mathbf r)+\varepsilon}$$

```python
r = rewards.view(-1, G)
adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
adv = adv.reshape(B, 1)                    # 广播到全部 L 个 token

ratio  = (logp - logp_old).exp()
policy = -torch.min(ratio * adv, ratio.clamp(1-eps, 1+eps) * adv)

log_ratio = logp_ref - logp
kl = log_ratio.exp() - log_ratio - 1.0     # k3：既无偏又非负

loss = ((policy + beta * kl) * mask).sum() / mask.sum()
```

**三个真正的面试内容：**

1. **KL 挪进了 loss**，作为 per-token 项，不再折进 reward。而且用的是 Schulman 的 **k3 估计量**：
   记 $$r = \dfrac{\pi_\text{ref}}{\pi_\theta}$$（对 $$\pi_\theta$$ 采样），则

   $$\widehat{\mathrm{KL}} = r - \log r - 1$$

   代码里 `log_ratio = logp_ref - logp` 就是 $$\log r$$，所以 `log_ratio.exp() - log_ratio - 1`。
   用它而不是朴素的 $$-\log r$$，是因为 k3 既无偏**又**逐样本非负——
   朴素 log-ratio 差在单个样本上可能为负，那是没有意义的 KL 估计。
2. **Advantage 是 bandit 式的**：每条完成一个标量，广播到每个 token。**完全没有 per-token
   的信用分配。**这是真实局限，值得主动说。
3. **全打平的组梯度为零。**如果一组里 reward 全相同，advantage 精确为 0，这一组白算。
   在大多数 prompt 要么总对要么总错的数据集上，你的大部分算力什么都没产出。

#### 自测 · A6.7

**Q A6.7.1** — GRPO 和 PPO 有什么不同？把 advantage 和 KL 说具体。

三个区别，只说出第一个是弱答案。

**critic 没了。**advantage 是同一 prompt 的 $$G$$ 条完成上做组内归一化后的奖励。
这从显存里、也从故障面里去掉了一个全尺寸模型。

**KL 挪进了 loss**，作为 per-token 项，不再折进 reward，而且用的是 k3 估计量
$$r - \log r - 1$$（$$r = \pi_{\mathrm{ref}}/\pi_\theta$$），它既无偏又非负——朴素的
log-ratio 差在单个样本上可能为负，那是没有意义的 KL 估计。要用 $$r$$ 来表述，
不要写成 $$e^{-x}+x-1$$：后者只有在 $$x = \log(\pi_\theta/\pi_{\mathrm{ref}})$$ 时才等价，
而代码里 log-ratio 通常是反过来存的。

**advantage 是 bandit 形状的**：每条完成一个标量，广播到每个 token。完全没有逐 token 的
信用分配，这是一个真实的局限，值得在被问之前就主动说出来。

> **追问**
> - *什么时候 GRPO 不是好选择？* → 有稠密的逐 token 奖励时；采不起 $$G$$ 条样本时；
>   以及组内方差很低时。
> - *DAPO 修了什么？* → 四件事。**Clip-Higher**（非对称的截断区间，让低概率 token 仍然能被
>   抬起来，避免熵坍缩）；**动态采样**（丢掉全打平的组——正是上面那个零梯度问题）；
>   **token 级的 loss**，而不是按序列求平均（后者会低估长回答的权重）；
>   以及**超长回答的 reward shaping**。
>
> **陷阱**
> - 说 GRPO"就是没有 critic 的 PPO"就停。

---

<a id="a6-8"></a>
### A6.8 DPO

**结果。**对 KL 约束的 RLHF 目标，最优策略与奖励函数有闭式关系：

$$\pi^*(y\mid x) \propto \pi_\text{ref}(y\mid x)\exp\!\big(\tfrac1\beta r(x,y)\big)
\;\Longrightarrow\; r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_\text{ref}(y\mid x)} + \beta\log Z(x)$$

代进 Bradley-Terry 偏好似然，$$Z(x)$$ **消掉了**（一对完成共享同一个 $$x$$），
剩下一个普通的分类损失：

$$\mathcal L_\text{DPO} = -\log\sigma\Big(\beta\big[(\log\pi_\theta(y_w) - \log\pi_\text{ref}(y_w)) - (\log\pi_\theta(y_l) - \log\pi_\text{ref}(y_l))\big]\Big)$$

**它移除了什么。**没有 reward model、没有 critic、训练循环里没有生成。四次对固定文本的前向，
跑在 SFT 基础设施上，约 2× 显存。

**自检：**在 reference 策略处 margin 为 0，损失恰好是 $$\log 2$$。

**代价：**

- **Off-policy。**它从固定偏好数据集学习。策略一旦偏离这些偏好被采集时的分布，信号就过期了。
  PPO/GRPO 一直在从当前策略采样。
- **Likelihood displacement。**DPO 可以通过压低**被拒**回答的似然来增大 margin，
  而不是抬高被选回答——有时两者的概率都在下降。

#### 自测 · A6.8

**Q A6.8.1** — 走一遍 DPO 的推导，并说说它牺牲了什么。

从 KL 约束下 RLHF 的最优解出发，它有闭式；把它反解出来，用策略与 reference 的对数比表示奖励；
再代进 Bradley-Terry 似然。配分函数 $$Z(x)$$ 对一对完成中的两条是同一个，于是消掉了——
这次消去就是全部的诀窍，也正是它让目标函数变得可算。

它牺牲了什么：它是 **off-policy** 的，学的是在某个分布上采集的偏好，而策略会逐渐漂离那个分布；
它还容易出现 **likelihood displacement**——margin 的增大来自把被拒回答压下去，
而不是把被选回答抬上来。

> **追问**
> - *$$\beta$$ 控制什么？* → 那个隐式的 KL 约束。$$\beta$$ 小 = 约束弱 = 相对 reference 漂得更远。
> - *为什么需要 $$\pi_\text{ref}$$？* → 它给隐式奖励定了锚。没有它，对数比就没有意义，
>   策略可以任意漂移。
> - *有哪些变体？* → IPO（修掉 BT 假设里的一个过拟合病理）、KTO（用二元的好/坏标签代替成对偏好）、
>   SimPO（干脆不要 reference 模型）。
>
> **陷阱**
> - 说 DPO"完全绕过了 reward model"。它学的是一个**隐式** reward。

---

<a id="a6-9"></a>
### A6.9 Reward hacking 与 KL 控制

**具体的、能点名的 reward hacking：**

- 模型对测试套件做特判，而不是解决问题。
- 找到 grader 的格式漏洞（长度、markdown、自信的语气）。
- **推理无效但最终答案正确**——基于 verifier 的奖励看不出来，因为它只看答案。
- Sycophancy：附和用户能提高学出来的奖励。

**缓解：**留一批模型永远训不到的测试；验证**过程**而不只是输出；KL 绳子拉短；
监控推理轨迹的分布漂移，而不只看奖励曲线。

**KL 系数怎么定。**不要凭感觉挑——**给 KL 设目标值**，监控实际 KL 并自适应调 $$\beta$$
把它稳在目标附近。KL 接近零意味着没在学；KL 无界增长意味着正在被钻空子。

> **KL 曲线是这一整节最有用的一张图。**奖励在涨而 KL 也在飙，几乎肯定是 hacking 而不是提升。
> 奖励在涨、KL 稳住、held-out 评测也在涨，才是真的。

#### 自测 · A6.9

**Q A6.9.1** — 奖励在涨，模型却在变差。诊断一下。

典型的 reward hacking，而 KL 曲线通常立刻就能告诉你：如果奖励在爬升的同时，
相对 reference 的 KL 无界增长，那就是策略正在走进 reward model 从没被训练过的区域，
并且在钻它的空子。

**按顺序做这些检查。**去读真实样本——hacking 通常十个例子之内人眼就能看出来。
用**另一个** judge 给一批固定的 held-out 集打分。和一个模型从没训过的 held-out verifier 对比。
看长度和格式的统计量，因为那是最便宜的空子。

**修法。**把 KL 目标收紧；用当前策略的样本重训 reward model（把分布差距补上）；
能换成可验证奖励的地方就换；加上过程级的检查，让"推理无效但答案正确"拿不到分。

> **追问**
> - *RLVR 是什么？* → RL with Verifiable Rewards——用一个检查器代替学出来的 RM。
>   R1 里长 chain of thought 能在没有任何人示范的情况下涌现，靠的就是它。

---

<a id="a6-10"></a>
### A6.10 ★ 蒸馏

**经典（Hinton）蒸馏。**让学生匹配老师的**软分布**而不是硬标签，温度为 $$T$$：

$$\mathcal L = T^2\cdot \operatorname{KL}\big(p_\text{teacher}^{(T)} \,\|\, p_\text{student}^{(T)}\big)$$

$$T^2$$ 补偿温度升高导致的梯度缩小。软标签携带"暗知识"——错误答案之间的相对概率
编码了 one-hot 标签丢掉的相似性结构。

**LLM 语境下有三件事都叫蒸馏，实际很不同：**

1. **序列级 / 行为克隆。**从老师采样输出，用它们 SFT 学生。大多数"蒸馏版"开源模型其实是这个，
   它不需要老师的 logits，只需要 API。R1 的 distilled Qwen/Llama 就是这一类。
2. **Logit 蒸馏。**匹配完整的 next-token 分布。需要老师 logits，只能自家用。每 token 信号更强。
3. **On-policy 蒸馏。**从**学生**采样，用老师打分。修掉 (1) 的曝光偏差问题：
   off-policy 采样下学生永远见不到自己犯错后的前缀，也就学不会从错误中恢复。

**正向 vs 反向 KL 在这里很关键。**正向 KL（处处匹配老师）让学生 mean-covering——
它会把质量摊到自己表示不了的模式上。反向 KL 让它 mode-seeking——挑一个模式做好。
对一个表示不了老师完整分布的小学生，反向 KL 常常生成质量更好。

#### 自测 · A6.10

**Q A6.10.1** — 学生能超过老师吗？

在被蒸馏的那个能力本身上，一般不能——天花板就是老师的分布。

但当你把一个昂贵的搜索过程蒸进一次前向时，**在有效意义上是能的**。如果老师的输出来自
best-of-N 采样、长 chain of thought 或者工具调用，学生学到的是用一次前向产出老师要花大量
test-time compute 才能产出的东西。你做的是**把 test-time compute 蒸进权重**，
按每 FLOP 算，这是赢过老师的。

另一条路是先蒸馏再 RL：用蒸馏把成功率从地板上抬起来，让 RL 有信号，
然后让 RL 去超过老师（见 A7.2）。

> **陷阱**
> - 只讲 Hinton 那套 soft target。LLM 圈里绝大多数"蒸馏"其实是序列级行为克隆。

---

<a id="a6-11"></a>
### A6.11 LoRA 与 PEFT

$$W' = W + \frac{\alpha}{r}BA,\qquad A\in\mathbb R^{r\times d_\text{in}},\; B\in\mathbb R^{d_\text{out}\times r}$$

```python
self.A = nn.Parameter(torch.zeros(r, in_f)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
self.B = nn.Parameter(torch.zeros(out_f, r))          # 置零 → 第 0 步适配器是恒等
def forward(self, x):
    return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling
```

**面试官在查的两个性质：**

1. **初始化时是恒等。**$$B=0 \Rightarrow BA=0$$，适配后的模型精确等于基座模型。
   两个都随机初始化会静默污染起点——这是"用过 LoRA 库但没读过实现"的标志。
2. **可无损合并。**$$W + \frac\alpha r BA$$ 就是一个权重矩阵，训完之后**推理零开销**——
   不像 adapter 层会加深度。这才是 LoRA 真正获胜的原因。

**为什么是 $$\alpha/r$$。**这样改 rank 时不用重调学习率。

**诚实的局限。**LoRA 在风格、格式、任务适配上很好，在注入大量新知识上很差——
低秩更新根本没有那个容量。

#### 自测 · A6.11

**Q A6.11.1** — LoRA 省下的显存到底来自哪里？

不是来自权重——基座模型照样要常驻。它来自**优化器状态和梯度**。用 AdamW 做全量微调
每参数约 16 字节（见 A5.1：bf16 权重 2 + bf16 梯度 2 + fp32 主副本 4 + Adam 两个矩各 4）；
用 LoRA 时基座是冻结的，只贡献它那 2 字节的 bf16 权重，剩下的 14 字节只作用在适配器上，
而适配器还不到模型的百分之一。

具体到 70B 模型：1,120 GB 的状态变成大约 140 GB 加一个舍入误差。

激活基本没变——你仍然要走完整个网络的前向——所以 gradient checkpointing 依然值得开。

> **追问**
> - *QLoRA 呢？* → 把冻结的基座量化到 4-bit（NF4），适配器保持较高精度，
>   再加上分页优化器和双重量化。能在单张 48GB 卡上微调 70B。
> - *挂在哪些层上？* → 默认挂注意力的投影矩阵；更难的任务上再加 MLP 的矩阵会有帮助。
>   rank 更高并不可靠地更好——$$r=8$$–$$64$$ 覆盖大部分情况。
>
> **陷阱**
> - 两个矩阵都随机初始化。

---

> **待补概念：**iterated / online DPO、process reward model、
> self-play 与 self-rewarding、对齐税的测量、
> RLHF 的完整口述版（从数据采集到上线）。

---

<a id="section-a7"></a>

## A7 · 推理模型与 test-time compute

★ 全新一节。o1 / R1 之后，这是最高频的话题之一，而它之前散落在 post-training 和 scaling
两节里，没有独立位置。

**这一节的分界线：**能说清 test-time scaling 是**第三条**扩展轴（不是采样技巧），
并且知道**在哪些任务上它不成立**。

---

<a id="a7-1"></a>
### A7.1 第三条扩展轴

前两条轴是**参数**和**数据**。第三条是**推理时的算力**——让模型在回答前生成更多 token。

**为什么它能工作。**Transformer 的每个前向是**固定深度**的计算。一个需要 20 步串行推理的
问题，无法在固定深度里完成。但如果模型把中间结果**写进上下文**，下一次前向就能读到它——
**上下文变成了工作记忆，自回归变成了串行计算的循环**。深度不够，就用长度换。

**它是一条真正的 scaling 律。**准确率随推理 token 数呈近似对数增长，跨越数个数量级。
这不是提示技巧的边际收益，而是和参数/数据同类的曲线。

**三种花掉推理算力的方式：**

| 方式 | 做法 | 特点 |
|---|---|---|
| **串行** | 更长的单条 CoT | 适合深度推理；受长上下文能力限制 |
| **并行** | 采样 $$k$$ 条后选一条 | 易并行；受 selector 质量限制 |
| **搜索** | 树/束搜索 + 过程评分 | 最强也最贵；需要 PRM 或 verifier |

> **并行的关键是 selector，不是采样。**pass@k（存在一条对的）远高于实际准确率
> （**选**出对的）。多数投票、reward model 打分、或可执行验证——这三者的质量决定了
> 并行 scaling 的上限。有 verifier 时并行极强；没有时，它很快就饱和。

#### 自测 · A7.1

**Q A7.1.1** — 从机制上讲，为什么想得更久能让模型更准？

一次前向是固定深度的计算：$$L$$ 层，走一遍。一个所需串行步数超过模型有效深度的问题，
无论参数量多大都不可能在一次前向里解完。

Chain of thought 把**深度换成了长度**。中间结果被写进上下文，下一次前向就能读到，
于是上下文充当工作记忆，自回归生成变成一个串行计算的循环。模型现在可以用 token 作代价，
做任意多的串行步骤。

这也解释了为什么 CoT 在多步问题上（算术、证明、规划）收益最大，在单步检索上几乎没用——
后者根本不需要串行深度。

> **追问**
> - *推理链忠实于真实的计算过程吗？* → 未必。模型可以给出一条并不反映答案真实依据的推理链。
>   有用 ≠ 忠实，而这一点对监控很要紧。
>
> **陷阱**
> - 把 test-time compute 说成「提示技巧」。它是一条和参数、数据同类的扩展曲线。

**Q A7.1.2** — 推理算力什么时候该并行花，什么时候该串行花？

问题需要**深度**时走串行（更长的 CoT）——每一步都依赖上一步，只有一条推理线要往前推。

问题需要**覆盖**时走并行（采 $$k$$ 条）——有好几种看着都行的思路，事先分不出哪条能成；
或者你有延迟预算但没有串行预算。

并行的决定性问题是你有没有 **selector**。有 verifier（单元测试、证明检查器）时并行 scaling
极强——pass@k 就是你实际拿到的。没有的话你只能靠多数投票或 reward model，
收益在 $$k \approx 10$$–$$100$$ 附近就饱和了。

实务上：并行对延迟友好（墙钟只等一条样本）但很吃吞吐；串行反过来。

---

<a id="a7-2"></a>
### A7.2 推理模型是怎么训出来的

**R1-Zero 的证明。**从基座模型直接做 RLVR（可验证奖励的 RL），**没有任何 SFT 冷启动**。
长链推理、自我检查、回溯——全都自己长了出来。响应长度在训练中**自发增长**，
因为更长的推理拿到更高奖励。

**这为什么重要。**它说明推理是**被激发的**，不是**被示范的**。基座模型已经有这个能力，
RL 只是找到了它。这改变了数据策略：你需要的是**可验证的问题**，不是人写的推理过程。

**完整配方**（R1 及之后的常见形态）：

1. **冷启动 SFT**（可选）：少量长 CoT 样本，主要解决可读性和格式，不是能力。
2. **RLVR**：在数学/代码上做大规模 RL，奖励来自执行或答案匹配。
3. **拒绝采样 + SFT**：从 RL 模型采样、筛出正确的，回炉做 SFT，蒸馏进更稳的形态。
4. **通用 RLHF**：恢复非推理任务上的对话质量与安全性。

**蒸馏出奇地有效。**把大推理模型的轨迹 SFT 进小模型，效果**好于**直接在小模型上做 RL。
解释是小模型自己很难探索到好轨迹——RL 需要偶尔成功才能有信号，而蒸馏直接给了成功轨迹。

#### 自测 · A7.2

**Q A7.2.1** — R1-Zero 完全没做 SFT 就学会了推理。这说明什么？

说明这个能力**在基座模型里本来就潜伏着**，RL 是在把它找出来，而不是装进去。基座模型已经能产出
像样的推理文本，它缺的是"稳定地这么做并自我检查"这套策略。一个可验证的奖励就足以把它筛出来。

两个实际后果。第一，数据策略从"收集人写的推理过程"变成"收集可验证的问题"——便宜得多，
也可扩展得多。第二，那些涌现行为（自我纠错、回溯、响应长度自发增长）在奖励里没有任何一处被
指定；它们之所以出现，是因为它们提高了最终答案正确的概率。

> **追问**
> - *那为什么发布版的 R1 还是用了冷启动 SFT？* → 可读性。R1-Zero 的推理链混着多种语言，很难读。
>   那一步 SFT 修的是呈现形式，不是能力。

**Q A7.2.2** — 为什么蒸馏推理轨迹比直接在小模型上跑 RL 更管用？

RL 要有非零奖励才能学。如果一个 7B 模型只能解出 1% 的题，几乎每次 rollout 的 advantage 都是零，
梯度基本不存在——卡住它的是探索问题，不是优化问题。

蒸馏把探索整个绕开了：大模型已经找到了成功轨迹，在这些轨迹上做 SFT 是每个 token 都有的稠密监督。
你转移的是大模型搜索的**结果**，而不是要求小模型自己再搜一遍。

正确的组合通常是先蒸馏（把成功率从地板上抬起来），再在上面做 RL（在它现在偶尔能解的问题上
超过老师）。

> **陷阱**
> - 别说"蒸馏总是更好"。蒸馏的上限是老师；RL 原则上可以超过。正确的说法是蒸馏解决**探索**
>   问题，RL 解决**优化**问题，而在小模型上探索是瓶颈。

---

<a id="a7-3"></a>
### A7.3 推理模型的代价

不是免费的。面试里主动说出代价，比只夸能力更有说服力。

| 代价 | 具体表现 |
|---|---|
| **延迟与成本** | 一个答案可能烧几千到几万 token；首 token 延迟不变但完成时间大增 |
| **KV cache** | 长推理链把 cache 撑大，并发数直接下降（见 A10-08） |
| **过度思考** | 简单问题也生成长推理——这是 RL 学到的"长=好"的副产品 |
| **校准变差** | 长链上的置信度往往更差，而不是更好（见 A13） |
| **不忠实** | 推理链未必反映真实计算过程，因此不能当作可信的监控信号 |

**过度思考（overthinking）是最实际的问题。**因为奖励只看最终正确性，
而更长的推理平均更容易正确，模型学到的是"总是长推理"。修法有：
在奖励里加长度惩罚、训练时混入短答案样本、或像 Qwen3 那样做成可切换模式。

#### 自测 · A7.3

**Q A7.3.1** — 你的推理模型为了回答"2+2 等于几"烧掉 4,000 个 token。为什么，怎么修？

**为什么：**RL 的奖励只看结果。在训练分布（全是难题）上，更长的推理和正确性相关，
于是策略无条件地学到了"就是要长推理"。奖励里没有任何东西告诉它长度是有代价的，
而简单问题在训练里又占比过低。

**修法，大致按我对它们的信任程度排序：**

1. **在奖励里加长度惩罚**——按 token 数减去一项，让模型学会在"再推下去也不会提高成功率"时
   停手。要仔细调：惩罚太强，它会在真正的难题上提前截断。
2. **混合难度的训练数据**，包含那些短答案就是对的简单题，让"短"有时候成为最大化奖励的行为。
3. **一个显式的模式开关**（Qwen3 的 hybrid thinking）——把控制权交给调用方，
   而不是让模型自己去猜该花多少预算。
4. **推理时设预算**——给思考 token 设上限，到点强制出答案。最便宜，但也最粗暴：
   在真正需要这份预算的问题上它就失败了。

> **追问**
> - *难度阈值怎么定？* → 基本上事先定不了，这正是模式开关流行的原因：
>   调用方对这个请求通常比模型知道得更多。
>
> **陷阱**
> - 只夸推理模型的能力不说代价。主动说出过度思考和校准变差，比被问出来强得多。

---

<a id="a7-4"></a>
### A7.4 训练算力 vs 推理算力：怎么分配

给定总预算，多训模型还是多推理？这是一个真实的、正在被研究的权衡。

**基本结论：**对**难**问题，把算力放到推理侧的收益往往更高——一个小模型加大量推理算力，
可以打平一个大得多的模型。对**简单**问题则相反，推理算力很快饱和。

**但要算清账。**训练算力是**一次性**的，推理算力是**每次请求都付**的。所以：

$$\text{总成本} \approx C_{\text{train}} + R \cdot C_{\text{inference-per-request}}$$

请求量 $$R$$ 越大，越应该把算力前移到训练侧。这和 A3.2 里 Llama 3 的逻辑是同一个。

> **一个漂亮的答法。**"这取决于 $$R$$ 和任务难度分布。研究场景（$$R$$ 小、问题难）
> 应该重推理；产品场景（$$R$$ 大、大部分问题简单）应该重训练，并且给难问题留一条
> 可选的重推理路径。"——这正是 hybrid thinking 的产品逻辑。

#### 自测 · A7.4

**Q A7.4.1** — 给定固定的 FLOP 预算，你会训一个更大的模型，还是把它花在推理上？

取决于两件事：**请求量**和**难度分布**。

训练算力只付一次，推理算力每个请求都要付。所以总成本是
$$C_{\text{train}} + R\cdot C_{\text{inf}}$$，$$R$$ 大就该往训练侧压。
这和 Llama 3 那套 inference-optimal 预训练是同一个论证。

难度把结论往另一边推。在难题上，test-time compute 的 scaling 曲线很好看——
一个小模型配上大思考预算，可以打平一个大得多的模型。在简单题上它几乎立刻饱和，钱就白花了。

实务上的综合答案是不做二选一：把那个更小的模型训好，默认用便宜的方式服务，
再给真正需要的请求开一个 thinking 模式。

> **追问**
> - *有 verifier 会改变结论吗？* → 会，而且变化很大。有可靠的 verifier 时，并行的
>   test-time compute 能 scale 得远得多，天平就往推理侧偏。

---

> **待补概念：**process reward model 的训练与使用、
> latent / continuous reasoning（不出 token 的思考）、
> 推理链的可监控性（chain-of-thought monitorability）、
> 推理模型的评测污染问题。
>
> **陷阱**
> - 不问请求量和难度分布就直接回答。这题的答案是「取决于 $$R$$」。

---

<a id="section-a8"></a>

## A8 · 推理与服务

这一节的组织原则只有一条：**prefill 和 decode 是两台不同的机器。**几乎所有服务侧的
设计决策都从这一个区分推出来。量化和长上下文是新增的（★）——Alisa 那本零覆盖。

---

<a id="a8-1"></a>
### A8.1 Prefill 与 decode 是两台机器

**Prefill** 一次处理整个 prompt。每个 token 都要和其他所有 token 交互，
所以每读一字节权重就有大量并行工作。算术强度高 → 落在 roofline 拐点右侧 → **计算受限**。
成本随 $$S$$ **线性**增长（权重矩阵乘），外加一个 $$S^2$$ 的注意力项——
但后者要到 $$S \approx N/(2Ld) \approx 53\text{k}$$ 才追平前者，
2k 的 prompt 上注意力只占 prefill 的约 4%。

**Decode** 生成一个 token。你要读**整个权重矩阵**——几十 GB——只为了算一个 token 的算术。
强度约 1 FLOP/byte，而 H100 的拐点在约 295 → **访存带宽受限**。算术单元几乎全部空闲。

**一个把它坐实的数字。**70B 模型 bf16 共 141 GB 权重，按**一张 H100 的带宽**算，
batch=1 的 decode 有一条硬下界：

$$\frac{1.41\times10^{11}\ \text{bytes}}{3.35\times10^{12}\ \text{bytes/s}} = 42\ \text{ms/token} \approx 24\ \text{tokens/s}$$

**关键在于这个下界由带宽决定，不由算力决定。**同样带宽的更快 GPU 帮不上忙；
把 batch 加大也不会让这一条序列更快。

**但加卡是有用的**——这一点要说准。张量并行把权重切到 $$N$$ 张卡上，每张只读 $$1/N$$
的字节且并行读，所以下界变成 $$(\text{bytes}/N)/\text{带宽}$$ 加上每层 all-reduce 的延迟。
这正是低延迟服务跑 TP=8 的原因。（顺带：141 GB 本来也放不进一张 80 GB 的 H100，
所以 TP≥2 是硬性要求，42 ms 是"一张卡的带宽"这个参照量，不是可实现的配置。）

**其余的一切都从这里推出来：**

| 技术 | 为什么存在 |
|---|---|
| Batching | 把权重读取摊到多条序列上 → decode 的主要杠杆 |
| Continuous batching | 静态 batch 在等最长序列时浪费尾部 |
| Paged KV cache | Cache 才是 batch 大小的限制；连续分配碎片严重 |
| Chunked prefill | 一个超长 prompt 会独占 GPU、毁掉所有人的 TPOT |
| Prefix caching | 共享的系统提示否则每次请求都要重算 |
| 投机解码 | Decode 有闲置 FLOPs；拿去验证草稿 token |
| P/D 分离 | 两个阶段想要的硬件配比不同 |

#### 自测 · A8.1

**Q A8.1.1** — 为什么 prefill 是计算受限、decode 是访存带宽受限？

算术强度。Prefill 读一遍权重换来 $$O(NS)$$ 的计算量，强度高，落在 roofline 拐点右侧。
Decode 完整读一遍每一个权重，只换来一个 token 的算术——强度约 1 FLOP/byte，
而 H100 的拐点在 295 附近——所以算术单元空转，HBM 被打满。

大家常漏掉的推论：batch=1 时 decode 速度就是 $$\text{model bytes} / \text{bandwidth}$$，
**加多少算力都改变不了**——bf16 的 70B 模型对上一张 H100 的 HBM 带宽，就是 42 ms/token。
加*带宽*确实有用，这也是张量并行能砍掉 batch=1 延迟的原因：每张卡只读自己那一片。

> **追问**
> - *batch 要多大才能让 decode 变成计算受限？* → 强度大致随 batch 大小增长，
>   所以 $$B \gtrsim 295$$。实际上 KV cache 会先耗尽，
>   所以 decode 基本上永远是带宽受限的。
>
> **陷阱**
> - 说"加算力能让 decode 更快"。不能——decode 缺的是带宽不是算力。
>   但**别反过来说成"加卡也没用"**：张量并行切的是每张卡要读的字节数，它确实有效。

---

<a id="a8-2"></a>
### A8.2 服务指标：先问要优化哪个

在设计任何东西之前先问这个。三个指标彼此冲突，不存在同时最优。

| 指标 | 定义 | 谁在乎 |
|---|---|---|
| **TTFT** | Time To First Token = 排队 + prefill | 交互式聊天的体感 |
| **TPOT** | Time Per Output Token = decode 每步 | 流式输出的流畅度 |
| **吞吐** | 总 token/s（跨所有请求） | 成本 |
| **Goodput** | **在 SLO 内**完成的请求数 | 真正该优化的那个 |

**为什么它们冲突。**加大 batch 提高吞吐，但每个请求的 TPOT 变差；chunked prefill 保护了
别人的 TPOT，但拉长了这个请求的 TTFT；投机解码降低 TPOT，但在高负载下反而降低吞吐。

> **Goodput 是唯一诚实的指标。**原始吞吐可以很漂亮，同时每个请求都错过延迟目标。
> 面试里主动区分吞吐和 goodput，是"做过线上服务"和"读过博客"的分界线。

#### 自测 · A8.2

**Q A8.2.1** — 你要把 p99 延迟砍掉一半。你会改什么？

先问**是哪个延迟**——TTFT 和 TPOT 的修法几乎不重叠，而 p99 这个指标本身通常指向排队，
而不是这两者中的任何一个。

**如果是 TTFT：**prefix caching（共享前缀的 prefill 整块省掉）、chunked prefill
（不让一个长 prompt 卡住队列）、加副本来压低队列深度，
或者做 prefill/decode 分离，给 prefill 专属算力。

**如果是 TPOT：**减小 batch（直接拿吞吐换延迟）、投机解码、用量化减少每步要读的字节数，
或者用张量并行把权重读取切到多张卡上。

**如果专门是 p99：**那通常是排队或调度问题，不是模型问题。去看准入控制、
看长请求有没有堵住短请求、看 KV cache 满了之后的抢占策略。

> **追问**
> - *最先试哪个最便宜？* → 有共享系统提示的话就是 prefix caching。
>   它几乎不花成本，而且常常能省掉大部分 prefill。

---

<a id="a8-3"></a>
### A8.3 KV cache

**为什么缓存 K/V 而不缓存 Q。**在 decode 第 $$t$$ 步你只有一个 query——新 token 的。
但你需要**全部**历史的 key 和 value 来做注意力。Q 是瞬时的，K/V 是累积的。
没有 cache 就要每步重算所有历史 token 的 K/V，即 $$O(T^2)$$ 的无谓开销。

**大小**

$$\text{bytes/token} = 2 \times L \times K \times H \times \text{每元素字节数}$$

那个 2 是 K 和 V。$$K$$ 是 **KV 头数**，不是 query 头数。

**Llama-3-70B，bf16，GQA 8 个 KV 头**

$$2\times80\times8\times128\times2 = 327{,}680\ \text{bytes} = 320\ \text{KiB/token}$$

128k 上下文下是**单条序列 40 GiB**。用完整 MHA 会是 320 GiB——一次对话就装不下一张卡。

#### 自测 · A8.3

**Q A8.3.1** — 为什么缓存 K 和 V，却不缓存 Q？

因为它们在 decode 第 $$t$$ 步的角色不同。你只有**一个** query——新 token 的——用一次就扔。
但你需要**全部**历史的 key 和 value 来做注意力，而它们和你在更早的步里算出来的完全一样，
因为每个位置的 K 和 V 只依赖那个位置的隐状态。

所以 Q 是瞬时的，K/V 是累积的。没有 cache，你每一步都要把所有历史 token 的 K 和 V 重算一遍，
生成过程变成 $$O(T^2)$$ 的无谓开销。

要说出来的正确性性质：带 cache 的增量 decode 必须和完整重算**在数值上完全一致**。
这就是我会写的那个测试——先跑 teacher forcing，再带 cache 逐 token 生成，断言 `allclose`。

**Q A8.3.2** — 推导 KV cache 的大小，并算出 70B 模型在 128k 上下文下是多少。

每 token $$2 \times L \times K \times H \times \text{bytes}$$：因子 2 来自 K 和 V，$$L$$ 是层数，
$$K$$ 是 **KV** 头数（不是 query 头数），$$H$$ 是头维度。

bf16 的 Llama-3-70B：$$2\times80\times8\times128\times2 = 320$$ KiB/token，所以 128k 上下文是
**单条序列 40 GiB**。这就是为什么 GQA 不是可选项——按 MHA 的 64 个 KV 头算是 320 GiB，
一次对话就要吃掉一个 8×H100 节点的大半，单卡更是差得远。

> **追问**
> - *cache 会改变数学吗？* → 不会。带 cache 的增量 decode 必须和完整重算**在数值上完全一致**。
>   这就是可以主动提出来的测试：先跑 teacher forcing，再带 cache 逐 token 生成，
>   断言 `allclose`。
> - *带 cache 时 mask 有什么微妙之处？* → 你的 query 块从位置 `T_full - T` 开始，不是从 0。
>   直接用 `tril` 是错的，得写 `diagonal=T_full - T`。这个 bug 上线后的表现是
>   "模型 eval 没问题，一生成就变差"。
>
> **陷阱**
> - 用 query 头数算 → 大 8 倍。忘记那个 2。

---

<a id="a8-4"></a>
### A8.4 Continuous batching 与 PagedAttention

**静态 batching 浪费尾部。**固定 batch 下短序列早早结束，槽位空着等最长的那条。
长度差 10× 时你浪费掉大部分容量。

**Continuous batching**（又叫 in-flight batching）在**每个 decode 步**逐出完成的序列、
放入新的，让 batch 始终满载。这是现代服务里最大的单项吞吐收益。

**PagedAttention** 解决显存那一侧。朴素分配按序列**可能的最大**长度预留连续块，
于是一个可能生成 4k 但实际生成 200 token 的请求浪费 95% 的预留，碎片化还会叠加。

修法就是虚拟内存：把 cache 切成固定大小的 **block**（如 16 token），
维护每序列的 block table，按需分配。收益：

- 碎片接近零 → 并发序列数大幅提高。
- **写时复制的前缀共享**：同一 prompt 的多条并行采样，或共享系统提示的多个请求，
  可以共用同一批物理 block。

#### 自测 · A8.4

**Q A8.4.1** — PagedAttention 是一种注意力算法吗？

不是——这题考的就是这个。它是 KV cache 的一种**显存分配策略**，完全不改变注意力的数学。
名字有误导性。

类比就是操作系统的虚拟内存：固定大小的 block、每序列一张 block table、按需分配、
共享用写时复制。它买到的是接近零的碎片，于是能同时装下的序列数上升，吞吐也跟着上升——
因为在 decode 里，正是并发度在摊薄那次权重读取。

> **追问**
> - *cache 还是满了怎么办？* → 你需要一个**抢占策略**：要么之后把被逐出序列的 prefill
>   重算一遍，要么把它的 block 换到主机内存。知道存在这个决策点本身就是很强的信号。
> - *chunked prefill 是干什么的？* → 把长 prompt 切成几块，和 decode 步交错执行，
>   这样一个大请求就没法拖垮所有人的 token 间延迟。

---

<a id="a8-5"></a>
### A8.5 Prefix caching

**想法。**如果很多请求共享一段前缀——系统提示、few-shot 示例、一份长文档——
可以把它的 KV 算一次然后复用。实现上维护一棵 radix/前缀树，配 LRU 逐出。

**什么时候收益巨大。**每次请求带 2,000 token 系统提示、用户轮次 100 token：
你跳过了 95% 的 prefill。多轮对话是另一个大场景——第 $$n$$ 轮和第 $$n-1$$ 轮共享全部历史。

**为什么分页让它成为可能。**连续分配没法共享；固定 block 加写时复制才能跨序列共享物理块。

#### 自测 · A8.5

**Q A8.5.1** — prefix caching 的正确性要求是什么？它对 prompt 设计意味着什么？

前缀必须**逐 token 完全一致**。有一个 token 不同，从那个位置往后的 cache 就全部失效，
因为后面每一个 key 和 value 都依赖它。

对设计的含义很具体：**把会变的部分放到最后**。一个在开头注入时间戳或用户 ID 的模板，
会让每个请求的 cache 都作废。先放静态系统提示，再放 few-shot 示例，最后放用户这一轮。

> **追问**
> - *它能改善 TPOT 吗？* → 不能，只改善 TTFT。它省掉的是 prefill 的工作，不是 decode 的。
> - *为什么各家厂商给 cached input 单独定价？* → 因为这份节省是真实的、量很大，
>   而且很容易归因到具体某个请求上。

---

<a id="a8-6"></a>
### A8.6 投机解码

**机制。**一个小的草稿模型自回归地提出 $$k$$ 个 token。大模型在**一次并行前向**里给这 $$k$$ 个
打分（它们是一条序列，所以这是 prefill 形状的操作）。然后用一条规则逐个接受或拒绝，
使输出分布**精确等于**目标模型的分布。

**为什么它是精确的。**草稿分布 $$q$$、目标分布 $$p$$，以概率 $$\min(1, p(x)/q(x))$$ 接受 token $$x$$；
拒绝时从残差分布 $$\propto \max(0, p(x)-q(x))$$ 采样。这是拒绝采样，可证明产生 $$p$$ 的样本。
**投机解码不是近似**——这一点常让人意外，也是考点。

**为什么它赢。**Decode 是带宽受限的，FLOPs 闲着。验证 $$k$$ 个 token 的墙钟时间约等于生成 1 个，
因为你仍然只读一遍权重。

#### 自测 · A8.6

**Q A8.6.1** — 投机解码什么时候就不再有用了？

batch 变大的时候。它全部的收益都来自带宽受限的 decode 阶段闲置的 FLOPs；
一旦 batch 大到你不再缺带宽，验证就要去抢现在变稀缺的算力，收益缩向零，然后**变成负的**。

所以它是面向交互式、中低负载服务的**延迟**优化，不是吞吐优化。
在一台跑满 batch-256 的服务器上，它通常是错的工具。

> **追问**
> - *草稿模型从哪来？* → 同一家族的小模型；或者目标模型的前几层（self-speculation）；
>   或者 Medusa 那种额外的头；或者代码场景下的 n-gram 查表，因为那里字面重复很常见。
> - *加速比由什么决定？* → 接受率。容易的 token（空白、模板代码）几乎总被接受，
>   难的很少被接受——这就是为什么实测加速比高度依赖负载类型。
>
> **陷阱**
> - 说它是近似方法、会改变输出分布。它不是。

---

<a id="a8-7"></a>
### A8.7 采样

**顺序有影响：**temperature → top-k → top-p。温度改变了截断所作用的那个分布。

```python
def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0:                       # greedy；同时防除零
        return int(logits.argmax())
    logits = logits / temperature

    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        srt, idx = torch.sort(logits, descending=True)
        probs = F.softmax(srt, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        drop = cum - probs >= top_p            # 偏移：保留越过阈值的那个 token
        srt = srt.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, srt)

    return int(torch.multinomial(F.softmax(logits, dim=-1), 1))
```

**两个必须写对的地方：**

- `cum - probs >= top_p` —— 保留的是累积质量**超过** p 的最短前缀，所以越过阈值的那个 token
  要被**包含**。差一位会静默改变采样分布。
- `temperature == 0` 需要显式分支，否则除零。这是真实推理服务里出现过的 bug。

**每个旋钮在做什么。**温度重新缩放 logits，在 argmax（$$\tau\to0$$）和均匀（$$\tau\to\infty$$）
之间插值，**不改变排序**。Top-k 截断到固定数量。Top-p（nucleus）截断到固定概率质量，
所以支撑集大小**随模型置信度自适应**——这就是它通常胜过 top-k 的原因。

#### 自测 · A8.7

**Q A8.7.1** — 实现带 temperature、top-k 和 top-p 的采样。顺序有影响吗？

（代码见上。）顺序有影响：temperature 要放最前，因为它改变了截断所作用的那个分布。
先做 top-p 再做 temperature，等于在错误的分布上挑 nucleus。

被考的两个实现细节：top-p 的那个偏移（`cum - probs >= top_p`，让越过阈值的那个 token
被保留——差一位就会静默改变采样分布），以及给 `temperature == 0` 写一个显式分支来避免除零。

> **追问**
> - *为什么 greedy decoding 会产生重复循环？* → 高概率的续写往往是自我强化的；
>   没有采样噪声，模型就可能进入一个循环。Nucleus 采样当初被提出来，就是为了修这个退化问题。
> - *Beam search——为什么不用在对话上？* → 它最大化的是序列似然，这适合翻译。
>   开放式生成里它产出的文本很平淡，因为似然并不是人想要的东西。
> - *这些会改变校准吗？* → 会。temperature 恰恰就是标准的事后校准旋钮。

---

<a id="a8-8"></a>
### A8.8 FlashAttention

**问题。**朴素注意力在 HBM 里物化一个 $$N\times N$$ 的分数矩阵。长上下文下这既是显存天花板
也是速度天花板，因为注意力是访存受限的。

**想法。**根本不物化它。这要求在没看到全部输入的情况下完成 softmax 归约，
而这是可能的，因为 softmax 有一个重缩放递推——维护 running max $$m$$、分母 $$\ell$$ 和分子，
每当某个 block 揭示出更大的最大值时，按 $$e^{m_\text{old}-m_\text{new}}$$ 重缩放。

```python
m_new = max(m, s.max())
correction = exp(m - m_new)
l   = l * correction + exp(s - m_new).sum()
acc = acc * correction + exp(s - m_new) @ v
```

**要说的三件事：**

1. **它是精确的**，不是近似。与完整 softmax 逐位可比。
2. 显存从 $$O(N^2)$$ 降到 $$O(N)$$。FLOPs 其实**上升**了一点，因为反向要在片上重算注意力，
   而不是读回存好的矩阵。
3. 它仍然更快，因为这个操作原本受限于 **HBM 流量**而不是算术。在 roofline 的访存受限一侧，
   拿 FLOPs 换访存流量是划算的。

#### 自测 · A8.8

**Q A8.8.1** — FlashAttention 的 FLOPs 更多。为什么它反而更快？

因为这个操作从来就不是计算受限的。朴素注意力受限于 HBM 流量——写出再读回一个 $$N\times N$$
的分数矩阵——而 FlashAttention 通过分块、把归约留在 SRAM 里，把这部分流量去掉了。
多出来的 FLOPs 来自反向时在片上重算注意力，而不是读回存好的矩阵；
在 roofline 的访存受限一侧，这笔交换非常划算。

另外要说明它是**精确**的，不是近似；而且它依赖的那个流式 softmax 递推在论文之前就有了
（Milakov & Gimelshein, 2018）。它的贡献是 IO 感知的分块和 kernel 融合，
正是这两点让它在真实硬件上取胜。

> **追问**
> - *它对 decode 有帮助吗？* → 小得多。batch=1 的 decode 里根本没有 $$N\times N$$ 矩阵可以省；
>   瓶颈在读权重和 KV cache。
> - *FA-2/3 改了什么？* → 更好的跨 warp 和 thread block 的工作划分、更少的非矩阵乘 FLOPs，
>   以及在 Hopper 上用异步拷贝和 FP8。
>
> **陷阱**
> - 说它是近似。或者说它"减少了计算量"——它增加了计算量。

---

<a id="a8-9"></a>
### A8.9 ★ 量化

**先说这句。**推理是访存受限的，所以缩小你要读的字节数**直接就是加速**——
这不只是"为了塞进小卡"。

**可以量化的东西，大致按安全程度排序：**

| 目标 | 常见精度 | 效果 |
|---|---|---|
| 权重 | INT8, INT4 | 带宽和显存降 2–4×；收益最大 |
| KV cache | FP8, INT8 | 长上下文并发数翻倍 |
| 激活 | FP8 | 真要用上 INT8 tensor core 就需要它 |
| 梯度 / 优化器 | FP8, 8-bit Adam | 训练侧，另一个问题 |

**PTQ vs QAT。**PTQ 是几乎所有人在做的：拿一个训好的模型，在小数据集上标定 scale，几分钟搞定。
QAT 在训练中模拟量化，能挽回更多质量，但要一次训练。

**朴素 INT8 为什么会坏：outlier feature。**Transformer 激活里有少数通道的量级比其余大
10–100×。单个 per-tensor scale 必须覆盖这些离群值，于是正常值可用的分辨率被压碎。

修法都是"不要共享一个 scale"的变体：

- **更细的粒度**——per-channel、per-group（如 128 个权重）、per-token。
- **LLM.int8()**——离群通道保持 fp16，其余量化。
- **SmoothQuant**——通过 per-channel 重缩放把难度从激活迁移到权重，
  该缩放在数学上被吸收进前一层。
- **GPTQ**——逐层的二阶（Hessian）舍入，最小化输出误差而不是权重误差。
- **AWQ**——保护由激活幅度识别出的、最重要的约 1% 权重。

**真正退化的是什么。**INT8 下困惑度几乎不动，INT4 下动一点。**最先退化的通常是长上下文行为、
推理链和长尾知识**——恰恰是通用语料上的困惑度测不到的东西。所以要在你在意的任务上评测，
不要用 wikitext。

#### 自测 · A8.9

**Q A8.9.1** — 你量化到了 INT4，困惑度几乎没动。这就算完事了吗？

没有。通用语料上的困惑度由高频、容易的 token 主导，而恰恰是这些预测在量化后活得最好。
最先退化的是长上下文行为、多步推理链和长尾事实知识——它们对平均困惑度的贡献都很小。

要在你实际上线的东西上评测：任务榜单、按位置分解的长上下文检索、真实 prompt 上的生成质量。
还要检查失败模式是不是集中的——量化的损伤常常平均看没事，落到某一个切片上就很严重。

> **追问**
> - *FP8 和 INT8 怎么选？* → FP8 有指数位，动态范围处理得更好，需要的标定机制也更少；
>   但它要 Hopper 级别的硬件。INT8 的支持面更广。
> - *量化权重能加速 prefill 吗？* → 比 decode 少得多。Prefill 是计算受限的，
>   所以只有当你真的用低精度去**算**才有收益，仅仅用低精度**存**是不够的。
> - *KV cache 量化呢？* → 长上下文下杠杆最高，而且退化得比较平缓——但 K 比 V 更敏感，
>   所以有些系统对两者做非对称量化。

---

<a id="a8-10"></a>
### A8.10 ★ 长上下文扩展

**开箱为什么不行。**RoPE 的低频分量在 8k 训练中转不满一圈，所以模型从没见过 100k 处出现的
角度。它在一个没有训练信号的区域外推，质量崩塌。

**修法，按复杂度排序：**

1. **Position Interpolation (PI)。**把位置按 $$s = L_\text{new}/L_\text{old}$$ 缩小，
   使 0–128k 映射进训练过的 0–8k。简单；代价是损失一些局部分辨率，
   因为相邻 token 在角度上挤得更近了。
2. **NTK-aware scaling。**不对所有频率一视同仁——高频（局部细节）基本不动，
   低频（全局位置）做插值。保住局部分辨率。
3. **YaRN。**NTK-by-parts 加上注意力 logits 的温度修正（熵随上下文长度增长，
   所以 logits 需要重缩放）。目前最强，而且只需要短暂微调。

这些**都需要在长序列上继续训练**才能真正好用——通常是一个长上下文数据配比的 midtraining 阶段。

**128k 下还会坏什么：**

- **KV cache 显存**——70B 配 GQA 是单序列 40 GiB。这通常才是真正的约束，不是质量。
- **注意力成本**——$$S^2$$；FlashAttention 让显存线性，但计算量不变。
- **Lost in the middle。**检索准确率在上下文的开头和结尾高、中间塌陷。
  一个"支持"128k 的模型未必**用得上**全部 128k。

#### 自测 · A8.10

**Q A8.10.1** — 一个在 8k 上训练的模型要服务 128k。讲讲你会怎么做。

**为什么会坏：**RoPE 的低频分量在 8k 训练里转不满一圈，所以到了 100k，
模型被要求去解释它从没见过的角度。

**修法**是 PI、NTK-aware scaling、YaRN 三者之一——它们都是把角度压回训练过的范围内，
区别在于保住了多少局部分辨率。YaRN 是目前的默认选择，还额外修正了注意力温度，
因为熵随序列长度增长。这些方法都需要在长序列上再训一段；没有一个能纯靠推理时改配置就生效。

**但真正的约束通常是显存，不是质量。**每条序列 40 GiB 的 KV cache，
意味着一个节点上只能并发几个 128k 请求。在扩上下文之前，我会先问产品到底需不需要 128k
的注意力，还是在同样的文档上做检索又便宜又好。

**还有，要按位置评测，不要只看汇总数。**大海捞针只是及格线，而且已经接近饱和；
要用多针版本和 RULER 那类任务，并且永远把准确率报成"信息所在位置"的函数。

> **陷阱**
> - 只答"用 PI 插值"。面试官想听你说出**显存才是真正的约束**，以及 lost-in-the-middle。

---

<a id="a8-11"></a>
### A8.11 Batching、packing 与 padding

**训练侧：packing。**把多份文档拼进一条定长序列，而不是各自 pad 到最长。
Pad 到 batch 内最长可能浪费 50% 以上的算力。

**关键细节：**朴素 packing 下 token 会**跨文档边界**做注意力。两个修法——
块对角注意力 mask（正确，需要 varlen kernel 支持），或者接受这个污染
（历史上很常见，在某些任务上可测地更差）。

**推理侧：continuous batching**（见 A8.4）。Padding 基本被消除，因为序列逐步进出。

#### 自测 · A8.11

**Q A8.11.1** — 变长序列怎么高效地组 batch？

训练侧：把多份文档 **pack** 进定长序列，而不是 padding，并用 varlen kernel
（带 `cu_seqlens` 的 FlashAttention），这样不用物化块对角 mask 就能阻断跨文档注意力。
Pad 到 batch 内最长序列，动辄浪费掉一半算力。

如果 packing 太麻烦，就**按长度分桶**，让同一 batch 里的序列长度接近——
拿到大部分收益，机制简单得多。

推理侧：**continuous batching**，它从结构上消除了 padding，因为序列每一步都在进出。

> **追问**
> - *为什么 batch 的组成会影响 MoE 的输出？* → 专家容量是按 batch 算的，
>   所以哪些 token 被丢掉取决于同一 batch 里还有什么。同样的输入可能产出不同的输出。

---

> **待补概念：**disaggregated prefill/decode 的部署形态、
> structured output / constrained decoding、多 LoRA 服务（S-LoRA）、
> Medusa/EAGLE 变体、CPU offload 与 NVMe、推理的确定性与可复现性。

---

<a id="section-a9"></a>

## A9 · 数据

★ 全新一节。Alisa 那本零覆盖，但数据是前沿实验室真正的护城河——而且它是我自己
`training-data-pipeline` 那篇的压缩版，所以项目深挖大概率会被拽到这里。

**这一节的组织原则：**每一个数据问题其实是同一个问题——**这条监督信号从哪来，
你怎么知道它是对的。**

---

<a id="a9-1"></a>
### A9.1 监督信号的三个来源

一切都归结到三个来源。能把它们命名出来，"数据从哪来"就从一个清单变成了一个结构化回答。

1. **人** —— 示范、偏好、标注。质量最高、扩展性最差，也是唯一能表达**真正新品味**的来源。
2. **模型** —— 合成生成、self-instruct、从更强老师蒸馏、模型写的批评。
   可以任意扩展；天花板是生成模型本身，除非你加上验证。
3. **世界** —— 执行结果、单元测试、编译器、模拟器、搜索结果、真实用户交互。
   唯一能告诉你**没人知道的事**的来源，所以也是唯一能突破老师天花板的那类。

**关键的不对称。**来源 1 和 2 被现有能力界定。来源 3 不是——verifier 可以认证一个
没有人写过、也没有模型能稳定产出的解。这就是为什么 RL 转向可验证领域（代码、数学）：
那里来源 3 便宜。

#### 自测 · A9.1

**Q A9.1.1** — 训练信号最终来自哪里？

三个来源：人（示范、偏好）、模型（合成、蒸馏）、世界（执行、测试、模拟器、真实交互）。

真正重要的是那个不对称。人和模型的信号都被现有能力界定，世界信号不是——verifier 可以认证一个
没人写过、也没有模型能稳定产出的解，这是唯一能越过老师的办法。这就是 RL 集中到代码和数学上的
全部原因：那两个领域的世界信号最便宜。

> **追问**
> - *那为什么不只用来源 3？* → 大多数有价值的任务不可验证。"写一份好摘要"没有 checker。
>   前沿的难题是把验证扩展到不可验证的领域（rubric、裁判、process reward），
>   同时不继承它们的偏差。
>
> **陷阱**
> - 只答"人工标注和合成数据"。漏掉 world / execution 那一类，就漏掉了唯一能突破天花板的那一类。

---

<a id="a9-2"></a>
### A9.2 预训练数据：过滤才是产品

**管线**

1. **采集** —— Common Crawl、代码、书、论文、精选语料。
2. **文本抽取** —— HTML → 文本。被严重低估：boilerplate 去不干净会污染下游一切，
   实际的质量差距很大一部分来自这里。
3. **语种识别与过滤。**
4. **质量过滤** —— 启发式（长度、符号比例、停用词）加分类器过滤
   （用"好"的参考文本 vs 随机爬取训一个分类器）。
5. **去重** —— 精确去重，然后 MinHash/LSH 做文档级近重复，现在越来越多做到子串级。
6. **去污染**，对着评测集。
7. **配比 / 上采样** —— 代码、数学、网页、书、多语言之间的权重。

**哪一步最重要。**去重和质量过滤，而且不接近。文献里一致的发现是
**激进过滤胜过增加原始 token**——FineWeb-Edu 式的分类器过滤（按模型判定的教育质量训练）
产出的模型，在同等算力下打败大得多的未过滤语料。

**为什么去重这么重要。**重复文本会被记住而不是被泛化，浪费算力，还通过污染抬高评测分。
近重复才是难点：同一篇文章被 500 个站点转载，每个的 boilerplate 都不一样。

#### 自测 · A9.2

**Q A9.2.1** — 预训练数据管线里哪一步最重要？

去重和质量过滤，而且不接近。一致的发现是**激进过滤胜过增加原始 token**——同等算力下，
分类器过滤后的语料产出的模型，比一个大得多的未过滤语料更好。

去重之所以重要，是因为重复文本会被记住而不是被泛化，浪费算力，还通过污染抬高评测分。
难点在近重复检测；在真实爬取数据上，精确匹配几乎抓不到什么。

文本抽取也值得提一句——它不起眼，但实际的质量差距有很大一部分能追溯到 boilerplate 清理上。

> **追问**
> - *多 epoch 有多糟？* → 高质量数据上重复到约 4 个 epoch，效果大致等同于新数据；
>   再往后收益迅速崩掉。这就是 data-constrained scaling 成为独立研究方向的原因。
> - *配比权重问题呢？* → 代码/数学/网页的比例是用小的 proxy run 调出来的，各家都当商业机密。
>   上采样代码即使在非代码任务上也能改善推理。
>
> **陷阱**
> - 说"数据越多越好"。前沿的共识是**过滤比加量重要**。

---

<a id="a9-3"></a>
### A9.3 Midtraining：没人写下来的那一阶段

**定义。**预训练和 SFT 之间的一个阶段：在**刻意重新加权的、更高质量的混合数据**上继续预训练，
通常配一段学习率衰减。

**用来做什么：**

- **长上下文扩展** —— 8k → 128k 实际发生在这里，用长文档配比。
- **领域注入** —— 大幅上采样代码、数学、推理轨迹。
- **质量退火** —— 用你最好的数据结束训练，让最终权重被它塑造。
- **多语言再平衡。**

**为什么要单独一个阶段。**两个理由。第一，你没法用优质数据跑完整个预训练——它不够多。
第二，**学习率调度让顺序变得重要**：最后衰减阶段见到的数据影响力超常，所以你想把最好的数据放最后。

**和 LR 调度的联系。**这就是 WSD（warmup-stable-decay）取代 cosine 流行起来的原因：
有一段恒定的 stable 阶段，你可以在任意点分叉出一段衰减，
于是 midtraining 变成一个可重复的操作，而不是在第 0 步就定死的一次性决定。

#### 自测 · A9.3

**Q A9.3.1** — Midtraining 为什么要单独成一个阶段，而不是并进预训练？

两个理由，第二个才有意思。

**供给**：优质数据不够多，撑不起整个预训练。

**顺序**：学习率调度让数据出现在训练的哪个位置变得重要。最后衰减阶段见到的数据对最终权重
影响超常，所以你想把最好的数据放在最后。这也是 WSD 调度取代 cosine 的原因——有一段恒定的
stable 阶段，你可以在任意点分叉出一段衰减，于是 midtraining 变成一个可重复的操作，
而不是在第 0 步就定死的决定。

> **追问**
> - *怎么知道它起作用了？* → 目标领域的 held-out loss 加针对性 benchmark，
>   还有最关键的一条：确认通用能力没有退化——灾难性遗忘就是在这个阶段冒出来的。
>
> **陷阱**
> - 把 midtraining 和 SFT 混为一谈。它仍然是**语言建模目标**。

---

<a id="a9-4"></a>
### A9.4 SFT 数据：一道就绪门，不是能力来源

**重新框定。**SFT 不教能力——基座模型已经有了。SFT 教的是
**格式、指令遵循、工具调用语法**：它让潜在能力变得可及。

支持这个框架的证据是 LIMA 式结果：**少量**（千量级）非常高质量、多样的示范就能走完大部分路。
质量和多样性以巨大优势压过数量。

**SFT 数据必须覆盖什么** —— 把它当作覆盖问题，不是体量问题：

- 你需要的每一种**响应格式**（JSON、代码块、工具调用、拒绝）。
- 每一种**轮次结构**（单轮、多轮、带工具结果的多轮）。
- **边界行为**：拒绝、要求澄清、承认不知道。

**SFT 做不到什么。**它只能模仿。没被示范过的行为，SFT 产不出来。
而且因为它是纯模仿，它有**曝光偏差**：模型只见过金标准前缀，
从来学不会从自己的错误中恢复。这正是 RL 要补的缺口。

#### 自测 · A9.4

**Q A9.4.1** — SFT 数据需要多少？

比大多数人以为的少，而且"多少"本身就是个错的问题——该问的是覆盖。千量级的高质量、
多样示范就能走完大部分路，因为 SFT 不是在安装能力，而是通过格式和指令遵循
让已有的潜在能力变得可及。

所以我会把它当作一张覆盖矩阵来规划：我需要输出的每一种响应格式、每一种轮次结构，
以及那些边界行为（拒绝、要求澄清、承认不知道）——只采集成功完成的任务，这些永远不会出现。

> **追问**
> - *怎么造？* → 主要靠模型生成再过滤，人写种子并做审核。在所需的多样性下，
>   全人工撰写的 SFT 数据已经不经济了。
> - *多轮怎么办？* → mask 掉所有用户轮，在**所有**助手轮上算 loss，不是只算最后一轮。
>
> **陷阱**
> - 说"SFT 数据越多越好"。

---

<a id="a9-5"></a>
### A9.5 RL 数据是题目，不是答案

**关键的重新框定。**做 RLVR 你**不**需要通常意义上的答案。你需要：

- 一个 **prompt**，
- 一个能给完成打分的 **verifier**，
- 以及（数学/代码场景）一个只被 verifier 使用的**参考答案或测试套件**。

模型自己生成轨迹。所以数据集是一堆*题目*，不是一堆*解答*——
这彻底改变了"采集数据"的含义。

**Prompt 选择才是全部，因为方差论证。**当前策略下成功率为 $$\hat p$$ 的任务，
二值结果的方差是 $$\hat p(1-\hat p)$$ —— **在 $$\hat p = 0.5$$ 处最大，在两端为零**。
策略总失败（$$\hat p=0$$）和总成功（$$\hat p=1$$）的任务对梯度的贡献**都是零**。

在 GRPO 里这是字面意义的：一组里所有完成拿到相同奖励时，advantage 精确为 0，这组是白烧的算力。
DAPO 的**动态采样**就是为此存在——重采样直到一组内有奖励方差。

**所以实用配方是按难度做课程**：持续估计每个 prompt 的成功率，
把 prompt 维持在 50% 附近，淘汰已解决的，搁置不可能的。

#### 自测 · A9.5

**Q A9.5.1** — 一个 RL 数据集由什么组成，prompt 怎么选？

prompt 加 verifier，不是 prompt 加答案——模型自己生成轨迹，所以你采集的是*题目*，不是*解答*。

prompt 的选择由方差论证决定：成功概率为 $$\hat p$$ 的二值结果，方差是 $$\hat p(1-\hat p)$$，
在 0.5 处最大、**在两端为零**。策略总是失败的任务和总是解决的任务都不贡献任何东西。
在 GRPO 里这是字面意义的——一组奖励全相同时 advantage 精确为 0。

所以配方是一条难度课程：持续跟踪每个 prompt 的成功率，把 prompt 维持在 50% 附近，
淘汰已解决的，搁置不可能的，再用动态重采样避免把 rollout 浪费在全平局的组上。

> **追问**
> - *"难度 ≠ 可训练性"是什么意思？* → 一个任务可以因为产生不了学习信号的原因而变难——
>   规范含糊、verifier 坏了、需要模型没有的知识。可训练意味着*既难又有信息量*，
>   这是一个严格更小的集合。
>
> **陷阱**
> - 说 RL 需要"高质量答案"。RLVR 需要的是**可验证的题目**。

---

<a id="a9-6"></a>
### A9.6 验证阶梯

**从最强到最弱的信号：**

1. **精确 / 程序化验证。**单元测试、编译器、符号数学检查器、模拟器。
   确定性、运行便宜、在通常意义上无法被钻空子。
2. **受约束的验证。**答案必须匹配某个规范形式（最终数值、正则、schema）。
   弱于 (1)，因为*过程*没被检查。
3. **基于 rubric 的 LLM 裁判。**带显式检查表的裁判模型。可扩展到不可验证领域；继承裁判的偏差。
4. **偏好比较。**成对，人或模型。排序可靠，没有绝对尺度。
5. **启发式。**长度、格式、关键词。快且极易被玩弄——只能当过滤器，绝不能当奖励。

**经验法则：**领域允许的话尽量往上爬；爬不上去时，用几个**失效方式互不相关**的弱信号，
而不是一个看起来很强的单一信号。

**每一级都有的陷阱：推理无效但答案正确。**结果验证看不见它。
这就是 process reward model 存在的原因。

#### 自测 · A9.6

**Q A9.6.1** — 你要大规模地给模型输出打分，有哪些选择？

（阶梯见上。）真正重要的框架是：领域允许爬多高就爬多高；爬不到顶时，宁可用
**几个失效方式互不相关的弱信号**，也不要用一个看起来很强的单一信号。
单一裁判模型离被玩弄只差一个相关的失效模式。

再点名每一级都存在的那个陷阱——推理无效但最终答案正确。结果验证在结构上就看不见它。

> **追问**
> - *怎么发现 verifier 被钻了空子？* → 留一批模型从不训练的测试；手工读奖励最高的那些轨迹
>   （这招见效最快）；盯住奖励在涨而 held-out 指标走平的情形。
>
> **陷阱**
> - 只说 unit test。要能给出没有 verifier 时的降级路径。

---

<a id="a9-7"></a>
### A9.7 Agent 级数据

**四种不同的产物，混为一谈是最常见的困惑：**

| 产物 | 是什么 | 谁生产 |
|---|---|---|
| **Environment** | 可执行的世界：文件系统、API、浏览器、模拟器 | 工程 |
| **Task** | 该环境内的一个目标 + 初始状态 + 成功条件 | 生成 + 过滤 |
| **Rubric / verifier** | 你如何判定任务完成了 | 工程，每个环境一套 |
| **Trajectory** | 一次 rollout：观察、动作、工具结果、结果 | 策略，在训练时产生 |

**瓶颈是 environment，不是 task。**环境一旦存在，任务可以廉价生成；
环境是定制工程。这就是"environment scaling"成为独立研究方向的原因——
这个领域卡在可执行世界上，不是卡在算法上。

**管线**是 Generate → Build → Verify → Filter → Evolve：合成候选任务、在环境里实例化、
检查它确实可解且确实可判定、丢掉不合格的、把幸存的往策略能力的边界上变异。

#### 自测 · A9.7

**Q A9.7.1** — 轨迹数据和 SFT 数据有什么不同？

它是 **on-policy** 的：策略一变它就过期。SFT 数据是一份采集一次、跨多个 run 反复使用的资产；
轨迹数据是当前策略的易腐副产品，攒不下来。这就是 agentic RL 的核心经济学，
也是成本中心落在 rollout 基础设施而不是数据采集上的原因。

第二个不同是：你除了验证结果，还必须验证**任务本身**。一个成功条件坏掉的生成任务
产出的是纯噪声，而在规模上这类候选占比不小。

> **追问**
> - *长时程的 credit assignment 怎么办？* → 几百次工具调用共用一个结果奖励。可选项有
>   中间步骤的 process reward、学出来的 critic、hindsight relabelling——没有一个是彻底解决的。
>
> **陷阱**
> - 把 task 和 environment 混为一谈。瓶颈在 environment。

---

<a id="a9-8"></a>
### A9.8 合成数据什么时候坍塌

**坍塌结论。**反复在自己的输出上训练、没有外部信号，会退化模型——
分布的尾部先消失，然后模型收敛到一个窄的、低方差的输出分布。
机制很直白：采样丢失尾部质量，在样本上训练把这个损失固化下来。

**合成数据什么时候安全——条件是外部锚定：**

| 设置 | 安全？ | 为什么 |
|---|---|---|
| 自生成、自训练、不过滤 | **否** | 纯坍塌 |
| 自生成 + **verifier 过滤** | **是** | Verifier 是外部信号（来源 3） |
| 从**更强**的老师蒸馏 | 是，上限是老师 | 外部信号 = 老师 |
| 生成 + 人工审核 | 是 | 外部信号 = 人 |
| 合成与新鲜真实数据混合 | 大体是 | 真实数据补充尾部 |

**统一原则：**合成数据是对你**已经拥有**的信息做**重组**。
只有当某个外部的东西——verifier、更强的模型、人、世界——进入循环，它才增加信息。

#### 自测 · A9.8

**Q A9.8.1** — 合成数据什么时候会导致 model collapse？

当循环里没有外部信号的时候。自己生成、自己训练、不做过滤——这就是纯粹的坍塌：
采样丢掉尾部质量，再在样本上训练把这个损失固化下来。

统一的说法是：合成数据**重组**你已经拥有的信息；只有当某个外部的东西进入循环——
verifier、更强的老师、人、或者世界——它才增加信息。

这也解释了为什么有坍塌这个结论、合成数据仍然遍地都是：重组本身确实有价值。
把原始文本变成指令-响应对、推理轨迹或多轮对话，是一次有真实价值的格式转换，
它并没有做出坍塌所警告的那种知识主张。

> **陷阱**
> - 一句"合成数据会导致 model collapse"就结束。条件是**没有外部信号**。

---

<a id="a9-9"></a>
### A9.9 污染

**它怎么发生的** —— 通常不是因为粗心：

- Benchmark 在你爬取之前就发表了，所以它字面上就在网页数据里。
- 有人把解答贴在 GitHub / StackOverflow / 博客上。
- 由一个自己见过该 benchmark 的模型生成的合成数据。
- **间接污染**：评测的*源材料*（SWE-bench 背后的 GitHub 仓库）在语料里，
  即使任务格式不在。

**检测**

- **N-gram 重叠**，评测项与语料之间。便宜，抓字面复制，抓不到改写。
- **Canary 字符串**植入评测集；模型能补全它就说明见过。
- **困惑度差**：在评测上的 loss 相对同类未见文本异常低。
- **行为上的破绽**：在公开划分上表现很强，在同分布新采集的私有划分上明显弱。
  这是实践中最可靠的信号。

**真正重要的框架。**你的评测集是数据管线的一部分——去污染是一个**管线步骤**，
不是事后想起来的事，而且它必须在每次训练前、对每个你在乎的评测跑一遍。

#### 自测 · A9.9

**Q A9.9.1** — 能做到彻底去污染吗？

不能。字面匹配你可以删掉，n-gram 重叠也能抓到那些。但你删不掉这个事实：
模型读过一篇讲解这道题答案的博客；或者某个 agent benchmark 背后的源仓库就在语料里，
哪怕任务格式不在。

唯一站得住的答案是**在训练截止日期之后创建的 held-out 集**，这也是各家实验室越来越多
自建私有、定期刷新评测的原因。其余都只是缓解。

值得补一句：污染让**测量**失效，不一定让模型变差。一个被污染的 benchmark 什么都说明不了，
但这不代表模型退步了。

> **陷阱**
> - 只说 n-gram overlap。要提到 held-out-after-cutoff 才是唯一可持续的做法。

---

> **待补概念：**数据配比的实验方法（proxy model / scaling law for mixtures）、
> 多语言数据、代码数据的特殊处理、长文档的构造、
> 隐私与 PII、数据版权与许可、data attribution。

---

<a id="section-a10"></a>

## A10 · 估算题

这一节全是**当场算**的题。它在 rapid-fire 轮里出现频率很高，而且几乎没有现成的练习材料——
Alisa 的笔记里公式很全，但一道习题都没有。所以这一节的题基本是新出的。

全节统一用 **Llama-3-70B** 的真实配置做算例，这样数字可以互相印证：

| 符号 | 含义 | Llama-3-70B |
|---|---|---|
| $$L$$ | 层数 | 80 |
| $$D$$ | hidden size / d_model | 8192 |
| $$N$$ | query 头数 | 64 |
| $$K$$ | KV 头数（GQA） | 8 |
| $$H$$ | head_dim | 128 |
| $$F$$ | FFN 中间维 | 28672 |
| $$V$$ | 词表 | 128256 |

> **一个通用建议：算之前先说单位。**面试里最常见的翻车不是算错，是把 bit 和 byte、
> 把 query 头和 KV 头、把每 token 和每序列搞混。开口第一句先把单位钉死。
>
> **本节的单位约定。**显存一律用**二进制**单位（GiB $$=2^{30}$$ bytes），因为
> `nvidia-smi` 和"放不放得下"这个问题用的都是它。带宽和 FLOP/s 一律用**十进制**
> （TB/s $$=10^{12}$$ bytes/s），因为硬件规格书是这么标的。两者混用会带来 7% 的误差
> —— 单看不致命，但连乘几次之后就会让你的结论跑偏，而且面试官会注意到。

---

<a id="a10-0"></a>
### A10.0 四个锚点数字与三条公式

这一节唯一的"概念"部分。**记住四个数字加三条公式，几乎所有估算题都能当场推出来**，
不需要背任何具体模型的参数表。

**四个锚点数字**（记这四个，其余靠推）：

| 量 | 数值 | 怎么用 |
|---|---|---|
| H100 峰值算力（bf16 dense） | $$\approx 1\times10^{15}$$ FLOP/s | 上限；实际乘 MFU |
| H100 HBM 带宽 | $$3.35$$ TB/s | decode 速度的分母 |
| H100 显存 | 80 GB | 放不放得下的分子 |
| 一天的秒数 | $$8.64\times10^{4}$$ | 把 FLOP/s 换成总预算 |

**三条公式：**

$$\underbrace{C = 6ND}_{\text{训练 FLOPs}}\qquad
\underbrace{2N}_{\text{推理每 token FLOPs}}\qquad
\underbrace{2LKH\times b}_{\text{KV cache 每 token 字节}}$$

- **$$6ND$$**：$$2ND$$ 前向 + $$4ND$$ 反向（反向是前向的两倍：要算输入梯度和权重梯度）。
- **$$2N$$**：每个参数一次乘一次加。
- **$$2LKH b$$**：2 是 K 和 V，$$K$$ 是 **KV** 头数，$$b$$ 是每元素字节数。

**估算的四步套路**（照这个顺序说，不容易漏）：

1. **说单位**——GiB 还是 GB，per token 还是 per sequence。
2. **写公式**——先符号后数字，这样错了也能看出是代入错还是理解错。
3. **代数量级**——用 $$10^x$$ 心算，不要追求有效数字。
4. **回头做常识检查**——"70B 模型 140 GB 权重，装不下一张 80 GB 卡"这类判断要能立刻给出。

> **面试里真正被评估的不是算术，是你会不会检查自己的答案。**算完之后主动说一句
> "这个数量级合理吗"，比小数点后两位准确重要得多。

**Q A10.0.1** — 为什么训练是 $$6ND$$，而推理每 token 只有 $$2N$$？

前向每 token 是 $$2N$$：每个参数参与一次乘、一次加。反向是前向的两倍，因为你要算两组梯度——
对输入的（继续往回传）和对权重的（更新这一层）。所以每 token 是 $$2N + 4N = 6N$$，
$$D$$ 个 token 就是 $$6ND$$。

有两个前提值得说出来：它不含 attention 的 $$S^2$$ 项，长上下文下这一项不可忽略；
MoE 的话，$$N$$ 指的是**激活**参数量，不是总参数量。

---

#### A10-01 · 推导 decoder-only LM 的参数量

`参数量` `高频` `必背`

**Q.** 用 $$V, D, L, F$$ 推导一个标准 decoder-only Transformer 的总参数量。
然后化简成常用的、只含 $$V, D, L$$ 的近似式。

**逐块数。**

Embedding：$$VD$$。Unembedding（lm_head）：$$VD$$。两者共 $$2VD$$。

每层 attention（标准 MHA，即 $$K=N$$）：

$$W_Q: (D,D),\quad W_K: (D,KH),\quad W_V: (D,KH),\quad W_O: (D,D)$$

$$\text{attn} = 2D^2 + 2DKH \;\xrightarrow{\;K=N,\; NH=D\;}\; 4D^2$$

每层 FFN（SwiGLU，**三个**矩阵，不是两个）：

$$W_\text{up}: (D,F),\quad W_\text{gate}: (D,F),\quad W_\text{down}: (F,D) \;\Rightarrow\; 3DF$$

每层 norm：pre-attn 和 pre-FFN 各一个 RMSNorm，每个 $$D$$ 个 $$\gamma$$，共 $$2D$$。

**合起来：**

$$P = 2VD + L\,(4D^2 + 3DF + 2D)$$

**化简。**取常见的 $$F = \tfrac{8}{3}D$$（SwiGLU 为了在三矩阵下保持和 $$4D$$ 的两矩阵 FFN
相同的参数量而选的比例），则 $$3DF = 8D^2$$：

$$P \approx 2VD + 12LD^2$$

那个 **12** 就是这么来的：4（attention）+ 8（FFN）。$$2D$$ 的 norm 项相对 $$D^2$$ 可忽略。

> **追问**
> - 为什么 SwiGLU 要用 $$F=\tfrac83 D$$ 而不是 $$4D$$？→ 因为它有三个矩阵而不是两个，
>   $$3D\cdot\tfrac83 D = 8D^2 = 2D\cdot 4D$$，参数量持平。
> - weight tying（embedding 和 unembedding 共享）能省多少？→ $$VD$$。对小模型很可观：
>   $$V=128256, D=2048$$ 时是 2.6 亿参数，可能占整个模型的 15% 以上。大模型上占比很小。
> - GQA 对参数量的影响？→ 只影响 $$2DKH$$ 那一项。Llama-3-70B 里 $$K=8$$ 而不是 64，
>   K/V 投影从 $$2D^2$$ 降到 $$2D\cdot 1024$$，每层省 $$1.17\times10^8$$ → 全模型省 **94 亿**。
>   下一题会用完整配置再验一次这个数。
>
> **陷阱**
> - FFN 写成 $$2DF$$。SwiGLU 是三个矩阵。
> - 忘记 unembedding，只算一个 $$VD$$。
> - 把 GQA 的 K/V 投影仍按 $$(D,D)$$ 算。


---

#### A10-02 · 验算一下：Llama-3-70B 真的是 70B 吗？

`参数量` `实算`

**Q.** 用上面那张配置表算出参数量，验证它确实落在 70B 附近。

**Embedding + unembedding**

$$2VD = 2 \times 128256 \times 8192 = 2.10 \times 10^9$$

**每层 attention**（注意 GQA：$$KH = 8\times128 = 1024$$）

$$\underbrace{8192^2}_{W_Q} + \underbrace{8192\times1024}_{W_K} + \underbrace{8192\times1024}_{W_V} + \underbrace{8192^2}_{W_O}$$

$$= 2(6.71\times10^7) + 2(8.39\times10^6) = 1.51\times 10^8$$

**每层 FFN**

$$3DF = 3 \times 8192 \times 28672 = 7.05\times 10^8$$

**每层合计** $$\approx 8.56\times10^8$$，乘以 80 层：

$$80 \times 8.56\times10^8 = 6.85\times10^{10}$$

**总计**

$$6.85\times10^{10} + 2.10\times10^9 = 7.06\times10^{10} \approx \mathbf{70.6B}\;\checkmark$$

注意 **FFN 占了每层的 82%**（7.05 / 8.56）。这是一个值得随口说出来的直觉：
现代 LLM 的参数绝大部分在 FFN 里，不在 attention 里。

> **追问**
> - 如果不用 GQA 而用完整 MHA，参数量变多少？→ K/V 投影从 $$2\times8192\times1024$$ 变成
>   $$2\times8192\times8192$$，每层多 $$1.17\times10^8$$，全模型多 **94 亿** → 约 80B。
>   所以相对 MHA 的 80B 基线，GQA 省了约 **12%** 的参数（$$9.4/80$$；
>   别拿 $$9.4/70.6=13\%$$ 去报，那是除以省完之后的大小）。
>   顺带把 KV cache 砍到 1/8，那才是重点。
> - 为什么 FFN 占这么大？→ $$F/D = 3.5$$，而 attention 的四个矩阵都是 $$O(D^2)$$ 量级。
>
> **陷阱**
> - 用 $$N=64$$ 去算 K/V 投影（那是 query 头数）。GQA 下 K/V 用的是 $$K=8$$。


---

#### A10-03 · 每层的激活显存

`激活` `显存`

**Q.** 用 $$B,S,D,N,F$$ 推导每个 Transformer 层为反向传播必须保留多少激活显存。
序列很长时哪一项主导？

**attention 部分**

| 张量 | 形状 | 大小 |
|---|---|---|
| norm 输入 | $$(B,S,D)$$ | $$BSD$$ |
| norm 输出 | $$(B,S,D)$$ | $$BSD$$ |
| Q, K, V | 各 $$(B,S,D)$$ | $$3BSD$$ |
| attention 分数 | $$(B,N,S,S)$$ | $$BNS^2$$ |
| attention 输出 | $$(B,S,D)$$ | $$BSD$$ |

小计 $$\approx 6BSD + BNS^2$$

**FFN 部分**

norm 输入 $$BSD$$，gate/up 输出各 $$BSF$$，down 输出 $$BSD$$
→ $$2BSD + 2BSF \xrightarrow{F=8D/3} 2BSD + \tfrac{16}{3}BSD \approx 8BSD$$

**每层合计**

$$\boxed{14BSD + BNS^2}$$

**哪一项主导？**两项之比：

$$\frac{BNS^2}{14BSD} = \frac{NS}{14D}$$

代入 $$N=64, D=8192$$：当 $$S > 14D/N = 1792$$ 时，$$S^2$$ 项开始主导。也就是说
**超过约 1.8k 上下文，attention 矩阵就是激活显存的大头**——而这正是 FlashAttention 的动机。

用了 FlashAttention 之后 $$S\times S$$ 矩阵不再物化，第二项从 $$BNS^2$$ 降到 $$O(BNS)$$，
激活显存重新变成随 $$BS$$（总 token 数）线性增长。

> **追问**
> - 梯度检查点（activation recomputation）能省多少，代价多少？→ 只存每层边界的激活，
>   反向时重算层内部分。显存从 $$O(L)$$ 降到 $$O(\sqrt L)$$ 或 $$O(1)$$（看策略），
>   代价约多 30% 的算力（多一次前向）。
> - 为什么 dropout mask 也要算激活？→ 反向要用同一个 mask，得存下来（通常按 bool/bit 存）。
>
> **陷阱**
> - 忘掉 $$BNS^2$$ 用的是 **query 头数 $$N$$**，不是 KV 头数——GQA 不减少 attention 矩阵大小。


---

#### A10-04 · 前向传播的 FLOPs

`FLOPs` `必背`

**Q.** 推导一次前向传播的 FLOPs。为什么说反向是前向的 2×？

**基本单位：**一次 $$(m,k)\times(k,n)$$ 的矩阵乘是 $$2mkn$$ FLOPs（每个输出元素做 $$k$$ 次
乘加，乘和加各算一次）。那个 **2** 是所有 FLOPs 估算的来源。

**每层 attention**

| 运算 | 形状 | FLOPs |
|---|---|---|
| Q 投影 | $$(B,S,D)\times(D,D)$$ | $$2BSD^2$$ |
| K 投影 | $$(B,S,D)\times(D,D)$$ | $$2BSD^2$$ |
| V 投影 | 同上 | $$2BSD^2$$ |
| $$QK^\top$$ | $$(B,N,S,H)\times(B,N,H,S)$$ | $$2BNS^2H = 2BS^2D$$ |
| $$AV$$ | $$(B,N,S,S)\times(B,N,S,H)$$ | $$2BS^2D$$ |
| O 投影 | $$(B,S,D)\times(D,D)$$ | $$2BSD^2$$ |

小计 $$= 8BSD^2 + 4BS^2D$$

**这套推导是 MHA。**GQA 下 K、V 投影到的是 $$K_{kv}H$$ 而不是 $$D$$，每个只要
$$2BSD\,K_{kv}H$$——Llama-3-70B 上是 $$D/(K_{kv}H) = 8192/1024 = 8$$ 倍的差距，
attention 小计降到 $$4.5BSD^2 + 4BS^2D$$。它改的是 $$BSD^2$$ 前面的系数，
$$4BS^2D$$ 那一项不动——这正是 A2.3 里那个不对称：GQA 缩的是投影和 KV cache，
从来不是 attention 矩阵本身。答 $$24BSD^2$$ 时要说清楚你假设的是哪一种。

**每层 FFN**：三个矩阵各 $$2BSDF$$ → $$6BSDF \xrightarrow{F=8D/3} 16BSD^2$$

**每层合计** $$= 24BSD^2 + 4BS^2D = 2BSD(12D + 2S)$$

**加上 unembedding** $$2BSDV$$，全模型：

$$\text{FLOPs}_\text{fwd} = 2BSD\,(12LD + 2LS + V)$$

**为什么反向是 2×？**每一层的反向要算两个矩阵乘而不是一个：

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z}W^\top \quad\text{（传给上一层）}$$
$$\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Z} \quad\text{（更新这一层）}$$

两个矩阵乘，规模都和前向那一个相同 → 反向 ≈ 2× 前向，前向+反向 ≈ **3× 前向**。

> **追问**
> - 用了梯度检查点之后呢？→ 反向时要重做一次前向，总量变成 4× 前向（1 前向 + 1 重算 + 2 反向）。
> - 为什么 attention 的 $$QK^\top$$ 和 $$AV$$ 不受 GQA 影响？→ K/V 会被 `repeat_interleave`
>   扩回 $$N$$ 个头再算，GQA 省的是**显存和带宽**，不是 FLOPs。
>
> **陷阱**
> - 漏掉那个 2（把矩阵乘算成 $$mkn$$）。
> - 忘记 unembedding —— 对小模型、大词表它占比可观。


---

#### A10-05 · $$6ND$$ 是怎么来的？

`FLOPs` `MFU` `★ 补充`

**Q.** 大家都用 $$C\approx 6ND$$ 估训练算力。这个 6 是怎么来的，这个近似在什么时候失效？

**来源。**每个参数在前向里参与**一次乘加** = 2 FLOPs。所以前向对每个 token 是 $$2N$$ FLOPs。
反向是 2×（见上题），所以：

$$\text{前向} + \text{反向} = 2N + 4N = 6N \;\text{FLOPs / token}$$

乘以总 token 数 $$D$$：

$$C \approx 6ND$$

**和上一题的公式对得上吗？**对得上。上题里非 attention 部分的前向是 $$24LBSD^2$$，
而 $$N \approx 12LD^2$$、token 数 $$= BS$$，所以 $$2N\cdot BS = 24LBSD^2$$ ✓。

**什么时候不准。**$$6ND$$ **忽略了 attention 里的 $$4BS^2D$$ 项**，因为那一项不含参数。
它相对非 attention 部分的占比是：

$$\frac{4BS^2D \cdot L}{24LBSD^2} = \frac{S}{6D}$$

所以当 $$S > 6D$$ 时 attention 的算力开始不可忽略。$$D=8192$$ 时临界点是
$$S \approx 49{,}000$$ —— **短上下文下 $$6ND$$ 很准，长上下文下它会低估**。

其他不准的来源：梯度检查点（→ $$8ND$$）、MoE（只有激活的专家参与，$$N$$ 要用
activated params 而不是 total params）。

> **追问**
> - MoE 模型怎么算？→ 用 activated parameters。DeepSeek-V3 总共 671B 但每 token 只激活 37B，
>   所以算力按 37B 走，显存按 671B 走。
> - 那 MFU 怎么算？→ 见下一题。
>
> **陷阱**
> - 把 $$6ND$$ 用在长上下文训练上还以为很准。


---

#### A10-06 · 算 MFU，以及它低了该查什么

`MFU` `★ 补充` `高频`

**Q.** 定义 MFU。对一个具体配置算出来，然后说说如果结果只有 20%，你会按什么顺序去查。

**定义。**Model FLOPs Utilization = 实际达到的模型 FLOP/s ÷ 硬件峰值 FLOP/s。

$$\text{MFU} = \frac{6N \cdot (\text{tokens/s})}{\text{GPU 数} \times \text{单卡峰值 FLOP/s}}$$

注意分子用的是**模型必需的** FLOPs（$$6N$$），不含重算、不含通信。所以梯度检查点会
**降低** MFU 却可能**提高**实际吞吐——这是一个很好的追问点。区别于 HFU（Hardware FLOPs
Utilization），后者把重算也算进分子。

**算例。**70B 模型，1024 张 H100（bf16 峰值 989 TFLOP/s），实测 12,000 tokens/s：

$$\text{分子} = 6 \times 7.06\times10^{10} \times 12000 = 5.08\times10^{15}\ \text{FLOP/s}$$

$$\text{分母} = 1024 \times 9.89\times10^{14} = 1.01\times10^{18}\ \text{FLOP/s}$$

$$\text{MFU} = \frac{5.08\times10^{15}}{1.01\times10^{18}} = \mathbf{0.50\%}$$

这个数字低得离谱 —— 说明这个假想场景里吞吐远远不够。反过来推：要达到 40% MFU，
需要 tokens/s $$= 0.40 \times 1.01\times10^{18} / (6\times7.06\times10^{10}) \approx 9.5\times10^5$$，
即约 **95 万 tokens/s**。这就是为什么前沿训练动辄几万亿 token 也只要几周。

**大规模训练的健康区间是 35–50%。**低于 30% 一般说明有具体问题。

**低了按这个顺序查：**

1. **通信没和计算重叠。**最常见。检查 DP 的 all-reduce 有没有和反向重叠、
   ZeRO-3 的参数 gather 有没有预取。
2. **Pipeline bubble。**$$p$$ 段、$$m$$ 个 micro-batch 时，空闲占墙钟的比例约
   $$(p-1)/(m+p-1)$$，$$p=m=8$$ 就是 47% 的浪费。（Megatron 报的 $$(p-1)/m$$ 是相对
   理想计算时间的口径，同样条件下是 87.5%——别把两个口径混起来。）
   加 micro-batch 或换 1F1B / zero-bubble 调度。
3. **每设备 batch 太小。**矩阵乘太瘦，GPU 打不满。
4. **Data loader 喂不上。**看 GPU 的 idle 时间分布，不是看平均利用率。
5. **TP 跨节点了。**TP 每层内部要 all-reduce，必须在 NVLink 域内。
6. **序列太长。**attention 的 $$S^2$$ 项不计入 $$6N$$，所以长上下文下 MFU 天然偏低——
   这时候 MFU 低不代表有问题。

> **追问**
> - MFU 和 HFU 差多少？→ 差的就是重算那部分。开了梯度检查点，HFU ≈ MFU × 4/3。
> - 为什么不直接看 GPU utilization（nvidia-smi）？→ 那个只说明 kernel 在跑，
>   不说明它在做有用的算术。一个纯访存的 kernel 也能让它显示 100%。
>
> **陷阱**
> - 分母用了稀疏峰值（H100 的 1979 TFLOP/s 是 2:4 稀疏，dense 是 989）。
> - MoE 用 total params 而不是 activated params 算分子。


---

#### A10-07 · KV cache 每 token 多少字节

`推理显存` `必背` `高频`

**Q.** 推导每 token 的 KV cache 大小。对 Llama-3-70B 算出来，并和完整 MHA 做对比。

**公式**

$$\text{bytes/token} = 2 \times L \times K \times H \times \text{bytes per element}$$

- $$2$$：K 和 V 各存一份
- $$L$$：每层都要存
- $$K \times H$$：**KV 头数** × head_dim（不是 query 头数！）

**Llama-3-70B（GQA，$$K=8$$，bf16）**

$$2 \times 80 \times 8 \times 128 \times 2 = 327{,}680\ \text{bytes} = \mathbf{320\ KiB/token}$$

**如果是完整 MHA（$$K=N=64$$）**

$$2 \times 80 \times 64 \times 128 \times 2 = 2{,}621{,}440\ \text{bytes} = \mathbf{2{,}560\ KiB/token}$$

**GQA 省了 8 倍**，正好等于 $$N/K = 64/8$$。

> **追问**
> - 128k 上下文单条序列多少？→ $$320\ \text{KiB} \times 131072 / 1024^2 = \mathbf{40\ GiB}$$。
>   用 MHA 的话是 **320 GiB** —— 一张 80GB 的卡连一条对话都放不下。这就是 GQA 让长上下文
>   在经济上可行的原因。
> - MQA（$$K=1$$）呢？→ 40 KiB/token，省 64 倍，但质量有可测量的下降。
> - MLA 呢？→ DeepSeek-V2 把 K/V 压成一个低秩 latent（512 维）加一个 64 维的解耦 RoPE key，
>   每层只存 $$(512+64)\times2$$ 字节 → $$80\times576\times2 = 92{,}160\ \text{bytes} = 90\ \text{KiB/token}$$。
>   而且他们的消融显示 MLA 的建模质量**优于** MHA，是少见的不用取舍的优化。
>
> **陷阱**
> - 用 query 头数 → 结果大 8 倍。这是这道题最常见的错法。
> - 忘记那个 2（K 和 V）。
> - 用 $$D$$ 代替 $$KH$$ —— GQA 下 $$KH \ne D$$。


---

#### A10-08 · 一个节点能放下多少条序列？

`推理显存` `容量规划` `高频`

**Q.** 4×H100（共 320 GiB）用 bf16 服务 Llama-3-70B，平均上下文 8k。能放下多少条并发序列？
128k 上下文呢？

**先算权重。**$$70.6\times10^9 \times 2 = 1.41\times10^{11}$$ bytes $$= 131\ \text{GiB}$$。

**剩余空间。**$$320 - 131 = 189$$ GiB。再扣掉框架开销、CUDA context、临时激活，
实际可用于 KV cache 的按 **170 GiB** 估。

**8k 上下文场景**

每条序列：$$320\ \text{KiB/token} \times 8192 = 2.5\ \text{GiB}$$

$$170 / 2.5 = \mathbf{68\ 条并发}$$

**128k 上下文场景**

每条序列 40 GiB → $$170/40 = \mathbf{4\ 条}$$。

这个反差是整个推理服务设计的核心矛盾：**同样的硬件，上下文长 16 倍，并发掉 16 倍**，
而吞吐几乎正比于并发数。所以长上下文产品的单 token 成本天然高一个量级。

> **追问**
> - 怎么提高并发？→ 按性价比排序：KV cache 量化到 fp8（并发翻倍）、
>   prefix caching（如果有共享 system prompt）、paged attention 消除碎片
>   （朴素连续分配要按最大长度预留，浪费可能超过 50%）、换 MLA 架构。
> - 吞吐怎么估？→ 若每条请求平均占用 5 秒，68 并发 $$\approx$$ 13.6 QPS/副本。
> - 为什么要留出"框架开销"？→ vLLM 之类会预分配一大块，另外 CUDA context 每卡约 0.5–1GB。
>
> **陷阱**
> - 忘了减权重，直接用 320GB 除。
> - 忘了张量并行下**权重是每张卡都要放一份分片，但 KV cache 也要分片**——
>   TP=4 时每卡放 1/4 权重和 1/4 的 KV，总量不变，所以按总和算是对的。


---

#### A10-09 · 训练显存：16 字节/参数是怎么来的？

`训练显存` `必背`

**Q.** 混合精度加 AdamW，每个参数占多少字节？逐项列出来。

**标准混合精度配方**

| 项 | 精度 | 字节/参数 |
|---|---|---|
| 权重（计算用） | bf16 | 2 |
| 梯度 | bf16 | 2 |
| 权重（master copy） | fp32 | 4 |
| Adam 一阶矩 $$m$$ | fp32 | 4 |
| Adam 二阶矩 $$v$$ | fp32 | 4 |
| **合计** | | **16** |

**为什么要 fp32 master copy？**因为更新量通常比权重小好几个数量级。
bf16 只有 7 位尾数，相对精度约 $$2^{-8}\approx 0.4\%$$；当
$$|\Delta w| / |w| < 0.4\%$$ 时，$$w + \Delta w$$ 在 bf16 里**直接舍入回 $$w$$**，
模型静默停止学习而 loss 曲线看着还行。

**注意这个数字随框架变。**有些实现把梯度也累加在 fp32（+2 字节 → 18），
有些用 8-bit Adam（$$m,v$$ 各 1 字节 → 10），有些完全 bf16 训练不留 master copy（→ 8）。
面试里把标准配方说清楚、再说一句"具体看框架"，比背一个数字好。

**算例：70B 模型**

$$70.6\times10^9 \times 16 = 1.13\times10^{12}\ \text{bytes} = 1.03\ \text{TiB}$$

**这还不含激活。**一张 80 GiB 的卡放不下它的 1/13。所以 70B 全量训练**必须**分片，
不是优化问题而是可行性问题。

> **追问**
> - 只做 LoRA 呢？→ 基座权重冻结（bf16，2 字节/参数），只有 adapter 有梯度和优化器状态。
>   $$r=16$$ 的 adapter 大概是全参数的 0.1–1%，所以总显存从 16 字节/参数掉到约 2 字节/参数
>   加上激活——这才是 LoRA 真正省的东西。
> - 激活占多少？→ 见 A10-03，$$L(14BSD + BNS^2)$$，通常是最大的一项。
>
> **陷阱**
> - 只算权重和梯度，忘了优化器状态是大头（8/16 = 50%）。
> - 把 Adam 状态算成 bf16。


---

#### A10-10 · 100B 的训练怎么切？

`训练显存` `并行` `设计`

**Q.** 你要在 512 张 H100（每张 80 GiB）上训一个 100B 模型。做一遍容量规划，
并说清每种并行策略各自解决什么问题。

**第一步：算总需求。**

$$100\times10^9 \times 16\ \text{bytes} = 1.6\ \text{TB}$$（不含激活）

512 × 80 GiB = 40 TiB $$\approx$$ 44 TB 总显存（注意是 GiB；按 80 GB 算会少 7%）。看起来绰绰有余——**但 DDP 是每张卡都放一份完整状态**，
所以朴素 DDP 需要单卡 1.6TB，直接不可行。问题不是总量，是**分布**。

**第二步：按内存方程逐项攻击。**

$$\text{memory} = \underbrace{P}_{\text{权重}} + \underbrace{P}_{\text{梯度}} + \underbrace{2P\text{–}4P}_{\text{优化器}} + \underbrace{\text{激活}}_{\propto BS}$$

| 策略 | 切哪一项 | 效果 |
|---|---|---|
| ZeRO-1 | 优化器状态 | $$4 + 12/N_\text{dp}$$ 字节/参数（$$N_\text{dp}=8$$ 时 5.5，切很多份趋近 4） |
| ZeRO-2 | + 梯度 | 再降 |
| ZeRO-3 / FSDP | + 权重 | 权重按需 gather，通信量上升 |
| TP | 层内切矩阵 | 同时切权重和**激活**，但要 NVLink |
| PP | 按层切 | 切权重，引入 bubble |
| 激活重算 | 激活 | 换 ~30% 算力 |

**第三步：给一个具体布局。**

节点内 8 卡 NVLink → **TP = 8**。跨节点 **PP = 8**。剩下 $$512/(8\times8) = 8$$ 路 **DP**，
DP 层面开 ZeRO-1 切优化器状态。

单卡权重相关：$$1.6\times10^{12} / (8\times8) = 2.5\times10^{10}$$ bytes $$= 23\ \text{GiB}$$，
再被 8 路 ZeRO-1 分掉优化器状态——每参数从 16 字节降到 $$4 + 12/8 = 5.5$$ 字节——
落到 $$1.5625\times10^9 \times 5.5 = \mathbf{8.0\ GiB}$$。剩下约 72 GiB 给激活和临时缓冲，
配合选择性重算绰绰有余。

**第四步：检查 bubble。**PP=8 时要让 bubble 低于 10%，需要
$$m \ge 9(p-1)/1 \approx 63$$ 个 micro-batch。这会反过来约束 global batch size。

> **追问**
> - 为什么 TP 不能跨节点？→ 每层内部两次 all-reduce，跨节点 InfiniBand 带宽比 NVLink 低
>   一个量级，会直接吃掉收益。
> - 为什么 PP 放在 TP 外面 DP 里面？→ PP 的通信量最小（只在段边界传激活），最适合跨节点。
>
> **陷阱**
> - 只看总显存够不够，忘了 DDP 是复制不是分片。
> - 忘了激活——它经常比权重还大。


---

#### A10-11 · prefill 与 decode 的算术强度

`roofline` `推理` `高频`

**Q.** 算出 prefill 和 decode 各自的算术强度（FLOP/byte），
并解释为什么它们本质上是两台不同的机器。

**定义。**算术强度 = 计算量 ÷ 访存量。H100 的 ridge point：

$$\frac{989\ \text{TFLOP/s}}{3.35\ \text{TB/s}} \approx 295\ \text{FLOP/byte}$$

强度高于 295 → 计算受限；低于 → 访存受限。

**Decode（batch=1，生成一个 token）**

- 计算：$$2N = 2\times7.06\times10^{10} = 1.41\times10^{11}$$ FLOPs
- 访存：要把整个权重读一遍 $$= 1.41\times10^{11}$$ bytes（bf16，2 字节/参数）
- 强度 $$= 1.41\times10^{11} / 1.41\times10^{11} = \mathbf{1\ \text{FLOP/byte}}$$

离 ridge point 差 **295 倍**。GPU 的算术单元基本全闲着，你在纯粹地等内存。

**这直接给出 decode 的速度上限：**

$$\text{每 token 时间} \ge \frac{1.41\times10^{11}\ \text{bytes}}{3.35\times10^{12}\ \text{bytes/s}} = 42\ \text{ms}$$

即 batch=1 时最多约 **24 tokens/s**，且这个上限**与算力无关**，换更强的卡也没用。

**Prefill（长 prompt）**

同样读一遍权重，但一次处理 $$S$$ 个 token，计算量 × $$S$$：

$$\text{强度} \approx S\ \text{FLOP/byte}$$

$$S = 2048$$ 时强度约 2048，远在 ridge point 右侧 → **计算受限**。

**结论：batch size 就是把 decode 往右推的旋钮。**batch = $$B$$ 时 decode 强度约 $$B$$，
所以要让 decode 变成计算受限需要 $$B \gtrsim 295$$。实际做不到那么大（KV cache 放不下），
所以 decode 几乎永远是访存受限的。

> **追问**
> - 为什么 speculative decoding 在高 batch 下失效？→ 它用的是"decode 有闲置算力"这个前提。
>   batch 大了之后算力不再闲置，验证 draft token 的开销就不再免费。
> - 那 KV cache 的读取算不算？→ 算。长上下文下 KV cache 的读取会超过权重读取，
>   此时 decode 时间开始随上下文长度增长——这是"长对话越聊越慢"的机制。
>
> **陷阱**
> - 用稀疏峰值 1979 TFLOP/s 算 ridge point（应该用 dense 989）。
> - 认为加卡能提高 batch=1 的 decode 速度。


---

#### A10-12 · 估算训练时间与成本

`成本` `容量规划`

**Q.** 用 2048 张 H100、40% MFU 在 15T token 上训一个 70B 模型。要多久，大概多少钱？

**总算力需求**

$$C = 6ND = 6 \times 7.06\times10^{10} \times 1.5\times10^{13} = 6.35\times10^{24}\ \text{FLOPs}$$

**集群有效算力**

$$2048 \times 9.89\times10^{14} \times 0.40 = 8.10\times10^{17}\ \text{FLOP/s}$$

**时间**

$$\frac{6.35\times10^{24}}{8.10\times10^{17}} = 7.84\times10^{6}\ \text{s} = \mathbf{91\ 天}$$

**成本**（按 H100 云价 ~$2/卡·小时 估）

$$2048 \times 24 \times 91 \times \$2 \approx \mathbf{\$8.9\ M}$$

**Sanity check：**这个量级和公开报道的前沿模型训练成本吻合（数百万到数千万美元），
说明估算没跑偏。面试里做完估算主动做一次 sanity check 是加分项。

> **追问**
> - 怎么缩短到一个月？→ 需要 3× 算力，即约 6000 张卡。但注意 MFU 会随规模下降
>   （通信占比上升），所以不是线性的。
> - 中断怎么办？→ 91 天里硬件故障几乎必然发生。需要算 checkpoint 频率：
>   若平均 4 小时一次故障，checkpoint 间隔应远小于它，通常 15–30 分钟。
>   这又引出 checkpoint 写入带宽的要求。
>
> **陷阱**
> - 忘了乘 MFU，直接用峰值算 → 时间少算 2.5 倍。
> - 用 total params 算 MoE。


---

> **这一节还会继续加题。**计划补充：MoE 的显存/算力账、量化后的显存重算、
> 多轮对话的 KV 增长、embedding 与 vocab 的取舍、batch size 与 LR 的缩放关系。

---

<a id="section-a11"></a>

## A11 · Scaling 与评测

Alisa 那本的 "Scaling laws" 一节其实只有 muP 和曲线拟合，**没有 Kaplan/Chinchilla**，
评测更是零覆盖。所以这一节基本是新写的。

**这一节的分界线：**scaling law 部分看你知不知道**它优化的是什么**；
评测部分看你会不会**主动说出自己方法的失效模式**。

---

<a id="a11-1"></a>
### A11.1 Kaplan 与 Chinchilla

**Kaplan (2020)。**Loss 在参数、数据、算力上都服从幂律。他们的分析暗示固定算力预算下
应该把增量主要花在**参数**上——所以 GPT-3 是 175B 配约 300B token（约 1.7 token/参数）。

**Chinchilla (2022)。**重做了分析，算力最优前沿变成了大致**等比例扩展**：约 **20 token/参数**。

**为什么两者结论不同，标准说法其实已经被修正了。**Hoffmann 等当年猜的是学习率调度
（Kaplan 的 run 复用同一个调度，惩罚了小模型/长训练的配置）。但 Porian 等
（[arXiv:2406.19146](https://arxiv.org/abs/2406.19146)，NeurIPS 2024）复现并拆开了这个差异，
发现真正的主因是另外两个：**Kaplan 没有把 embedding/解码层的 FLOPs 计入算力**，
以及**固定长度的 warmup 对小模型太长**。他们直接测了学习率衰减这一项，指数只从 0.60 动到 0.57——
"careful learning rate decay is not essential"。

这题值得记住三个因子而不是一个，因为面试官如果读过这篇，只答调度会被追下去。

$$N_\text{opt} \propto C^{0.5},\qquad D_\text{opt}\propto C^{0.5}$$

Chinchilla（70B，1.4T token）在同等算力下打败了 Gopher（280B，300B token）。
整个领域立刻把模型做小、数据集做大。

**然后又变了。**Chinchilla 优化的是**训练**算力。如果你要给几百万用户服务，
**推理成本主导终身成本**，那么把一个更小的模型训到远超算力最优点是理性的。
Llama 3 正是这么做的：8B 模型训约 15T token，约 1,875 token/参数，是 Chinchilla 点的约 90 倍。

**"最优模型大小是多少"的正确回答是一个反问：**对训练成本最优，还是对终身总成本最优？

#### 自测 · A11.1

**Q A11.1.1** — Chinchilla 改变了什么，它现在还是正确的目标吗？

Chinchilla 用逐个 run 单独调过的学习率调度重做了 Kaplan 的分析，发现算力最优前沿是参数和数据
大致等比例扩展——约 20 token/参数，而不是 Kaplan 那个偏重参数的建议。

就它所优化的东西——**训练**算力——而言，它仍然是对的。当推理主导终身成本时它就是错的目标：
这时候你要训得更小、更久，因为小模型在此后的每一次请求上都更便宜，而多出来的训练只付一次。
Llama 3 8B 训约 15T token，是刻意地跑到 Chinchilla 点之外约 90 倍。

这个策略的上限是数据。重复大约四个 epoch 之后收益就崩了，所以"训小训久"会在你用光
不重复的高质量 token 时走到头。

> **追问**
> - *irreducible loss 那一项是什么？* → 拟合式是
>   $$\mathcal L(C) = \mathcal L_\infty + \beta C^{-\alpha}$$。没有 $$\mathcal L_\infty$$
>   （数据本身的熵）这一项，拟合会推出算力 → ∞ 时 loss → 0，这是荒谬的。
> - *scaling law 对下游任务也成立吗？* → 干净程度差很多。Loss 平滑扩展；
>   benchmark 准确率可能看起来是不连续的，主要因为那个指标带阈值。
>
> **陷阱**
> - 说 Chinchilla 是"20 tokens per parameter 的规律"就停。

---

<a id="a11-2"></a>
### A11.2 muP

**问题。**标准参数化下，最优学习率**随宽度移动**。所以你在 1B proxy 上调好的超参
对 70B 是错的——而 70B 你调不起。

**muP 做什么。**重新缩放初始化方差和每层学习率，使*更新相对于权重的幅度*在不同宽度下保持一致。
于是最优超参变得**与宽度无关**，可以在小 proxy 上调完直接迁移。

**怎么用。**在几个不同宽度的小模型上扫 LR（和其他超参），确认最优点不移动，
然后迁移到目标宽度。这是"只能跑一次的 run"的标准做法。

#### 自测 · A11.2

**Q A11.2.1** — muP 解决的是什么问题？

标准参数化下，最优学习率随宽度移动，所以在小 proxy 上调好的超参到了目标规模就是错的——
而在目标规模上你只有一次机会。

muP 同时重新缩放初始化方差**和**每层学习率，使更新相对于权重的幅度与宽度无关。
最优点于是不再移动，你可以在一族小模型上调完直接迁移过去。

> **追问**
> - *它能跨深度迁移吗？* → 原始结果主要是关于宽度的；深度迁移没那么干净，有专门的后续工作。
>
> **陷阱**
> - 说 muP 是"一种初始化方法"。它同时改初始化**和**每层学习率。

---

<a id="a11-3"></a>
### A11.3 Test-time compute 对评测的影响

第三条轴的机制在 A7 讲过。这里只讲它**对评测意味着什么**，因为那是评测这一节的问题。

**一个 benchmark 数字现在是欠定的。**同样的权重，贪心解码和大搜索预算下实际上是两个不同的系统。
公平比较需要三者之一：

1. 各系统共享一个**固定推理预算**，
2. 一条**分数对 token/成本/延迟**的曲线，
3. 能力提升**连同**买到它的额外预算一起报告。

#### 自测 · A11.3

**Q A11.3.1** — 两个模型在同一个推理 benchmark 上报了相同的分数。你会问什么？

各自花了多少 token。当 test-time compute 成为一条活跃的轴，单个数字不足以确定一个系统——
同样的权重，贪心解码和 best-of-64 是两个成本不同的产品。

我要的是一条**分数对预算的曲线**，不是一个点。这同时会暴露更有意思的性质：谁的*斜率*更好。
一个在低预算下落后、但随思考 token 扩展得更好的模型，往往是更值得押的那个。

> **追问**
> - *额外算力从哪里开始不再划算？* → 事实召回几乎立刻走平——思考不创造知识。
>   搜索形状的工作（竞赛数学、调试）则会持续得到回报。
> - *best-of-N 的瓶颈在哪？* → 在 **verifier**，不在采样器。pass@k 远高于你实际能挑出来的水平。
>
> **陷阱**
> - 报 benchmark 分数不说推理预算。

---

<a id="a11-4"></a>
### A11.4 困惑度

$$\text{PPL} = \exp\Big(-\frac1T\sum_{t=1}^T \log p(x_t\mid x_{<t})\Big)$$

指数化的平均负对数似然——可解释为**有效分支因子**（模型在多少个等概率选项之间选择）。

**它什么时候骗人：**

1. **跨分词器时。**不同词表对文本的切分不同，每 token 似然不可比。
   **永远不要跨分词器比困惑度。**必须比就归一化到每字节或每字符。
2. **它被简单 token 主导。**大多数 token 是空白、标点和虚词。模型可以在困惑度上大幅改善，
   而在你在乎的任何事情上毫无长进。
3. **RLHF 之后它通常*变差*而模型变得更有用。**对齐把概率质量集中到一种偏好的风格上，
   这会抬高通用语料上的 NLL。

#### 自测 · A11.4

**Q A11.4.1** — 定义困惑度，并给出三种它会误导人的情形。

（定义与三点见上。）容易翻车的是第三点：RLHF 之后，通用语料上的困惑度通常会**上升**，
而模型变得更有用，因为对齐把概率质量集中到了一种偏好的风格上。
如果你拿困惑度当后训练指标，你会得出"对齐这一轮把模型搞坏了"的结论。

跨模型比较要用 **bits per byte**，它与分词器无关。

> **追问**
> - *那为什么还在报它？* → 它便宜、平滑，而且 scaling law 就是拟合在这个量上的。
>   它是一个好的*训练*信号，一个差的*产品*指标。

---

<a id="a11-5"></a>
### A11.5 无法验证答案时怎么评测

**评测阶梯，按顺序说出来：**

1. **Verifier**，只要存在就用。单元测试、数学检查器、编译器。最便宜，而且在通常意义上
   无法被钻空子——它是一个函数，不是一个模型。
2. **人类偏好**，在没有 verifier 时。贵、慢，但对"有用性"是 ground truth。
3. **LLM-as-judge** 作为人类的可扩展代理——而且要主动点名它的失效模式：
   **位置偏差**（偏好第一个或第二个）、**长度偏差**（偏好更长的）、
   **自我偏好**（偏好自己和同族模型的输出）、以及对格式敏感。
4. **成对比较而不是绝对打分**，因为人和裁判模型在排序上都远比在 1–10 打分上可靠。

**裁判偏差的缓解：**随机化位置并对两种顺序取平均；控制长度；
用与被测模型不同族的裁判；用人工标注的子集做校准。

#### 自测 · A11.5

**Q A11.5.1** — 给开放式生成排一条评测阶梯。

（阶梯见上。）真正拉开差距的地方，是**不等人问**就主动点名裁判的失效模式——位置偏差、
长度偏差、自我偏好、格式敏感——并且每一个都给出缓解办法：随机化位置并对两种顺序取平均、
控制长度、用不同族的裁判、拿人工标注的子集做校准。

还有，优先用**成对**比较而不是绝对打分，因为人和模型在排序上都远比在打分上可靠。

> **追问**
> - *你怎么知道你的 benchmark 不在训练数据里？* → 对着语料做 n-gram 重叠检查；
>   用训练截止**之后**构建的 held-out 集；在评测集里植入 canary 字符串；
>   以及公开划分和新采集的私有划分之间那个说明问题的差距。
> - *Elo / Arena 是什么？* → 把成对的人类投票聚合成一个评分。它测的是偏好，
>   与正确性相关但不等同，而且可以被风格刷分。
>
> **陷阱**
> - 只说"用 LLM-as-judge"而不说它的偏差。

---

<a id="a11-6"></a>
### A11.6 涌现是真的吗

**主张。**某些能力不连续地出现——在某个规模阈值前一直是随机水平，之后急剧改善。

**批评**（Schaeffer 等）。这个不连续常常是**度量**的产物，不是模型的。
多步任务上的精确匹配准确率是每 token 准确率的非线性阈值函数：
需要 5 个 token 都对、每个概率为 $$p$$ 时，精确匹配是 $$p^5$$——
即使 $$p$$ 在平滑改善，$$p^5$$ 看起来也是先平后爆。换成连续度量
（token 编辑距离、答案的对数似然），曲线就平滑了。

**诚实的立场。**两件事都成立。底层能力一般平滑扩展，而系统的*可用性*确实可以是不连续的，
因为产品有阈值——一个 20% 成功率的代码 agent 和一个 80% 成功率的，
即使底层曲线平滑，也是两个不同的产品。

#### 自测 · A11.6

**Q A11.6.1** — 涌现能力是真的吗？

两种立场都对了一半，好的回答会把**能力**和**可用性**分开。

底层能力一般是平滑扩展的。表面上的不连续在很大程度上是度量的产物：多步任务上的精确匹配
是每 token 准确率的阈值函数，所以 $$p$$ 在平滑改善时 $$p^5$$ 看起来先平后爆。
换成连续度量，曲线就平滑了。

但可用性确实是不连续的，因为产品有阈值。一个 20% 成功率的代码 agent 和一个 80% 的，
无论底层曲线是什么形状，都是两个不同的产品。

实践上这件事为什么重要：平滑的底层指标可以从小规模 run 外推，带阈值的产品指标不行。
这正是能力预测困难的原因。

---

<a id="a11-7"></a>
### A11.7 设计一个评测

**先说可信评测的三个性质：**它测的是有人真正在乎的工作；分数变高意味着系统真的变好了；
轨迹能解释这个分数是怎么挣来的。

**然后是设计选择：**

- **任务来源。**带真实测试套件的真实 GitHub issue（SWE-bench 式）胜过合成，
  因为它继承了真实工作的难度分布。但它也继承了污染。
- **验证。**跑仓库自己的测试。这就是编程成为好 RL/评测领域的全部原因——verifier 免费。
- **污染控制。**用模型截止日期**之后**创建的任务。否则你测的是记忆。
- **预算控制。**固定步数、token 数或墙钟时间。否则你测的是脚手架，不是模型。
- **按位置/难度分层报告**，不要只给一个总分。平均值掩盖了模型是在简单任务上变好还是难任务上。

**要点名的难题：评测延迟。**如果一个任务要跑一小时，你就无法迭代。
你需要一个快速冒烟子集做内循环、全套做外循环。在前沿，
一个诚实的周级 agent 任务评测本身可能就要跑一周——比训下一个模型还久。

#### 自测 · A11.7

**Q A11.7.1** — 为长时程编程设计一个评测。

（设计要点见上。）两个选择承担了绝大部分价值。

**污染控制**：用模型训练截止之后创建的仓库和 issue 来构造任务。没有这一条，你测的就是记忆，
而在公开 benchmark 上这是默认结果。

**预算控制**：在各个系统之间固定步数、token 数或墙钟时间。否则你比的是脚手架而不是模型——
在 agent benchmark 上，脚手架通常比模型更能左右结果。

pass@k **和** pass^k 都要报。前者奖励探索（$$k$$ 次尝试中任意一次成功），
后者度量可靠性（$$k$$ 次全部成功）。产品需要的是后者，
而两者之间的差距正是不稳定行为藏身的地方。

> **追问**
> - *测试本身就 flaky 怎么办？* → 跑 $$k$$ 次，两个指标都报，并把已知 flaky 的任务隔离出去。
>   只跑一次的话，flaky 和"部分具备能力"是分不开的。
>
> **陷阱**
> - 不控推理预算就比较两个 agent。

---

> **待补概念：**benchmark 的具体谱系（MMLU / GPQA / SWE-bench / τ-bench / ARC-AGI）、
> 数据污染的检测方法细节、reward model 的评测、
> 多语言与公平性评测、A/B 测试与线上指标。

---

<a id="section-a12"></a>

## A12 · Agentic RL 与环境

★ 全新一节，压缩自我自己的 `env-scaling` / `agentic-rl-qa` / `self-evolving-harness`。
Alisa 那本零覆盖。

**为什么必须有这一节：**这是我简历上的方向，项目深挖一定会拽到这里，而面试官会顺着往下问。
在这一块答得含糊，比在任何别的地方答得含糊都更致命。

---

<a id="a12-1"></a>
### A12.1 从 chat 到 agent：形式上变了什么


**形式上。**单轮 LM 是一个 bandit：一个 prompt、一个动作、一个奖励。Agent 则是一个
**部分可观测 MDP**：在第 $$t$$ 步 agent 看到观察 $$o_t$$，发出动作 $$a_t$$，
世界转移到 $$s_{t+1}\sim T(\cdot\mid s_t,a_t)$$，episode 结束时 verifier 返回 $$r = R(s_T)$$。

策略 $$\pi_\theta(a_t \mid o_{\le t}, a_{<t})$$ 以**整段历史**为条件，
因为脚手架把它一直留在上下文里。

**实际会坏掉的东西，按严重程度排序：**

1. **Credit assignment。**几百个决策共用一个奖励。到底是哪一次工具调用做对了？
2. **环境变成了模型的一部分。**你的策略上限就是它能行动的那个世界的上限——
   而环境是定制工程，不是数据。
3. **评测延迟。**一个跑一周的任务，诚实地评一次就要一周。你没法用这个速度迭代，
   于是瓶颈从训练变成了评测。
4. **非平稳性。**工具 API 会变，网站会变，环境在你脚下漂移。
5. **轨迹数据是 on-policy 的，而且会腐坏。**策略一变它就过期，不像 SFT 数据可以攒起来。


#### 自测 · A12.1

**Q A12.1.1** — 一个 agent 在 5 步任务上有 80% 成功率。20 步任务你预测多少？

朴素外推是 $$0.8^{4} \approx 41\%$$，把 5 步任务当作一个可靠性单元。先把这个说出来，
然后说清它在**两个方向上**都错了。

**太悲观**，因为 agent 会恢复。如果 agent 观察到错误并重试，一步失败不等于任务失败——
恢复恰恰就是长时程策略存在的意义，而它打破了独立性假设。

**太乐观**，因为失败是相关的。一个误解了目标的 agent 会在之后的每一步都失败，
而且上下文退化是累积的，不会自己重置。

诚实的回答是：每步可靠性根本无法外推。这也是为什么 **operating horizon**——
agent 能保持*连贯*多久，而不是能跑多久——是直接测出来的，不是推出来的。

> **追问**
> - *为什么大家都去抓 "operating horizon" 这个指标？* → 它测的不是模型能*跑*多久——
>   一个坏掉的循环可以永远跑下去——而是它能保持**连贯**多久：抓得住目标、
>   犯的是可恢复而非致命的错、并且持续产出有人愿意留下的工作。
> - *"这个模型做研究行不行"的实用代理指标是什么？* → 你敢无人监督地交给它多少张 GPU。
>   这是一个信任指标，不是 benchmark 指标。
>
> **陷阱**
> - 只说"多轮对话"。要说出 POMDP、credit assignment、以及环境本身成了系统的一部分。


---

<a id="a12-2"></a>
### A12.2 环境的解剖


五个部件。能把它们分开命名，后面的讨论才谈得下去：

1. **状态 / 世界** —— 文件系统、数据库、浏览器 DOM、模拟器。
2. **动作空间** —— 工具 schema。这是一个后果很大的设计决定：太粗，agent 表达不了它需要的东西；
   太细，时程直接爆炸。
3. **观察** —— 返回来的是什么，以及关键的**它怎么被截断**。一份 10MB 的测试日志必须变成
   能塞进上下文、又不丢掉报错的东西。
4. **转移** —— 通常是真实执行，也就意味着慢、有状态、偶尔还不确定。
5. **奖励 / verifier** —— 成功条件。

**外加两条大家总会忘的运维要求：**

- **Reset / 隔离。**每次 rollout 都需要一个干净的世界：容器、快照，或者纯模拟器。
  没有这个，rollout 之间会互相污染，你的梯度就是垃圾。
- **吞吐。**RL 需要成千上万次 rollout。如果一次 reset 要 30 秒，它就主导一切。
  环境工程在很大程度上是个吞吐问题。


#### 自测 · A12.2

**Q A12.2.1** — 一个 RL 环境由哪些部件组成，哪一个最被低估？

五个：状态/世界、动作空间（工具 schema）、观察（包括**它怎么被截断**）、
转移（通常是真实执行——慢、有状态、有时不确定）、奖励/verifier。

大多数回答漏掉的两个都是运维层面的：**reset 与隔离**（每次 rollout 都需要干净的世界，
否则 rollout 之间互相污染、梯度就是垃圾）和**吞吐**（RL 要跑成千上万次 rollout，
一次 30 秒的 reset 就主导一切）。环境工程在很大程度上是吞吐问题，不是建模问题。

> **追问**
> - *为什么观察的截断是建模决定，而不是管道细节？* → 因为你截掉什么，决定了 agent
>   能学会关注什么。丢掉 stack trace 只留摘要，教出来的是另一个策略。
> - *沙箱怎么做？* → 断网、资源限制、超时、可重置的文件系统。一旦 agent 会写并执行代码，
>   这既是正确性要求，也是安全要求。
>
> **陷阱**
> - 只答"state, action, reward"。reset/isolation 和吞吐才是实际工程里的大头。


---

<a id="a12-3"></a>
### A12.3 难度 ≠ 可训练性


**把方差论证说精确。**对一个在当前策略下成功概率为 $$\hat p$$ 的任务，二值结果的方差是

$$\operatorname{Var} = \hat p(1-\hat p)$$

在 $$\hat p = 0.5$$ 处最大，**在两端为零**。策略总是失败（$$\hat p=0$$）的任务和
总是解决（$$\hat p=1$$）的任务，对梯度的贡献**都是零**。

在 GRPO 里这不是近似，而是精确成立的。一组里每个完成都拿到相同奖励时，advantage 恒等于零，
这组就是白烧的算力。

**所以：难度不等于可训练性。**一个任务可以因为产生不了信号的原因而变难：

- 规范含糊，于是 verifier 实际上是随机的。
- Verifier 坏了，于是成功与质量不相关。
- 它需要基座模型没有的知识——RL 装不进知识。
- 它长到 credit assignment 毫无希望。

**可训练**意味着*既难又有信息量*，这是比*难*严格更小的一个集合。

**那该怎么做。**从近期 rollout 持续估计每个 prompt 的成功率；把题池集中在 50% 附近；
淘汰已解决的任务；把从未解决的搁置起来留待以后（策略变强后它们可能变得可训练）。
这是一条**会移动的**课程，因为 $$\hat p$$ 随策略变化。


#### 自测 · A12.3

**Q A12.3.1** — 你手上有一池子任务。实际该在哪些上面训？

成功率在 50% 附近的那些，因为 $$\operatorname{Var} = \hat p(1-\hat p)$$ 在那里最大、
**在两端为零**。在 GRPO 里这是精确的而非近似：一组里每个完成都拿到相同奖励时，
advantage 恒等于零，这组算力就白烧了。

这也是难度不等于可训练性的原因。一个任务可以因为产生不了信号而变难——规范含糊、
verifier 坏了、缺少 RL 装不进去的知识，或者时程长到 credit assignment 毫无希望。
可训练意味着*既难又有信息量*，是一个严格更小的集合。

而且课程必须**会移动**，因为 $$\hat p$$ 随策略变化。

> **追问**
> - *这和 DAPO 的动态采样是一回事吗？* → 原理相同，层级不同。DAPO 在一个 batch 内重采样，
>   直到某组有奖励方差；课程则是在整个训练过程中对任务池做调度。
> - *那些从来解不出来的任务怎么办？* → 要么拆解（给子目标或部分解当提示），
>   要么先放一边，等策略长到能吃下它们为止。
>
> **陷阱**
> - 说"用最难的题训练"。最难的题梯度为零。


---

<a id="a12-4"></a>
### A12.4 长时程的信用分配


**要诚实地说这个问题没有解决。**几个选项，以及各自买到什么、代价是什么：

1. **把结果奖励广播到所有 token**（GRPO 的做法）。简单、期望上无偏，但长时程上方差极大。
   episode 短的时候效果好得出乎意料。
2. **学出来的 critic / 价值函数**（PPO）。能给出每步 advantage，但 critic 恰恰是这个设定下
   最难拟合的东西：奖励稀疏、目标在动，还要再养一个同等规模的模型。
3. **Process reward model（PRM）。**给中间步骤打分。credit assignment 更好，
   但你现在需要步骤级标注——很贵，而且 PRM 自己也会变得可被钻空子。
4. **Hindsight relabelling。**一条失败但*做成了别的什么*的轨迹，对*那个*目标来说就是成功轨迹。
   便宜的额外信号；风险是教会模型去追容易的目标。
5. **步骤级 verifier**，在领域允许的地方：这次编辑之后代码还能编译吗、通过的测试数涨了吗。
   编程里这种 process 信号是免费的，这也是编程成为 agentic RL 主战场的重要原因。
6. **从构造上就缩短时程。**把任务拆开，让每个子 episode 有自己可验证的结果。
   这往往是最实际的答案。

**值得给出的重新框定。**credit assignment 之所以难，是因为奖励*来得晚*。
任何能让信号更早出现的东西——步骤 verifier、任务拆解、更稠密的环境反馈——
都比一个更聪明的估计器管用。


#### 自测 · A12.4

**Q A12.4.1** — 一个奖励，三百次工具调用。你有哪些选择？

先诚实地说这个问题没有解决。把结果奖励广播到所有 token（GRPO）简单且无偏，
但长时程上方差极大。学出来的 critic 能给每步 advantage，但恰恰在这个设定下最难拟合——
奖励稀疏、目标在动、还要多养一个同等规模的模型。Process reward model 改善了 credit assignment，
但需要步骤级标注，而且它自己也会变得可被钻空子。Hindsight relabelling 是便宜的额外信号，
风险是教会模型去追容易的目标。

我会给出的重新框定是：credit assignment 之所以难，是因为奖励**来得晚**。
任何能让信号更早出现的东西——步骤级 verifier（能编译吗、通过的测试数涨了吗）、
拆成各有结果的子 episode、更稠密的环境反馈——都比一个更聪明的估计器管用。

> **追问**
> - *为什么 critic 偏偏在这里特别难？* → 它要从一条不完整的 agent 轨迹预测未来期望奖励。
>   而这个分布每次策略变化都会漂移，奖励还只是最后的一个 bit。
> - *outcome 和 process reward 哪个更安全？* → 结果奖励可以被无效推理满足；
>   process reward 可以被逐步玩弄。谁也不占优；通常的答案是两个都用，以结果奖励作为锚。
>
> **陷阱**
> - 说"用 PPO 的 critic 就行"。要说出为什么 critic 在这个设定下特别难训。


---

<a id="a12-5"></a>
### A12.5 环境扩展的管线


**管线：Generate → Build → Verify → Filter → Evolve。**

1. **Generate。**合成候选任务——从模板、从真实产物（GitHub issue、文档），
   或者从一个以当前策略失效模式为条件的模型生成。
2. **Build。**把每个候选实例化到一个可执行的环境里。这是最贵的一步，也是真正的瓶颈。
3. **Verify。**分开检查两件事，而且**两件都要**：
   - 它**可解**吗？跑一个强参考模型，或者一份脚本解法。
   - 它**可判定**吗？成功条件在正确解上确实触发、在错误解上确实不触发吗？
4. **Filter。**丢掉任一检查没过的那一大批，再加上重复的和退化的任务。
   然后按可训练性（A12-03）过滤，而不只是按有效性。
5. **Evolve。**把幸存者往策略能力的边界上变异——把已解决的变难，把没解决的拆开。

**诚实的数字。**从生成到可用训练任务的良率很低——生成的候选里有很大一部分不可解、
不可判定，或者一眼就能解。预算要按这个来做。


#### 自测 · A12.5

**Q A12.5.1** — 你需要 10,000 个训练环境。怎么搞出来？

Generate → Build → Verify → Filter → Evolve。从模板、真实产物，或者一个以当前策略失败模式为
条件的模型生成候选；把每个实例化到可执行环境里（最贵的一步，也是真正的瓶颈）；
然后验证**两件独立的事**。

这个双重检查正是大多数人会漏掉的地方。**可解**——一个强参考模型或者脚本解法能不能做完它？
以及**可判定**——成功条件在正确解上确实触发、在错误解上确实*不*触发吗？
verifier 的一次假阳性会主动教给策略一个错误的东西，这比压根没有这个任务还糟。

然后按可训练性而不只是有效性来过滤，并把幸存者往策略能力的边界上变异。
良率要按低了预算：生成的候选里有很大一部分不可解、不可判定，或者太简单。

> **追问**
> - *为什么"多样性更高"不会自动有帮助？* → verifier 打不了分的多样性，
>   或者落在可训练区间之外的多样性，都是噪声。多样性只有**以信号为条件**才有价值。
> - *什么是 self-evolving harness？* → 把同一个循环延伸到脚手架本身：模型周围的 prompt、
>   工具定义和控制流变成被优化的对象，用的还是 generate-verify-select 那套结构。
>   它是在脚手架上做无梯度优化。
>
> **陷阱**
> - 只做 solvable 检查，不做 checkable 检查。verifier 假阳性会直接教坏策略。


---

<a id="a12-6"></a>
### A12.6 工具设计与失效模式


**设计原则：**

- **粒度要匹配决策单元。**太细的工具（`move_cursor`）会让时程爆炸；
  太粗的（`solve_task`）则什么都没留给模型学。
- **错误必须有信息量、且可恢复。**这是杠杆最高的一件事：只返回 "Error" 的工具什么都教不了；
  返回 stack trace 加一条提示的工具，教会的是恢复。
- **能幂等就幂等**，这样重试是安全的。
- **观察必须可摘要** —— 该由工具而不是模型去截断一份 10MB 的日志，而且必须留住关键的那部分。

**要点名的失效模式：**

| 失效 | 修法 |
|---|---|
| 死循环（反复发同一个调用） | 步数预算；循环检测；惩罚重复 |
| 幻觉出的工具名 / 参数 | 对着 schema 做约束解码 |
| 无视工具返回结果 | 通常是上下文排版问题，不是推理问题 |
| 级联错误 | 在 SFT 数据里放显式的恢复示例 |
| 静默的部分失败 | 工具必须大声失败；含糊的成功比失败更糟 |

**安全。**一旦 agent 会写并执行代码，沙箱就不是可选项：断网、资源限制、超时、全新文件系统。
另外有一条值得明确画出来的区分：**惩罚动作，不惩罚想法**——你要约束的是 agent *做*了什么，
同时让它的推理保持可读，因为惩罚推理只会教它把推理藏起来。


#### 自测 · A12.6

**Q A12.6.1** — 工具接口怎么设计，会出什么问题？

粒度要匹配决策单元——太细（`move_cursor`）时程爆炸，太粗（`solve_task`）什么都没留给模型学。
工具能幂等就幂等，这样重试是安全的；一份 10 MB 的日志该由**工具**而不是模型来截断，
而且要留住关键的那部分。

杠杆最高的单一因素是**错误信息的质量**。只返回 "Error" 的工具什么都教不了；
返回 stack trace 加提示的工具教会的是恢复，而在一个长 episode 里，恢复就是大部分技能。

安全上要把这条区分明确画出来：**惩罚动作，不惩罚想法**。执行放沙箱里，
不可逆的动作要求确认，但让推理保持可读——惩罚推理只会教模型把它藏起来。

> **追问**
> - *为什么惩罚坏想法会适得其反？* → 它优化出的是不可监控的推理。一旦 chain of thought
>   成了安全训练的目标，它就不再是一扇忠实反映计算过程的窗——而可监控性本身才是它的价值所在。
> - *MCP 呢？* → 一个标准化的工具接口，让 agent 和工具提供方不必两两做定制集成。
>   它在 2025 年通过广泛采用成为事实标准，随后于 2025 年 12 月捐给 Linux Foundation 的
>   Agentic AI Foundation 以保持治理中立——被问到时，这个先后顺序是有讲究的。
>
> **陷阱**
> - 只讲 schema 设计。错误信息的质量对学习的影响更大。


---

<a id="a12-7"></a>
### A12.7 Agent 评测


**一个可信的 agent 评测需要同时具备三条性质：**

1. 它测的是**有人真正在乎的工作**——不是一个没人愿意付钱的合成代理任务。
2. 分数变高意味着系统**真的变好了**——不是脚手架运气好。
3. **轨迹能解释**这个分数是怎么挣来的——你可以审计它*为什么*通过。

**为什么它更难：**

- **评测延迟。**一个跑一周的任务，诚实地评一次就要一周。这可能超过训练下一个模型的成本，
  并且直接卡住你的迭代速度。
- **不确定性。**工具返回、超时和网络会让同一个策略在不同 run 上拿到不同分数。
  你需要重复 $$k$$ 次，并且同时报 **pass@k**（任意一次成功——探索）和 **pass^k**
  （全部成功——可靠性）。产品需要的是后者。
- **脚手架混淆。**两个"agent"之间被测出来的差距，很多时候大部分来自脚手架而不是模型。
  比模型时固定脚手架，比脚手架时固定模型。
- **预算混淆。**不固定步数/token/时间预算，你测的就是花钱的意愿，不是能力。


#### 自测 · A12.7

**Q A12.7.1** — 为什么评测一个 agent 比评测一个 chat 模型更难？

四个原因，其中后两个足以让大多数已发表的对比失效。

**评测延迟**——一个跑一周的任务，诚实地评一次就要一周，这可能超过训练下一个模型的成本，
并卡住迭代速度。**不确定性**——工具返回、超时和网络会让同一个策略拿到不同分数，
所以你需要重复 $$k$$ 次，并且同时报 pass@k（探索）和 pass^k（可靠性）；产品需要的是后者。

**脚手架混淆**——两个"agent"之间被测出来的差距，大部分来自脚手架而不是模型。
比模型的时候要固定脚手架。**预算混淆**——不固定步数/token/时间预算，
你测的就是花钱的意愿，不是能力。

> **追问**
> - *最便宜又有用的一步是加什么？* → 一个跑得快的冒烟子集，用在内循环。
>   全套每晚跑一次，冒烟集每次改动都跑。
> - *要不要给部分分？* → 通常比二值更有信息量：通过的测试数、达成的子目标。
>   如果你之后要拿这个评测当训练奖励，它也提供更稠密的信号——但要小心，
>   这会让它立刻变得可被钻空子。
>
> **陷阱**
> - 只报平均成功率。要报**在固定预算下**的 pass^k，并分难度看。


---

<a id="a12-8"></a>
### A12.8 为什么要 RL，而不是在好轨迹上做 SFT


**你确实应该——先做。**在好轨迹上做行为克隆便宜、稳定，而且能走完大部分路。
这是正确的第一步，也是标准的冷启动。

**然后是它做不到的：**

1. **曝光偏差。**SFT 只展示金标准质量的前缀。模型从来看不到自己的错误，
   所以永远学不会从错误中恢复。在一个 300 步的 episode 里，恢复就是大部分技能。
2. **它的天花板是示范者。**模仿超不过来源。而对着 verifier 做 RL 可以找到示范者从未产出的解，
   因为定义成功的是 verifier，不是示范者。
3. **它表达不了对*怎么做*的偏好。**"用更少的工具调用解决它"、"不要删文件"、
   "含糊的时候停下来问"，这些都是轨迹分布的性质，SFT 只能靠把每种情形都示范一遍来编码。

**所以标准配方是两个都要：**先用 SFT 拿到一个格式正确、有基本能力的起始策略，
再对着 verifier 做 RL，越过示范者并教会恢复。


#### 自测 · A12.8

**Q A12.8.1** — 你手上有一个强模型产出的 10 万条成功轨迹。为什么不直接在上面做 SFT？

你确实应该——先做。行为克隆便宜、稳定，是标准的冷启动。然后再点名它做不到的三件事。

**曝光偏差**：SFT 只展示金标准质量的前缀，所以模型从来看不到自己的错误，也就学不会恢复——
而在 300 步的 episode 里恢复就是大部分技能。**它的天花板是示范者**：模仿超不过来源，
而对着 verifier 做 RL 可以找到示范者从未产出的解。**它表达不了对*怎么做*的偏好**：
"更少的工具调用"、"永远不要删文件"、"含糊时先问"，这些都是轨迹分布的性质。

所以配方是两个都要，而诚实的中间地带是 rejection-sampling fine-tuning——从当前策略采样，
留下验证正确的轨迹，在这些上面做 SFT，然后重复。on-policy 的数据、SFT 的机器、
不需要 RL 基础设施。这是一个非常强的 baseline，也常常就是"我们做了 RL"的真实含义。

> **追问**
> - *什么是 rejection sampling fine-tuning（RFT / STaR）？* → 中间地带：从当前策略采样，
>   只留验证正确的轨迹，在这些上面做 SFT，然后重复。on-policy 的数据、SFT 的机器、
>   不需要 RL 基础设施。非常强的 baseline，也常常就是"我们做了 RL"的实际含义。
> - *什么时候 RL 不值得做？* → 没有 verifier 的时候；episode 短到 SFT 就能覆盖分布的时候；
>   或者基础设施成本超过边际收益的时候——这种情况其实很常见，能说出这一点是有判断力的表现。
>
> **陷阱**
> - 直接说"RL 更好"。正确顺序是先 SFT，然后说清 SFT 的三个天花板。


---

> **待补：**multi-agent 与 agent 间通信、memory 架构（短期/长期/情景）、
> planning 与 reflection 的具体机制、RL 基础设施（rollout/训练分离、
> 异步 off-policy 的偏差）、真实产品里的 human-in-the-loop。

---

<a id="section-a13"></a>

## A13 · 对齐、校准与持续学习

★ 全新一节，压缩自我自己的 `agentic-uncertainty` / `agentic-post-training` /
`continual-learning`。Alisa 那本零覆盖。

注意这一节是**技术层面**的对齐与安全——价值观那种"你怎么看 AI safety"的问题在第三篇。

---

<a id="a13-1"></a>
### A13.1 完整的 RLHF 流程


**经典管线（InstructGPT）：**

1. **预训练**一个基座 LM。
2. 在人类对目标行为的示范上做 **SFT**。
3. **采集偏好**：每个 prompt **从 SFT 策略**采样 $$k$$ 个完成，让人来排序。
   从策略采样这一点很关键——你需要的是你将要优化的那个分布上的偏好。
4. 用 Bradley-Terry loss 在这些成对数据上**训一个 reward model**。
5. 对着这个奖励跑 **PPO**，并加一个到 SFT 策略的 KL 惩罚。

**此后变了什么：**

- 只要可能就用**可验证奖励**。checker 胜过学出来的 RM：因果链更短，也不会以同样的方式被钻空子。
- 推理方向上**用 GRPO 取代 PPO**——去掉 critic，改用组均值做 baseline。
- 有静态偏好数据、又想要简单时用 **DPO**，代价是 off-policy。
- **迭代多轮**而不是一次过：生成、评判、重训、再来一遍（Tülu-3 那类配方把这点写得很明确）。
- **AI 反馈**（RLAIF / Constitutional AI）替代了大部分人工标注，人来写*原则*而不是写*标签*。


#### 自测 · A13.1

**Q A13.1.1** — 从头到尾讲一遍 RLHF。InstructGPT 之后有什么变化？

预训练，在示范上做 SFT，**从 SFT 策略**采样 $$k$$ 个完成并让人排序来采集偏好，
训一个 Bradley-Terry reward model，然后对着它跑 PPO，并加一个到 SFT 策略的 KL 惩罚。

值得强调的细节是第三步：偏好是**对策略样本的排序**，不是人类示范。
你需要的是你将要优化的那个分布上的偏好。

此后的变化：领域允许的地方一律用可验证奖励；推理方向上用 GRPO 而不是 PPO；
有静态偏好数据、看重简单性时用 DPO；用迭代多轮而不是一次过；
以及 AI 反馈替代了大部分人工标注——人来写*原则*而不是写*标签*。

> **追问**
> - *KL 惩罚是干什么的？* → 约束相对 SFT 策略的漂移。没有它，优化一个学出来的 reward model
>   就会找到它的失效模式，奖励数字一路上涨而策略丢掉通用能力。
> - *KL 在各个算法里加在哪？* → PPO：从奖励里减掉。GRPO：作为 loss 里的一个 per-token 项，
>   通常用 k3 估计量。DPO：隐式的，通过参考模型体现。
>
> **陷阱**
> - 说偏好数据是"人写的答案"。是**对策略采样的排序**，不是示范。


---

<a id="a13-2"></a>
### A13.2 Constitutional AI 与 RLAIF


**两个阶段。**

*监督阶段。*给模型一个有害的 prompt，让它回答；然后让**模型自己**对照一条写下来的原则
批评这个回答并做修改。在修改后的回答上做微调。模型是自己的标注员。

*RL 阶段（RLAIF）。*生成成对回答，让一个以 constitution 为条件的模型挑出更好的那个。
在这些 AI 生成的标签上训一个偏好模型，然后对着它做 RL。

**它买到了什么：**

- **可扩展性。**人工标注有害性既贵又慢，而且对标注者的心理是一种消耗。
- **透明性。**行为是由一份写下来的、可检视、可编辑的文档规定的，而不是隐含在一堆标签里。
  你可以和一部 constitution *争论*。
- **一致性。**人类标注员之间会有分歧；由模型施加的一条原则至少是统一的。

**该由你自己提出来的那个显然的反对意见。**它继承了模型自身的盲点。如果模型认不出某种危害，
再多的自我批评也不会让它浮出来。所以它扩展的是价值观的*施加*，不是价值观的*发现*——
这也是它并不能免除人工 red-teaming 的原因。


#### 自测 · A13.2

**Q A13.2.1** — 相比 RLHF，Constitutional AI 买到了什么，又没能解决什么？

可扩展性（人工标注有害性既贵又慢，对标注者还是心理消耗）、透明性（行为由一份写下来的、
可检视、可编辑的文档规定，而不是隐含在一堆标签里——你可以和一部 constitution *争论*）、
以及一致性（标注员之间有分歧；由模型施加的一条原则至少是统一的）。

它没能解决的，而且我会自己主动提出来：它继承了模型自身的盲点。如果模型认不出某种危害，
再多的自我批评也不会让它浮出来。它扩展的是价值观的**施加**，不是价值观的**发现**，
这也是它并不能免除人工 red-teaming 的原因。

> **追问**
> - *这和 scalable oversight 是什么关系？* → 同一类问题：你怎么在自己都评不了的任务上
>   监督一个系统。其他路子还有：debate（两个模型辩论，一个更弱的裁判来判）、
>   recursive reward modelling、weak-to-strong generalisation。
> - *什么是 weak-to-strong generalisation？* → 一个弱的监督者能不能引出强模型的全部能力？
>   经验上是部分可以——在弱标签上微调的强模型会超过那个弱监督者，
>   这对 superalignment 的情形算是一个不算强的鼓励信号。
>
> **陷阱**
> - 只说"用 AI 代替人标注"。要主动说出它继承模型盲点这个局限。


---

<a id="a13-3"></a>
### A13.3 校准的定义与度量


**定义。**一个模型是校准的，如果它声明的置信度与它的经验准确率一致：
在所有以置信度 $$c$$ 做出的预测里，应该有 $$c$$ 的比例是对的。

$$\mathbb P\big(\text{correct} \mid \text{confidence}=c\big) = c \quad \forall c$$

**度量 —— Expected Calibration Error。**按置信度把预测分箱，在每个箱内比较准确率与平均置信度：

$$\text{ECE} = \sum_{b=1}^{B}\frac{n_b}{n}\big|\,\text{acc}(b) - \text{conf}(b)\,\big|$$

**ECE 的陷阱，至少要能说出两条：**

- **依赖分箱。**箱的数量和位置会改变这个值；自适应分箱（等质量而不是等宽）更稳健。
- **它不是 proper scoring rule。**一个永远输出基础率的模型能拿到 ECE 0，同时毫无用处。
  **永远要把 accuracy 一起报出来。**
- **它把你真正在乎的区域平均掉了。**危险的是高置信度下的错误，而 ECE 按频率而不是按代价
  给它们加权。

更好的搭档：**Brier score**（proper，可分解为 calibration + refinement）、
**选择性准确率 / risk-coverage 曲线**（准确率作为你选择作答的比例的函数）、
以及**错误预测的 AUROC**。


#### 自测 · A13.3

**Q A13.3.1** — 定义校准，并说出 ECE 的两个陷阱。

校准的意思是声明的置信度与经验准确率一致：在以置信度 $$c$$ 做出的预测里，有 $$c$$ 的比例是对的。

ECE 的陷阱：它**依赖分箱**，箱的数量和位置会改变这个值；以及它**不是 proper scoring rule**——
一个永远输出基础率的模型能拿到 ECE 0 却毫无用处，所以你必须总是把 accuracy 一起报出来。
还值得补第三条：它把你真正在乎的区域平均掉了，对高置信度下的错误按频率而不是按代价加权。

更好的搭档是 Brier score——它是 proper 的，并且可以分解为 calibration 加 refinement——
以及 risk-coverage 曲线，它回答的是你真正关心的问题：如果模型对最没把握的那一部分选择不答，
准确率能提高多少。

> **追问**
> - *那个置信度数字从哪来？* → 三个来源，性质各不相同：答案的 token 概率；
>   口头表述的置信度（"我有 80% 把握"）；或者采样一致性（$$k$$ 个样本之间的 self-consistency）。
>   它们互相不一致，哪个最好取决于任务。
> - *基座模型是校准的吗？* → 在多选题上还算是。后训练会把它破坏掉——见下一节。
>
> **陷阱**
> - 只报 ECE 不报 accuracy。ECE 单独看可以被平凡地刷到 0。


---

<a id="a13-4"></a>
### A13.4 为什么后训练会破坏校准


**先说这一句。**每一个后训练算子优化的目标都**对校准无所谓**，其中好几个还在主动奖励自信。

逐个算子看：

- **SFT。**训练数据是一水儿自信且正确的示范。模型学到的是自信的*风格*，
  与它到底知不知道完全解耦。
- **用人类偏好的 RLHF。**人偏好自信、流畅的回答，模棱两可读起来就是没帮上忙。
  于是自信被直接奖励，与正确性无关。
- **RLVR。**奖励是最终答案对/错的二值信号。里面没有任何东西约束模型*声明的*置信度，
  所以置信度会朝优化推动的方向漂——通常是往上，因为在 SFT 数据里自信格式的回答与正确回答相关。
- **Best-of-N / 拒绝采样。**挑出最好的样本会锐化输出分布，并丢掉模型自身的不确定性信号。

**机制版本。**在二值奖励上做 RL 会把概率质量推向 argmax，熵坍塌。
模型的 token 概率不再是一个可用的不确定性估计——不是因为它知道得更少，
而是因为这个分布不再代表信念。

**修法，一句话：把置信度训练到模型自己的成功率上。**与其事后校准，
不如把目标设成模型在那类输入上的经验准确率，让置信度由结果而不是由风格来监督。


#### 自测 · A13.4

**Q A13.4.1** — 经过 RLHF 的模型以过度自信著称。是哪个算子造成的？

全都有份，原因各不相同，好的回答会把这个阶梯走一遍。**SFT** 训练在一水儿自信的示范上，
所以模型学到的是自信的*风格*，与知不知道解耦。**用人类偏好的 RLHF** 直接奖励自信，
因为模棱两可读起来就是没帮上忙。**RLVR** 对声明的置信度只字未提，
所以它会朝优化推动的方向漂——通常往上。**Best-of-N** 锐化分布，并丢掉不确定性信号。

机制上：在二值奖励上做 RL 会把质量推向 argmax，熵坍塌，于是 token 概率不再代表信念——
不是因为模型知道得更少，而是因为这个分布已经不是信念分布了。

修法一句话：**把置信度训练到模型自己的成功率上**，让它由结果而不是由风格来监督。
Temperature scaling 是便宜的事后选项，它能修好平均校准，但修不好高置信度那条尾巴，
而那条尾巴才是要紧的部分。

> **追问**
> - *推理有帮助吗？* → 部分有。更长的链条能改善*准确率*，跨样本的 self-consistency
>   也确实给出比单个口头数字更好的不确定性信号。但推理模型口头表述的置信度并不会自动更校准——
>   它同样被那几个算子优化过。
> - *有事后的修法吗？* → 在 held-out 集上做 temperature scaling 是便宜的标准做法。
>   它修好平均校准，但修不好高置信度那条尾巴，而那才是要紧的部分。
>
> **陷阱**
> - 只说"RLHF 让模型过度自信"。要能逐个 operator 说清机制。


---

<a id="a13-5"></a>
### A13.5 Agent 的校准有什么不同


**三件事变了：**

1. **不确定性会累积。**每步可靠性 95% 的 20 步轨迹，成功率是 36%。
   要看的量是**轨迹级**置信度而不是每步的，而且它不是每步置信度的乘积，因为步骤之间相关。
2. **你可以据此行动。**chat 模型只能在文字上模棱两可，agent 却可以
   **问一个澄清问题、跑一次便宜的验证、先做一个可逆的动作，或者上报给人。**
   不确定性变成了控制信号，而不只是一份报告。
3. **犯错的代价是非对称且不可逆的。**删掉一个文件、发出一封邮件、完成一笔付款。
   所以正确的阈值不是"置信度 > 0.5"，而是一个权衡了动作代价的决策论阈值。

**这对设计意味着什么。**你要的是**决策点上**的置信度估计，而不是最后才给；
而且它要绑定一条策略：低于阈值 → 验证、询问，或上报。有意思的研究问题是：
怎么训练模型产出一个*针对那个决策*校准的置信度，而不是一个通用的置信度。


#### 自测 · A13.5

**Q A13.5.1** — 为什么长时程 agent 的不确定性比单个答案的更难处理？

三点变化。**它会累积**——每步可靠性 95% 的 20 步任务，成功率是 36%，
而你需要的量是轨迹级置信度，它*不是*每步置信度的乘积，因为步骤之间相关。
**你可以据此行动**——agent 可以问一个澄清问题、跑一次便宜的验证、先做一个可逆动作，或者上报；
不确定性成了控制信号而不是报告。**犯错的代价是非对称且常常不可逆的**，
所以阈值是决策论的，不是"置信度 > 0.5"。

对设计的含义是：你要的是**决策点上**校准好的置信度，并且绑定一条策略——低于阈值就验证或上报。
评估则用轨迹层面的 risk-coverage 曲线：如果 agent 在最没把握的 $$x\%$$ 上选择上报，
剩下那部分的成功率能提高多少？

> **追问**
> - *怎么评估 agent 的校准？* → 轨迹层面的 risk-coverage 曲线：如果 agent 在最没把握的
>   $$x\%$$ 上弃答或上报，剩下那部分的成功率能提高多少？校准良好的 agent 曲线会很陡。
> - *这和"你敢交给它多少张 GPU"那个指标有什么联系？* → 那个指标测的是信任，
>   而信任恰好就是校准好的不确定性加上有界的下行风险。
>   当你能预测它什么时候会失败，你就敢多授权。
>
> **陷阱**
> - 把 agent 的校准当成"每步概率相乘"。步骤是相关的，而且真正要估的是轨迹级成败。


---

<a id="a13-6"></a>
### A13.6 灾难性遗忘


**到底发生了什么。**在新分布上做梯度下降，会挪动那些编码了旧分布的权重。
目标函数里没有任何一项在说"继续保持你已经会的东西"——旧数据只是干脆不在 loss 里。

**缓解手段，大致按实用程度排序：**

1. **Replay / 数据混合。**把一部分原始分布掺进微调数据里。无聊，但也是效果远远最好的做法。
   常见比例是 5–20%。
2. **更低的学习率 + 更少的步数。**大部分遗忘来自在新领域上训过头。
3. **参数高效方法。**LoRA 把更新约束在一个低秩子空间里，限制了你能走多远——
   遗忘几乎是被构造性地界住的。而且你可以把 adapter *卸下来*，全量微调做不到这一点。
4. **对原模型做 KL / 蒸馏正则。**在一个参考分布上显式惩罚漂移。
   这和 RLHF 的 KL 惩罚是同一套机制，用途不同。
5. **经典方法** —— EWC（惩罚那些 Fisher 信息认为重要的参数发生移动）、梯度投影。
   优雅，但在 LLM 规模上很少用，因为 replay 花更少力气就能做得更好。


#### 自测 · A13.6

**Q A13.6.1** — 你在一个新领域上微调之后，通用能力掉了。怎么办？

先上 replay。把 5–20% 的原始分布掺进微调数据里。这做法很无聊，但效果远远最好，
因为真正的原因就是旧数据压根不在 loss 里——目标函数里没有任何一项在说"继续保持你已经会的东西"。

然后：降学习率、减步数，因为大部分遗忘来自在新领域上训过头。再然后是 LoRA，
它几乎是构造性地把更新界定在一个低秩子空间里，还有一个被低估的性质——你可以把它*卸下来*。
再然后是对原模型做 KL 或蒸馏正则——和 RLHF 的 KL 惩罚是同一套机制，用途不同。

EWC 这类经典方法很优雅，但在 LLM 规模上很少用，因为 replay 花更少力气就能做得更好。
上来就讲 EWC，会让人看出这个答案来自教科书而不是来自真实的 run。

> **追问**
> - *遗忘一定是坏事吗？* → 不是。unlearning 有时候正是目标（移除某种能力、PII、
>   某个受版权保护的作品）。问题在于它目前是不加区分的。
> - *多时间尺度这个框架为什么有用？* → 它把"什么东西该按哪个时钟变"分开了：
>   权重（慢、贵、永久）、上下文（快、便宜、易逝）、外部记忆（居中，可编辑）。
>   大多数"持续学习"的产品需求其实是记忆需求，不是权重更新需求——
>   把两者混淆的结果，就是本该建一个存储却去做了微调。
>
> **陷阱**
> - 直接跳到 EWC 之类的算法。**replay 是最有效的**，先说它。


---

<a id="a13-7"></a>
### A13.7 部署之后的学习


**这个循环：**

1. **记录**带结果的轨迹——有显式反馈就用显式的，没有就用隐式信号
   （用户有没有重试、有没有接受这个 diff、测试过没过）。
2. **筛选**出结果信号可信的轨迹。这是难的一步：大部分生产流量没有 ground truth。
3. **验证**，在可能的地方——重跑测试，检查代码是否仍能编译。
4. **整理**成训练数据，去重，并对着你的评测集做去污染。
5. **训练**——在验证成功的轨迹上做 RFT 是最安全的形式；如果你有可靠的奖励就做 RL。
6. **评测**，上线前对着一套冻结的评测跑一遍，专门检查有没有退化。

**风险，而风险才是这道题的实质：**

- **反馈回路。**模型塑造了它随后要训练的那个分布。受欢迎的行为会被强化，不管它好不好，
  而分布随时间不断收窄。
- **分布往错误的方向漂。**留下来的用户正是模型已经服务得不错的那批，
  于是你对他们过拟合，对其他所有人变得更差。
- **没有 ground truth。**隐式信号被严重混淆——用户重试可能意味着答案错了，
  也可能只是他改了主意。
- **污染自己的评测集**，如果生产数据漏了进去。
- **隐私。**生产数据就是用户数据。这首先是一条法律约束，其次才是技术约束。


#### 自测 · A13.7

**Q A13.7.1** — 设计一个从生产用量中学习的循环。风险是什么？

记录带结果的轨迹，筛出结果信号可信的那些，能验证的地方重跑测试来验证，
整理成训练数据并对着评测集去污染，训练——在验证成功的轨迹上做 rejection-sampling fine-tuning
是最安全的形式——上线前再对着一套冻结的评测跑一遍。

风险才是实质。**反馈回路**：模型塑造了它随后要训练的分布，于是受欢迎的行为不管好坏都被强化，
分布不断收窄。**往错误方向漂**：留下来的用户正是已经被服务得不错的那批，于是你对他们过拟合。
**没有 ground truth**：隐式信号被严重混淆——一次重试可能意味着答案错了，也可能是用户改了主意。
**评测污染**，如果生产数据漏了进去。以及**隐私**，它首先是一条法律约束，其次才是技术约束。

第一条的检测办法：监控输出多样性随时间的变化，而不只是看总体质量。
坍塌总是先表现为收窄，然后才表现为质量下降。

> **追问**
> - *这个循环最安全的版本是什么？* → 只做已验证的 RFT：只保留自动 checker 确认成功的轨迹。
>   良率低，可信度高。
> - *怎么发现反馈回路？* → 监控输出多样性和行为分布随时间的变化，不只是总体质量。
>   坍塌总是先表现为收窄，然后才表现为质量下降。
>
> **陷阱**
> - 只画流程不说风险。反馈回路和缺乏 ground truth 才是这题的重点。


---

<a id="a13-8"></a>
### A13.8 监控，以及为什么不要在 CoT 上做训练


**简短的回答：不该，而这个理由值得仔细说清楚。**

chain of thought 作为一个**监控面**之所以有价值，恰恰因为它没有被优化过。
它是一扇相对忠实地反映计算过程的窗。你一旦把它变成训练目标——去惩罚"坏想法"——
你优化出来的就是*看起来*可接受的推理，而不是*本身*可接受的推理。
你没有消除那个行为，你消除的是你看见它的能力。

**这条原则，一句话：惩罚动作，不惩罚想法。**约束 agent *做*了什么；让推理保持可读。

**要承认的那个不舒服的取舍。**这意味着接受模型有时会想一些你不喜欢的东西，
并且选择让它保持可见，而不是把它逼到地下。这也意味着 CoT 的可监控性是一个
**你可能会失去的性质**——随着模型被优化得越来越狠，即使没有任何人直接盯着它，
它也可能自行退化。


#### 自测 · A13.8

**Q A13.8.1** — chain of thought 该不该成为安全训练的目标？

不该，而这个理由值得仔细说清楚。chain of thought 作为一个**监控面**之所以有价值，
恰恰因为它没有被优化过——它是一扇相对忠实地反映计算过程的窗。把它变成训练目标，
你优化出来的就是*看起来*可接受的推理，而不是*本身*可接受的推理。
你没有消除那个行为，你消除的是你看见它的能力。

这条原则一句话：**惩罚动作，不惩罚想法**。

要承认的不舒服的部分：这意味着接受模型有时会想一些你不喜欢的东西，
并且选择让它保持可见，而不是把它逼到地下。而且可监控性是一个你可能会**失去**的性质——
随着模型被优化得越来越狠，即使没有人盯着它，它也可能自行退化。

> **追问**
> - *今天的 CoT 忠实吗？* → 只是部分忠实。模型有时会产出并不决定答案的推理（事后合理化），
>   也可能被它从未提及的线索影响。所以它是一个有用但不完美的信号——值得监控，
>   但不值得完全信任。
> - *要监控什么？* → 推理模式的分布漂移，不只是奖励。奖励在涨而 held-out 走平，
>   是经典的 reward hacking 特征。
>
> **陷阱**
> - 说"当然要对 CoT 做安全训练"。这会让你失去唯一的观测窗口。


---

<a id="a13-9"></a>
### A13.9 越狱与对抗鲁棒性


**它们为什么有效。**对齐训练覆盖的是一个输入分布，jailbreak 找的就是这个分布之外的输入。
基座模型保留了全部能力——拒绝只是一层薄薄的行为层，不是把能力删掉了。
所以攻击面就是"找一个让拒绝行为不触发的框架"。

**几大类：**

- **角色扮演 / 换框架** —— 虚构、假设情境、"你现在是 DAN"。
- **编码** —— base64、低资源语言、leetspeak；有害内容确实在那里，但拒绝分类器认不出来。
- **Many-shot** —— 用大量顺从的例子填满长上下文；in-context learning 压过了训练出来的行为，
  而且上下文窗口越大它*越严重*。
- **基于优化** —— GCG 那类用梯度搜索找到的对抗后缀；值得注意的是它们能跨模型**迁移**。
- **Prompt injection** —— 对 agent 来说，攻击是从*检索到的内容或工具输出*进来的，
  不是从用户那一轮。这是对产品最要紧的一类。

**防御，诚实地排个序：**

1. **纵深防御。**独立于模型之外的输入和输出分类器。互相独立的失效模式胜过一层很强的防线。
2. **对抗训练**，针对已知攻击。对那些攻击有用；泛化很差。
3. **对 agent：最小权限。**把所有检索到的内容都当作不可信，把数据通道和指令通道分开，
   不可逆的动作要求确认。**这是对 prompt injection 唯一的结构性防御**——
   你没法靠写 prompt 绕出去。
4. **监控与限流。**假设总有攻击会成功；限制爆炸半径。


#### 自测 · A13.9

**Q A13.9.1** — jailbreak 为什么有效，真正能防住它的是什么？

它们有效，是因为对齐训练覆盖的是一个输入分布，而 jailbreak 找的是这个分布之外的输入。
基座模型保留了每一项能力——拒绝只是一层薄薄的行为层，不是删除——
所以攻击就是"找一个让拒绝不触发的框架"。

对产品来说最要紧的一类是 **prompt injection**：攻击是从检索内容或工具输出进来的，
不是从用户那一轮。当前的共识是它**在模型层面无解**：
当指令和数据都以文本形式到达时，模型无法可靠地区分二者。

所以防御是结构性的，不是对齐训练。对 agent 用最小权限：把所有检索到的内容当作不可信，
把数据通道和指令通道分开，不可逆的动作要求确认。围绕这一层再加纵深防御——
用独立于模型失效的输入/输出分类器——以及在"总有攻击会成功"的假设下做监控和限流。

> **追问**
> - *prompt injection 在模型层面能解决吗？* → 当前共识是不能。当指令和数据都以文本形式到达时，
>   模型无法可靠地区分二者。这是一个架构和权限问题。
> - *为什么 many-shot jailbreak 随着上下文变长而变严重？* → 更多的上下文示例意味着更强的
>   in-context learning，它会直接和训练出来的拒绝行为竞争。
>
> **陷阱**
> - 只答"用 RLHF 训练拒绝"。对 agent 来说 prompt injection 需要的是**权限设计**，不是对齐训练。


---

> **待补：**可解释性（SAE、features、circuits）、debate 与 recursive reward modeling、
> unlearning 的技术方案、model organism 与 alignment faking、
> 评测对齐税（alignment tax）的方法。

---

<a id="section-refs"></a>

## 参考文献

按依赖它们的章节分组，方便从某个概念直接跳到原文。下面每一个 arXiv ID 都过了
arXiv API 核验，见 `refs.py`。


### A1 · 基础

- **Adam** — Adam: A Method for Stochastic Optimization. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- **AdamW / decoupled weight decay** — Decoupled Weight Decay Regularization. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- **Layer Normalization** — Layer Normalization. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- **RMSNorm** — Root Mean Square Layer Normalization. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- **Deep double descent** — Deep Double Descent: Where Bigger Models and More Data Hurt. [arXiv:1912.02292](https://arxiv.org/abs/1912.02292)
- **Batch Normalization** — Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

### A2 · 架构

- **Attention Is All You Need** — Attention Is All You Need. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **RoPE (RoFormer)** — RoFormer: Enhanced Transformer with Rotary Position Embedding. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **MQA (Fast Transformer Decoding)** — Fast Transformer Decoding: One Write-Head is All You Need. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- **GQA** — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- **SwiGLU (GLU Variants)** — GLU Variants Improve Transformer. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- **Pre-LN vs post-LN** — On Layer Normalization in the Transformer Architecture. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- **Sparsely-gated MoE** — Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- **Switch Transformer** — Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- **BPE for NMT** — Neural Machine Translation of Rare Words with Subword Units. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- **Mamba** — Mamba: Linear-Time Sequence Modeling with Selective State Spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- **Linear attention** — Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
- **Longformer (sliding window)** — Longformer: The Long-Document Transformer. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
- **LLaVA (vision projector)** — Visual Instruction Tuning. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
- **BERT** — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **T5** — Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)

### A3 · 常见模型

- **The Llama 3 Herd of Models** — The Llama 3 Herd of Models. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- **DeepSeek-V2 (MLA)** — DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- **DeepSeek-V3** — DeepSeek-V3 Technical Report. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- **DeepSeek-R1** — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- **Qwen3** — Qwen3 Technical Report. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- **Mixtral of Experts** — Mixtral of Experts. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
- **GPT-3 (few-shot)** — Language Models are Few-Shot Learners. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

### A4 · 预训练

- **Multi-token prediction** — Better & Faster Large Language Models via Multi-token Prediction. [arXiv:2404.19737](https://arxiv.org/abs/2404.19737)
- **Chinchilla** — Training Compute-Optimal Large Language Models. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- **muP / muTransfer** — Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)

### A5 · 训练基础设施

- **ZeRO** — ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- **Megatron-LM** — Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- **Efficient large-scale training (PTD-P)** — Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
- **GPipe** — GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
- **Zero Bubble pipeline** — Zero Bubble Pipeline Parallelism. [arXiv:2401.10241](https://arxiv.org/abs/2401.10241)
- **Mixed precision training** — Mixed Precision Training. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- **Gradient checkpointing** — Training Deep Nets with Sublinear Memory Cost. [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- **Ring attention** — Ring Attention with Blockwise Transformers for Near-Infinite Context. [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)

### A6 · Post-training 与 RL

- **PPO** — Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- **GAE** — High-Dimensional Continuous Control Using Generalized Advantage Estimation. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- **InstructGPT / RLHF** — Training language models to follow instructions with human feedback. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **GRPO (DeepSeekMath)** — DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **DPO** — Direct Preference Optimization: Your Language Model is Secretly a Reward Model. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **DAPO** — DAPO: An Open-Source LLM Reinforcement Learning System at Scale. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- **IPO** — A General Theoretical Paradigm to Understand Learning from Human Preferences. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
- **KTO** — KTO: Model Alignment as Prospect Theoretic Optimization. [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)
- **SimPO** — SimPO: Simple Preference Optimization with a Reference-Free Reward. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
- **Distilling the knowledge** — Distilling the Knowledge in a Neural Network. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- **LoRA** — LoRA: Low-Rank Adaptation of Large Language Models. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **QLoRA** — QLoRA: Efficient Finetuning of Quantized LLMs. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **Reward model overoptimization** — Scaling Laws for Reward Model Overoptimization. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)

### A7 · 推理模型与 test-time compute

- **Chain-of-thought prompting** — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- **Self-consistency** — Self-Consistency Improves Chain of Thought Reasoning in Language Models. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- **Scaling test-time compute** — Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
- **Process supervision (PRM)** — Let's Verify Step by Step. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)

### A8 · 推理与服务

- **FlashAttention** — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- **FlashAttention-2** — FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- **Online softmax** — Online normalizer calculation for softmax. [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- **PagedAttention / vLLM** — Efficient Memory Management for Large Language Model Serving with PagedAttention. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- **Speculative decoding** — Fast Inference from Transformers via Speculative Decoding. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- **Speculative sampling** — Accelerating Large Language Model Decoding with Speculative Sampling. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- **Medusa** — Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- **Nucleus sampling** — The Curious Case of Neural Text Degeneration. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
- **LLM.int8()** — LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- **SmoothQuant** — SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- **GPTQ** — GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- **AWQ** — AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- **Position interpolation** — Extending Context Window of Large Language Models via Positional Interpolation. [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)
- **YaRN** — YaRN: Efficient Context Window Extension of Large Language Models. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- **Lost in the middle** — Lost in the Middle: How Language Models Use Long Contexts. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
- **Chunked prefill (Sarathi)** — SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills. [arXiv:2308.16369](https://arxiv.org/abs/2308.16369)

### A9 · 数据

- **Deduplicating training data** — Deduplicating Training Data Makes Language Models Better. [arXiv:2107.06499](https://arxiv.org/abs/2107.06499)
- **Data-constrained scaling (4 epochs)** — Scaling Data-Constrained Language Models. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)
- **FineWeb** — The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
- **LIMA** — LIMA: Less Is More for Alignment. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
- **Model collapse** — The Curse of Recursion: Training on Generated Data Makes Models Forget. [arXiv:2305.17493](https://arxiv.org/abs/2305.17493)
- **Self-Instruct** — Self-Instruct: Aligning Language Models with Self-Generated Instructions. [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)

### A11 · Scaling 与评测

- **Kaplan scaling laws** — Scaling Laws for Neural Language Models. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- **Porian et al. — resolving the discrepancy** — Resolving Discrepancies in Compute-Optimal Scaling of Language Models. [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)
- **Emergent abilities** — Emergent Abilities of Large Language Models. [arXiv:2206.07682](https://arxiv.org/abs/2206.07682)
- **Emergence as a mirage** — Are Emergent Abilities of Large Language Models a Mirage?. [arXiv:2304.15004](https://arxiv.org/abs/2304.15004)
- **LLM-as-a-judge / MT-Bench** — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- **RULER (long-context eval)** — RULER: What's the Real Context Size of Your Long-Context Language Models?. [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)

### A12 · Agentic RL

- **ReAct** — ReAct: Synergizing Reasoning and Acting in Language Models. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- **SWE-bench** — SWE-bench: Can Language Models Resolve Real-World GitHub Issues?. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- **tau-bench** — $\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. [arXiv:2406.12045](https://arxiv.org/abs/2406.12045)

### A13 · 对齐与校准

- **Constitutional AI** — Constitutional AI: Harmlessness from AI Feedback. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- **RLAIF** — RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. [arXiv:2309.00267](https://arxiv.org/abs/2309.00267)
- **Weak-to-strong generalization** — Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision. [arXiv:2312.09390](https://arxiv.org/abs/2312.09390)
- **On calibration of modern neural networks** — On Calibration of Modern Neural Networks. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- **GCG universal adversarial attacks** — Universal and Transferable Adversarial Attacks on Aligned Language Models. [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
- **Alignment faking** — Alignment faking in large language models. [arXiv:2412.14093](https://arxiv.org/abs/2412.14093)
- **EWC** — Overcoming catastrophic forgetting in neural networks. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)

### 非 arXiv 来源


- Alisa Liu，*The Book of LLMs* — [https://alisawuffles.notion.site/](https://alisawuffles.notion.site/)  
  她 2026 年从博士到 OpenAI 求职过程中公开的笔记，是 A1–A6 的主要底本。
- Stas Bekman，*Machine Learning Engineering* — [https://github.com/stas00/ml-engineering](https://github.com/stas00/ml-engineering)  
  A5.5 里 loss 尖峰的分类和 data sampler 那条警告出自这里。
- John Schulman，*Approximating KL divergence* — [http://joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html)  
  A6.7 的 GRPO loss 里用的 k3 估计量。
- NVIDIA H100 datasheet — [https://resources.nvidia.com/en-us-hopper-architecture](https://resources.nvidia.com/en-us-hopper-architecture)  
  A10.0 的硬件锚点：989 TFLOP/s dense bf16、3.35 TB/s HBM、80 GB。
