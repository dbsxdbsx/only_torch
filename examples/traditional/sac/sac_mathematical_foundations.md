# SAC 变体的数学基础与严谨性分析

> 本文档系统梳理 SAC（Soft Actor-Critic）在离散、连续、混合动作空间下的数学基础，
> 重点分析各变体的收敛保证、已知理论缺口、以及实践中的应对策略。
>
> 参考论文：
> - Haarnoja et al. 2018a — SAC v1（连续，手动 α）
> - Haarnoja et al. 2018b — SAC v2（连续，自动 α）
> - Christodoulou 2019 — SAC-Discrete（离散）
> - Delalleau et al. 2019 — Hybrid SAC（混合）
> - Chou et al. 2017 — Beta Policy（有界连续）
> - Ward et al. 2019 — SAC + Normalizing Flows（更具表达力的策略）
>
> 论文副本存放在 [`.doc/paper/RL/`](../../.doc/paper/RL/) 和 AI 论文目录。

---

## 1. Hybrid Action 的通用框架（Delalleau 2019）

### 1.1 通用动作分解

Hybrid SAC 将动作 `a` 分解为离散部分和连续部分：

- **离散**：`ad = (ad_1, ..., ad_D)`，每个 `ad_i` 取值 `1..K_i`
- **连续**：`ac = (ac_1, ..., ac_C)`，每个 `ac_j` 是 `m_j` 维连续向量

策略分解（条件独立假设）：

```
π(a|s) = Π_i π(ad_i|s) · Π_j π(ac_j|s, ad)
```

### 1.2 六种实例化模式

论文列举了 6 种具体模式，涵盖了实际场景的绝大多数需求：

| # | 描述 | D | C | 典型场景 |
|---|------|:-:|:-:|----------|
| 1 | 单组离散动作 | 1 | 0 | Atari 游戏 |
| 2 | 多组独立 1D 连续动作 | 0 | 多 | 标准连续控制（每维独立 Gaussian） |
| 3 | 单组 m 维连续动作 | 0 | 1 | Normalizing Flows（允许维度间相关） |
| 4 | 1 离散 + 1 连续（连续**依赖**离散） | 1 | 1 | 参数化动作空间（连续头以离散动作为输入） |
| 5 | 1 离散 + K 组独立连续（每个离散值配专属连续头） | 1 | K | 参数化动作空间变体（更灵活，支持不同范围/维度） |
| 6 | 多组独立离散动作 | 多 | 0 | 离散化连续空间（按维度分桶） |

**要点**：

- 模式 #5 天然支持**不同离散动作对应不同连续范围/维度**（如：动作 1 连续范围 [-1,1]，动作 2 连续范围 [-2,2]）
- 模式 #4 通过将离散动作作为连续头的输入，也能实现类似效果，但参数更紧凑
- 上述模式可自由组合：D > 1 且 C > 0 的一般形式覆盖了几乎所有游戏/机器人场景

---

## 2. 收敛性保证对比

### 2.1 原版连续 SAC 的理论保证

Haarnoja 2018 提供了三个严格定理（表格形式收敛证明）：

| 定理 | 内容 | 关键条件 |
|------|------|---------|
| **Lemma 1: 软策略评估** | 软 Bellman 算子 T^π 是压缩映射，重复应用收敛到 Q^π | γ < 1 |
| **Lemma 2: 软策略改进** | 每次更新 `π_new = argmin KL(π \|\| exp(Q/α)/Z)` 后 `Q^{π_new} ≥ Q^{π_old}` | 单一 α，KL 方向为 `π` 在前 |
| **Theorem 1: 软策略迭代** | 交替评估与改进收敛到最优 π* | 策略类足够丰富 |

### 2.2 各变体的严谨性

| 维度 | 连续 SAC (Haarnoja) | 离散 SAC (Christodoulou) | Hybrid SAC (Delalleau) |
|------|:------------------:|:----------------------:|:---------------------:|
| **形式化收敛证明** | 有（Theorem 1） | 有（积分→求和，直接套用） | **无** |
| **熵定义严谨性** | 微分熵（可为负，有隐患） | Shannon 熵（严格非负） | 混合两种熵（隐患更大） |
| **温度参数** | 单一 α | 单一 α | 双 α_d + α_c（无证明） |
| **策略类限制** | Gaussian（可用 NF 扩展） | Softmax | 因式化乘积族 |
| **实际可靠性** | 高 | 高 | 中高（视任务而定） |

### 2.3 Hybrid SAC 的理论现状

Hybrid SAC 是一篇 AAAI Workshop 论文（非主会），**偏实用性，不追求完整理论**。

- **没有 fundamental 错误**：每个组件（链式法则、重参数化、精确枚举、KL 最小化）都数学正确
- **缺少端到端的收敛证明**：各组件正确不等于整体有收敛保证
- 属于"证明还没写出来"而非"存在反例证明它不成立"

---

## 3. KL 散度的方向问题

### 3.1 不对称性

```
KL(P || Q) = E_P[log(P/Q)]    ≠    KL(Q || P) = E_Q[log(Q/P)]
```

两个方向的行为截然不同：

| 方向 | 名称 | 行为 |
|------|------|------|
| KL(π \|\| target) | reverse KL / I-projection | **模式追踪**（mode-seeking）：π 倾向坍缩到 target 最高峰 |
| KL(target \|\| π) | forward KL / M-projection | **均值覆盖**（mean-seeking）：π 必须覆盖 target 的所有峰 |

### 3.2 SAC 的选择

**所有版本的 SAC 统一使用 KL(π \|\| target)**，即 **π 在前**：

```
J_π(φ) = E_s [ KL( π_φ(·|s)  ||  exp(Q(s,·)/α) / Z(s) ) ]
```

展开后丢掉不依赖 φ 的 log Z(s)：

```
J_π(φ) = E_{s,ε} [ α · log π(f_φ(ε;s) | s) - Q(s, f_φ(ε;s)) ]
```

### 3.3 选择此方向的三个原因

**原因 1 — 可计算性（最关键）**：
- KL(π \|\| target) 对 **π 采样**，可用重参数化技巧，且**不需要知道 Z(s)**
- KL(target \|\| π) 需要对 `exp(Q/α)/Z` 采样，Z(s) 在连续空间不可计算

**原因 2 — 收敛证明依赖此方向**：
- Lemma 2 的推导利用了 `KL(π \|\| target) ≥ 0` 的性质
- 换方向需要全新证明

**原因 3 — mode-seeking + 熵正则化 = 合理平衡**：
- KL(π \|\| target) 的 mode-seeking 会使策略坍缩到 Q 最高的动作
- SAC 的 `+α·H(π)` 熵奖励对抗坍缩，两者形成动态平衡
- 如果用 mean-seeking + 熵正则化，策略会过于分散

### 3.4 在各变体中的一致性

| 版本 | KL 方向 | 实现方式 |
|------|--------|---------|
| 连续 SAC | KL(π_c \|\| exp(Q/α)/Z) | 重参数化采样，丢掉 log Z |
| 离散 SAC | KL(π_d \|\| exp(Q/α)/Z) | 精确求和（有限动作） |
| Hybrid 离散部分 | KL(π_d \|\| exp(q_d/α_d)/Z_d) | 精确求和 |
| Hybrid 连续部分 | KL(π_c(·\|s,ad) \|\| exp(Q(s,ad,·)/α_c)/Z_c) | 重参数化 + 对 ad 加权 |

**方向完全一致**，这一点在所有版本中都严格成立。

### 3.5 KL 方向的已知局限

Delalleau 论文在 Normalizing Flows 实验（Figure 3）中展示了一个重要发现：

- KL(π \|\| target)（SAC 的选择）→ 无论 Gaussian 还是 NF，策略都坍缩到目标的单峰
- KL(target \|\| π)（反方向）→ NF 策略可以捕获多峰
- Jensen-Shannon（对称）→ NF 效果最好

结论：**SAC 的 KL 方向限制了多峰策略的表达能力**，但换方向目前不可计算。这是一个开放研究方向。

---

## 4. 双温度 (α_d, α_c) 的理论分析

### 4.1 引入动机

离散和连续的熵量级天差地别。例如 3 个离散动作 + 10 维连续：

```
H_d_max ≈ 1.1       （= ln(3)）
H_c 典型值 ≈ 14.2    （10 维 Gaussian，σ ≈ 1）
```

单一 α 时，93% 的熵信号来自连续部分，离散部分的探索被"淹没"。

### 4.2 双温度的理论地位

| 方面 | 单一 α | 双 α_d + α_c |
|------|:------:|:------------:|
| 收敛证明 | 可直接套用原版 SAC | **无形式化证明** |
| 尺度平衡 | 差（一方淹没另一方） | 好（各自独立调节） |
| 统一目标函数 | 有（单一联合 KL） | 无（分解为两个独立 KL） |
| 自动调节 | 一个 target entropy | 两个 target entropy（交互未分析） |

### 4.3 可能的中间路线

1. **归一化后用单一 α**：`H_normalized = H_d / ln(K) + H_c / H_c_ref`，保持收敛证明
2. **固定比例**：`α_d = k · α_c`，只自动调节一个，保持单一自由度
3. **重新诠释为修改参考测度**：双 α 可视为在缩放后的乘积测度上的单 α 问题（事后合理化，非严格）

---

## 5. 测度论基础

### 5.1 核心问题

Hybrid SAC 将两种不同的"熵"相加：

- **Shannon 熵**（离散）：`H_d = -Σ p(x) log p(x)`，恒非负
- **微分熵**（连续）：`H_c = -∫ f(x) log f(x) dx`，可以为负

这两者是数学上**不同类型的对象**。

### 5.2 严格处理方式

在乘积空间 `{1,...,K} × R^m` 上定义乘积参考测度：

```
μ = 计数测度(离散) × Lebesgue 测度(连续)
```

策略密度：`f(ad, ac) = π(ad) · p(ac | ad)`

统一熵：

```
H_μ(π) = -Σ_{ad} ∫ π(ad)·p(ac|ad) · log[π(ad)·p(ac|ad)] dac
        = H_Shannon(ad) + E_{ad}[H_differential(ac|ad)]     ← 链式法则
```

在此统一测度下，用单一 α 的 SAC 收敛证明**可以直接推广到混合空间**。

### 5.3 论文的处理

Delalleau 论文原文承认：

> "Here we slightly abuse notations... A rigorous treatment would rely on measure theory but is beyond the scope of this paper."

这是整个 MaxEnt RL 领域在混合空间上的**共同开放问题**，不是这篇论文独有的。

---

## 6. Beta 分布的理论优势

> 详细介绍见 [Beta 分布备选方案笔记](beta_distribution_note.md)

### 6.1 与 TanhNormal 的关键差异

> **术语说明**：TanhNormal（又名 Squashed Gaussian）= Gaussian + tanh squashing + Jacobian 修正。
> 这是同一个分布的不同叫法——前者是分布名，后者是实现描述。

| 性质 | TanhNormal | Beta(α,β) |
|------|:---------:|:---------:|
| 支撑集 | (-∞,+∞) → 压缩到 (-1,1) | 天然 [0,1]（可缩放） |
| 边界偏差 | 有（tanh 部分缓解） | **完全无偏** |
| 熵值范围 | (-∞, +∞) | 有界 |
| Jacobian 修正 | 需要 | 不需要 |
| 模态 | 单峰 | 单峰（α,β > 1 时） |

### 6.2 对 Hybrid SAC 的潜在改善

Beta 的熵有界性对混合场景有额外好处：
- 离散部分（Shannon 熵）和 Beta 连续部分的熵**尺度更接近**（都有界）
- 相比 Gaussian，使用 Beta 时**单一 α 的尺度失衡问题显著缓解**
- 但 Beta 不能解决**多峰分布**的表达力问题（单峰限制仍在）

---

## 7. Normalizing Flow 与策略表达力

> 参考论文：Ward et al. 2019 — *Improving Exploration in SAC with Normalizing Flows Policies*（ICML 2019 Workshop）

### 7.1 直觉：tanh 就是一层 Flow

标准 SAC 的采样链：`z₀ = μ + σε → a = tanh(z₀)`，其 log 概率为：

```
log π(a|s) = log p(z₀) - Σ log(1 - tanh²(z₀_i))
                          ↑ tanh 的 Jacobian 行列式
```

这与 Normalizing Flow 的换元公式 `log p(z') = log p(z₀) - log|det(∂f/∂z₀)|` 完全一致。**tanh squashing 本质上就是一层可逆变换（1-layer flow）。**

### 7.2 多层 Flow 的扩展

SAC+NF 在 Gaussian 采样与 tanh 之间插入多层 RealNVP 变换：

```
标准 SAC:     z₀ = μ + σε  →  tanh  →  a           （1 层 flow）
SAC + NF:     z₀ = μ + σε  →  f₁ → ... → fK  →  tanh  →  a  （K+1 层 flow）
```

log 概率逐层累加 Jacobian，仍然可精确计算：

```
log π(a|s) = log p(z₀) - Σ_{k=1..K} log|det(∂fk/∂z_{k-1})| - Σ log(1 - tanh²(zK_i))
             基础 Gaussian     中间 flow 层 Jacobian          最终 tanh 层 Jacobian
```

### 7.3 表达力层级

各种策略分布的表达力构成严格的包含关系：

```
Gaussian ⊂ Beta ⊂ NF(Gaussian base) ⊂ NF(Beta base)
   ↑         ↑           ↑                  ↑
 单峰椭圆   单峰有界    任意形状无界       任意形状有界
```

理论上，更大的策略类只会让 SAC 的最优解更好或持平（因为恒等变换是 NF 的特例）。

### 7.4 实际效果与局限

| 论文 | 环境 | 结果 |
|------|------|------|
| Ward et al. 2019 | 2D Grid World（稀疏奖励） | **NF 显著更好**（几乎立刻找到目标） |
| Ward et al. 2019 | 2D Grid World（稠密奖励） | NF ≈ Gaussian（两者都快速收敛） |
| Mazoure et al. 2019 | Roboschool（稠密奖励） | 初期有优势，训练久了优势**消失** |
| Delalleau et al. 2019 | Roboschool + Figure 3 | **几乎无改善**（KL 方向限制了多模态） |

**根本原因**（连接第 3.5 节）：

- NF 的核心能力 = 表达多峰/非对称分布
- SAC 的 KL(π \|\| target) = mode-seeking，压制多峰
- 两者形成矛盾：NF 有"超能力"但被 SAC 的优化目标**束缚**
- 只有在稀疏奖励（需要大范围有方向性的探索）时，NF 的分布变形能力才能绕过这个限制发挥价值

### 7.5 工程注意事项

NF 在 RL 中的数值稳定性不如在生成模型中（因为 RL 的梯度信号更嘈杂），Ward et al. 用了两个关键 trick：

1. **权重裁剪**（Weight Clipping）：限制 flow 层参数范围，防止可逆变换产生极端值（借鉴 WGAN）
2. **去掉 BatchNorm**：RealNVP 原版的 BatchNorm 在 RL 中导致 log σ 溢出，需删除

### 7.6 NF + Beta 的组合前景

| 组合 | 解决什么 | 不解决什么 | 实践验证 |
|------|---------|-----------|---------|
| TanhNormal（标准 SAC） | — | 边界偏差 + 单峰 | 充分 |
| Beta | 边界偏差 | 单峰 | 充分（ICML 2017） |
| NF + TanhNormal | 单峰限制 | 边界偏差 | 部分（仅小规模环境） |
| **NF + Beta base** | **边界偏差 + 单峰** | SAC 的 KL 方向仍压制多模态 | **无实验验证** |

理论上 NF + Beta 是最优策略类，但：
- 实现复杂度最高（Beta 重参数化 + Flow 层 + 稳定性 trick）
- 收益高度依赖任务特征（稀疏奖励 + 有界动作时收益最大）
- 在标准 benchmark 上的边际收益可能不值得额外复杂性

---

## 8. 实践建议总结

### 对本项目的指导

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| Phase 2-3（连续 SAC） | TanhNormal（标准方案） | 生态成熟，对照资料多 |
| Phase 4（Hybrid SAC） | 双 α + TanhNormal | 实践效果可靠，与论文一致 |
| 备选实验（短期） | Beta 替代 TanhNormal | 成本低，收益确定（消除边界偏差） |
| 备选实验（远期） | NF + Gaussian/Beta base | 成本高，收益依赖任务（稀疏奖励时值得） |
| 架构设计 | 分布模块可替换 | 保留切换 Gaussian/Beta/NF 的能力 |

### 理论风险与应对

| 风险 | 严重程度 | 应对 |
|------|---------|------|
| Hybrid SAC 无形式化收敛证明 | 低（各组件正确，实验有效） | 保留切换到单一 α 的能力 |
| 双温度探索失衡 | 中（论文观察到 collapse） | 监控两部分熵的比值 |
| 微分熵可为负 | 低（tanh squashing 限制了范围） | 使用 clamp 保护 log_α |
| mode-seeking 坍缩 | 低（熵正则化对抗） | 合理设置 target entropy |
| NF 数值不稳定 | 中（RL 梯度信号嘈杂） | 权重裁剪 + 去掉 BatchNorm |

---

## 相关文档

- [SAC 示例总览与核心概念](README.md) — 熵、α、Target Entropy 的实践指南
- [Beta 分布备选方案](beta_distribution_note.md) — 有界分布替代 Gaussian+tanh
- [概率分布模块设计](../../.doc/design/distributions_design.md) — Categorical/Normal/TanhNormal 的 API 设计
- [RL 路线图](../../.cursor/plans/) — Phase 1-4 的开发规划
- SAC+NF 论文原文：Ward et al., *Improving Exploration in Soft-Actor-Critic with Normalizing Flows Policies*（[arXiv:1906.02771](https://arxiv.org/abs/1906.02771)）
