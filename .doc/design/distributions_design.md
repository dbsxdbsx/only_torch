# 概率分布模块设计

> 模块路径：`src/nn/distributions/`

## 概述

概率分布模块提供 Categorical、Normal、TanhNormal 三种分布的计算图内实现，用于 SAC-Discrete / SAC-Continuous / Hybrid SAC 等强化学习算法。

## 核心设计原则

### 1. 需要梯度 → Var，不需要梯度 → Tensor

这是最根本的判断标准：

| 情况 | 返回类型 | 原因 |
|------|----------|------|
| 参与 loss 计算、需要反向传播 | `Var` | 在计算图中，梯度可追踪 |
| 采样（不可微） | `Tensor` | 采样过程断开梯度 |
| 重参数化采样（可微） | `Var` | 重参数化技巧保留梯度路径 |

**唯一例外**：`sample()`（非重参数化采样）返回 `Tensor`，因为离散采样不可微。`rsample()`（重参数化采样）返回 `Var`，因为 `mean + std * ε` 对 mean 和 std 可微。

### 2. 构造时预计算并缓存共享的 Var 中间值

分布的多个方法经常共享中间计算。为避免冗余图节点，在 `new()` 构造时就创建并缓存这些共享节点：

| 分布 | 缓存内容 | 被哪些方法共享 |
|------|----------|--------------|
| **Categorical** | `probs`（softmax）、`log_probs`（log_softmax） | entropy、log_prob、sample、用户的 probs() |
| **Normal** | `log_std`（ln(σ)） | entropy、log_prob |
| **TanhNormal** | 继承 Normal 的缓存 | — |

这保证了同一 logits 上最多只有 2 个图节点（softmax + log_softmax），与手写公式的拓扑一致。

### 3. 返回 Var（clone），不返回 &Var

Var 内部是 `Rc<NodeInner>`，clone 只增加引用计数（纳秒级），但返回 `Var` 比 `&Var` 更简洁：
- 用户无需关心生命周期
- 所有方法签名统一
- `&Var` 和 `Var`（clone）指向同一个图节点，对梯度追踪零影响

## 分布 API 一览

### Categorical（离散分类分布）

```rust
let dist = Categorical::new(logits); // logits: Var [batch, num_classes]

// Var 级（计算图内，可反向传播）
dist.probs()              // Var [batch, num_classes] — 缓存的 softmax
dist.log_probs()          // Var [batch, num_classes] — 缓存的 log_softmax
dist.logits()             // Var [batch, num_classes] — 原始 logits
dist.entropy()            // Var [batch, 1] — H = -Σ p log p
dist.log_prob(&action)    // Var [batch, 1] — 指定动作的 log 概率

// Tensor 级（不参与梯度）
dist.sample()             // Tensor [batch, 1] — 按概率采样索引
```

### Normal（正态分布）

```rust
let dist = Normal::new(mean, std); // mean: Var, std: Var

// Var 级
dist.mean()               // Var — 均值
dist.std()                // Var — 标准差
dist.log_std()            // Var — 缓存的 ln(σ)
dist.rsample()            // Var — 重参数化采样 μ + σε
dist.log_prob(&value)     // Var — 对数概率密度
dist.entropy()            // Var — 分布熵
```

### TanhNormal（Squashed Gaussian）

```rust
let dist = TanhNormal::new(mean, std);

// Var 级
dist.mean()                     // Var — 均值
dist.std()                      // Var — 标准差
dist.rsample()                  // (Var, Var) — (tanh(u), u)
dist.log_prob(&raw_action)      // Var — 带 Jacobian 修正的 log prob
dist.rsample_and_log_prob()     // (Var, Var) — 采样+log_prob 一步完成
```

## 与 PyTorch 的对比

| 方面 | PyTorch | only_torch |
|------|---------|------------|
| Tensor/Var 区分 | 不区分（Tensor 天然带 autograd） | 显式区分 Tensor（纯数据）和 Var（图节点） |
| 缓存策略 | `__init__` 中计算并缓存 probs/logits | `new()` 中计算并缓存 probs/log_probs/log_std |
| sample 返回值 | 返回 detached Tensor | 返回 Tensor（天然不在图中） |
| rsample 返回值 | 返回 Tensor（带 grad） | 返回 Var（在计算图中） |

## 参考

- PyTorch: `torch.distributions.Categorical`, `torch.distributions.Normal`
- Haarnoja et al. 2018 Appendix C（TanhNormal 的 Jacobian 修正）
- [RL 路线图](../../.cursor/plans/) Phase 2 详细规划
