# API 分层与种子管理设计

> 创建日期: 2025-12-21
> 最后更新: 2026-04-19
> 状态: **已实现** (Graph 级别种子 + 严格确定性保证)
> 关联: [architecture_roadmap.md](../architecture_roadmap.md), [optimizer_architecture_design.md](./optimizer_architecture_design.md)

## 1. 背景与动机

### 1.1 问题陈述

随着项目发展，我们需要回答一个核心问题：**如何设计 API 层次，使其既对新用户友好，又为高级用户和 NEAT 进化算法保留灵活性？**

种子管理（Seed Management）是这个问题的一个典型切入点：
- 测试需要可重复性
- 用户希望简单设置一次
- NEAT 需要多图并行独立运行

### 1.2 架构约束（来自 NEAT 需求）

```
⚠️ 核心约束：避免全局状态
   原因：NEAT 算法可能同时进化多个 Graph 实例
```

这意味着 PyTorch 风格的 `torch.manual_seed()` 全局种子方案**不适用**。

---

## 2. API 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│  High-Level API (远期)                                       │
│  ─────────────────                                          │
│  • nn::Sequential, nn::Module 风格                          │
│  • 自动管理种子、设备、模式                                    │
│  • 目标用户：快速原型、教学                                    │
├─────────────────────────────────────────────────────────────┤
│  Graph-Level API (中期目标) ← 推荐的默认层                    │
│  ─────────────────────────                                  │
│  • Graph::new_with_seed(seed) / graph.set_seed(seed)        │
│  • 每个 Graph 独立 RNG 状态                                  │
│  • 目标用户：标准训练流程、NEAT                               │
├─────────────────────────────────────────────────────────────┤
│  Granular API (当前) ← 底层工具                              │
│  ─────────────────                                          │
│  • Tensor::normal_seeded(), shuffle_mut_seeded()            │
│  • new_parameter_node_seeded()                              │
│  • 显式控制每个随机操作                                       │
│  • 目标用户：单元测试、库开发者、精确调试                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 各层职责

| 层级 | 种子管理 | 适用场景 | 状态 |
|-----|---------|---------|------|
| Granular | `fn_seeded(..., seed)` | 单元测试、精确控制 | ✅ 已实现 |
| Graph-Level | `Graph::new_with_seed(seed)` / `Graph::set_seed(seed)` | 训练脚本、NEAT | ✅ 已实现 |
| High-Level | 自动/配置化 | 快速原型 | 📋 远期 |

---

## 3. Graph 级别种子设计（中期目标）

### 3.1 设计方案

```rust
// 方案 A：构造时设置（推荐）
let graph = Graph::new();           // 随机种子（非确定性）
let graph = Graph::new_with_seed(42); // 固定种子（确定性）

// 方案 B：运行时设置
let mut graph = Graph::new();
graph.set_seed(42);  // 重置 RNG 状态

// 内部实现
pub struct Graph {
    // ... existing fields ...
    rng: Option<StdRng>,  // None = 使用 thread_rng()
}
```

### 3.2 影响范围

当 Graph 有种子时，以下操作使用 Graph 的 RNG：
- `graph.parameter()` — 参数初始化（`Init::generate_with_rng`）
- `graph.randn()` — 随机张量（`Tensor::normal_with_rng`）
- `var.dropout(p)` — Dropout 种子（`GraphInner::next_seed()`）
- `var.rand_like()` / `var.randn_like()` — 随机噪声（Graph RNG）
- `Normal::rsample()` — 重参数化采样的 ε 噪声（Graph RNG）
- `Categorical::sample()` — 离散采样（`multinomial_with_rng`）
- 演化系统 `descriptor_rebuild` 中的 Dropout 重建同样使用 `next_seed()`

### 3.3 与 Granular API 的关系

```rust
// Granular API 仍然可用，优先级更高
let graph = Graph::new_with_seed(42);

// 使用 Graph 的 RNG
let w = graph.new_parameter_node(&[3, 1], Some("w"))?;

// 显式覆盖，使用指定种子（Granular API）
let b = graph.new_parameter_node_seeded(&[1, 1], Some("b"), 999)?;
```

### 3.4 NEAT 兼容性验证

```rust
// 多个 Graph 并行进化，互不干扰
let graphs: Vec<Graph> = (0..100)
    .map(|i| Graph::new_with_seed(i as u64))
    .collect();

// 每个 graph 独立初始化，结果可重复
for graph in &mut graphs {
    let param = graph.new_parameter_node(&[10, 10], None)?;
    // 相同种子的 graph 产生相同的参数值
}
```

---

## 4. 实现路径

### 阶段 1：当前状态 ✅
- Granular API (`_seeded` 方法) 已实现
- 集成测试使用显式种子，结果可重复

### 阶段 2：Graph 级别种子 ✅ (2025-12-21 完成)
- [x] 为 `Graph` 添加 `rng: Option<StdRng>` 字段
- [x] 实现 `Graph::new_with_seed(seed)`
- [x] 实现 `Graph::set_seed(seed)`
- [x] 实现 `Graph::with_name_and_seed(name, seed)`
- [x] 实现 `Graph::has_seed()` 检查方法
- [x] 修改 `new_parameter_node()` 使用 Graph 的 RNG（如有）
- [x] 8 个单元测试验证功能

### 阶段 2.5：严格确定性保证 ✅ (2026-04-19 完成)

核心原则：**默认行为不变（不指定 seed = 随机），但一旦指定 seed，所有随机操作 100% 确定性。**

**传统模式修复：**
- [x] `Var::dropout()` — 改用 `GraphInner::next_seed()` 替代 `SystemTime`
- [x] `Graph::randn()` — 改用 `Tensor::normal_with_rng()` 替代 `Tensor::normal()`（thread_rng）
- [x] `Graph` handle 添加 `set_seed()` / `has_seed()` 代理方法
- [x] `descriptor_rebuild` 中 Dropout 重建改用 `next_seed()` 替代固定 seed 42

**演化模式修复：**
- [x] `population_size` / `offspring_batch_size` — 指定 seed 时自动固定为常量（20/12），避免因线程数差异导致 RNG 序列不同
- [x] `rebuild_pareto_member()` — 使用保存的 `evolution_seed` 而非 `from_entropy()`
- [x] `EvolutionResult` 新增 `evolution_seed: Option<u64>` 字段

**测试：**
- [x] `test_seeded_graph_dropout_deterministic` — 验证 seeded Graph 下 `.dropout()` 可复现
- [x] `test_different_graph_seed_dropout_differs` — 不同 seed 产生不同 mask
- [x] `test_seeded_graph_multiple_dropouts_deterministic` — 多 Dropout 互不干扰
- [x] `test_seeded_graph_randn_deterministic` — 验证 seeded Graph 下 `randn` 可复现
- [x] `test_different_seed_randn_differs` — 不同 seed 产生不同值
- [x] `test_set_seed_proxy` — `set_seed()` 等价于 `new_with_seed()`
- [x] `test_seeded_graph_full_pipeline_deterministic` — 完整 pipeline（parameter + randn + dropout）可复现

### 阶段 3：High-Level API（远期）
- 在 `nn::Module` 或类似抽象中自动处理种子
- 可能提供配置文件/环境变量方式设置

---

## 5. 设计决策记录

| 决策 | 选择 | 原因 |
|-----|------|------|
| 全局种子 vs Graph 种子 | Graph 种子 | NEAT 需要多图并行 |
| 默认行为（无种子） | 非确定性 | 符合用户预期，生产环境常态 |
| 指定 seed 后的保证 | 严格 100% 确定性 | 可复现性对调试和科研至关重要 |
| Granular API 保留 | 是 | 测试需要、向后兼容 |
| RNG 类型 | `StdRng` | 跨平台一致、可序列化 |
| 演化 seed 后固定种群参数 | 是（population_size=20, offspring_batch_size=12） | 消除线程数差异导致的不确定性 |

---

## 6. 开放问题

1. **种子是否需要序列化？** - NEAT 保存/加载进化状态时可能需要
2. **是否支持 `Graph::fork()`？** - 从当前 RNG 状态分叉，用于进化分支
3. ~~**Tensor 层面是否也需要 RNG 上下文？**~~ — ✅ 已解决：`Graph::randn()` 有 seed 时使用 `normal_with_rng`
4. **Data transforms / DataLoader 的 RNG** — `data/transforms/*` 中的随机增强目前使用 `thread_rng`，尚未纳入 Graph RNG。DataLoader 可通过 `.seed()` 控制 shuffle 顺序

---

## 7. 参考

- PyTorch: `torch.manual_seed()`, `torch.Generator`
- JAX: 显式 PRNGKey 传递（函数式风格）
- TensorFlow: `tf.random.set_seed()` + `tf.random.Generator`

