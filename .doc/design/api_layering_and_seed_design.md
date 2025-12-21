# API 分层与种子管理设计

> 创建日期: 2025-12-21
> 状态: **草案**
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
| Graph-Level | `Graph::set_seed(seed)` | 训练脚本、NEAT | ⏳ 待实现 |
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
- `new_parameter_node()` - 参数初始化
- 未来可能：Dropout、数据增强等

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

### 阶段 2：Graph 级别种子（建议在 M4 前完成）
- [ ] 为 `Graph` 添加 `rng: Option<StdRng>` 字段
- [ ] 实现 `Graph::new_with_seed(seed)`
- [ ] 实现 `Graph::set_seed(seed)`
- [ ] 修改 `new_parameter_node()` 使用 Graph 的 RNG（如有）
- [ ] 更新集成测试使用 Graph 级别种子（简化代码）

### 阶段 3：High-Level API（远期）
- 在 `nn::Module` 或类似抽象中自动处理种子
- 可能提供配置文件/环境变量方式设置

---

## 5. 设计决策记录

| 决策 | 选择 | 原因 |
|-----|------|------|
| 全局种子 vs Graph 种子 | Graph 种子 | NEAT 需要多图并行 |
| 默认行为（无种子） | 非确定性 | 符合用户预期，生产环境常态 |
| Granular API 保留 | 是 | 测试需要、向后兼容 |
| RNG 类型 | `StdRng` | 跨平台一致、可序列化 |

---

## 6. 开放问题

1. **种子是否需要序列化？** - NEAT 保存/加载进化状态时可能需要
2. **是否支持 `Graph::fork()`？** - 从当前 RNG 状态分叉，用于进化分支
3. **Tensor 层面是否也需要 RNG 上下文？** - 目前 `Tensor::normal()` 独立于 Graph

---

## 7. 参考

- PyTorch: `torch.manual_seed()`, `torch.Generator`
- JAX: 显式 PRNGKey 传递（函数式风格）
- TensorFlow: `tf.random.set_seed()` + `tf.random.Generator`

