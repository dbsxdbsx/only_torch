# MyZero — 统一 Model-Based RL 算法

> only_torch 的终极强化学习算法：learned-model MCTS，按环境内置已验收组件组合。  
> **战略 / 组件裁决 / backlog（唯一权威）**：[RL 状态总览](../../.doc/design/rl_myzero_status.md)  
> **环境配置**：[rl_python_env_setup.md](../../.doc/rl_python_env_setup.md)  
> **实测数字按环境分账**：[CartPole](cartpole/README.md) / [Gomoku](gomoku/README.md) / [MinAtar](minatar/README.md) / [Pong](pong/README.md)

本页只做**入口索引**与快速开始；组件矩阵、处方表、未完成事项一律见状态总览，勿在此维护副本。

## 设计理念

- **一个算法，持续迭代**：维护不断进化的 MyZero，不再为每篇论文建独立实现
- **用户零组件概念**：`MyZero::new(env_id)` 自动套用库内 recipe；promote 时只改 [`recipe.rs`](../../src/rl/algo/my_zero/recipe.rs)
- **生产路径**：learned-world-model；真规则仅 `cfg(test)` diagnostic

## 环境 × 状态

| 环境 | 动作类型 | 门禁 | 状态 |
|------|---------|------|------|
| [**CartPole-v1**](cartpole/README.md) | 离散（2） | greedy ≥ 475 | ✅ 回归哨兵（数字见[账本](cartpole/README.md)） |
| [**Pendulum-v1**](pendulum/README.md) | 连续（1） | return ≥ -200 | 诊断·已降级（[issue](../../.issue/items/pendulum_failure_diagnosis.md)） |
| [**ALE/Pong-v5**](pong/README.md) | 离散图像 | greedy ≥ −18 | ❌ 预算不足平直（[issue](../../.issue/items/my_zero_pong_image_flat_negative.md)） |
| [**MinAtar**](minatar/README.md) | 小网格图像 | Breakout ≥ 8 | ⚠️ 管线能学 peak 4.1（未达标） |
| [**Gomoku 9×9**](gomoku/README.md) | 棋盘离散 | vs random ≥0.95 | ✅ 已闭环（recipe=base）；naive0 墙见 [issue](../../.issue/items/gomoku_naive0_tactical_wall.md) |
| [**StochasticCartPole**](cartpole_stochastic/README.md) | 离散（2） | greedy ≥ 300 | ✅ K=8 3/3 vs K=1 2/3 |
| Platform-v0 | 混合 Tuple | — | 💨 smoke 通过，未做基准 |
| Schema toys | MultiDiscrete / Hybrid / … | loss 有限 | ✅ 接口纵切 |
| CartPole POMDP-lite | 离散（2） | greedy ≥ 400 | ❌ posterior 默认关（[issue](../../.issue/items/my_zero_pomdp_lite_posterior_negative.md)） |

组件裁决与处方表 → [状态总览 §3](../../.doc/design/rl_myzero_status.md#3-已验收环境与组件)。

## 快速开始

```bash
cargo run --example my_zero_cartpole --release
```

```rust
use only_torch::rl::algo::my_zero::MyZero;

let mz = MyZero::new("CartPole-v1")
    .solved(475.0)
    .max_episodes(2000)
    .save_model_when_eval("models/my_zero/CartPole-v1/best")
    .train()?;

let best = mz.train_report().and_then(|r| r.model_path.clone());
let mz = match best {
    Some(p) => mz.load_model_if_exists(p)?,
    None => mz,
};
mz.eval(10)?;
```

验收分层与改动纪律见 [状态总览 §4](../../.doc/design/rl_myzero_status.md#4-验收协议三层) 与 [`.github/instructions/rl.instructions.md`](../../.github/instructions/rl.instructions.md)。

## 训练与推理生命周期

| 时机 | 权重 | 说明 |
|------|------|------|
| `.train()` 返回 / 训后直接 eval | **latest** | 训末权重 |
| periodic greedy eval 创新高 | 写入 `{path}.otm` | 须 `.save_model_when_eval(path)` |
| `.load_model_if_exists(path)` 后 eval | **best** | 部署 / 演示用 |

`TrainReport`：`final_greedy` = latest；`best_greedy` / `model_path` = 训练期 periodic eval 最优（有落盘时）。

## 评判口径

| 维度 | 操作定义 |
|------|---------|
| **好** | greedy(temp=0) eval 达门槛（唯一成功判据） |
| **快** | env-steps-to-solved 为主（官方口径 **3-seed 中位 + 达标率**），wall-clock 为辅 |

CartPole 是 sanity/regression **哨兵**，不是组件价值证明台。

## 代码组织

- `src/rl/algo/my_zero/`：自包含 MyZero 库（**唯一** `*Zero` 实现）
- `recipe.rs`：按 `env_id` 内置组件开关
- `examples/my_zero/*/main.rs`：薄示例（env + 训练契约 + opt-in 落盘）
