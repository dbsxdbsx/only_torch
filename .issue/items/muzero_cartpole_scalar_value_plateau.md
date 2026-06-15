---
status: suspended
created: 2026-06-15
updated: 2026-06-15
---

# MuZero CartPole 标量 value 表示导致训练停在 ~40 分平台期，无法达到 195

## 背景

v0.23 主计划要求 MuZero 在 `CartPole-v0` 上达到 reward ≥ 195（与 SAC / PPO 同一架构跑通门槛）。

当前 MuZero 示例（`examples/muzero/cartpole/`）已实现 canonical MuZero 的骨架：

- representation / dynamics / prediction 三网络（隐藏层 128）
- MCTS-on-latent（复用 `src/rl/mcts/`，通过 `Dynamics` + `DynamicsModel` 桥接）
- K=5 步 unroll 训练 + n-step(50) value target
- 标量 value/reward + value transform `h(x) = sign(x)(sqrt(|x|+1)-1)+εx`
- 温度退火 + Dirichlet 根噪声 + MinMaxStats Q 归一化

## 现象 / 影响

修复了 8 个 MCTS / MuZero 逻辑 bug（见「已尝试」）后，训练**从完全不学习变为能学习，但卡在低位平台期**：

| 阶段 | avg_R（最近 100 局） | 说明 |
|------|---------------------|------|
| 逻辑 bug 修复前 | ~9.4（恒定） | 与随机策略无异，完全不学习 |
| 修复 6 个 SEVERE 后（v3） | ~28 | 开始学习，1000 局收敛到 ~28 |
| 再修 2 个 MODERATE 后（v4） | ~40（峰值 180+） | 继续学习，但 760 局仍卡 ~40 |

特征：avg_R 缓慢爬升但方差极大（单局在 9 ~ 180 之间剧烈跳动），呈现明显的**低位平台 + 高方差**形态。这是标量 value 回归噪声大的典型症状。

距离 195 门槛差距显著，继续延长训练或微调超参收益有限。

## 已尝试

### 第一轮：batch 训练 + 超参（让训练能跑起来）

- `train_batch` 从「每 position 独立 zero_grad+backward+step（实质 batch=1）」改为「一次 zero_grad + N 次 backward 梯度累积 + 一次 step」
- lr 3e-3 → 0.02、td_steps 10 → 50、batch_games → 8、trains_per_episode → 8

### 第二轮：6 个 MCTS / MuZero 逻辑 bug（Reviewer 审查发现）

| 级别 | 问题 | 位置 |
|------|------|------|
| FATAL | `terminal = reward < -0.5` 在 CartPole 永不触发（reward 恒 +1） | `model.rs` recurrent_impl |
| SEVERE | 首次 terminal backup 用 `rec_out.value` 而非 0 | `src/rl/mcts/search.rs` |
| SEVERE | `root_value` 漏算 `reward + discount * V(child)`，只取了子 subtree value | `main.rs` self_play |
| SEVERE | `MinMaxStats` 未初始化时返回 raw Q（压死 PUCT exploration） | `src/rl/mcts/min_max.rs` |
| SEVERE | 梯度缩放错误地乘在整个 step loss 上（含 prediction head） | `model.rs` train_unroll |
| FATAL/工程 | `value_transform_inv` 放大未训练网络噪声（±5 → ±35），毒害早期搜索 | `model.rs`（已加 clamp） |

### 第三轮：2 个 MODERATE

- 训练只采样 full unroll，不覆盖 episode 尾部 → 改为允许 short unroll，越界用 uniform policy + value 0
- `make_targets` / `recommend` 对 `visit_count=0` 用 `max(1)` → 改为真实 visit count，全 0 才 uniform fallback

**所有 99 个 RL 单元测试 + 11 个 MCTS 测试全绿**，`mcts_cartpole_env` 真环境 gate 测试用真实环境当 model 能跑到 90+ 步——证明 **MCTS 搜索核心本身正确**，瓶颈在 learned dynamics 的训练。

## 根因分析

相比 canonical MuZero（Schrittwieser et al., 2020），当前实现仍简化掉**两个**关键组件：

### 主因：Categorical value/reward 表示

- **现状**：标量回归 + `h(x)` 变换 + MSE loss
- **canonical**：把标量映射到固定 support（如 601 维，覆盖 [-300,300]）的 two-hot 编码，用**交叉熵**学习分布，取期望还原标量
- **为什么关键**：交叉熵对 value 的梯度信号远比 MSE 稳定、样本效率高。muzero-general / EfficientZero 即便在 CartPole 也用 categorical——它不是为 Atari 才需要的奢侈品，而是 value 学习稳定性的地基。当前的「低位平台 + 高方差」正是标量 MSE 回归噪声大的直接体现。

### 次因：Latent state 未归一化

- **现状**：representation / dynamics 的 latent 输出只过 `.relu()`，值域 [0, ∞)
- **canonical**：每步把 hidden state min-max 归一化到 [0,1]
- **为什么关键**：latent 不归一化会让其量级随训练漂移，dynamics 网络更难学到稳定的状态转移。

> 严格分离两者的贡献需要消融实验。但 categorical 是最可能的主导因素，latent 归一化是强次要候选。

## 当前卡点

标量 value 回归在 CartPole（value target 范围 0~150）下梯度信号不够稳定，即使配合 `h(x)` 变换仍无法把 value 学准到支撑 195 分所需的精度；MCTS backup 依赖的 value 估计噪声大，搜索质量受限。

## 暂缓原因

Categorical value/reward 与 latent 归一化都是 **EfficientZero V2（v0.24）的核心增量**，与 reanalyze / value prefix / SVE 一起实现更经济。在 v0.23 单独补 categorical 会提前把 v0.24 的工作量拉进来，且需要新增 support 编解码 + two-hot + 交叉熵的一整套基础设施。

v0.23 的实际价值已兑现：**MCTS 算法底座（search / backup / PUCT / MinMaxStats / Dynamics 桥接）经 8 个 bug 修复后已验证正确**，这对 AlphaZero 系列所有后续算法都是共享地基。MuZero 示例骨架完整、能学习，只是受标量 value 限制无法达标。

## 下次恢复条件

v0.24 EfficientZero V2 开工时。届时 categorical value/reward 应作为**首批基础设施**实现（先于 reanalyze / SVE），因为它是 value 学习栈的地基。

## 下一步建议

1. **v0.24 优先实现 categorical value/reward**：
   - 新增 support 配置（如 size=51，覆盖 CartPole value 范围足够，无需 601）
   - `scalar → two-hot(h(x))` 编码 + `categorical → scalar` 期望解码
   - prediction / dynamics 的 value/reward head 输出 logits，loss 改交叉熵
   - 入库 `src/rl/algo/muzero/`（与现有 `value_transform` 同模块，扩展为 `support.rs`）
2. **同期补 latent state 归一化**：representation / dynamics forward 末尾加 min-max scale 到 [0,1]（约 10 行）。
3. 两者落地后，先在 `CartPole-v0`（或 v1）复测达标，再推 EZ-V2 的 reanalyze / SVE 等更重的增量。

## 关联

- 主计划：[`rl_主线实施计划`](../../../../Users/Administrator/.cursor/plans/rl_主线实施计划_5966956a.plan.md) v0.23 / v0.24 章节
- 受影响代码：`examples/muzero/cartpole/{model.rs, main.rs}`、`src/rl/mcts/`、`src/rl/algo/muzero/`
