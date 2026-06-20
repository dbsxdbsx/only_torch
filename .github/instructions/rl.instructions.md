---
applyTo: "{src/rl/**,examples/my_zero/**,examples/sac/**,tests/python/gym/**}"
description: "Use when editing reinforcement learning code, Gym/Gymnasium integration, or the PyO3 bridge in only_torch."
---

# RL Instructions

- 在线 RL 环境 **仅 Gymnasium**（`gymnasium.make`）；禁止 `GymEnv` 回退 OpenAI Gym。混合动作 benchmark：**`Platform-v0`**（`pip install hybrid-platform`，`import gym_platform`），**不用** gym-hybrid / Moving-v0。离线数据用 Minari。
- 优先保持环境交互层与训练逻辑解耦，避免把 Python 侧细节散落到算法代码。
- 改动前先看 [RL 路线图 §8 MyZero](../../.doc/design/rl_roadmap.md#8-v025-myzero-统一算法2026-06-16-方向定稿)、[MyZero 算法纲领](../../.doc/design/my_zero_algorithm_vision.md) 与 [Python 环境配置](../../.doc/rl_python_env_setup.md)。
- **验收分层**：SAC / MyZero / PPO 架构跑通用 **`CartPole-v1` 且 greedy(temp=0) eval 均值 ≥ 475**（Gymnasium 官方 solved；**2026-06-16 全项目 CartPole 统一 v1、废弃 v0**——v0 在新 Gymnasium 仅 DeprecationWarning）；**MyZero**（项目**唯一**的 `*Zero` 实现，学术上承接 [EfficientZero V2](https://arxiv.org/abs/2403.00564) 谱系）为终极调优算法，全部示例环境用 **`-v1` / 新版 ID**，侧重样本效率（env-steps）（见 [RL 路线图](../../.doc/design/rl_roadmap.md)）。
- **MyZero 改动纪律（搬运 ≠ 改进）**：
  - **搬运**（挪纯函数、改 import、折 config、删旧模块）：行为零变化 → 可批量；须 **CartPole-v1 greedy eval ≥ 475** 证明无回归。
  - **改进**（修 stop-grad / SVE 自适应 / value 坍缩等）：改行为 → **一次一项**，单独 A/B 消融，单独过 CartPole 哨兵；**不得与搬运混进同一改动**。
  - 战略裁决见 [MyZero 算法纲领](../../.doc/design/my_zero_algorithm_vision.md)；组件×环境实测见 [MyZero 示例总览](../../examples/my_zero/README.md)。
- 遇到运行或测试异常时，先检查 Python 环境、gymnasium 安装、解释器版本，以及导入竞态问题。
- RL / PyO3 相关测试优先单线程运行；排查时先用小范围示例或串行测试。
- 除非用户明确要求，不要默认运行耗时的 RL 训练示例。
- 若改到策略分布、采样或 log_prob，顺手核对 `src/nn/distributions/` 下实现的数值稳定性。
