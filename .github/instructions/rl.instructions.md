---
applyTo: "{src/rl/**,examples/traditional/sac/**,tests/python/gym/**}"
description: "Use when editing reinforcement learning code, Gym/Gymnasium integration, or the PyO3 bridge in only_torch."
---

# RL Instructions

- 当前 RL 环境以 Python Gym / Gymnasium 为主，Rust 侧通过 PyO3 提供桥接接口；不要默认改成纯 Rust 环境实现。
- 优先保持环境交互层与训练逻辑解耦，避免把 Python 侧细节散落到算法代码。
- 改动前先看 [RL 路线图](../../.doc/design/rl_roadmap.md) 与 [Python 环境配置](../../.doc/rl_python_env_setup.md)。
- 遇到运行或测试异常时，先检查 Python 环境、gymnasium 安装、解释器版本，以及导入竞态问题。
- RL / PyO3 相关测试优先单线程运行；排查时先用小范围示例或串行测试。
- 除非用户明确要求，不要默认运行耗时的 RL 训练示例。
- 若改到策略分布、采样或 log_prob，顺手核对 `src/nn/distributions/` 下实现的数值稳定性。
