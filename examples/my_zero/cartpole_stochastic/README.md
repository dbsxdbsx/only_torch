# MyZero · Stochastic CartPole

Stochastic MuZero 验证环境：在 CartPole-v1 基础上注入随机力扰动，
模拟 agent 无法控制的环境随机性（chance events）。

## 环境

`StochasticCartPole`（`stochastic_cartpole.py`）：每步物理模拟后，
对 cart velocity 和 pole angular velocity 各添加 `U(-noise_scale, +noise_scale)` 噪声。
默认 `noise_scale=0.05`。

## 运行

```bash
cargo run --example my_zero_cartpole_stochastic --release
SMOKE=1 cargo run --example my_zero_cartpole_stochastic   # 管线验证
```

## 目的

验证 Stochastic MuZero chance node 机制在随机环境下的表现：
- K=1（默认）= 确定性快路径，不插入 chance 节点
- K=8 = stochastic 分支，dynamics 产生 chance logits + MCTS chance 节点

预期：K=8 在随机环境下应优于 K=1（chance model 捕获环境随机性）。
