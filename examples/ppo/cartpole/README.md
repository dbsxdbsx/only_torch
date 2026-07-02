# PPO · CartPole-v1

> on-policy model-free 基线（与 SAC 同为 MyZero 的对照算法，见 [算法纲领 §3 双轨架构](../../../.doc/design/my_zero_algorithm_vision.md#3-双轨架构已定)）。
> 跨算法样本效率对照表见 [CartPole 基准账本](../../my_zero/cartpole/README.md)。

## 运行

```bash
cargo run --example ppo_cartpole --release

# 管线验证（2 个 update，不验收敛）
SMOKE=1 cargo run --example ppo_cartpole

# 多 seed 基线重测协议（默认 42）
SEED=43 cargo run --example ppo_cartpole --release

# 限制 update 数（默认 600）
MAX_UPD=100 cargo run --example ppo_cartpole --release
```

门禁：greedy(temp=0) eval 20 局（固定 seed）均值 ≥ **475**（Gymnasium CartPole-v1 官方 solved）。

## 基线（v0.25 收口 · release + MKL · seed 42/43/44）

| seed | 到 475 env-steps | 最终 Eval |
|------|------------------|-----------|
| 42 | 81,920 | 487.1 |
| 43 | 81,920 | 479.4 |
| 44 | 102,400 | 484.2 |
| **中位** | **81,920**（3/3 达标） | — |

PPO 按 2048-step rollout 为单位训练，env-steps 呈 2048 的整数倍粒度；eval 每 10 个 update 触发一次，因此达标点也是粗粒度的。

## 算法组成

- 入库 helper（`src/rl/algo/ppo/`）：`compute_gae`（GAE-λ，正确处理 `terminated` / `truncated` 边界 bootstrap）、`clipped_policy_loss`、`value_loss`、`entropy_bonus`、`normalize_advantages`
- 示例侧（本目录）：Actor-Critic 网络（`model.rs`）、rollout 采集与 minibatch 训练循环（`main.rs`）
- 超参：`n_steps=2048` · `lr=3e-4` · `γ=0.99` · `λ=0.95` · `clip=0.2` · `ent_coef=0.01` · `vf_coef=0.5` · 4 epochs × minibatch 64

## 实现注意点

- **truncated 边界**：rollout 采集时对每步存真实 `next_value`（`terminated` 为 0，其余用 critic 评估后继状态），GAE 才能正确 bootstrap 撞 500 步上限的截断局。
- **greedy eval 用独立 env**：PPO rollout 跨 update 连续，若用训练 env 做 eval 会打断 obs 连续性。
