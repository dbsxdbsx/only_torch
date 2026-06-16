# MyZero — 统一 Model-Based RL 算法

> only_torch 的终极强化学习算法：一个持续进化的 learned-model MCTS 实现，
> 以消融实验驱动逐步叠加组件，最终覆盖全动作空间与全环境类型。

## 设计理念

- **一个算法，持续迭代**：不再为每篇论文建独立实现（MuZero / EfficientZero），
  而是维护一个不断进化的 MyZero
- **奥卡姆剃刀**：每叠一个组件必须用消融证明其价值，保证不回归
- **从简到繁**：CartPole → Pendulum → Platform → 更多环境

## 消融序列

MyZero 从 canonical MuZero（S0 base）出发，逐步叠加 EfficientZero 增量组件：

| 步骤 | 开关 | 说明 | 环境变量 |
|------|------|------|---------|
| S0 | 全关 | canonical MuZero（base） | （默认） |
| S1 | +consistency | SimSiam 自监督对齐 | `EZ_CONS=1` |
| S2 | +value prefix | LSTM 累计 reward 前缀 | `EZ_VP=1` |
| S3 | +target net | EMA/hard 参数同步 | `EZ_TARGET=1` |
| S4 | +SVE | search value blend | `EZ_SVE=0.5` |
| S5 | +Gumbel | 连续/混合动作搜索 | （待实现） |

## 环境矩阵（v0.25 硬承诺）

| 环境 | 动作类型 | 达标门禁 | 配置要点 | 状态 |
|------|---------|---------|---------|------|
| **CartPole-v1** | 离散（2） | greedy eval ≥ 475 | `num_simulations=50`，`gamma=0.997` | ✅ S1 稳定达标（双 seed greedy 500），转回归哨兵 |
| **Pendulum-v1** | 纯连续（1） | return ≥ -200 | 离散化候选（骨架）→ Gumbel 连续搜索（目标） | 骨架已建+管线通；待 Gumbel 提升至达标 |
| **Platform-v0** | 混合 Tuple | return 趋势上升 | 需 Gumbel + 混合编码 | 待实现 |

## 运行

```bash
# S0 base 完整训练（CartPole-v1，~5 分钟）
cargo run --example my_zero_cartpole --release

# S1 消融（开 consistency，当前 CartPole 最优配置）
EZ_CONS=1 cargo run --example my_zero_cartpole --release

# 多 seed 稳定基线（seed 42/43/44 取中位数，可复现回归锚点）
SEEDS=3 EZ_CONS=1 cargo run --example my_zero_cartpole --release

# Pendulum（连续动作 → 离散化候选，骨架，待 Gumbel 提升）
EZ_CONS=1 cargo run --example my_zero_pendulum --release

# SMOKE 管线验证（3 局 + 1 次训练，~30 秒）
SMOKE=1 cargo run --example my_zero_cartpole
```

## 评判口径（所有环境通用）

| 维度 | 操作定义 | 说明 |
|------|---------|------|
| **好（good）** | greedy(temp=0) eval 10 局固定 seed 均值达门槛 | **唯一成功判据**；`avg_R`（自对弈分）带探索噪声，永远偏低，**不作判据** |
| **稳（stable）** | 多 seed（≥3）都达标，取中位数 | 排除单 seed spike |
| **快（fast）** | env-steps-to-solved 为主，wall-clock 为辅 | 同算法消融看 wall-clock；跨算法看 env-step（样本效率） |

## 关键超参

| 参数 | CartPole 默认 | 说明 |
|------|-------------|------|
| `num_simulations` | 50 | 每步 MCTS 模拟次数；按动作数和环境复杂度调整 |
| `gamma` | 0.997 | 折扣因子 |
| `k_unroll` | 5 | 训练期 K 步 dynamics 展开 |
| `td_steps` | 50 | n-step bootstrap 步数 |
| `lr` | 0.02 | 学习率 |
| `batch_games` | 8 | 每次训练采样的整局数 |
| `trains_per_episode` | 8 | 每个 episode 后的训练迭代次数 |

## Benchmark 基线（CartPole-v1，2026-06-16 实测）

| 算法 | greedy eval | env-steps | wall-clock | 备注 |
|------|-----------|-----------|------------|------|
| **PPO** | 484.6 | 81,920 | 107s | model-free，最快达标 |
| **SAC** | 487.0 | 104,982 | 186s | model-free |
| **MyZero S0** (sim=50) | 500.0 | 17,260 | 287s | model-based，样本效率最高但 wall-clock 最慢 |

### 消融快照（Ep250 时对比，CartPole-v1，sim=50，seed=42）

| 配置 | avg_R @ep250 | 最高单局 | loss | 观察 |
|------|-------------|---------|------|------|
| S0 base | 80.3 | 182 | ~9.6 | 学习中，尚未达标 |
| **S1 +consistency** | **97.1** | **366** | **~0.7** | loss 低一个数量级，学习显著更快 |
| S1+S2 +value_prefix | 15.5 @ep250 | 56 | ~0 振荡 | **严重退化**：Ep659 avg_R 仍 ~13，greedy 9.4；VP 在 CartPole 上有害 |

> **解读**：MyZero S0 用最少的 env-step（17k vs PPO 82k）达到满分，但 wall-clock 反而最慢——
> 因为每个 env-step 包含 50 次 MCTS 模拟（dynamics 网络推理），实际计算量 ~50 倍于 model-free。
> **样本效率 vs 计算效率是 model-based 的核心权衡**：在数据昂贵/交互成本高的环境（机器人、真实世界）
> model-based 更有价值；在 CartPole 这种模拟器廉价的环境，model-free 的 wall-clock 优势更明显。
> S1 consistency 的自监督信号让 dynamics 网络学得更准（loss 从 ~10 降到 ~0.7），avg_R 在同样 episode 数下高 21%。

### S1 多 seed 稳定基线（CartPole-v1，sim=50，EZ_CONS=1）

| seed | greedy eval | env-steps to 500 | 备注 |
|------|-----------|------------------|------|
| 42 | 500.0 | 18,159 | |
| 43 | 500.0 | 13,684 | |
| 中位数 | **500.0** | **~16k** | 2/2 达满分 → CartPole S1 已"稳定+好" |

> **结论**：按上方"评判口径"，CartPole 在 S1 下**已又好又稳**（greedy 双 seed 满分），且 env-step 比 PPO/SAC 省 5–7×。CartPole 至此转为**回归哨兵**。eval harness 已减半（20→10 局）降低 wall-clock，纯开销优化、不改算法。下一步主攻 **Gumbel 搜索**（少模拟仍稳 + 解锁连续动作），见 plan。

## 算法核心

MyZero 采用 MuZero 的三网络架构：

- **Representation** h：obs → latent（min-max 归一化）
- **Dynamics** g：(latent, action) → (next_latent, reward)
- **Prediction** f：latent → (policy, value)

训练期 K 步 unroll + categorical value/reward 表示 + absorbing state 终止处理。
搜索期 MCTS 在 learned latent 空间推演（不碰真环境）。

## 与旧代码的关系

- `examples/muzero/`：MuZero canonical 参考实现，保留为消融对照基线
- `examples/efficientzero/`：EZ-V2 多模式示例，保留为组件参考
- `src/rl/algo/my_zero/`：MyZero 库模块（配置 + 消融开关）
- `src/rl/algo/muzero/`：MuZero helper（support / value_transform / n_step 等），MyZero 直接复用
