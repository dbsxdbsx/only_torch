# MyZero · CartPole-v1

> [← 返回 MyZero 总览](../README.md)｜组件裁决（✅/❌/⏳）见总览的「组件 × 环境 效果矩阵」

- **规格**：离散（2 动作）· 门禁 greedy eval ≥ 475 · 默认 `num_simulations=50`，`gamma=0.997`
- **状态**：✅ consistency + completedQ 已「又好又稳」（双 / 三 seed greedy 满分），转为**回归哨兵**。eval harness 已减半（20→10 局）降低 wall-clock，纯开销优化、不改算法。下一步主攻 Gumbel-root（在判别环境上验证）。

## 运行

```bash
# base 全关 完整训练（~5 分钟）
cargo run --example my_zero_cartpole --release

# +consistency（最优配置之一）
CONSISTENCY=1 cargo run --example my_zero_cartpole --release

# +consistency +completedQ（低 sims 仍满分，最快路径，~40 秒）
CONSISTENCY=1 CQ=1 SIMS=16 cargo run --example my_zero_cartpole --release

# 多 seed 稳定基线（seed 42/43/44 取中位数，可复现回归锚点）
SEEDS=3 CONSISTENCY=1 cargo run --example my_zero_cartpole --release

# SMOKE 管线验证（~30 秒）
SMOKE=1 cargo run --example my_zero_cartpole
```

训练产物默认写入 `checkpoints/cartpole/seed_{seed}/best`（权重 + manifest + meta）；仅当 periodic greedy eval 创新高时覆盖写盘。加载：

```bash
# 须与训练时相同的 env 契约
cargo run --example my_zero_cartpole --release
# 或在代码中：MyZero::new("CartPole-v1").solved(475.0).max_episodes(2000).load("checkpoints/cartpole/seed_42/best")?
```

## 关键超参（CartPole 默认）

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_simulations` | 50 | 每步 MCTS 模拟次数；按动作数与环境复杂度调整（completedQ 下可降至 16） |
| `gamma` | 0.997 | 折扣因子 |
| `k_unroll` | 5 | 训练期 K 步 dynamics 展开 |
| `td_steps` | 50 | n-step bootstrap 步数 |
| `lr` | 0.02 | 学习率 |
| `batch_games` | 8 | 每次训练采样的整局数 |
| `trains_per_episode` | 8 | 每个 episode 后的训练迭代次数 |

## 跨算法 Benchmark 基线（2026-06-16 实测）

| 算法 | greedy eval | env-steps | wall-clock | 备注 |
|------|-----------|-----------|------------|------|
| **PPO** | 484.6 | 81,920 | 107s | model-free，最快达标 |
| **SAC** | 487.0 | 104,982 | 186s | model-free |
| **MyZero base**（sim=50） | 500.0 | 17,260 | 287s | model-based，样本效率最高但 wall-clock 最慢 |

> **解读**：MyZero 用最少 env-step（17k vs PPO 82k）达满分，但 wall-clock 最慢——每个 env-step 含 50 次 MCTS 模拟（dynamics 推理），计算量 ~50× model-free。**样本效率 vs 计算效率是 model-based 的核心权衡**：数据昂贵 / 交互成本高的环境（机器人、真实世界）model-based 更有价值；CartPole 这种廉价模拟器上 model-free 的 wall-clock 占优。

## 消融快照（Ep250，sim=50，seed=42）

| 配置 | avg_R @ep250 | 最高单局 | loss | 观察 |
|------|-------------|---------|------|------|
| base（全关） | 80.3 | 182 | ~9.6 | 学习中，尚未达标 |
| **+consistency** | **97.1** | **366** | **~0.7** | loss 低一个数量级，学习显著更快 → ✅ 留下 |
| +consistency +value_prefix | 15.5 | 56 | ~0 振荡 | Ep659 avg_R 仍 ~13、greedy 9.4 → ❌ value_prefix 在 CartPole 有害 |

> **value_prefix 为什么在 CartPole 有害（关键洞见，别误删组件）**：CartPole 每步 reward 恒 +1，「累计 reward 前缀」退化成「步数计数器」，LSTM 画蛇添足。**但这不代表 value_prefix 是坏组件**——它正是 EfficientZero 在 Atari / 稀疏奖励 / 长 horizon 上拉开差距的核心。所以裁决是「**CartPole 删，但留待 Pendulum / 稀疏奖励环境重测**」，不是全局删除。这正是「判别环境原则」的活例。

## completedQ 目标实测（+consistency +completedQ，低 sims A/B，3 seed 中位数）

| 配置 | greedy（中位数） | env-steps to 475 | wall（中位数） | 结论 |
|------|:---:|:---:|:---:|------|
| visit-count @ sims=16 | 299.5 | **未达标** | 248s | ❌ 少模拟下 visit-count 学不动 |
| completedQ `c_scale=0.1` @ sims=16 | 138.5 | 未达标 | 242s | ❌ 过锐、噪声 Q 上过度自信 |
| **completedQ `c_scale=0.02` @ sims=16** | **500.0** | **5,141**（3/3 达标） | **39.6s** | ✅ 稳定达标 |
| 参照：visit-count @ sims=50 | 500.0 | ~16k | ~287s | 旧基线 |

> **结论**：completedQ（`c_scale=0.02`）让 sims 50→16 仍 **3/3 seed 满分**，env-step ~3× 更省、wall-clock ~7× 更快 → **经实证留下**。超参 `σ(q)=(c_visit+max_b N(b))·c_scale·q`；`c_scale` 偏锐（0.1）在 2 动作噪声 Q 上反而更差，需按环境调。**闭式 ≠ 无超参**（无需 Grill ¯π 二分搜索，但 `c_visit/c_scale` 仍是旋钮）。

## 多 seed 稳定基线（+consistency，sim=50）

| seed   | greedy eval | env-steps to 500 |
| ------ | ----------- | ---------------- |
| 42     | 500.0       | 18,159           |
| 43     | 500.0       | 13,684           |
| 中位数 | **500.0**   | **~16k**         |
