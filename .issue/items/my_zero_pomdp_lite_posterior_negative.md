---
status: open
created: 2026-07-14
updated: 2026-07-14
owners: []
reviewers: []
---

# MyZero · recurrent posterior (POMDP-lite)：velocity-masked CartPole 对照未达标

> **状态**：**open（负结果记录）**——代码保留、recipe 默认关；架构未否定，归因为超参/容量/预算。
> **代码**：`src/rl/algo/my_zero/network.rs`（`PosteriorEncoder`）·
> `runner.rs`（self-play / eval hidden state）·
> `config.rs` / `builder.rs`（`recurrent_posterior` / `obs_mask`）·
> `tests/posterior.rs`（5 单测全绿）·
> `tests/pomdp_cartpole_bench.rs`（手动 `--ignored` bench）
> **关联**：[RL 路线图 §5](../../.doc/design/rl_roadmap.md) ·
> [world model 地基 §7.2](../../.doc/design/my_zero_world_model_foundation.md) ·
> [大纲阶段 4](../../.doc/design/rl_closure_plan.md)

---

## 一、设计目标

路线图阶段 4「recurrent posterior / POMDP-lite」的验证门槛：

> 先用 observation-aliasing toy 验证**单帧失败、history posterior 成功**，
> 再进入真实图像 POMDP。

选择 **CartPole-v1 velocity-masked**（遮蔽索引 1, 3 即 cart_velocity 与
pole_velocity）作为 observation-aliasing toy：单帧只看位置无法推导速度，
posterior 理论上可从历史序列恢复速度信息。

## 二、工程实现（已完成，全绿）

| 步骤 | 内容 | commit |
|------|------|--------|
| S0-S1 | 记忆单元 `step()` + `recurrent_posterior` 开关 | `a9a86f9` |
| S2-S3 | `PosteriorEncoder`（GRU）+ `LatentState` hidden 扩展 | `955084c` |
| S4-S7 | 训练 burn-in（无反传暖 hidden）+ self-play hidden 管理 + `PrecomputedRootDynamics` | `f8edca8` |
| 验证环境 | `obs_mask` + velocity-masked CartPole | `3c7f656` |
| bugfix | greedy eval 路径补 posterior hidden state 管理 | `6cb94e3` |

单元测试 5 个（`posterior.rs`）：OFF 等价、ON 梯度流、推理链路、
PrecomputedRoot 一致性、masked+posterior 梯度——全绿。

## 三、实验结果（2026-07-14，3-seed，3000 episodes）

| 组 | posterior | 中位 greedy | 达标率 | 判定 |
|----|-----------|-----------|--------|------|
| B（负对照） | **OFF** + masked | ~39.7 | 0/3 | 符合预期（单帧应失败） |
| C（正组） | **ON** + masked | ~45.4 | 0/3 | **未达标** |

门槛：正组 greedy ≥ 400（3/3），负组 ≤ 200 或 0/3。

**结果**：正组仅比负组高约 14%，远未达到 400 门槛。cold-start 现象：
posterior=ON 前 ~1000 episodes greedy 钉在 ~9.1（GRU 暖隐+探索成本），
后段追上但天花板未拉开。

## 四、归因分析

1. **容量不足**：posterior GRU hidden_size = latent_dim = 32；CartPole 默认
   网络容量本就偏小，masked obs 复杂度更高。
2. **预算不足**：3000 episodes 对 POMDP 可能太少——普通 CartPole 需 ~200
   episodes 收敛，masked 变体难度数量级增加。
3. **burn-in 长度**：默认 burn_in_steps=10，对 CartPole ~200 步的局可能不够。
4. **探索 cold-start**：posterior 开启后早期 hidden = 零向量、latent 信息差，
   MCTS 搜索质量在前 1000 episodes 显著劣化。

**不归因于架构错误**：GRU posterior + burn-in 是 MuZero/Dreamer 族的标准
POMDP 方案，单元测试证明梯度流正确、推理链路通畅。

## 五、裁决

- **不 promote**：recipe 默认保持 `recurrent_posterior = false`。
- **代码保留**：完整 POMDP-lite 骨架（posterior、burn-in、obs_mask、hidden state
  管理）保留在库中，作为后续调参/扩容的基础设施。
- **不否定架构**：当前实验否定的是「默认超参 + 3000-episode 预算下
  recurrent posterior 对 CartPole masked 有效」，不否定 POMDP-lite 架构本身。

## 六、复活条件

以下任一条件变化时可重新预注册 A/B 消融：

1. **扩容**：latent_dim / hidden_size 翻倍或更大网络配置。
2. **加预算**：max_episodes 10000+ 或配合图像环境的长预算方案。
3. **burn-in 调优**：burn_in_steps 20+、sequence_length 扩大。
4. **native POMDP 环境**：真实部分可观测环境（而非人工遮蔽）可能有不同特征。
5. **posterior 参与 unroll**：当前 unroll K 步仍走 dynamics prior，
   posterior 只在根节点暖——训练期 posterior 全程参与可能改善梯度。

复活时须按一次一项纪律 + 3-seed 统计口径预注册。
