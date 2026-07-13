# 强化学习路线图（当前态）

> **定位**：RL 模块的**当前状态 + 验收协议 + 下一版方向**。战略层（为什么选这条路）见 [MyZero 算法纲领](./my_zero_algorithm_vision.md)；实测数字按环境分账：[CartPole](../../examples/my_zero/cartpole/README.md) / [Gomoku](../../examples/my_zero/gomoku/README.md)。
> v0.20–v0.24「每版一个算法」时代的设计决策与实施计划已整体归档：[rl_roadmap_v020_v024.md](./_archive/rl_roadmap_v020_v024.md)（MCTS 接缝设计、能力边界矩阵、SAC 技术笔记等长期有效内容也在归档 §2.5 / §2.2.1b / §4，按需链回，不重复维护）。
>
> **创建日期**：2026-02-14 · **本版重写**：2026-07-02（v0.25 收口）

---

## 1. 文档分工（先读这张表）

| 文档 | 回答什么 |
|------|----------|
| **本文** | 当前模块状态、验收协议、v0.25 结果、v0.26 方向 |
| [my_zero_algorithm_vision.md](./my_zero_algorithm_vision.md) | 算法哲学、文献谱系、做/不做决策、战略目标 |
| [my_zero_world_model_foundation.md](./my_zero_world_model_foundation.md) | 通用 learned world model 前两阶段实现、契约与后续接缝 |
| [examples/my_zero/README.md](../../examples/my_zero/README.md) | 组件×环境实测矩阵、命令、门禁 |
| [examples/my_zero/cartpole/README.md](../../examples/my_zero/cartpole/README.md) | CartPole benchmark 数字 owner |
| [examples/my_zero/gomoku/README.md](../../examples/my_zero/gomoku/README.md) | Gomoku / 棋盘 benchmark 数字 owner |
| [rl_python_env_setup.md](../rl_python_env_setup.md) | Python / Gymnasium 环境搭建 |
| [rl.instructions.md](../../.github/instructions/rl.instructions.md) | 改 RL 代码时的 agent 约束 |
| [_archive/rl_roadmap_v020_v024.md](./_archive/rl_roadmap_v020_v024.md) | 历史：v0.20–v0.24 设计决策与实施计划 |

---

## 2. 当前状态（v0.25 收口，2026-07-02）

### 2.1 双轨架构

| 轨道 | 位置 | 状态 |
|------|------|------|
| **MyZero**（主线，项目唯一 `*Zero` 实现） | `src/rl/algo/my_zero/`（自包含：模型 + 训练循环 + MCTS 接入 + recipe） | CartPole ✅ 哨兵；Pendulum ⏳ 诊断中 |
| **SAC / PPO**（model-free 基线） | 纯函数 helper 入库 `src/rl/algo/{sac,ppo}/`；Agent 与训练循环留在 `examples/{sac,ppo}/` | SAC 四环境 + PPO CartPole 示例可跑 |

支撑层：`src/rl/env/`（`GymEnv` Gymnasium-only + `MinariDataset`）、`src/rl/buffer/`（`Transition` / `ReplayBuffer` / `RolloutBuffer` / `SelfPlayGame`）、`src/rl/mcts/`（搜索内核：PUCT / Gumbel / Sampled，吃 model 不吃 env——接缝设计见[归档 §2.5](./_archive/rl_roadmap_v020_v024.md#25-mcts-抽象边界内核吃-model不吃-env选择规则可插拔v022-定形)）。

### 2.2 MyZero 组件裁决现状

组件由 [`recipe.rs`](../../src/rl/algo/my_zero/recipe.rs) 按 env 注入，用户 API 不暴露开关。

| 状态 | 组件 |
|------|------|
| ✅ CartPole promoted | consistency(coef 2) + reconstruction(**coef 16**，v0.26 P0 重标定) + Sampled（PUCT · sims=20 · td=5 · continuation 二值门） |
| ❌ CartPole 负结果（代码保留、recipe 关） | completedQ / Gumbel-root（[issue](../../.issue/_archive/my_zero_gumbel_completedq_cartpole_negative.md)）、reanalyze 写回（[issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md)）、value_prefix |
| ⏳ 已入库待接/待测 | target_net、SVE |

组件×环境全矩阵见 [MyZero 总览](../../examples/my_zero/README.md#内部组件进展团队--promote-时改-recipers)；实测数字一律以[基准账本](../../examples/my_zero/cartpole/README.md)为准。

---

## 3. 验收协议（三层，发版固定关卡）

| 层 | 命令 | 判什么 | 何时跑 |
|----|------|--------|--------|
| **单元/集成测试** | `just test`（RL 子集 `just test-filter rl`；2026-07-12 当前 **351 passed**） | 正确性 | 每次改动 |
| **管线 smoke** | `just smoke-rl`（聚合 8 目标：MyZero cartpole/pendulum/gomoku + PPO cartpole + SAC cartpole/pendulum/platform/lunarlander） | 管线通、loss 有限；**不验收敛** | 每次发版 |
| **统计基线** | MyZero：`SEEDS=3 cargo run --example my_zero_cartpole --release`；增量链 4 档：`my_zero::tests::baseline_matrix_bench`（`--ignored` 手动）；PPO/SAC：`SEED=42/43/44` 各跑一次 | **3-seed 中位 env-steps-to-solved + 达标率**（官方哨兵口径） | 发版前 / 改算法行为后 |

### 3.1 基线判据原则（2026-07-02 定稿）

- **变慢 ≠ 失败**：重测数字（哪怕比历史口径慢）直接接受为新基线并回填账本；历史行仅作方向性参考。
- 只有「跑完仍无法收敛 / 无法达标（CartPole greedy < 475）」才记 known-fail `.issue`，且不阻塞发版。
- 账本每行必须带**口径列**（profile / BLAS / seeds / 日期）；跨口径对比 wall-clock 无效，env-steps 有效。
- CartPole 是 **sanity/regression 哨兵**（叠组件不崩的证明），**不是**组件价值的证明台，更不拿 wall-clock 判生死——规划红利要到更难的环境验证（见 §5）。

### 3.2 改动纪律（搬运 ≠ 改进，沿用）

- **搬运**（挪代码、改 import、折 config）：行为零变化 → 可批量；CartPole 哨兵过即无回归。
- **改进**（改行为）：**一次一项**，单独 A/B 消融（3-seed 统计口径），单独过哨兵；不得与搬运混提交。

---

## 4. v0.25 结果小结（MyZero 统一）

1. **MyZero 成为唯一 `*Zero` 实现**：算法主体 + 全部组件（value_encoding / n_step / consistency / reconstruction / value_prefix / target_net / SVE / reanalyze / completedQ / Gumbel / Sampled）统一进库 `src/rl/algo/my_zero/`；旧 `muzero/` + `efficientzero/` 模块与示例整体删除。
2. **用户 API 定形**：`MyZero::new(env_id)` 链式 builder + `train/eval/run` 生命周期 + `.otm` 统一持久化；示例瘦身至 ~40 行。
3. **框架级 autograd 修复改变基线语义**：MSE/MAE/BCE/Huber 反向此前忽略 `upstream_grad`（作中间 loss 时丢链式缩放），修复后辅助 loss 回到正确量级——**所有历史 benchmark 数字失效**，哨兵改 3-seed 统计口径并全量重测（数字见[基准账本](../../examples/my_zero/cartpole/README.md)）。同批：非连续张量守卫、MCTS recurrent 单趟前向提速 ~3.4×、训练 batch-native（batch=8 快 2.2×）。
4. **负结果沉淀**：completedQ / Gumbel-root / reanalyze 在 CartPole 实测无增益或有害 → recipe 关、issue 记录（不否定其在 `|A| > sims`、低延迟 acting、图像等场景的价值，v0.26+ 复测）。
5. **Pendulum 转入可学习性诊断**：value 头容量已证伪为瓶颈，根因收敛到上游 target/搜索（[issue](../../.issue/items/pendulum_failure_diagnosis.md)）；**不作为发版门禁**。

---

## 5. v0.26 方向（2026-07-01 战略转向定稿）

> **执行次序总纲**：v0.26→v0.28 五阶段收口战役（终态验收、阶段退出判据、issue 裁决映射、制度化条款）见 [RL 全面收口规划](./rl_closure_plan.md)；本节只保留方向级优先级。
>
> **战略复盘已定档（2026-07-10）**：生产树坚持全程 learned dynamics，真规则 /
> env snapshot 仅作 reference diagnostic；先建设“同一稳定契约下可退化的模块族”，不再在
> 棋类真规则 / learned 预算 / 图像三条载体之间三选一。前两阶段（零行为契约重构 +
> 输入/动作最小纵切）已完成，事实源见[通用 world model 地基](./my_zero_world_model_foundation.md)；
> 主动数据生成 3A0 已于 2026-07-12 独立负裁：continuation Brier 虽与任务关键
> 局面相关，但新增真实数据训练后 2/3 seed 回归，未进入 ErrorQ/Collector。
> 下一阶段顺位为 recurrent posterior/POMDP，再做 stochastic chance search。

> 依据：真实目标是**中国象棋**（离散完美信息 self-play）与**商业图像游戏**（图像 obs + 实时 + 样本贵）——两者都不在「动作空间广度」轴上。路线从「磨动作空间（Pendulum 连续 / Platform 混合）」**转向「磨观测空间（图像/CNN）+ self-play」**。完整论证见[算法纲领 §2.3](./my_zero_algorithm_vision.md#23-战略目标与优先轴2026-07-01-定稿)。

| 优先级 | 方向 | 说明 |
|--------|------|------|
| **P0 ✅（2026-07-02 完成）** | loss 系数重标定消融 | 已裁决：**recon_coef 1→16 promote**（新哨兵中位 **~9.8k** env-steps、3/3，超过 bug 时代 13.1k；cont 保持 1.0，cons 无重标定理由）；同批 5-seed 复裁「recon=1 有害、recon 去留终审留图像环境」。数字与预注册协议见[基准账本](../../examples/my_zero/cartpole/README.md#v026-p0loss-系数重标定2026-07-02--当前官方哨兵) |
| **P0 ✅（2026-07-02 完成）** | **收口规划 Phase 0 全项闭环**（训练信号收口） | 梯度流审计：现状 canonical，sg 两臂不追加；**HL-Gauss 编码负结果**（中位 9.8k→27.6k，回退 two-hot，开关留库 Phase 1 图像域复测）；**obs symlog 负结果**（三臂系数不回移，recon_coef=16 裁决为权衡旋钮非单位换算，开关留库）。recipe 零变更定稿、哨兵逐 bit 复现 ~9.8k，**CartPole 自此冻结为纯回归哨兵（条款二生效）**。细节见[收口规划 §1](./rl_closure_plan.md)与[账本](../../examples/my_zero/cartpole/README.md#v026-phase-0编码--量纲消融2026-07-02) |
| **P0 ✅（2026-07-10 完成）** | 通用 learned-world-model 地基 | `ObservationSchema / ActionSchema+Codec / LatentState / WorldModel` 四契约落地；矩形/可配 history、Image+Dense Dict、token+mask、MultiDiscrete、2D continuous、Platform Hybrid 纵切全通；Reference board 仅 `cfg(test)`；CartPole 哨兵 8,741 中位、3/3 逐值复现 |
| **P0 ❌（2026-07-12 3A0 负裁）** | 主动数据 error proxy | 真实 reset 后的多样化轨迹审计观察到 continuation Brier 与战术关键局面关联；但同一 fixed block 未稳健下降，新增 30 局真实 game 再训练仅 1/3 seed 改善、2/3 回归 → 当前协议未证明 proxy 可跨 seed 稳定降低，停止 ErrorQ/Collector/H=K/5+5，不用 SAC/WGAN 补救；结论边界与 raw reward CE 熵下界见[数字账本](../../examples/my_zero/gomoku/README.md#phase-3a0--主动数据误差-proxy-审计2026-07-12) |
| **P0 下一阶段** | recurrent posterior / POMDP-lite | 增加 sequence replay、burn-in、mask 与跨真实时间步 hidden；先用 observation-aliasing toy 验证单帧失败、history posterior 成功，再进入真实图像 POMDP。不得与 stochastic chance nodes 同批 |
| **P0** | CNN 图像表征 + 图像离散基准（Atari-100k 类） | 商业游戏直接代理；复用已验收 consistency + reconstruction（自监督正是图像+少样本的命门组件） |
| **P1 ✅（2026-07-10 纵深终审完成）** | Gomoku self-play → 象棋踏脚石 | **棋盘支柱 M0–M4 已立，naive 战术纵深 ①–⑱ 全裁决**：真规则上限 0.90 证明纯 self-play 可行；learned dynamics 规则学习税主导。PER、consistency、reconstruction 均未获未见 seed 稳健增益（recon1 发现集 0.35，未见 seed 对照 0.28 vs recon1 0.20）；G3 selected 100 局/档 holdout 0.19/0.09/0.07/0，交付未过线，recipe 维持 base。全账见[棋盘账本](../../examples/my_zero/gomoku/README.md)与[战术墙 issue](../../.issue/items/gomoku_naive0_tactical_wall.md)；下一步为预算 / 真规则 / 图像线战略复盘，不自动追加组件 |
| **P1** | reanalyze 复活 + acting/reanalyze 解耦 | 「实时轻 acting（少 sim / policy 先验）+ 离线重 reanalyze（榨样本）」是商业游戏路线的战略组件；CartPole 负结果不构成否定 |
| **P2 ✅（2026-07-05 完成）** | Gumbel 少 sim acting 复测 | Gomoku s16（\|A\|=81 ≫ sims）复裁**中性**：无 CartPole 式灾难亦无增益 → 全域 recipe 关、代码保留，少 sim acting 降档可用；[issue 终局归档](../../.issue/_archive/my_zero_gumbel_completedq_cartpole_negative.md) |
| **降级** | Pendulum / Platform | 不在两大目标关键路径；Pendulum 保留诊断态，Platform 待具体需求再排 |

**一级风险**（显式管理）：CPU-only × 图像 CNN × MCTS × 实时的结构性冲突——见 [.issue/items/cpu_only_mcts_image_realtime_risk.md](../../.issue/items/cpu_only_mcts_image_realtime_risk.md)（§三b 已录 planning-free 退路）。

**P0/P1 消融候选补充**（2026-07-02 论文清账产出）：HL-Gauss value 编码、辅助 loss stop-gradient 解耦（挂 P0 批次）、loss 优先回放（挂 P1 reanalyze）——组件映射与执行顺序见 [Simulus 组件采纳消融计划](./my_zero_simulus_ablation_plan.md)。

## 6. 长期 backlog（可能做，不承诺）

Stochastic MuZero（chance node，动树核心）· MENTS/DENTS 熵 backup · MCTS 并行（virtual loss / 批量叶子）· PER · Beta 分布 · DQN/TD3（教学）· 演化+RL 联合搜索。详细谱系与接缝对照见[归档 §5.10](./_archive/rl_roadmap_v020_v024.md#510-mcts--zero-变体-backlog接缝已留做不做按需)与[算法纲领 §5](./my_zero_algorithm_vision.md#5-决策表可能做--暂缓--不做)。

**明确不做**：BetaZero（belief 规划，[纲领 §5.3](./my_zero_algorithm_vision.md#53-betazero-裁决2026-06-21-定稿)）· 不完全信息博弈（CFR 族）· 多智能体 N>2 · SAC 升格母算法。
