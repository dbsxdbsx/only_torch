# 更新日志

## [Unreleased]

> **v0.25 MyZero 统一算法（Phase 0/1）**：算法主体统一进库 + 全部组件吸收，MyZero 自包含；**旧 `muzero/` + `efficientzero/` 已整体删除，MyZero 成为项目唯一的 `*Zero` 实现**。CartPole-v1 回归哨兵全程 greedy 500 不变。

### Added

- **feat(mcts): 标准 Gumbel MuZero 根搜索**（`GumbelPolicy` + Sequential Halving + `RootScheduler::on_search_start`）；`MyZeroSearchPolicy` 接入 self-play / greedy eval / reanalyze；builder `.gumbel()` / `.gumbel_standard()`；CartPole 阶段 A/B bench 用例

- **feat(rl): MyZero `.otm` 统一持久化 + `model_io`**
  - 删除 `manifest.rs` 与旁路 `.bin`；契约写入 `OtmMetadata.myzero`（`env_id` / action / `reward_scale` / `latent_dim`）
  - `save_myzero_model` / `load_weights_into`；用户 API 仅 `load_model(path)`（path 不含 `.otm` 后缀）
  - 训练期 `BestTracker` 在 periodic greedy eval 创新高时写 `models/my_zero/{env_id}/seed_{seed}/best.otm`

### Changed

- **chore(rl): MyZero 默认 MCTS `num_simulations` 50→20**；CartPole cons+recon 基线 ~12.2k env-steps（seed=42）；sim=10/15 扫参见 `examples/my_zero/cartpole/README.md`
- **refactor(rl): MyZero 用户侧 API 链式 builder + train/eval/run 生命周期**
  - `MyZero::new(env_id)` 唯一入口；`.solved` / `.max_episodes` 仅绑 `.train()`
  - 去掉 `restore_best`：`.train()` 返回实例持有 **latest** 训末权重；训后 `eval`/`run` 沿用 latest
  - 要用磁盘 best → 显式 `load_model(TrainReport.model_path)`；`final_greedy` = latest 分，`best_greedy` = 训练期历史 best
- **feat(rl): MyZero 统一进库 `src/rl/algo/my_zero/`（Phase 0，算法主体下沉）**
  - 5 层 `MyZeroConfig`（`EnvConfig` / `ModelConfig` / `TrainConfig` / `ComponentConfig` / `RunConfig`）+ `apply_env_overrides`（`EZ_CONS/CQ/SIMS/SEEDS/SMOKE/DIAG/GAMMA/LR/MAX_EP/NUM_ACTIONS/RSCALE/SOLVED` 旋钮集中一处）
  - `network.rs`：三网络模型（repr/dyn/pred + value-prefix LSTM + SimSiam 分支）从示例迁入
  - `action.rs`：`ActionAdapter` 从 `GymEnv` **自动推断**动作空间（离散/连续/范围/档数）+ idx→env 映射；`ActionPlan::{Auto, Discretize}`——动作类型是 env 事实（库自动推断），「连续如何近似」才是用户选择
  - `runner.rs`：统一 `run()`（self-play + 训练 + greedy eval + 多 seed + SMOKE + DIAG），内部 `Python::attach`
- **feat(rl): MyZero 吸收全部算法组件，自包含（Phase 1，不再 import muzero/ez）**
  - 从 `muzero/` 吸收：`support` / `value_transform` / `n_step` / `reanalyze` / `loss`
  - 从 `efficientzero/` 吸收：`consistency` / `value_prefix` / `sve` / `target_net`（含本地 `TargetConfig`）
  - `my_zero` 模块单测 31 个全绿；旧 `muzero/` + `efficientzero/` 模块与示例已删除（见下方 Removed）
- **test(rl): MyZero value-head 容量诊断单测**
  - `my_zero::tests::value_head_capacity`：喂高方差可分 value 目标，head 把高/低组预测间隔训到精确 **14.0** → 证伪「value head 学不动」，Pendulum value 坍缩根因缩到上游 target/搜索（见 [`pendulum_failure_diagnosis.md`](.issue/items/pendulum_failure_diagnosis.md)）

### Changed

- **refactor(rl): MyZero 示例瘦身为 thin `main.rs`**：`cartpole` 663→41 行、`pendulum` 842→45 行（只填 config + 调 `run`）；删 `examples/my_zero/cartpole/model.rs`（并入库 `network.rs`），移除 pendulum 的 `#[path]` 复用
- **refactor(rl): MyZero 配置命名归位**：`FeatureSet`→`ComponentConfig`（对齐文档「组件」术语）；示例侧 `MuZeroConfig` 依赖 → my_zero 自有 `TrainConfig`（告别 MuZero 名）；字段统一 `*_config: *Config`
- **refactor(rl): MyZero 命名清理（去 EZ/MuZero 残留）**：`support.rs`→`value_encoding.rs`（自报「value/reward 分类编码层」用途，"support" 退为模块内术语）；消融环境变量去 `EZ_` 前缀、改用组件名 `CONSISTENCY` / `VALUE_PREFIX` / `TARGET_NET` / `SVE`（`TARGET_NET` 刻意避开 Rust 构建系统占用的 `TARGET`）；9 个组件文件头「吸收自 MuZero/EfficientZero」改为纯描述 + 论文引用（保留 `canonical MuZero` / Schrittwieser 等**学术溯源**）

### Removed

- **chore(rl): 删除旧 `muzero/` + `efficientzero/` 全部代码，MyZero 成为项目唯一的 `*Zero` 实现**
  - 删 `src/rl/algo/muzero/` 与 `src/rl/algo/efficientzero/`（组件已于 Phase 1 吸收进 `my_zero/`，无 kept-code 依赖；`cargo check` 通过）
  - 删 `examples/muzero/`（CartPole）与 `examples/efficientzero/`（cartpole/pendulum/platform/gomoku/atari/ant/minari 七格矩阵），并移除 `Cargo.toml` 对应 `[[example]]` 条目
  - 删 `src/rl/tests/algo_muzero.rs`（n-step / value transform 已由 `my_zero` 内部单测覆盖）；`justfile` 去掉 `example-cartpole-muzero` / `smoke-cartpole-muzero`，改提供对等的 `example-cartpole-my-zero` / `smoke-my-zero-cartpole`
  - 文档（`AGENTS.md` / `rl_roadmap.md` / `rl.instructions.md` / `rl_python_env_setup.md` / `examples/my_zero/README.md`）同步收口为「MyZero 唯一实现」；论文溯源（MuZero / EfficientZero / SimSiam / Gumbel 等）作为**学术引用**保留

## [0.24.0] - 2026-06-16

> **EfficientZero V2 框架管线完备（SMOKE 级）**：Phase 0a 五根接缝契约 + EZ 算法 helper 入库 + 六格示例矩阵 SMOKE 全绿（CartPole/Pendulum/Platform/Gomoku/Atari/Ant/Minari）+ 全项目 CartPole 统一 v1。分数未全面压测达标——简化点（离散化候选替代忠实 Gumbel、降采样替代 CNN、小盘 best-effort）均在示例 doc 标注。**v0.25 起由 MyZero 统一算法接棒，以消融实验方式逐增量迭代，最终取代分散的 MuZero/EfficientZero 实现。**「发版」= bump 版本号 + 更新 CHANGELOG，不 `cargo publish`。

### Added

- **feat(rl): EZ-V2 Phase 0a 五根接缝契约（MCTS 可扩展性地基）**
  - `ActionSampler` trait + `DiscreteActionSampler`（`src/rl/mcts/traits.rs`）：独立接缝负责候选动作生成，与 `SearchPolicy` 解耦；离散/连续/混合/Sampled 统一入口
  - `RootScheduler` trait + `PuctScheduler`（搜索生命周期 hook）：Gumbel sequential halving 留位；默认实现零开销等价历史 PUCT
  - `SearchPolicy::make_root_scheduler` 默认方法
  - `mcts_search` RNG 注入：签名加 `rng: &mut dyn RngCore` 参数，所有调用点（示例/reanalyze/测试）统一改用外部 seeded RNG，训练可复现
  - `SelfPlayStepExtras`（`src/rl/buffer/self_play.rs`）：builder 模式承载算法增量字段（`value_prefix_target`），禁裸 Option 坟场
  - `ReplayBuffer::sample_indexed`（`src/rl/buffer/replay.rs`）：返回 `(index, item)` 对，完整 reanalyze 回写 / PER priority 更新预留
  - `MctsModel::State` 不透明契约文档（第 5 根接缝）+ 契约测试 `mcts_recurrent_state.rs`：证明 EZ value prefix 忠实版（LSTM hidden 穿 MCTS 树）无需改内核 backup
  - `mcts_sampler.rs` ActionSampler 接缝契约测试

- **feat(rl): `src/rl/algo/efficientzero/` EZ-V2 函数式 helper 入库**
  - `EfficientZeroConfig`（组合 base/search/gumbel/reanalyze/target/loss 六子配置，非扁平）
  - `negative_cosine_similarity`（SimSiam consistency loss，stop-gradient）
  - `reward_prefix_targets` / `prefix_to_delta`（value prefix 累计目标 + 增量还原校验）
  - `ema_update` / `hard_update` / `sync_target`（target network 同步，hard/EMA + 间隔调度）
  - `sve_blend`（search-based value estimation，n-step 与 search root value 混合）
  - 各子模块含单元测试（consistency 3 / value_prefix 4 / target 3 / sve 2 / config 2）

- **feat(rl): EZ-V2 多模式示例矩阵 SMOKE 全绿（Phase 2–4 管线打通）**
  - 新增 5 个 EZ 示例（复用 `examples/efficientzero/cartpole/model.rs`，零库层改动）：
    - `examples/efficientzero/pendulum/`（Pendulum-v1，纯连续→离散化 9 档）
    - `examples/efficientzero/platform/`（Platform-v0，混合 Tuple→离散化候选）
    - `examples/efficientzero/gomoku/`（双人 learned-model，自定义 `to_play` 携带适配器**首次触发内核 negamax 双人路径**，6×6 小盘）
    - `examples/efficientzero/ant/`（Ant-v5，8 维连续→固定候选）
    - `examples/efficientzero/atari/`（ALE/Breakout-v5，像素降采样→MLP）
    - `examples/efficientzero/minari/`（Minari 离线 load + 训练，无本地数据集优雅跳过）
  - 六格示例 `SMOKE=1` 全部跑通（交互/自对弈 + 训练 + loss 有限 + 无 panic）；本机 ale-py/mujoco/minari 依赖齐全，Atari/Ant/Minari 真跑通
  - **定位**：v0.24 EZ **框架管线完备（smoke 级）**，分数未压测达标；简化点（离散化候选替代忠实 Gumbel 连续/混合搜索、Atari 降采样替代 CNN repr、Gomoku 小盘 best-effort）均在示例 doc 与 `examples/efficientzero/README.md` 标注为 TODO

### Changed

- **refactor(rl): CartPole 示例/测试/文档统一 v1，废弃 v0**
  - `examples/{sac,ppo,muzero}/cartpole/`：环境硬编码 `CartPole-v1`，达标门槛 `greedy eval ≥ 475`，并打印「到 475 所需 episode/env-step」样本效率指标；移除临时 `ENV_ID` 切换与 v0 分支
  - `src/rl/tests/mcts_cartpole_env.rs`：`CartPole-v0` → `CartPole-v1`
  - 同步更新 AGENTS.md / rl_roadmap.md / rl.instructions.md / rl_python_env_setup.md / RL 主线 plan 的验收分层（v0→v1、195→475）
  - 一次性测得四算法 v1 样本效率（到 500 满分所需 env-step）：MuZero ~3.8k（噪声大、spike）/ EZ(cons+vp) ~31k / PPO ~102k / SAC ~129k——model-based 样本效率碾压 model-free，详见 [`.issue/items/post_ez_v2_research_backlog.md`](.issue/items/post_ez_v2_research_backlog.md)

## [0.23.1] - 2026-06-15

> **MuZero canonical 完全体达标**：补齐 categorical value/reward + latent min-max 归一化 + absorbing state + canonical 梯度缩放后，MuZero CartPole-v0 从「卡 ~40 平台期」收口到 **greedy(temp=0) eval 20 局均值 199.5 ≥ 195**。真因是搜索在 learned model 上的 **no-terminal 价值膨胀**，由 absorbing state 直击修复。这套机制是整个 `*Zero` 家族（AlphaZero / MuZero / EfficientZero）的共享地基，已逐项正确性验证。「发版」= bump 版本号 + 更新 CHANGELOG，不 `cargo publish`。

### Added

- **feat(nn): `Var::scale_gradient(scale)` 梯度缩放算子**（`src/nn/var/mod.rs`）
  - 前向恒等、反向梯度 ×scale；恒等分解 `s·x + (1-s)·detach(x)`，复用已测 `detach` + 标量乘加，零新增手写反传
  - `detach()` 是其 `scale=0` 特例；通用 autograd 原语，供 `*Zero` 家族 K 步 unroll 复用
  - 7 个单元测试（前向恒等 / 反向 ×s / 链式半衰 / detach 等价）

- **feat(nn): `Var::amax` / `Var::amin`**（`src/nn/var/ops/reduce.rs`）
  - 复用带 backward 的 `Amax`/`Amin` raw node，Var 级包装 + min-max 组合前向/反向单测；支撑 latent 归一化

- **feat(rl): MuZero categorical value/reward 表示**（`src/rl/algo/muzero/support.rs`）
  - `SupportConfig` + `scalar_to_two_hot`（h(x) 变换域 two-hot 编码）+ `two_hot_to_scalar`（期望解码），对齐 canonical MuZero 附录 F

- **feat(rl): `MuZeroConfig` 超参容器**（`src/rl/algo/muzero/config.rs`）
  - 训练/搜索超参按环境配置（`num_simulations` 等：棋类 ~800、向量/Atari ~50），跨 `*Zero` 家族复用；2 个单元测试

- **feat(rl): MuZero Reanalyze**（`src/rl/algo/muzero/reanalyze.rs`）
  - `reanalyze_game`：用**最新**网络对旧轨迹重跑 MCTS，刷新 policy/value 目标（reward/terminated 不变）
  - 共享 `SearchResult::root_value()`：self-play 与 reanalyze 统一 root value 口径
  - 示例训练循环配置门控接入（`reanalyze_fraction`，默认关、`REANALYZE=` 可开）+ mock 单元测试

### Changed

- **refactor(rl): MuZero 示例补齐 canonical 完全体**（`examples/muzero/cartpole/`）
  - categorical value/reward 头 + 交叉熵 loss（替换标量 MSE）
  - representation / dynamics 末尾 latent min-max 归一化到 [0,1]
  - canonical 梯度缩放：每 dynamics step latent 梯度 ×0.5 + 每 recurrent step loss 梯度 ×(1/K)（`DYNAMICS_GRADIENT_SCALE` 真正生效，替换原 `1/(K+1)` 替身）
  - n-step target 区分 terminated/truncated（truncation 仍 bootstrap，避免低估满分局末端 value）
  - 超参经 `MuZeroConfig` 注入；`SelfPlayStep` 增加 `terminated` 字段

### Fixed

- **fix(rl): MuZero no-terminal 价值膨胀（CartPole 平台期真因）**
  - 搜索在 learned model 上「幻想无限存活、每步累加 +1」→ root_value 膨胀 + policy 坍缩；canonical **absorbing state**（终止后 unroll 填 reward0/value0/uniform）直击修复 → greedy eval 199.5 达标
  - 实测修正：categorical 单上反而 regressed（~9），证伪「categorical 是平台期主因」；真因是 no-terminal

## [0.23.0] - 2026-06-15

> SAC ✅ / PPO ✅ CartPole-v0 ≥195；MCTS 算法底座收口（Dynamics + MinMaxStats）+ 8 个逻辑 bug 修复；on-policy buffer 入库。**MuZero 架构已验证（能学习），但 ≥195 推 v0.24**（缺 categorical value + latent 归一化，见 `.issue/items/muzero_cartpole_scalar_value_plateau.md`）。「发版」= bump 版本号 + 更新 CHANGELOG，不 `cargo publish`。

### Added

- **feat(rl): `src/rl/algo/ppo/` PPO 函数式 helper 入库**
  - `compute_gae`：GAE 优势估计（terminated/truncated 分离，只 mask terminated）
  - `clipped_policy_loss` / `value_loss` / `entropy_bonus`：PPO 三损失构件
  - `PpoBatch` + `rollout_to_batch` + `normalize_advantages`
  - 5 个单元测试（GAE 手算 + terminated/truncated 双路径 + 优势标准化）

- **feat(rl): `RolloutStep` + `RolloutBuffer` 入库**（`src/rl/buffer/rollout*.rs`）
  - On-policy 采集缓冲区，用完即弃，不 impl BufferItem
  - 7 个单元测试

- **feat(rl): `Dynamics` trait + `DynamicsModel` adapter**（`src/rl/mcts/dynamics.rs`）
  - MuZero learned dynamics 接口（representation + dynamics + prediction 三段式）
  - `DynamicsModel<D>` 适配器桥接到 `MctsModel`（State = Vec<f32> latent）
  - 3 个单元测试

- **feat(rl): `MinMaxStats` Q 值归一化**（`src/rl/mcts/min_max.rs`）
  - PUCT 中 Q 值 min-max 归一化，解决 value 无界环境的探索失效问题
  - 穿线至 `mcts_search` → `select` → `PuctPolicy::select_child`

- **feat(rl): PPO CartPole-v0 示例**（`examples/ppo/cartpole/`）
  - actor-critic 独立 MLP（128 隐藏层），离散 Categorical
  - GAE + clipped surrogate + value loss + entropy bonus
  - SMOKE 模式支持

- **feat(rl): `src/rl/algo/muzero/` MuZero 函数式 helper 入库**
  - `value_transform` / `value_transform_inv`：标量 value/reward 变换 `h(x)=sign(x)(sqrt(|x|+1)-1)+εx`
  - `compute_n_step_target`：n-step bootstrapped return
  - `loss` 模块：value/reward loss 系数 + 梯度缩放常量
  - 9 个单元测试（变换 round-trip + 单调性 + 压缩 + n-step 手算）

- **feat(rl): MuZero CartPole-v0 示例**（`examples/muzero/cartpole/`）
  - representation / dynamics / prediction 三网络（128 隐藏层）
  - MCTS-on-latent（复用 mcts_search + DynamicsModel）+ value transform + 温度退火 + 真 batch 梯度累积训练
  - K=5 步 unroll + n-step(50) value target；`ReplayBuffer<SelfPlayGame>` 整局存储
  - **现状**：训练能学习（avg 9.4→~40，峰值 180+）但卡平台期，未达 195；缺 categorical value + latent 归一化（推 v0.24，见 issue）

### Fixed

- **fix(rl): 修复 8 个 MCTS/MuZero 逻辑 bug（AlphaZero 系列共享地基校正）**
  - `search.rs`：首次 terminal 叶子 backup 用 0（而非网络预测 value），终局 bootstrap 口径一致
  - `min_max.rs`：`MinMaxStats` 无有效 range 时归一化返回 0.5 中性值（而非 raw Q，原会压死 PUCT exploration）
  - `puct.rs`：`recommend` / `make_targets` 用真实 visit count（去掉 `max(1)`，全 0 才 uniform fallback）
  - MuZero 示例：`root_value` 按 `reward + γ·V(child)` 口径加权（原漏即时奖励与折扣）；移除恒不触发的 `reward<-0.5` terminal 误判；`value_transform_inv` 输出 clamp 防未训练网络噪声放大；梯度缩放不再误缩 prediction head；训练改真 batch 梯度累积 + 允许 short unroll 覆盖 episode 尾部

### Changed

- **refactor(rl): `SelfPlayStep` 扩展 MuZero 字段**
  - 新增 `reward: f32`（必填，AlphaZero 填 0.0）
  - 新增 `root_value: Option<f32>`（可选，AlphaZero 无需）

- **refactor(rl): `SearchPolicy::select_child` 签名增加 `&MinMaxStats`**
  - v0.22 已预告的唯一搜索签名变更
  - AlphaZero（value ∈ [-1,1]）下 min-max 近似恒等，无害

- **refactor(rl): SAC CartPole 示例增强**
  - 网络宽度 64→128
  - 收敛判据：100 局均值 >= 195（原为单局）
  - 新增 20 局 deterministic eval 函数

### Docs

- justfile：新增 `example-cartpole-ppo` / `smoke-cartpole-ppo` / `example-cartpole-muzero` / `smoke-cartpole-muzero`
- `.issue/items/muzero_cartpole_scalar_value_plateau.md`：记录 MuZero CartPole 标量 value 表示导致 ~40 平台期的根因（缺 categorical + latent 归一化）与 v0.24 补齐方案

## [0.22.0] - 2026-06-15

> AlphaZero 基础设施：库内 MCTS + Python 五子棋环境 + 规划桥接 + Agent trait。「发版」= bump 版本号 + 更新 CHANGELOG，不 `cargo publish`。

### Added

- **feat(rl): `python/gym_env/` 五子棋 Gymnasium 环境包**
  - `board.py`：纯规则层（增量获胜检查 ~6μs、numpy legal_mask ~0.8μs）
  - `env.py`：`GomokuSelfPlayEnv`（无对手，AlphaZero 训练）+ `GomokuEnv`（带 naive 对手，评测）
  - `opponents.py`：5 级 naive 对手（random/naive0-3）
  - `pip install -e python/gym_env`；Gymnasium 注册 `Gomoku-selfplay-v0` 等

- **feat(rl): `src/rl/mcts/` MCTS 搜索引擎**
  - `MctsModel` trait（root + recurrent，mctx 同构；State 关联类型不透明）
  - `SearchPolicy` trait（hook 粒度：prepare_root / select_child / recommend / make_targets）
  - `Predictor` trait（day-1 `predict_batch` 接口）
  - `PuctPolicy`：Dirichlet 根噪声 + UCB 选择 + 温度采样
  - `mcts_search(model, policy, obs, cfg)` 不吃 `&GymEnv`
  - Arena 树结构（`Vec<Node>` + `NodeId`）
  - Backup 统一公式（perspective_factor 支持单智能体 / 双人零和）
  - 3 个单元测试

- **feat(rl): `GymEnv` 规划桥接方法**
  - `legal_mask` / `snapshot` / `restore` / `current_player` / `is_terminal` / `board_step` / `board_observation_flat`
  - SAC 用户无感；AlphaZero 用它驱动 MCTS

- **feat(rl): `SelfPlayGame` 数据结构入库**（`src/rl/buffer/self_play.rs`）
  - `SelfPlayStep` / `SelfPlayGame` / `GameOutcome`
  - `impl BufferItem for SelfPlayGame`，复用 `ReplayBuffer` 容器机制
  - 5 个单元测试

- **feat(rl): `Agent` / `PlanningAgent` 双 trait**（`src/rl/agent.rs`）

### Changed

- **refactor(rl): 五子棋环境迁移至 `python/gym_env/gomoku/`**
  - 删除旧 `tests/python/custom_envs/gomoku.py` + `test_08_gomoku.py`
  - 默认棋盘 9×9（旧为 15×15）；15×15 用 `Gomoku-*-15x15-v0`
  - 旧测试适配新注册 ID

- **refactor(rl): `GymEnv` 自定义环境注册改为统一 `import gym_env`**
  - 去除 `Gomoku-` 硬编码 if 分支，新游戏只改 Python 侧

### Docs

- `rl_roadmap.md` 目录结构更新（含 mcts / agent / self_play / python/gym_env）

## [0.21.0] - 2026-06-15

> SAC helper 入库 + 示例瘦身 + LunarLander + 目录重组。「发版」= bump 版本号 + 更新 CHANGELOG，不 `cargo publish`。

### Added

- **feat(rl): `src/rl/algo/sac/` 函数式 helper 入库**
  - `SacBatch` + `transitions_to_batch`：`&[Transition]` 一次性转批量 Tensor
  - `compute_td_target`：`r + γ·(1-terminated)·V(s')`
  - `compute_v_discrete` / `compute_v_continuous` / `compute_v_hybrid`：三变体软 V 值
  - `update_alpha`：温度梯度步进 + clamp
  - 22 个纯 Rust 单元测试

- **feat(rl): LunarLander-v3 SAC-Discrete 示例**（`examples/sac/lunarlander/`，78 行）
  - 验证 SAC helper 跨环境复用（与 CartPole 同构，obs=8 / actions=4）

### Changed

- **refactor(rl): 三个 SAC 示例瘦身**
  - CartPole 469→125 行 / Pendulum 442→117 行 / Platform 319→129 行
  - 删除各示例本地 Experience / ReplayBuffer，改用库的 `Transition` + `ReplayBuffer` + SAC helper

- **refactor(rl): 目录重组 `examples/traditional/sac/` → `examples/sac/`**
  - Cargo.toml / justfile / README / 全项目路径引用同步

### Docs

- SAC `README.md`：加 LunarLander 行、更新 helper 说明
- justfile：加 `example-lunarlander-sac` 目标
- README.md：修正 `moving_sac` → `platform_sac`、加 LunarLander 行

## [0.20.0] - 2026-06-15

> RL 主线首个**代码**版本：Gymnasium-only 环境清理 + buffer 基础 + smoke 门禁。「发版」= bump 版本号 + 更新 CHANGELOG，不 `cargo publish`。

### Added

- **feat(rl): `Transition` + `ReplayBuffer<T: BufferItem>` 入库**
  - `Transition`：单步交互数据，存 `terminated` + `truncated`（不合并 `done`），含 action 编码约定 doc
  - `BufferItem` trait：`Clone + 'static`（砍 `Send`，CPU-only 单线程）
  - `ReplayBuffer<T>`：泛型 FIFO 淘汰 + 有放回 `gen_range` 采样（禁全长索引）
  - 15 个纯 Rust 单元测试（FIFO / 有放回 / 空 buffer / seed 可复现 / action shape×3 / 终止字段保真 / clone 独立）

- **feat(rl): `cartpole_sac` smoke 模式（`SMOKE=1`）**
  - 3 episode 短跑，每步断言 `loss.is_finite()`，不验证 reward 收敛
  - justfile `smoke-cartpole-sac` 目标

- **feat(rl): `GymEnv::flatten_obs` Tuple 观察展平 helper**
  - 按空间原生顺序拼接所有子空间为单一 `Vec<f32>`

### Changed

- **refactor(rl): Gymnasium-only `GymEnv`（Phase 0 legacy 大清理）**
  - 删除 `use_legacy_gym` 字段及所有 gym 回退分支（reset / step / get_module_name）
  - 删除 `try_make_env` gym 回退、`extract_gym_reset_obs`、`gym_hybrid` 导入
  - `gymnasium.make` 失败时 panic + 中文安装指引（不再 `import gym`）
  - `GymEnv::step` 返回 `(obs, reward, terminated, truncated)` 四元组
  - legacy 三条 grep 全零：`import("gym")` / `use_legacy_gym` / `gym_hybrid`

- **refactor(rl): Platform-v0 替换 Moving-v0（Phase 0b）**
  - `examples/traditional/sac/moving/` → `platform/`，Cargo.toml `moving_sac` → `platform_sac`
  - 模型简化：统一连续头（3 维），无条件分支
  - `try_import_gym_env_module`：`Platform-v0 → import gym_platform`
  - 3 个 Platform-v0 Rust 测试 + Python `test_07_hybrid.py` 全面重写

- **refactor(rl): 三个 SAC 示例 TD target 改用 `1 - terminated`**
  - 修正 CartPole truncation 场景的 bootstrap 错误

### Removed

- `Step` struct（已替换为 `Transition`）
- Moving-v0 / Sliding-v0 测试用例（已替换为 Platform-v0）
- `import gym` / `gym_hybrid` 所有代码路径

### Docs

- `rl_roadmap.md`：新增 §2.2.1b 覆盖边界表 + Phase 0 大清理范围逐文件表
- `rl_python_env_setup.md`：Gymnasium 版本锁定 `>=1.3.0,<2.0` + 实测记录
- `distributions_design.md` / `rl_roadmap.md`：修正 `examples/sac/` 断链为 `examples/traditional/sac/`
- justfile：新增 `examples-rl` / `py-gym-platform` / `smoke-cartpole-sac` 目标

## [0.19.0] - 2026-06-15

> RL 主线首个版本：**规划与设计决策定稿**（纯文档 / 规划发版——bump 版本号 + 更新 `CHANGELOG.md`，**不** `cargo publish`）。运行时改造（Gymnasium-only `GymEnv`、buffer 落库、smoke 门禁）自 **v0.20.0** 起实施，详见 [RL 路线图](.doc/design/rl_roadmap.md) 与主线实施计划。

### Changed

- **docs(rl): RL 主线规划定稿 + 2026-06-07 体检决策同步**
  - 环境：`GymEnv` 定调 **Gymnasium-only**（删 legacy gym 回退）；混合动作改用 **`Platform-v0`**（`hybrid-platform`），弃 gym-hybrid / Moving；离线数据用 Minari
  - 老 gym / 其他库环境：不在 Rust 层兼容，改在 **Python 侧用 [`shimmy`](https://shimmy.farama.org/) 适配**成标准 Gymnasium 环境后经 `GymEnv` 接入（Rust 永不 `import gym`）
  - 验收分层：SAC / MuZero / PPO 架构跑通统一 **`CartPole-v0` reward ≥ 195**；**EfficientZero V2（EZ-V2，第二代）** 为唯一终极调优算法（全 `-v1` 环境）
  - buffer 设计：`Transition` 取代 `Step`；`ReplayBuffer<T: Clone + 'static>` 泛型（砍 `Send`）；`sample` 定义为按存储单位有放回抽样、**非训练采样器**，禁建全长索引、返 owned `Vec<T>`
  - **终止语义**：`Transition` 存 `terminated` + `truncated`（对齐 Gymnasium），`truncated` 仍 bootstrap；`GymEnv::step` 透出两信号，修正 `CartPole-v0` 200 步截断被当真终止导致的 TD target 误算
  - **MCTS 抽象边界（§2.5）**：`mcts_search` 吃 `MctsModel`(root+recurrent) + `SearchPolicy`、**不吃 `&GymEnv`**；`SearchResult` 暴露根孩子原始统计，为后续 AlphaZero / MuZero / EZ-V2 复用同一搜索预留
  - `GymEnv` 维持 panic 为主、不全面 `Result` 化；seed 显式注入契约
  - 文档同步：`rl_roadmap.md`（新增 §2.5 MCTS 抽象、§5.10 变体 backlog、§7.7 体检决策）、`AGENTS.md`、`rl.instructions.md`、`rl_python_env_setup.md`、`sac/README.md`、`sac_mathematical_foundations.md`；`examples/traditional/sac/cartpole` `target_reward` 190 → 195

### Docs

- 同步 `AGENTS.md` / `README` / `memory_mechanism_design.md` 至 v0.18.0 完成状态（Attention 阶段进度、RL 主线与 Phase D 留坑索引）
- 统一 17 个文档的数学表达式为 LaTeX `$...$` / `$$...$$` 格式，修复此前公式无法正确渲染的问题

## [0.18.0] - 2026-05-27

### Added

- **feat(nn): Attention / Transformer 闭环并入演化系统**
  - **Layer**：`MultiHeadAttention` 补齐 `input_size`、`from_vars`、因果 / padding mask 工具与 `forward_masked`；新增 `SinusoidalPositionalEncoding` / `LearnableAbsolutePositionalEncoding`；新增 Pre-LN `TransformerEncoder` / `TransformerEncoderLayer`
  - **Evolution**：`CellAttention` 复合模板节点（QKV 父边顺序与 Layer 对齐）、`SequenceOpSet::{RecurrentOnly, AttentionOnly, RecurrentWithAttention}`、`expand_attention` / `resize_attention_out`、descriptor rebuild / mutation / net2net 占位路径；ONNX 导出与函数保持 net2net 扩宽留 Phase 3
  - **示例**：`parity_transformer_var_len`（传统 API，桶式同长度变长 parity + Transformer Encoder）；`evolution_parity_seq_attention`（演化 API，`RecurrentWithAttention` 混合搜索 RNN/LSTM/GRU 与 MHA）
  - **测试**：37 个单元测试（`layer_attention` / `layer_positional` / `layer_transformer` / `cell_attention` / `attention_evolution` / `attention_rebuild`）
  - **文档**：`memory_mechanism_design.md` Phase 3.5 / 4.5、`neural_architecture_evolution_design.md` Sequence 章节同步

- **feat(vision/detection): prediction 侧 letterbox→原图反映射 API（与 label 侧对称）**
  - `Detection::map_to_origin(self, &LetterboxResult) -> Self`：单框反映射，保留 `score` / `class_id`，bbox 几何由 `LetterboxResult::bbox_to_origin` 接管（含原图边界 clip）；调用方不再需要手拼 `Detection::new(lb.bbox_to_origin(d.bbox), d.score, d.class_id)`
  - `vision::detection::restore_letterbox_detections(detections, &LetterboxResult, DetectionLabelFilter) -> Vec<Detection>`：批量反映射 + clip + min_area 过滤，与 label 侧 `restore_letterbox_labels` 在 prediction 侧形态完全对称（共用 `DetectionLabelFilter`）
  - `vision::detection` mod 顶部 rustdoc 新增 **Quick Start 卡片**：按"推理第三方 ONNX YOLO / 训练自己的 detector / bbox 通用积木 / mAP 评估"四类高频入口给出导航，避免新用户翻 CHANGELOG 才发现 `adapter::yolo::v5::detect`
  - `vision::detection::transform` mod rustdoc 改为"label / prediction 双侧"形态说明
  - 配套 3 个新单元测试：单框 `map_to_origin` 保留 score/class_id 且 bbox 跨界自动 clip；批量版正常框 / 跨界框 clip / 过小框被 min_area 过滤的端到端组合行为；`min_area=0` 时批量版逐元素 ≡ 单框版的等价性锚定（防两条路径未来漂移）
  - 三个改动文件相对仓库现存 207 条预存 clippy warning **零新增**

- **feat(vision/metrics): 把空间域 example 的通用 helper 沉淀到库**
  - `src/vision/mask`：像素级 mask 处理（`argmax_to_class_map` / `foreground_from_multiclass` / `mask_to_ascii_lines`），统一替代 example 各自手写的 argmax / 多类→前景 / mask 文本化
  - `src/vision/viz`：展示画布工具——`Palette`（含 `default_categorical()` Tab10 风格 8 色调色板）、`pixel_block_scale`（toy 像素 N 倍放大）、`blend_alpha`（mask 半透明叠加）、`TinyFont` 5x3 像素字体（内置 0-9 / A-Z / 标点 / 小写自动落到大写，含 `draw_with_box` 一键画带底框的标签）
  - `src/vision/detection/adapter/yolo/v5`：YOLOv5 ONNX 输出解码 + per-class NMS（`detect` 端到端 + `decode` 低层），从 `chess_yolo_onnx_detect/yolo_decode.rs` 上提；为后续 v3/v4/v8/x 等独立子模块预留位置
  - `src/metrics/segmentation`：新增 `mean_instance_iou` / `mean_valid_slot_iou` / `empty_slot_accuracy` 三个实例分割指标，沿用现有运行时 shape 反推风格，与 `binary_iou` / `mean_iou` 系列 API 保持一致
  - `BBox::vec_from_tensor` / `vec_to_tensor`：`[N, 4]` Tensor ↔ `Vec<BBox>` 批量转换；故意不附带任何 clip 行为，让调用方按归一化 / letterbox / 原图坐标空间显式选择 `clip(0.0, 1.0)` 或 `clip_to_size(w, h)`
  - `vision::io::save_rgb_image`：直接保存 `RgbImage`，避免 `RgbImage → DynamicImage` 的 clone
  - 配套 42 个新单元测试，覆盖 BBox 批量 API、mask 工具、viz 调色板/字体/像素操作、yolo v5 decoder 阈值过滤与 NMS、instance segmentation 指标空 slot / valid slot / 完美匹配等边界

- **feat(vision/data/metrics): 沉淀通用 2D detection 能力**
  - `src/vision/preprocess`：通用 letterbox（保比缩放 + 填充）+ `LetterboxResult::to_origin` 反向映射 + `image_to_nchw_normalized` 归一化
  - `src/vision/detection`：`BBox` 统一封装 + `BoxFormat::{XyXy, CxCyWh}` 显式互转，`iou` / `giou` / `diou` / `ciou` 全家桶，`scale_translate` / `horizontal_flip` 几何变换，`Detection` / `GroundTruthBox` 标准载体，支持坐标契约、像素 / 归一化互转、clip/filter、score threshold、pre-NMS top-k、max detections 与 batch NMS
  - `src/vision/preprocess`：letterbox 扩展到矩形输出，补充 bbox 与 letterbox / 原图坐标互转，NCHW 归一化支持矩形输入
  - `src/data/detection`：`DetectionSample` / `DetectionBatch` 处理变长检测标签，并补充 letterbox、restore、horizontal flip、clip/filter 等 bbox 同步变换 helper
  - `src/data/datasets/yolo`：YOLO `.txt` 标签解析（支持空行、行内 `#` 注释、错误行 [文件:行号] 定位）
  - `src/metrics/detection`：mAP / Precision / Recall / F1 复用 `vision::detection::BBox`，预置 `VOC_IOU_THRESHOLDS`（`mAP@0.5`）与 `COCO_IOU_THRESHOLDS`（10 点 0.5..=0.95），新增 `DetectionMetricOptions`、per-class AP、per-threshold AP、score threshold 与 max detections 评估协议
  - `src/vision/detection/iou_loss`：新增 `BBoxLossKind::{IoU, GIoU, DIoU, CIoU}`（移到 `vision::detection`，更贴近其语义层级）与 `VarLossOps::{bbox_loss, giou_loss, diou_loss, ciou_loss}`，支持 `[N, 4]` 已匹配检测框回归训练；用基础算子拼接（`Maximum / Minimum / ReLU / Square / Atan2 / Sign / Mean`）+ autograd 自动反向，避开 fused 节点手推解析梯度的工程成本，可视化层面用 `NodeGroupContext` 折叠成单个 IoULoss / GIoULoss / DIoULoss / CIoULoss cluster；新增 `Atan2` 可微节点（CIoU 角度差所需）+ `tests/bbox_loss_reference.py` PyTorch oracle + 18 个对照测试（`epsilon = 1e-5` 逐元素一致）
  - `src/nn/detection_loss`：新增 `DetectionLossComponents` / `DetectionLossWeights`，作为 adapter 组合 bbox / objectness / class loss 的通用积木，不内置 YOLOv5 anchor/grid matching
  - 配套单元测试覆盖格式互转、IoU 家族数值、坐标契约、几何变换、NMS、指标选项、同步标签变换与 detection loss 组合；YOLO label 解析覆盖正/异常路径
- **feat(nn): `RebuildResult` 推理便捷 API**
  - 新增 `predict(input)` / `predict_head(name, input)` 一行调用，免去 `set_value → forward → value` 三步
  - 新增 `input_by_name(name)` / `output_by_name(name)` 按名取节点；多 input/output 场景默认走第一个，并补充按名访问
  - 新增 `tests/model_save.rs::test_rebuild_result_predict_uses_first_input_and_output` / `test_rebuild_result_predict_head_reports_missing_output` 覆盖默认与按名访问行为
- **bench(devops): ONNX 直接载入 vs OTM 中转载入决策 bench**
  - 新增 `tests/onnx_otm_load_bench.rs`，对比冷启动耗时、磁盘体积、参数保真度、推理速度，定位为低频"决策性 bench"，独立于 Criterion 回归体系（不进 `bench-save` / `bench-compare`）
  - `justfile` 新增 `bench-onnx-vs-otm`，跑法 `just bench-onnx-vs-otm`
  - 真机数据：OTM 文件大 ~1.9%，冷启动未优于 ONNX 直载；补充 missing parameter diff 观测，当前 VinXiangQi round-trip 参数名集合一致（135 → 135）
- **chore(devops): 新增保留 benchmark 结果的构建缓存清理命令**
  - `justfile` 新增 `clean-cache`，用于清理 `target/debug/incremental`、`target/debug/examples`、`target/release`、`target/ra` 等大体积可重建产物
  - 保留 `target/criterion` 与宏基准导出结果；`just clean` 仍保持 `cargo clean` 的彻底清理语义
- **feat(bench): 建立 benchmark 可观测性工作流**
  - 新增 `smoke`、`pool2d`、`optimizer`、`normalization` 四组 Criterion benchmark，覆盖快速回归、Pool2d、优化器和归一化层关键路径
  - 补齐 `loss`、`rnn`、`attention` 三组 focused benchmark，覆盖 Loss、循环层和 MultiHeadAttention 的 forward + backward 路径
  - `justfile` 新增 `bench-smoke`、`bench-save`、`bench-compare`、`bench-macro`、`bench-macro-core`，支持改动前保存 baseline、改动后对比和 release example 宏基准
  - 文档补充性能验证标准流程，并已保存 `Mode` 重构前 `pre-execution-context` Criterion baseline
- **feat(vision/detection): 立检测任务接口契约**
  - 新增 `vision/detection/contract.rs`：`Backbone` / `BackboneOutput`、`DetectionHeadDecode`、`Assigner<P>` / `AssignmentResult`
  - 契约比实现先行——本次只立类型约定（不写第一个 backbone 实现），避免后续 example 各自发明互不兼容的 head / assignment 接口
- **feat(data/transforms): 引入 `SampleTransform` 与 image+label 同步变换**
  - 新增 `SampleTransform<S>` trait，与 image-only `Transform` 正交；适用于 detection / segmentation 训练时让 image 与 bbox / mask 几何同步
  - 新增 `ClassificationSample` / `SegmentationSample` 数据载体（`DetectionSample` 复用 `data::detection`）
  - `RandomHorizontalFlip` / `CenterCrop` / `RandomCrop` 各自为三种 Sample 类型补齐 `SampleTransform` 实现：detection 路径自动同步 bbox（hflip 用 `image_w - x`，crop 用平移 + 与 crop window 求交集 + filter 过小框）；segmentation 路径同步翻转 / 裁剪 mask
  - `CenterCrop` / `RandomCrop` 新增 `with_label_filter(DetectionLabelFilter)` builder，控制 detection bbox crop 后的最小面积过滤
  - 新增 `data/transforms/crop_helpers.rs` 抽出 image-only / paired 共用的 crop / pad / bbox-shift / clip-filter 逻辑，避免重复
  - 配套 5 个 paired hflip 测试 + 7 个 paired crop 测试覆盖 cls / det / seg 三档；image-only 老测试保持兼容
- **feat(data/transforms): 补齐 `SampleTransform` 实现矩阵（Rotation / Affine / Erasing）**
  - `RandomRotation` / `RandomAffine`：为三种 Sample 补齐 paired 实现——detection 路径把 bbox 4 个角点按正向变换后取 AABB，再经 `clip_filter_labels` 裁到图像边界并过滤过小框；segmentation 路径 mask 改用**最近邻**插值（避免 bilinear 把离散类别混出非法中间值）；新增 `with_label_filter(DetectionLabelFilter)` builder 控制最小面积过滤
  - `RandomErasing`：按 torchvision v2 的 A 方案——**只擦 image，labels / mask 保留**，保留"训练抗遮挡能力"的意图；`sample_erase_window` 把"概率掷骰 + 采样窗口"封成一次返回 `Option`，三档 sample 共用同一路径
  - 新增 `data/transforms/affine_kernel.rs`：`AffineParams` + `affine_bilinear` / `affine_nearest` / `affine_bbox` 三个纯函数；`RandomAffine::apply` 重构为 "sample_params → kernel" 两步，为 paired 路径让 image / mask / bbox 共用同一组随机采样（否则随机性各自独立，paired 不再 paired）
  - `transforms/mod.rs` 顶部 rustdoc 更新为完整的 `SampleTransform` 实现矩阵（6 个 transform × 3 档 Sample）
  - 配套 26 个新 paired 测试（9 rotation + 10 affine + 7 erasing）覆盖 identity 短路、纯旋转 / 纯缩放 / 纯平移数学、mask nearest 离散类别保持、Erasing A 方案下 labels / mask 必须保留等断言

### Changed

- **refactor(vision/detection)!: bbox loss 从 fused 节点迁到拼接式 helper（Breaking, autograd composition）**
  - 删除 `src/nn/nodes/raw_node/loss/bbox.rs`（fused `BBoxLoss` + 有限差分梯度）以及 11 处引用：`NodeTypeDescriptor::BBoxLoss` variant、`create_bbox_loss_node` 节点构造器、`descriptor_rebuild` 分支、ONNX `TrainingOnly` 分类、evolution `node_gene` shape 推断、`var/descriptor` 分支、`raw_node` re-export；旧 `src/nn/tests/node_bbox_loss.rs` 整文件移除
  - 新增 `Atan2` 可微节点（CIoU 角度差所需）：覆盖前向 + 解析梯度 $∂out/∂y = x / (x² + y²)$ / $∂out/∂x = -y / (x² + y²)$，`(0, 0)` 处梯度 fallback 为 `0`（**与 PyTorch 的 `NaN` 不一致**——避免污染下游训练，特别是 CIoU 退化样本）；descriptor / rebuild / ONNX `Unsupported`（ONNX 缺原生 op）/ evolution / node_builders 全链路注册；含 6 个 known-value 单测
  - 新增 `src/vision/detection/iou_loss.rs`：把 `BBoxLossKind`（从 `nn::nodes::raw_node::loss` 搬到 `vision::detection`，与其语义所属模块对齐）+ 4 套 IoU 损失统一收口；用 `Maximum / Minimum / ReLU / Square / Atan2 / Sign / Mean` 基础算子拼接，autograd 自动反向；CIoU 零面积退化复用 ReLU 已保证非负的特性，用 `Sign` 直接得到 `w > 0` 的运行时硬 mask（梯度恒 0），不引入 `DEGENERATE_EPS` 数值阈值，行为严格对齐 `BBox::ciou` 的 `w/h <= 0` 判断；`NodeGroupContext` 把 30+ 内部节点折叠成单个 cluster，`.dot` 可视化与原 fused 节点形态等价
  - `Var::bbox_loss / giou_loss / diou_loss / ciou_loss` 直接 delegate 到拼接式 helper；helper 入口显式 `target.detach()`，target 即便是 Parameter 也不会反向收梯度（fused 时代由节点 hard-reject 实现）；`f32 - Var` 标量减法运算符重载补齐
  - 测试体系：`tests/bbox_loss_reference.py` 用 PyTorch autograd 在 8 组典型样本（含部分重叠 / 对角偏移 / pred 包住 target / aspect ratio 错配 / CxCyWh 中心偏移 / CxCyWh aspect 改变 / N=2 batch / 完全不重叠）上算 4 套 IoU 的 forward + backward oracle，输出 Rust 常量；新建 `src/vision/tests/bbox_loss_composed.rs` 取代旧 `src/nn/tests/node_bbox_loss.rs`，18 个 test 覆盖 forward known-value（手算 IoU=2/3、4/3）+ 8 组 backward 对照 PyTorch oracle（`epsilon = 1e-5` 逐元素一致，相比旧 `2e-3` 有限差分容忍收紧 200×）+ shape 严格相等（拒绝 `[3, 4]` vs `[1, 4]` 隐式 broadcast）+ target Parameter 无梯度 + CIoU 零面积退化 → DIoU 等价 + CIoU 极小正宽（1e-8）保留 aspect penalty（防 epsilon 阈值化误判）+ 4 套 IoU 的 NodeGroupTag 可视化分组（遍历 `backward_topo_order` 内部节点逐个断言 group_type / instance_id 一致）
  - 文档同步：`architecture_roadmap.md` 节点统计 `BBoxLoss` 移除（损失 6→5、合计 75→74，新增 Atan2 算术 +1 → 75）；`node_vs_layer_design.md` 损失行同步；`spatial_vision_tasks_roadmap.md` 删除"`bbox_loss` 反向传播仍使用有限差分"caveat
  - 归档 `.issue/_archive/bbox_loss_analytical_grad.md`（resolved，记录拼接式 + autograd 的 final 闭环路径）

- **rename: tests/ 集成测试与中国象棋 example 命名规范化**
  - **tests/ 去 `test_` 前缀**(向同目录已有的 `onnx_otm_load_bench.rs` / `yolov5_xiangqi_import.rs` 看齐):`tests/test_cse_dedup.rs` → `tests/cse_dedup.rs`、`tests/test_mode_invariants.rs` → `tests/mode_invariants.rs`;`mode_invariants.rs` 内部 `target/test_mode_invariants` 临时目录跟着改成 `target/mode_invariants`;同步 `.doc/design/mode_design.md` / `src/nn/tests/{graph_handle,gradient_flow_control}.rs` 共 4 处链接
  - **chess example 改名为 `chinese_chess_*`,且把 task / 模型版本写清楚**:旧名 `chess_yolo_onnx_detect` 看不出"做的是空间域 detection 还是棋局识别(recognition)"且 `yolo` 不带版本号;新名按"领域 + 模型(带版本) + 互通方式 + 核心动作"四段式表达
    - `chess_yolo_onnx_detect` → `chinese_chess_yolov5_onnx_recognize_fen`(YOLOv5 检测 + ONNX 互通,产出 FEN 字符串,跟 example 内部已统一使用的 `board_align::recognize` 函数对齐)
    - `chess_cnn_onnx_finetune` → `chinese_chess_cnn_onnx_finetune`(只补 `chinese_` 前缀,CNN+finetune 已准确表达 task)
    - 影响 18 个文件 60+ 处引用:Cargo.toml + justfile(task 名 / bench-macro / examples-traditional 聚合)+ 两个 example 内部所有路径常量与文档 + `tests/onnx_models/yolov5_xiangqi/{README,export.py}` 转发链接 + 主 README 概览表 / 特性矩阵 / ONNX 互通章节 + `.doc/design/{onnx_import_strategy,spatial_vision_tasks_roadmap}.md` + `.doc/optimization_candidates.md` + `src/nn/tests/node_max_pool2d.rs` SPPF 注释
    - 运行时数据 / 模型路径(`data/chess_cnn_onnx_finetune/` → `data/chinese_chess_cnn_onnx_finetune/`、`models/chess_cnn_onnx_finetune.otm` 同改)同步迁移,以保证 example 内部 `DATA_DIR` / `OTM_PATH` 与 `generate_data.py` / `train_pytorch.py` 默认输出目录一致
  - **新增命名规范文档**:`.doc/terminology_convention.md` §七「外部模型与版本命名」沉淀长期规则——跟随上游官方 / 论文标题写法(YOLO 用 `YOLOv5` / `yolov5` 紧贴 `v`,MobileNet 用 `mobilenet_v2` 下划线分隔,各取所长不强行内部一致),含常见模型族(YOLO / MobileNet / EfficientNet / ResNet / BERT / GPT / LLaMA)速查表
  - **故意保留**:CHANGELOG 历史条目(时光胶囊属性,记录改名前的旧名)+ `src/nn/graph/onnx_import/mod.rs:43` 注释里的 plan 文件名引用(plan 文件已不在仓库,作为历史工作代号保留)
  - 验收:`cargo check` + `cargo test --lib` 2970/2970 全过 + 4 个集成测试 binary 全部被 cargo 正确识别(cse_dedup / mode_invariants / yolov5_xiangqi_import / onnx_otm_load_bench)+ `cargo check --example chinese_chess_yolov5_onnx_recognize_fen --example chinese_chess_cnn_onnx_finetune` 通过

- **fix(data/transforms): `RandomErasing` image-only 路径放宽 shape 支持**
  - 原 `RandomErasing::apply` assert 输入必须是 3D `[C, H, W]`——这是早期遗留限制而非设计决策；放宽后同时接受 2D `[H, W]` 灰度图像，与 paired 路径（`SampleTransform`）行为一致，也与 torchvision 对齐
  - 现有 3D 测试不受影响；新增 `test_erasing_supports_2d_grayscale` 锁住新支持
- **refactor(examples): 9 个空间域 example 改用库 API，共减重约 1100 行**
  - `traditional/single_object_detection`：删除本地 `tensor_rows_to_clipped_bboxes` / `draw_bbox` / `draw_hline` / `draw_vline` / `bbox_to_canvas_rect` / `tiny_text_*` / `put_pixel_checked` 等约 150 行可视化辅助；改用 `BBox::vec_from_tensor` + `vision::draw::draw_bbox`(DynamicImage 中转)+ `vision::viz::TinyFont::draw_with_box`
  - `traditional/single_object_segmentation` / `multi_instance_segmentation` / `overlapping_fixed_slot_instance_segmentation`：删除本地 `mask_row` / `fill_scaled_pixel` / `overlay` / `blend_channel` / `slot_pixel_accuracy` / `mean_instance_iou` / `mean_valid_slot_iou` / `empty_slot_accuracy` / `instance_iou` / `slot_color` 等，改用 `vision::mask` + `vision::viz` + `metrics::segmentation` + `vision::io::save_rgb_image`
  - `traditional/overlapping_shapes_semantic_segmentation` / `overlapping_shapes_unet_lite_segmentation`：删除本地 `argmax_class` / `foreground_probability_mask` / `foreground_target_mask` / `class_color` 等，改用 `vision::mask::{argmax_to_class_map, foreground_from_multiclass}` + `Palette::default_categorical()`
  - `traditional/deformable_conv2d_segmentation` 与 `evolution/{overlapping_shapes_unet_lite, deformable_conv2d}_segmentation`：用 `vision::viz::pixel_block_scale` 替换本地 `fill_scaled_pixel`；保留各自的连续概率→颜色或 binary_color 等任务特化映射
  - `traditional/chess_yolo_onnx_detect`：**整个 `yolo_decode.rs`(98 行)删除**，改用 `vision::detection::adapter::yolo::v5::detect`
  - 行为不变：所有 example 在重构前后产出相同的 mean IoU / Pixel Accuracy / Mean Instance IoU / Empty-slot Accuracy 数值

- **refactor(examples/shared): 跨 traditional / evolution 共享合成形状数据集生成**
  - 新增 `examples/shared/synthetic_shapes.rs`(99 行)：统一封装 R/C/T 三类形状的 `ShapeKind` / `ShapeObject` / `contains` / `generate_objects`；class_id 由 kind 派生(Rectangle=1, Circle=2, Triangle=3)，参数 `(image_size, max_objects)` 由调用方显式传
  - `traditional/{overlapping_shapes_semantic, overlapping_shapes_unet_lite}_segmentation` 与 `evolution/overlapping_shapes_unet_lite_segmentation` 通过 `#[path = "../../shared/synthetic_shapes.rs"] mod synthetic_shapes;` 跨大类引用，消除约 245 行 100% 复制粘贴
  - 与 shared 不一致的 example(`evolution/overlapping_shapes_semantic_segmentation` 用 R/C 两类 + 随机 class_id；`deformable_conv2d_segmentation` 用 16x16 + small range)保留各自本地 helper，避免过度抽象

- **feat(example): DeformableConv2d evolution 示例新增审计矩阵**
  - `examples/evolution/deformable_conv2d_segmentation` 保持默认 deformable-only smoke 路径，用于验证算子进入 evolution 主流程
  - 新增 `ONLY_TORCH_EVOLUTION_DEFORMABLE_SEG_AUDIT=1`，对比 deformable-only、默认 segmentation portfolio、`vision_segmentation_with_deformable()`、heuristic 开关与小幅预算提升
  - 示例输出全测试集 PixelAccuracy、BinaryIoU min/mean/max、Dice mean，并保存最差 IoU 样本图片，避免单样本或低阈值误导判断
  - 当前矩阵显示默认 dense segmentation 族明显优于 deformable-only，因此暂不把 DeformableConv2d 提升为默认 heuristic family
- **refactor(example): chess_yolo_onnx_detect 大幅精简至库版 pipeline**
  - `main.rs` 393 → 162 行：去掉手工 ONNX → Graph 双步、各阶段计时、`ImportReport` 打印、`num_classes` 手算、五段式后处理调用，改为 `Graph::from_onnx → RebuildResult::predict → yolo_decode::detect → board_align::recognize` 三步
  - `examples/traditional/chess_yolo_onnx_detect/letterbox.rs` 删除，整体迁移到 `src/vision::preprocess`
  - `yolo_decode.rs` / `board_align.rs` 复用库内 `Detection` / `LetterboxResult` / `nms`，新增 `detect()` / `recognize() -> BoardOutput` 端到端封装
  - `examples/traditional/chess_yolo_onnx_detect/README.md` 同步到库版 pipeline 描述

- **refactor(graph)!: 用 `Mode { Train, Inference }` 统一执行上下文（Breaking）**
  - 删除 `ExecutionContext { training, grad_enabled }` 双字段；新 `Mode` 一个枚举同时承载层行为切换、backward 缓存策略与 backward 是否被允许，详见 `.doc/design/mode_design.md`
  - `Inference` 模式下 `Graph::backward()` / `Var::backward()` 现在会在 ensure-forward 前直接返回 `GraphError::InvalidOperation`，不再降级为警告
  - `forward_recursive` 与 `TraitNode::set_mode` 改为按值传递 `Mode`；新增 12 个重缓存节点（Softmax / LogSoftmax / LayerNorm / RMSNorm / Abs / Square / Pow / Clip / Reciprocal / Ln / Log2 / Log10）实现按 mode 切缓存
  - `Graph::inference_scope()` / `GraphInner::inference_scope()` 改为 panic-safe 恢复：闭包 panic 后若被上层捕获，图 mode 仍恢复到进入前状态
  - Evolution `Trainer::predict_*` / `EvolutionTask::evaluate` 自动进入 `Inference` 模式，候选评估期间不再重复保留 backward 缓存；`evaluate` 结束或出错后恢复进入前的 mode
  - `tests/test_execution_context_invariants.rs` 整文件迁移到 `tests/test_mode_invariants.rs`，旧 `gradient_flow_control_design.md` 归档至 `.doc/_archive/`，新文档落到 `.doc/design/mode_design.md`；新增 `mode_cache` 测试覆盖重缓存节点的 Inference 跳缓存契约
  - **API 迁移表**：

| 旧 API | 新等价物 |
|---|---|
| `ExecutionContext::training()` | `Mode::Train` |
| `ExecutionContext::inference()` | `Mode::Inference` |
| `Graph::eval()` | `Graph::inference()` |
| `Graph::training()` | `Graph::is_training()` |
| `Graph::set_train_mode()` | `Graph::train()` |
| `Graph::set_eval_mode()` | `Graph::inference()` |
| `Graph::is_train_mode()` | `Graph::is_training()` |
| `Graph::is_grad_enabled()` | `Graph::is_training()` |
| `Graph::execution_ctx()` | `Graph::mode()` |
| `Graph::set_execution_ctx(ctx)` | `Graph::set_mode(mode)` |
| `Graph::no_grad_scope(\|g\| ...)` | `Graph::inference_scope(\|g\| ...)` |
| `TraitNode::set_execution_ctx(&ctx)` | `TraitNode::set_mode(mode)` |

- **perf(nn): 优化 Conv2d Debug 推理路径**
  - `Conv2d` 在 `Inference` 推理模式下对 `1x1 stride=1 padding=0` 卷积启用直接 GEMM 快路径，避免为 backward 生成无用 `im2col` 缓存
  - `Conv2d` padding 与 `im2col` 热循环改用连续 slice 索引，减少 Debug 模式下动态 Tensor 索引开销
  - `chess_yolo_onnx_detect` Debug forward 从约 1871 ms 降到约 596 ms，总耗时从约 2030 ms 降到约 745 ms，并保持两张 sample 的 FEN 位级匹配
- **refactor(vision)!: vision 模块按职能重组（Breaking）**
  - `vision/` 子模块从"`Vision::xxx` impl method + 裸 Tensor"切换到"模块函数 + `&DynamicImage` 强类型"，对齐 `torchvision.transforms.functional` / `torchvision.utils` / `torchvision.io`：
    - 新增 `vision/io.rs`（`load_image` / `save_image`）、`vision/color.rs`（`to_luma`）、`vision/geom.rs`（`resize_exact` / `resize_keep_ratio` / `center_crop`）、`vision/filter.rs`（`median_blur`）
    - 新增 `vision/draw.rs`（`draw_bbox` / `draw_circle` / `draw_rectangle_xyxy`，接收 `BBox` 等强类型 + `&mut DynamicImage`，断言可验证像素而非"保存到 temp 再删"）
    - 新增 `vision/cv/` 子模块收纳传统 CV 算法（Hough 圆检测等，与 PyTorch / JAX 不收录的范畴对齐）
  - **删除** `vision/detect.rs`（迁 `vision/cv/hough_circles.rs`）、`vision/process.rs`（重写为 `vision/filter.rs`）、`vision/shape.rs`（重写为 `vision/geom.rs`）
  - **删除** `pub struct Vision` 命名空间与 `ImageBufferEnum`：`Vision::load_image / save_image / to_luma` 全部迁到对应模块函数
  - `Tensor::to_luma` 内部不再调 `Vision::to_luma`，改走 `vision::color::to_luma`；外部 API 不变
- **refactor(vision/detection)!: detection 任务级 helper 收口到 `vision/detection/`（Breaking）**
  - `nn/detection_loss.rs` → `vision/detection/loss.rs`：`DetectionLossWeights` / `DetectionLossComponents` 现从 `crate::vision::detection::` 引入（不再从 `crate::nn::`）
  - `data/detection.rs` 拆分：数据载体 `DetectionSample` / `DetectionBatch` 留 `data/`；label 几何变换 `letterbox_labels` / `restore_letterbox_labels` / `horizontal_flip_labels` / `clip_filter_labels` 与 `DetectionLabelFilter` 迁 `vision/detection/transform.rs`
  - `data/datasets/yolo.rs` → `vision/detection/io.rs`：`parse_yolo_txt_file` / `parse_yolo_txt_labels` 现从 `crate::vision::detection::` 引入
- **refactor(metrics)!: detection 指标 API 升级到强类型（Breaking）**
  - **删除** `mean_box_iou_cxcywh(&Tensor, &Tensor)`，新增 `mean_box_iou(&[BBox], &[BBox])`；调用方需要先做 `Tensor → Vec<BBox>` 转换。整个 detection 系统现在统一只走 `BBox::iou`，不再维护"裸 Tensor 配对 IoU"代码路径
  - `single_object_detection` example 同步适配

### Fixed

- **fix(onnx): 修复 Resize roi initializer 被误导入为参数**
  - ONNX `Resize` 的 `roi` / `scales` / `sizes` 统一标记为 metadata consumed，避免非空 `roi` initializer 被注册成不参与输出路径的死 `Parameter`
  - VinXiangQi YOLOv5 `Graph::from_onnx` → `Graph::save_model` → `Graph::load_model` 参数数量已从 137 → 135 的异常收口为 135 → 135，并在 bench 中保留缺失参数名 / shape / origin diff
- **fix(onnx): 补齐 BatchNormalization 导入语义与 BatchNorm 状态持久化**
  - ONNX `BatchNormalization(X, scale, B, mean, var)` 导入时展开为确定推理算术子图，不再错误映射成训练态 `BatchNormOp`
  - `BatchNormOp` 增加 `eps`、`momentum`、running stats 形状校验和单样本训练态报错，避免 running variance 被无效统计污染
  - `.otm` / `GraphDescriptor` 保存并恢复 BatchNorm `running_mean` / `running_var`，ONNX 导出侧拒绝生成缺少 scale/bias/mean/var 的不完整 BatchNormalization
- **fix(onnx/evolution): 收口卷积 bias 导入导出与演化识别**
  - ONNX `Conv` / `ConvTranspose` 导入时将一维 bias 自动升维为内部 `[1, C, 1, 1]` 广播参数，导出时把安全的 `Conv + Add(bias)` 合并为标准三输入卷积
  - Evolution 的 Conv2d block 分解和通道 resize 改为通过 `Add(conv, bias)` 父边识别真实 bias，避免同 block 其它参数被误判或误改
  - 补充 ConvTranspose bias 拆分、ONNX bias 融合、无 bias Conv2d 保存加载和 FM 分解误判防护测试
- **fix(onnx): 修复 YOLOv5 `Pow` 常量指数导入与数值漂移**
  - ONNX `Pow(base, exponent)` 导入时读取 Constant / initializer 标量 exponent，并折叠为 only_torch `Pow { exponent }` 属性，避免 `x^2` 被误导成默认 `x^1`
  - VinXiangQi YOLOv5 增加 raw output 与 ORT 对照、逐节点中间张量 drift 诊断和最小 `Pow` 常量指数回归测试；strict 数值门下 raw output `max_abs` 已降至 `5.19e-4`
  - 更新 Chinese YOLOv5 示例与 fixture 文档，明确真实模型不含独立 `BatchNormalization`，NMS 数量差异不再被当作算子根因

## [0.17.0] - 2026-04-29

### Fixed

- **fix(nn): 修复 DeformableConv2d 动态 batch 与 BinaryIoU 评估**
  - `DeformableConv2d` 前向 / 反向改为读取运行时 batch size，修复构图期 batch=1 时评估 `[N,C,H,W]` 仍输出单样本的问题
  - Evolution `TaskMetric::BinaryIoU` / `ReportMetric::BinaryIoU` / `Dice` / 二值 `PixelAccuracy` 在 BCE logits 下先按 `>=0.0` 解码预测，再按 0.5 阈值对齐 0/1 标签，避免标签 0.0 被误判为正类
  - 补充 `BinaryIoU` batch、`DeformableConv2d` runtime dynamic batch、BinaryIoU + Deformable evolution 评估回归测试

### Added

- **feat(example): 新增 Overlapping Shapes U-Net-lite 分割强基线**
  - 新增 `examples/traditional/overlapping_shapes_unet_lite_segmentation`，复用 64x64 overlapping shapes 语义分割数据与 Mean IoU / Dice / per-class IoU 指标
  - 模型采用 `Conv2d -> MaxPool2d -> ConvTranspose2d -> Var::concat(axis=1)` 的轻量 encoder-decoder + skip connection 结构，作为后续 Segmentation Evolution 扩大 benchmark 的传统对照
  - 注册 `cargo run --example overlapping_shapes_unet_lite_segmentation` 与 `just example-overlapping-shapes-unet-lite-segmentation`；debug + BLAS 下约 27.4 秒达到 Mean IoU 75.6%
- **feat(evolution): 新增 U-Net-lite benchmark 对齐的分割演化示例**
  - 新增 `examples/evolution/overlapping_shapes_unet_lite_segmentation`，使用同类 64x64 / 4 类 / 0..3 个可重叠形状数据，作为 U-Net-lite 强基线的 evolution 对照
  - 示例默认 segmentation portfolio 已纳入 U-Net-lite encoder-decoder + skip concat 初始族；输出 input / target / prediction 可视化，避免把类别颜色误读成实例 ID
  - 注册 `cargo run --example evolution_overlapping_shapes_unet_lite_segmentation` 与 `just example-evolution-overlapping-shapes-unet-lite-segmentation`；默认 target Mean IoU 提升到 0.60，并新增 `ONLY_TORCH_EVOLUTION_UNET_LITE_SEED`、`ONLY_TORCH_EVOLUTION_UNET_LITE_TARGET`、`ONLY_TORCH_EVOLUTION_UNET_LITE_SAVE_ARTIFACTS=0` 便于稳定性复核
- **feat(evolution): 支持固定多头 supervised evolution 第一阶段**
  - 新增 `SupervisedSpec` / `HeadSpec` 显式入口；旧 `Evolution::supervised(...).with_*().run()` 链式写法保持兼容并自动包装为单 head supervised task
  - `NetworkGenome` 记录命名 `OutputHead` 元数据，`BuildResult` 新增 `outputs: Vec<Var>` 并保留默认 `output: Var`，支持共享 trunk + 多个物理输出 head
  - 多头训练按 head 创建 target/loss 并用 `loss_weight` 聚合；评估生成逐 head `HeadMetricReport`，`FitnessScore.primary` 默认取 primary head
  - `EvolutionResult` 新增 `predict_head` / `predict_heads`，`.otm` 保存/加载、可视化和 ONNX 导出路径改为使用所有输出 head；当前阶段限定为平坦共享输入、固定 head 数量，detection matching / NMS / mAP 留待后续
- **feat(example): 新增多头 supervised evolution 示例**
  - 新增 `examples/evolution/multi_head_quadrant_radius`，用二维点共享输入同时训练 `quadrant` 四分类 head 与 `radius` 回归 head
  - 示例覆盖 `SupervisedSpec::head_targets(...)`、逐 head metric report、`predict_head` / `predict_heads` 选择性推理和 `.otm` 保存/加载
  - 注册 `cargo run --example evolution_multi_head_quadrant_radius` 与 `just example-evolution-multi-head-quadrant-radius`
- **feat(nn/evolution): 新增 offset-only DeformableConv2d 通用算子**
  - 新增 `NodeTypeDescriptor::DeformableConv2d`、raw node 前向 / 反向、descriptor rebuild 与 ONNX unsupported 标记，并补 PyTorch / torchvision 数值对照测试
  - 新增 `DeformableConv2d` Layer，offset predictor 初始为零，使传统手写网络可直接使用该算子
  - Evolution NodeLevel 新增 DeformableConv2d block 展开、形状 / FLOPs 推导和 segmentation InsertLayer 最小接入
- **feat(example): 新增 DeformableConv2d 传统分割示例**
  - 新增 `examples/traditional/deformable_conv2d_segmentation`，使用 16x16 多形状二值前景分割数据展示 `Conv -> DeformableConv2d -> Conv -> 1x1 head` 手写网络基线
  - 示例输出 `test_in.png` / `test_out.png` 和计算图 `.dot` / `.png`，`test_out.png` 以绿色热力图展示 foreground 概率
  - 注册 `cargo run --example deformable_conv2d_segmentation` 与 `just example-deformable-conv2d-segmentation`
- **feat(example): 新增 DeformableConv2d 分割演化示例**
  - 新增 `examples/evolution/deformable_conv2d_segmentation`，使用 16x16 二值前景分割数据验证 DeformableConv2d seed 进入 evolution 主流程
  - `InitialPortfolioConfig` 新增 `include_deformable_tiny` 与 `vision_segmentation_with_deformable()`，并新增 `spatial_segmentation_deformable_tiny` 初始基因组
  - 示例关闭 P5-lite learned / heuristic 预筛路径，默认 seed=42 在 4 个测试样本上达到 Binary IoU 40.3%，最终基因组包含 DeformableConv2d
  - 注册 `cargo run --example evolution_deformable_conv2d_segmentation` 与 `just example-evolution-deformable-conv2d-segmentation`

### Changed

- **feat(evolution): 收敛 MNIST 默认演化搜索路径**
  - 空间分类任务默认启用初始候选族、family-diverse P5-lite、ASHA 多样性保护、final refit、FLOPs 上限和合适的 batch / population 设置，用户侧不再需要手动选择 `smoke / quality / audit / search` profile
  - `examples/evolution/mnist` 删除 profile 分层，示例收敛为 `Evolution::supervised(...).with_target_metric(0.95).run()`，默认仍输出最新可视化图
  - 新增 `ONLY_TORCH_MNIST_SEED` 与 `ONLY_TORCH_MNIST_SAVE_ARTIFACTS=0`，用于多 seed 稳定性复核；默认路径 5 个 seed 全部达到 95% 准确率
- **feat(evolution): 迁移 P5-lite 审计到 Segmentation evolution**
  - segmentation 默认启用最小分割头、`spatial_segmentation_tiny` 与 `spatial_segmentation_unet_lite` 初始候选族，并接入 family-diverse P5-lite 预筛
  - 候选族统计改为通用计数容器；`dense_seg_head` / `dense_seg_deep` / `encoder_decoder_seg` 作为内部候选族继续服务 `eval-family` 与 `p5-lite-family` 观测
  - segmentation Phase 1/2 注册 `InsertEncoderDecoderSkipMutation`，一次性插入 `Pool2d -> Conv2d -> ConvTranspose2d -> Concat(skip) -> Conv2d`，让 U-Net/FPN 风格结构能通过 mutation 进入搜索
  - `NodeBlockKind` 补齐 `ConvTranspose2d` 识别与参数维度修复，`SkipAgg` 的通道数级联改为读取实际推导形状，避免 concat 后 fuse conv 输入通道被误修回单分支通道
  - `evolution_overlapping_shapes_unet_lite_segmentation` 在 target Mean IoU 0.60 下完成 5-seed 稳定性验证：seed 1..5 全部 `TargetReached`，Mean IoU 为 93.3% / 98.4% / 90.3% / 77.1% / 63.0%
  - `loss_var.backward()` 计时拆分为 `backward_total`、`backward_forward`、`backward_propagate`，BCEWithLogits 前向改为单次扫描生成 sigmoid 缓存与稳定 loss，并移除 target 缓存 clone
  - `Conv2d` forward 对 padding 为 0 的 1x1 / valid conv 不再深拷贝输入作为 `padded_input`，减少 dense segmentation head 的无效内存拷贝
  - `evolution_overlapping_shapes_semantic_segmentation` 示例移除默认强制 verbose 审计日志；本轮 debug + BLAS 最新单次复验约 2.9 秒达到 Mean IoU 63.0%

## [0.16.0] - 2026-04-28

### 修复

- **fix(nn): GraphDescriptor 加 `explicit_output_ids`,精确还原 ONNX `graph.output`** [`761e4e8`]
  - **真根因**:`from_descriptor` 用"无后继 = 输出"拓扑推断,复杂模型(YOLOv5)经过常量折叠 + Split 重写后会留下若干无后继的中间节点(常量 Parameter / 拆出的 Narrow 副本等),它们都被误当成输出节点 → main.rs 拿到 3 个输出且选错(VinXiangQi 实际只有 1 个 `output` shape `[1, 25200, 20]`)
  - **修复**:`GraphDescriptor` 加 `explicit_output_ids: Option<Vec<u64>>` 字段;ONNX import 从 `graph.output` 名称取 ID 列表填到 `descriptor.explicit_output_ids`;`descriptor_rebuild` 优先用此列表,fallback 到原拓扑推断(演化/手写 Layer 等内部路径不变)
  - **回归门**:`tests/yolov5_xiangqi_import.rs::yolov5_xiangqi_rebuild_succeeds` 加 `outputs.len() == 1` + `inputs.len() == 1` 断言;原冗余的 `forward_outputs_correct_shape` 测试删除(由 example 的 FEN 自动对比兼任,更强)
  - 验收:`cargo test --lib` 3105/3105;example chinese_chess_yolo 端到端跑通(下条)

- **fix(nn): 修复 chinese_chess_yolo spatial shape 传播,补齐 ONNX MaxPool padding/ceil_mode 语义** [`f88e0a7`]
  - **真根因**:`MaxPool2d` 不读 ONNX `pads` 属性,YOLOv5 SPPF 模块 (k=5, pads=2, s=1) 输出错算成 (20-5)/1+1=16,期望 20
  - 同时修复 Conv/ConvTranspose 解析的潜在 bug:`(pads[0], pads[2])` 把 H_end 当 W_begin,改为对称语义 `(pads[0], pads[1])`,非对称四角报 actionable 错(提示 ZeroPad2d / onnxsim)
  - **MaxPool2d 加 padding (4 维) + ceil_mode 字段**:`#[serde(default)]` 兼容旧 .otm,前向用虚拟 padding 避免污染 max,反向 unpad 还原原始坐标
  - **Constant 节点未被消费时建成 Parameter 节点**:保留数值常量(如 YOLOv5 头部 Mul 常数因子),修复下游 `resolve_parents` 找不到父节点 id
  - `infer_output_shape_placeholder` 给 Concat 沿 axis 累加、给 Permute 按 dims 重排:修复下游 Reshape `-1` 推导拿到错的 input total
  - `Parameter` 节点放宽维度上限,允许 5D+ 张量(YOLOv5 anchor 表 `[1, na, 1, 1, 2]` 等场景)
  - 11 个新单测(MaxPool SPPF padding / ceil_mode / backward + ONNX MaxPool with pads/ceil_mode + Conv 对称/非对称 pads + yolov5_xiangqi_rebuild_succeeds 集成回归)
  - **验收**:rebuild OK,参数量 140,3 个输出节点;evolution 模块 4 处 MaxPool 字面量同步适配

- **fix(onnx): 支持 PyTorch eval-mode 导出的 Conv with bias** [`40dc67d`]
  - 自动拆分 3 输入 Conv 为 `Conv2d + Add`
  - bias 形状自动从 `[1, C]` reshape 到 `[1, C, 1, 1]` 以正确广播

- **fix(flatten): 修复动态 batch 维度（`dim=0`）下的除零 panic** [`40dc67d`]

### 新增

- **feat(example): 中国象棋 CNN 示例改造为 ONNX 互通端到端流程** [`40dc67d`]
  - PyTorch 训练 → ONNX 导出 → only_torch 加载 → 继续训练 → `.otm` 保存/加载/验证
  - 文件迁移到 `examples/traditional/chess_cnn_onnx_finetune/`，后续示例重命名突出 ONNX 互通与继续训练能力
  - `train_pytorch.py` 新增 ONNX 导出步骤
  - 实测：基线 97.1% → 微调 5 epoch 后 97.8% → `.otm` 重载差异 0.00%

- **feat(graph): `RebuildResult` 新增 `parameters` 字段** [`40dc67d`]
  - 与 `inputs` / `outputs` 对称，加载完模型直接拿到可训练参数 `Var` 列表
  - 适用于 `Graph::from_onnx` / `Graph::load_model` 后接优化器的场景

- **feat(evolution): 彻底收口 NodeLevel-only genome 主路径**
  - `NetworkGenome::minimal*` 构造器直接生成 NodeLevel genome，不再依赖 LayerLevel → NodeLevel 迁移层
  - 删除 `migration.rs` 生产模块与旧迁移测试，保留的节点展开算法改名搬迁到 `node_expansion`
  - builder / mutation / gene / model_io / net2net 测试改为围绕 NodeLevel block、参数节点快照、跨层 connection 与可视化分组语义验证
  - 封装 evolution 内部 ASHA rung seed 派生，并修复 vision 并发测试覆盖共享 fixture、RandomAffine 测试精确浮点断言导致的完整测试偶发失败

- **feat(data/examples): 新增 `SyntheticRng` 统一合成数据可复现随机生成**
  - `data` 模块新增 public `SyntheticRng`，用于 examples / tests / synthetic dataset 的确定性伪随机数生成；模型参数初始化仍使用 `Graph::new_with_seed`，演化流程仍使用 `Evolution::with_seed`
  - 将传统与演化 examples 中手写的 `mix()` / `wrapping_mul` / `DefaultHasher` 数据生成逻辑统一迁移到 `SyntheticRng`，避免示例主路径暴露 hash mixing 常量与 debug 溢出风险
  - 补充 `SyntheticRng` 单元测试，覆盖同 seed 可复现、seed parts 派生、range 边界与 fork 不消耗父流

- **feat(vision/evolution): 补齐 Segmentation P1 benchmark 与单输出分割演化接入**
  - 新增 `overlapping_shapes_semantic_segmentation`：64x64 多形状、多对象、允许重叠的 visible semantic mask benchmark，报告 Pixel Accuracy、Dice、per-class IoU、Mean IoU
  - 新增 `overlapping_fixed_slot_instance_segmentation`：1..3 个可重叠实例、固定 slot、空 slot 与 visible mask 规则的 instance segmentation lite benchmark
  - `metrics` 补充 Dice、semantic pixel accuracy、per-class IoU、Mean IoU，并加入分割指标测试
  - Evolution 新增 `BinaryIoU` / `MeanIoU` primary metric、分割报告指标、`[C,H,W] -> [classes,H,W]` shape 协议与不经过 `Flatten` 的 spatial-to-spatial minimal genome
  - 新增 `evolution_overlapping_shapes_semantic_segmentation` smoke 示例和对应测试；后续已通过 segmentation portfolio、P5-lite 审计与 dense 前向优化闭环空间域 Evolution 慢路径

- **feat(metrics/example): 新增 Single Object Segmentation 传统示例**
  - `metrics` 新增 `pixel_accuracy` / `binary_iou` 二值分割指标，并补充空 mask、shape mismatch 等单元测试
  - 新增 `examples/traditional/single_object_segmentation`：内置固定 seed 的 16x16 合成矩形 / 圆形 mask 数据，小型 CNN 直接输出 `[N, 1, H, W]` logits，用 4D `BCEWithLogits` 快速训练
  - 示例注册到 `Cargo.toml` / `just example-single-object-segmentation` / README，并输出 `test_in.png`、`test_out.png` 便于直观看原图与预测 mask overlay

- **feat(metrics/example): 新增 Single Object Detection 传统示例**
  - `metrics` 新增 `mean_box_iou_cxcywh` 单目标 bbox IoU 指标，并补充完全重叠、部分重叠、不重叠、空 Tensor、shape mismatch 等单元测试
  - 新增 `examples/traditional/single_object_detection`：内置固定 seed 的 16x16 合成单矩形数据，小型 CNN 输出归一化 `[cx, cy, w, h]` bbox，用 Huber loss 快速训练
  - 示例注册到 `Cargo.toml` / `just example-single-object-detection` / README，并输出 `test_in.png`、`test_out.png` 展示原图与预测 bbox overlay

- **feat(example): 新增 Multi Instance Segmentation 传统示例**
  - 新增 `examples/traditional/multi_instance_segmentation`：内置固定 seed 的 16x16 合成图像，每张图恰好 2 个非重叠矩形实例，输出 `[N, 2, H, W]` 固定 slot mask
  - 小型全卷积网络 `Conv(1→8→8→2)` 用 4D `BCEWithLogits` 训练，示例内报告 Slot Pixel Accuracy 与 Mean Instance IoU
  - 示例注册到 `Cargo.toml` / `just example-multi-instance-segmentation` / README，并明确它是教学用固定两实例 toy 示例，不覆盖通用 Mask R-CNN / YOLO-seg 系统

- **feat(evolution): 收敛用户 API 到 NodeLevel-only 路线**
  - `evolution` 不再公开 `gene` / `builder` / `migration` / `mutation` 等内部模块，用户侧从顶层导入 `TaskMetric`、`ReportMetric`、`EvolutionCallback`、`ConvergenceConfig` 等任务级类型
  - 删除 `build_layer_level()` 逐层构图后端，`NetworkGenome::build()` 统一通过 NodeLevel → `GraphDescriptor` → `Graph` 构图
  - 层块插入变异统一命名为 `InsertLayerMutation` / `InsertLayer`，Linear / Conv / RNN / Dropout 等仍作为内部层规格搜索空间保留
  - examples 与演化设计文档同步改为任务 API + NodeLevel-only 表述，不再推荐用户理解或使用 LayerLevel 概念

- **feat(metrics/evolution): 通用指标补齐并接入演化评估报告**
  - `metrics` 新增回归误差指标：`mean_squared_error` / `mean_absolute_error` / `root_mean_squared_error`，与现有 `r2_score` 共用 `RegressionMetric` 接口
  - 演化侧新增 `ReportMetric` / `MetricValue` / `MetricReport`，`FitnessScore::report` 默认按任务类型报告 Accuracy/Precision/Recall/F1、R²/MSE/MAE/RMSE 或多标签 loose/strict accuracy
  - `Evolution::with_report_metrics(...)` 支持追加报告指标；报告只用于日志、回调与结果展示，不进入 primary fitness、target 判断、NSGA-II objective 或 archive 收敛
  - 默认 `DefaultCallback` 日志显示 `metrics=...`，补充单测覆盖指标计算、去重/兼容性、多标签 BCE logit 阈值和“报告不影响选择”边界

- **feat(nn): 全链路新增节点 ONNX provenance(`origin_onnx_nodes`)** [`b115ff2`]
  - `NodeDescriptor.origin_onnx_nodes: Vec<String>`:`#[serde(default)]` 兼容旧 .otm,`skip_serializing_if = "Vec::is_empty"` 让无 origin 的节点不写入 JSON 体积;`NodeDescriptor::new` 签名不变 + 链式 builder `with_origin_onnx_nodes` 让 18 处历史调用零修改
  - `NodeInner` 加 `RefCell<Vec<String>>` 字段 + getter / setter(后置注入风格):零侵入到所有 Layer / Var / 算子构造路径
  - `descriptor_rebuild::rebuild_node` 创建 Var 后从 NodeDescriptor 注入 origin 到 NodeInner
  - `SnapshotNode` 加 origin 字段;`build_snapshot` 透传;`snapshot_to_dot` 渲染规则:空 Vec 不显示(演化路径零影响)/ ≤3 项列出 / >3 项显示 `+N more`,tooltip 始终含完整列表
  - import 期 ~10 个填充点(BasicInput / Initializer / Constant→Param / 1:1 默认 / Conv+bias / Gemm / Reshape 折叠 / Resize 折叠 / Split 重写)
  - 9 个新单测:6 个 provenance + 3 个 DOT 渲染(覆盖 1:1 / Conv+bias / Split / `.otm` round-trip / legacy 兼容 / 演化空 Vec / DOT 空守卫 / DOT 渲染 / DOT +N more 摘要)

- **feat(nn): Upsample2d 算子（2D 最近邻上采样）+ ONNX 双向桥接** [`899c3d5`]
  - 完整新增 `NodeTypeDescriptor::Upsample2d { scale_h, scale_w }`：raw_node 前向（nearest 像素复制）+ 完整反向（sum_pool，等价 avg_pool × scale_h × scale_w）+ builder + descriptor_rebuild + onnx_ops 双向映射
  - ONNX 导入：`Resize` / `Upsample` 自动桥接到 `Upsample2d`（mode="nearest"，scale 由常量折叠填）
  - ONNX 导出：`Upsample2d → Resize` opset 13 nearest（按策略文档 §4.4 不承诺 round-trip）
  - 18 个单测：tensor 级 + 节点级，对照 PyTorch `nn.Upsample(mode='nearest')` 数值一致

- **feat(nn/graph): ONNX Constant 折叠 + Split 重写（路线 B 模式重写）** [`fdc61e7`]
  - 在 `assemble` 加预处理 pass：扫描 `OpType::Constant` 节点 + initializer 建立常量表
  - `Reshape`：从常量表读 shape 输入折叠到 `Reshape::target_shape`，支持 ONNX `-1`（按 parent 形状静态推导）和 `0`（保留对应 parent 维度）
  - `Resize`：scales 折叠到 `Upsample2d::scale_h/scale_w`（仅 4D NCHW + 整数倍 nearest）
  - `Split`：展开为 N 个 `Narrow` 节点，`split_sizes` 来自 attribute（opset≤12）或 input 常量（opset 13+）
  - 9 个新单测覆盖 Reshape via initializer/Constant、Reshape -1 推导、Reshape 0 维保留、Reshape 拒绝多个 -1、Resize scales 整数倍、Resize 拒绝非整数倍、Split via attribute / Constant input

- **feat(nn/graph): ONNX `Transpose` 导入支持** [`602466c`]
  - `OpType::Transpose` 读 `perm` 属性映射到 `NodeTypeDescriptor::Permute`（缺 perm 报 actionable 错，提示 onnxsim 预处理）
  - `Permute` 从 Unsupported 列表挪到独立导出分支，导出为 `Transpose` 含 perm 属性
  - 5 个单测覆盖 4D NCHW→NHWC、2D 转置、缺 perm 报错、导出验证、完整 round-trip

- **feat(nn/graph): `ImportReport` 最小骨架 + `RebuildResult.import_report` 透传** [`4e8c913`]
  - 新增 `ImportReport { rewritten, warnings }` + `RewriteRecord { pattern, consumed_onnx_nodes, produced_descriptor_nodes }`
  - `OnnxImportResult` 挂 `import_report` 字段；`RebuildResult` 加 `import_report: Option<ImportReport>` 让 ONNX 路径全程透传
  - 把现有 `Conv+bias` 拆分（pattern=`conv_with_bias_to_conv_plus_add`）和 `Gemm→MatMul+Add` 拆分（pattern=`gemm_to_matmul_plus_add`）作为已知 rewrite 填进去
  - 3 个单测验证字段被正确填充
  - 范围严控：不含 `folded` / `shape_inference` / `provenance` / `ImportOptions` 等扩展字段，等真正撞到对应需求时再补

- **feat(nn): ONNX 导入路径可观测性 API 公开** [`bfd6afc`]
  - 新增 `nn::load_onnx` / `nn::load_onnx_from_bytes` / `nn::ImportReport` / `nn::OnnxImportResult` / `nn::RewriteRecord` 类型导出
  - 让用户在 `Graph::from_onnx` 的 rebuild 阶段失败时，仍能拿到 ImportReport 做诊断

- **feat(example): `chinese_chess_yolo` 端到端打通 + 内置 sample 截图** [`9b1ac85`]
  - 之前 example 卡在 forward + FEN(README 写为"已知 limitation"),实际是框架 `explicit_output_ids` bug + 业务侧多个 bug 叠加。框架 bug 已在前一个 commit 修复,本 commit 修业务侧并补内置 sample 让 example 开箱即跑
  - **修业务 bug**:类别字典之前 14 类瞎猜,改为按 VinXiangQi v1.4.0 官方源码 `YoloXiangQiModel.cs` 对齐 15 类 `[n,b,a,k,r,c,p,R,N,A,K,B,C,P,board]`(class 14 整盘 bbox 不进 FEN);ROI 之前写死整图,改为 `auto_detect_board_roi` 优先用棋子检测中心包络 + fallback board 类 bbox 内缩 5%;新增 `detect_red_on_top` + `rotate_grid_180` 视觉朝向自动检测,支持反向截图(红方在原图上方时整盘转回标准方向)
  - **视觉朝向作为独立输出项**:FEN 是逻辑棋局表示(标准约定永远红方在 row 9 底,字符串本身无法表达原图视觉朝向),所以拆成两份输出——视觉朝向报告"红方在棋盘上方/下方",标准 FEN 永远红方在底与视觉朝向解耦
  - **内置 sample + 自动对比**:`samples/{sample_red_bottom.png, sample_red_top.png, example_answer.txt}`,跑 samples/ 下的图自动从 answer.txt 找答案做位级对比,输出 `✓ 匹配` 或 `✗ 不匹配 期望=... 实际=...`,把 example 同时升级为真正的端到端回归测试
  - **CLI 参数 + 默认开箱即跑**:默认 `cargo run --example chinese_chess_yolo` 跑 sample 1;sample 2 用 `-- <路径>.png` 指定;删掉之前合成截图脚本的废话提示
  - **验收**:两个 sample FEN 位级匹配人类标注(中盘残局 29/90 + 初始局面 32/90),朝向输出对两种朝向都正确

- **feat(example): `chinese_chess_yolo` 端到端 example 框架(VinXiangQi YOLOv5 模型)** [`a5529a1`, `bfd6afc`]
  - `download_model.py`:拉取 VinXiangQi v1.4.0 release(93 MB)+ 解压 `.onnx` + 用 `onnx` 库审计算子缺口;中间产物放跨平台 cache 目录(默认 `~/.cache/only_torch_yolo_cache/`,可用 `XIANGQI_CACHE_DIR` 环境变量覆盖),模型落 `models/vinxiangqi.onnx`(已被 `.gitignore` 排除)
  - `letterbox.rs`(~80 行):等比缩放 + 灰色填充到 640×640 + NCHW 归一化
  - `yolo_decode.rs`(~120 行):YOLOv5 输出解码 + 纯 Rust per-class O(N²) NMS
  - `board_align.rs`(~120 行):bbox → 9×10 网格对齐 + FEN 序列化(含类别字典 / ROI 自动锁定 / 视觉朝向检测)
  - `main.rs`(~250 行):分两步 import + rebuild,rebuild 失败时优雅降级 + actionable 提示,仍展示 ImportReport
  - `README.md`:用法 + 调优指引 + 已知 limitation
  - `Cargo.toml` 注册 `[[example]] chinese_chess_yolo`

- **test(onnx_models): `yolov5_xiangqi` 回归 fixture** [`1cec9f9`]
  - 按 `.doc/design/onnx_import_strategy.md` §8.1 目录约定布局：`README.md` + `.gitignore` + `export.py`（转发 example download_model.py）+ `numeric_check.py`（用 onnxruntime 跑参考输出）
  - `tests/yolov5_xiangqi_import.rs`：1 个 CI 默认跑（fixture 元信息）+ 2 个 `#[ignore]`（descriptor 拓扑 + ImportReport 4 模式覆盖）
  - 防止 `chinese_chess_yolo` import 路径被未来改动悄悄回退

### 修改

- **refactor(nn): 删 `.otm` 跨版本兼容兜底 + MaxPool2d IR 层 padding 4→2 元组** [`e6686b4`]
  - **动作 A:删 .otm 跨版本兼容兜底**:删 `default_dilation` / `default_output_padding` / `default_max_pool_padding` 三个辅助函数 + Conv2d.dilation / ConvTranspose2d.output_padding / MaxPool2d.{padding, ceil_mode} 上的 `#[serde(default)]` 标注;`origin_onnx_nodes` 保留 `default + skip_serializing_if`(配对惯用法,空 Vec 不写入 JSON 体积,不属于版本兜底);`GraphDescriptor::from_json` 加 actionable error 包装(失败时报本地版本号 + 文件版本号 + "请用对应版本重新加载或在新版本下重新 train/save");删 `test_provenance_legacy_otm_compatible` + 新增 2 个 fail-fast 测试覆盖版本不匹配/完全损坏 JSON 场景
  - **动作 B:MaxPool2d.padding IR 层 4 元组 → 2 元组(对称语义,与 Conv2d 对齐)**:`NodeTypeDescriptor::MaxPool2d.padding` 从 `(usize×4)` 改 `(usize×2)` 对称 `(pad_h, pad_w)`;Layer `with_padding` API + `create_max_pool2d_node` 形参同步改 2 元组,内部展开为 `(p_h, p_h, p_w, p_w)` 传给 raw_node(raw_node 仍保留 4 维表示——算法实现需要,前向用 `NEG_INFINITY` 虚拟填充避免污染 max,反向需按边 unpad);`var/descriptor.rs` 转换处加 `debug_assert` 对称性检查,只取 `(top, left)` 进 IR;ONNX 导入复用 `parse_symmetric_2d_pads`(与 Conv 一致,非对称报 actionable error 提示用 `ZeroPad2d` / `onnxsim`);导出从 `(p_h, p_w)` 展开为 4 维 ONNX `pads = [p_h, p_w, p_h, p_w]`;evolution + tests 共 30 处字面量 `(0,0,0,0) → (0,0)`,原"非对称 H 维 padding"测试改为"非对称应被拒绝"
  - **顺手刷新过时快照**:`examples/evolution/parity_seq_var_len/{.dot,.png}` 的 committed 版本来自 commit 712e619(演化阶段 F),之后多个 commit(b115ff2 / f88e0a7 等)间接影响演化轨迹但没人重生成快照,本次跑 example 顺手刷新(FEN 准确率 87%→96%,seed=42 确定性可复现)
  - **验收**:`cargo check + tests + examples + benches` 全部通过零新 lint;`cargo test --lib` 3105/3105 全过;16 traditional + 4 evolution examples(不含 RL/中国象棋/MNIST 演化)端到端跑通

- **refactor(nn/graph): `onnx_import.rs` 拆分为子目录,启用 `ImportReport.warnings` 字段** [`722e9e0`]
  - 把单文件 `src/nn/graph/onnx_import.rs` (~1080 行) 拆分到 `src/nn/graph/onnx_import/` 子目录的 7 个文件:`mod.rs` / `assemble.rs` / `const_table.rs` / `fold_reshape.rs` / `fold_resize.rs` / `split_narrow.rs` / `util.rs`
  - 不抽 `PatternRewrite` trait(设计文档 §7.2 提议)——实测 5 种 rewrite 全在 ONNX 节点装配阶段做(不是 GraphDescriptor 后处理),强抽 trait 会引入"胖 Context struct + 大量泛型"的反向复杂度
  - **启用 `ImportReport.warnings`**(此前定义为 `Vec<String>` 但全程无人 push,死字段):Conv+bias 拆分时 bias 升维 → warning;Gemm 转置 B → warning;Resize 折叠为 Upsample2d → warning(提示 coordinate_transformation_mode 子模式差异在整数倍场景可忽略)
  - 实测 VinXiangQi 导入产出 71 条 rewrite + **62 条 warning**(60 Conv bias 升维 + 2 Resize 折叠)
  - 1 个新单测 `test_import_report_warnings_populated`:验证 Gemm with transB=1 触发 "transB=1 ... gemm0" 风格 warning

- **chore(nn/graph): ONNX `MIN_OPSET_VERSION` 从 13 降到 12** [`fdc61e7`]
  - 兼容 VinXiangQi 等 YOLOv5 老版本导出（opset 12 引入了 Constant/Split/Pow 的稳定形式，本 import 已覆盖）

- **chore: 历史遗留私有/隐私信息脱敏** [`4f698e7`]
  - `download_model.py` 之前硬编码本机绝对路径，不适合作为公开 example 的默认行为，改为跨平台 cache 标准做法：默认 `~/.cache/only_torch_yolo_cache/`（Windows 落到 `%USERPROFILE%/.cache/...`），允许 `XIANGQI_CACHE_DIR` 环境变量覆盖
  - CHANGELOG.md 老条目里残留的本机路径 + 个人项目名同步脱敏，下游集成应用相关描述泛化
  - `.doc/design/onnx_import_strategy.md` §9.1 中的个人项目名改为通用“下游连线器应用”
  - **已知遗留**（本次不动）：`examples/traditional/chinese_chess/prepare_real_pieces.py` 还有硬编码第三方软件安装目录，属另一个 chess CNN example 的脚本，留作独立 backlog 任务

- **chore: 删 `prepare_real_pieces.py` + 清理 `--real-data` 真实数据混合路径** [`1ed9acb`]
  - 处理 `4f698e7` 留下的“已知遗留”：`prepare_real_pieces.py` 本质是为开发者本地从第三方象棋软件安装目录提取真实棋子贴图作 fine-tune mixin 的私货脚本，公开仓库不需要，且含硬编码本机安装路径
  - 整文件删除(-526 行)+ `data.rs` 移除 `has_real` 真实数据混合分支(-34 行,含 `concat_tensors` 辅助函数 + doc-comment 同步)+ `train_pytorch.py` 移除 `--real-data` CLI 参数及全部相关路径(-113 行,含 `load_and_merge_data` 简化为 `load_data` / `evaluate(real_mask_all)` 简化 / `best_real_acc` 跟踪删 / 真实数据子集统计段删)
  - 影响:对应示例仍能完整演示「PyTorch → ONNX → only_torch continue-train → .otm 保存/加载」全流程,合成数据 baseline 实测 97.1%(本次发版前实测数据);改动局限在 example 内部,README / 设计文档 / Cargo.toml 都不需要改

- **refactor(example): chess 系列重命名突出框架能力** [`0a9d1fd`]
  - 旧名 `chinese_chess` / `chinese_chess_yolo` 看不出在演示 only_torch 的什么能力,对外(公开仓库)体验不友好。改为"领域 + 模型 + 核心能力"三段式:
    - `chinese_chess` → `chess_cnn_onnx_finetune`(CNN + ONNX 互通 + 继续训练)
    - `chinese_chess_yolo` → `chess_yolo_onnx_detect`(YOLO + ONNX 互通 + 检测推理)
  - 两个示例形成"训练侧 vs 推理侧"的互补对比,一眼看清各自定位
  - 主 README chess 折叠章节从「中国象棋示例」改为「ONNX 互通示例(Chess 系列)」,内部重写为对比表格 + 两个示例分别详细描述;`chess_yolo_onnx_detect` 首次进入主 README 概览表(此前只在子 README + CHANGELOG 出现)
  - 设计文档 `.doc/design/onnx_import_strategy.md` §8.3 supported models matrix 顺手新增 VinXiangQi YOLOv5 ✅ 行(本来就该加,example 早已端到端跑通且 FEN 位级匹配,只是上次 commit 没顺手补)
  - 25 个文件改动:Cargo.toml + justfile 注册同步 + 主 README 三处(概览表 / 折叠节 / 特性矩阵列名 `chess_cnn` 缩写) + `tests/yolov5_xiangqi_import.rs` + `tests/onnx_models/yolov5_xiangqi/{README.md, export.py}` + `src/nn/` 2 处注释 + 各示例内部 35 处路径引用(含 `OTM_PATH` / `DATA_DIR` / `SAMPLES_DIR` 等常量,运行时 .otm / 数据 / sample 路径全部跟着改名)
  - **故意保留 chinese_chess 引用**:CHANGELOG.md 历史条目(时光胶囊属性) + `src/nn/graph/onnx_import/mod.rs:43` 注释里的 plan 文件名引用(历史工作代号,plan 文件已不在仓库)
  - 验收:`cargo check` 两个新 example 通过 + `cargo test --lib` 3105/3105 全过 + ReadLints 全清

### 文档

- **docs: 添加 ONNX 导入/互通策略设计文档** [`6fcc013`]
  - 新增 `.doc/design/onnx_import_strategy.md`，沉淀 ONNX import/export 的支持边界、路线选择与后续 backlog
  - 为后续 Upsample2d、Transpose、Constant folding、Split rewrite 与真实 YOLOv5 模型导入提供设计依据

- **docs(design): 扩充 onnx_import 设计文档 §9 为权威 backlog** [`d410c87`]
  - 把散落在旧 plan §9 / 新 plan §6 / 设计文档原 §9 / R4 风险注释的 11 项 backlog 整合到 `.doc/design/onnx_import_strategy.md` §9 作为权威入口
  - 5 类组织:业务层(4)/ 算子层(3)/ ImportReport 扩充(3,R4 显式守住)/ 架构层(1)/ 永远不做(1)
  - 每项带:触发条件、预期产出、来源 plan/章节、立项 plan、风险评估
  - 机制约定:任何新 plan 立项前先看本表;立项时填 plan 文件名;完成后从表中移除并落 CHANGELOG(避免 backlog 与 CHANGELOG 重复维护)

### 实测里程碑

- **VinXiangQi YOLOv5 模型 ONNX 导入 + rebuild 端到端跑通**(`yolo_followup_three_commits` plan 5 个 commit 完成)
  - import 阶段 release 12.9 ms / 423 个 descriptor 节点(Constant→Parameter 后增到 443)
  - rebuild 阶段:**spatial shape 传播 bug 已修复**(MaxPool padding/ceil_mode 补全 + Conv 对称 padding 修对 + Constant→Parameter 保留 + Concat/Permute placeholder 精化)
  - ImportReport 71 条 rewrite + 62 条 warning,4 种 rewrite 模式齐全:`conv_with_bias_to_conv_plus_add`(60)/ `constant_fold_into_reshape`(6)/ `constant_fold_into_resize`(2)/ `split_to_narrows`(3)
  - 集成回归 `tests/yolov5_xiangqi_import.rs::yolov5_xiangqi_rebuild_succeeds` 持续通过

## [0.15.0] - 2026-04-20

### 新增

- **feat(evolution): NodeLevel 统一内核重构（Phase 1-10）——演化系统架构级大改**
  - Phase 1+2：`NodeGene` 统一中间表示（IR）+ `LayerConfig` 迁移层，所有 Layer 配置统一收敛到 `NodeGene` 粒度
  - Phase 3：NodeLevel `capture_weights` / `restore_weights` 权重快照，参数级精确保存与恢复
  - Phase 4：NodeLevel 变异算子（`InsertNode` / `RemoveNode` / `GrowHiddenSize` / `ShrinkHiddenSize` / `ChangeActivation` 等）+ 确定性修复
  - Phase 5：Parameter 节点粒度权重继承——Lamarckian 继承从层级下沉到参数级
  - Phase 6：节点级演化收口与持久化验收——序列化 / 反序列化完整性验证
  - Phase 7：NodeLevel 通用跨层连接变异（`AddConnection` / `RemoveConnection`），替代旧 `SkipEdge` 层级操作
  - Phase 8：NodeLevel 循环网络支持——RNN / LSTM / GRU 均通过节点级基因表达
  - Phase 9：LayerLevel 从演化内核降级为用户入口 DSL——用户仍用 `LayerGene` 描述初始网络，内部自动转为 NodeLevel 运行
  - Phase 10：ONNX 双向桥接——`NodeGene` ↔ ONNX 导出/导入，支持与外部工具链互通

- **feat(evolution): Pareto 种群搜索 + NSGA-II 选择**
  - 多目标搜索（primary fitness + complexity）替代单目标 greedy
  - NSGA-II 非支配排序 + 拥挤度距离选择
  - 并行评估（`rayon` 多线程 `evaluate_batch`），显著加速大种群演化
  - `EvolutionResult` 返回 Pareto 前沿全部成员，用户可按偏好选择

- **feat(evolution): 阶段 A — Spatial 域增强**
  - 解决 MNIST 演化瓶颈：自动推断 Spatial 输入形状、Flatten 维度计算、Conv2d padding/stride 合法性校验
  - Conv2d / Pool2d 从空间模式必需层降级为可演化层——演化可自由插入/删除 CNN 组件

- **feat(evolution): 阶段 B — InsertAtomicNode 变异 + 归一化层/Dropout 纳入演化**
  - `InsertAtomicNode`：在任意两个已有节点间插入单个激活/归一化节点，细粒度拓扑探索
  - 通用循环边支持：演化可在任意层间创建 recurrent connection
  - BatchNorm / LayerNorm / GroupNorm / RMSNorm / Dropout 全部纳入可演化变异空间

- **feat(evolution): 阶段 C — EXACT 级别 Spatial 域 Feature Map 粒度演化**
  - FM（Feature Map）级别基因表示：每个 Conv 块内独立管理 per-channel 连接
  - 10 种 FM 级别变异：`AddFMEdge` / `RemoveFMEdge` / `SplitFM` / `MergeFM` / `ChangeFMKernel` 等
  - FM 掩码融合（FM Mask Fusion）：构图时自动检测同构 FM 边，合并为单个 dense Conv2d，减少计算图节点数
  - `FMFusionAnalysis`：per-block 同构性检测 + 融合矩阵构建

- **feat(evolution): 阶段 F — 流程修复（F1-F4）**
  - **F1 Net2Net 函数保持性扩容**：`GrowHiddenSize` 扩容时新增维度复制已有列 + 小扰动，下游消费者行按复制次数缩放；覆盖 Linear / Conv2d / RNN / LSTM / GRU + 下游 + BN/LN/RMS pass-through
  - **F2 Cell 类型切换权重迁移**：`migrate_cell_weights()` 覆盖 6 种迁移（RNN↔LSTM / RNN↔GRU / LSTM↔GRU），特征门保留权重，饱和门用 `W=0 + bias=±6` 使 σ 饱和
  - **F3 学习速度代理（LossSlope）**：`FitnessScore` 新增 `primary_proxy: Option<f32>`，`ProxyKind::LossSlope` 计算 loss 下降斜率；NSGA-II plateau 时用 proxy 打破平局（默认启用）
  - **F4 ASHA 多保真评估**：`AshaConfig { rung_epochs, eta }` 默认 `[1,2,4]/eta=3`，阶梯式 Successive Halving 将训练预算集中到头部候选（默认启用）
  - F3/F4 默认启用 + LayerLevel Lamarckian 继承修复

- **feat(evolution): 序列域演化支持**
  - 自动推断序列输入维度、支持 `minimal_sequential` 初始基因组
  - 演化激活函数池扩展至 13 种（新增 GELU / Swish / ELU / SELU / Mish / HardSwish / HardSigmoid / Softplus）

- **feat(evolution): CNN 空间演化 + 记忆单元演化**
  - Conv2d / Pool2d 可被演化自由插入/删除/参数化
  - 记忆单元（RNN/LSTM/GRU）可被 `MutateCellType` 在运行中切换

- **feat: 统一 .otm 模型格式**
  - 手动构建的模型和演化生成的模型均可保存拓扑 + 权重到 `.otm` 文件
  - `Graph` 权重 API：`save_weights()` / `load_weights()`

- **feat(nn): LR Scheduler 模块**
  - `CosineAnnealingLR`：余弦退火学习率调度
  - `StepLR`：阶梯式衰减
  - `LambdaLR`：自定义函数调度

- **feat(vision): 新增 3 种数据增强变换**
  - `RandomErasing`：随机擦除
  - `RandomResizedCrop`：随机缩放裁剪（双线性插值）
  - `RandomAffine`：随机仿射变换（旋转 + 平移 + 缩放 + 剪切）

- **feat(nn): 新增 API**
  - `Graph::set_seed(seed)` / `Graph::has_seed()` 代理方法
  - `EvolutionResult` 新增 `evolution_seed` 字段，支持 Pareto 成员确定性重建
  - `EvolutionTask::train()` 返回类型变更为 `TrainOutcome { final_loss, proxy }`

- **feat(examples): 新增演化示例**
  - `evolution_parity_seq`：序列数据演化，记忆单元自动选择
  - `evolution_parity_seq_var_len`：变长序列演化，zero-pad 自动处理

### 性能优化

- **perf(evolution): Spatial 域演化速度多项优化**
  - 收紧 `SizeConstraints::auto()` 的 `fc_base` 计算，防止 Flatten→Linear 参数爆炸
  - `ComplexityMetric` 默认值从 `ParamCount` 切换为 `FLOPs`
  - `GrowHiddenSize` 变异权重从 0.25 降至 0.12
  - 新增 BLAS 线程守卫：`parallelism > 1` 时自动设置 `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `OMP_NUM_THREADS = 1`

### 修复

- **fix(nn): 种子确定性严格保证 — 指定 seed 后所有随机操作 100% 可复现**
  - `Var::dropout()` / `Graph::randn()` / `Var::rand_like()` / `Var::randn_like()` / `Normal::rsample()` / `Categorical::sample()` 全部改用 Graph RNG
  - `descriptor_rebuild` 中 Dropout 重建改用 `next_seed()` 替代固定 seed 42
  - 演化系统 `rebuild_pareto_member()` 使用保存的 `evolution_seed` 替代 `from_entropy()`
  - 演化系统指定 seed 时自动固定 `population_size`（20）和 `offspring_batch_size`（12），消除跨机器线程数差异
- **fix(nn): BatchNorm 4D 广播 bug + running stats 跨 forward 丢失 bug**
- **fix(nn): GroupNorm gamma/beta 梯度链修复**
- **fix(nn): Kaiming/Xavier init fan_in 计算修复**
- **fix(nn): ConvTranspose2d output_padding 参数在 ONNX 导出/导入时丢失**
- **fix(evolution): Pareto 演化系统正确性与收敛效率修复 + 测试补全**
- **fix(evolution): skip edge 域重新验证 + `is_domain_valid` 语义修正**
- **fix(evolution): NodeLevel Cluster 可视化缺少输入形状描述**
- **fix(evolution): RNN 重建路径 `NodeGroupTag` 被 backfill 覆盖的可视化 bug**
- **fix(net2net): 堆叠循环层 + Conv2d→Flatten→Linear 扩宽路径修复**

### 重构

- **refactor: examples 目录重构为 `traditional/` 和 `evolution/` 两组**
- **refactor: `save_model()` / `load_model()` → `save_weights()` / `load_weights()` 重命名**
- **refactor(evolution): Conv2d / Pool2d 从空间模式必需层降级为可演化层**
- **refactor(evolution): 演化系统内部自适应改造（6 项）**——变异概率动态调整、停滞检测参数优化等
- **refactor(evolution): 移除所有 Phase N 工程阶段注释**

### 文档

- 演化设计文档全面更新：Phase 1-10、A-C、F 阶段完成状态、优先级图更新
- 更新种子设计文档：标记阶段 2.5 完成
- 新增演化、强化学习和测试指令文档
- 更新 ONNX 双向桥接规划与完成记录
- 中国象棋示例增强：合并真实数据、增加 RandomAffine、batch=256

### 已知问题

- MNIST 演化示例运行较慢（阶段 D/E 优化项待后续版本跟进）

## [0.14.0] - 2026-03-09

### 新增

- **feat(evolution): 神经架构演化模块 MVP（核心特色功能）**
  - 完整的 Genome-centric 层级演化系统，用户只需提供数据和目标——零模型代码
  - `gene.rs`: 基因数据结构（`NetworkGenome`、`LayerGene`、`SkipEdge`、`TrainingConfig` 等）
  - `mutation.rs`: `Mutation` trait + `MutationRegistry`，内置 12 种变异操作
    - 结构变异：`InsertLayer`、`RemoveLayer`、`AddSkipEdge`、`RemoveSkipEdge`
    - 参数变异：`GrowHiddenSize`、`ShrinkHiddenSize`、`ChangeActivation`
    - 训练超参数变异：`MutateLearningRate`、`MutateOptimizer`、`MutateBatchSize`、`MutateLossFunction`
    - 聚合变异：`ChangeAggregateStrategy`
  - `builder.rs`: Genome → Graph 自动转换 + Lamarckian 权重继承（跨代权重复用）
  - `convergence.rs`: `ConvergenceDetector` 训练收敛检测（loss plateau + gradient norm 双判据）
  - `task.rs`: `EvolutionTask` trait + `SupervisedTask` 监督学习实现，支持 full-batch / mini-batch
  - `callback.rs`: 回调接口（`EvolutionCallback` + `DefaultCallback`），支持自定义日志/停止策略
  - `Evolution` 主控结构体：Builder 模式 API，`run()` 驱动完整演化主循环
  - `EvolutionResult`：`predict()` 推理 + `visualize()` 计算图可视化
  - `SkipEdge` DAG 拓扑演化：支持 `Add`/`Concat`/`Mean`/`Max` 四种聚合策略
  - `NetworkGenome` Display：主路径摘要 + skip edge 注解 + 重名层自动消歧
  - 停滞探测机制：连续 N 代 primary 未提升后强制结构变异
  - 完整的单元测试和集成测试覆盖

- **feat(evolution): 新增 2 个演化示例**
  - `evolution_xor`: XOR 零模型代码演化，从 `Input(2) → [Linear(1)]` 自动发现解决方案
  - `evolution_iris`: Iris 鸢尾花演化，150 样本自动 mini-batch + CrossEntropy 推断

- **feat(examples): 新增中国象棋棋子 CNN 分类器示例**
  - 15 类分类（空位 + 红方 7 子 + 黑方 7 子），28x28 合成 patch
  - Conv(3→16) → Pool → Conv(16→32) → Pool → FC(1568→128) → FC(128→15)
  - 运行时数据增强（ColorJitter）、early stopping、per-class 准确率报告

- **feat(nn): 批量新增 18 项基础节点（节点总数 41 → 53）**
  - 已在 0.13.0 CHANGELOG 中列出（该批提交实际落入本版本）

- **feat(nn): 将 ReLU 从 LeakyReLU 中独立为一等节点**

- **feat: 可选 BLAS 加速（Intel MKL / OpenBLAS）**
  - 通过 `--features blas-mkl` 或 `--features blas-openblas` 启用
  - justfile 自动检测本地 BLAS 后端（MKL > OpenBLAS > 纯 Rust）

### 性能优化

- **perf(conv2d): im2col + GEMM 替换嵌套循环卷积，训练速度提升 2.6-4.4x**
- perf(conv2d): 反向传播 im2col 批量化，N 次小 GEMM 合并为 1 次大 GEMM
- perf(conv2d): 前向传播 padded_input 缓存改用 move 消除 clone
- perf(nn): 反向传播全局优化——in-place 梯度累加 + ReLU 融合 + MaxPool 预分配
- perf(nn): `GradResult` 零拷贝梯度传递 + benchmark 基础设施
- perf(optimizer): `set_value_owned` 零拷贝参数更新 + Adam 临时分配优化

### 修复

- fix: 消除编译警告 + 补充 `GradResult::Negated` 路径单元测试
- fix: 补齐 roadmap 遗漏项（Tensor 测试 + 独立节点 + Var API）

### 重构

- refactor: 计算图表示中 LeakyReLU 替换为 ReLU
- refactor(evolution): 移除所有 Phase N 工程阶段注释
- refactor(evolution): `EvolutionError` + 延迟实例化，`supervised()` 恢复无错构造
- refactor(evolution): 隐藏 `Graph`，`EvolutionResult` 仅暴露 `predict()` / `visualize()` API
- refactor(examples): 更新中国象棋模型架构和数据增强

### 文档

- docs: 归档已完成的规划文档，整合至 architecture_roadmap
- docs: 更新性能优化文档，反映 Phase 1-5 完成状态
- docs: 更新文档反映 roadmap 完成状态
- docs: 新增 oneDNN CPU 内核优化参考
- docs: 数据共享可视化已通过 source_id 实现，更新未来方向

### 其他

- feat: Phase 1-5 feature expansion（CNN / data augmentation / Transformer / API convenience methods / Repeat node / Chunk / Norm variants / error refinement / utility activation methods）

## [0.13.0] - 2026-02-14

### 破坏性变更

- **feat: 全面分离 Stack 与 Concat 为独立操作**
  - `Stack` 和 `Concat` 不再合并为同一节点，各自拥有独立的语义和实现
  - 新增 `Var::cat` 便捷方法（对应 PyTorch 的 `torch.cat`）

- **refactor(nn): 将 Detach 从 Identity 标志位拆分为独立节点类型**
  - `Detach` 不再是 `Identity` 的特殊标志，而是完整独立的计算图节点

- **refactor(vis): 统一节点分组机制，删除旧 LayerGroup/RecurrentLayerMeta 体系**
  - 新的节点分组上下文机制取代旧式 `LayerGroup` / `RecurrentLayerMeta`

### 新增

- **feat(graph): 实现通用 CSE（公共子表达式消除）节点去重机制**

- **feat(nn): 新增概率分布模块**
  - `Categorical`：离散分类分布（支持 log_prob / entropy / sample）
  - `Normal`：正态分布
  - `TanhNormal`：Tanh 压缩正态分布（SAC 连续动作策略核心）

- **feat(nn): 新增计算图节点**
  - `Exp`：指数函数
  - `Clip`：值域裁剪
  - `Sqrt`：平方根
  - `Negate`：取负（补全基础算术运算对称性）

- **feat(nn): 批量新增基础节点（18 项，节点总数 41 → 53）**
  - 7 个现代激活函数节点：`GELU`、`Swish/SiLU`、`ELU`、`SELU`、`Mish`、`HardSwish`、`HardSigmoid`
  - 形状操作节点：`Narrow`（沿轴连续切片）、`Permute`（维度重排 / 转置）
  - 条件/筛选节点：`Where`（掩码选择）、`TopK`（取前 K 大值）、`Sort`（沿轴排序）
  - 3 个 Var 便捷方法（无独立 NodeType）：`squeeze`、`unsqueeze`、`split`
  - 统一 Tensor → Node → Var 三层架构，每层均有独立测试
  - 11 个 Python 对照脚本（PyTorch 前向值 + Jacobian 验证）

- **refactor(nn): 补齐 3 个已有节点的 Tensor 层方法**
  - `LeakyReLU`、`SoftPlus`、`Step` 的前向计算下沉到 Tensor 方法，统一三层调用路径
  - 附带将 `Concat` 内的 `slice_along_axis` 重构为 `Tensor::narrow`

- **feat(vis): Graph 快照可视化 + 多 Loss 路径边着色**
  - 支持在任意时刻对计算图进行快照可视化
  - 多 Loss 场景下自动为不同 Loss 路径着色

- **feat(vis): 节点分组上下文机制 + 分布 cluster 可视化**
  - 基于上下文的灵活分组，支持概率分布模块的 cluster 展示

- **feat(vis): Tensor source_id 追踪 + 同源数据节点链式虚线标注**
  - 追踪数据来源，同源输入以虚线可视化关联

- **feat(rl): 新增 SAC 示例**
  - SAC-Continuous Pendulum 示例
  - Moving-v0 Hybrid SAC 示例（方式 B — 独立连续分支）

### 修复

- fix(vis): 修复 `.dot` 输出中同源数据虚线边顺序不确定的问题
- fix(vis): 修复 RNN/LSTM/GRU 场景 Input 节点未归入模型 scope 的 bug
- fix(docs): 移除公开文档中的本地私有路径

### 重构

- refactor: 大文件按功能域拆分，降低单文件复杂度
- refactor(examples): 8 个示例改用 snapshot 可视化 + GAN 多 Loss 着色 + detach 节点命名
- refactor(examples): 4 个示例从逐样本训练改为 full-batch 模式
- refactor(test): 将内联单元测试迁移到独立 tests/ 目录

### 测试

- test(tensor): 补充 source_id corner case 单元测试

### 文档

- 新增 Input 节点语义与数据共享可视化设计文档
- 新增 RL 路线图，整理 RL 相关文档过时内容
- 新增 SAC 数学基础分析文档

### 其他

- chore: Minari 联网测试加 `#[ignore]`，justfile 细化测试命令
- chore: rustfmt 格式化 + lint 清理

## [0.12.0] - 2026-02-12

### 破坏性变更

- **refactor(nn): 动态图架构迁移（方案 C）**
  - `Var` 持有 `Rc<NodeInner>`，节点生命周期由引用计数自动管理
  - 移除 `ModelState`、`Criterion` — 不再需要闭包式缓存机制
  - 移除 `GraphInner::new_*_node()` / `forward(NodeId)` / `get_node_value(NodeId)` 等旧 API
  - 新 API：`Graph` + `Var` 算子重载 + `Module` trait + `Optimizer`

- **refactor(nn): 移除旧式循环机制**
  - 删除 `connect_recurrent` / `step` / `backward_through_time` 等旧 API
  - 删除 `StepSnapshot` / `recurrent_edges` / `prev_values` 等旧字段
  - 展开式 RNN/LSTM/GRU 设计完全取代旧式显式时间步方案

- **refactor(nn): 移除 `backward_ex()` 和 `retain_graph` 参数**
  - 动态图架构下节点自动管理生命周期，`retain_graph` 不再需要
  - 统一使用 `backward()` 即可支持多 loss 梯度累积、多次反向传播

### 新增

- **feat(nn): PyTorch 风格动态图 API**
  - `graph.input()` / `graph.parameter()` 创建变量
  - `&a + &b`、`a.matmul(&b)` 等算子重载
  - `var.forward()` / `var.backward()` 自动前向/反向传播
  - `var.mse_loss()` / `var.cross_entropy_loss()` 等损失函数方法链

- **feat(rl): 强化学习基础设施**
  - `GymEnv`：与 Python Gymnasium 环境交互
  - `Minari`：离线 RL 数据集加载
  - CartPole SAC-Discrete 示例（Twin Q、自动温度调节、目标网络软更新）

- **feat(nn): RNN/LSTM/GRU 展开式设计**
  - 一次性处理整个序列，标准 `backward()` 自动完成 BPTT
  - 支持动态 batch_size 和变长序列

### 测试

- **test: 全量测试迁移完成**
  - 1579 个单元测试全部通过（0 failed, 0 ignored）
  - 12 个 Batch 的节点测试从旧 API 迁移到新 API
  - 16 个示例全部迁移到新 API 并验证通过（含 cartpole_sac RL 示例）

### 文档

- 更新 README：移除 `ModelState` 引用，更新为新 API 描述
- 更新动态图设计文档状态为"已完成"

## [0.11.0] - 2026-01-29

### 新增

- **feat(tensor): 实现统一的 Stack 操作**
  - 覆盖 PyTorch 的 `stack` 和 `cat` 功能
  - 支持 Tensor 层和节点层操作

- **feat(nn): 多输入/多输出 API**
  - `forward2`/`forward3` 多输入前向传播
  - `ModelState` 支持多输出及 `retain_graph` 反向传播
  - 新增 `dual_input_add`、`siamese_similarity`、`dual_output_classify`、`multi_io_fusion` 示例

- **feat(nn): 新增损失函数**
  - `MAE`（Mean Absolute Error）损失节点
  - `BCE Loss` 二元交叉熵损失（支持多标签分类）
  - `Huber Loss`（Smooth L1 Loss）

- **feat(metrics): 评估指标模块**
  - 分类指标：Accuracy、Precision、Recall、F1Score 等
  - 回归指标：MSE、MAE、R² 等
  - 统一 API，用户无需导入 `Metric` trait

- **feat(nn): Dropout 正则化节点**
  - 支持训练/推理模式自动切换

- **feat(tensor): Abs 绝对值算子**
  - Tensor 层和节点层完整支持

### 重构

- **refactor: 统一浮点类型为 f32**
  - 移除 `f64` 过度设计，简化代码

- **refactor: 统一损失节点命名**
  - `MSELoss` → `MSE`，与其他损失节点命名风格一致

### 文档

- docs: README 添加多输入/多输出示例说明

### 其他

- fix: rust lint 修复

## [0.10.2] - 2026-01-28

### 重构

- **refactor(graph): 模块化重构 graph.rs 为 graph/ 目录结构**
  - 拆分为 `core.rs`、`forward.rs`、`backward.rs`、`visualization.rs` 等子模块
  - 提升代码可维护性，为 NEAT 演化架构做准备

- **refactor(cnn): 统一 CNN 层为 Batch-First 4D 格式**
  - Conv2d/MaxPool2d/AvgPool2d 输入输出格式统一为 `[N, C, H, W]`

- **refactor: 统一术语，明确 Batch-First 设计原则**
  - 文档和代码注释统一使用 Batch-First 术语

- **refactor: 代码质量提升**
  - 统一错误信息格式，避免"节点"前缀重复
  - 改进参数文件格式错误的提示信息
  - 清理代码注释中的版本/阶段历史痕迹
  - 清理冗余代码并新增通用下载模块 (`src/utils/download.rs`)

### 文档

- **docs: 新增 NEAT 神经架构演化设计文档**
  - 新增 `.doc/design/neural_architecture_evolution_design.md`
  - 整合循环边变异机制设计
  - 为后续 NEAT/强化学习功能做架构准备

## [0.10.1] - 2026-01-25

### 新增

- **feat(rnn): 添加 RNN 展开缓存机制**
  - 支持动态 batch，避免重复展开相同序列长度的计算图

### 修复

- **fix(rnn): 修复 RNN/LSTM/GRU 缓存 key 问题**
  - 缓存 key 仅用 seq_len 导致变 batch 失效，现已修正

### 重构

- **refactor(vis): 统一可视化 API**
  - 默认启用层分组显示
  - 可视化边线从 ortho 改为 polyline
  - 优化循环层时间步标签及 ZerosLike 节点样式

### 文档

- 新增计算图可视化指南 (`.doc/design/visualization_guide.md`)

### 其他

- test: 更新 forward 行为测试以反映新设计
- chore: rust lint format

## [0.10.0] - 2026-01-25

### 重构

- **refactor(nn): 统一 Input 节点类型架构**
  - 将 `Input` 和 `GradientRouter` 统一为 `InputVariant` 枚举
  - 三种变体：`Data`（通用输入）、`Target`（Loss 目标值）、`Smart`（模型入口，原 GradientRouter）
  - 详见 [设计文档](.doc/design/input_node_unification_design.md)

- **refactor(nn): 可视化样式区分不同输入类型**
  - `Data`：浅蓝色，标签 `Input`
  - `Target`：浅橙色，标签 `Target`
  - `Smart`：浅绿色，标签 `Input`

### 新增

- **feat(examples): 所有示例添加计算图可视化**
  - 新增 `.dot` 和 `.png` 文件：xor、iris、sine_regression、california_housing、mnist、parity_rnn_fixed_len、parity_rnn_var_len、parity_lstm_var_len、parity_gru_var_len
  - 更新 mnist_gan 可视化

### 文档

- 新增 Input 节点统一设计文档
- README 可视化示例改用 examples 目录图片

## [0.9.0] - 2026-01-22

### 新增

- **feat(nn): DynamicShape 动态形状系统**
  - 新增 `DynamicShape` 类型，支持动态维度（类似 Keras 的 `None`）
  - 所有节点实现 `dynamic_expected_shape()` 和 `supports_dynamic_batch()`
  - `NodeDescriptor` 存储 `dynamic_shape` 用于可视化和序列化
  - 可视化中动态维度显示为 `?`（如 `[?, 128]`）

- **feat(nn): GradientRouter 节点和函数式 detach 机制**
  - 新增 `GradientRouter` 节点，支持动态梯度路由
  - 实现 `DetachedVar` 轻量 detach 包装
  - 支持 GAN 训练的 `fake.detach()` 模式

- **feat(nn): ModelState 智能缓存 + Criterion 损失封装**
  - `ModelState` 按特征形状缓存计算图，忽略 batch 维度
  - `MseLoss` / `CrossEntropyLoss` PyTorch 风格封装
  - `ForwardInput` trait 统一输入类型

- **feat(nn): PyTorch 风格 RNN/LSTM/GRU API**
  - `Rnn`/`Lstm`/`Gru` struct + `forward()` 模式
  - 支持变长序列（`BucketedDataLoader`）
  - `ZerosLike` 节点动态生成初始隐藏状态

- **feat(data): PyTorch 风格 DataLoader**
  - `DataLoader` 统一批处理接口
  - `BucketedDataLoader` 变长序列分桶

- **feat(tensor): argmax/argmin 方法**
  - 分类任务预测必需

### 示例

- 新增 10 个完整示例：
  - `xor`: 基础 MLP
  - `sine_regression`: 回归任务
  - `iris`: 多分类
  - `mnist`: 图像分类（MLP + CNN）
  - `mnist_gan`: GAN 训练 + detach
  - `california_housing`: 房价回归
  - `parity_rnn_fixed_len`: RNN 定长
  - `parity_rnn_var_len`: RNN 变长 + 智能缓存
  - `parity_lstm_var_len`: LSTM 变长
  - `parity_gru_var_len`: GRU 变长

### 修复

- fix(layer): RNN/LSTM/GRU 层 h0/c0 不再缓存，每次 forward 动态创建
  - 解决 `BucketedDataLoader` 变长批次的形状不兼容问题

### 重构

- refactor(nn): `check_shape_consistency` 使用 `DynamicShape.is_compatible_with_tensor()`
- refactor(seed): Graph seed 自动传播到 Layer

### 测试

- 单元测试从 822 增加到 1017
- 所有节点新增 DynamicShape 单元测试
- 新增 `node_softmax.rs`、`node_zeros_like.rs` 测试文件

## [0.8.0] - 2026-01-20

### ⚠️ 破坏性变更 (Breaking Changes)

- **refactor(layer)!: 统一所有 Layer 为 PyTorch 风格 API**
  - `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Rnn`, `Lstm`, `Gru` 统一为 struct + `forward()` 模式
  - 旧函数式 API 已删除
  - 详见 [架构 V2 设计](.doc/design/architecture_v2_design.md)

- **refactor(nn): 移除 `ScalarMultiply` 和 `ChannelBiasAdd` 节点**
  - 功能由通用 `Add`/`Subtract`/`Multiply` + 广播替代
  - `Conv2d` bias 形状从 `[1, C]` 改为 `[1, C, 1, 1]`

- **refactor(optimizer): 统一优化器 API**
  - V1 API 已删除，V2 成为默认实现
  - Optimizer 内部持有图引用，`zero_grad()`/`step()` 不再需要 `&mut Graph` 参数

### 新增

- **feat(tensor): 实现完整 NumPy 风格广播机制**
  - Tensor 层：8 个运算符（`+`/`-`/`*`/`/` 及其 `Assign` 版本）支持广播
  - Node 层：`Add`/`Subtract`/`Multiply`/`Divide` 支持广播
  - 工具函数：`broadcast_shape()`, `sum_to_shape()`
  - 新增 `Subtract` 节点
  - 详见 [广播机制设计](.doc/design/broadcast_mechanism_design.md)

- **feat(nn): 实现 Module trait 和 PyTorch 风格 API**
  - `Module` trait：`parameters()` 返回 `Vec<Var>`
  - `Var` 支持算子重载（`&a + &b`）和链式调用（`x.relu().sigmoid()`）
  - `Graph` 句柄：`Rc<RefCell<GraphInner>>` 允许 `Var` 持有图引用

### 重构

- refactor(layer): 简化 Layer 层，使用原生广播替代 `ones @ bias` 模式
- refactor(test): 改进 RNN/LSTM/GRU reset 测试的健壮性

### 文档

- docs: 更新架构 V2 设计文档，添加广播机制设计决策
- docs: 新增广播机制设计文档

### 测试

- 单元测试从 ~800 增加到 822+
- 新增 V2 集成测试：`test_mnist_linear_v2.rs`, `test_mnist_batch_v2.rs`

## [0.7.0] - 2026-01-08

### ⚠️ 破坏性变更 (Breaking Changes)

- **refactor(autodiff): 自动微分 API 统一 (Jacobian → VJP)**
  - 删除 Jacobian 模式，统一使用 VJP (Vector-Jacobian Product)
  - API 重命名：
    - `forward_node()` → `forward()`
    - `backward_nodes()` / `backward_batch()` → `backward()`
    - `clear_jacobi()` / `clear_grad()` → `zero_grad()`
    - `one_step()` / `one_step_batch()` / `update()` → `step()`
  - 删除：所有节点的 `jacobi` 字段、`calc_jacobi_to_a_parent()` 方法
  - `backward()` 返回 `f32` (loss 值)，简化训练循环
  - 详见 [自动微分统一设计](.doc/design/autodiff_unification_design.md)

## [0.6.0] - 2026-01-01

### 新增

- feat(layer): **Phase 3 完成** - RNN/LSTM/GRU Layer API
  - `rnn()`: Vanilla RNN 层 (h_t = tanh(x@W_ih + h_{t-1}@W_hh + b))
  - `lstm()`: LSTM 层 (4 门: 输入门、遗忘门、候选细胞、输出门)
  - `gru()`: GRU 层 (2 门: 重置门、更新门)
  - 所有层支持 BPTT 训练与层分组可视化
  - 集成测试验收：RNN 95.3%、LSTM 93.8%、GRU 90.6% 准确率
- feat: 实现 State 节点与 BPTT 循环机制
  - 支持时序状态记忆
  - `graph.step()` / `backward_through_time()` API
- feat: 添加 Sign 节点（Tensor 层 + NN 节点层）
  - 输出 {-1, 0, 1}，与 PyTorch 行为一致
- feat: 添加 Conv2d bias 支持与层分组可视化功能
  - 新增 ChannelBiasAdd 节点用于 bias 广播
  - 新增 `LayerGroup` 和 `save_visualization()` 实现层分组可视化

### 性能优化

- perf: 优化赋值算子 (+=/-=/*=/÷=) 并减少不必要的 clone
  - jacobi 累加、优化器梯度计算等处避免临时张量分配

### 重构

- refactor: 重组 Python 测试目录结构 (`tests/python/layer_reference/`)
- refactor(test): 增强 `assert_err!` 宏，支持多种简洁语法
  - 新增 `Variant(literal)`、`ShapeMismatch(exp, got, msg)` 等语法
  - 重构所有测试文件，消除冗长的 if guard 形式

### 测试

- test: 补充各层 PyTorch 数值对照及覆盖测试
  - 层测试总数从 128 增加到 143
  - 新增 AvgPool2d/MaxPool2d/Linear/Conv2d 的 forward/backward PyTorch 对照
  - 新增 RNN/LSTM/GRU batch_backward、chain_batch_training 等测试

### 文档

- docs: 新增五层架构设计文档 (`architecture_v2_design.md`)
- docs: 添加记忆机制设计文档及 NEAT/EXAMM 论文笔记
- docs: 更新梯度流控制设计文档
- docs: 修复 README 笔误 (waht→what, ndoes→nodes, fis→fix)

### 其他

- chore: 删除 README 中已完成的正确性验证 section（所有项已被现有测试覆盖）

## [0.5.0] - 2025-12-27

### 新增

- feat: 实现计算图序列化与可视化功能
  - `GraphDescriptor` 统一 IR 设计
  - `save_model()` / `load_model()` 模型保存加载（JSON + bin）
  - `to_dot()` / `save_visualization()` Graphviz 可视化
  - `summary()` / `summary_markdown()` Keras 风格摘要输出
- feat: 实现完整的梯度流控制机制
  - `no_grad_scope()` 无梯度作用域
  - `detach_node()` / `attach_node()` 梯度截断
  - `backward_nodes_ex(..., retain_graph)` 多次反向传播
- feat: 优化器 `with_params()` 方法，支持指定参数列表优化（用于 GAN/迁移学习）
- feat(Input): Input 节点拒绝设置雅可比矩阵

### 文档

- docs: 添加 Graph 序列化与可视化设计文档
- docs: 添加梯度流控制设计文档 (no_grad/detach/retain_graph)
- docs: README 添加计算图可视化展示
- docs: 精简 README TODO 列表

### 重构

- refactor: 将 Python 测试脚本移至 `tests/python/` 目录
- refactor: summary 标题改为中文「模型摘要」

### 其他

- chore: 添加 MNIST GAN 示例
- chore: 修正 GitHub 语言检测，忽略 issues 目录

## [0.4.0] - 2025-12-22

### 新增

- feat(layer): 实现 Linear 层（Batch-First 设计）
- feat: 实现 Conv2d 节点（2D 卷积）
- feat: 实现 MaxPool2d 节点（2D 最大池化）
- feat: 实现 AvgPool2d 节点（2D 平均池化）
- feat: 添加 CNN Layer 便捷函数 (conv2d, max_pool2d, avg_pool2d) 及 MNIST CNN 集成测试
- feat: 添加 Softplus 激活函数节点
- feat(nn): 实现 MSELoss 损失节点
- feat: California Housing 房价回归数据集与集成测试

### 性能优化

- perf: 使用 Rayon 并行化 CNN 层 (conv2d, max_pool2d, avg_pool2d)
- perf: 添加 dev profile 优化配置以加速 debug 模式下的计算密集测试
- perf: 为 SoftmaxCrossEntropy 添加 Rayon 并行优化

### 文档

- docs: 更新 CNN 节点状态为已完成

## [0.3.0] - 2025-12-21

### 新增

- feat: 实现 ScalarMultiply 和 Multiply 节点，修复 batch 训练梯度链
- feat: 添加带种子的随机函数以确保集成测试可重复性
- feat: 实现 Tanh 节点和 XOR 集成测试 (MVP M2+M3 完成)
- feat: M4 - 验证 Graph 动态扩展能力（NEAT 友好性）
- feat: M4b - Graph 级别种子 API
- feat: 实现 Sigmoid 激活节点 + jacobi_diag() 重构
- feat: 实现 SoftmaxCrossEntropyLoss 融合节点
- feat: 实现 data 模块（DataLoader + MNIST 数据集）
- feat: 实现 Batch Forward/Backward 机制
- feat: MNIST batch 测试添加 bias 支持
- feat: 实现 LeakyReLU/ReLU 激活函数节点
- feat: 为 Tensor 实现 AbsDiffEq trait，统一测试中的浮点比较
- feat: 实现 Reshape 节点
- feat: 实现 Flatten 节点

### 重构

- refactor: 统一集成测试命名规范
- refactor: 重构 tensor_slice 宏解决临时值生命周期问题

### 文档

- docs: 添加 API 分层与种子管理设计文档
- docs: 更新文档反映阶段二核心完成

### 其他

- chore: 统一术语规范，API 参数 axis 改为 dim

## [0.2.0] - 2025-12-20

### 新增

- feat: 实现优化器架构 (SGD/Adam) 及相关测试

### 重构

- refactor(optimizer): 模块化测试并封装内部实现细节

### 文档

- 架构设计重构：`.doc/high_level_architecture_design.md` 全面重写
- Hybrid 执行引擎设计：借鉴 MXNet hybrid 思想，设计 Eager/Graph 双模式执行方案
- 五层架构设计：用户 API 层、演化 API 层、执行引擎层、中间表示层、底层计算层
- OTMF 模型格式设计：OnlyTorch Model Format 规范，支持演化信息和跨语言部署
- NEAT 演化 API 设计：完整的演化模型接口、基因表示和演化引擎
- PyTorch 风格 API 设计：Module trait、functional 模块、优化器系统
- 整理全部文档

### 其他

- chore: update .gitignore
- chore: 将 MatrixSlow Python 参考项目纳入版本控制
- chore: 应用 clippy 和 rustfmt 自动修复

## [0.1.0] - 2025-07-23

### 文档

- 搁置底层计算图重构计划，当前重心为完善上层 API。
