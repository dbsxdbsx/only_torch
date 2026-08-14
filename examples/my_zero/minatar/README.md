# MyZero · MinAtar（小网格图像）

> MinAtar（Young & Tian 2019）：10×10 二值通道网格版 Atari，保留策略难度、移除视觉感知难度。
> 用作图像域效率组件的裁决场（门槛未达标前不做组件 promote）；战略状态见 [RL 状态总览](../../../.doc/design/rl_myzero_status.md)。

## 环境特性

| 游戏 | obs shape | channels | 动作数 | reward 类型 | 平均局长 |
|------|-----------|----------|--------|-------------|----------|
| Breakout-v1 | (10,10,4) | 4 | 3 | 稠密（+1/brick） | ~25 步 |
| Freeway-v1 | (10,10,7) | 7 | 3 | 稀疏（+1/到顶） | ~50 步 |

## 预注册协议

### DQN 文献基线（条款三锚定）

| 游戏 | DQN 参考 | 随机基线 |
|------|----------|----------|
| Breakout | ~16-27 | ~1-2 |
| Freeway | ~55-60 | ~0 |

### 预注册门槛（证明"管线能学"，非 SOTA）

| 游戏 | 门槛 | 判据 |
|------|------|------|
| **Breakout** | 3-seed 中位 best greedy ≥ **8** | 明确脱离随机 ~1-2 |
| **Freeway** | 3-seed 中位 best greedy ≥ **20** | 明确脱离随机 ~0 |

### 预算

- 每 seed: **200k env-steps** 或 **wall-clock 2h** 先到为准（`main.rs` 已接线：`max_env_steps` / `max_wall_secs`；局数安全上限 20k）
- 中间判停：100k env-steps 时若 peak greedy 仍 < 3（Breakout）/ < 1（Freeway），提前结束（`flat_stop`）

### 退出判据

| 结果 | 条件 | 后续 |
|------|------|------|
| 🟢 绿灯 | 任一游戏达标 | 进入组件消融（见下方优先级） |
| 🔴 红灯 | 双游戏皆平直 | planning-free 退路评估上桌 |
| 🧊 冻结 | 组件消融完成后 | MinAtar 降级为回归哨兵（防元过拟合） |

## 训练口径

| 超参 | 值 | 备注 |
|------|-----|------|
| gamma | 0.99 | 短局无需长折扣 |
| k_unroll | 5 | 标准 |
| td_steps | 10 | n-step return |
| num_simulations | 25 | 思考 / greedy eval / reanalyze；`SIMS` 仍覆盖此项 |
| acting_simulations | 1 | self-play / `run` 只出 π；要动手前搜索用 `.acting_simulations(n)` |
| lr | 0.005 | |
| train_batch_size | 32 | |
| trains_per_episode | 16 | |
| buffer_capacity | 128 (按局) | |
| latent_dim | 64 | BoardConvRepresentationNet |
| representation | Conv3×3 stride-1 (N→32→64) → fc | 与 Gomoku 同架构 |
| 组件 | consistency ON, reconstruction OFF | minatar_base_stack |
| stochastic | K=8 always-on | 默认配置 |

## 组件消融优先级（绿灯后）

每臂 3-seed，半额预算（100k steps/seed），与 base 配对比较：

1. **ROSMO reanalyze** — offline 效率第一张账单
2. **target_net + SVE** — 门槛达标后解封
3. **HL-Gauss** — 图像域复测
4. **loss 优先回放** — 效率提升

## 运行方式

```bash
# Smoke（管线验证，3 局）
SMOKE=1 cargo run --example my_zero_minatar

# 单 seed 正式（Breakout，默认）
cargo run --example my_zero_minatar --release --features blas-mkl

# 3-seed 统计
SEEDS=3 cargo run --example my_zero_minatar --release --features blas-mkl

# 切换游戏
GAME=Freeway SEEDS=3 cargo run --example my_zero_minatar --release --features blas-mkl

# 自定义局数
MAX_EP=3000 cargo run --example my_zero_minatar --release --features blas-mkl

# 诊断覆盖（不改 recipe 默认）：搜索次数 / n-step 步数 / wall-clock 秒 / 独立 target net
SIMS=8 WALL_SECS=3600 cargo run --example my_zero_minatar --release --features blas-mkl
TARGET_NET=1 WALL_SECS=3600 cargo run --example my_zero_minatar --release --features blas-mkl
TD_STEPS=5 WALL_SECS=3600 cargo run --example my_zero_minatar --release --features blas-mkl
```

## 账本

### Breakout-v1 · 正式 3-seed（2026-08-12）

> 口径：预注册预算 **2h / 200k 先到**（本轮三 seed 均先触 **wall-clock 2h**，未到 200k）；门槛 = 中位 **best** greedy ≥ 8。  
> 日志：`.bench/minatar_breakout_3seed_20260812_085356.log`

| seed | best greedy（训练期 peak） | final greedy（训末） | env-steps | wall-clock | 判定 |
|------|---------------------------|----------------------|-----------|------------|------|
| 42 | **4.1** | 3.6 | 65,686 | 7202s | ❌ < 8 |
| 43 | **2.1** | 1.3 | 42,772 | 7202s | ❌ < 8 |
| 44 | **3.9** | 2.7 | 73,778 | 7202s | ❌ < 8 |
| **中位** | **3.9** | **2.7** | — | ~7202s | **0/3 达标** |

**判读**：三 seed 均能脱离随机（~1–2），但 2h 预算内锁在约 **2–4** 分平台，**未绿灯**。吞吐瓶颈明显（2h 仅 ~43k–74k env-steps，远低于名义 200k）。不满足「双游戏皆平直」红灯；Freeway 仍按协议暂缓。

**主轴**：MinAtar 是图像效率裁决场（以点破面），不是给 Breakout 炼专属配方。只收已验证、可按环境结构路由的组件。FNC / COMBO 整包不进主线；ROSMO 默认关（无裁判探针不开）。

### Breakout-v1 · acting-sims 诊断（已收口，2026-08-13）

> 唯一变量：`num_simulations` 25→8（环境变量 `SIMS`，**不写入 recipe**）。  
> 预算：单 seed=42，`WALL_SECS=3600`（1h），其余配方与正式 3-seed 相同。  
> 问题：同一 wall-clock 内少搜后，best greedy 是仍锁 2–4、随 env-steps 上涨，还是变差？  
> **禁止**：不把 8 写入默认；无与游戏无关的路由规则则不 promote。

| seed | sims | best greedy | final greedy | env-steps | wall-clock | 判定 |
|------|------|-------------|--------------|-----------|------------|------|
| 42 | 8 | **1.5**（~6.4k steps） | 1.3 | 27,980 | 3601s（先触 1h） | ❌ 未达标；训末 eval best=1.5 |

对照（正式 3-seed seed 42）：sims=25，best **4.1**，65,686 steps @ 2h。

**判读**：

- **少搜没有买到吞吐**。1h 仅 27,980 env-steps（约 7.8 step/s）；对照 seed 42 在 sims=25 下约 9.1 step/s。wall-clock 仍被每局 16 次 unroll 训练占满；策略更差 → 局更短 → 每小时更多「局末训练税」，steps 反而更少。
- **学习更差，不是同平台更快涨**。greedy 在 ~6.4k steps 摸到 1.5 后锁在 1.2–1.3，1h 未随 steps 抬向对照的 2–4。
- **不 promote** `sims=8`。下一刀不要砍 `trains_per_episode`；若动训练量只允许加，且须「相近 env-steps 下 greedy 是否抬升」对照。

日志：`.bench/minatar_breakout_sims8_1h_20260813.log`

### Breakout-v1 · target_net 独立接线诊断（已收口，2026-08-13）

> 唯一变量：`TARGET_NET=1`（n-step 尾值改用滞后网 V + hard update 400 训练步）。  
> **不开** reanalyze / ROSMO；`SIMS` 不设（默认 **25**）。  
> 对照：正式 3-seed seed 42（sims=25，best **4.1**，65,686 steps @ 2h）。  
> **禁止**：不写入 recipe。

| seed | target_net | best greedy | final greedy | env-steps | wall-clock | 判定 |
|------|------------|-------------|--------------|-----------|------------|------|
| 42 | ON | **2.3**（~15.4k steps） | 1.9 | 26,096 | 标称 1h；日志 9916s | ❌ 未达标；训末 eval best=2.3 |

**判读**：

- **等步数仍低于对照**。~26k steps 时 greedy 约 1.6–1.9，best 2.3；对照 seed 42 在 ~26.6k 已是 3.3，且更早在 ~20k 到过 3.9、~32k 到 4.1。
- **没有换成「更稳的更高平台」**。摸到 2.3 后仍在 1.5–2.3 晃，不是少回撤的 3–4。
- **早期更钝成立，后期没追上。** 不 promote。SVE / ROSMO 不要绑进来救它。
- **wall 9916s 不是多训了 2.75 小时。** 第 1118 局只有 6 步，却记了 `t=6839s`（邻局约 3.2s）。预算在**每局结束**才检查，这一局卡住后才触发 `[budget]`。卡住前约 1117 局 / 2.6 万步，有效训练仍是 1h 量级。

日志：`.bench/minatar_breakout_targetnet_1h_20260813.log`

### Breakout-v1 · td_steps 10→5 诊断（已收口，2026-08-13）

> 唯一变量：`TD_STEPS=5`（n-step bootstrap 10→5，对齐 canonical MuZero / `k_unroll`）。  
> `SIMS` / `TARGET_NET` 不设（sims=25，无独立 target net）。  
> 对照：正式 3-seed seed 42（td_steps=10，best **4.1** @ ~32k）。  
> **禁止**：不写入 recipe，默认仍 10。

| seed | td_steps | best greedy | final greedy | env-steps | wall-clock | 判定 |
|------|----------|-------------|--------------|-----------|------------|------|
| 42 | 5 | **2.6**（~25.7k steps） | 2.4 | 32,134 | 3602s（先触 1h） | ❌ 未达标；训末 eval best=2.6 |

**判读**：

- **等步数低于对照**。~32k 时 greedy 2.3–2.4、全程 best 2.6；对照 seed 42 在 ~32k 到 **4.1**，且更早 ~4k 已 3.0、~20k 到 3.9。本臂 ~4k 仅 1.1、~20k 仅 2.1。
- **几乎不加税，也没买吞吐**。约 8.9 step/s，与对照 sims=25 的 ~9.1 同量级。缩短 n 没有让训练变快，只是目标更短、学得更浅。
- **不 promote** `td_steps=5`。2–4 分天花板不是「n=10 把目标搞噪了」。

日志：`.bench/minatar_breakout_tdsteps5_1h_20260813.log`

### Breakout-v1 Base Pilot（历史，2026-07-23）

| seed | best greedy | env-steps | wall-clock | 判定 |
|------|-------------|-----------|------------|------|
| 1 | **4.1** / final 4.1 | 63,656 | ~90 min（`MAX_EP=2000`，未接 2h 预算） | ⚠️ 单 seed 试探 |

### Freeway-v1（待 Breakout 绿灯后）

| seed | best greedy | env-steps | wall-clock | 判定 |
|------|-------------|-----------|------------|------|
| — | — | — | — | 待执行 |
