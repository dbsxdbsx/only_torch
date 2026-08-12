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
| num_simulations | 25 | CPU 单 worker |
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

**下一步（未跑，需立项）**：在同一预算纪律下一次一项调参（降 lr / 增 sims / 缩短 td_steps）或接受「CPU 墙」后评估 planning-free 退路；组件消融仍等绿灯。

### Breakout-v1 Base Pilot（历史，2026-07-23）

| seed | best greedy | env-steps | wall-clock | 判定 |
|------|-------------|-----------|------------|------|
| 1 | **4.1** / final 4.1 | 63,656 | ~90 min（`MAX_EP=2000`，未接 2h 预算） | ⚠️ 单 seed 试探 |

### Freeway-v1（待 Breakout 绿灯后）

| seed | best greedy | env-steps | wall-clock | 判定 |
|------|-------------|-----------|------------|------|
| — | — | — | — | 待执行 |
