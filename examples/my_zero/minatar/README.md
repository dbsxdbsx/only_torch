# MyZero · MinAtar 第三支柱

> MinAtar（Young & Tian 2019）：10×10 二值通道网格版 Atari，保留策略难度、移除视觉感知难度。
> 作为 MyZero 的第三支柱裁决场，用于批量终审积压的 offline/效率组件。

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

- 每 seed: **200k env-steps** 或 **wall-clock 2h** 先到为准
- 中间判停：100k env-steps 时若曲线仍完全平直（Breakout greedy < 3），记录并停止

### 退出判据

| 结果 | 条件 | 后续 |
|------|------|------|
| 🟢 绿灯 | 任一游戏达标 | Phase 3 解封，进入组件消融 |
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
2. **target_net + SVE** — Phase 3 解封条件
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

### Breakout-v1 Base Pilot

| seed | best greedy | env-steps | wall-clock | 判定 |
|------|-------------|-----------|------------|------|
| 1 | **4.1**（峰值 eval）/ final eval 4.1 | 63,656 | ~90 min | ⚠️ 学到了但未达门槛 8 |

**学习曲线摘要**（单 seed，2000 局，2026-07-23）：
- Ep 50-100: greedy 0.2（随机水平）
- Ep 150: greedy 2.0（开始学习）
- Ep 300: greedy 3.2
- Ep 1050: greedy **4.1**（峰值）
- Ep 1500-2000: greedy 波动 2.5-3.7（疑似收敛/轻微过拟合）
- final best model eval(×10): **4.1**
- avg_R(self-play): 2.8

**诊断**：管线明确能学（脱离随机 ~1-2），但在 ~64k env-steps 内陷入 3-4 分平台。
可能瓶颈：训练预算不足（仅用 64k / 200k）/ td_steps 过长 / sims 不够 / lr 偏高。
下一步：调参（增大 max_episodes / 降 lr / 增 sims）后再试；或直接 3-seed 确认趋势。

### Freeway-v1（待 Breakout 绿灯后）

| seed | best greedy | env-steps | wall-clock | 判定 |
|------|-------------|-----------|------------|------|
| — | — | — | — | 待执行 |
