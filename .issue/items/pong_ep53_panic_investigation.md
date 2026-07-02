# Pong Ep53 `Tensor::new` panic：未能复现，转被动监视（2026-07-02）

## 原始现象（仅一次，现场已丢失）

- **命令**：`SEEDS=3 cargo run --example my_zero_pong --release --features blas-mkl`
- **崩溃点**：seed 42，Ep 52 日志行之后（即 Ep 53 的 self-play 或训练期间）
- **错误**：`Tensor::new`（`tensor/mod.rs:101`）形状与数据长度不匹配（ndarray `from_shape_vec` OutOfBounds）
- 当时 `Tensor::new` 还没有增强版 panic 信息，具体是哪个张量、长度多少均未留下

## 复现尝试（均失败）

| 跑法 | 口径 | 结果 |
|------|------|------|
| Run A `MAX_EP=60` | temp 退火曲线与 150 局不同 | Ep 53–60 正常，`EXIT=0` |
| Run B 默认 150 局 | 与原始 seed 42 **完全一致** | Ep 53–60 正常，无 panic（Ep 60 后被重启杀掉） |

两次共耗时约 3h，均带 `RUST_BACKTRACE=full`。

## 静态排查结论（两轮，未找到确定性路径）

对 Pong 路径全部 `Tensor::new` 调用点做过两轮静态审查（`runner.rs` 训练循环 /
`obs_pipeline.rs` / `network.rs` batch 堆叠 / `target.rs` policy 目标）：

- `process_frame` 输出长度由构造期 h×w 决定，**与环境实际返回长度无关**（`to_gray`
  输出恒为 plane 长度，`chunks_exact` 对短输入只会静默截断），恒 7056；
- 堆叠恒 4×7056=28224；policy 目标经 `scatter_policy_target` 恒 action_dim；
  two-hot 恒 SUPPORT.size()=41；onehot / continuation 按固定维度预分配；
- MCTS latent 从图节点读回，长度恒 latent_dim。

**图像模式下不存在"数据长度随运行时变化"的确定性 `Tensor::new` 路径**。原崩溃属于
罕见数据依赖边角状态；MKL 多线程浮点归约不保证跨进程逐 bit 一致，轨迹分岔后无法用同
seed 踩回原状态。**根因不明，但非系统性**。

## 裁决：停止主动复现，被动陷阱就位

1. `Tensor::new` panic 信息已增强（打印 data len + shape），再发生即可自诊断张量类别；
2. 所有 RL 长跑统一带 `RUST_BACKTRACE=full`；
3. Phase 1 官方 3-seed 基准（`SEEDS=3`，与原崩溃同命令）继续照跑——基准本身即持续复现试验。

## 若再次触发

1. 从 panic 信息读 data len + shape，对照上文"固定长度表"定位张量类别；
2. 从 backtrace 定位 call site；
3. 记录当时 episode 号、buffer 状态（容量 32，Ep 32 起满）、是否在 eval / save 边界；
4. 回填本 issue 并转主动修复。

## 关联

- 示例入口：`examples/my_zero/pong/main.rs`
- 另一独立观察（不一定与 panic 相关）：单局 wall-clock 随 episode 增长
  （Ep1 ~2.5s → Ep52 ~200s，重启前系统亦处于降级状态）；由基准跑的 `PROFILE=1`
  分解数据另行裁决。
