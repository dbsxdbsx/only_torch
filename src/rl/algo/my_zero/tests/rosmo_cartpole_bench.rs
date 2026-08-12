//! ROSMO 阶梯一 · CartPole 回归闸门（手动档，不纳入 CI）。
//!
//! # 定位（条款二：CartPole 只答「崩没崩」，不答「好不好」）
//!
//! reanalyze 复活阶梯一（[RL 状态总览 · backlog](../../../../../.doc/design/rl_myzero_status.md#5-未完成事项收口时的-backlog)）的
//! **回归覆盖 + bug 探测**，不做价值裁决（价值裁决在图像域 Pong A/B，须先立无
//! reanalyze 图像基线）。
//!
//! **bug 探测逻辑**：旧全树 reanalyze 在 CartPole 灾难性失败时，「实现 bug」与「机制病理
//! （弱网深搜投毒）」不可分（reanalyze issue §三假设 4）。ROSMO 一步展开对弱模型鲁棒
//! （arXiv:2210.05980 论文主张 + BSuite cartpole 实测 ~990/1000），若它在 CartPole 上
//! 仍灾难性失败（greedy 长期钉在随机水平），则强指向我方管线实现 bug 而非机制问题。
//!
//! # 预注册判读（跑之前定死）
//!
//! - **口径**：release + MKL · seeds 42/43/44 · promoted recipe + `.rosmo(true)`（单变量）；
//! - **绿（回归通过）**：3-seed 内 ≥2 达标（greedy ≥475 within 2000 局），中位 env-steps
//!   不超过哨兵基线（~8.7k）的 **5×**——不要求更快（样本免费环境测省样本组件，信号天然弱）；
//! - **黄（可疑劣化）**：达标 <2/3 或中位 >5× 基线 → 记 issue，查 α / bootstrap 口径，不 promote；
//! - **红（bug 指向）**：greedy 全程 ≈ 随机（~9-10 分不动）→ 按实现 bug 排查
//!   （写回已消灭，首查现算 target 槽位对齐与 BC 权重符号）。
//! - 结果只回填 reanalyze issue 复活条目与 Pong 账本备注，**不改 CartPole recipe**。
//!
//! ```bash
//! cargo test --release --features blas-mkl cartpole_rosmo -- --ignored --nocapture --test-threads=1
//! ```

use crate::nn::GraphError;
use crate::rl::algo::my_zero::MyZero;
use crate::rl::algo::my_zero::runner::train_all_seeds;

/// ROSMO 回归闸门：promoted recipe + rosmo（单变量），seeds 42/43/44。
#[test]
#[ignore = "manual: ROSMO 阶梯一 CartPole 回归闸门 × 3 seeds"]
fn cartpole_rosmo_regression_gate() -> Result<(), GraphError> {
    let cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .rosmo(true)
        .seeds(3)
        .build()?;
    println!(
        "[rosmo-gate] CartPole-v1 · promoted recipe + ROSMO(α={}) · SEEDS=3 · 判读见模块文档",
        cfg.components.rosmo_alpha
    );
    train_all_seeds(cfg)?;
    Ok(())
}
