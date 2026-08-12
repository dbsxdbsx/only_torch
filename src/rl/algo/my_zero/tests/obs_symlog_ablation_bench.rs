//! v0.26 Phase 0：obs 无量纲化（symlog）消融（CartPole 哨兵；手动档）。
//!
//! 历史 symlog 消融臂（结论见 `.doc/design/rl_myzero_status.md` 与 CartPole 账本）：全家 loss 仅
//! reconstruction 直接暴露在环境单位下，symlog 统一量纲后 recon 系数才可跨环境迁移。
//!
//! # 预注册协议（跑之前定死）
//!
//! - **落点**：`MyZeroModel` obs 入口单点 `symlog(x)=sign(x)·ln(1+|x|)`（repr 输入 +
//!   recon 目标同源；无状态；buffer/env 恒存 raw）。开关 `Components.obs_symlog`（默认关）。
//! - **三臂**：symlog × recon_coef {16, 4, 1}，各 3-seed（42/43/44）；
//!   其余 recipe 不动（cons=2 · Sampled · sims=20 · td=5）。
//! - **口径**：release + MKL，3-seed 中位 env-steps-to-solved + 达标率。
//!   baseline 不重跑：账本当前哨兵（12,519 / 8,643 / 9,826，中位 ~9.8k，range 8.6k–12.5k）。
//! - **裁决**：
//!   - 小系数臂（1 或 4）与 16 臂打平（range 重叠）→ 按简单性取小系数并 promote symlog
//!     （系数回归论文默认 = 量纲工作被撤走的证明）；
//!   - symlog 各臂均显著差于哨兵（range 完全更差不重叠 / 达标率 < 3/3）→ 负结果留档、
//!     保持现状（量纲问题留图像线 [0,1] 归一自然解决）；
//!   - CartPole 判据 = 「不显著变差 + 最优系数向 1 回移」，**不预期变快**。
//! - **终极兑现判据在 Phase 1**：图像环境 recon 系数免重调。
//!
//! ```bash
//! cargo test --release --features blas-mkl cartpole_symlog -- --ignored --nocapture --test-threads=1
//! ```
//!
//! 实测结果回填 `examples/my_zero/cartpole/README.md`（唯一基准账本）。

use crate::nn::GraphError;
use crate::rl::algo::my_zero::MyZero;
use crate::rl::algo::my_zero::runner::train_all_seeds;

/// 跑一个 symlog 臂：promoted recipe 为底，开 obs_symlog 并设 recon_coef（3 seeds）。
fn run_symlog_arm(recon_coef: f32) -> Result<(), GraphError> {
    let mut cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .seeds(3)
        .build()?;
    cfg.components.obs_symlog = true;
    cfg.components.reconstruction_coef = recon_coef;
    println!(
        "[symlog-ablation] CartPole-v1 arm=symlog+recon{recon_coef} · cons={} cont={} · SEEDS=3",
        cfg.components.consistency_coef, cfg.components.continuation_coef,
    );
    train_all_seeds(cfg)?;
    Ok(())
}

/// s1：symlog + recon_coef=16（现默认系数）。
#[test]
#[ignore = "manual: v0.26 Phase 0 symlog 消融 recon=16"]
fn cartpole_symlog_s1_recon16() -> Result<(), GraphError> {
    run_symlog_arm(16.0)
}

/// s2：symlog + recon_coef=4。
#[test]
#[ignore = "manual: v0.26 Phase 0 symlog 消融 recon=4"]
fn cartpole_symlog_s2_recon4() -> Result<(), GraphError> {
    run_symlog_arm(4.0)
}

/// s3：symlog + recon_coef=1（论文默认；系数回移的目标点位）。
#[test]
#[ignore = "manual: v0.26 Phase 0 symlog 消融 recon=1"]
fn cartpole_symlog_s3_recon1() -> Result<(), GraphError> {
    run_symlog_arm(1.0)
}
