//! v0.26 Phase 0：value/reward 编码消融 two-hot → HL-Gauss（CartPole 哨兵；手动档）。
//!
//! = Simulus 采纳计划 **A1**（`.doc/design/rl_myzero_status.md`）；
//! 历史编码消融臂（结论见 `.doc/design/rl_myzero_status.md` 与 CartPole 账本）。
//!
//! # 预注册协议（跑之前定死）
//!
//! - **单变量**：promoted recipe（cons2 + recon16 + Sampled）为底，仅切 `hl_gauss=true`
//!   （σ = 0.75 × bin 宽，非旋钮）；解码端不变。
//! - **口径**：release + MKL + seeds 42/43/44，3-seed 中位 env-steps-to-solved + 达标率。
//!   baseline 臂**不重跑**：账本当前哨兵（12,519 / 8,643 / 9,826，中位 ~9.8k，range 8.6k–12.5k）。
//! - **裁决（Farebrother/Simulus 文献先验 + CartPole 哨兵分辨率有限）**：
//!   - 持平（range 与哨兵重叠且 3/3 达标）→ **promote**（文献口径 HL-Gauss ≥ two-hot，
//!     CartPole 分不出高下时按先验取 HL-Gauss，真正收益押图像线大 value 噪声场景）；
//!   - 显著劣化（range 完全更差不重叠，或达标率 < 3/3）→ 回退 two-hot、负结果留档；
//!   - 显著更优 → promote（直接受益）。
//! - CartPole 单环境证据为**临时结论**；Phase 1 图像域 native 复裁后定稿。
//!
//! ```bash
//! cargo test --release --features blas-mkl cartpole_hlgauss -- --ignored --nocapture --test-threads=1
//! ```
//!
//! 实测结果回填 `examples/my_zero/cartpole/README.md`（唯一基准账本）。

use crate::nn::GraphError;
use crate::rl::algo::my_zero::MyZero;
use crate::rl::algo::my_zero::runner::train_all_seeds;

/// HL-Gauss 臂：promoted recipe 为底，仅切编码（3 seeds）。
#[test]
#[ignore = "manual: v0.26 Phase 0 编码消融 HL-Gauss"]
fn cartpole_hlgauss_arm() -> Result<(), GraphError> {
    let mut cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .seeds(3)
        .build()?;
    cfg.components.hl_gauss = true;
    println!(
        "[hl-gauss-ablation] CartPole-v1 arm=HL-Gauss · cons={} recon={} cont={} · SEEDS=3",
        cfg.components.consistency_coef,
        cfg.components.reconstruction_coef,
        cfg.components.continuation_coef,
    );
    train_all_seeds(cfg)?;
    Ok(())
}
