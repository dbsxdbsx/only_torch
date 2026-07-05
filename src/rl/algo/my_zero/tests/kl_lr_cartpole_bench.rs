//! KL 自适应 lr · CartPole「等效不劣」闸门（手动档，不纳入 CI）。
//!
//! # 定位（去旋钮组件的公共化验收，2026-07-05 与用户定稿）
//!
//! `kl_adaptive_lr` 是「用户不调 lr」愿景的第一块落地件——**验收标准 =
//! 等效不劣 + 少一个超参，不要求正增益**（价值主张是去旋钮，非提分）。
//! 棋盘域 ⑨ 臂已验证无害（batch 512 被自动配平、护栏全绿）；本闸门裁决
//! 单智能体域（CartPole promoted recipe）能否 promote 默认开。
//!
//! # 预注册判读（跑之前定死）
//!
//! - **口径**：release + MKL · seeds 42/43/44 · promoted recipe + `.kl_adaptive_lr(true)`
//!   （单变量 A/B，对照 = 官方哨兵中位 ~8.7k env-steps）；
//! - **等效不劣（promote）**：3/3 达标（greedy ≥475 within 2000 局）且中位 env-steps
//!   ≤ 哨兵基线的 **2×**——lr 自适应在已调好 lr 的域允许小幅波动，但不得档位级劣化；
//!   promote 动作 = recipe 默认开 + **重标哨兵基线**（历史可比性以重标点为界）；
//! - **劣化（不 promote）**：达标 <3/3 或中位 >2× 基线 → 留库默认关，
//!   组件矩阵 CartPole 格记 ❌，触发条件回归「某域正增益后再议」。
//!
//! # ✅ 已裁决（2026-07-05）：❌ 灾难级失败，不 promote
//!
//! 0/3 达标（greedy 3-seed 全程钉死 ~9.2 ≈ 随机水平，wall ~140s/seed）。
//! 诊断（单 seed 轨迹 `.bench/cartpole_kl_lr_diag_20260705.log`）= **lr 死亡螺旋**：
//! CartPole batch 8 × buffer 1000 下当局探针局面大概率不被训练触碰 → 测得 KL ≈
//! 噪声（1e-5–0.47 乱跳）→ 连续偏小时乘子棘轮 1.5× 顶到 10× 上限（lr=0.2，
//! ep≈34）→ 网络被打散（KL 暴跳 0.24）→ 训练永不起步。**默认路径对照完好**
//! （同 HEAD 不开 KL-lr 单 seed 8,741 env-steps 达标 = 哨兵基线），非搬运 bug。
//! 域适配前提修正：机制要求「每局训练样本量大 + 探针与训练分布相关」（棋盘
//! batch 512×5/buffer 300 成立；参照实现原口径是在**刚训 minibatch 上**测 KL）。
//! 重试条件 = 改 minibatch-KL 口径后重过本闸门。裁决入组件矩阵 CartPole 格。
//!
//! ```bash
//! cargo test --release --features blas-mkl cartpole_kl_lr -- --ignored --nocapture --test-threads=1
//! ```

use crate::nn::GraphError;
use crate::rl::algo::my_zero::MyZero;
use crate::rl::algo::my_zero::runner::train_all_seeds;

/// KL-lr 闸门：promoted recipe + kl_adaptive_lr（单变量），seeds 42/43/44。
#[test]
#[ignore = "manual: KL 自适应 lr CartPole 等效不劣闸门 × 3 seeds"]
fn cartpole_kl_lr_equivalence_gate() -> Result<(), GraphError> {
    let cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .kl_adaptive_lr(true)
        .seeds(3)
        .build()?;
    println!(
        "[kl-lr-gate] CartPole-v1 · promoted recipe + kl_adaptive_lr(kl_targ=0.02) · SEEDS=3 · 判读见模块文档（对照哨兵中位 ~8.7k）"
    );
    train_all_seeds(cfg)?;
    Ok(())
}
