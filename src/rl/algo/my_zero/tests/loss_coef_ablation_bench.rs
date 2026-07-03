//! v0.26 P0：loss 系数重标定消融（CartPole 哨兵环境；手动档，不纳入 CI）。
//!
//! # 背景（为什么只测 reconstruction / continuation）
//!
//! autograd `upstream_grad` 修复（`b379ab4`）只影响 **MSE 系**反向——MyZero 中即
//! **reconstruction** 与 **continuation** 两项辅助 loss；consistency（负余弦，由原语组合）
//! 与 policy/value/reward（SoftmaxCrossEntropy）修复前后均正确，**无重标定的逻辑理由**。
//!
//! bug 时代的等效放大倍数可推导：MSE 反向丢上游 `step_scale (1/K=1/5)` 与
//! batch 均值 `(1/B=1/8)`，故 unroll 步（k≥1）等效系数 ≈ **K×B = 40**（k=0 recon ≈ 8）。
//! 旧 ~13.1k env-steps 基线即在该隐式放大下取得。
//!
//! # 预注册协议（跑之前定死，防事后解释）
//!
//! - **档位**：对数间隔 **{1（论文默认）, 4, 16}**，锚点 = 1 与 ~40（bug 等效）之间；
//!   不做精调（要的是汇率的数量级，不是小数点——"模糊的正确"）。
//! - **口径**：release + MKL + seeds 42/43/44，3-seed 中位 env-steps-to-solved + 达标率
//!   （与账本一致）。baseline 臂**不重跑**：直接用账本 t3 promoted
//!   （45,308 / 82,720 / 66,166，中位 66.2k）。
//! - **裁决**：候选臂 3/3 达标 **且** 三 seed range 与 baseline range（45.3k–82.7k）
//!   **不重叠且更优** → 才提案改 recipe 默认（另行一次一项过哨兵）；
//!   range 重叠 → 判「无差异，保持论文默认」。
//! - **单变量**：每臂只动一个系数；若两个单变量臂各自显著，才追加联合臂。
//! - CartPole 单环境证据只作**临时结论**；图像 / 第二环境复验后才定稿。
//!
//! # seed 扩展（reconstruction 去留复裁，同属 P0）
//!
//! v0.25 遗留悬案：t1(+cons) 中位 17.5k 反而优于 t2/t3 的 66.2k，但 t1 方差极大
//! （3.5k–151k），3 seeds 不足裁决。t1 与 t3 各补 seeds **45/46**，与账本已有 42/43/44
//! 合并成 5-seed 中位后再判。
//!
//! ```bash
//! # 系数消融（4 臂，各 3 seeds）
//! cargo test --release --features blas-mkl cartpole_coef -- --ignored --nocapture --test-threads=1
//! # seed 扩展（t1 / t3 各补 45/46）
//! cargo test --release --features blas-mkl cartpole_seedext -- --ignored --nocapture --test-threads=1
//! ```
//!
//! 实测结果回填 `examples/my_zero/cartpole/README.md`（唯一基准账本）。
//!
//! # 哨兵红灯复裁（2026-07-03 · 新数值流；见 `.issue/items/cartpole_sentinel_red_ndarray_drift.md`）
//!
//! ndarray 0.16/0.17 升级致 BLAS 轨迹漂移后，recon=16 达标率跌破门槛（3/5 与 1/3）。
//! 按红灯 issue 待办，在**新依赖栈 + 当前 HEAD**上重跑系数矩阵重点档位：
//!
//! - **档位**：recon ∈ {4, 16}（旧网格的两个候选平台点），单变量、其余 promoted recipe 不动；
//! - **口径**：release + MKL · seeds 42–46（5-seed）· 中位 env-steps + 达标率；
//!   旧数值流数字（recon16 中位 12.5k / 5/5）仅作历史参考，不作对照臂；
//! - **裁决（预注册）**：达标率 ≥ 4/5 的臂中取中位更优者重新 promote；
//!   两臂均 <4/5 → 升级追加温度调度常数臂（红灯 issue 可选臂）再裁；
//! - 结果回填账本「哨兵红灯」节并收口该 issue。
//!
//! ```bash
//! cargo test --release --features blas-mkl cartpole_recal -- --ignored --nocapture --test-threads=1
//! ```

use super::super::component::Components;
use crate::nn::GraphError;
use crate::rl::algo::my_zero::MyZero;
use crate::rl::algo::my_zero::runner::train_all_seeds;

/// 跑一个系数臂：promoted recipe（cons+recon+Sampled）为底，只改指定系数（3 seeds）。
fn run_coef_arm(label: &str, mutate: impl FnOnce(&mut Components)) -> Result<(), GraphError> {
    let mut cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .seeds(3)
        .build()?;
    mutate(&mut cfg.components);
    println!(
        "[coef-ablation] CartPole-v1 arm={label} · cons={} recon={} cont={} · SEEDS=3",
        cfg.components.consistency_coef,
        cfg.components.reconstruction_coef,
        cfg.components.continuation_coef,
    );
    train_all_seeds(cfg)?;
    Ok(())
}

/// a1：reconstruction coef 1 → 4（其余不动）。
#[test]
#[ignore = "manual: v0.26 P0 系数消融 recon=4"]
fn cartpole_coef_a1_recon4() -> Result<(), GraphError> {
    run_coef_arm("recon=4", |c| c.reconstruction_coef = 4.0)
}

/// a2：reconstruction coef 1 → 16。
#[test]
#[ignore = "manual: v0.26 P0 系数消融 recon=16"]
fn cartpole_coef_a2_recon16() -> Result<(), GraphError> {
    run_coef_arm("recon=16", |c| c.reconstruction_coef = 16.0)
}

/// a3：continuation coef 1 → 4。
#[test]
#[ignore = "manual: v0.26 P0 系数消融 cont=4"]
fn cartpole_coef_a3_cont4() -> Result<(), GraphError> {
    run_coef_arm("cont=4", |c| c.continuation_coef = 4.0)
}

/// a4：continuation coef 1 → 16。
#[test]
#[ignore = "manual: v0.26 P0 系数消融 cont=16"]
fn cartpole_coef_a4_cont16() -> Result<(), GraphError> {
    run_coef_arm("cont=16", |c| c.continuation_coef = 16.0)
}

/// a5：reconstruction coef 边界外扩核查（16 为预注册网格边缘赢家；本臂只判「16 是否在平台上」，
/// 不用于继续加码——若 64 未显著优于 16，默认取 16）。
#[test]
#[ignore = "manual: v0.26 P0 系数消融 recon=64（边界核查）"]
fn cartpole_coef_a5_recon64() -> Result<(), GraphError> {
    run_coef_arm("recon=64", |c| c.reconstruction_coef = 64.0)
}

/// 红灯复裁公共路径：promoted recipe 为底、只改 recon 系数，seeds 42–46 一次跑齐 5-seed。
fn run_recal_arm(label: &str, recon: f32) -> Result<(), GraphError> {
    let mut cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .seeds(5)
        .build()?;
    cfg.components.reconstruction_coef = recon;
    println!(
        "[sentinel-recal] CartPole-v1 arm={label} · recon={recon} · seeds 42-46 · 新数值流（ndarray 0.17 + 当前 HEAD）"
    );
    train_all_seeds(cfg)?;
    Ok(())
}

/// 红灯复裁 r1：recon=4（旧网格次优档，漂移后可能反超）。
#[test]
#[ignore = "manual: 哨兵红灯复裁 recon=4 × 5 seeds"]
fn cartpole_recal_r1_recon4() -> Result<(), GraphError> {
    run_recal_arm("recon=4", 4.0)
}

/// 红灯复裁 r2：recon=16（现 promoted 默认，红灯当事档）。
#[test]
#[ignore = "manual: 哨兵红灯复裁 recon=16 × 5 seeds"]
fn cartpole_recal_r2_recon16() -> Result<(), GraphError> {
    run_recal_arm("recon=16", 16.0)
}

/// seed 扩展公共路径：base_seed=45、seeds=2（即 45/46），与账本 42/43/44 合并成 5-seed。
fn run_seed_ext(label: &str, components: Components) -> Result<(), GraphError> {
    let mut cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .seed(45)
        .seeds(2)
        .build()?;
    cfg.components = components;
    println!("[seed-ext] CartPole-v1 tier={label} · seeds 45/46 · max_episodes=2000");
    train_all_seeds(cfg)?;
    Ok(())
}

/// t1(+consistency) 补 seeds 45/46（recon 去留复裁）。
#[test]
#[ignore = "manual: v0.26 P0 seed 扩展 t1 +consistency"]
fn cartpole_seedext_t1_cons() -> Result<(), GraphError> {
    let mut c = Components::base();
    c.consistency = true;
    run_seed_ext("+consistency", c)
}

/// t3(promoted recipe) 补 seeds 45/46（recon 去留复裁对照）。
#[test]
#[ignore = "manual: v0.26 P0 seed 扩展 t3 promoted"]
fn cartpole_seedext_t3_promoted() -> Result<(), GraphError> {
    let cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .build()?;
    run_seed_ext("promoted(recipe)", cfg.components)
}

/// recon=16 候选默认补 seeds 45/46（promote 前的 5-seed 稳健性证据）。
#[test]
#[ignore = "manual: v0.26 P0 seed 扩展 recon=16 候选"]
fn cartpole_seedext_recon16() -> Result<(), GraphError> {
    let mut cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .build()?;
    cfg.components.reconstruction_coef = 16.0;
    run_seed_ext("promoted+recon16", cfg.components)
}
