//! `value_encoding.rs` 编码层测试：two-hot 基线性质 + HL-Gauss 软标签性质与两者关系。

use super::super::value_encoding::{
    HL_GAUSS_SIGMA, SupportConfig, scalar_to_hl_gauss, scalar_to_two_hot, two_hot_to_scalar,
};
use super::super::value_transform::value_transform;

const CFG: SupportConfig = SupportConfig::new(20);

#[test]
fn two_hot_sums_to_one_and_roundtrips() {
    for &x in &[0.0f32, 1.0, -1.0, 10.0, -50.0, 200.0] {
        let p = scalar_to_two_hot(x, &CFG);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "x={x} two-hot 概率和 {sum} ≠ 1");
        let back = two_hot_to_scalar(&p, &CFG);
        assert!(
            (back - x).abs() < 0.15 * x.abs().max(1.0),
            "x={x} two-hot round-trip 得 {back}"
        );
    }
}

#[test]
fn hl_gauss_sums_to_one() {
    for &x in &[0.0f32, 1.0, -1.0, 10.0, -50.0, 200.0, 420.0, -420.0] {
        let p = scalar_to_hl_gauss(x, &CFG);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "x={x} HL-Gauss 概率和 {sum} ≠ 1");
        assert!(p.iter().all(|&v| v >= 0.0), "x={x} 出现负概率");
    }
}

#[test]
fn hl_gauss_expectation_matches_transform_domain_interior() {
    // 内点（远离 support 边界）：变换域期望应 ≈ h(x)（截断可忽略，离散化偏差 < 1e-3）
    for &x in &[0.0f32, 1.0, -3.0, 25.0, -25.0, 120.0] {
        let y = value_transform(x);
        let p = scalar_to_hl_gauss(x, &CFG);
        let mean: f32 = p.iter().enumerate().map(|(i, &pi)| pi * CFG.atom(i)).sum();
        assert!(
            (mean - y).abs() < 1e-2,
            "x={x}（h(x)={y:.4}）HL-Gauss 变换域期望 {mean:.4} 偏差过大"
        );
    }
}

#[test]
fn hl_gauss_roundtrips_through_decoder() {
    // 解码端与 two-hot 共用（期望 → h⁻¹）：编码后解码应还原原标量
    for &x in &[0.0f32, 1.0, -1.0, 10.0, -50.0, 200.0] {
        let p = scalar_to_hl_gauss(x, &CFG);
        let back = two_hot_to_scalar(&p, &CFG);
        assert!(
            (back - x).abs() < 0.15 * x.abs().max(1.0),
            "x={x} HL-Gauss round-trip 得 {back}"
        );
    }
}

#[test]
fn hl_gauss_mass_spreads_over_neighborhood() {
    // 与 two-hot 的本质差异：质量摊到 ~±3σ 邻域（σ=0.75 → 至少 4 个原子非平凡），
    // 峰值原子占比显著小于 1。
    let p = scalar_to_hl_gauss(0.0, &CFG);
    let nontrivial = p.iter().filter(|&&v| v > 1e-3).count();
    assert!(
        nontrivial >= 4,
        "σ={HL_GAUSS_SIGMA} 下非平凡原子仅 {nontrivial} 个，应 ≥ 4"
    );
    let peak = p.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        peak < 0.75,
        "峰值原子占比 {peak}，高斯平滑后应显著 < 1（two-hot 整点处为 1.0）"
    );

    let two = scalar_to_two_hot(0.0, &CFG);
    let two_nontrivial = two.iter().filter(|&&v| v > 1e-3).count();
    assert!(two_nontrivial <= 2, "two-hot 非零原子应 ≤ 2");
}

#[test]
fn hl_gauss_clamps_out_of_support_values() {
    // 超出 support 的值：clamp 后质量堆在边界 bin 附近，和仍为 1，期望接近边界原子
    let p = scalar_to_hl_gauss(1e6, &CFG);
    let sum: f32 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    let mean: f32 = p.iter().enumerate().map(|(i, &pi)| pi * CFG.atom(i)).sum();
    assert!(
        mean > CFG.half_size() as f32 - 1.0,
        "clamp 后期望 {mean} 应贴近上边界 {}",
        CFG.half_size()
    );
}

#[test]
fn hl_gauss_symmetric_for_negated_input() {
    let p_pos = scalar_to_hl_gauss(7.0, &CFG);
    let p_neg = scalar_to_hl_gauss(-7.0, &CFG);
    let n = p_pos.len();
    for i in 0..n {
        assert!(
            (p_pos[i] - p_neg[n - 1 - i]).abs() < 1e-5,
            "对称性破坏：p_pos[{i}]={} vs p_neg[{}]={}",
            p_pos[i],
            n - 1 - i,
            p_neg[n - 1 - i]
        );
    }
}
