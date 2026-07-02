//! `sve.rs` blend 测试：端点/中点与权重 clamp

use super::super::sve::sve_blend;

#[test]
fn endpoints_and_midpoint() {
    assert!(
        (sve_blend(1.0, 5.0, 0.0) - 1.0).abs() < 1e-6,
        "w=0 → n-step"
    );
    assert!(
        (sve_blend(1.0, 5.0, 1.0) - 5.0).abs() < 1e-6,
        "w=1 → 搜索 value"
    );
    assert!(
        (sve_blend(2.0, 4.0, 0.5) - 3.0).abs() < 1e-6,
        "w=0.5 → 中点"
    );
}

#[test]
fn weight_is_clamped() {
    assert!(
        (sve_blend(1.0, 9.0, 2.0) - 9.0).abs() < 1e-6,
        "w>1 clamp 到 1"
    );
    assert!(
        (sve_blend(1.0, 9.0, -1.0) - 1.0).abs() < 1e-6,
        "w<0 clamp 到 0"
    );
}
