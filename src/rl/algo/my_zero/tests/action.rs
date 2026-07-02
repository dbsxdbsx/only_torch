//! `action.rs` 连续离散化测试：bin 中点语义（Sampled MuZero Appendix 对齐）

use super::super::action::idx_to_continuous;

#[test]
fn discretize_uses_bin_centers() {
    // [0, 10] × 10 档 → 中点 0.5, 1.5, …, 9.5
    assert!((idx_to_continuous(0, 0.0, 10.0, 10) - 0.5).abs() < 1e-5);
    assert!((idx_to_continuous(9, 0.0, 10.0, 10) - 9.5).abs() < 1e-5);
    assert!((idx_to_continuous(4, 0.0, 10.0, 10) - 4.5).abs() < 1e-5);
}

#[test]
fn discretize_single_bucket_is_midpoint() {
    assert!((idx_to_continuous(0, -2.0, 2.0, 1) - 0.0).abs() < 1e-5);
}
