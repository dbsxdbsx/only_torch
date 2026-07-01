//! **回归测试**：非连续张量输入喂给 `src/data` 变换。
//!
//! 变换内部按行主序读取（`flatten_view` / 手写 `row*w+col` 偏移）。在
//! `permute`/`transpose`/`narrow` 等非连续视图上，`flatten_view`（内部 `into_shape`）
//! 会 **panic**，手写偏移则可能按物理序 **静默算错**。统一加 `contiguous()` 守卫
//! 或改 `to_vec()` 后应能正确处理任意布局。
//!
//! 每个用例都与「把非连续视图显式物化为连续张量」的等价计算逐元素对比，
//! 既抓 panic 也抓静默错序。

use crate::data::transforms::{GaussianNoise, Normalize, RandomHorizontalFlip, Transform};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// 构造逻辑形状 `[C=3, H=2, W=2]` 的**非连续**张量：由 `[H,W,C]` permute 得到。
fn noncontiguous_chw() -> Tensor {
    let base = Tensor::new(&(0..12).map(|v| v as f32).collect::<Vec<_>>(), &[2, 2, 3]);
    let nc = base.permute(&[2, 0, 1]); // [C=3, H=2, W=2]，非连续
    assert_eq!(nc.shape(), &[3, 2, 2]);
    assert!(!nc.is_contiguous(), "构造用例应产出非连续张量");
    nc
}

/// 把非连续视图显式物化为同逻辑值的连续张量，作为对照基准。
fn materialized(nc: &Tensor) -> Tensor {
    Tensor::new(&nc.to_vec(), nc.shape())
}

#[test]
fn test_normalize_noncontiguous_input() {
    let nc = noncontiguous_chw();
    let contig = materialized(&nc);

    let norm = Normalize::new(vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 8.0]);
    let out_nc = norm.apply(&nc); // 此前 flatten_view 会 panic
    let out_contig = norm.apply(&contig);

    assert_eq!(out_nc.shape(), &[3, 2, 2]);
    for (a, b) in out_nc.to_vec().iter().zip(out_contig.to_vec().iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
    }
}

#[test]
fn test_horizontal_flip_noncontiguous_input() {
    // p=1.0 → 确定性翻转，走 flip_horizontal 的手写 row*w+col 重排（最易静默错序处）。
    let nc = noncontiguous_chw();
    let contig = materialized(&nc);

    let flip = RandomHorizontalFlip::new(1.0);
    let out_nc = flip.apply(&nc);
    let out_contig = flip.apply(&contig);

    assert_eq!(out_nc.shape(), &[3, 2, 2]);
    for (a, b) in out_nc.to_vec().iter().zip(out_contig.to_vec().iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
    }
}

#[test]
fn test_gaussian_noise_noncontiguous_input_no_panic() {
    // 高斯噪声随机，无法逐元素对比；仅验证非连续输入不 panic、形状正确。
    let nc = noncontiguous_chw();
    let noise = GaussianNoise::new(0.0, 0.1);
    let out = noise.apply(&nc);
    assert_eq!(out.shape(), &[3, 2, 2]);
    assert_eq!(out.size(), 12);
}
