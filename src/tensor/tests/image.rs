//! **回归测试**：`Tensor::is_image` / `to_image` 在非连续输入上的健壮性。
//!
//! `is_image` 此前用 `self.data.as_slice().unwrap()` 遍历像素做范围校验，在
//! `permute`/`transpose` 等非连续视图上会 **panic**。改用 `iter()`（布局无关）后，
//! 任意布局都应正确工作。`to_image` 走 `self[[y,x,c]]` 逻辑索引，本就布局安全，
//! 此处一并验证非连续与连续等价张量产出一致图像。

use crate::tensor::Tensor;
use image::ColorType;

/// `[H=2, W=2, C=3]` 的**非连续** RGB 图像视图（由 CHW permute 得到）。
/// 像素值 0,20,…,220 均落在 `[0,255]` 且为整数，满足 `is_image` 校验。
fn noncontiguous_hwc_image() -> Tensor {
    let base = Tensor::new(
        &(0..12).map(|v| (v * 20) as f32).collect::<Vec<_>>(),
        &[3, 2, 2],
    );
    let nc = base.permute(&[1, 2, 0]); // [H=2, W=2, C=3]，非连续
    assert_eq!(nc.shape(), &[2, 2, 3]);
    assert!(!nc.is_contiguous(), "构造用例应产出非连续张量");
    nc
}

#[test]
fn test_is_image_noncontiguous_no_panic() {
    let nc = noncontiguous_hwc_image();
    // 此前 as_slice().unwrap() 在非连续上会 panic
    assert_eq!(nc.is_image().unwrap(), ColorType::Rgb8);
}

#[test]
fn test_to_image_noncontiguous_matches_contiguous() {
    let nc = noncontiguous_hwc_image();
    let contig = Tensor::new(&nc.to_vec(), nc.shape());
    let img_nc = nc.to_image().unwrap();
    let img_contig = contig.to_image().unwrap();
    assert_eq!(img_nc.to_rgb8(), img_contig.to_rgb8());
}
