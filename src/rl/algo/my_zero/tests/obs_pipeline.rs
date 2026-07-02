//! 图像 obs 管线单测（v0.26 Phase 1）：灰度 / 双线性缩放 / 帧堆叠 / 训练期组装一致性。

use crate::rl::algo::my_zero::obs_pipeline::{
    OUT_SIDE, STACK, assemble_stacked_obs, bilinear_resize, frame_len, stacked_obs_dim,
};
use approx::assert_abs_diff_eq;

// ---- bilinear_resize ----

/// 恒值图像缩放后仍恒值（插值不引入偏移）
#[test]
fn bilinear_constant_image_stays_constant() {
    let src = vec![127.0f32; 210 * 160];
    let out = bilinear_resize(&src, 210, 160, 84, 84);
    assert_eq!(out.len(), 84 * 84);
    for &v in &out {
        assert_abs_diff_eq!(v, 127.0, epsilon = 1e-4);
    }
}

/// 同尺寸缩放 = 恒等（像素中心对齐口径的基线性质）
#[test]
fn bilinear_identity_when_same_size() {
    let src: Vec<f32> = (0..12).map(|v| v as f32).collect(); // 3×4
    let out = bilinear_resize(&src, 3, 4, 3, 4);
    for (a, b) in src.iter().zip(&out) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-5);
    }
}

/// 2×2 → 1×1：四像素均值（像素中心映射到正中）
#[test]
fn bilinear_2x2_to_1x1_is_mean() {
    let src = vec![0.0, 10.0, 20.0, 30.0];
    let out = bilinear_resize(&src, 2, 2, 1, 1);
    assert_abs_diff_eq!(out[0], 15.0, epsilon = 1e-5);
}

/// 水平线性渐变缩小后仍保持单调（无振铃/越界）
#[test]
fn bilinear_gradient_stays_monotonic() {
    let (h, w) = (8, 64);
    let src: Vec<f32> = (0..h * w).map(|i| (i % w) as f32).collect();
    let out = bilinear_resize(&src, h, w, 4, 16);
    for row in 0..4 {
        for col in 1..16 {
            assert!(
                out[row * 16 + col] > out[row * 16 + col - 1],
                "第 {row} 行第 {col} 列不单调"
            );
        }
    }
    // 值域不越界
    for &v in &out {
        assert!((0.0..=(w as f32 - 1.0)).contains(&v));
    }
}

// ---- assemble_stacked_obs（训练期组装 = acting 期滑窗语义）----

/// episode 起点前向填充首帧；中段取连续 4 帧（老 → 新）
#[test]
fn assemble_stack_padding_and_order() {
    let f: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32; 2]).collect();
    let frames: Vec<&[f32]> = f.iter().map(|v| v.as_slice()).collect();

    // t=0：全部前向填充首帧 → [f0,f0,f0,f0]
    assert_eq!(
        assemble_stacked_obs(&frames, 0, 4),
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    // t=1：[f0,f0,f0,f1]
    assert_eq!(
        assemble_stacked_obs(&frames, 1, 4),
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    );
    // t=5（中段）：[f2,f3,f4,f5]
    assert_eq!(
        assemble_stacked_obs(&frames, 5, 4),
        vec![2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
    );
}

// ---- ImagePipe 常量契约 ----

#[test]
fn pipeline_dims_consistent() {
    assert_eq!(frame_len(), OUT_SIDE * OUT_SIDE);
    assert_eq!(stacked_obs_dim(), STACK * frame_len());
    assert_eq!(OUT_SIDE, 84);
    assert_eq!(STACK, 4);
}
