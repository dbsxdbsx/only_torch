//! 图像 obs 管线单测（v0.26 Phase 1）：灰度 / 双线性缩放 / 帧堆叠 / 训练期组装一致性。

use crate::rl::StoredObs;
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
    let f: Vec<StoredObs> = (0..6).map(|i| vec![i as f32; 2].into()).collect();
    let frames: Vec<&StoredObs> = f.iter().collect();

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

/// 融合组装（`ObsSource::Stacked` 直写 flat buffer）与参照实现逐 bit 一致
///
/// 生产路径已改为 batch 组装时就地反量化堆叠（零中间 Vec），本测试守住
/// 它与 [`assemble_stacked_obs`]（语义参照）的逐 bit 等价，含起点前向填充。
#[test]
fn obs_source_stacked_matches_reference_bit_exact() {
    use crate::rl::SelfPlayStep;
    use crate::rl::algo::my_zero::network::ObsSource;

    let steps: Vec<SelfPlayStep> = (0..6)
        .map(|i| SelfPlayStep {
            obs: StoredObs::quantize_pixels(&[i as f32 * 40.0, i as f32 * 40.0 + 0.7]),
            action: vec![0.0],
            policy_target: vec![1.0],
            player: 0,
            reward: 0.0,
            root_value: None,
            terminated: false,
            truncated: false,
            continuation: 1.0,
            extras: Default::default(),
        })
        .collect();
    let frames: Vec<&StoredObs> = steps.iter().map(|s| &s.obs).collect();

    for t in [0usize, 1, 3, 5] {
        let reference = assemble_stacked_obs(&frames, t, 4);
        let src = ObsSource::Stacked {
            steps: &steps,
            t,
            stack: 4,
        };
        assert_eq!(src.dim(), reference.len(), "t={t} dim 应一致");
        let mut fused = Vec::new();
        src.append_into(&mut fused);
        assert_eq!(fused, reference, "t={t} 融合组装应与参照逐 bit 一致");
    }

    // Single（向量 obs 直通）同样逐 bit
    let v: StoredObs = vec![0.5f32, -1.25, 3.0].into();
    let single = ObsSource::Single(&v);
    let mut out = Vec::new();
    single.append_into(&mut out);
    assert_eq!(out, v.to_f32_vec());
}

// ---- ImagePipe 像素域校验（0-255 前提守卫）----

/// 正常像素域（0–255）通过校验
#[test]
fn pixel_domain_accepts_raw_pixels() {
    use crate::rl::algo::my_zero::obs_pipeline::ImagePipe;
    ImagePipe::validate_pixel_domain(&[0.0, 12.0, 236.0, 255.0]);
}

/// 已归一化 [0,1] 的图像 obs 应被拦截（量化会静默压成 0/1）
#[test]
#[should_panic(expected = "疑似已归一化")]
fn pixel_domain_rejects_normalized_obs() {
    use crate::rl::algo::my_zero::obs_pipeline::ImagePipe;
    ImagePipe::validate_pixel_domain(&[0.0, 0.5, 1.0]);
}

/// 值域越界（负值 / >255）应被拦截
#[test]
#[should_panic(expected = "超出 0–255 像素域")]
fn pixel_domain_rejects_out_of_range() {
    use crate::rl::algo::my_zero::obs_pipeline::ImagePipe;
    ImagePipe::validate_pixel_domain(&[-1.0, 100.0]);
}

// ---- ImagePipe 常量契约 ----

#[test]
fn pipeline_dims_consistent() {
    assert_eq!(frame_len(), OUT_SIDE * OUT_SIDE);
    assert_eq!(stacked_obs_dim(), STACK * frame_len());
    assert_eq!(OUT_SIDE, 84);
    assert_eq!(STACK, 4);
}

// ---- StoredObs 量化语义（u8 帧存储）----

/// 量化往返误差 ≤ 半个量化步长（0.5/255），且整数像素无损
#[test]
fn stored_obs_quantize_roundtrip_error_bound() {
    // 覆盖整数点 + 小数点 + 越界值
    let pixels: Vec<f32> = vec![
        0.0, 1.0, 127.0, 254.0, 255.0, 0.4, 127.5, 254.6, -3.0, 260.0,
    ];
    let q = StoredObs::quantize_pixels(&pixels);
    let deq = q.to_f32_vec();
    assert_eq!(deq.len(), pixels.len());
    for (&orig, &got) in pixels.iter().zip(&deq) {
        let expect_norm = (orig / 255.0).clamp(0.0, 1.0);
        assert!(
            (got - expect_norm).abs() <= 0.5 / 255.0 + 1e-7,
            "像素 {orig} 反量化 {got} 偏离归一期望 {expect_norm} 超过半步长"
        );
    }
    // 整数像素严格无损（round 恒等）
    let q2 = StoredObs::quantize_pixels(&[0.0, 42.0, 255.0]);
    assert_eq!(q2.to_f32_vec(), vec![0.0, 42.0 / 255.0, 1.0]);
}

/// F32 直通变体与旧 `Vec<f32>` 语义逐 bit 一致（From / as_f32 / to_f32_vec）
#[test]
fn stored_obs_f32_passthrough_bit_exact() {
    let v = vec![1.5f32, -2.25, 0.1, f32::MIN_POSITIVE];
    let s: StoredObs = v.clone().into();
    assert_eq!(s.as_f32(), v.as_slice());
    assert_eq!(s.to_f32_vec(), v);
    assert_eq!(s.len(), 4);
}

/// 量化帧堆叠组装 = 逐帧反量化后拼接（训练期与 acting 期共用同一反量化口径）
#[test]
fn assemble_quantized_frames_matches_manual_dequant() {
    let f: Vec<StoredObs> = (0..5)
        .map(|i| StoredObs::quantize_pixels(&[i as f32 * 50.0, i as f32 * 50.0 + 0.6]))
        .collect();
    let frames: Vec<&StoredObs> = f.iter().collect();

    let assembled = assemble_stacked_obs(&frames, 4, 4);
    let mut manual = Vec::new();
    for idx in 1..=4 {
        manual.extend(f[idx].to_f32_vec());
    }
    assert_eq!(assembled, manual);
    // 起点前向填充语义在量化帧下同样成立
    let padded = assemble_stacked_obs(&frames, 0, 4);
    let f0 = f[0].to_f32_vec();
    assert_eq!(padded, [&f0[..], &f0[..], &f0[..], &f0[..]].concat());
}
