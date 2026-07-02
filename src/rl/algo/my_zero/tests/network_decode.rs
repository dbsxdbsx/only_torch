//! 搜索期无 Tensor 解码路径（`softmax_row` / `decode_categorical_slice`）的等价性测试
//!
//! 这些函数替代了旧的 `Tensor::softmax(1)` 链（约 6 次小 Tensor 分配），
//! 是 MCTS 推理热路径优化的一部分——必须与 Tensor 链**逐 bit 一致**，
//! 否则搜索轨迹会漂移、破坏哨兵可复现性。

use super::super::network::{MyZeroModel, softmax_row};
use super::super::value_encoding::two_hot_to_scalar;
use crate::rl::algo::my_zero::network::SUPPORT;
use crate::tensor::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// softmax_row 必须与 `Tensor::softmax(1)`（[1, N] 输入）逐 bit 一致
///
/// 覆盖 MCTS 实际用到的尺寸：policy（2/9 动作）、value/reward support（41）、latent（64）。
#[test]
fn softmax_row_bitwise_matches_tensor_softmax() {
    let mut rng = StdRng::seed_from_u64(42);
    for &n in &[2usize, 3, 9, 41, 64, 225] {
        for case in 0..20 {
            // 混合量纲：常规 logits + 大偏移（考验数值稳定路径一致性）
            let offset = if case % 3 == 0 { 50.0 } else { 0.0 };
            let logits: Vec<f32> = (0..n)
                .map(|_| rng.gen_range(-8.0f32..8.0) + offset)
                .collect();

            let expected = Tensor::new(&logits, &[1, n]).softmax(1);
            let expected_slice = expected.data_as_slice();
            let actual = softmax_row(&logits);

            assert_eq!(actual.len(), n);
            for i in 0..n {
                assert_eq!(
                    actual[i].to_bits(),
                    expected_slice[i].to_bits(),
                    "softmax_row 与 Tensor::softmax 不逐 bit 一致：n={n} case={case} i={i} \
                     actual={} expected={}",
                    actual[i],
                    expected_slice[i]
                );
            }
        }
    }
}

/// decode_categorical_slice 必须与旧链（Tensor softmax → data_as_slice → two_hot_to_scalar）逐 bit 一致
#[test]
fn decode_categorical_slice_bitwise_matches_tensor_chain() {
    let mut rng = StdRng::seed_from_u64(7);
    let n = SUPPORT.size();
    for _ in 0..50 {
        let logits: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0f32..10.0)).collect();

        // 旧路径：建 Tensor → softmax(1) → data_as_slice → two_hot_to_scalar
        let probs = Tensor::new(&logits, &[1, n]).softmax(1);
        let expected = two_hot_to_scalar(probs.data_as_slice(), &SUPPORT);

        // 新路径：切片直达
        let actual = MyZeroModel::decode_categorical_slice(&logits);

        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "decode_categorical_slice 与旧 Tensor 链不逐 bit 一致：actual={actual} expected={expected}"
        );
    }
}
