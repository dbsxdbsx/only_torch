/*
 * @Author       : 老董
 * @Date         : 2026-02-14
 * @Description  : Tensor::topk 单元测试
 */

use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// 基本 topk：[2, 4] axis=1, k=2, sorted=true
#[test]
fn test_topk_basic() {
    let t = Tensor::new(&[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0], &[2, 4]);
    let (values, indices) = t.topk(2, 1, true);

    assert_eq!(values.shape(), &[2, 2]);
    assert_eq!(indices.shape(), &[2, 2]);

    // 第一行 [1, 4, 2, 3] → 前 2 大: 4.0 (idx=1), 3.0 (idx=3)
    assert_abs_diff_eq!(values[[0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(values[[0, 1]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(indices[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(indices[[0, 1]], 3.0, epsilon = 1e-6);

    // 第二行 [8, 5, 7, 6] → 前 2 大: 8.0 (idx=0), 7.0 (idx=2)
    assert_abs_diff_eq!(values[[1, 0]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(values[[1, 1]], 7.0, epsilon = 1e-6);
    assert_abs_diff_eq!(indices[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(indices[[1, 1]], 2.0, epsilon = 1e-6);
}

/// k=1 等价于 max
#[test]
fn test_topk_k1() {
    let t = Tensor::new(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]);
    let (values, indices) = t.topk(1, 1, true);

    assert_eq!(values.shape(), &[2, 1]);
    // 第一行 max = 4.0 at idx 2
    assert_abs_diff_eq!(values[[0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(indices[[0, 0]], 2.0, epsilon = 1e-6);
    // 第二行 max = 9.0 at idx 2
    assert_abs_diff_eq!(values[[1, 0]], 9.0, epsilon = 1e-6);
    assert_abs_diff_eq!(indices[[1, 0]], 2.0, epsilon = 1e-6);
}

/// k = axis_len（全选）
#[test]
fn test_topk_k_full() {
    let t = Tensor::new(&[3.0, 1.0, 2.0], &[1, 3]);
    let (values, _indices) = t.topk(3, 1, true);

    assert_eq!(values.shape(), &[1, 3]);
    // sorted=true → 降序：3.0, 2.0, 1.0
    assert_abs_diff_eq!(values[[0, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(values[[0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(values[[0, 2]], 1.0, epsilon = 1e-6);
}

/// axis=0 topk
#[test]
fn test_topk_axis0() {
    let t = Tensor::new(&[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0], &[4, 2]);
    let (values, indices) = t.topk(2, 0, true);

    assert_eq!(values.shape(), &[2, 2]);

    // 列 0: [1, 5, 3, 7] → 前 2 大: 7 (idx=3), 5 (idx=1)
    assert_abs_diff_eq!(values[[0, 0]], 7.0, epsilon = 1e-6);
    assert_abs_diff_eq!(values[[1, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(indices[[0, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(indices[[1, 0]], 1.0, epsilon = 1e-6);

    // 列 1: [2, 6, 4, 8] → 前 2 大: 8 (idx=3), 6 (idx=1)
    assert_abs_diff_eq!(values[[0, 1]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(values[[1, 1]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(indices[[0, 1]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(indices[[1, 1]], 1.0, epsilon = 1e-6);
}

/// sorted=false：前 k 大值都应出现，但顺序不保证
#[test]
fn test_topk_unsorted() {
    let t = Tensor::new(&[1.0, 5.0, 3.0, 4.0, 2.0], &[1, 5]);
    let (values, _indices) = t.topk(3, 1, false);

    assert_eq!(values.shape(), &[1, 3]);

    // 前 3 大应为 {5.0, 4.0, 3.0}，顺序不保证
    let mut vals: Vec<f32> = (0..3).map(|i| values[[0, i]]).collect();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_abs_diff_eq!(vals[0], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(vals[1], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(vals[2], 3.0, epsilon = 1e-6);
}

/// 无效参数应 panic
#[test]
#[should_panic(expected = "topk: axis")]
fn test_topk_invalid_axis() {
    let t = Tensor::new(&[1.0, 2.0], &[2]);
    t.topk(1, 1, true); // axis=1 但张量只有 1 维
}

#[test]
#[should_panic(expected = "topk: k=")]
fn test_topk_k_zero() {
    let t = Tensor::new(&[1.0, 2.0], &[1, 2]);
    t.topk(0, 1, true);
}

#[test]
#[should_panic(expected = "topk: k=")]
fn test_topk_k_exceeds() {
    let t = Tensor::new(&[1.0, 2.0], &[1, 2]);
    t.topk(3, 1, true);
}
