/*
 * @Author       : 老董
 * @Description  : Tensor::pad() 单元测试
 *
 * Python 对照 (numpy):
 *   np.pad([[1,2,3],[4,5,6]], ((1,1),(2,2)), constant_values=0)
 *   → shape [4,7], 边缘填 0
 */

use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// 测试 2D pad: 每维前后各填充
#[test]
fn test_tensor_pad_2d() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let y = x.pad(&[(1, 1), (2, 2)], 0.0);
    assert_eq!(y.shape(), &[4, 7]);
    // 原始数据应在 [1,2]..[1,4] 和 [2,2]..[2,4]
    assert_abs_diff_eq!(y[[1, 2]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[1, 4]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[2, 2]], 4.0, epsilon = 1e-6);
    // 填充区域应为 0
    assert_abs_diff_eq!(y[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[3, 6]], 0.0, epsilon = 1e-6);
}

/// 测试非零填充值
#[test]
fn test_tensor_pad_nonzero_value() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = x.pad(&[(0, 1), (1, 0)], -1.0);
    assert_eq!(y.shape(), &[3, 3]);
    // 填充区域应为 -1
    assert_abs_diff_eq!(y[[0, 0]], -1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[2, 0]], -1.0, epsilon = 1e-6);
    // 原始数据
    assert_abs_diff_eq!(y[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[1, 2]], 4.0, epsilon = 1e-6);
}

/// 测试 1D pad
#[test]
fn test_tensor_pad_1d() {
    let x = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let y = x.pad(&[(2, 1)], 0.0);
    assert_eq!(y.shape(), &[6]);
    assert_abs_diff_eq!(y[[0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[2]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[4]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[5]], 0.0, epsilon = 1e-6);
}

/// 测试零 padding（不变）
#[test]
fn test_tensor_pad_no_padding() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = x.pad(&[(0, 0), (0, 0)], 0.0);
    assert_eq!(y.shape(), &[2, 2]);
    assert_abs_diff_eq!(y[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[1, 1]], 4.0, epsilon = 1e-6);
}

/// 测试 3D pad
#[test]
fn test_tensor_pad_3d() {
    let x = Tensor::ones(&[2, 3, 4]);
    let y = x.pad(&[(1, 0), (0, 1), (1, 1)], 0.0);
    assert_eq!(y.shape(), &[3, 4, 6]);
    // 第一行应全是 0（前面 pad 了 1 行）
    assert_abs_diff_eq!(y[[0, 0, 0]], 0.0, epsilon = 1e-6);
    // 原始数据从 [1,0,1] 开始
    assert_abs_diff_eq!(y[[1, 0, 1]], 1.0, epsilon = 1e-6);
}

/// 测试 paddings 长度不匹配应 panic
#[test]
#[should_panic(expected = "paddings 长度")]
fn test_tensor_pad_wrong_padding_len() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let _y = x.pad(&[(1, 1)], 0.0); // 只给了 1 维，但张量是 2 维
}
