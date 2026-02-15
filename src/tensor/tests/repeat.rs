/*
 * @Author       : 老董
 * @Description  : Tensor::repeat() 单元测试
 *
 * Python 对照 (numpy):
 *   np.tile([[1,2],[3,4]], (2,3))
 *   → shape [4,6]
 */

use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// 测试 2D repeat
#[test]
fn test_tensor_repeat_2d() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = x.repeat(&[2, 3]);
    assert_eq!(y.shape(), &[4, 6]);
    // 第一行: [1,2,1,2,1,2]
    assert_abs_diff_eq!(y[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[0, 2]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[0, 5]], 2.0, epsilon = 1e-6);
    // 第三行 = 第一行（重复）
    assert_abs_diff_eq!(y[[2, 0]], 1.0, epsilon = 1e-6);
    // 第四行 = 第二行（重复）
    assert_abs_diff_eq!(y[[3, 1]], 4.0, epsilon = 1e-6);
}

/// 测试 repeat [1,1]（不变）
#[test]
fn test_tensor_repeat_identity() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = x.repeat(&[1, 1]);
    assert_eq!(y.shape(), &[2, 2]);
    assert_abs_diff_eq!(y[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[1, 1]], 4.0, epsilon = 1e-6);
}

/// 测试 1D repeat
#[test]
fn test_tensor_repeat_1d() {
    let x = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let y = x.repeat(&[4]);
    assert_eq!(y.shape(), &[12]);
    assert_abs_diff_eq!(y[[0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[3]], 1.0, epsilon = 1e-6); // 重复
    assert_abs_diff_eq!(y[[4]], 2.0, epsilon = 1e-6);
}

/// 测试 3D repeat
#[test]
fn test_tensor_repeat_3d() {
    let x = Tensor::ones(&[2, 3, 4]);
    let y = x.repeat(&[1, 2, 3]);
    assert_eq!(y.shape(), &[2, 6, 12]);
}

/// 测试 repeats 长度不匹配应 panic
#[test]
#[should_panic(expected = "repeats 长度")]
fn test_tensor_repeat_wrong_repeats_len() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let _y = x.repeat(&[2]); // 只给了 1 维，但张量是 2 维
}
