/*
 * @Author       : 老董
 * @Description  : Tensor::powf() 单元测试
 *
 * Python 对照 (numpy):
 *   np.array([1,2,3,4])**2 = [1, 4, 9, 16]
 *   np.array([1,2,3,4])**0.5 = [1, 1.4142, 1.7321, 2]
 *   np.array([1,2,3,4])**(-1) = [1, 0.5, 0.3333, 0.25]
 */

use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// 测试 powf(2.0) = 平方
#[test]
fn test_tensor_powf_square() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = x.powf(2.0);
    assert_eq!(y.shape(), &[2, 2]);
    assert_abs_diff_eq!(y[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[0, 1]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[1, 0]], 9.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[1, 1]], 16.0, epsilon = 1e-6);
}

/// 测试 powf(0.5) = 开方
#[test]
fn test_tensor_powf_sqrt() {
    let x = Tensor::new(&[1.0, 4.0, 9.0, 16.0], &[4]);
    let y = x.powf(0.5);
    assert_eq!(y.shape(), &[4]);
    assert_abs_diff_eq!(y[[0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[2]], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y[[3]], 4.0, epsilon = 1e-5);
}

/// 测试 powf(-1.0) = 倒数
#[test]
fn test_tensor_powf_inverse() {
    let x = Tensor::new(&[1.0, 2.0, 4.0, 5.0], &[2, 2]);
    let y = x.powf(-1.0);
    assert_abs_diff_eq!(y[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[0, 1]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[1, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[1, 1]], 0.2, epsilon = 1e-6);
}

/// 测试 powf(0.0) = 全 1
#[test]
fn test_tensor_powf_zero_exponent() {
    let x = Tensor::new(&[2.0, 3.0, 5.0, 7.0], &[4]);
    let y = x.powf(0.0);
    for i in 0..4 {
        assert_abs_diff_eq!(y[[i]], 1.0, epsilon = 1e-6);
    }
}

/// 测试 powf(1.0) = 恒等
#[test]
fn test_tensor_powf_identity() {
    let x = Tensor::new(&[2.5, 3.7, 1.1, 0.9], &[4]);
    let y = x.powf(1.0);
    let flat = x.flatten_view();
    for i in 0..4 {
        assert_abs_diff_eq!(y[[i]], flat[i], epsilon = 1e-6);
    }
}

/// 测试 3D 张量
#[test]
fn test_tensor_powf_3d() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let y = x.powf(3.0);
    assert_eq!(y.shape(), &[2, 2, 2]);
    assert_abs_diff_eq!(y[[0, 0, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y[[1, 1, 1]], 512.0, epsilon = 1e-3);
}
