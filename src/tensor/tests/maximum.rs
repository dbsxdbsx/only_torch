//! maximum 相关测试

use crate::tensor::Tensor;

#[test]
fn test_maximum_same_shape_1d() {
    let a = Tensor::new(&[1.0, 4.0, 3.0], &[3]);
    let b = Tensor::new(&[2.0, 2.0, 5.0], &[3]);
    let result = a.maximum(&b);

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result[[0]], 2.0); // max(1, 2) = 2
    assert_eq!(result[[1]], 4.0); // max(4, 2) = 4
    assert_eq!(result[[2]], 5.0); // max(3, 5) = 5
}

#[test]
fn test_maximum_same_shape_2d() {
    // [[1, 4],
    //  [3, 2]]
    let a = Tensor::new(&[1.0, 4.0, 3.0, 2.0], &[2, 2]);
    // [[2, 3],
    //  [1, 5]]
    let b = Tensor::new(&[2.0, 3.0, 1.0, 5.0], &[2, 2]);
    let result = a.maximum(&b);

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result[[0, 0]], 2.0); // max(1, 2) = 2
    assert_eq!(result[[0, 1]], 4.0); // max(4, 3) = 4
    assert_eq!(result[[1, 0]], 3.0); // max(3, 1) = 3
    assert_eq!(result[[1, 1]], 5.0); // max(2, 5) = 5
}

#[test]
fn test_maximum_broadcast_scalar() {
    // 向量与标量（形状 [1]）比较
    let a = Tensor::new(&[1.0, 4.0, 3.0, 5.0], &[4]);
    let b = Tensor::new(&[2.5], &[1]);
    let result = a.maximum(&b);

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result[[0]], 2.5); // max(1.0, 2.5) = 2.5
    assert_eq!(result[[1]], 4.0); // max(4.0, 2.5) = 4.0
    assert_eq!(result[[2]], 3.0); // max(3.0, 2.5) = 3.0
    assert_eq!(result[[3]], 5.0); // max(5.0, 2.5) = 5.0
}

#[test]
fn test_maximum_broadcast_row_col() {
    // [2, 3] 与 [3] 广播
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[2.5, 2.5, 2.5], &[3]);
    let result = a.maximum(&b);

    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result[[0, 0]], 2.5); // max(1, 2.5) = 2.5
    assert_eq!(result[[0, 1]], 2.5); // max(2, 2.5) = 2.5
    assert_eq!(result[[0, 2]], 3.0); // max(3, 2.5) = 3
    assert_eq!(result[[1, 0]], 4.0); // max(4, 2.5) = 4
    assert_eq!(result[[1, 1]], 5.0); // max(5, 2.5) = 5
    assert_eq!(result[[1, 2]], 6.0); // max(6, 2.5) = 6
}

#[test]
fn test_maximum_broadcast_different_dims() {
    // [2, 1] 与 [1, 3] 广播为 [2, 3]
    let a = Tensor::new(&[1.0, 4.0], &[2, 1]);
    let b = Tensor::new(&[2.0, 3.0, 5.0], &[1, 3]);
    let result = a.maximum(&b);

    assert_eq!(result.shape(), &[2, 3]);
    // 第一行：max(1, [2, 3, 5]) = [2, 3, 5]
    assert_eq!(result[[0, 0]], 2.0);
    assert_eq!(result[[0, 1]], 3.0);
    assert_eq!(result[[0, 2]], 5.0);
    // 第二行：max(4, [2, 3, 5]) = [4, 4, 5]
    assert_eq!(result[[1, 0]], 4.0);
    assert_eq!(result[[1, 1]], 4.0);
    assert_eq!(result[[1, 2]], 5.0);
}

#[test]
fn test_maximum_with_negative_values() {
    let a = Tensor::new(&[-1.0, 2.0, -3.0], &[3]);
    let b = Tensor::new(&[0.0, -1.0, -2.0], &[3]);
    let result = a.maximum(&b);

    assert_eq!(result[[0]], 0.0); // max(-1, 0) = 0
    assert_eq!(result[[1]], 2.0); // max(2, -1) = 2
    assert_eq!(result[[2]], -2.0); // max(-3, -2) = -2
}

#[test]
fn test_maximum_with_equal_values() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let result = a.maximum(&b);

    assert_eq!(result[[0]], 1.0);
    assert_eq!(result[[1]], 2.0);
    assert_eq!(result[[2]], 3.0);
}

#[test]
fn test_maximum_3d_tensors() {
    // [2, 2, 2] 形状
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::new(&[4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0], &[2, 2, 2]);
    let result = a.maximum(&b);

    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(result[[0, 0, 0]], 4.0); // max(1, 4)
    assert_eq!(result[[0, 0, 1]], 3.0); // max(2, 3)
    assert_eq!(result[[0, 1, 0]], 3.0); // max(3, 2)
    assert_eq!(result[[0, 1, 1]], 4.0); // max(4, 1)
    assert_eq!(result[[1, 0, 0]], 8.0); // max(5, 8)
    assert_eq!(result[[1, 0, 1]], 7.0); // max(6, 7)
    assert_eq!(result[[1, 1, 0]], 7.0); // max(7, 6)
    assert_eq!(result[[1, 1, 1]], 8.0); // max(8, 5)
}

#[test]
fn test_maximum_clamp_use_case() {
    // 模拟 clamp lower bound 场景：max(x, 0) 类似 ReLU
    let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let zero = Tensor::new(&[0.0], &[1]);
    let result = x.maximum(&zero);

    assert_eq!(result.shape(), &[5]);
    assert_eq!(result[[0]], 0.0); // max(-2, 0) = 0
    assert_eq!(result[[1]], 0.0); // max(-1, 0) = 0
    assert_eq!(result[[2]], 0.0); // max(0, 0) = 0
    assert_eq!(result[[3]], 1.0); // max(1, 0) = 1
    assert_eq!(result[[4]], 2.0); // max(2, 0) = 2
}

#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_maximum_incompatible_shapes() {
    // [3] 和 [4] 无法广播
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let _ = a.maximum(&b);
}
