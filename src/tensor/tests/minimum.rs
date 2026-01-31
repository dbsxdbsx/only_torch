//! minimum 相关测试

use crate::tensor::Tensor;

#[test]
fn test_minimum_same_shape_1d() {
    let a = Tensor::new(&[1.0, 4.0, 3.0], &[3]);
    let b = Tensor::new(&[2.0, 2.0, 5.0], &[3]);
    let result = a.minimum(&b);

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result[[0]], 1.0); // min(1, 2) = 1
    assert_eq!(result[[1]], 2.0); // min(4, 2) = 2
    assert_eq!(result[[2]], 3.0); // min(3, 5) = 3
}

#[test]
fn test_minimum_same_shape_2d() {
    // [[1, 4],
    //  [3, 2]]
    let a = Tensor::new(&[1.0, 4.0, 3.0, 2.0], &[2, 2]);
    // [[2, 3],
    //  [1, 5]]
    let b = Tensor::new(&[2.0, 3.0, 1.0, 5.0], &[2, 2]);
    let result = a.minimum(&b);

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result[[0, 0]], 1.0); // min(1, 2) = 1
    assert_eq!(result[[0, 1]], 3.0); // min(4, 3) = 3
    assert_eq!(result[[1, 0]], 1.0); // min(3, 1) = 1
    assert_eq!(result[[1, 1]], 2.0); // min(2, 5) = 2
}

#[test]
fn test_minimum_broadcast_scalar() {
    // 向量与标量（形状 [1]）比较
    let a = Tensor::new(&[1.0, 4.0, 3.0, 5.0], &[4]);
    let b = Tensor::new(&[2.5], &[1]);
    let result = a.minimum(&b);

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result[[0]], 1.0); // min(1.0, 2.5) = 1.0
    assert_eq!(result[[1]], 2.5); // min(4.0, 2.5) = 2.5
    assert_eq!(result[[2]], 2.5); // min(3.0, 2.5) = 2.5
    assert_eq!(result[[3]], 2.5); // min(5.0, 2.5) = 2.5
}

#[test]
fn test_minimum_broadcast_row_col() {
    // [2, 3] 与 [3] 广播
    // [[1, 2, 3],    [[10, 20, 30]]
    //  [4, 5, 6]] min
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[2.5, 2.5, 2.5], &[3]);
    let result = a.minimum(&b);

    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result[[0, 0]], 1.0); // min(1, 2.5) = 1
    assert_eq!(result[[0, 1]], 2.0); // min(2, 2.5) = 2
    assert_eq!(result[[0, 2]], 2.5); // min(3, 2.5) = 2.5
    assert_eq!(result[[1, 0]], 2.5); // min(4, 2.5) = 2.5
    assert_eq!(result[[1, 1]], 2.5); // min(5, 2.5) = 2.5
    assert_eq!(result[[1, 2]], 2.5); // min(6, 2.5) = 2.5
}

#[test]
fn test_minimum_broadcast_different_dims() {
    // [2, 1] 与 [1, 3] 广播为 [2, 3]
    let a = Tensor::new(&[1.0, 4.0], &[2, 1]);
    let b = Tensor::new(&[2.0, 3.0, 5.0], &[1, 3]);
    let result = a.minimum(&b);

    assert_eq!(result.shape(), &[2, 3]);
    // 第一行：min(1, [2, 3, 5]) = [1, 1, 1]
    assert_eq!(result[[0, 0]], 1.0);
    assert_eq!(result[[0, 1]], 1.0);
    assert_eq!(result[[0, 2]], 1.0);
    // 第二行：min(4, [2, 3, 5]) = [2, 3, 4]
    assert_eq!(result[[1, 0]], 2.0);
    assert_eq!(result[[1, 1]], 3.0);
    assert_eq!(result[[1, 2]], 4.0);
}

#[test]
fn test_minimum_with_negative_values() {
    let a = Tensor::new(&[-1.0, 2.0, -3.0], &[3]);
    let b = Tensor::new(&[0.0, -1.0, -2.0], &[3]);
    let result = a.minimum(&b);

    assert_eq!(result[[0]], -1.0); // min(-1, 0) = -1
    assert_eq!(result[[1]], -1.0); // min(2, -1) = -1
    assert_eq!(result[[2]], -3.0); // min(-3, -2) = -3
}

#[test]
fn test_minimum_with_equal_values() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let result = a.minimum(&b);

    assert_eq!(result[[0]], 1.0);
    assert_eq!(result[[1]], 2.0);
    assert_eq!(result[[2]], 3.0);
}

#[test]
fn test_minimum_3d_tensors() {
    // [2, 2, 2] 形状
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::new(&[4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0], &[2, 2, 2]);
    let result = a.minimum(&b);

    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(result[[0, 0, 0]], 1.0); // min(1, 4)
    assert_eq!(result[[0, 0, 1]], 2.0); // min(2, 3)
    assert_eq!(result[[0, 1, 0]], 2.0); // min(3, 2)
    assert_eq!(result[[0, 1, 1]], 1.0); // min(4, 1)
    assert_eq!(result[[1, 0, 0]], 5.0); // min(5, 8)
    assert_eq!(result[[1, 0, 1]], 6.0); // min(6, 7)
    assert_eq!(result[[1, 1, 0]], 6.0); // min(7, 6)
    assert_eq!(result[[1, 1, 1]], 5.0); // min(8, 5)
}

#[test]
fn test_minimum_rl_use_case() {
    // 模拟 SAC/TD3 中的 min(Q1, Q2) 场景
    // batch_size=4, 两个 Q 网络的输出
    let q1 = Tensor::new(&[1.5, 2.3, 0.8, 3.1], &[4]);
    let q2 = Tensor::new(&[1.2, 2.5, 1.0, 2.9], &[4]);
    let target_q = q1.minimum(&q2);

    assert_eq!(target_q.shape(), &[4]);
    assert_eq!(target_q[[0]], 1.2); // min(1.5, 1.2)
    assert_eq!(target_q[[1]], 2.3); // min(2.3, 2.5)
    assert_eq!(target_q[[2]], 0.8); // min(0.8, 1.0)
    assert_eq!(target_q[[3]], 2.9); // min(3.1, 2.9)
}

#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_minimum_incompatible_shapes() {
    // [3] 和 [4] 无法广播
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let _ = a.minimum(&b);
}
