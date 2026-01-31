//! max(axis) 相关测试

use crate::tensor::Tensor;

#[test]
fn test_max_1d() {
    let x = Tensor::new(&[5.0, 3.0, 4.0, 1.0, 2.0], &[5]);
    let result = x.max(0);
    // 1D 张量沿 axis=0 的 max 返回标量（0 维张量）
    assert!(result.is_scalar());
    assert_eq!(result.get_data_number().unwrap(), 5.0);
}

#[test]
fn test_max_2d_axis0() {
    // [[5, 3, 6],
    //  [1, 4, 2]]
    let x = Tensor::new(&[5.0, 3.0, 6.0, 1.0, 4.0, 2.0], &[2, 3]);
    let result = x.max(0);

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result[[0]], 5.0); // max(5, 1) = 5
    assert_eq!(result[[1]], 4.0); // max(3, 4) = 4
    assert_eq!(result[[2]], 6.0); // max(6, 2) = 6
}

#[test]
fn test_max_2d_axis1() {
    // [[5, 3, 6],
    //  [1, 4, 2]]
    let x = Tensor::new(&[5.0, 3.0, 6.0, 1.0, 4.0, 2.0], &[2, 3]);
    let result = x.max(1);

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result[[0]], 6.0); // max(5, 3, 6) = 6
    assert_eq!(result[[1]], 4.0); // max(1, 4, 2) = 4
}

#[test]
fn test_max_3d() {
    // [[[1, 2],
    //   [3, 4]],
    //  [[5, 6],
    //   [7, 8]]]
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);

    // axis=0: 在第一个维度上找最大
    let result0 = x.max(0);
    assert_eq!(result0.shape(), &[2, 2]);
    assert_eq!(result0[[0, 0]], 5.0); // max(1, 5)
    assert_eq!(result0[[0, 1]], 6.0); // max(2, 6)
    assert_eq!(result0[[1, 0]], 7.0); // max(3, 7)
    assert_eq!(result0[[1, 1]], 8.0); // max(4, 8)

    // axis=2: 在最后一个维度上找最大
    let result2 = x.max(2);
    assert_eq!(result2.shape(), &[2, 2]);
    assert_eq!(result2[[0, 0]], 2.0); // max(1, 2)
    assert_eq!(result2[[0, 1]], 4.0); // max(3, 4)
    assert_eq!(result2[[1, 0]], 6.0); // max(5, 6)
    assert_eq!(result2[[1, 1]], 8.0); // max(7, 8)
}

#[test]
fn test_max_with_negative_values() {
    let x = Tensor::new(&[-1.0, -5.0, -3.0, -2.0], &[4]);
    let result = x.max(0);
    assert_eq!(result.get_data_number().unwrap(), -1.0);
}

#[test]
fn test_max_batch_logits() {
    // 模拟 batch 中找最大 logit 的场景：batch=4, num_classes=3
    let logits = Tensor::new(
        &[
            0.5, 0.8, 0.3, // sample 0
            0.2, 0.6, 0.9, // sample 1
            0.7, 0.1, 0.4, // sample 2
            0.4, 0.4, 0.6, // sample 3
        ],
        &[4, 3],
    );

    let max_logits = logits.max(1);
    assert_eq!(max_logits.shape(), &[4]);
    assert_eq!(max_logits[[0]], 0.8); // max(0.5, 0.8, 0.3)
    assert_eq!(max_logits[[1]], 0.9); // max(0.2, 0.6, 0.9)
    assert_eq!(max_logits[[2]], 0.7); // max(0.7, 0.1, 0.4)
    assert_eq!(max_logits[[3]], 0.6); // max(0.4, 0.4, 0.6)
}

#[test]
fn test_max_consistent_with_argmax() {
    // 验证 max 和 argmax 的一致性
    let x = Tensor::new(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3]);

    let max_vals = x.max(1);
    let max_indices = x.argmax(1);

    // max_vals[i] 应该等于 x[i, argmax_indices[i]]
    assert_eq!(max_vals[[0]], 3.0);
    assert_eq!(max_indices[[0]], 1.0); // index 1 -> x[0, 1] = 3.0

    assert_eq!(max_vals[[1]], 6.0);
    assert_eq!(max_indices[[1]], 2.0); // index 2 -> x[1, 2] = 6.0
}

#[test]
#[should_panic(expected = "max: axis 2 超出维度范围 2")]
fn test_max_invalid_axis() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let _ = x.max(2); // 只有 axis 0 和 1 有效
}

#[test]
fn test_max_all_same_values() {
    let x = Tensor::new(&[3.0, 3.0, 3.0, 3.0], &[2, 2]);
    let result = x.max(0);
    assert_eq!(result[[0]], 3.0);
    assert_eq!(result[[1]], 3.0);
}
