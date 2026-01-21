//! argmin 相关测试

use crate::tensor::Tensor;

#[test]
fn test_argmin_1d() {
    let x = Tensor::new(&[5.0, 3.0, 4.0, 1.0, 2.0], &[5]);
    let result = x.argmin(0);
    // 1D 张量沿 axis=0 的 argmin 返回标量（0 维张量）
    assert!(result.is_scalar());
    assert_eq!(result.get_data_number().unwrap(), 3.0); // 最小值 1.0 在索引 3
}

#[test]
fn test_argmin_2d_axis0() {
    // [[5, 3, 6],
    //  [1, 4, 2]]
    let x = Tensor::new(&[5.0, 3.0, 6.0, 1.0, 4.0, 2.0], &[2, 3]);
    let result = x.argmin(0);

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result[[0]], 1.0); // 第 0 列最小值 1.0 在行索引 1
    assert_eq!(result[[1]], 0.0); // 第 1 列最小值 3.0 在行索引 0
    assert_eq!(result[[2]], 1.0); // 第 2 列最小值 2.0 在行索引 1
}

#[test]
fn test_argmin_2d_axis1() {
    // [[5, 3, 6],
    //  [1, 4, 2]]
    let x = Tensor::new(&[5.0, 3.0, 6.0, 1.0, 4.0, 2.0], &[2, 3]);
    let result = x.argmin(1);

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result[[0]], 1.0); // 第 0 行最小值 3.0 在索引 1
    assert_eq!(result[[1]], 0.0); // 第 1 行最小值 1.0 在索引 0
}

#[test]
fn test_argmin_3d() {
    // [[[8, 7],
    //   [6, 5]],
    //  [[4, 3],
    //   [2, 1]]]
    let x = Tensor::new(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], &[2, 2, 2]);

    // axis=0: 在第一个维度上找最小
    let result0 = x.argmin(0);
    assert_eq!(result0.shape(), &[2, 2]);
    // 每个位置 (i, j)，比较 x[0,i,j] 和 x[1,i,j]，第二个总是更小
    assert_eq!(result0[[0, 0]], 1.0);
    assert_eq!(result0[[0, 1]], 1.0);
    assert_eq!(result0[[1, 0]], 1.0);
    assert_eq!(result0[[1, 1]], 1.0);

    // axis=2: 在最后一个维度上找最小
    let result2 = x.argmin(2);
    assert_eq!(result2.shape(), &[2, 2]);
    // 每行 [a, b]，b < a，所以 argmin = 1
    assert_eq!(result2[[0, 0]], 1.0);
    assert_eq!(result2[[0, 1]], 1.0);
    assert_eq!(result2[[1, 0]], 1.0);
    assert_eq!(result2[[1, 1]], 1.0);
}

#[test]
fn test_argmin_with_ties() {
    // 有多个最小值时，返回第一个
    let x = Tensor::new(&[5.0, 1.0, 1.0, 3.0], &[4]);
    let result = x.argmin(0);
    assert_eq!(result.get_data_number().unwrap(), 1.0); // 第一个 1.0 在索引 1
}

#[test]
fn test_argmin_batch_losses() {
    // 模拟选择最小损失的场景：batch=4, num_options=3
    let losses = Tensor::new(
        &[
            0.9, 0.1, 0.5, // 样本 0: 选项 1 最小
            0.1, 0.8, 0.7, // 样本 1: 选项 0 最小
            0.3, 0.2, 0.1, // 样本 2: 选项 2 最小
            0.2, 0.2, 0.5, // 样本 3: 选项 0 最小（tie，取第一个）
        ],
        &[4, 3],
    );

    let selections = losses.argmin(1);
    assert_eq!(selections.shape(), &[4]);
    assert_eq!(selections[[0]], 1.0);
    assert_eq!(selections[[1]], 0.0);
    assert_eq!(selections[[2]], 2.0);
    assert_eq!(selections[[3]], 0.0); // tie 时取第一个
}

#[test]
#[should_panic(expected = "argmin: axis 2 超出维度范围 2")]
fn test_argmin_invalid_axis() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let _ = x.argmin(2); // 只有 axis 0 和 1 有效
}

#[test]
fn test_argmin_negative_values() {
    // 测试负数场景
    let x = Tensor::new(&[-1.0, -5.0, -3.0, -2.0], &[4]);
    let result = x.argmin(0);
    assert_eq!(result.get_data_number().unwrap(), 1.0); // 最小值 -5.0 在索引 1
}
