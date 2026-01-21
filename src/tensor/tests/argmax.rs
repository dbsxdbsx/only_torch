//! argmax/argmin 相关测试

use crate::tensor::Tensor;

#[test]
fn test_argmax_1d() {
    let x = Tensor::new(&[1.0, 3.0, 2.0, 5.0, 4.0], &[5]);
    let result = x.argmax(0);
    // 1D 张量沿 axis=0 的 argmax 返回标量（0 维张量）
    assert!(result.is_scalar());
    assert_eq!(result.get_data_number().unwrap(), 3.0); // 最大值 5.0 在索引 3
}

#[test]
fn test_argmax_2d_axis0() {
    // [[1, 3, 2],
    //  [5, 4, 6]]
    let x = Tensor::new(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3]);
    let result = x.argmax(0);

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result[[0]], 1.0); // 第 0 列最大值 5.0 在行索引 1
    assert_eq!(result[[1]], 1.0); // 第 1 列最大值 4.0 在行索引 1
    assert_eq!(result[[2]], 1.0); // 第 2 列最大值 6.0 在行索引 1
}

#[test]
fn test_argmax_2d_axis1() {
    // [[1, 3, 2],
    //  [5, 4, 6]]
    let x = Tensor::new(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3]);
    let result = x.argmax(1);

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result[[0]], 1.0); // 第 0 行最大值 3.0 在索引 1
    assert_eq!(result[[1]], 2.0); // 第 1 行最大值 6.0 在索引 2
}

#[test]
fn test_argmax_3d() {
    // [[[1, 2],
    //   [3, 4]],
    //  [[5, 6],
    //   [7, 8]]]
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);

    // axis=0: 在第一个维度上找最大
    let result0 = x.argmax(0);
    assert_eq!(result0.shape(), &[2, 2]);
    // 每个位置 (i, j)，比较 x[0,i,j] 和 x[1,i,j]，第二个总是更大
    assert_eq!(result0[[0, 0]], 1.0);
    assert_eq!(result0[[0, 1]], 1.0);
    assert_eq!(result0[[1, 0]], 1.0);
    assert_eq!(result0[[1, 1]], 1.0);

    // axis=2: 在最后一个维度上找最大
    let result2 = x.argmax(2);
    assert_eq!(result2.shape(), &[2, 2]);
    // 每行 [a, b]，b > a，所以 argmax = 1
    assert_eq!(result2[[0, 0]], 1.0);
    assert_eq!(result2[[0, 1]], 1.0);
    assert_eq!(result2[[1, 0]], 1.0);
    assert_eq!(result2[[1, 1]], 1.0);
}

#[test]
fn test_argmax_with_ties() {
    // 有多个最大值时，返回第一个
    let x = Tensor::new(&[1.0, 5.0, 5.0, 3.0], &[4]);
    let result = x.argmax(0);
    assert_eq!(result.get_data_number().unwrap(), 1.0); // 第一个 5.0 在索引 1
}

#[test]
fn test_argmax_batch_logits() {
    // 模拟分类任务：batch=4, num_classes=3
    // 每行是一个样本的 logits
    let logits = Tensor::new(
        &[
            0.1, 0.9, 0.0, // 样本 0: 类别 1 最大
            0.8, 0.1, 0.1, // 样本 1: 类别 0 最大
            0.2, 0.3, 0.5, // 样本 2: 类别 2 最大
            0.4, 0.4, 0.2, // 样本 3: 类别 0 最大（tie，取第一个）
        ],
        &[4, 3],
    );

    let predictions = logits.argmax(1);
    assert_eq!(predictions.shape(), &[4]);
    assert_eq!(predictions[[0]], 1.0);
    assert_eq!(predictions[[1]], 0.0);
    assert_eq!(predictions[[2]], 2.0);
    assert_eq!(predictions[[3]], 0.0); // tie 时取第一个
}

#[test]
#[should_panic(expected = "argmax: axis 2 超出维度范围 2")]
fn test_argmax_invalid_axis() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let _ = x.argmax(2); // 只有 axis 0 和 1 有效
}
