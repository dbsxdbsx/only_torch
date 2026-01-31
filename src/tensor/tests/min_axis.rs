//! amin(axis) 相关测试

use crate::tensor::Tensor;

#[test]
fn test_amin_1d() {
    let x = Tensor::new(&[5.0, 3.0, 4.0, 1.0, 2.0], &[5]);
    let result = x.amin(0);
    // 1D 张量沿 axis=0 的 amin 返回标量（0 维张量）
    assert!(result.is_scalar());
    assert_eq!(result.get_data_number().unwrap(), 1.0);
}

#[test]
fn test_amin_2d_axis0() {
    // [[5, 3, 6],
    //  [1, 4, 2]]
    let x = Tensor::new(&[5.0, 3.0, 6.0, 1.0, 4.0, 2.0], &[2, 3]);
    let result = x.amin(0);

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result[[0]], 1.0); // min(5, 1) = 1
    assert_eq!(result[[1]], 3.0); // min(3, 4) = 3
    assert_eq!(result[[2]], 2.0); // min(6, 2) = 2
}

#[test]
fn test_amin_2d_axis1() {
    // [[5, 3, 6],
    //  [1, 4, 2]]
    let x = Tensor::new(&[5.0, 3.0, 6.0, 1.0, 4.0, 2.0], &[2, 3]);
    let result = x.amin(1);

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result[[0]], 3.0); // min(5, 3, 6) = 3
    assert_eq!(result[[1]], 1.0); // min(1, 4, 2) = 1
}

#[test]
fn test_amin_3d() {
    // [[[8, 7],
    //   [6, 5]],
    //  [[4, 3],
    //   [2, 1]]]
    let x = Tensor::new(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], &[2, 2, 2]);

    // axis=0: 在第一个维度上找最小
    let result0 = x.amin(0);
    assert_eq!(result0.shape(), &[2, 2]);
    assert_eq!(result0[[0, 0]], 4.0); // min(8, 4)
    assert_eq!(result0[[0, 1]], 3.0); // min(7, 3)
    assert_eq!(result0[[1, 0]], 2.0); // min(6, 2)
    assert_eq!(result0[[1, 1]], 1.0); // min(5, 1)

    // axis=2: 在最后一个维度上找最小
    let result2 = x.amin(2);
    assert_eq!(result2.shape(), &[2, 2]);
    assert_eq!(result2[[0, 0]], 7.0); // min(8, 7)
    assert_eq!(result2[[0, 1]], 5.0); // min(6, 5)
    assert_eq!(result2[[1, 0]], 3.0); // min(4, 3)
    assert_eq!(result2[[1, 1]], 1.0); // min(2, 1)
}

#[test]
fn test_amin_with_negative_values() {
    let x = Tensor::new(&[-1.0, -5.0, -3.0, -2.0], &[4]);
    let result = x.amin(0);
    assert_eq!(result.get_data_number().unwrap(), -5.0);
}

#[test]
fn test_amin_batch_losses() {
    // 模拟 batch 中找最小损失的场景：batch=4, num_options=3
    let losses = Tensor::new(
        &[
            0.5, 0.3, 0.8, // sample 0
            0.2, 0.6, 0.4, // sample 1
            0.9, 0.1, 0.7, // sample 2
            0.4, 0.4, 0.2, // sample 3
        ],
        &[4, 3],
    );

    let min_losses = losses.amin(1);
    assert_eq!(min_losses.shape(), &[4]);
    assert_eq!(min_losses[[0]], 0.3); // min(0.5, 0.3, 0.8)
    assert_eq!(min_losses[[1]], 0.2); // min(0.2, 0.6, 0.4)
    assert_eq!(min_losses[[2]], 0.1); // min(0.9, 0.1, 0.7)
    assert_eq!(min_losses[[3]], 0.2); // min(0.4, 0.4, 0.2)
}

#[test]
fn test_amin_consistent_with_argmin() {
    // 验证 amin 和 argmin 的一致性
    let x = Tensor::new(&[5.0, 3.0, 6.0, 1.0, 4.0, 2.0], &[2, 3]);

    let min_vals = x.amin(1);
    let min_indices = x.argmin(1);

    // min_vals[i] 应该等于 x[i, argmin_indices[i]]
    assert_eq!(min_vals[[0]], 3.0);
    assert_eq!(min_indices[[0]], 1.0); // index 1 -> x[0, 1] = 3.0

    assert_eq!(min_vals[[1]], 1.0);
    assert_eq!(min_indices[[1]], 0.0); // index 0 -> x[1, 0] = 1.0
}

#[test]
#[should_panic(expected = "amin: axis 2 超出维度范围 2")]
fn test_amin_invalid_axis() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let _ = x.amin(2); // 只有 axis 0 和 1 有效
}

#[test]
fn test_amin_all_same_values() {
    let x = Tensor::new(&[3.0, 3.0, 3.0, 3.0], &[2, 2]);
    let result = x.amin(0);
    assert_eq!(result[[0]], 3.0);
    assert_eq!(result[[1]], 3.0);
}
