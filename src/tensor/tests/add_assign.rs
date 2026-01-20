use crate::tensor::Tensor;

#[test]
fn test_add_assign_f32_to_tensor() {
    // 标量
    let mut tensor = Tensor::new(&[1.0], &[]);
    tensor += 2.0;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(tensor, expected);
    // 向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    tensor += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(tensor, expected);
    // 矩阵
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    tensor += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(tensor, expected);
    // 3维张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    tensor += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(tensor, expected);
}

#[test]
fn test_add_assign_f32_to_tensor_ref() {
    // 标量
    let mut tensor = Tensor::new(&[1.0], &[]);
    let tensor_ref = &mut tensor;
    *tensor_ref += 2.0;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(*tensor_ref, expected);
    // 向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor_ref = &mut tensor;
    *tensor_ref += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(*tensor_ref, expected);
    // 矩阵
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor_ref = &mut tensor;
    *tensor_ref += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(*tensor_ref, expected);
    // 3维张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let tensor_ref = &mut tensor;
    *tensor_ref += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(*tensor_ref, expected);
}

#[test]
fn test_add_assign_tensor_to_tensor() {
    // 标量
    let mut tensor1 = Tensor::new(&[1.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    tensor1 += tensor2;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(tensor1, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    tensor1 += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(tensor1, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    tensor1 += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(tensor1, expected);
    // 3维张量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    tensor1 += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(tensor1, expected);
}

#[test]
fn test_add_assign_tensor_ref_to_tensor() {
    // 标量
    let mut tensor1 = Tensor::new(&[1.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    tensor1 += &tensor2;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(tensor1, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    tensor1 += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(tensor1, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    tensor1 += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(tensor1, expected);
    // 3维张量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    tensor1 += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(tensor1, expected);
}

#[test]
fn test_add_assign_tensor_to_tensor_ref() {
    // 标量
    let mut tensor1 = Tensor::new(&[1.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += tensor2;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(*tensor1_ref, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(*tensor1_ref, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(*tensor1_ref, expected);
    // 3维张量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(*tensor1_ref, expected);
}

#[test]
fn test_add_assign_tensor_ref_to_tensor_ref() {
    // 标量
    let mut tensor1 = Tensor::new(&[1.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += &tensor2;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(*tensor1_ref, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(*tensor1_ref, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(*tensor1_ref, expected);
    // 3维张量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(*tensor1_ref, expected);
}

/// += 广播测试（基于 NumPy 参考结果）
///
/// 规则：`a += b` 支持广播，但广播后的结果形状必须与 a 形状相同
///
/// 参考脚本: tests/python/tensor_reference/tensor_add_assign_broadcast_reference.py
#[test]
fn test_add_assign_broadcast() {
    // 格式: (shape_a, data_a, shape_b, data_b, expected_data)
    // 注意: 结果形状始终等于 shape_a
    let test_cases: &[(&[usize], &[f32], &[usize], &[f32], &[f32])] = &[
        // 1. 相同形状
        // [3] += [3]
        (
            &[3],
            &[1.0, 2.0, 3.0],
            &[3],
            &[10.0, 20.0, 30.0],
            &[11.0, 22.0, 33.0],
        ),
        // [2, 3] += [2, 3]
        (
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0],
        ),
        // 2. 标量广播
        // [3] += []
        (&[3], &[1.0, 2.0, 3.0], &[], &[10.0], &[11.0, 12.0, 13.0]),
        // [2, 3] += []
        (
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[],
            &[10.0],
            &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ),
        // 3. 低维广播到高维
        // [2, 3] += [3]
        (
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3],
            &[10.0, 20.0, 30.0],
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        ),
        // [2, 3] += [1, 3]
        (
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 3],
            &[10.0, 20.0, 30.0],
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        ),
        // 4. Linear 层典型场景: [batch, out] += [out] 或 [1, out]
        // [3, 4] += [4]
        (
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[4],
            &[10.0, 20.0, 30.0, 40.0],
            &[
                11.0, 22.0, 33.0, 44.0, 15.0, 26.0, 37.0, 48.0, 19.0, 30.0, 41.0, 52.0,
            ],
        ),
        // [3, 4] += [1, 4]
        (
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[1, 4],
            &[10.0, 20.0, 30.0, 40.0],
            &[
                11.0, 22.0, 33.0, 44.0, 15.0, 26.0, 37.0, 48.0, 19.0, 30.0, 41.0, 52.0,
            ],
        ),
        // 5. 列广播: [3, 4] += [3, 1]
        (
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 1],
            &[10.0, 20.0, 30.0],
            &[
                11.0, 12.0, 13.0, 14.0, 25.0, 26.0, 27.0, 28.0, 39.0, 40.0, 41.0, 42.0,
            ],
        ),
        // 6. 3D 广播
        // [2, 3, 4] += [4]
        (
            &[2, 3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
            &[4],
            &[10.0, 20.0, 30.0, 40.0],
            &[
                11.0, 22.0, 33.0, 44.0, 15.0, 26.0, 37.0, 48.0, 19.0, 30.0, 41.0, 52.0, 23.0, 34.0,
                45.0, 56.0, 27.0, 38.0, 49.0, 60.0, 31.0, 42.0, 53.0, 64.0,
            ],
        ),
        // [2, 3, 4] += [3, 4]
        (
            &[2, 3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
            &[3, 4],
            &[
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
            ],
            &[
                11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0, 110.0, 121.0, 132.0, 23.0,
                34.0, 45.0, 56.0, 67.0, 78.0, 89.0, 100.0, 111.0, 122.0, 133.0, 144.0,
            ],
        ),
    ];

    for (shape_a, data_a, shape_b, data_b, expected_data) in test_cases {
        let mut a = Tensor::new(*data_a, *shape_a);
        let b = Tensor::new(*data_b, *shape_b);
        let expected = Tensor::new(*expected_data, *shape_a); // 形状不变

        a += &b;
        assert_eq!(
            a, expected,
            "+= broadcast failed: {:?} += {:?}",
            shape_a, shape_b
        );
    }
}

/// += 广播失败: 标量 += 向量（形状会改变）
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_add_assign_broadcast_fail_scalar_to_vector() {
    // [] += [3] → 失败
    let mut a = Tensor::new(&[1.0], &[]);
    let b = Tensor::new(&[10.0, 20.0, 30.0], &[3]);
    a += b;
}

/// += 广播失败: 向量 += 矩阵（形状会改变）
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_add_assign_broadcast_fail_vector_to_matrix() {
    // [3] += [2, 3] → 失败
    let mut a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);
    a += b;
}

/// += 广播失败: 最后一维不兼容
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_add_assign_broadcast_fail_incompatible_last_dim() {
    // [2, 3] += [4] → 失败（3 != 4）
    let mut a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[4]);
    a += b;
}

/// += 广播失败: 转置形状不兼容
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_add_assign_broadcast_fail_transposed_shape() {
    // [2, 3] += [3, 2] → 失败
    let mut a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]);
    a += b;
}
