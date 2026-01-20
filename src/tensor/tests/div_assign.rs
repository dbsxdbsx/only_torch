use crate::tensor::Tensor;

#[test]
fn test_div_assign_f32_to_tensor() {
    // 标量
    let mut tensor = Tensor::new(&[2.0], &[]);
    tensor /= 2.0;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(tensor, expected);
    // 向量
    let mut tensor = Tensor::new(&[2.0, 4.0, 6.0], &[3]);
    tensor /= 2.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(tensor, expected);
    // 矩阵
    let mut tensor = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    tensor /= 2.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(tensor, expected);
    // 3维张量
    let mut tensor = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 1, 2]);
    tensor /= 2.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(tensor, expected);
}

#[test]
fn test_div_assign_f32_to_tensor_ref() {
    // 标量
    let mut tensor = Tensor::new(&[2.0], &[]);
    let tensor_ref = &mut tensor;
    *tensor_ref /= 2.0;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(*tensor_ref, expected);
    // 向量
    let mut tensor = Tensor::new(&[2.0, 4.0, 6.0], &[3]);
    let tensor_ref = &mut tensor;
    *tensor_ref /= 2.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(*tensor_ref, expected);
    // 矩阵
    let mut tensor = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    let tensor_ref = &mut tensor;
    *tensor_ref /= 2.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(*tensor_ref, expected);
    // 3维张量
    let mut tensor = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 1, 2]);
    let tensor_ref = &mut tensor;
    *tensor_ref /= 2.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(*tensor_ref, expected);
}

#[test]
fn test_div_assign_tensor_to_tensor() {
    // 标量
    let mut tensor1 = Tensor::new(&[2.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    tensor1 /= tensor2;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(tensor1, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0], &[3]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0], &[3]);
    tensor1 /= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(tensor1, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);
    tensor1 /= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(tensor1, expected);
    // 3维张量
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 1, 2]);
    tensor1 /= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(tensor1, expected);
}

#[test]
fn test_div_assign_tensor_ref_to_tensor() {
    // 标量
    let mut tensor1 = Tensor::new(&[2.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    tensor1 /= &tensor2;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(tensor1, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0], &[3]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0], &[3]);
    tensor1 /= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(tensor1, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);
    tensor1 /= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(tensor1, expected);
    // 3维张量
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 1, 2]);
    tensor1 /= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(tensor1, expected);
}

#[test]
fn test_div_assign_tensor_to_tensor_ref() {
    // 标量
    let mut tensor1 = Tensor::new(&[2.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref /= tensor2;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(*tensor1_ref, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0], &[3]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0], &[3]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref /= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(*tensor1_ref, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref /= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(*tensor1_ref, expected);
    // 3维张量
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 1, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref /= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(*tensor1_ref, expected);
}

#[test]
fn test_div_assign_tensor_ref_to_tensor_ref() {
    // 标量
    let mut tensor1 = Tensor::new(&[2.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref /= &tensor2;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(*tensor1_ref, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0], &[3]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0], &[3]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref /= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(*tensor1_ref, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref /= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(*tensor1_ref, expected);
    // 3维张量
    let mut tensor1 = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 1, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref /= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(*tensor1_ref, expected);
}

/// /= 广播测试（基于 NumPy 参考结果）
///
/// 规则：`a /= b` 支持广播，但广播后的结果形状必须与 a 形状相同
/// 注意：测试数据避免除数为 0（Only Torch 会 panic）
///
/// 参考脚本: tests/python/tensor_reference/tensor_div_assign_broadcast_reference.py
#[test]
fn test_div_assign_broadcast() {
    // 格式: (shape_a, data_a, shape_b, data_b, expected_data)
    // 使用整除结果避免浮点精度问题
    let test_cases: &[(&[usize], &[f32], &[usize], &[f32], &[f32])] = &[
        // [] /= []
        (&[], &[10.0], &[], &[2.0], &[5.0]),
        // [3] /= [3]
        (
            &[3],
            &[10.0, 12.0, 18.0],
            &[3],
            &[2.0, 3.0, 6.0],
            &[5.0, 4.0, 3.0],
        ),
        // [2, 3] /= [2, 3]
        (
            &[2, 3],
            &[10.0, 12.0, 18.0, 20.0, 24.0, 30.0],
            &[2, 3],
            &[2.0, 3.0, 6.0, 4.0, 6.0, 10.0],
            &[5.0, 4.0, 3.0, 5.0, 4.0, 3.0],
        ),
        // [3] /= []
        (&[3], &[10.0, 12.0, 14.0], &[], &[2.0], &[5.0, 6.0, 7.0]),
        // [2, 3] /= []
        (
            &[2, 3],
            &[10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            &[],
            &[2.0],
            &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        // [2, 3] /= [3]
        (
            &[2, 3],
            &[10.0, 12.0, 18.0, 20.0, 24.0, 36.0],
            &[3],
            &[2.0, 3.0, 6.0],
            &[5.0, 4.0, 3.0, 10.0, 8.0, 6.0],
        ),
        // [3, 4] /= [4] (normalize 场景)
        (
            &[3, 4],
            &[10.0, 12.0, 18.0, 20.0, 20.0, 24.0, 36.0, 40.0, 30.0, 36.0, 54.0, 60.0],
            &[4],
            &[2.0, 3.0, 6.0, 10.0],
            &[5.0, 4.0, 3.0, 2.0, 10.0, 8.0, 6.0, 4.0, 15.0, 12.0, 9.0, 6.0],
        ),
        // [3, 4] /= [3, 1] (列广播)
        (
            &[3, 4],
            &[10.0, 20.0, 30.0, 40.0, 20.0, 40.0, 60.0, 80.0, 30.0, 60.0, 90.0, 120.0],
            &[3, 1],
            &[2.0, 4.0, 6.0],
            &[5.0, 10.0, 15.0, 20.0, 5.0, 10.0, 15.0, 20.0, 5.0, 10.0, 15.0, 20.0],
        ),
    ];

    for (shape_a, data_a, shape_b, data_b, expected_data) in test_cases {
        let mut a = Tensor::new(*data_a, *shape_a);
        let b = Tensor::new(*data_b, *shape_b);
        let expected = Tensor::new(*expected_data, *shape_a);

        a /= &b;
        assert_eq!(
            a, expected,
            "/= broadcast failed: {:?} /= {:?}",
            shape_a, shape_b
        );
    }
}

/// /= 广播失败: 标量 /= 向量
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_div_assign_broadcast_fail_scalar_to_vector() {
    let mut a = Tensor::new(&[10.0], &[]);
    let b = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    a /= b;
}

/// /= 广播失败: 向量 /= 矩阵
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_div_assign_broadcast_fail_vector_to_matrix() {
    let mut a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    a /= b;
}

/// /= 广播失败: 最后一维不兼容
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_div_assign_broadcast_fail_incompatible_last_dim() {
    let mut a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    a /= b;
}

// ====================== 除以 0 测试 ======================

/// 张量除以纯数 0
#[test]
#[should_panic(expected = "除数为零")]
fn test_div_assign_zero_number() {
    let mut tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let scalar = 0.;
    tensor1 /= scalar;
}

/// 相同形状：除数包含 0 元素
#[test]
#[should_panic(expected = "作为除数的张量中存在为零元素")]
fn test_div_assign_zero_same_shape() {
    let mut tensor1 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    let tensor2 = Tensor::new(&[1., 0., 3., 4.], &[2, 2]);
    tensor1 /= tensor2;
}

/// 广播场景：标量 0 广播
/// [3] /= [] where [] = 0
#[test]
#[should_panic(expected = "作为除数的张量中存在为零元素")]
fn test_div_assign_zero_broadcast_scalar() {
    let mut tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[0.], &[]);
    tensor1 /= &tensor2;
}

/// 广播场景：向量中含 0 广播到矩阵
/// [2, 3] /= [3] where [3] contains 0
#[test]
#[should_panic(expected = "作为除数的张量中存在为零元素")]
fn test_div_assign_zero_broadcast_vector() {
    let mut tensor1 = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let tensor2 = Tensor::new(&[1., 0., 3.], &[3]); // 中间元素为 0
    tensor1 /= &tensor2;
}

/// 广播场景：[1, 4] 含 0 广播到 [3, 4]
#[test]
#[should_panic(expected = "作为除数的张量中存在为零元素")]
fn test_div_assign_zero_broadcast_row() {
    let mut tensor1 = Tensor::new(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], &[3, 4]);
    let tensor2 = Tensor::new(&[1., 2., 0., 4.], &[1, 4]); // 第3个元素为 0
    tensor1 /= &tensor2;
}
