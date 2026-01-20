use crate::tensor::Tensor;
use ndarray::Array;
use ndarray::IxDyn;

use crate::tensor::tests::TensorCheck;

#[test]
fn test_div_with_or_without_ownership() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[4., 5., 6.], &[3]);

    // f32 / 不带引用的张量
    let result = 5. / tensor1.clone();
    let expected = Tensor::new(&[5., 2.5, 1.6666666], &[3]);
    assert_eq!(result, expected);

    // f32 / 带引用的张量
    let result = 5. / &tensor1;
    let expected = Tensor::new(&[5., 2.5, 1.6666666], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 / f32
    let result = tensor1.clone() / 5.;
    let expected = Tensor::new(&[0.2, 0.4, 0.6], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 / f32
    let result = &tensor1 / 5.;
    let expected = Tensor::new(&[0.2, 0.4, 0.6], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 / 不带引用的张量
    let result = tensor1.clone() / tensor2.clone();
    let expected = Tensor::new(&[0.25, 0.4, 0.5], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 / 带引用的张量
    let result = tensor1.clone() / &tensor2;
    let expected = Tensor::new(&[0.25, 0.4, 0.5], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 / 不带引用的张量
    let result = &tensor1 / tensor2.clone();
    let expected = Tensor::new(&[0.25, 0.4, 0.5], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 / 带引用的张量
    let result = &tensor1 / &tensor2;
    let expected = Tensor::new(&[0.25, 0.4, 0.5], &[3]);
    assert_eq!(result, expected);

    // 验证原始张量仍然可用
    assert_eq!(tensor1, Tensor::new(&[1., 2., 3.], &[3]));
    assert_eq!(tensor2, Tensor::new(&[4., 5., 6.], &[3]));
}

#[test]
fn test_div_vectors_with_same_shape() {
    let shapes: &[&[usize]] = &[&[3], &[3, 1], &[1, 3]];
    for shape in shapes {
        let vector1 = Tensor::new(&[1., 2., 3.], shape);
        let vector2 = Tensor::new(&[4., 5., 6.], shape);
        let result = vector1 / vector2;
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(shape), vec![0.25, 0.4, 0.5]).unwrap()
        );
    }
}

#[test]
fn test_div_matrices_with_same_shape() {
    let shape = &[2, 2];
    let matrix1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let matrix2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = matrix1 / matrix2;
    assert_eq!(
        result.data,
        Array::from_shape_vec(IxDyn(shape), vec![0.2, 1. / 3., 3. / 7., 0.5]).unwrap()
    );
}

#[test]
fn test_div_high_dim_tensors_with_same_shape() {
    let shape = &[2, 1, 2];
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let tensor2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = tensor1 / tensor2;
    assert_eq!(
        result.data,
        Array::from_shape_vec(IxDyn(shape), vec![0.2, 1. / 3., 3. / 7., 0.5]).unwrap()
    );
}

#[test]
fn test_div_number_and_tensor() {
    let number = 2.;
    // 每个test_cases的元素是个三元组，其元素分别是：张量的形状、张量的数据、正确的结果
    let test_cases = vec![
        // 标量型张量
        TensorCheck {
            input_shape: vec![],
            input_data: vec![1.],
            expected_output: vec![vec![2.], vec![1. / number]],
        },
        TensorCheck {
            input_shape: vec![1],
            input_data: vec![1.],
            expected_output: vec![vec![2.], vec![1. / number]],
        },
        TensorCheck {
            input_shape: vec![1, 1],
            input_data: vec![1.],
            expected_output: vec![vec![2.], vec![1. / number]],
        },
        // 向量型张量
        TensorCheck {
            input_shape: vec![2],
            input_data: vec![1., 2.],
            expected_output: vec![vec![2., 1.], vec![1. / number, 1.]],
        },
        TensorCheck {
            input_shape: vec![2, 1],
            input_data: vec![1., 2.],
            expected_output: vec![vec![2., 1.], vec![1. / number, 1.]],
        },
        TensorCheck {
            input_shape: vec![1, 2],
            input_data: vec![1., 2.],
            expected_output: vec![vec![2., 1.], vec![1. / number, 1.]],
        },
        // 矩阵型张量
        TensorCheck {
            input_shape: vec![2, 3],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![
                vec![
                    number / 1.,
                    number / 2.,
                    number / 3.,
                    number / 4.,
                    number / 5.,
                    number / 6.,
                ],
                vec![
                    1. / number,
                    2. / number,
                    3. / number,
                    4. / number,
                    5. / number,
                    6. / number,
                ],
            ],
        },
        // 高维张量
        TensorCheck {
            input_shape: vec![2, 3, 1],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![
                vec![
                    number / 1.,
                    number / 2.,
                    number / 3.,
                    number / 4.,
                    number / 5.,
                    number / 6.,
                ],
                vec![
                    1. / number,
                    2. / number,
                    3. / number,
                    4. / number,
                    5. / number,
                    6. / number,
                ],
            ],
        },
        TensorCheck {
            input_shape: vec![2, 1, 3, 1],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![
                vec![
                    number / 1.,
                    number / 2.,
                    number / 3.,
                    number / 4.,
                    number / 5.,
                    number / 6.,
                ],
                vec![
                    1. / number,
                    2. / number,
                    3. / number,
                    4. / number,
                    5. / number,
                    6. / number,
                ],
            ],
        },
    ];
    for test_case in test_cases {
        let tensor = Tensor::new(&test_case.input_data, &test_case.input_shape);
        // 1.纯数在前，张量在后
        let result = number / tensor.clone();
        assert_eq!(
            result.data,
            Array::from_shape_vec(
                IxDyn(&test_case.input_shape),
                test_case.expected_output[0].clone()
            )
            .unwrap(),
            "纯数在前，张量在后：使用的纯数为：{:?}，张量为：{:?}",
            number,
            test_case.input_data
        );
        // 2.张量在前，纯数在后
        let result = tensor / number;
        assert_eq!(
            result.data,
            Array::from_shape_vec(
                IxDyn(&test_case.input_shape),
                test_case.expected_output[1].clone()
            )
            .unwrap(),
            "张量在前，纯数在后：使用的纯数为：{:?}，张量为：{:?}",
            number,
            test_case.input_data
        );
    }
}

/// NumPy 风格广播除法测试（基于 Python NumPy 参考结果）
///
/// 注意：除法**不满足交换律** (a / b != b / a)
/// 注意：测试数据避免除数为 0（Only Torch 会 panic）
///
/// 参考脚本: tests/python/tensor_reference/tensor_div_broadcast_reference.py
#[test]
fn test_div_broadcast() {
    // 格式: (shape_a, data_a, shape_b, data_b, expected_shape, expected_data)
    // 使用整除结果避免浮点精度问题
    let test_cases: &[(&[usize], &[f32], &[usize], &[f32], &[usize], &[f32])] = &[
        // 1. 相同形状
        // [3] / [3] -> [3]
        (
            &[3],
            &[2.0, 4.0, 6.0],
            &[3],
            &[1.0, 2.0, 3.0],
            &[3],
            &[2.0, 2.0, 2.0],
        ),
        // [2, 3] / [2, 3] -> [2, 3]
        (
            &[2, 3],
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ),
        // 2. 标量广播
        // [] / [3] -> [3]
        (&[], &[12.0], &[3], &[1.0, 2.0, 3.0], &[3], &[12.0, 6.0, 4.0]),
        // [3] / [] -> [3]
        (&[3], &[2.0, 4.0, 6.0], &[], &[2.0], &[3], &[1.0, 2.0, 3.0]),
        // [] / [2, 3] -> [2, 3]
        (
            &[],
            &[12.0],
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 6.0, 12.0],
            &[2, 3],
            &[12.0, 6.0, 4.0, 3.0, 2.0, 1.0],
        ),
        // [2, 3] / [] -> [2, 3]
        (
            &[2, 3],
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            &[],
            &[2.0],
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        // 3. 低维广播到高维 (normalize 场景)
        // [3, 4] / [4] -> [3, 4]
        (
            &[3, 4],
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
            &[4],
            &[1.0, 2.0, 2.0, 4.0],
            &[3, 4],
            &[2.0, 2.0, 3.0, 2.0, 10.0, 6.0, 7.0, 4.0, 18.0, 10.0, 11.0, 6.0],
        ),
        // 4. 双向广播: [3, 1] / [1, 4] -> [3, 4]
        (
            &[3, 1],
            &[12.0, 24.0, 36.0],
            &[1, 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[3, 4],
            &[12.0, 6.0, 4.0, 3.0, 24.0, 12.0, 8.0, 6.0, 36.0, 18.0, 12.0, 9.0],
        ),
    ];

    for (shape_a, data_a, shape_b, data_b, expected_shape, expected_data) in test_cases {
        let a = Tensor::new(*data_a, *shape_a);
        let b = Tensor::new(*data_b, *shape_b);
        let expected = Tensor::new(*expected_data, *expected_shape);

        // 直接比对 Tensor
        let result = &a / &b;
        assert_eq!(
            result, expected,
            "/ broadcast failed: {:?} / {:?}",
            shape_a, shape_b
        );

        // 注意：除法不满足交换律，不测试 b / a
    }
}

/// 广播失败测试: 维度不匹配 [3] / [4]
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_div_broadcast_fail_dim_mismatch() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let _ = &a / &b;
}

/// 广播失败测试: 形状不兼容 [2, 3] / [3, 2]
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_div_broadcast_fail_shape_incompatible() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let _ = &a / &b;
}

/// 广播失败测试: 最后一维不匹配 [2, 3] / [4]
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_div_broadcast_fail_last_dim_mismatch() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let _ = &a / &b;
}

// ====================== 除以 0 测试 ======================

/// 张量除以纯数 0
#[test]
#[should_panic(expected = "除数为零")]
fn test_div_zero_number() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let scalar = 0.;
    let _ = tensor1 / scalar;
}

/// 相同形状：除数包含 0 元素
#[test]
#[should_panic(expected = "作为除数的张量中存在为零元素")]
fn test_div_zero_same_shape() {
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    let tensor2 = Tensor::new(&[1., 0., 3., 4.], &[2, 2]);
    let _ = tensor1 / tensor2;
}

/// 广播场景：标量 0 广播到向量
/// [3] / [] where [] = 0
#[test]
#[should_panic(expected = "作为除数的张量中存在为零元素")]
fn test_div_zero_broadcast_scalar() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[0.], &[]);
    let _ = &tensor1 / &tensor2;
}

/// 广播场景：向量中含 0 广播到矩阵
/// [2, 3] / [3] where [3] contains 0
#[test]
#[should_panic(expected = "作为除数的张量中存在为零元素")]
fn test_div_zero_broadcast_vector() {
    let tensor1 = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let tensor2 = Tensor::new(&[1., 0., 3.], &[3]); // 中间元素为 0
    let _ = &tensor1 / &tensor2;
}

/// 广播场景：[1, 4] 含 0 广播到 [3, 4]
#[test]
#[should_panic(expected = "作为除数的张量中存在为零元素")]
fn test_div_zero_broadcast_row() {
    let tensor1 = Tensor::new(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], &[3, 4]);
    let tensor2 = Tensor::new(&[1., 2., 0., 4.], &[1, 4]); // 第3个元素为 0
    let _ = &tensor1 / &tensor2;
}
