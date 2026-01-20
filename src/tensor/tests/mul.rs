use crate::tensor::Tensor;
use ndarray::Array;
use ndarray::IxDyn;

use super::TensorCheck;

#[test]
fn test_mul_with_or_without_ownership() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[4., 5., 6.], &[3]);

    // f32 * 不带引用的张量
    let result = 5. * tensor1.clone();
    let expected = Tensor::new(&[5., 10., 15.], &[3]);
    assert_eq!(result, expected);

    // f32 * 带引用的张量
    let result = 5. * &tensor1;
    let expected = Tensor::new(&[5., 10., 15.], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 * f32
    let result = tensor1.clone() * 5.;
    let expected = Tensor::new(&[5., 10., 15.], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 * f32
    let result = &tensor1 * 5.;
    let expected = Tensor::new(&[5., 10., 15.], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 * 不带引用的张量
    let result = tensor1.clone() * tensor2.clone();
    let expected = Tensor::new(&[4., 10., 18.], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 * 带引用的张量
    let result = tensor1.clone() * &tensor2;
    let expected = Tensor::new(&[4., 10., 18.], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 * 不带引用的张量
    let result = &tensor1 * tensor2.clone();
    let expected = Tensor::new(&[4., 10., 18.], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 * 带引用的张量
    let result = &tensor1 * &tensor2;
    let expected = Tensor::new(&[4., 10., 18.], &[3]);
    assert_eq!(result, expected);

    // 验证原始张量仍然可用
    assert_eq!(tensor1, Tensor::new(&[1., 2., 3.], &[3]));
    assert_eq!(tensor2, Tensor::new(&[4., 5., 6.], &[3]));
}

#[test]
fn test_mul_vectors_with_same_shape() {
    let shapes: &[&[usize]] = &[&[3], &[3, 1], &[1, 3]];
    for shape in shapes {
        let vector1 = Tensor::new(&[1., 2., 3.], shape);
        let vector2 = Tensor::new(&[4., 5., 6.], shape);
        let result = vector1 * vector2;
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(shape), vec![4., 10., 18.]).unwrap()
        );
    }
}

#[test]
fn test_mul_matrices_with_same_shape() {
    let shape = &[2, 2];
    let matrix1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let matrix2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = matrix1 * matrix2;
    assert_eq!(
        result.data,
        Array::from_shape_vec(IxDyn(shape), vec![5., 12., 21., 32.]).unwrap()
    );
}

#[test]
fn test_mul_high_dim_tensors_with_same_shape() {
    let shape = &[2, 1, 2];
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let tensor2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = tensor1 * tensor2;
    assert_eq!(
        result.data,
        Array::from_shape_vec(IxDyn(shape), vec![5., 12., 21., 32.]).unwrap()
    );
}

#[test]
fn test_mul_number_and_tensor() {
    let number = 2.;
    let test_cases = vec![
        // 标量型张量
        TensorCheck {
            input_shape: vec![],
            input_data: vec![1.],
            expected_output: vec![vec![2.]],
        },
        TensorCheck {
            input_shape: vec![1],
            input_data: vec![1.],
            expected_output: vec![vec![2.]],
        },
        TensorCheck {
            input_shape: vec![1, 1],
            input_data: vec![1.],
            expected_output: vec![vec![2.]],
        },
        // 向量型张量
        TensorCheck {
            input_shape: vec![2],
            input_data: vec![1., 2.],
            expected_output: vec![vec![2., 4.]],
        },
        TensorCheck {
            input_shape: vec![2, 1],
            input_data: vec![1., 2.],
            expected_output: vec![vec![2., 4.]],
        },
        TensorCheck {
            input_shape: vec![1, 2],
            input_data: vec![1., 2.],
            expected_output: vec![vec![2., 4.]],
        },
        // 矩阵型张量
        TensorCheck {
            input_shape: vec![2, 3],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![vec![2., 4., 6., 8., 10., 12.]],
        },
        // 高维张量
        TensorCheck {
            input_shape: vec![2, 3, 1],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![vec![2., 4., 6., 8., 10., 12.]],
        },
        TensorCheck {
            input_shape: vec![2, 1, 3, 1],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![vec![2., 4., 6., 8., 10., 12.]],
        },
    ];

    for test_case in test_cases {
        let tensor = Tensor::new(&test_case.input_data, &test_case.input_shape);
        // 1.纯数在前，张量在后
        let result = number * tensor.clone();
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
        let result = tensor * number;
        assert_eq!(
            result.data,
            Array::from_shape_vec(
                IxDyn(&test_case.input_shape),
                test_case.expected_output[0].clone()
            )
            .unwrap(),
            "张量在前，纯数在后：使用的纯数为：{:?}，张量为：{:?}",
            number,
            test_case.input_data
        );
    }
}

/// NumPy 风格广播乘法测试（基于 Python NumPy 参考结果）
///
/// 参考脚本: tests/python/tensor_reference/tensor_mul_broadcast_reference.py
#[test]
fn test_mul_broadcast() {
    // 格式: (shape_a, data_a, shape_b, data_b, expected_shape, expected_data)
    let test_cases: &[(&[usize], &[f32], &[usize], &[f32], &[usize], &[f32])] = &[
        // 1. 相同形状
        // [3] * [3] -> [3]
        (
            &[3],
            &[1.0, 2.0, 3.0],
            &[3],
            &[1.0, 2.0, 3.0],
            &[3],
            &[1.0, 4.0, 9.0],
        ),
        // [2, 3] * [2, 3] -> [2, 3]
        (
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
        ),
        // 2. 标量广播
        // [] * [3] -> [3]
        (&[], &[1.0], &[3], &[1.0, 2.0, 3.0], &[3], &[1.0, 2.0, 3.0]),
        // [3] * [] -> [3]
        (&[3], &[1.0, 2.0, 3.0], &[], &[2.0], &[3], &[2.0, 4.0, 6.0]),
        // [] * [2, 3] -> [2, 3]
        (
            &[],
            &[1.0],
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        // [2, 3] * [] -> [2, 3]
        (
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[],
            &[2.0],
            &[2, 3],
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        ),
        // 3. 低维广播到高维 (mask/scale 场景)
        // [3, 4] * [4] -> [3, 4]
        (
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[4],
            &[1.0, 2.0, 3.0, 4.0],
            &[3, 4],
            &[
                1.0, 4.0, 9.0, 16.0, 5.0, 12.0, 21.0, 32.0, 9.0, 20.0, 33.0, 48.0,
            ],
        ),
        // [3, 4] * [1, 4] -> [3, 4]
        (
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[1, 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[3, 4],
            &[
                1.0, 4.0, 9.0, 16.0, 5.0, 12.0, 21.0, 32.0, 9.0, 20.0, 33.0, 48.0,
            ],
        ),
        // 4. 双向广播: [3, 1] * [1, 4] -> [3, 4]
        (
            &[3, 1],
            &[1.0, 2.0, 3.0],
            &[1, 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[3, 4],
            &[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0],
        ),
        // [1, 4] * [3, 1] -> [3, 4]
        (
            &[1, 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[3, 1],
            &[1.0, 2.0, 3.0],
            &[3, 4],
            &[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0],
        ),
    ];

    for (shape_a, data_a, shape_b, data_b, expected_shape, expected_data) in test_cases {
        let a = Tensor::new(*data_a, *shape_a);
        let b = Tensor::new(*data_b, *shape_b);
        let expected = Tensor::new(*expected_data, *expected_shape);

        // 直接比对 Tensor
        let result = &a * &b;
        assert_eq!(
            result, expected,
            "* broadcast failed: {:?} * {:?}",
            shape_a, shape_b
        );

        // 验证乘法交换律
        let result_swap = &b * &a;
        assert_eq!(
            result_swap, expected,
            "* commutative failed: {:?} * {:?}",
            shape_b, shape_a
        );
    }
}

/// 广播失败测试: 维度不匹配 [3] * [4]
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_mul_broadcast_fail_dim_mismatch() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let _ = &a * &b;
}

/// 广播失败测试: 形状不兼容 [2, 3] * [3, 2]
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_mul_broadcast_fail_shape_incompatible() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let _ = &a * &b;
}

/// 广播失败测试: 最后一维不匹配 [2, 3] * [4]
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_mul_broadcast_fail_last_dim_mismatch() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let _ = &a * &b;
}
