use crate::tensor::Tensor;
use ndarray::Array;
use ndarray::IxDyn;

use crate::tensor::tests::TensorCheck;

#[test]
fn test_add_with_or_without_ownership() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[4., 5., 6.], &[3]);

    // f32 + 不带引用的张量
    let result = 5. + tensor1.clone();
    let expected = Tensor::new(&[6., 7., 8.], &[3]);
    assert_eq!(result, expected);

    // f32 + 带引用的张量
    let result = 5. + &tensor1;
    let expected = Tensor::new(&[6., 7., 8.], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 + f32
    let result = tensor1.clone() + 5.;
    let expected = Tensor::new(&[6., 7., 8.], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 + f32
    let result = &tensor1 + 5.;
    let expected = Tensor::new(&[6., 7., 8.], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 + 不带引用的张量
    let result = tensor1.clone() + tensor2.clone();
    let expected = Tensor::new(&[5., 7., 9.], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 + 带引用的张量
    let result = tensor1.clone() + &tensor2;
    let expected = Tensor::new(&[5., 7., 9.], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 + 不带引用的张量
    let result = &tensor1 + tensor2.clone();
    let expected = Tensor::new(&[5., 7., 9.], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 + 带引用的张量
    let result = &tensor1 + &tensor2;
    let expected = Tensor::new(&[5., 7., 9.], &[3]);
    assert_eq!(result, expected);

    // 验证原始张量仍然可用
    assert_eq!(tensor1, Tensor::new(&[1., 2., 3.], &[3]));
    assert_eq!(tensor2, Tensor::new(&[4., 5., 6.], &[3]));
}

#[test]
fn test_add_vectors_with_same_shape() {
    let shapes: &[&[usize]] = &[&[3], &[3, 1], &[1, 3]];
    for shape in shapes {
        let vector1 = Tensor::new(&[1., 2., 3.], shape);
        let vector2 = Tensor::new(&[4., 5., 6.], shape);
        let result = vector1 + vector2;
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(shape), vec![5., 7., 9.]).unwrap()
        );
    }
}

#[test]
fn test_add_matrices_with_same_shape() {
    let shape = &[2, 2];
    let matrix1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let matrix2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = matrix1 + matrix2;
    assert_eq!(
        result.data,
        Array::from_shape_vec(IxDyn(shape), vec![6., 8., 10., 12.]).unwrap()
    );
}

#[test]
fn test_add_high_dim_tensors_with_same_shape() {
    let shape = &[2, 1, 2];
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let tensor2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = tensor1 + tensor2;
    assert_eq!(
        result.data,
        Array::from_shape_vec(IxDyn(shape), vec![6., 8., 10., 12.]).unwrap()
    );
}

#[test]
fn test_add_number_and_tensor() {
    let number = 2.;
    // 每个test_cases的元素是个三元组，其元素分别是：张量的形状、张量的数据、正确的结果
    let test_cases = vec![
        // 标量型张量
        TensorCheck {
            input_shape: vec![],
            input_data: vec![1.],
            expected_output: vec![vec![3.]],
        },
        TensorCheck {
            input_shape: vec![1],
            input_data: vec![1.],
            expected_output: vec![vec![3.]],
        },
        TensorCheck {
            input_shape: vec![1, 1],
            input_data: vec![1.],
            expected_output: vec![vec![3.]],
        },
        // 向量型张量
        TensorCheck {
            input_shape: vec![2],
            input_data: vec![1., 2.],
            expected_output: vec![vec![3., 4.]],
        },
        TensorCheck {
            input_shape: vec![2, 1],
            input_data: vec![1., 2.],
            expected_output: vec![vec![3., 4.]],
        },
        TensorCheck {
            input_shape: vec![1, 2],
            input_data: vec![1., 2.],
            expected_output: vec![vec![3., 4.]],
        },
        // 矩阵型张量
        TensorCheck {
            input_shape: vec![2, 3],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![vec![3., 4., 5., 6., 7., 8.]],
        },
        // 高维张量
        TensorCheck {
            input_shape: vec![2, 3, 1],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![vec![3., 4., 5., 6., 7., 8.]],
        },
        TensorCheck {
            input_shape: vec![2, 1, 3, 1],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![vec![3., 4., 5., 6., 7., 8.]],
        },
    ];

    for test_case in test_cases {
        let tensor = Tensor::new(&test_case.input_data, &test_case.input_shape);
        // 1.纯数在前，张量在后
        let result = number + tensor.clone();
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
        let result = tensor + number;
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

/// NumPy 风格广播加法测试（基于 Python NumPy 参考结果）
///
/// 参考脚本: tests/python/tensor_reference/tensor_add_broadcast_reference.py
#[test]
fn test_add_broadcast() {
    // 格式: (shape_a, data_a, shape_b, data_b, expected_shape, expected_data)
    let test_cases: &[(&[usize], &[f32], &[usize], &[f32], &[usize], &[f32])] = &[
        // 1. 相同形状
        // [] + [] -> []
        (&[], &[1.0], &[], &[10.0], &[], &[11.0]),
        // [3] + [3] -> [3]
        (
            &[3],
            &[1.0, 2.0, 3.0],
            &[3],
            &[10.0, 20.0, 30.0],
            &[3],
            &[11.0, 22.0, 33.0],
        ),
        // [2, 3] + [2, 3] -> [2, 3]
        (
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            &[2, 3],
            &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0],
        ),
        // 2. 标量广播
        // [] + [3] -> [3]
        (
            &[],
            &[1.0],
            &[3],
            &[10.0, 20.0, 30.0],
            &[3],
            &[11.0, 21.0, 31.0],
        ),
        // [3] + [] -> [3]
        (
            &[3],
            &[1.0, 2.0, 3.0],
            &[],
            &[10.0],
            &[3],
            &[11.0, 12.0, 13.0],
        ),
        // [] + [2, 3] -> [2, 3]
        (
            &[],
            &[1.0],
            &[2, 3],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            &[2, 3],
            &[11.0, 21.0, 31.0, 41.0, 51.0, 61.0],
        ),
        // [2, 3] + [] -> [2, 3]
        (
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[],
            &[10.0],
            &[2, 3],
            &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ),
        // 3. Linear 层典型场景: [batch, out] + [1, out] 或 [out]
        // [3, 4] + [1, 4] -> [3, 4]
        (
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[1, 4],
            &[10.0, 20.0, 30.0, 40.0],
            &[3, 4],
            &[
                11.0, 22.0, 33.0, 44.0, 15.0, 26.0, 37.0, 48.0, 19.0, 30.0, 41.0, 52.0,
            ],
        ),
        // [3, 4] + [4] -> [3, 4]
        (
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[4],
            &[10.0, 20.0, 30.0, 40.0],
            &[3, 4],
            &[
                11.0, 22.0, 33.0, 44.0, 15.0, 26.0, 37.0, 48.0, 19.0, 30.0, 41.0, 52.0,
            ],
        ),
        // 4. 不同维度数的广播
        // [2, 3, 4] + [4] -> [2, 3, 4]
        (
            &[2, 3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
            &[4],
            &[10.0, 20.0, 30.0, 40.0],
            &[2, 3, 4],
            &[
                11.0, 22.0, 33.0, 44.0, 15.0, 26.0, 37.0, 48.0, 19.0, 30.0, 41.0, 52.0, 23.0, 34.0,
                45.0, 56.0, 27.0, 38.0, 49.0, 60.0, 31.0, 42.0, 53.0, 64.0,
            ],
        ),
        // [2, 3, 4] + [3, 4] -> [2, 3, 4]
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
            &[2, 3, 4],
            &[
                11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0, 110.0, 121.0, 132.0, 23.0,
                34.0, 45.0, 56.0, 67.0, 78.0, 89.0, 100.0, 111.0, 122.0, 133.0, 144.0,
            ],
        ),
        // 5. 双向广播: [3, 1] + [1, 4] -> [3, 4]
        (
            &[3, 1],
            &[1.0, 2.0, 3.0],
            &[1, 4],
            &[10.0, 20.0, 30.0, 40.0],
            &[3, 4],
            &[
                11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,
            ],
        ),
    ];

    for (shape_a, data_a, shape_b, data_b, expected_shape, expected_data) in test_cases {
        let a = Tensor::new(*data_a, *shape_a);
        let b = Tensor::new(*data_b, *shape_b);
        let expected = Tensor::new(*expected_data, *expected_shape);

        // 直接比对 Tensor（同时验证 shape 和 data）
        let result = &a + &b;
        assert_eq!(
            result, expected,
            "Broadcast failed: {:?} + {:?}",
            shape_a, shape_b
        );

        // 验证加法交换律
        let result_swap = &b + &a;
        assert_eq!(
            result_swap, expected,
            "Commutative failed: {:?} + {:?}",
            shape_b, shape_a
        );
    }
}

/// 广播失败测试: 维度不匹配 [3] + [4]
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_add_broadcast_fail_dim_mismatch() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let _ = &a + &b;
}

/// 广播失败测试: 形状不兼容 [2, 3] + [3, 2]
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_add_broadcast_fail_shape_incompatible() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let _ = &a + &b;
}

/// 广播失败测试: 最后一维不匹配 [2, 3] + [4]
#[test]
#[should_panic(expected = "张量形状不兼容")]
fn test_add_broadcast_fail_last_dim_mismatch() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let _ = &a + &b;
}
