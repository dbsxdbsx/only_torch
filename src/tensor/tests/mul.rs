use crate::assert_panic;
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
        // 高阶张量
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
        // 1.标量在前，张量在后
        let result = number * tensor.clone();
        assert_eq!(
            result.data,
            Array::from_shape_vec(
                IxDyn(&test_case.input_shape),
                test_case.expected_output[0].clone()
            )
            .unwrap(),
            "标量在前，张量在后：使用的标量为：{:?}，张量为：{:?}",
            number,
            test_case.input_data
        );
        // 2.张量在前，标量在后
        let result = tensor * number;
        assert_eq!(
            result.data,
            Array::from_shape_vec(
                IxDyn(&test_case.input_shape),
                test_case.expected_output[0].clone()
            )
            .unwrap(),
            "张量在前，标量在后：使用的标量为：{:?}，张量为：{:?}",
            number,
            test_case.input_data
        );
    }
}

#[test]
fn test_mul_scalars_among_various_shapes() {
    let number = 2.;
    let scalar_shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1]];

    // 测试不同形状标量间的乘法组合
    for shape1 in scalar_shapes.iter() {
        let scalar1 = Tensor::new(&[number], shape1);

        for shape2 in scalar_shapes.iter() {
            let scalar2 = Tensor::new(&[1.0], shape2);

            if shape1 == shape2 {
                // 相同形状的标量张量相乘应该成功
                let result = scalar1.clone() * scalar2.clone();
                let expected = Tensor::new(&[2.0], shape1);
                assert_eq!(result, expected);

                let result = scalar2 * scalar1.clone();
                let expected = Tensor::new(&[2.0], shape1);
                assert_eq!(result, expected);
            } else {
                // 不同形状的标量张量相乘应该失败
                let expected_msg = format!(
                    "形状不一致，故无法相乘：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
                    shape1, shape2
                );
                assert_panic!(scalar1.clone() * scalar2.clone(), expected_msg);

                let expected_msg = format!(
                    "形状不一致，故无法相乘：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
                    shape2, shape1
                );
                assert_panic!(scalar2 * scalar1.clone(), expected_msg);
            }
        }
    }
}
