use crate::assert_panic;

use crate::tensor::Tensor;
use ndarray::Array;
use ndarray::IxDyn;

use crate::tensor::tests::TensorCheck;

#[test]
fn test_sub_with_or_without_ownership() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[4., 5., 6.], &[3]);

    // f32 - 不带引用的张量
    let result = 5. - tensor1.clone();
    let expected = Tensor::new(&[4., 3., 2.], &[3]);
    assert_eq!(result, expected);

    // f32 - 带引用的张量
    let result = 5. - &tensor1;
    let expected = Tensor::new(&[4., 3., 2.], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 - f32
    let result = tensor1.clone() - 5.;
    let expected = Tensor::new(&[-4., -3., -2.], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 - f32
    let result = &tensor1 - 5.;
    let expected = Tensor::new(&[-4., -3., -2.], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 - 不带引用的张量
    let result = tensor1.clone() - tensor2.clone();
    let expected = Tensor::new(&[-3., -3., -3.], &[3]);
    assert_eq!(result, expected);

    // 不带引用的张量 - 带引用的张量
    let result = tensor1.clone() - &tensor2;
    let expected = Tensor::new(&[-3., -3., -3.], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 - 不带引用的张量
    let result = &tensor1 - tensor2.clone();
    let expected = Tensor::new(&[-3., -3., -3.], &[3]);
    assert_eq!(result, expected);

    // 带引用的张量 - 带引用的张量
    let result = &tensor1 - &tensor2;
    let expected = Tensor::new(&[-3., -3., -3.], &[3]);
    assert_eq!(result, expected);

    // 验证原始张量仍然可用
    assert_eq!(tensor1, Tensor::new(&[1., 2., 3.], &[3]));
    assert_eq!(tensor2, Tensor::new(&[4., 5., 6.], &[3]));
}

#[test]
fn test_sub_vectors_with_same_shape() {
    let shapes: &[&[usize]] = &[&[3], &[3, 1], &[1, 3]];
    for shape in shapes {
        let vector1 = Tensor::new(&[1., 2., 3.], shape);
        let vector2 = Tensor::new(&[4., 5., 6.], shape);
        let result = vector1 - vector2;
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(shape), vec![-3., -3., -3.]).unwrap()
        );
    }
}

#[test]
fn test_sub_matrices_with_same_shape() {
    let shape = &[2, 2];
    let matrix1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let matrix2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = matrix1 - matrix2;
    assert_eq!(
        result.data,
        Array::from_shape_vec(IxDyn(shape), vec![-4., -4., -4., -4.]).unwrap()
    );
}

#[test]
fn test_sub_high_dim_tensors_with_same_shape() {
    let shape = &[2, 1, 2];
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let tensor2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = tensor1 - tensor2;
    assert_eq!(
        result.data,
        Array::from_shape_vec(IxDyn(shape), vec![-4., -4., -4., -4.]).unwrap()
    );
}

#[test]
fn test_sub_number_and_tensor() {
    let number = 2.;
    // 每个test_cases的元素是个三元组，其元素分别是：张量的形状、张量的数据、正确的结果
    let test_cases = vec![
        // 标量型张量
        TensorCheck {
            input_shape: vec![],
            input_data: vec![1.],
            expected_output: vec![vec![1.], vec![-1.]],
        },
        TensorCheck {
            input_shape: vec![1],
            input_data: vec![1.],
            expected_output: vec![vec![1.], vec![-1.]],
        },
        TensorCheck {
            input_shape: vec![1, 1],
            input_data: vec![1.],
            expected_output: vec![vec![1.], vec![-1.]],
        },
        // 向量型张量
        TensorCheck {
            input_shape: vec![2],
            input_data: vec![1., 2.],
            expected_output: vec![vec![1., 0.], vec![-1., 0.]],
        },
        TensorCheck {
            input_shape: vec![2, 1],
            input_data: vec![1., 2.],
            expected_output: vec![vec![1., 0.], vec![-1., 0.]],
        },
        TensorCheck {
            input_shape: vec![1, 2],
            input_data: vec![1., 2.],
            expected_output: vec![vec![1., 0.], vec![-1., 0.]],
        },
        // 矩阵型张量
        TensorCheck {
            input_shape: vec![2, 3],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![
                vec![1., 0., -1., -2., -3., -4.],
                vec![-1., 0., 1., 2., 3., 4.],
            ],
        },
        // 高阶张量
        TensorCheck {
            input_shape: vec![2, 3, 1],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![
                vec![1., 0., -1., -2., -3., -4.],
                vec![-1., 0., 1., 2., 3., 4.],
            ],
        },
        TensorCheck {
            input_shape: vec![2, 1, 3, 1],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![
                vec![1., 0., -1., -2., -3., -4.],
                vec![-1., 0., 1., 2., 3., 4.],
            ],
        },
    ];
    for test_case in test_cases {
        let tensor = Tensor::new(&test_case.input_data, &test_case.input_shape);
        // 1.纯数在前，张量在后
        let result = number - tensor.clone();
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
        let result = tensor - number;
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

#[test]
fn test_sub_scalars_among_various_shapes() {
    let number = 2.;
    let scalar_shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1]];

    // 测试不同形状标量间的减法组合
    for shape1 in scalar_shapes.iter() {
        let scalar1 = Tensor::new(&[number], shape1);

        for shape2 in scalar_shapes.iter() {
            let scalar2 = Tensor::new(&[1.0], shape2);

            if shape1 == shape2 {
                // 相同形状的标量张量相减应该成功
                let result = scalar1.clone() - scalar2.clone();
                let expected = Tensor::new(&[1.0], shape1);
                assert_eq!(result, expected);

                let result = scalar2 - scalar1.clone();
                let expected = Tensor::new(&[-1.0], shape1);
                assert_eq!(result, expected);
            } else {
                // 不同形状的标量张量相减应该失败
                let expected_msg = format!(
                    "形状不一致，故无法相减：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
                    shape1, shape2
                );
                assert_panic!(scalar1.clone() - scalar2.clone(), expected_msg);

                let expected_msg = format!(
                    "形状不一致，故无法相减：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
                    shape2, shape1
                );
                assert_panic!(scalar2 - scalar1.clone(), expected_msg);
            }
        }
    }
}
