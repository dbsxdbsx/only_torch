/*
 * @Author       : 老董
 * @Date         : 2023-08-17 17:24:24
 * @LastEditors  : 老董
 * @LastEditTime : 2023-08-22 12:27:21
 * @Description  : 张量的除法，实现了两个张量“逐元素”相除的运算，并返回一个新的张量。
 *                 除法运算支持以下情况：
 *                 1. 若两个张量的形状严格一致, 则相除后的张量形状不变；
 *                 2. 若其中一个张量为标量或纯数---统称为一阶张量。
 *                  2.1 若两个都是一阶张量，则相除后返回一个标量，其形状为[1];
 *                  2.2 若其中一个是二阶以上的张量，则相除后的形状为该张量的形状；
 *                 注意：这里的除法概念与线性代数中的矩阵除法类似，但适用于更高阶的张量。
 */

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
        // 高阶张量
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
            Array::from_shape_vec(IxDyn(&test_case.input_shape), test_case.expected_output[0].clone()).unwrap(),
            "纯数在前，张量在后：使用的纯数为：{:?}，张量为：{:?}",
            number,
            test_case.input_data
        );
        // 2.张量在前，纯数在后
        let result = tensor / number;
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(&test_case.input_shape), test_case.expected_output[1].clone()).unwrap(),
            "张量在前，纯数在后：使用的纯数为：{:?}，张量为：{:?}",
            number,
            test_case.input_data
        );
    }
}

#[test]
fn test_div_scalar_and_tensor() {
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
        // 高阶张量
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

    let scalar_shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1]];

    for test_case in test_cases {
        let tensor = Tensor::new(&test_case.input_data, &test_case.input_shape);
        for scalar_shape in scalar_shapes.iter() {
            let scalar_tensor = Tensor::new(&[number], scalar_shape);
            let correct_shape = if scalar_tensor.is_scalar() && tensor.is_scalar() {
                vec![1]
            } else {
                test_case.input_shape.clone()
            };
            // 1.标量在前，张量在后
            let result = scalar_tensor.clone() / tensor.clone();
            assert_eq!(
                result.data,
                Array::from_shape_vec(IxDyn(&correct_shape), test_case.expected_output[0].clone())
                    .unwrap(),
                "标量在前，张量在后：使用的标量为：{:?}，张量为：{:?}",
                &[number],
                test_case.input_data
            );
            // 2.张量在前，标量在后
            let result = tensor.clone() / scalar_tensor;
            assert_eq!(
                result.data,
                Array::from_shape_vec(IxDyn(&correct_shape), test_case.expected_output[1].clone())
                    .unwrap(),
                "张量在前，标量在后：使用的标量为：{:?}，张量为：{:?}",
                &[number],
                test_case.input_data
            );
        }
    }
}

#[test]
#[should_panic(
    expected = "形状不一致且两个张量（且没有一个是标量），故无法相除：第一个张量的形状为[3]，第二个张量的形状为[2]"
)]
fn test_div_operator_for_inconsistent_shape_1() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[2., 3.], &[2]);
    let _ = tensor1 / tensor2;
}
#[test]
#[should_panic(
    expected = "形状不一致且两个张量（且没有一个是标量），故无法相除：第一个张量的形状为[3]，第二个张量的形状为[3, 1]"
)]
fn test_div_operator_for_inconsistent_shape_2() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[4., 5., 6.], &[3, 1]);
    let _ = tensor1 / tensor2;
}

#[test]
#[should_panic(expected = "除数为零")]
fn test_div_zero_number() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let scalar = 0.;
    let _ = tensor1 / scalar;
}

#[test]
#[should_panic(expected = "作为除数的张量中存在为零元素")]
fn test_div_zero_scalar() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[0.], &[1, 1]);
    let _ = tensor1 / tensor2;
}

#[test]
#[should_panic(expected = "作为除数的张量中存在为零元素")]
fn test_div_matrix_and_zero_matrix() {
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    let tensor2 = Tensor::new(&[0., 2., 3., 0.], &[2, 2]);
    let _ = tensor1 / tensor2;
}
