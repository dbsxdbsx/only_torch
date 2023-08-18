use crate::tensor::Tensor;
use ndarray::Array;
use ndarray::IxDyn;

use super::TensorCheck;

#[test]
fn test_mul_vectors_with_same_shape() {
    let shapes: &[&[usize]] = &[&[3], &[3, 1], &[1, 3]];
    for shape in shapes {
        let vector1 = Tensor::new(&[1.0, 2.0, 3.0], shape);
        let vector2 = Tensor::new(&[4.0, 5.0, 6.0], shape);
        let result = vector1 * vector2;
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(shape), vec![4.0, 10.0, 18.0]).unwrap()
        );
    }
}

#[test]
fn test_mul_matrices_with_same_shape() {
    let shape = &[2, 2];
    let matrix1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], shape);
    let matrix2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], shape);
    let result = matrix1 * matrix2;
    assert_eq!(
        result.data,
        Array::from_shape_vec(IxDyn(shape), vec![5.0, 12.0, 21.0, 32.0]).unwrap()
    );
}

#[test]
fn test_mul_high_order_tensors_with_same_shape() {
    let shape = &[2, 1, 2];
    let tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], shape);
    let tensor2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], shape);
    let result = tensor1 * tensor2;
    assert_eq!(
        result.data,
        Array::from_shape_vec(IxDyn(shape), vec![5.0, 12.0, 21.0, 32.0]).unwrap()
    );
}

#[test]
fn test_mul_number_and_tensor() {
    let number = 2.0;
    let test_cases = vec![
        // 标量型张量
        TensorCheck {
            shape: vec![],
            data: vec![1.0],
            expected: vec![vec![2.0]],
        },
        TensorCheck {
            shape: vec![1],
            data: vec![1.0],
            expected: vec![vec![2.0]],
        },
        TensorCheck {
            shape: vec![1, 1],
            data: vec![1.0],
            expected: vec![vec![2.0]],
        },
        // 向量型张量
        TensorCheck {
            shape: vec![2],
            data: vec![1.0, 2.0],
            expected: vec![vec![2.0, 4.0]],
        },
        TensorCheck {
            shape: vec![2, 1],
            data: vec![1.0, 2.0],
            expected: vec![vec![2.0, 4.0]],
        },
        TensorCheck {
            shape: vec![1, 2],
            data: vec![1.0, 2.0],
            expected: vec![vec![2.0, 4.0]],
        },
        // 矩阵型张量
        TensorCheck {
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            expected: vec![vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]],
        },
        // 高阶张量
        TensorCheck {
            shape: vec![2, 3, 1],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            expected: vec![vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]],
        },
        TensorCheck {
            shape: vec![2, 1, 3, 1],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            expected: vec![vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]],
        },
    ];

    for test_case in test_cases {
        let tensor = Tensor::new(&test_case.data, &test_case.shape);
        // 1.标量在前，张量在后
        let result = number * tensor.clone();
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(&test_case.shape), test_case.expected[0].clone()).unwrap(),
            "标量在前，张量在后：使用的标量为：{:?}，张量为：{:?}",
            number,
            test_case.data
        );
        // 2.张量在前，标量在后
        let result = tensor * number;
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(&test_case.shape), test_case.expected[0].clone()).unwrap(),
            "张量在前，标量在后：使用的标量为：{:?}，张量为：{:?}",
            number,
            test_case.data
        );
    }
}

#[test]
fn test_mul_scalar_and_tensor() {
    let number = 2.0;
    let test_cases = vec![
        // 标量型张量
        TensorCheck {
            shape: vec![],
            data: vec![1.0],
            expected: vec![vec![2.0]],
        },
        TensorCheck {
            shape: vec![1],
            data: vec![1.0],
            expected: vec![vec![2.0]],
        },
        TensorCheck {
            shape: vec![1, 1],
            data: vec![1.0],
            expected: vec![vec![2.0]],
        },
        // 向量型张量
        TensorCheck {
            shape: vec![2],
            data: vec![1.0, 2.0],
            expected: vec![vec![2.0, 4.0]],
        },
        TensorCheck {
            shape: vec![2, 1],
            data: vec![1.0, 2.0],
            expected: vec![vec![2.0, 4.0]],
        },
        TensorCheck {
            shape: vec![1, 2],
            data: vec![1.0, 2.0],
            expected: vec![vec![2.0, 4.0]],
        },
        // 矩阵型张量
        TensorCheck {
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            expected: vec![vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]],
        },
        // 高阶张量
        TensorCheck {
            shape: vec![2, 3, 1],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            expected: vec![vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]],
        },
        TensorCheck {
            shape: vec![2, 1, 3, 1],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            expected: vec![vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]],
        },
    ];
    let scalar_shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1]];

    for test_case in test_cases {
        let tensor = Tensor::new(&test_case.data, &test_case.shape);
        for scalar_shape in scalar_shapes.iter() {
            let scalar_tensor = Tensor::new(&[number], scalar_shape);
            let correct_shape = if scalar_tensor.is_scalar() && tensor.is_scalar() {
                vec![1]
            } else {
                test_case.shape.clone()
            };
            // 1.标量在前，张量在后
            let result = scalar_tensor.clone() * tensor.clone();
            assert_eq!(
                result.data,
                Array::from_shape_vec(IxDyn(&correct_shape), test_case.expected[0].clone())
                    .unwrap(),
                "标量在前，张量在后：使用的标量为：{:?}，张量为：{:?}",
                &[number],
                test_case.data
            );
            // 2.张量在前，标量在后
            let result = tensor.clone() * scalar_tensor;
            assert_eq!(
                result.data,
                Array::from_shape_vec(IxDyn(&correct_shape), test_case.expected[0].clone())
                    .unwrap(),
                "张量在前，标量在后：使用的标量为：{:?}，张量为：{:?}",
                &[number],
                test_case.data
            );
        }
    }
}

#[test]
#[should_panic(
    expected = "形状不一致且两个张量没有一个是标量，故无法相乘：第一个张量的形状为[3]，第二个张量的形状为[2]"
)]
fn test_mul_operator_for_inconsistent_shape_1() {
    let tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[2.0, 3.0], &[2]);
    let _ = tensor1 * tensor2;
}
#[test]
#[should_panic(
    expected = "形状不一致且两个张量没有一个是标量，故无法相乘：第一个张量的形状为[3]，第二个张量的形状为[3, 1]"
)]
fn test_mul_operator_for_inconsistent_shape_2() {
    let tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[4.0, 5.0, 6.0], &[3, 1]);
    let _ = tensor1 * tensor2;
}
