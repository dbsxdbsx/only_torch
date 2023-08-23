use crate::tensor::ops::others::DotSum;
use crate::tensor::Tensor;
use ndarray::Array;
use ndarray::IxDyn;

use super::TensorCheck;

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`sum`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#[test]
fn test_sum_scalar() {
    let shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1]];
    let expected_sum = Tensor::from(5.);
    for shape in shapes {
        let tensor = Tensor::new(&[5.], shape);
        assert_eq!(tensor.sum(), expected_sum);
    }
}

#[test]
fn test_sum_vector() {
    let shapes: &[&[usize]] = &[&[4], &[1, 4], &[4, 1]];
    let expected_sum = Tensor::from(10.);
    for shape in shapes {
        let tensor = Tensor::new(&[1., 2., 3., 4.], shape);
        assert_eq!(tensor.sum(), expected_sum);
    }
}

#[test]
fn test_sum_matrix() {
    let shapes: &[&[usize]] = &[&[2, 2], &[1, 2, 2], &[2, 2, 1]];
    let expected_sum = Tensor::from(10.);
    for shape in shapes {
        let tensor = Tensor::new(&[1., 2., 3., 4.], shape);
        assert_eq!(tensor.sum(), expected_sum);
    }
}

#[test]
fn test_sum_high_dim_tensor() {
    let shapes: &[&[usize]] = &[
        &[2, 2, 2],
        &[1, 2, 2, 2],
        &[2, 1, 2, 2],
        &[2, 2, 1, 2],
        &[2, 2, 2, 1],
    ];
    let expected_sum = Tensor::from(36.);
    for shape in shapes {
        let tensor = Tensor::new(&[1., 2., 3., 4., 5., 6., 7., 8.], shape);
        assert_eq!(tensor.sum(), expected_sum);
    }
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑`sum`↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`dot_sum`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#[test]
fn test_dot_sum_vectors_with_same_shape() {
    let shapes: &[&[usize]] = &[&[3], &[3, 1], &[1, 3]];
    for shape in shapes {
        let vector1 = Tensor::new(&[1., 2., 3.], shape);
        let vector2 = Tensor::new(&[4., 5., 6.], shape);
        let result = vector1.dot_sum(vector2);
        assert_eq!(result.number().unwrap(), 32.);
    }
}

#[test]
fn test_dot_sum_matrices_with_same_shape() {
    let shape = &[2, 2];
    let matrix1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let matrix2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = matrix1.dot_sum(matrix2);
    assert_eq!(result.number().unwrap(), 70.);
}

#[test]
fn test_dot_sum_high_dim_tensors_with_same_shape() {
    let shape = &[2, 1, 2];
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let tensor2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = tensor1.dot_sum(tensor2);
    assert_eq!(result.number().unwrap(), 70.);
}

#[test]
fn test_dot_sum_number_and_tensor() {
    let number = 2.;
    let test_cases = vec![
        // 标量型张量
        TensorCheck {
            shape: vec![],
            data: vec![1.],
            expected: vec![vec![2.]],
        },
        TensorCheck {
            shape: vec![1],
            data: vec![1.],
            expected: vec![vec![2.]],
        },
        TensorCheck {
            shape: vec![1, 1],
            data: vec![1.],
            expected: vec![vec![2.]],
        },
        // 向量型张量
        TensorCheck {
            shape: vec![2],
            data: vec![1., 2.],
            expected: vec![vec![6.]],
        },
        TensorCheck {
            shape: vec![2, 1],
            data: vec![1., 2.],
            expected: vec![vec![6.]],
        },
        TensorCheck {
            shape: vec![1, 2],
            data: vec![1., 2.],
            expected: vec![vec![6.]],
        },
        // 矩阵型张量
        TensorCheck {
            shape: vec![2, 3],
            data: vec![1., 2., 3., 4., 5., 6.],
            expected: vec![vec![42.]],
        },
        // 高阶张量
        TensorCheck {
            shape: vec![2, 3, 1],
            data: vec![1., 2., 3., 4., 5., 6.],
            expected: vec![vec![42.]],
        },
        TensorCheck {
            shape: vec![2, 1, 3, 1],
            data: vec![1., 2., 3., 4., 5., 6.],
            expected: vec![vec![42.]],
        },
    ];

    for test_case in test_cases {
        let tensor = Tensor::new(&test_case.data, &test_case.shape);
        // 1.标量在前，张量在后
        let result = number.dot_sum(tensor.clone());
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(&[1]), test_case.expected[0].clone()).unwrap(),
            "标量在前，张量在后：使用的标量为：{:?}，张量为：{:?}",
            number,
            test_case.data
        );
        // 2.张量在前，标量在后
        let result = tensor.dot_sum(number);
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(&[1]), test_case.expected[0].clone()).unwrap(),
            "张量在前，标量在后：使用的标量为：{:?}，张量为：{:?}",
            number,
            test_case.data
        );
    }
}

#[test]
fn test_dot_sum_scalar_and_tensor() {
    let number = 2.;
    let test_cases = vec![
        // 标量型张量
        TensorCheck {
            shape: vec![],
            data: vec![1.],
            expected: vec![vec![2.]],
        },
        TensorCheck {
            shape: vec![1],
            data: vec![1.],
            expected: vec![vec![2.]],
        },
        TensorCheck {
            shape: vec![1, 1],
            data: vec![1.],
            expected: vec![vec![2.]],
        },
        // 向量型张量
        TensorCheck {
            shape: vec![2],
            data: vec![1., 2.],
            expected: vec![vec![6.]],
        },
        TensorCheck {
            shape: vec![2, 1],
            data: vec![1., 2.],
            expected: vec![vec![6.]],
        },
        TensorCheck {
            shape: vec![1, 2],
            data: vec![1., 2.],
            expected: vec![vec![6.]],
        },
        // 矩阵型张量
        TensorCheck {
            shape: vec![2, 3],
            data: vec![1., 2., 3., 4., 5., 6.],
            expected: vec![vec![42.]],
        },
        // 高阶张量
        TensorCheck {
            shape: vec![2, 3, 1],
            data: vec![1., 2., 3., 4., 5., 6.],
            expected: vec![vec![42.]],
        },
        TensorCheck {
            shape: vec![2, 1, 3, 1],
            data: vec![1., 2., 3., 4., 5., 6.],
            expected: vec![vec![42.]],
        },
    ];
    let scalar_shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1]];

    for test_case in test_cases {
        let tensor = Tensor::new(&test_case.data, &test_case.shape);
        for scalar_shape in scalar_shapes.iter() {
            let scalar_tensor = Tensor::new(&[number], scalar_shape);
            // 1.标量在前，张量在后
            let result = scalar_tensor.clone().dot_sum(tensor.clone());
            assert_eq!(
                result.data,
                Array::from_shape_vec(IxDyn(&[1]), test_case.expected[0].clone()).unwrap(),
                "标量在前，张量在后：使用的标量为：{:?}，张量为：{:?}",
                &[number],
                test_case.data
            );
            // 2.张量在前，标量在后
            let result = tensor.clone().dot_sum(scalar_tensor);
            assert_eq!(
                result.data,
                Array::from_shape_vec(IxDyn(&[1]), test_case.expected[0].clone()).unwrap(),
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
fn test_dot_sum_operator_for_inconsistent_shape_1() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[2., 3.], &[2]);
    let _ = tensor1 * tensor2;
}

#[test]
#[should_panic(
    expected = "形状不一致且两个张量没有一个是标量，故无法相乘：第一个张量的形状为[3]，第二个张量的形状为[3, 1]"
)]
fn test_dot_sum_operator_for_inconsistent_shape_2() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[4., 5., 6.], &[3, 1]);
    let _ = tensor1 * tensor2;
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑`dot_sum`↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`order`和`shuffle`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#[test]
fn test_order() {
    let tensor1 = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let tensor2 = Tensor::new(&[3., 4., 1., 2., 5., 6.], &[2, 3]);
    let ordered_tensor = tensor2.order();
    assert_ne!(tensor1, tensor2);
    assert_eq!(tensor1, ordered_tensor);
}

#[test]
fn test_shuffle() {
    let tensor = Tensor::new(&[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 4]);
    let shuffled_tensor = tensor.shuffle();
    shuffled_tensor.print();
    assert_eq!(tensor.shape(), shuffled_tensor.shape());
    assert_ne!(tensor.data, shuffled_tensor.data);
    let ordered_tensor = shuffled_tensor.order();
    assert_eq!(tensor, ordered_tensor);
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑`order`和`shuffle`↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
