use crate::assert_panic;
use crate::tensor::Tensor;
use crate::tensor::ops::reduce::DotSum;
use ndarray::{Array, IxDyn};

use super::TensorCheck;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`sum`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_sum_scalar() {
    let shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1]];
    let expected_tensor = Tensor::new(&[5.], &[1, 1]);
    for shape in shapes {
        let tensor = Tensor::new(&[5.], shape);
        assert_eq!(tensor.sum(), expected_tensor);
    }
}

#[test]
fn test_sum_vector() {
    let shapes: &[&[usize]] = &[&[4], &[1, 4], &[4, 1]];
    let expected_tensor = Tensor::new(&[10.], &[1, 1]);
    for shape in shapes {
        let tensor = Tensor::new(&[1., 2., 3., 4.], shape);
        assert_eq!(tensor.sum(), expected_tensor);
    }
}

#[test]
fn test_sum_matrix() {
    let shapes: &[&[usize]] = &[&[2, 2], &[1, 2, 2], &[2, 2, 1]];
    let expected_tensor = Tensor::new(&[10.], &[1, 1]);
    for shape in shapes {
        let tensor = Tensor::new(&[1., 2., 3., 4.], shape);
        assert_eq!(tensor.sum(), expected_tensor);
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
    let expected_tensor = Tensor::new(&[36.], &[1, 1]);
    for shape in shapes {
        let tensor = Tensor::new(&[1., 2., 3., 4., 5., 6., 7., 8.], shape);
        assert_eq!(tensor.sum(), expected_tensor);
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑`sum`↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`dot_sum`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_dot_sum_vectors_with_same_shape() {
    let shapes: &[&[usize]] = &[&[3], &[3, 1], &[1, 3]];
    for shape in shapes {
        let vector1 = Tensor::new(&[1., 2., 3.], shape);
        let vector2 = Tensor::new(&[4., 5., 6.], shape);
        let result = vector1.dot_sum(vector2);
        assert_eq!(result.get_data_number().unwrap(), 32.);
    }
}

#[test]
fn test_dot_sum_matrices_with_same_shape() {
    let shape = &[2, 2];
    let matrix1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let matrix2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = matrix1.dot_sum(matrix2);
    assert_eq!(result.get_data_number().unwrap(), 70.);
}

#[test]
fn test_dot_sum_high_dim_tensors_with_same_shape() {
    let shape = &[2, 1, 2];
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], shape);
    let tensor2 = Tensor::new(&[5., 6., 7., 8.], shape);
    let result = tensor1.dot_sum(tensor2);
    assert_eq!(result.get_data_number().unwrap(), 70.);
}

#[test]
fn test_dot_sum_number_and_tensor() {
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
            expected_output: vec![vec![6.]],
        },
        TensorCheck {
            input_shape: vec![2, 1],
            input_data: vec![1., 2.],
            expected_output: vec![vec![6.]],
        },
        TensorCheck {
            input_shape: vec![1, 2],
            input_data: vec![1., 2.],
            expected_output: vec![vec![6.]],
        },
        // 矩阵型张量
        TensorCheck {
            input_shape: vec![2, 3],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![vec![42.]],
        },
        // 高维张量
        TensorCheck {
            input_shape: vec![2, 3, 1],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![vec![42.]],
        },
        TensorCheck {
            input_shape: vec![2, 1, 3, 1],
            input_data: vec![1., 2., 3., 4., 5., 6.],
            expected_output: vec![vec![42.]],
        },
    ];

    for test_case in test_cases {
        let tensor = Tensor::new(&test_case.input_data, &test_case.input_shape);
        // 1.纯数在前，张量在后
        let result = number.dot_sum(tensor.clone());
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(&[1, 1]), test_case.expected_output[0].clone()).unwrap(),
            "纯数在前，张量在后：使用的纯数为：{:?}，张量为：{:?}",
            number,
            test_case.input_data
        );
        // 2.张量在前，纯数在后
        let result = tensor.dot_sum(number);
        assert_eq!(
            result.data,
            Array::from_shape_vec(IxDyn(&[1, 1]), test_case.expected_output[0].clone()).unwrap(),
            "张量在前，纯数在后：使用的纯数为：{:?}，张量为：{:?}",
            number,
            test_case.input_data
        );
    }
}

#[test]
fn test_dot_sum_scalars_among_various_shapes() {
    let number = 2.;
    let scalar_shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1]];

    // 测试不同形状标量间的点积和组合
    for shape1 in scalar_shapes.iter() {
        let scalar1 = Tensor::new(&[number], shape1);

        for shape2 in scalar_shapes.iter() {
            let scalar2 = Tensor::new(&[1.0], shape2);

            if shape1 == shape2 {
                // 相同形状的标量张量点积和应该成功
                let result = scalar1.clone().dot_sum(scalar2.clone());
                let expected = Tensor::new(&[2.0], &[1, 1]); // 2 * 1 = 2
                assert_eq!(result, expected);

                let result = scalar2.dot_sum(scalar1.clone());
                let expected = Tensor::new(&[2.0], &[1, 1]); // 1 * 2 = 2
                assert_eq!(result, expected);
            } else {
                // 不同形状的标量张量点积和应该失败
                assert_panic!(
                    scalar1.clone().dot_sum(scalar2.clone()),
                    format!(
                        "形状不一致，故无法点积和：第1个张量的形状为{:?}，第2个张量的形状为{:?}",
                        shape1, shape2
                    )
                );
                assert_panic!(
                    scalar2.dot_sum(scalar1.clone()),
                    format!(
                        "形状不一致，故无法点积和：第1个张量的形状为{:?}，第2个张量的形状为{:?}",
                        shape2, shape1
                    )
                );
            }
        }
    }
}

#[test]
fn test_dot_sum_with_or_without_ownership() {
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[4., 5., 6.], &[3]);
    let expected = Tensor::new(&[32.], &[1, 1]); // (1*4 + 2*5 + 3*6) = 32

    // f32 标量情况
    // 直接使用 f32
    let result = tensor1.dot_sum(2.0);
    assert_eq!(result, Tensor::new(&[12.], &[1, 1])); // (1*2 + 2*2 + 3*2) = 12

    // Tensor 情况
    // 1. 不带引用的张量
    let result = tensor1.clone().dot_sum(tensor2.clone());
    assert_eq!(result, expected);

    // 2. tensor1 带引用，tensor2 不带引用
    let result = (&tensor1).dot_sum(tensor2.clone());
    assert_eq!(result, expected);

    // 3. tensor1 不带引用，tensor2 带引用
    let result = tensor1.clone().dot_sum(&tensor2);
    assert_eq!(result, expected);

    // 4. 都带引用
    let result = (&tensor1).dot_sum(&tensor2);
    assert_eq!(result, expected);

    // 验证原始张量未被修改
    assert_eq!(tensor1, Tensor::new(&[1., 2., 3.], &[3]));
    assert_eq!(tensor2, Tensor::new(&[4., 5., 6.], &[3]));
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑`dot_sum`↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`variance`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

/// 全局方差：2x3 矩阵（numpy 对照值：2.9166667）
#[test]
fn test_variance_2d() {
    let t = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let var = t.variance();
    assert_eq!(var.shape(), &[1, 1]);
    assert!((var[[0, 0]] - 2.9166667).abs() < 1e-5);
}

/// 全局方差：单元素张量（方差应为 0）
#[test]
fn test_variance_single_element() {
    let t = Tensor::new(&[5.0], &[1]);
    let var = t.variance();
    assert!((var[[0, 0]] - 0.0).abs() < 1e-7);
}

/// 全局方差：1D 向量（numpy 对照值：5.0）
#[test]
fn test_variance_1d() {
    let t = Tensor::new(&[2., 4., 6., 8.], &[4]);
    let var = t.variance();
    assert!((var[[0, 0]] - 5.0).abs() < 1e-5);
}

/// 全局方差：3D 张量（numpy 对照值：5.25）
#[test]
fn test_variance_3d() {
    let t = Tensor::new(&[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 2, 2]);
    let var = t.variance();
    assert!((var[[0, 0]] - 5.25).abs() < 1e-5);
}

/// 沿轴方差：2x3 矩阵 axis=0（numpy 对照值：[2.25, 2.25, 2.25]）
#[test]
fn test_var_axis_keepdims_2d_axis0() {
    let t = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let var = t.var_axis_keepdims(0);
    assert_eq!(var.shape(), &[1, 3]);
    let data = var.data_as_slice();
    assert!((data[0] - 2.25).abs() < 1e-5);
    assert!((data[1] - 2.25).abs() < 1e-5);
    assert!((data[2] - 2.25).abs() < 1e-5);
}

/// 沿轴方差：2x3 矩阵 axis=1（numpy 对照值：[0.6666667, 0.6666667]）
#[test]
fn test_var_axis_keepdims_2d_axis1() {
    let t = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let var = t.var_axis_keepdims(1);
    assert_eq!(var.shape(), &[2, 1]);
    let data = var.data_as_slice();
    assert!((data[0] - 0.6666667).abs() < 1e-5);
    assert!((data[1] - 0.6666667).abs() < 1e-5);
}

/// 沿轴方差：2x2x2 张量各轴（numpy 对照值）
#[test]
fn test_var_axis_keepdims_3d() {
    let t = Tensor::new(&[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 2, 2]);

    // axis=0 → [1, 2, 2]，值全为 4.0
    let var0 = t.var_axis_keepdims(0);
    assert_eq!(var0.shape(), &[1, 2, 2]);
    for &v in var0.data_as_slice() {
        assert!((v - 4.0).abs() < 1e-5);
    }

    // axis=1 → [2, 1, 2]，值全为 1.0
    let var1 = t.var_axis_keepdims(1);
    assert_eq!(var1.shape(), &[2, 1, 2]);
    for &v in var1.data_as_slice() {
        assert!((v - 1.0).abs() < 1e-5);
    }

    // axis=2 → [2, 2, 1]，值全为 0.25
    let var2 = t.var_axis_keepdims(2);
    assert_eq!(var2.shape(), &[2, 2, 1]);
    for &v in var2.data_as_slice() {
        assert!((v - 0.25).abs() < 1e-5);
    }
}

/// 沿轴方差：axis 超出范围应 panic
#[test]
fn test_var_axis_keepdims_out_of_range() {
    let t = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    assert_panic!(t.var_axis_keepdims(2));
}

/// 沿轴方差：1D 向量 axis=0（numpy 对照值：5.0）
#[test]
fn test_var_axis_keepdims_1d() {
    let t = Tensor::new(&[2., 4., 6., 8.], &[4]);
    let var = t.var_axis_keepdims(0);
    assert_eq!(var.shape(), &[1]);
    assert!((var.data_as_slice()[0] - 5.0).abs() < 1e-5);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑`variance`↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
