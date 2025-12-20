use crate::assert_panic;
use crate::errors::TensorError;
use crate::tensor::Tensor;
use crate::tensor::ops::others::DotSum;
use ndarray::IxDyn;
use ndarray::{Array, Axis};

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
        // 高阶张量
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
                        "形状不一致，故无法点积和：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
                        shape1, shape2
                    )
                );
                assert_panic!(
                    scalar2.dot_sum(scalar1.clone()),
                    format!(
                        "形状不一致，故无法点积和：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
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

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓order↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_order() {
    // 1. 2维张量
    let tensor1 = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let tensor2 = Tensor::new(&[3., 4., 1., 2., 5., 6.], &[2, 3]);
    let ordered_tensor = tensor2.order();
    assert_eq!(tensor1, ordered_tensor);

    // 2. 3维张量
    let tensor1 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 2, 3],
    );
    let tensor2 = Tensor::new(
        &[7., 8., 9., 10., 11., 12., 3., 4., 1., 2., 5., 6.],
        &[2, 2, 3],
    );
    let ordered_tensor = tensor2.order();
    assert_eq!(tensor1, ordered_tensor);
}

#[test]
fn test_order_mut() {
    // 1. 2维张量
    let tensor1 = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let mut tensor2 = Tensor::new(&[3., 4., 1., 2., 5., 6.], &[2, 3]);
    tensor2.order_mut();
    assert_eq!(tensor1, tensor2);

    // 2. 3维张量
    let tensor1 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 2, 3],
    );
    let mut tensor2 = Tensor::new(
        &[7., 8., 9., 10., 11., 12., 3., 4., 1., 2., 5., 6.],
        &[2, 2, 3],
    );
    tensor2.order_mut();
    assert_eq!(tensor1, tensor2);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑order↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓shuffle↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_shuffle() {
    let data = &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0, 33.0, 34.0, 35.0, 36.0,
    ];
    let shape = &[6, 6];
    let tensor = Tensor::new(data, shape);

    // 1. 仅打乱第一个维度（打乱后的形状仍一致，但数据不一致）
    let shuffled_tensor_row = tensor.shuffle(Some(0));
    assert_eq!(tensor.shape(), shuffled_tensor_row.shape());
    assert_ne!(tensor.data, shuffled_tensor_row.data);
    // 1.1 虽然打乱后整体数据是不一致的，但是该张量每行的数据总是能在另一个张量中的某行找到完全一致的数据
    for row in shuffled_tensor_row.data.axis_iter(Axis(0)) {
        assert!(tensor.data.axis_iter(Axis(0)).any(|r| r == row));
    }

    // 2. 仅打乱第二个维度（打乱后的形状仍一致，但数据不一致）
    let shuffled_tensor_col = tensor.shuffle(Some(1));
    assert_eq!(tensor.shape(), shuffled_tensor_col.shape());
    assert_ne!(tensor.data, shuffled_tensor_col.data);
    // 2.1 虽然打乱后整体数据是不一致的，但是该张量每列的数据总是能在另一个张量中的某列找到完全一致的数据
    for col in shuffled_tensor_col.data.axis_iter(Axis(1)) {
        assert!(tensor.data.axis_iter(Axis(1)).any(|c| c == col));
    }

    // 3. 全局打乱（打乱后的形状仍一致，但数据不一致）
    let tensor_shuffle = tensor.shuffle(None);
    assert_eq!(tensor.shape(), tensor_shuffle.shape());
    assert_ne!(tensor.data, tensor_shuffle.data);
    // 3.1 确保没有一行或一列和原来一样的
    assert!(
        tensor_shuffle
            .data
            .axis_iter(Axis(0))
            .all(|row| { tensor.data.axis_iter(Axis(0)).all(|r| r != row) })
    );
    assert!(
        tensor_shuffle
            .data
            .axis_iter(Axis(1))
            .all(|col| { tensor.data.axis_iter(Axis(1)).all(|r| r != col) })
    );
    // 3.2 重新排序后则应完全一致
    let ordered_tensor = tensor_shuffle.order();
    assert_eq!(tensor, ordered_tensor);
}

#[test]
fn test_shuffle_mut() {
    let data = &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0, 33.0, 34.0, 35.0, 36.0,
    ];
    let shape = &[6, 6];
    let tensor = Tensor::new(data, shape);

    // 1. 仅打乱第一个维度（打乱后的形状仍一致，但数据不一致）
    let mut tensor_shuffle_row = Tensor::new(data, shape);
    tensor_shuffle_row.shuffle_mut(Some(0));
    assert_eq!(tensor.shape(), tensor_shuffle_row.shape());
    assert_ne!(tensor.data, tensor_shuffle_row.data);
    // 1.1 虽然打乱后整体数据是不一致的，但是该张量每行的数据总是能在另一个张量中的某行找到完全一致的数据
    for row in tensor_shuffle_row.data.axis_iter(Axis(0)) {
        assert!(tensor.data.axis_iter(Axis(0)).any(|r| r == row));
    }

    // 2. 仅打乱第二个维度（打乱后的形状仍一致，但数据不一致）
    let mut tensor_shuffle_col = Tensor::new(data, shape);
    tensor_shuffle_col.shuffle_mut(Some(1));
    assert_eq!(tensor.shape(), tensor_shuffle_col.shape());
    assert_ne!(tensor.data, tensor_shuffle_col.data);
    // 2.1 虽然打乱后整体数据是不一致的，但是该张量每列的数据总是能在另一个张量中的某行找到完全一致的数据
    for row in tensor_shuffle_col.data.axis_iter(Axis(1)) {
        assert!(tensor.data.axis_iter(Axis(1)).any(|r| r == row));
    }

    // 3. 全局打乱（打乱后的形状仍一致，但数据不一致）
    let mut tensor_shuffle = Tensor::new(data, shape);
    tensor_shuffle.shuffle_mut(None);
    assert_eq!(tensor.shape(), tensor_shuffle.shape());
    assert_ne!(tensor.data, tensor_shuffle.data);
    // 3.1 确保没有一行或一列和原来一样的
    assert!(
        tensor_shuffle
            .data
            .axis_iter(Axis(0))
            .all(|row| { tensor.data.axis_iter(Axis(0)).all(|r| r != row) })
    );
    assert!(
        tensor_shuffle
            .data
            .axis_iter(Axis(1))
            .all(|col| { tensor.data.axis_iter(Axis(1)).all(|r| r != col) })
    );
    let ordered_tensor = tensor_shuffle.order();
    // 3.2 重新排序后则应完全一致
    assert_eq!(tensor, ordered_tensor);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑shuffle↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓reshape↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_reshape() {
    // 1.标量reshape
    let data = &[5.];
    let shape = &[];
    let tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[1, 1, 1];
    assert_eq!(tensor.reshape(new_shape).shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2];
    assert_panic!(tensor.reshape(incompatible_shape));

    // 2.向量reshape
    let data = &[1., 2., 3., 4.];
    let shape = &[4, 1];
    let tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[2, 2];
    assert_eq!(tensor.reshape(new_shape).shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2, 3];
    assert_panic!(tensor.reshape(incompatible_shape));

    // 3.矩阵reshape
    let data = &[1., 2., 3., 4., 5., 6.];
    let shape = &[2, 3];
    let tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[3, 2];
    assert_eq!(tensor.reshape(new_shape).shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2, 2];
    assert_panic!(tensor.reshape(incompatible_shape));
}

#[test]
fn test_reshape_mut() {
    // 1.标量reshape
    let data = &[5.];
    let shape = &[];
    let mut tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[1, 1, 1];
    tensor.reshape_mut(new_shape);
    assert_eq!(tensor.shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2];
    assert_panic!(tensor.reshape_mut(incompatible_shape));

    // 2.向量reshape
    let data = &[1., 2., 3., 4.];
    let shape = &[4, 1];
    let mut tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[2, 2];
    tensor.reshape_mut(new_shape);
    assert_eq!(tensor.shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2, 3];
    assert_panic!(tensor.reshape_mut(incompatible_shape));

    // 3.矩阵reshape
    let data = &[1., 2., 3., 4., 5., 6.];
    let shape = &[2, 3];
    let mut tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[3, 2];
    tensor.reshape_mut(new_shape);
    assert_eq!(tensor.shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2, 2];
    assert_panic!(tensor.reshape_mut(incompatible_shape));
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑reshape↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓stack(concat)↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_stack_without_new_dim() {
    // 1.空张量的堆叠
    assert_panic!(Tensor::stack(&[], false), TensorError::EmptyList);
    // 2.标量的堆叠
    let t1 = Tensor::new(&[5.0], &[]);
    let t2 = Tensor::new(&[6.0], &[1]);
    let t3 = Tensor::new(&[7.0], &[1, 1]);
    let stacked = Tensor::stack(&[&t1, &t2, &t3], false);
    assert_eq!(stacked, Tensor::new(&[5.0, 6.0, 7.0], &[3]));

    // 3.向量的堆叠
    let t1 = Tensor::new(&[1.0, 2.0], &[2]);
    let t2 = Tensor::new(&[3.0, 4.0], &[2]);
    let stacked = Tensor::stack(&[&t1, &t2], false);
    assert_eq!(stacked, Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]));
    // (不添加新维度的情况下，张量的第一个维度可以不同)
    let t1 = Tensor::new(&[1., 2.], &[2]);
    let t2 = Tensor::new(&[6.0, 7.0, 8.0], &[3]);
    let stacked = Tensor::stack(&[&t1, &t2], false);
    assert_eq!(stacked, Tensor::new(&[1.0, 2.0, 6.0, 7.0, 8.0], &[5]));
    // (不添加新维度的情况下，除张量的第一个维度不同外，其他维度若不同则会报错)
    let t1 = Tensor::new(&[1., 2.], &[2]);
    let t2 = Tensor::new(&[6.0, 7.0, 8.0], &[3, 1]);
    assert_panic!(
        Tensor::stack(&[&t1, &t2], false),
        TensorError::InconsitentShape
    );

    // 4.矩阵的堆叠
    let t1 = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let t2 = Tensor::new(&[7., 8., 9., 10., 11., 12.], &[2, 3]);
    let stacked = Tensor::stack(&[&t1, &t2], false);
    assert_eq!(
        stacked,
        Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5., 6., 7., 8., 9., 10., 11., 12.],
            &[4, 3]
        )
    );
    // (不添加新维度的情况下，张量的第一个维度可以不同)
    let t1 = Tensor::new(&[1., 2., 3.0, 4.0, 5., 6.], &[2, 3]);
    let t2 = Tensor::new(&[7.0, 8.0, 9.0], &[1, 3]);
    let stacked = Tensor::stack(&[&t1, &t2], false);
    assert_eq!(
        stacked,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
    );
    // (不添加新维度的情况下，除张量的第一个维度不同外，其他维度若不同则会报错)
    let t1 = Tensor::new(&[1., 2., 3.0, 4.0, 5., 6.], &[2, 3]);
    let t2 = Tensor::new(&[7.0, 8.0, 9.0, 10.0], &[1, 4]);
    assert_panic!(
        Tensor::stack(&[&t1, &t2], false),
        TensorError::InconsitentShape
    );

    // 5.高维张量的堆叠
    let t1 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 3, 2, 1],
    );
    let t2 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 3, 2, 1],
    );
    let stacked = Tensor::stack(&[&t1, &t2], false);
    assert_eq!(
        stacked,
        Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, 5., 6., 7., 8., 9., 10., 11., 12., 1., 2., 3., 4., 5., 6., 7.,
                8., 9., 10., 11., 12.
            ],
            &[4, 3, 2, 1]
        )
    );
    // (不添加新维度的情况下，张量的第一个维度可以不同)
    let t1 = Tensor::new(&[1., 2., 3., 4.], &[2, 1, 2, 1]);
    let t2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 1, 2, 1]);
    let stacked = Tensor::stack(&[&t1, &t2], false);
    assert_eq!(
        stacked,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,], &[4, 1, 2, 1])
    );
    // (不添加新维度的情况下，除张量的第一个维度不同外，其他维度若不同则会报错)
    let t1 = Tensor::new(&[1., 2., 3., 4.], &[2, 1, 2, 1]);
    let t2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 1, 2]);
    assert_panic!(
        Tensor::stack(&[&t1, &t2], false),
        TensorError::InconsitentShape
    );
}

#[test]
fn test_stack_with_new_dim() {
    // 1. 空张量的堆叠
    assert_panic!(Tensor::stack(&[], true), TensorError::EmptyList);

    // 2. 标量的堆叠
    let t1 = Tensor::new(&[5.0], &[]);
    let t2 = Tensor::new(&[6.0], &[1]);
    let t3 = Tensor::new(&[7.0], &[1, 1]);
    let stacked = Tensor::stack(&[&t1, &t2, &t3], true);
    assert_eq!(stacked, Tensor::new(&[5.0, 6.0, 7.0], &[3, 1]));

    // 3. 向量的堆叠
    let t1 = Tensor::new(&[1.0, 2.0], &[2]);
    let t2 = Tensor::new(&[3.0, 4.0], &[2]);
    let stacked = Tensor::stack(&[&t1, &t2], true);
    assert_eq!(stacked, Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
    // (添加新维度的情况下，形状必须严格一致，若不同则会报错)
    let t1 = Tensor::new(&[1., 2.], &[2, 1]);
    let t2 = Tensor::new(&[5.0, 6.0, 7.0], &[3, 1]);
    assert_panic!(
        Tensor::stack(&[&t1, &t2], true),
        TensorError::InconsitentShape
    );

    // 4. 矩阵的堆叠
    let t1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let t2 = Tensor::new(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3]);
    let stacked = Tensor::stack(&[&t1, &t2], true);
    assert_eq!(
        stacked,
        Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ],
            &[2, 2, 3]
        )
    );
    // (添加新维度的情况下，形状必须严格一致，若不同则会报错)
    let t1 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    let t2 = Tensor::new(&[5.0, 6.0], &[2, 1]);
    assert_panic!(
        Tensor::stack(&[&t1, &t2], true),
        TensorError::InconsitentShape
    );

    // 5.高维张量的堆叠
    let t1 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 3, 2],
    );
    let t2 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 3, 2],
    );
    let stacked = Tensor::stack(&[&t1, &t2], true);
    assert_eq!(
        stacked,
        Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, //
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ],
            &[2, 2, 3, 2]
        )
    );
    // (添加新维度的情况下，形状必须严格一致，若不同则会报错)
    let t1 = Tensor::new(&[1., 2., 3., 4.], &[2, 2, 1, 1]);
    let t2 = Tensor::new(&[1., 2., 3., 4.], &[2, 2, 1]);
    assert_panic!(
        Tensor::stack(&[&t1, &t2], true),
        TensorError::InconsitentShape
    );
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑stack(concat)↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓(un)squeeze↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_squeeze() {
    // 测试标量
    let data = &[1.];
    let shape = &[];
    let squeezed_tensor = Tensor::new(data, shape).squeeze();
    assert_eq!(squeezed_tensor.shape(), &[]);

    let data = &[1.];
    let shape = &[1];
    let squeezed_tensor = Tensor::new(data, shape).squeeze();
    assert_eq!(squeezed_tensor.shape(), &[]);

    // 测试向量
    let data = &[1., 2., 3., 4.];
    let shape = &[4];
    let squeezed_tensor = Tensor::new(data, shape).squeeze();
    assert_eq!(squeezed_tensor.shape(), &[4]);

    // 测试矩阵
    let data = &[1., 2., 3., 4.];
    let shapes: &[&[usize]] = &[&[4], &[1, 4], &[4, 1]];
    for shape in shapes {
        let squeezed_tensor = Tensor::new(data, shape).squeeze();
        assert_eq!(squeezed_tensor.shape(), &[4]);
    }

    // 测试高维张量
    let data = &[1., 2., 3., 4.];
    let shape = &[1, 1, 1, 4];
    let squeezed_tensor = Tensor::new(data, shape).squeeze();
    assert_eq!(squeezed_tensor.shape(), &[4]);

    let data = &[1., 2., 3., 4., 5., 6.];
    let shape = &[1, 2, 1, 3];
    let squeezed_tensor = Tensor::new(data, shape).squeeze();
    assert_eq!(squeezed_tensor.shape(), &[2, 3]);
}
#[test]
fn test_squeeze_mut() {
    // 测试标量
    let data = &[1.];
    let shape = &[];
    let mut tensor = Tensor::new(data, shape);
    tensor.squeeze_mut();
    assert_eq!(tensor.shape(), &[]);

    let data = &[1.];
    let shape = &[1];
    let mut tensor = Tensor::new(data, shape);
    tensor.squeeze_mut();
    assert_eq!(tensor.shape(), &[]);

    // 测试向量
    let data = &[1., 2., 3., 4.];
    let shape = &[4];
    let mut tensor = Tensor::new(data, shape);
    tensor.squeeze_mut();
    assert_eq!(tensor.shape(), &[4]);

    // 测试矩阵
    let data = &[1., 2., 3., 4.];
    let shapes: &[&[usize]] = &[&[4], &[1, 4], &[4, 1]];
    for shape in shapes {
        let mut tensor = Tensor::new(data, shape);
        tensor.squeeze_mut();
        assert_eq!(tensor.shape(), &[4]);
    }

    // 测试高维张量
    let data = &[1., 2., 3., 4.];
    let shape = &[1, 1, 1, 4];
    let mut tensor = Tensor::new(data, shape);
    tensor.squeeze_mut();
    assert_eq!(tensor.shape(), &[4]);

    let data = &[1., 2., 3., 4., 5., 6.];
    let shape = &[1, 2, 1, 3];
    let mut tensor = Tensor::new(data, shape);
    tensor.squeeze_mut();
    assert_eq!(tensor.shape(), &[2, 3]);
}

#[test]
fn test_unsqueeze() {
    // 测试在最前面增加一个维度
    let data = &[1., 2., 3., 4.];
    let shape = &[4];
    let unsqueezed_tensor = Tensor::new(data, shape).unsqueeze(0);
    assert_eq!(unsqueezed_tensor.shape(), &[1, 4]);
    // 测试在最后面增加一个维度
    let unsqueezed_tensor = Tensor::new(data, shape).unsqueeze(-1);
    assert_eq!(unsqueezed_tensor.shape(), &[4, 1]);
    // 测试在中间增加一个维度
    let shape = &[2, 2];
    let unsqueezed_tensor = Tensor::new(data, shape).unsqueeze(1);
    assert_eq!(unsqueezed_tensor.shape(), &[2, 1, 2]);
    // 测试负索引
    let unsqueezed_tensor = Tensor::new(data, shape).unsqueeze(-2);
    assert_eq!(unsqueezed_tensor.shape(), &[2, 1, 2]);
    let unsqueezed_tensor = Tensor::new(data, shape).unsqueeze(-3);
    assert_eq!(unsqueezed_tensor.shape(), &[1, 2, 2]);
    // 测试超出范围的索引
    assert_panic!(Tensor::new(data, shape).unsqueeze(3));
    assert_panic!(Tensor::new(data, shape).unsqueeze(-4));
}
#[test]
fn test_unsqueeze_mut() {
    // 测试在最前面增加一个维度
    let mut tensor = Tensor::new(&[1., 2., 3., 4.], &[4]);
    tensor.unsqueeze_mut(0);
    assert_eq!(tensor.shape(), &[1, 4]);
    // 测试在最后面增加一个维度
    let mut tensor = Tensor::new(&[1., 2., 3., 4.], &[4]);
    tensor.unsqueeze_mut(-1);
    assert_eq!(tensor.shape(), &[4, 1]);
    // 测试在中间增加一个维度
    let mut tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    tensor.unsqueeze_mut(1);
    assert_eq!(tensor.shape(), &[2, 1, 2]);
    // 测试负索引
    let mut tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    tensor.unsqueeze_mut(-2);
    assert_eq!(tensor.shape(), &[2, 1, 2]);
    let mut tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    tensor.unsqueeze_mut(-3);
    assert_eq!(tensor.shape(), &[1, 2, 2]);
    // 测试超出范围的索引
    assert_panic!(Tensor::new(&[1., 2., 3., 4.], &[2, 2]).unsqueeze_mut(3));
    assert_panic!(Tensor::new(&[1., 2., 3., 4.], &[2, 2]).unsqueeze_mut(-4));
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑(un)squeeze↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓permute↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_permute() {
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // 应该成功的情况
    let permuted_tensor = tensor.permute(&[1, 0]);
    let expected_tensor = Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2]);
    assert_eq!(permuted_tensor, expected_tensor);
    // 应该失败的情况
    assert_panic!(tensor.permute(&[]), TensorError::PermuteNeedAtLeast2Dims);
    assert_panic!(tensor.permute(&[1]), TensorError::PermuteNeedAtLeast2Dims);
    assert_panic!(
        tensor.permute(&[1, 1]),
        TensorError::PermuteNeedUniqueAndInRange
    );
}

#[test]
fn test_permute_mut() {
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // 应该成功的情况
    tensor.permute_mut(&[1, 0]);
    let expected_tensor = Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2]);
    assert_eq!(tensor, expected_tensor);
    // 应该失败的情况
    assert_panic!(
        tensor.permute_mut(&[]),
        TensorError::PermuteNeedAtLeast2Dims
    );
    assert_panic!(
        tensor.permute_mut(&[1]),
        TensorError::PermuteNeedAtLeast2Dims
    );
    assert_panic!(
        tensor.permute_mut(&[1, 1]),
        TensorError::PermuteNeedUniqueAndInRange
    );
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑permute↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓transpose↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_transpose() {
    // 测试标量
    let tensor = Tensor::new(&[1.0], &[]);
    let transposed = tensor.transpose();
    assert_eq!(transposed.shape(), &[]);

    // 测试向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let transposed = tensor.transpose();
    assert_eq!(transposed.shape(), &[3]); // 一维张量的转置仍然是一维的

    // 测试矩阵
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let transposed = tensor.transpose();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed, Tensor::new(&[1.0, 3.0, 2.0, 4.0], &[2, 2]));

    // 测试高维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    let transposed = tensor.transpose();
    assert_eq!(transposed.shape(), &[3, 2, 1]);
    assert_eq!(
        transposed,
        Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2, 1])
    );
}

#[test]
fn test_transpose_mut() {
    // 测试标量
    let mut tensor = Tensor::new(&[1.0], &[]);
    tensor.transpose_mut();
    assert_eq!(tensor.shape(), &[]);

    // 测试向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    tensor.transpose_mut();
    assert_eq!(tensor.shape(), &[3]); // 一维张量的转置仍然是一维的

    // 测试矩阵
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    tensor.transpose_mut();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor, Tensor::new(&[1.0, 3.0, 2.0, 4.0], &[2, 2]));

    // 测试高维张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    tensor.transpose_mut();
    assert_eq!(tensor.shape(), &[3, 2, 1]);
    assert_eq!(
        tensor,
        Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2, 1])
    );
}

#[test]
fn test_transpose_dims() {
    // 1. 交换第0和第1维
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    let transposed = tensor.transpose_dims(0, 1);
    assert_eq!(transposed.shape(), &[3, 2, 1]);
    assert_eq!(
        transposed,
        Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2, 1])
    );

    // 2. 交换第1和第2维
    let transposed = tensor.transpose_dims(1, 2);
    assert_eq!(transposed.shape(), &[2, 1, 3]);
    assert_eq!(
        transposed,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3])
    );

    // 3. 测试维度超出范围的情况
    assert_panic!(tensor.transpose_dims(0, 3));
}

#[test]
fn test_transpose_dims_mut() {
    // 1. 交换第0和第1维
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    tensor.transpose_dims_mut(0, 1);
    assert_eq!(tensor.shape(), &[3, 2, 1]);
    assert_eq!(
        tensor,
        Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2, 1])
    );

    // 2. 交换第1和第2维
    tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    tensor.transpose_dims_mut(1, 2);
    assert_eq!(tensor.shape(), &[2, 1, 3]);
    assert_eq!(
        tensor,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3])
    );

    // 3. 测试维度超出范围的情况
    assert_panic!(tensor.transpose_dims_mut(0, 3));
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑transpose↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓flatten↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_flatten() {
    // 测试标量
    let tensor = Tensor::new(&[5.0], &[]);
    let flattened = tensor.flatten();
    assert_eq!(flattened.shape(), &[1]);
    assert_eq!(flattened, Tensor::new(&[5.0], &[1]));

    // 测试向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let flattened = tensor.flatten();
    assert_eq!(flattened.shape(), &[3]);
    assert_eq!(flattened, Tensor::new(&[1.0, 2.0, 3.0], &[3]));

    // 测试矩阵
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let flattened = tensor.flatten();
    assert_eq!(flattened.shape(), &[4]);
    assert_eq!(flattened, Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]));

    // 测试高维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    let flattened = tensor.flatten();
    assert_eq!(flattened.shape(), &[6]);
    assert_eq!(
        flattened,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6])
    );
}

#[test]
fn test_flatten_mut() {
    // 测试标量
    let mut tensor = Tensor::new(&[5.0], &[]);
    tensor.flatten_mut();
    assert_eq!(tensor.shape(), &[1]);
    assert_eq!(tensor, Tensor::new(&[5.0], &[1]));

    // 测试向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    tensor.flatten_mut();
    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor, Tensor::new(&[1.0, 2.0, 3.0], &[3]));

    // 测试矩阵
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    tensor.flatten_mut();
    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor, Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]));

    // 测试高维张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    tensor.flatten_mut();
    assert_eq!(tensor.shape(), &[6]);
    assert_eq!(tensor, Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]));
}

#[test]
fn test_flatten_view() {
    // 测试标量
    let tensor = Tensor::new(&[5.0], &[]);
    let flattened = tensor.flatten_view();
    assert_eq!(flattened.len(), 1);
    assert_eq!(flattened[0], 5.0);

    // 测试向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let flattened = tensor.flatten_view();
    assert_eq!(flattened.len(), 3);
    assert_eq!(flattened.to_vec(), vec![1.0, 2.0, 3.0]);

    // 测试矩阵
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let flattened = tensor.flatten_view();
    assert_eq!(flattened.len(), 4);
    assert_eq!(flattened.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

    // 测试高维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    let flattened = tensor.flatten_view();
    assert_eq!(flattened.len(), 6);
    assert_eq!(flattened.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑flatten↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓diag↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_diag() {
    // 1. 测试标量 -> 标量 (保持形状不变)
    // 1维标量
    let tensor = Tensor::new(&[1.0], &[1]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[1]);
    assert_eq!(diag, Tensor::new(&[1.0], &[1]));

    // 2维标量
    let tensor = Tensor::new(&[1.0], &[1, 1]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[1, 1]);
    assert_eq!(diag, Tensor::new(&[1.0], &[1, 1]));

    // 2. 测试向量 -> 对角方阵
    // 一维向量
    let tensor = Tensor::new(&[1.0, 2.0], &[2]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[2, 2]);
    assert_eq!(diag, Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]));

    // 列向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[3, 3]);
    assert_eq!(
        diag,
        Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3])
    );

    // 行向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[3, 3]);
    assert_eq!(
        diag,
        Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3])
    );

    // 3. 测试方阵 -> 对角向量
    // 2x2方阵
    let tensor = Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[2]);
    assert_eq!(diag, Tensor::new(&[1.0, 2.0], &[2]));

    // 3x3方阵
    let tensor = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[3]);
    assert_eq!(diag, Tensor::new(&[1.0, 2.0, 3.0], &[3]));

    // 4. 测试非法输入
    // 0维标量
    let tensor = Tensor::new(&[1.0], &[]);
    assert_panic!(tensor.diag(), "张量维度必须为1或2");

    // 非方阵 (2x3)
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_panic!(tensor.diag(), "张量必须是标量、向量或方阵");

    // 非方阵 (3x2)
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_panic!(tensor.diag(), "张量必须是标量、向量或方阵");

    // 三维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2, 1]);
    assert_panic!(tensor.diag(), "张量维度必须为1或2");

    // 四维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
    assert_panic!(tensor.diag(), "张量维度必须为1或2");
}

#[test]
fn test_diag_mut() {
    // 1. 测试标量 -> 标量 (保持形状不变)
    // 1维标量
    let mut tensor = Tensor::new(&[1.0], &[1]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[1]);
    assert_eq!(tensor, Tensor::new(&[1.0], &[1]));

    // 2维标量
    let mut tensor = Tensor::new(&[1.0], &[1, 1]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[1, 1]);
    assert_eq!(tensor, Tensor::new(&[1.0], &[1, 1]));

    // 2. 测试向量 -> 对角方阵
    // 一维向量
    let mut tensor = Tensor::new(&[1.0, 2.0], &[2]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor, Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]));

    // 列向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[3, 3]);
    assert_eq!(
        tensor,
        Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3])
    );

    // 行向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[3, 3]);
    assert_eq!(
        tensor,
        Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3])
    );

    // 3. 测试方阵 -> 对角向量
    // 2x2方阵
    let mut tensor = Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[2]);
    assert_eq!(tensor, Tensor::new(&[1.0, 2.0], &[2]));

    // 3x3方阵
    let mut tensor = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor, Tensor::new(&[1.0, 2.0, 3.0], &[3]));

    // 4. 测试非法输入
    // 0维标量
    let mut tensor = Tensor::new(&[1.0], &[]);
    assert_panic!(tensor.diag_mut(), "张量维度必须为1或2");

    // 非方阵 (2x3)
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_panic!(tensor.diag_mut(), "张量必须是标量、向量或方阵");

    // 非方阵 (3x2)
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_panic!(tensor.diag_mut(), "张量必须是标量、向量或方阵");

    // 三维张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2, 1]);
    assert_panic!(tensor.diag_mut(), "张量维度必须为1或2");

    // 四维张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
    assert_panic!(tensor.diag_mut(), "张量维度必须为1或2");
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑diag↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
