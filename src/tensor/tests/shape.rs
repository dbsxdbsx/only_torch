use crate::assert_panic;
use crate::errors::TensorError;
use crate::tensor::Tensor;

#[test]
fn test_compare_shapes_with_same_shapes() {
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], &[1, 4]);
    let tensor2 = Tensor::new(&[1., 2., 3., 4.], &[1, 4]);
    assert!(tensor1.is_same_shape(&tensor2));
}

#[test]
fn test_compare_shapes_with_diff_shapes() {
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], &[1, 4]);
    let tensor2 = Tensor::new(&[1., 2., 3., 4.], &[4]);
    assert!(!tensor1.is_same_shape(&tensor2));

    let tensor1 = Tensor::new(&[1., 2., 3., 4.], &[1, 4]);
    let tensor2 = Tensor::new(&[1., 2., 3., 4.], &[4, 1]);
    assert!(!tensor1.is_same_shape(&tensor2));
}

#[test]
fn test_dims() {
    let tensor = Tensor::new(&[1.], &[]);
    assert_eq!(tensor.dims(), 0);

    let tensor = Tensor::new(&[1., 2., 3., 4.], &[4]);
    assert_eq!(tensor.dims(), 1);

    let tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    assert_eq!(tensor.dims(), 2);

    let tensor = Tensor::new(&[1.], &[1, 1, 1]);
    assert_eq!(tensor.dims(), 3);
}

#[test]
fn test_is_scalar() {
    let scalar_tensor = Tensor::new(&[1.], &[]);
    assert!(scalar_tensor.is_scalar());

    let scalar_tensor = Tensor::new(&[1.], &[1]);
    assert!(scalar_tensor.is_scalar());

    let scalar_tensor = Tensor::new(&[1.], &[1, 1]);
    assert!(scalar_tensor.is_scalar());

    let non_scalar_tensor = Tensor::new(&[1., 2.], &[2]);
    assert!(!non_scalar_tensor.is_scalar());
}

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓stack↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
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
    let t1 = Tensor::new(&[1., 2., 3.0, 4.0], &[2, 1, 2, 1]);
    let t2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 1, 2, 1]);
    let stacked = Tensor::stack(&[&t1, &t2], false);
    assert_eq!(
        stacked,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,], &[4, 1, 2, 1])
    );
    // (不添加新维度的情况下，除张量的第一个维度不同外，其他维度若不同则会报错)
    let t1 = Tensor::new(&[1., 2., 3.0, 4.0], &[2, 1, 2, 1]);
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
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
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
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑stack↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓(un)squeeze↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
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
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑(un)squeeze↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓permute↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
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
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑permute↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
