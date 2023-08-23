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

#[test]
fn test_stack_with_same_shapes() {
    // 创建三个形状为[2, 4]的张量
    let tensor1 = Tensor::new(&[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 4]);
    let tensor2 = Tensor::new(&[9., 10., 11., 12., 13., 14., 15., 16.], &[2, 4]);
    let tensor3 = Tensor::new(&[17., 18., 19., 20., 21., 22., 23., 24.], &[2, 4]);
    // 在新维度上堆叠三个张量
    let stacked_tensor1 = Tensor::stack(&[&tensor1, &tensor2, &tensor3], true).unwrap();
    assert_eq!(stacked_tensor1.shape(), &[3, 2, 4]);
    // 在现有维度上拼接三个张量
    let stacked_tensor2 = Tensor::stack(&[&tensor1, &tensor2, &tensor3], false).unwrap();
    assert_eq!(stacked_tensor2.shape(), &[6, 4]);
}

#[test]
fn test_stack_with_diff_shapes() {
    // 创建多个形状不一致的标量
    let tensor1 = Tensor::new(&[1.], &[]);
    let tensor2 = Tensor::new(&[2.], &[1]);
    let tensor3 = Tensor::new(&[3.], &[1, 1]);
    let tensor4 = Tensor::new(&[3.], &[1, 1, 1]);
    // 在新维度上堆叠三个张量
    let stacked_tensor = Tensor::stack(&[&tensor1, &tensor2, &tensor3, &tensor4], true).unwrap();
    assert_eq!(stacked_tensor.shape(), &[4, 1]);
    // 在现有维度上拼接三个张量
    let stacked_tensor = Tensor::stack(&[&tensor1, &tensor2, &tensor3, &tensor4], false).unwrap();
    assert_eq!(stacked_tensor.shape(), &[4]);

    // 创建两个形状不一致的非标量型张量
    let tensor4 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    let tensor5 = Tensor::new(&[5., 6.], &[2, 1]);
    // 在新维度上堆叠两个形状不一致的张量，应该返回None
    let stacked_tensor3 = Tensor::stack(&[&tensor4, &tensor5], true);
    assert_eq!(stacked_tensor3, None);
    // 在现有维度上拼接两个形状不一致的张量，应该返回None
    let stacked_tensor4 = Tensor::stack(&[&tensor4, &tensor5], false);
    assert_eq!(stacked_tensor4, None);
}

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
