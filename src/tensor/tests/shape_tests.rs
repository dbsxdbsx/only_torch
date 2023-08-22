use crate::tensor::Tensor;

#[test]
fn test_compare_shapes_with_same_shapes() {
    let tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let tensor2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    assert!(tensor1.is_same_shape(&tensor2));
}

#[test]
fn test_compare_shapes_with_diff_shapes() {
    let tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let tensor2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    assert!(!tensor1.is_same_shape(&tensor2));

    let tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let tensor2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4, 1]);
    assert!(!tensor1.is_same_shape(&tensor2));
}

#[test]
fn test_dims() {
    let tensor = Tensor::new(&[1.0], &[]);
    assert_eq!(tensor.dims(), 0);

    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    assert_eq!(tensor.dims(), 1);

    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(tensor.dims(), 2);

    let tensor = Tensor::new(&[1.0], &[1, 1, 1]);
    assert_eq!(tensor.dims(), 3);
}

#[test]
fn test_is_scalar() {
    let scalar_tensor = Tensor::new(&[1.0], &[]);
    assert!(scalar_tensor.is_scalar());

    let scalar_tensor = Tensor::new(&[1.0], &[1]);
    assert!(scalar_tensor.is_scalar());

    let scalar_tensor = Tensor::new(&[1.0], &[1, 1]);
    assert!(scalar_tensor.is_scalar());

    let non_scalar_tensor = Tensor::new(&[1.0, 2.0], &[2]);
    assert!(!non_scalar_tensor.is_scalar());
}

#[test]
fn test_stack_with_same_shapes() {
    // 创建三个形状为[2, 4]的张量
    let tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let tensor2 = Tensor::new(&[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], &[2, 4]);
    let tensor3 = Tensor::new(&[17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0], &[2, 4]);
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
    let tensor1 = Tensor::new(&[1.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[1]);
    let tensor3 = Tensor::new(&[3.0], &[1, 1]);
    let tensor4 = Tensor::new(&[3.0], &[1, 1, 1]);
    // 在新维度上堆叠三个张量
    let stacked_tensor = Tensor::stack(&[&tensor1, &tensor2, &tensor3, &tensor4], true).unwrap();
    assert_eq!(stacked_tensor.shape(), &[4, 1]);
    // 在现有维度上拼接三个张量
    let stacked_tensor = Tensor::stack(&[&tensor1, &tensor2, &tensor3, &tensor4], false).unwrap();
    assert_eq!(stacked_tensor.shape(), &[4]);

    // 创建两个形状不一致的非标量型张量
    let tensor4 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor5 = Tensor::new(&[5.0, 6.0], &[2, 1]);
    // 在新维度上堆叠两个形状不一致的张量，应该返回None
    let stacked_tensor3 = Tensor::stack(&[&tensor4, &tensor5], true);
    assert_eq!(stacked_tensor3, None);
    // 在现有维度上拼接两个形状不一致的张量，应该返回None
    let stacked_tensor4 = Tensor::stack(&[&tensor4, &tensor5], false);
    assert_eq!(stacked_tensor4, None);
}
