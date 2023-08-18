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
