use crate::tensor::Tensor;

#[test]
fn test_eq_with_or_without_ownership() {
    // 相同的情况
    let tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);

    assert_eq!(tensor1, tensor2);
    assert_eq!(tensor1, &tensor2);
    assert_eq!(&tensor1, tensor2);
    assert_eq!(&tensor1, &tensor2);

    // 不同的情况：含有不一致的元素
    let tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 4.0, 3.0], &[3]);

    assert_ne!(tensor1, tensor2);
    assert_ne!(tensor1, &tensor2);
    assert_ne!(&tensor1, tensor2);
    assert_ne!(&tensor1, &tensor2);

    // 不同的情况: 形状不同
    let tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 4.0, 3.0], &[3, 1]);

    assert_ne!(tensor1, tensor2);
    assert_ne!(tensor1, &tensor2);
    assert_ne!(&tensor1, tensor2);
    assert_ne!(&tensor1, &tensor2);
}
