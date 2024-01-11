use crate::tensor::Tensor;
#[test]
fn test_sub_assign_f32_to_tensor() {
    // 标量
    let mut tensor = Tensor::new(&[3.0], &[]);
    tensor -= 2.0;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(tensor, expected);
    // 向量
    let mut tensor = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    tensor -= 1.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(tensor, expected);
    // 矩阵
    let mut tensor = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    tensor -= 1.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(tensor, expected);
    // 三阶张量
    let mut tensor = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    tensor -= 1.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(tensor, expected);
}

#[test]
fn test_sub_assign_f32_to_tensor_ref() {
    // 标量
    let mut tensor = Tensor::new(&[3.0], &[]);
    let tensor_ref = &mut tensor;
    *tensor_ref -= 2.0;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(*tensor_ref, expected);
    // 向量
    let mut tensor = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    let tensor_ref = &mut tensor;
    *tensor_ref -= 1.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(*tensor_ref, expected);
    // 矩阵
    let mut tensor = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let tensor_ref = &mut tensor;
    *tensor_ref -= 1.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(*tensor_ref, expected);
    // 三阶张量
    let mut tensor = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    let tensor_ref = &mut tensor;
    *tensor_ref -= 1.0;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(*tensor_ref, expected);
}

#[test]
fn test_sub_assign_tensor_to_tensor() {
    // 标量
    let mut tensor1 = Tensor::new(&[3.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    tensor1 -= tensor2;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(tensor1, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    tensor1 -= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(tensor1, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    tensor1 -= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(tensor1, expected);
    // 三阶张量
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    tensor1 -= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(tensor1, expected);
}

#[test]
fn test_sub_assign_tensor_ref_to_tensor() {
    // 标量
    let mut tensor1 = Tensor::new(&[3.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    tensor1 -= &tensor2;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(tensor1, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    tensor1 -= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(tensor1, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    tensor1 -= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(tensor1, expected);
    // 三阶张量
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    tensor1 -= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(tensor1, expected);
}

#[test]
fn test_sub_assign_tensor_to_tensor_ref() {
    // 标量
    let mut tensor1 = Tensor::new(&[3.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref -= tensor2;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(*tensor1_ref, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref -= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(*tensor1_ref, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref -= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(*tensor1_ref, expected);
    // 三阶张量
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref -= tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(*tensor1_ref, expected);
}

#[test]
fn test_sub_assign_tensor_ref_to_tensor_ref() {
    // 标量
    let mut tensor1 = Tensor::new(&[3.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref -= &tensor2;
    let expected = Tensor::new(&[1.0], &[]);
    assert_eq!(*tensor1_ref, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref -= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(*tensor1_ref, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref -= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(*tensor1_ref, expected);
    // 三阶张量
    let mut tensor1 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref -= &tensor2;
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    assert_eq!(*tensor1_ref, expected);
}

#[test]
#[should_panic(
    expected = "形状不一致且两个张量没有一个是标量，故无法自相减：第一个张量的形状为[3]，第二个张量的形状为[2]"
)]
fn test_sub_assign_incompatible_shapes() {
    let mut tensor1 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    let tensor2 = Tensor::new(&[2.0, 2.0], &[2]);
    tensor1 -= tensor2;
}
