use crate::assert_panic;
use crate::tensor::Tensor;

#[test]
fn test_add_assign_f32_to_tensor() {
    // 标量
    let mut tensor = Tensor::new(&[1.0], &[]);
    tensor += 2.0;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(tensor, expected);
    // 向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    tensor += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(tensor, expected);
    // 矩阵
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    tensor += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(tensor, expected);
    // 三阶张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    tensor += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(tensor, expected);
}

#[test]
fn test_add_assign_f32_to_tensor_ref() {
    // 标量
    let mut tensor = Tensor::new(&[1.0], &[]);
    let tensor_ref = &mut tensor;
    *tensor_ref += 2.0;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(*tensor_ref, expected);
    // 向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor_ref = &mut tensor;
    *tensor_ref += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(*tensor_ref, expected);
    // 矩阵
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor_ref = &mut tensor;
    *tensor_ref += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(*tensor_ref, expected);
    // 三阶张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let tensor_ref = &mut tensor;
    *tensor_ref += 1.0;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(*tensor_ref, expected);
}

#[test]
fn test_add_assign_tensor_to_tensor() {
    // 标量
    let mut tensor1 = Tensor::new(&[1.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    tensor1 += tensor2;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(tensor1, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    tensor1 += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(tensor1, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    tensor1 += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(tensor1, expected);
    // 三阶张量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    tensor1 += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(tensor1, expected);
}

#[test]
fn test_add_assign_tensor_ref_to_tensor() {
    // 标量
    let mut tensor1 = Tensor::new(&[1.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    tensor1 += &tensor2;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(tensor1, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    tensor1 += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(tensor1, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    tensor1 += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(tensor1, expected);
    // 三阶张量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    tensor1 += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(tensor1, expected);
}

#[test]
fn test_add_assign_tensor_to_tensor_ref() {
    // 标量
    let mut tensor1 = Tensor::new(&[1.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += tensor2;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(*tensor1_ref, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(*tensor1_ref, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(*tensor1_ref, expected);
    // 三阶张量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(*tensor1_ref, expected);
}

#[test]
fn test_add_assign_tensor_ref_to_tensor_ref() {
    // 标量
    let mut tensor1 = Tensor::new(&[1.0], &[]);
    let tensor2 = Tensor::new(&[2.0], &[]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += &tensor2;
    let expected = Tensor::new(&[3.0], &[]);
    assert_eq!(*tensor1_ref, expected);
    // 向量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    assert_eq!(*tensor1_ref, expected);
    // 矩阵
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(*tensor1_ref, expected);
    // 三阶张量
    let mut tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let tensor2 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2]);
    let tensor1_ref = &mut tensor1;
    *tensor1_ref += &tensor2;
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 1, 2]);
    assert_eq!(*tensor1_ref, expected);
}

#[test]
fn test_add_assign_scalar_or_ref_to_scalar_or_ref() {
    let number = 2.;
    let scalar_shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1]];

    // 测试不同形状标量间的加法组合
    for shape1 in scalar_shapes.iter() {
        let scalar1 = Tensor::new(&[number], shape1);

        for shape2 in scalar_shapes.iter() {
            let scalar2 = Tensor::new(&[1.0], shape2);

            if shape1 == shape2 {
                // 相同形状的标量相加应该成功
                // 1. 标量 += 标量
                let mut result = scalar1.clone();
                result += scalar2.clone();
                let expected = Tensor::new(&[3.0], shape1);
                assert_eq!(result, expected);

                // 2. 标量 += &标量
                let mut result = scalar1.clone();
                result += &scalar2;
                assert_eq!(result, expected);

                // 3. &标量 += 标量
                let mut result = scalar1.clone();
                let result_ref = &mut result;
                *result_ref += scalar2.clone();
                assert_eq!(result, expected);

                // 4. &标量 += &标量
                let mut result = scalar1.clone();
                let result_ref = &mut result;
                *result_ref += &scalar2;
                assert_eq!(result, expected);
            } else {
                // 不同形状的标量相加应该失败
                let expected_msg = format!(
                    "形状不一致，故无法相加：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
                    shape1, shape2
                );

                // 1. 标量 += 标量
                let mut result = scalar1.clone();
                assert_panic!(result += scalar2.clone(), expected_msg);

                // 2. 标量 += &标量
                let mut result = scalar1.clone();
                assert_panic!(result += &scalar2, expected_msg);

                // 3. &标量 += 标量
                let mut result = scalar1.clone();
                let result_ref = &mut result;
                assert_panic!(*result_ref += scalar2.clone(), expected_msg);

                // 4. &标量 += &标量
                let mut result = scalar1.clone();
                let result_ref = &mut result;
                assert_panic!(*result_ref += &scalar2, expected_msg);
            }
        }
    }
}
