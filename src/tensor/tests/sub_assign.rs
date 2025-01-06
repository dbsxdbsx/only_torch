use super::TensorCheck;
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};

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
    let expected = Tensor::new(&[1.0], &[1]);
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
    let expected = Tensor::new(&[1.0], &[1]);
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
    let expected = Tensor::new(&[1.0], &[1]);
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
    let expected = Tensor::new(&[1.0], &[1]);
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
fn test_sub_assign_scalar_or_ref_to_tensor_or_ref() {
    let number = 2.;
    // 每个test_cases的元素是个三元组，其元素分别是：张量的形状、张量的数据、正确的结果
    let test_cases = vec![
        // 标量型张量
        TensorCheck {
            shape: vec![],
            data: vec![1.],
            expected: vec![vec![-1.]],
        },
        TensorCheck {
            shape: vec![1],
            data: vec![1.],
            expected: vec![vec![-1.]],
        },
        TensorCheck {
            shape: vec![1, 1],
            data: vec![1.],
            expected: vec![vec![-1.]],
        },
        // 向量型张量
        TensorCheck {
            shape: vec![2],
            data: vec![1., 2.],
            expected: vec![vec![-1., 0.]],
        },
        TensorCheck {
            shape: vec![2, 1],
            data: vec![1., 2.],
            expected: vec![vec![-1., 0.]],
        },
        TensorCheck {
            shape: vec![1, 2],
            data: vec![1., 2.],
            expected: vec![vec![-1., 0.]],
        },
        // 矩阵型张量
        TensorCheck {
            shape: vec![2, 3],
            data: vec![1., 2., 3., 4., 5., 6.],
            expected: vec![vec![-1., 0., 1., 2., 3., 4.]],
        },
        // 高阶张量
        TensorCheck {
            shape: vec![2, 3, 1],
            data: vec![1., 2., 3., 4., 5., 6.],
            expected: vec![vec![-1., 0., 1., 2., 3., 4.]],
        },
        TensorCheck {
            shape: vec![2, 1, 3, 1],
            data: vec![1., 2., 3., 4., 5., 6.],
            expected: vec![vec![-1., 0., 1., 2., 3., 4.]],
        },
    ];

    let scalar_shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1]];

    for test_case in test_cases {
        for scalar_shape in scalar_shapes.iter() {
            let mut tensor = Tensor::new(&test_case.data, &test_case.shape);
            let scalar_tensor = Tensor::new(&[number], scalar_shape);
            let correct_shape = if scalar_tensor.is_scalar() && tensor.is_scalar() {
                vec![1]
            } else {
                test_case.shape.clone()
            };
            let expect_tensor =
                Array::from_shape_vec(IxDyn(&correct_shape), test_case.expected[0].clone())
                    .unwrap();
            // 1.张量-=标量
            tensor -= scalar_tensor.clone();
            assert_eq!(
                tensor.data,
                expect_tensor,
                "`张量-=标量`出错！使用的标量为：{:?}，张量为：{:?}",
                &[number],
                test_case.data
            );
            // 2.张量-=&标量
            let mut tensor = Tensor::new(&test_case.data, &test_case.shape);
            tensor -= &scalar_tensor;
            assert_eq!(
                tensor.data,
                expect_tensor,
                "`张量-=&标量`出错！使用的标量为：{:?}，张量为：{:?}",
                &[number],
                test_case.data
            );
            // 3.&张量-=标量
            let mut tensor = Tensor::new(&test_case.data, &test_case.shape);
            let tensor_ref = &mut tensor;
            *tensor_ref -= scalar_tensor.clone();
            assert_eq!(
                tensor.data,
                expect_tensor,
                "`&张量-=标量`出错！使用的标量为：{:?}，张量为：{:?}",
                &[number],
                test_case.data
            );
            // 4.&张量-=&标量
            let mut tensor = Tensor::new(&test_case.data, &test_case.shape);
            let tensor_ref = &mut tensor;
            *tensor_ref -= &scalar_tensor;
            assert_eq!(
                tensor.data,
                expect_tensor,
                "`&张量-=&标量`出错！使用的标量为：{:?}，张量为：{:?}",
                &[number],
                test_case.data
            );
        }
    }
}

#[test]
#[should_panic(
    expected = "形状不一致且两个张量（且没有一个是标量），故无法相减：第一个张量的形状为[3]，第二个张量的形状为[2]"
)]
fn test_sub_assign_incompatible_shapes() {
    let mut tensor1 = Tensor::new(&[1.0, 1.0, 1.0], &[3]);
    let tensor2 = Tensor::new(&[2.0, 2.0], &[2]);
    tensor1 -= tensor2;
}
