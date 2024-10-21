use crate::tensor::Tensor;

#[test]
fn test_eq_number_and_tensor_with_or_without_ownership() {
    let shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1]];
    // 相同的情况
    for shape in shapes {
        let tensor = Tensor::new(&[1.], shape);
        assert_eq!(1., tensor);
        assert_eq!(1., &tensor);
        assert_eq!(tensor, 1.);
        assert_eq!(&tensor, 1.);
    }
    // 不同的情况
    for shape in shapes {
        let tensor = Tensor::new(&[1.], shape);
        assert_ne!(2., tensor);
        assert_ne!(2., &tensor);
        assert_ne!(tensor, 2.);
        assert_ne!(&tensor, 2.);
    }
}

#[test]
fn test_eq_tensors_with_or_without_ownership() {
    // 相同的情况
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[1., 2., 3.], &[3]);

    assert_eq!(tensor1, tensor2);
    assert_eq!(tensor1, &tensor2);
    assert_eq!(&tensor1, tensor2);
    assert_eq!(&tensor1, &tensor2);

    // 不同的情况：含有不一致的元素
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[1., 4., 3.], &[3]);

    assert_ne!(tensor1, tensor2);
    assert_ne!(tensor1, &tensor2);
    assert_ne!(&tensor1, tensor2);
    assert_ne!(&tensor1, &tensor2);

    // 不同的情况: 形状不同
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let tensor2 = Tensor::new(&[1., 4., 3.], &[3, 1]);

    assert_ne!(tensor1, tensor2);
    assert_ne!(tensor1, &tensor2);
    assert_ne!(&tensor1, tensor2);
    assert_ne!(&tensor1, &tensor2);
}

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓和快照相关的eq测试↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_eq_view_and_tensor_with_or_without_ownership() {
    // 相同的情况
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let mut tensor = Tensor::new(&data, &shape);
    let cloned_tensor = tensor.clone();
    let view = tensor.view();

    assert_eq!(view, tensor);
    assert_eq!(tensor, view);
    assert_eq!(view, &tensor);
    assert_eq!(&tensor, view);
    let view_mut = tensor.view_mut();
    assert_eq!(cloned_tensor, view_mut);
    assert_eq!(view_mut, cloned_tensor);
    assert_eq!(view_mut, &cloned_tensor);
    assert_eq!(&cloned_tensor, view_mut);

    // 不同的情况：含有不一致的元素
    let tensor1 = Tensor::new(&[1., 2., 3.], &[3]);
    let mut tensor2 = Tensor::new(&[1., 4., 3.], &[3]);
    let cloned_tensor1 = tensor1.clone();
    let view2 = tensor2.view();
    assert_ne!(view2, tensor1);
    assert_ne!(tensor1, view2);
    assert_ne!(view2, &tensor1);
    assert_ne!(&tensor1, view2);
    let view2_mut = tensor2.view_mut();
    assert_ne!(cloned_tensor1, view2_mut);
    assert_ne!(view2_mut, cloned_tensor1);
    assert_ne!(view2_mut, &cloned_tensor1);
    assert_ne!(&cloned_tensor1, view2_mut);

    // 不同的情况: 形状不同
    let tensor1 = Tensor::new(&[1.], &[1]);
    let mut tensor2 = Tensor::new(&[1.], &[1, 1]);
    let cloned_tensor1 = tensor1.clone();
    let view2 = tensor2.view();
    assert_ne!(view2, tensor1);
    assert_ne!(tensor1, view2);
    assert_ne!(view2, &tensor1);
    assert_ne!(&tensor1, view2);
    let view2_mut = tensor2.view_mut();
    assert_ne!(cloned_tensor1, view2_mut);
    assert_ne!(view2_mut, cloned_tensor1);
    assert_ne!(view2_mut, &cloned_tensor1);
    assert_ne!(&cloned_tensor1, view2_mut);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑和快照相关的eq测试↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
