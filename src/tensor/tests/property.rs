use crate::assert_panic;
use crate::tensor::Tensor;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓快照/view(_mut)↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_view() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let tensor = Tensor::new(&data, &shape);
    let view = tensor.view();
    // 检查可否正常打印
    println!("{:?}", view);
    // 检查view的索引是否正确，若非指向具体的某个元素，则会panic
    assert_panic!(view[[0]]);
    // 修改view通过索引的元素是否和原始张量保持一致
    assert_eq!(view[[0, 0]], 1.0);
    assert_eq!(view[[0, 1]], 2.0);
    assert_eq!(view[[1, 0]], 3.0);
    assert_eq!(view[[1, 1]], 4.0);
}

#[test]
fn test_view_mut() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let mut tensor = Tensor::new(&data, &shape);
    let mut view_mut = tensor.view_mut();
    // 检查可否正常打印
    println!("{:?}", view_mut);
    // 检查view_mut的索引是否正确，若非指向具体的某个元素，则会panic
    assert_panic!(view_mut[[0]]);
    // 修改view_mut中的值，并检查原始张量是否也发生了改变
    view_mut[[0, 0]] = 5.0;
    view_mut[[0, 1]] = 6.0;
    view_mut[[1, 0]] = 7.0;
    view_mut[[1, 1]] = 8.0;
    assert_eq!(tensor.data[[0, 0]], 5.0);
    assert_eq!(tensor.data[[0, 1]], 6.0);
    assert_eq!(tensor.data[[1, 0]], 7.0);
    assert_eq!(tensor.data[[1, 1]], 8.0);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑快照/view(_mut)↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓shape↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
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
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑shape↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

#[test]
fn test_dimension() {
    let tensor = Tensor::new(&[1.], &[]);
    assert_eq!(tensor.dimension(), 0);

    let tensor = Tensor::new(&[1., 2., 3., 4.], &[4]);
    assert_eq!(tensor.dimension(), 1);

    let tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    assert_eq!(tensor.dimension(), 2);

    let tensor = Tensor::new(&[1.], &[1, 1, 1]);
    assert_eq!(tensor.dimension(), 3);
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

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓size↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_size() {
    // 测试标量
    let tensor = Tensor::new(&[1.0], &[]);
    assert_eq!(tensor.size(), 1);

    // 测试一维向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(tensor.size(), 3);

    // 测试二维矩阵
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(tensor.size(), 4);

    // 测试三维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    assert_eq!(tensor.size(), 6);

    // 测试高维张量
    let tensor = Tensor::new(&[1.0; 24], &[2, 3, 2, 2]);
    assert_eq!(tensor.size(), 24);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑size↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
