use ndarray::{Array, IxDyn};

use crate::assert_panic;
use crate::tensor::Tensor;

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓get（返回克隆的张量）↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#[test]
fn test_get_with_scalar() {
    let shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1]];
    for shape in shapes {
        let tensor = Tensor::new(&[1.], shape);
        let result = tensor.get(&[]);
        assert_eq!(result, Tensor::new(&[1.], &[]));

        let result = tensor.get(&[1]);
        assert_eq!(result, Tensor::new(&[1.], &[]));

        let result = tensor.get(&[1, 1]);
        assert_eq!(result, Tensor::new(&[1.], &[]));

        let result = tensor.get(&[4, 1]);
        assert_eq!(result, Tensor::new(&[1.], &[]));

        let result = tensor.get(&[4, 1, 2, 3]);
        assert_eq!(result, Tensor::new(&[1.], &[]));
    }
}

#[test]
fn test_get_with_vector() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let shape = &[6];
    let tensor = Tensor::new(data, shape);

    let result = tensor.get(&[2]);
    let expected = Tensor::new(&[3.], &[]);
    assert_eq!(result, expected);
}

#[test]
fn test_get_with_matrix() {
    let data = &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let shape = &[4, 3];
    let tensor = Tensor {
        data: Array::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap(),
    };

    let result = tensor.get(&[0]);
    let expected = Tensor::new(&[1., 2., 3.], &[3]);
    assert_eq!(result, expected);

    let result = tensor.get(&[0, 1]);
    let expected = Tensor::new(&[2.], &[]);
    assert_eq!(result, expected);

    let result = tensor.get(&[1, 2]);
    let expected = Tensor::new(&[6.], &[]);
    assert_eq!(result, expected);
}

#[test]
fn test_get_with_high_dim_tensor() {
    let data = &[
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let shape = &[2, 2, 2, 2];
    let tensor = Tensor::new(data, shape);

    let result = tensor.get(&[0, 0]);
    let expected = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    assert_eq!(result, expected);

    let result = tensor.get(&[0, 1, 1]);
    let expected = Tensor::new(&[7., 8.], &[2]);
    assert_eq!(result, expected);

    let result = tensor.get(&[1, 1, 1, 1]);
    let expected = Tensor::new(&[16.], &[]);
    assert_eq!(result, expected);
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑get（返回克隆的张量）↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓快照：view(mut)↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
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
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑快照：view(mut)↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
