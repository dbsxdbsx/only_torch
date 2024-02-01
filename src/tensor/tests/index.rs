use ndarray::{Array, IxDyn};

use crate::assert_panic;
use crate::tensor::Tensor;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓get（返回克隆的张量）↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
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
//*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑get（返回克隆的张量）↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓快照：view(mut)↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
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
//*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑快照：view(mut)↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓index↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_index() {
    // 1.测试标量
    let shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1]];
    for shape in shapes {
        let tensor = Tensor::new(&[1.], shape);
        assert_eq!(tensor[[]], 1.);
    }

    // 2.测试向量
    let data = vec![1.0, 2.0, 3.0];
    let shape = vec![3];
    let tensor = Tensor::new(&data, &shape);
    // 检查索引是否正确，若非指向具体的某个元素，则会panic
    assert_panic!(tensor[[0, 0]]);
    assert_panic!(tensor[[0, 0, 0]]);
    // 检查索引是否正确，若指向具体的某个元素，则返回该元素的值
    assert_eq!(tensor[[0]], 1.0);
    assert_eq!(tensor[[1]], 2.0);
    assert_eq!(tensor[[2]], 3.0);

    // 3.测试矩阵
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let tensor = Tensor::new(&data, &shape);
    // 检查索引是否正确，若非指向具体的某个元素，则会panic
    assert_panic!(tensor[[0]]);
    assert_panic!(tensor[[0, 0, 0]]);
    // 检查索引是否正确，若指向具体的某个元素，则返回该元素的值
    assert_eq!(tensor[[0, 0]], 1.0);
    assert_eq!(tensor[[0, 1]], 2.0);
    assert_eq!(tensor[[1, 0]], 3.0);
    assert_eq!(tensor[[1, 1]], 4.0);

    // 3.测试三维张量
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let shape = vec![2, 2, 3];
    let tensor = Tensor::new(&data, &shape);
    // 检查索引是否正确，若非指向具体的某个元素，则会panic
    assert_panic!(tensor[[0]]);
    assert_panic!(tensor[[0, 0]]);
    assert_panic!(tensor[[0, 0, 0, 0]]);
    // 检查索引是否正确，若指向具体的某个元素，则返回该元素的值
    assert_eq!(tensor[[0, 0, 0]], 1.0);
    assert_eq!(tensor[[0, 0, 1]], 2.0);
    assert_eq!(tensor[[0, 0, 2]], 3.0);
    assert_eq!(tensor[[0, 1, 0]], 4.0);
    assert_eq!(tensor[[0, 1, 1]], 5.0);
    assert_eq!(tensor[[0, 1, 2]], 6.0);
    assert_eq!(tensor[[1, 0, 0]], 7.0);
    assert_eq!(tensor[[1, 0, 1]], 8.0);
    assert_eq!(tensor[[1, 0, 2]], 9.0);
    assert_eq!(tensor[[1, 1, 0]], 10.0);
    assert_eq!(tensor[[1, 1, 1]], 11.0);
    assert_eq!(tensor[[1, 1, 2]], 12.0);

    // 4.测试四维张量
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, //
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, //
        19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    ];
    let shape = vec![2, 2, 3, 2];
    let tensor = Tensor::new(&data, &shape);
    assert_panic!(tensor[[]]);
    assert_panic!(tensor[[0]]);
    assert_panic!(tensor[[0, 0]]);
    assert_panic!(tensor[[0, 0, 0]]);
    assert_eq!(tensor[[0, 0, 0, 0]], 1.0);
    assert_eq!(tensor[[0, 1, 2, 1]], 12.0);
    assert_eq!(tensor[[1, 1, 2, 1]], 24.0);

    // 5.(五维及以上张量的索引是不支持的)
}

#[test]
fn test_index_mut() {
    // 1.测试标量
    let shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1], &[1, 1, 1, 1]];
    for shape in shapes {
        let mut tensor = Tensor::new(&[1.], shape);
        tensor[[]] = 2.0;
        assert_eq!(tensor[[]], 2.0);
    }

    // 2.测试向量
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let mut tensor = Tensor::new(&data, &shape);
    tensor[[0, 0]] = 5.0;
    tensor[[1, 1]] = 6.0;
    assert_eq!(tensor[[0, 0]], 5.0);
    assert_eq!(tensor[[0, 1]], 2.0);
    assert_eq!(tensor[[1, 0]], 3.0);
    assert_eq!(tensor[[1, 1]], 6.0);

    // 3.测试三维张量
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let shape = vec![2, 2, 3];
    let mut tensor = Tensor::new(&data, &shape);
    tensor[[0, 0, 0]] = 13.0;
    tensor[[1, 1, 2]] = 14.0;
    assert_eq!(tensor[[0, 0, 0]], 13.0);
    assert_eq!(tensor[[0, 1, 2]], 6.0);
    assert_eq!(tensor[[1, 0, 1]], 8.0);
    assert_eq!(tensor[[1, 1, 2]], 14.0);

    // 4.测试四维张量
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, //
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, //
        19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    ];
    let shape = vec![2, 2, 3, 2];
    let mut tensor = Tensor::new(&data, &shape);
    tensor[[0, 0, 0, 0]] = 25.0;
    tensor[[1, 1, 1, 1]] = 26.0;
    assert_eq!(tensor[[0, 0, 0, 0]], 25.0);
    assert_eq!(tensor[[0, 1, 2, 1]], 12.0);
    assert_eq!(tensor[[1, 0, 1, 0]], 15.0);
    assert_eq!(tensor[[1, 0, 1, 1]], 16.0);
    assert_eq!(tensor[[1, 1, 1, 1]], 26.0);

    // 5.(五维及以上张量的索引是不支持的)
}
//*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑index↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
