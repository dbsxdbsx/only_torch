use ndarray::{Array, IxDyn};

use crate::tensor::Tensor;

#[test]
fn test_index_with_scalar() {
    let shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[1, 1, 1]];
    for shape in shapes {
        let tensor = Tensor::new(&[1.], shape);
        let result = tensor.index(&[]);
        assert_eq!(result, Tensor::new(&[1.], &[]));

        let result = tensor.index(&[1]);
        assert_eq!(result, Tensor::new(&[1.], &[]));

        let result = tensor.index(&[1, 1]);
        assert_eq!(result, Tensor::new(&[1.], &[]));

        let result = tensor.index(&[4, 1]);
        assert_eq!(result, Tensor::new(&[1.], &[]));

        let result = tensor.index(&[4, 1, 2, 3]);
        assert_eq!(result, Tensor::new(&[1.], &[]));
    }
}

#[test]
fn test_index_with_vector() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let shape = &[6];
    let tensor = Tensor::new(data, shape);

    let result = tensor.index(&[2]);
    let expected = Tensor::new(&[3.], &[]);
    assert_eq!(result, expected);
}

#[test]
fn test_index_with_matrix() {
    let data = &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let shape = &[4, 3];
    let tensor = Tensor {
        data: Array::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap(),
    };

    let result = tensor.index(&[0]);
    let expected = Tensor::new(&[1., 2., 3.], &[3]);
    assert_eq!(result, expected);

    let result = tensor.index(&[0, 1]);
    let expected = Tensor::new(&[2.], &[]);
    assert_eq!(result, expected);

    let result = tensor.index(&[1, 2]);
    let expected = Tensor::new(&[6.], &[]);
    assert_eq!(result, expected);
}

#[test]
fn test_index_with_high_dim_tensor() {
    let data = &[
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let shape = &[2, 2, 2, 2];
    let tensor = Tensor::new(data, shape);

    let result = tensor.index(&[0, 0]);
    let expected = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    assert_eq!(result, expected);

    let result = tensor.index(&[0, 1, 1]);
    let expected = Tensor::new(&[7., 8.], &[2]);
    assert_eq!(result, expected);

    let result = tensor.index(&[1, 1, 1, 1]);
    let expected = Tensor::new(&[16.], &[]);
    assert_eq!(result, expected);
}
