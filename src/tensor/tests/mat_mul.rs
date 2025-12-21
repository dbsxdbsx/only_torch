use crate::tensor::Tensor;

#[test]
fn test_mat_mul_vector_vector() {
    // 结果为标量的情况
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let b = Tensor::new(&[4.0, 5.0, 6.0], &[3, 1]);
    let result = a.mat_mul(&b);
    let expected = Tensor::new(&[32.0], &[1, 1]);
    assert_eq!(result.data, expected.data);
    // 结果为矩阵的情况
    let result = b.mat_mul(&a);
    let expected = Tensor::new(&[4.0, 8.0, 12.0, 5.0, 10.0, 15.0, 6.0, 12.0, 18.0], &[3, 3]);
    assert_eq!(result.data, expected.data);
    // 构造2个，使得结果正好等于第2个张量
    let a = Tensor::eyes(2);
    let b = Tensor::new(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 3]);
    let result = a.mat_mul(&b);
    assert_eq!(result.data, b.data);
}

#[test]
fn test_mat_mul_vector_matrix() {
    let a = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let b = Tensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);
    let result = a.mat_mul(&b);
    let expected = Tensor::new(&[13.0, 16.0], &[1, 2]);
    assert_eq!(result.data, expected.data);
}

#[test]
fn test_mat_mul_matrix_vector() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(&[5.0, 6.0], &[2, 1]);
    let result = a.mat_mul(&b);
    let expected = Tensor::new(&[17.0, 39.0], &[2, 1]);
    assert_eq!(result.data, expected.data);
}

#[test]
fn test_mat_mul_matrix_matrix() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]);
    b.print();
    let result = a.mat_mul(&b);
    let expected = Tensor::new(&[21.0, 24.0, 27.0, 47.0, 54.0, 61.0], &[2, 3]);
    assert_eq!(result.data, expected.data);
}

#[test]
#[should_panic(expected = "输入的张量维度必须为2")]
fn test_mat_mul_panic_on_invalid_dimension() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::new(&[1.0, 2.0], &[2]);
    a.mat_mul(&b);
}

#[test]
#[should_panic(expected = "前一个张量的列数必须等于后一个张量的行数")]
fn test_mat_mul_panic_on_invalid_shape() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let b = Tensor::new(&[4.0, 5.0], &[2, 1]);
    a.mat_mul(&b);
}
