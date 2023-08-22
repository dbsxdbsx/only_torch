#[test]
fn test_dims() {
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(tensor.dims(), 2);

    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    assert_eq!(tensor.dims(), 1);

    let tensor = Tensor::new(&[1.0], &[]);
    assert_eq!(tensor.dims(), 0);
}