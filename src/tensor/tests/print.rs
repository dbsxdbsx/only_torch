use crate::tensor::Tensor;

#[test]
fn test_print() {
    use std::fmt::Write;

    // 测试标量
    let tensor = Tensor::new(&[1.], &[]);
    let mut buffer = String::new();
    write!(&mut buffer, "{}", tensor).unwrap();
    assert_eq!(buffer, "  1.0000\n形状: []\n");

    let tensor = Tensor::new(&[1.], &[1]);
    let mut buffer = String::new();
    write!(&mut buffer, "{}", tensor).unwrap();
    assert_eq!(buffer, "[  1.0000]\n形状: [1]\n");

    let tensor = Tensor::new(&[1.], &[1, 1]);
    let mut buffer = String::new();
    write!(&mut buffer, "{}", tensor).unwrap();
    assert_eq!(buffer, "[[  1.0000]]\n形状: [1, 1]\n");

    let tensor = Tensor::new(&[1.], &[1, 1, 1]);
    let mut buffer = String::new();
    write!(&mut buffer, "{}", tensor).unwrap();
    assert_eq!(buffer, "[[[  1.0000]]]\n形状: [1, 1, 1]\n");

    let tensor = Tensor::new(&[1.], &[1, 1, 1, 1]);
    let mut buffer = String::new();
    write!(&mut buffer, "{}", tensor).unwrap();
    assert_eq!(buffer, "[[[[  1.0000]]]]\n形状: [1, 1, 1, 1]\n");

    // 测试向量
    let tensor = Tensor::new(&[1., 2., 3.], &[3]);
    let mut buffer = String::new();
    write!(&mut buffer, "{}", tensor).unwrap();
    assert_eq!(buffer, "[  1.0000, \n   2.0000, \n   3.0000]\n形状: [3]\n");

    let tensor = Tensor::new(&[1., 2., 3.], &[1, 3]);
    let mut buffer = String::new();
    write!(&mut buffer, "{}", tensor).unwrap();
    assert_eq!(buffer, "[[  1.0000,   2.0000,   3.0000]]\n形状: [1, 3]\n");

    let tensor = Tensor::new(&[1., 2., 3.], &[3, 1]);
    let mut buffer = String::new();
    write!(&mut buffer, "{}", tensor).unwrap();
    assert_eq!(
        buffer,
        "[[  1.0000], \n [  2.0000], \n [  3.0000]]\n形状: [3, 1]\n"
    );

    // 测试矩阵
    let tensor = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let mut buffer = String::new();
    write!(&mut buffer, "{}", tensor).unwrap();
    assert_eq!(
        buffer,
        "[[  1.0000,   2.0000,   3.0000], \n [  4.0000,   5.0000,   6.0000]]\n形状: [2, 3]\n"
    );

    // 测试高阶张量
    let tensor = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3, 1]);
    let mut buffer = String::new();
    write!(&mut buffer, "{}", tensor).unwrap();
    assert_eq!(
        buffer,
        "<对于阶数大于二（rank>2）的张量（形状：[2, 3, 1]）无法展示具体数据>\n"
    );
}
