use crate::tensor::Tensor;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`sign`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_sign_basic() {
    // 1. 基本符号测试：正数、负数、零
    let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let y = x.sign();
    let expected = Tensor::new(&[-1.0, -1.0, 0.0, 1.0, 1.0], &[5]);
    assert_eq!(y, expected);
}

#[test]
fn test_sign_special_values() {
    // 2. 特殊值测试：无穷大、NaN
    let x = Tensor::new(&[f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -0.0], &[4]);
    let y = x.sign();

    // INFINITY -> 1.0
    assert_eq!(y.get(&[0]).get_data_number().unwrap(), 1.0);
    // NEG_INFINITY -> -1.0
    assert_eq!(y.get(&[1]).get_data_number().unwrap(), -1.0);
    // NaN -> NaN
    assert!(y.get(&[2]).get_data_number().unwrap().is_nan());
    // -0.0 == 0.0 在 Rust 中为 true，所以返回 0.0（与 PyTorch 行为一致）
    assert_eq!(y.get(&[3]).get_data_number().unwrap(), 0.0);
}

#[test]
fn test_sign_shapes() {
    // 3. 不同形状的张量
    // 标量
    let scalar = Tensor::new(&[-5.0], &[]);
    assert_eq!(scalar.sign(), Tensor::new(&[-1.0], &[]));

    // 向量
    let vec = Tensor::new(&[3.0, -3.0], &[2]);
    assert_eq!(vec.sign(), Tensor::new(&[1.0, -1.0], &[2]));

    // 矩阵
    let mat = Tensor::new(&[-1.0, 2.0, 0.0, -3.0], &[2, 2]);
    assert_eq!(mat.sign(), Tensor::new(&[-1.0, 1.0, 0.0, -1.0], &[2, 2]));

    // 高维张量
    let high_dim = Tensor::new(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], &[2, 2, 2]);
    let expected = Tensor::new(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], &[2, 2, 2]);
    assert_eq!(high_dim.sign(), expected);
}

#[test]
fn test_sign_mut() {
    // 4. 就地修改版本
    let mut x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    x.sign_mut();
    let expected = Tensor::new(&[-1.0, -1.0, 0.0, 1.0, 1.0], &[5]);
    assert_eq!(x, expected);
}

#[test]
fn test_sign_preserves_shape() {
    // 5. 确保 sign 操作保持形状不变
    let shapes: &[&[usize]] = &[&[], &[1], &[3], &[2, 3], &[2, 3, 4]];
    for shape in shapes {
        let size: usize = shape.iter().product::<usize>().max(1);
        let data: Vec<f32> = (0..size)
            .map(|i| (i as f32) - (size as f32 / 2.0))
            .collect();
        let x = Tensor::new(&data, shape);
        let y = x.sign();
        assert_eq!(y.shape(), *shape, "sign 应保持形状不变");
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑`sign`↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`abs`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_abs_basic() {
    // 基本绝对值测试：正数、负数、零
    let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let y = x.abs();
    let expected = Tensor::new(&[2.0, 1.0, 0.0, 1.0, 2.0], &[5]);
    assert_eq!(y, expected);
}

#[test]
fn test_abs_special_values() {
    // 特殊值测试：无穷大、NaN
    let x = Tensor::new(&[f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -0.0], &[4]);
    let y = x.abs();

    // INFINITY -> INFINITY
    assert_eq!(y.get(&[0]).get_data_number().unwrap(), f32::INFINITY);
    // NEG_INFINITY -> INFINITY
    assert_eq!(y.get(&[1]).get_data_number().unwrap(), f32::INFINITY);
    // NaN -> NaN
    assert!(y.get(&[2]).get_data_number().unwrap().is_nan());
    // -0.0 -> 0.0
    assert_eq!(y.get(&[3]).get_data_number().unwrap(), 0.0);
}

#[test]
fn test_abs_shapes() {
    // 不同形状的张量
    // 标量
    let scalar = Tensor::new(&[-5.0], &[]);
    assert_eq!(scalar.abs(), Tensor::new(&[5.0], &[]));

    // 向量
    let vec = Tensor::new(&[3.0, -3.0], &[2]);
    assert_eq!(vec.abs(), Tensor::new(&[3.0, 3.0], &[2]));

    // 矩阵
    let mat = Tensor::new(&[-1.0, 2.0, 0.0, -3.0], &[2, 2]);
    assert_eq!(mat.abs(), Tensor::new(&[1.0, 2.0, 0.0, 3.0], &[2, 2]));

    // 高维张量
    let high_dim = Tensor::new(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], &[2, 2, 2]);
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    assert_eq!(high_dim.abs(), expected);
}

#[test]
fn test_abs_mut() {
    // 就地修改版本
    let mut x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    x.abs_mut();
    let expected = Tensor::new(&[2.0, 1.0, 0.0, 1.0, 2.0], &[5]);
    assert_eq!(x, expected);
}

#[test]
fn test_abs_preserves_shape() {
    // 确保 abs 操作保持形状不变
    let shapes: &[&[usize]] = &[&[], &[1], &[3], &[2, 3], &[2, 3, 4]];
    for shape in shapes {
        let size: usize = shape.iter().product::<usize>().max(1);
        let data: Vec<f32> = (0..size)
            .map(|i| (i as f32) - (size as f32 / 2.0))
            .collect();
        let x = Tensor::new(&data, shape);
        let y = x.abs();
        assert_eq!(y.shape(), *shape, "abs 应保持形状不变");
    }
}

#[test]
fn test_abs_idempotent() {
    // 幂等性测试：对已经是正数的张量，abs 应该不变
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    assert_eq!(x.abs(), x);

    // 对绝对值结果再次 abs，结果应该相同
    let y = Tensor::new(&[-1.0, -2.0, 3.0, -4.0], &[4]);
    let abs_y = y.abs();
    assert_eq!(abs_y.abs(), abs_y);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑`abs`↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
