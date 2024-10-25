use crate::tensor::Tensor;
use approx::assert_abs_diff_eq; // 添加这个引用

#[test]
fn test_where_with() {
    // 基本功能测试
    let t = Tensor::new(&[0.9, 1.0, 1.1, -1.0], &[4]);

    // 测试固定值转换
    let result = t.where_with(|x| x >= 0.0, |_| 0.0, |x| -x);
    let expected = Tensor::new(&[0.0, 0.0, 0.0, 1.0], &[4]);
    // 使用近似比较
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );

    // 测试条件值转换
    let result = t.where_with(|x| x > 1.0, |x| x * 2.0, |x| x / 2.0);
    let expected = Tensor::new(&[0.45, 0.5, 2.2, -0.5], &[4]);
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );

    // 测试不同维度的张量
    let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2], &[2, 1, 2]);
    let result = t.where_with(|x| x >= 1.0, |x| x + 1.0, |x| x - 1.0);
    let expected = Tensor::new(&[-0.1, 2.0, 2.1, 2.2], &[2, 1, 2]);
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );

    // 同时验证形状
    assert_eq!(result.shape(), expected.shape());
}

#[test]
fn test_where_with_special_values() {
    let t = Tensor::new(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0], &[4]);

    // 测试特殊值处理
    let result = t.where_with(|x| x >= 0.0, |x| x + 1.0, |x| x - 1.0);
    let result_slice = result.data.as_slice().unwrap();

    assert!(result_slice[0].is_nan()); // NaN应保持NaN
    assert!(result_slice[1].is_infinite() && result_slice[1].is_sign_positive()); // INFINITY + 1 = INFINITY
    assert!(result_slice[2].is_infinite() && result_slice[2].is_sign_negative()); // NEG_INFINITY - 1 = NEG_INFINITY
    assert_eq!(result_slice[3], 1.0); // 0.0 + 1.0 = 1.0
}

#[test]
fn test_where_with_boundary_values() {
    let t = Tensor::new(&[f32::MIN, f32::MAX, 0.0], &[3]);

    // 测试边界值处理
    let result = t.where_with(
        |x| x > 0.0,
        |x| x / 2.0, // 对正数减半
        |x| x * 2.0, // 对非正数翻倍
    );
    let result_slice = result.data.as_slice().unwrap();

    assert_eq!(result_slice[0], f32::MIN * 2.0); // MIN * 2
    assert_eq!(result_slice[1], f32::MAX / 2.0); // MAX / 2
    assert_eq!(result_slice[2], 0.0); // 0.0 * 2 = 0.0
}

#[test]
fn test_where_with_complex_conditions() {
    let t = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);

    // 测试复杂条件：abs(x) > 1.0
    let result = t.where_with(
        |x| x.abs() > 1.0,
        |x| x.signum(), // 返回符号（1.0 或 -1.0）
        |x| x,          // 保持原值
    );
    let expected = Tensor::new(&[-1.0, -1.0, 0.0, 1.0, 1.0], &[5]);
    assert_eq!(result, expected);

    // 测试区间条件：-1.0 <= x <= 1.0
    let result = t.where_with(
        |x| (-1.0..=1.0).contains(&x),
        |x| x * x, // 在区间内的值平方
        |_| 0.0,   // 区间外的值设为0
    );
    let expected = Tensor::new(&[0.0, 1.0, 0.0, 1.0, 0.0], &[5]);
    assert_eq!(result, expected);
}
