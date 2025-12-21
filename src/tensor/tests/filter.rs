use crate::tensor::Tensor;
use crate::{assert_panic, tensor_where, tensor_where_f32, tensor_where_tensor};
use approx::assert_abs_diff_eq; // 添加这个引用

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓where_with↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_where_with() {
    // 基本功能测试
    let t = Tensor::new(&[0.9, 1.0, 1.1, -1.0], &[4]);

    // 测试固定值转换
    let result = t.where_with_f32(|x| x >= 0.0, |_| 0.0, |x| -x);
    let expected = Tensor::new(&[0.0, 0.0, 0.0, 1.0], &[4]);
    // 使用近似比较
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );

    // 测试条件值转换
    let result = t.where_with_f32(|x| x > 1.0, |x| x * 2.0, |x| x / 2.0);
    let expected = Tensor::new(&[0.45, 0.5, 2.2, -0.5], &[4]);
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );

    // 测试不同维度的张量
    let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2], &[2, 1, 2]);
    let result = t.where_with_f32(|x| x >= 1.0, |x| x + 1.0, |x| x - 1.0);
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
    let result = t.where_with_f32(|x| x >= 0.0, |x| x + 1.0, |x| x - 1.0);
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
    let result = t.where_with_f32(
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
    let result = t.where_with_f32(
        |x| x.abs() > 1.0,
        |x| x.signum(), // 返回符号（1.0 或 -1.0）
        |x| x,          // 保持原值
    );
    let expected = Tensor::new(&[-1.0, -1.0, 0.0, 1.0, 1.0], &[5]);
    assert_eq!(result, expected);

    // 测试区间条件：-1.0 <= x <= 1.0
    let result = t.where_with_f32(
        |x| (-1.0..=1.0).contains(&x),
        |x| x * x, // 在区间内的值平方
        |_| 0.0,   // 区间外的值设为0
    );
    let expected = Tensor::new(&[0.0, 1.0, 0.0, 1.0, 0.0], &[5]);
    assert_eq!(result, expected);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑where_with↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓where_with_tensor↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_where_with_tensor() {
    let t = Tensor::new(&[0.9, 1.0, 1.1, -1.0], &[4]);
    let y = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[4]);

    // 1. 测试过滤后的真假条件值仅使用主张量的lambda函数
    // 1.1
    let result = t.where_with_tensor(&y, |x, y| x >= y, |x, _| x + 1.0, |x, _| x - 1.0);
    let expected = Tensor::new(&[-0.1, 2.0, 2.1, -2.0], &[4]);
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );
    // 1.2 测试不同的比较条件
    let result = t.where_with_tensor(&y, |x, y| x < y, |x, _| x * 2.0, |x, _| x / 2.0);
    let expected = Tensor::new(&[1.8, 0.5, 0.55, -2.0], &[4]);
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );

    // 2. 测试过滤后的真假条件值混合使用主张量的lambda函数和常量
    // 2.1 真值条件为常量，假值条件为lambda函数
    let result = t.where_with_tensor(
        &y,
        |x, y| x >= y,
        |_, _| 1.0,   // true时返回常量1.0
        |x, _| x * x, // false时返回x的平方
    );
    let expected = Tensor::new(&[0.81, 1.0, 1.0, 1.0], &[4]);
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );

    // 2.2 真值条件为lambda函数，假值条件为常量
    let result = t.where_with_tensor(
        &y,
        |x, y| x >= y,
        |x, _| x * 2.0, // true时返回x的2倍
        |_, _| -1.0,    // false时返回常量-1.0
    );
    let expected = Tensor::new(&[-1.0, 2.0, 2.2, -1.0], &[4]);
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );

    // 3. 测试过滤后的真假条件值混合使用主副2个张量的值进行计算
    let result = t.where_with_tensor(
        &y,
        |x, y| x >= y,
        |x, y| x + y, // true时返回两个值的和
        |x, y| x * y, // false时返回两个值的积
    );
    let expected = Tensor::new(&[0.9, 2.0, 2.1, -1.0], &[4]);
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );

    // 4. 测试过滤后的真假条件值混合使用副张量的lambda函数和常量
    // 4.1 真值条件为常量，假值条件为副张量lambda函数
    let result = t.where_with_tensor(
        &y,
        |x, y| x >= y,
        |_, _| -1.0,    // true时返回常量-1
        |_, y| y * 3.0, // false时返回副张量值的3倍
    );
    let expected = Tensor::new(&[3.0, -1.0, -1.0, 3.0], &[4]);
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );
    // 4.2 真值条件为副张量lambda函数，假值条件为常量
    let result = t.where_with_tensor(
        &y,
        |x, y| x >= y,
        |_, y| y + 2.0, // true时返回副张量值加2
        |_, _| 0.0,     // false时返回常量0
    );
    let expected = Tensor::new(&[0.0, 3.0, 3.0, 0.0], &[4]);
    assert_abs_diff_eq!(
        result.data.as_slice().unwrap(),
        expected.data.as_slice().unwrap(),
        epsilon = 1e-6
    );
}

#[test]
fn test_where_with_tensor_special_values() {
    let t = Tensor::new(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0], &[4]);
    let y = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[4]);

    // 测试特殊值处理
    let result = t.where_with_tensor(&y, |x, y| x >= y, |x, _| x + 1.0, |x, _| x - 1.0);
    let result_slice = result.data.as_slice().unwrap();

    assert!(result_slice[0].is_nan()); // NaN应保持NaN
    assert!(result_slice[1].is_infinite() && result_slice[1].is_sign_positive()); // INFINITY + 1 = INFINITY
    assert!(result_slice[2].is_infinite() && result_slice[2].is_sign_negative()); // NEG_INFINITY - 1 = NEG_INFINITY
    assert_eq!(result_slice[3], 1.0); // 0.0 + 1.0 = 1.0
}

#[test]
#[should_panic(expected = "两个张量的形状必须相同")]
fn test_where_with_tensor_shape_mismatch() {
    let t = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let y = Tensor::new(&[1.0, 2.0], &[2]);

    let _ = t.where_with_tensor(&y, |x, y| x >= y, |x, _| x + 1.0, |x, _| x - 1.0);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑where_with_tensor↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓macro:tensor_where_f32!↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_tensor_where_f32_macro() {
    let t = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);

    // 1. 测试基本比较运算符
    // 1.1 大于
    let result = tensor_where_f32!(t > 0.0, t + 1.0, -1.0);
    let expected = Tensor::new(&[-1.0, -1.0, -1.0, 2.0, 3.0], &[5]);
    assert_eq!(result, expected);

    // 1.2 小于等于
    let result = tensor_where_f32!(t <= 0.0, t * 2.0, t / 2.0);
    let expected = Tensor::new(&[-4.0, -2.0, 0.0, 0.5, 1.0], &[5]);
    assert_eq!(result, expected);

    // 1.3 等于
    let result = tensor_where_f32!(t == 0.0, 1.0, t);
    let expected = Tensor::new(&[-2.0, -1.0, 1.0, 1.0, 2.0], &[5]);
    assert_eq!(result, expected);

    // 2. 测试复杂表达式
    // 2.1 条件值为表达式
    let result = tensor_where_f32!(t >= -1.0, t * t + 1.0, 0.0);
    let expected = Tensor::new(&[0.0, 2.0, 1.0, 2.0, 5.0], &[5]);
    assert_eq!(result, expected);

    // 2.2 真值为复杂表达式
    let result = tensor_where_f32!(t > 0.0, t * 2.0 + 1.0, -1.0);
    let expected = Tensor::new(&[-1.0, -1.0, -1.0, 3.0, 5.0], &[5]);
    assert_eq!(result, expected);

    // 3. 测试特殊值
    let t = Tensor::new(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0], &[4]);
    let result = tensor_where_f32!(t >= 0.0, t + 1.0, t - 1.0);
    let result_slice = result.data.as_slice().unwrap();

    assert!(result_slice[0].is_nan()); // NaN保持NaN
    assert!(result_slice[1].is_infinite() && result_slice[1].is_sign_positive()); // INFINITY + 1 = INFINITY
    assert!(result_slice[2].is_infinite() && result_slice[2].is_sign_negative()); // NEG_INFINITY - 1 = NEG_INFINITY
    assert_eq!(result_slice[3], 1.0); // 0.0 + 1.0 = 1.0

    // 4. 测试边界值
    let t = Tensor::new(&[f32::MIN, f32::MAX, 0.5], &[3]);
    let result = tensor_where_f32!(t > 0.0, t / 2.0, t * 2.0);
    let result_slice = result.data.as_slice().unwrap();

    assert_eq!(result_slice[0], f32::MIN * 2.0); // MIN * 2
    assert_eq!(result_slice[1], f32::MAX / 2.0); // MAX / 2
    assert_eq!(result_slice[2], 0.25); // 0.5 / 2 = 0.25

    // 3. 测试f32变量实例参与的表达式
    let t = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let f = -1.0;
    let result = tensor_where_f32!(t >= (-1.0 - f), t * t + 2.0 - f, 0.0 + f);
    let expected = Tensor::new(&[-1.0, -1.0, 3.0, 4.0, 7.0], &[5]);
    assert_eq!(result, expected);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑macro:tensor_where_f32!↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓macro:tensor_where_tensor!↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_tensor_where_tensor_macro() {
    let t = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let y = Tensor::new(&[0.0, 0.0, 0.0, 0.0, 0.0], &[5]);

    // 1. 测试基本比较运算符
    // 1.1 大于等于
    let result = tensor_where_tensor!(t >= y, t + y, t - y);
    let expected = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    assert_eq!(result, expected);

    // 1.2 小于
    let result = tensor_where_tensor!(t < y, t * y, t / 2.0);
    let expected = Tensor::new(&[0.0, 0.0, 0.0, 0.5, 1.0], &[5]);
    assert_eq!(result, expected);

    // 2. 测试复杂表达式
    let a = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let b = Tensor::new(&[-1.0, -2.0, 0.0, 1.0, 3.0], &[5]);

    // 2.1 使用复杂的条件表达式
    let result = tensor_where_tensor!(a >= b, a * a + b, a * b);
    let expected = Tensor::new(&[2.0, -1.0, 0.0, 2.0, 6.0], &[5]);
    assert_eq!(result, expected);

    // 2.2 使用常量表达式
    let result = tensor_where_tensor!(a > b, 1.0, -1.0);
    let expected = Tensor::new(&[-1.0, 1.0, -1.0, -1.0, -1.0], &[5]);
    assert_eq!(result, expected);

    // 2.3 逆转主副张量表达式
    let result = tensor_where_tensor!(b > a, a * a + b, a * b);
    let expected = Tensor::new(&[3.0, 2.0, 0.0, 1.0, 7.0], &[5]);
    assert_eq!(result, expected);

    // 3. 测试特殊值
    let t = Tensor::new(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0], &[4]);
    let y = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[4]);
    let result = tensor_where_tensor!(t >= y, t + 1.0, t - 1.0);
    let result_slice = result.data.as_slice().unwrap();

    assert!(result_slice[0].is_nan()); // NaN保持NaN
    assert!(result_slice[1].is_infinite() && result_slice[1].is_sign_positive()); // INFINITY + 1 = INFINITY
    assert!(result_slice[2].is_infinite() && result_slice[2].is_sign_negative()); // NEG_INFINITY - 1 = NEG_INFINITY
    assert_eq!(result_slice[3], 1.0); // 0.0 + 1.0 = 1.0

    // 4. 测试完整语法形式
    let result = tensor_where_tensor!(t >= y, t + y, t - y);
    let expected = Tensor::new(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0], &[4]);
    let result_slice = result.data.as_slice().unwrap();
    let expected_slice = expected.data.as_slice().unwrap();

    for (r, e) in result_slice.iter().zip(expected_slice.iter()) {
        if r.is_nan() {
            assert!(e.is_nan());
        } else {
            assert_eq!(r, e);
        }
    }

    // 5. 测试因形状不一致导致的错误
    let y = Tensor::new(&[0.0, 0.0, 0.0], &[3]);
    assert_panic!(
        tensor_where_tensor!(t >= y, t + y, t - y),
        "两个张量的形状必须相同，当前张量形状为[4]，比较张量形状为[3]"
    );

    // 6. 测试高维张量
    let a = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 3, 2],
    );
    let b = Tensor::new(
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[2, 3, 2],
    );
    let result = tensor_where_tensor!(a > b, a * 2.0, a / 2.0);
    let expected = Tensor::new(
        &[
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
        ],
        &[2, 3, 2],
    );
    assert_eq!(result, expected);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑macro:tensor_where_tensor!↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

#[test]
fn test_unified_where_macro() {
    let t = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let y = Tensor::new(&[0.0, 0.0, 0.0, 0.0, 0.0], &[5]);

    // 1. 测试f32常量比较
    // 1.1 大于
    let result = tensor_where!(t > 0.0, t + 1.0, -1.0);
    let expected = Tensor::new(&[-1.0, -1.0, -1.0, 2.0, 3.0], &[5]);
    assert_eq!(result, expected);

    // 1.2 使用f32变量
    let threshold = 1.0;
    let result = tensor_where!(t > (-threshold), t * 2.0, t / 2.0);
    let expected = Tensor::new(&[-1.0, -0.5, 0.0, 2.0, 4.0], &[5]);
    assert_eq!(result, expected);

    // 2. 测试张量比较
    // 2.1 大于等于
    let result = tensor_where!(t >= y, t + y, t - y); // 使用专门的张量比较宏
    let expected = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    assert_eq!(result, expected);

    // 2.2 复杂表达式
    let a = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let b = Tensor::new(&[-1.0, -2.0, 0.0, 1.0, 3.0], &[5]);
    let result = tensor_where!(a >= b, a * a + b, a * b); // 使用专门的张量比较宏
    let expected = Tensor::new(&[2.0, -1.0, 0.0, 2.0, 6.0], &[5]);
    assert_eq!(result, expected);

    // 3. 测试特殊值
    let t = Tensor::new(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0], &[4]);

    // 3.1 f32常量比较
    let result = tensor_where!(t >= 0.0, t + 1.0, t - 1.0);
    let result_slice = result.data.as_slice().unwrap();
    assert!(result_slice[0].is_nan());
    assert!(result_slice[1].is_infinite() && result_slice[1].is_sign_positive());
    assert!(result_slice[2].is_infinite() && result_slice[2].is_sign_negative());
    assert_eq!(result_slice[3], 1.0);

    // 3.2 张量比较
    let y = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[4]);
    let result = tensor_where!(t >= y, t + y, t - y); // 使用专门的张量比较宏
    let expected = Tensor::new(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0], &[4]);
    let result_slice = result.data.as_slice().unwrap();
    let expected_slice = expected.data.as_slice().unwrap();

    for (r, e) in result_slice.iter().zip(expected_slice.iter()) {
        if r.is_nan() {
            assert!(e.is_nan());
        } else {
            assert_eq!(r, e);
        }
    }
}
