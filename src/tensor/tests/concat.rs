/*
 * @Author       : 老董
 * @Description  : Tensor::concat 单元测试（沿现有维度拼接）
 */

use crate::assert_panic;
use crate::tensor::{Tensor, TensorError};

#[test]
fn test_concat_basic() {
    // 1.空张量的拼接
    assert_panic!(Tensor::concat(&[], 0), TensorError::EmptyList);
    // 2.标量的拼接
    let t1 = Tensor::new(&[5.0], &[]);
    let t2 = Tensor::new(&[6.0], &[1]);
    let t3 = Tensor::new(&[7.0], &[1, 1]);
    let result = Tensor::concat(&[&t1, &t2, &t3], 0);
    assert_eq!(result, Tensor::new(&[5.0, 6.0, 7.0], &[3]));

    // 3.向量的拼接
    let t1 = Tensor::new(&[1.0, 2.0], &[2]);
    let t2 = Tensor::new(&[3.0, 4.0], &[2]);
    let result = Tensor::concat(&[&t1, &t2], 0);
    assert_eq!(result, Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]));
    // (拼接维度可以不同)
    let t1 = Tensor::new(&[1., 2.], &[2]);
    let t2 = Tensor::new(&[6.0, 7.0, 8.0], &[3]);
    let result = Tensor::concat(&[&t1, &t2], 0);
    assert_eq!(result, Tensor::new(&[1.0, 2.0, 6.0, 7.0, 8.0], &[5]));
    // (除拼接维度外，其他维度若不同则会报错)
    let t1 = Tensor::new(&[1., 2.], &[2]);
    let t2 = Tensor::new(&[6.0, 7.0, 8.0], &[3, 1]);
    assert_panic!(
        Tensor::concat(&[&t1, &t2], 0),
        "concat: 张量 1 的维度 2 与第一个张量的维度 1 不一致"
    );

    // 4.矩阵的拼接
    let t1 = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let t2 = Tensor::new(&[7., 8., 9., 10., 11., 12.], &[2, 3]);
    let result = Tensor::concat(&[&t1, &t2], 0);
    assert_eq!(
        result,
        Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5., 6., 7., 8., 9., 10., 11., 12.],
            &[4, 3]
        )
    );
    // (拼接维度可以不同)
    let t1 = Tensor::new(&[1., 2., 3.0, 4.0, 5., 6.], &[2, 3]);
    let t2 = Tensor::new(&[7.0, 8.0, 9.0], &[1, 3]);
    let result = Tensor::concat(&[&t1, &t2], 0);
    assert_eq!(
        result,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
    );
    // (除拼接维度外，其他维度若不同则会报错)
    let t1 = Tensor::new(&[1., 2., 3.0, 4.0, 5., 6.], &[2, 3]);
    let t2 = Tensor::new(&[7.0, 8.0, 9.0, 10.0], &[1, 4]);
    assert_panic!(
        Tensor::concat(&[&t1, &t2], 0),
        "concat: 张量 1 在维度 1 的大小 4 与第一个张量的 3 不一致"
    );

    // 5.高维张量的拼接
    let t1 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 3, 2, 1],
    );
    let t2 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 3, 2, 1],
    );
    let result = Tensor::concat(&[&t1, &t2], 0);
    assert_eq!(
        result,
        Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, 5., 6., 7., 8., 9., 10., 11., 12., 1., 2., 3., 4., 5., 6., 7.,
                8., 9., 10., 11., 12.
            ],
            &[4, 3, 2, 1]
        )
    );
    // (拼接维度可以不同)
    let t1 = Tensor::new(&[1., 2., 3., 4.], &[2, 1, 2, 1]);
    let t2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 1, 2, 1]);
    let result = Tensor::concat(&[&t1, &t2], 0);
    assert_eq!(
        result,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,], &[4, 1, 2, 1])
    );
    // (除拼接维度外，其他维度若不同则会报错)
    let t1 = Tensor::new(&[1., 2., 3., 4.], &[2, 1, 2, 1]);
    let t2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 1, 2]);
    assert_panic!(
        Tensor::concat(&[&t1, &t2], 0),
        "concat: 张量 1 的维度 3 与第一个张量的维度 4 不一致"
    );
}

/// 测试沿非首维度（axis != 0）的 concat 操作
#[test]
fn test_concat_with_axis() {
    // 1. 沿 axis=1 拼接 2D 张量
    let t1 = Tensor::new(&[1.0, 2.0], &[1, 2]); // [1, 2]
    let t2 = Tensor::new(&[3.0, 4.0, 5.0], &[1, 3]); // [1, 3]
    let concat = Tensor::concat(&[&t1, &t2], 1);
    assert_eq!(concat.shape(), &[1, 5]);
    assert_eq!(concat, Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]));

    // 2. 沿 axis=1 拼接更大的 2D 张量
    let t1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]); // [2, 2]
    let t2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]); // [2, 3]
    let concat = Tensor::concat(&[&t1, &t2], 1);
    assert_eq!(concat.shape(), &[2, 5]);
    assert_eq!(
        concat,
        Tensor::new(
            &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0],
            &[2, 5]
        )
    );

    // 3. 沿 axis=2 拼接 3D 张量
    let t1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]); // [1, 2, 2]
    let t2 = Tensor::new(&[5.0, 6.0], &[1, 2, 1]); // [1, 2, 1]
    let concat = Tensor::concat(&[&t1, &t2], 2);
    assert_eq!(concat.shape(), &[1, 2, 3]);
    assert_eq!(
        concat,
        Tensor::new(&[1.0, 2.0, 5.0, 3.0, 4.0, 6.0], &[1, 2, 3])
    );

    // 4. axis 超出维度应 panic
    let t1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let t2 = Tensor::new(&[3.0, 4.0], &[1, 2]);
    assert_panic!(
        Tensor::concat(&[&t1, &t2], 2),
        "concat: axis 2 超出张量维度 2"
    );

    // 5. 除 axis 外其他维度不一致应 panic
    let t1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let t2 = Tensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);
    assert_panic!(
        Tensor::concat(&[&t1, &t2], 1),
        "concat: 张量 1 在维度 0 的大小 2 与第一个张量的 1 不一致"
    );

    // 6. [1,1] 形状张量沿 dim=1 拼接应保留 2D 维度
    //    回归测试：确保不被误判为标量而产生错误的 1D 结果
    let u1 = Tensor::new(&[1.0], &[1, 1]);
    let u2 = Tensor::new(&[2.0], &[1, 1]);
    let result = Tensor::concat(&[&u1, &u2], 1);
    assert_eq!(result.shape(), &[1, 2]);
    assert_eq!(result, Tensor::new(&[1.0, 2.0], &[1, 2]));

    // 7. 三个 [1,1] 张量沿 dim=1 拼接
    let u3 = Tensor::new(&[3.0], &[1, 1]);
    let result = Tensor::concat(&[&u1, &u2, &u3], 1);
    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result, Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]));

    // 8. [1,1] 沿 dim=0 拼接也应保留 2D
    let result = Tensor::concat(&[&u1, &u2], 0);
    assert_eq!(result.shape(), &[2, 1]);
    assert_eq!(result, Tensor::new(&[1.0, 2.0], &[2, 1]));
}
