/*
 * @Author       : 老董
 * @Description  : Tensor::stack 单元测试（沿新维度堆叠）
 */

use crate::assert_panic;
use crate::tensor::{Tensor, TensorError};

#[test]
fn test_stack_basic() {
    // 1. 空张量的堆叠
    assert_panic!(Tensor::stack(&[], 0), TensorError::EmptyList);

    // 2. 标量的堆叠
    let t1 = Tensor::new(&[5.0], &[]);
    let t2 = Tensor::new(&[6.0], &[1]);
    let t3 = Tensor::new(&[7.0], &[1, 1]);
    let stacked = Tensor::stack(&[&t1, &t2, &t3], 0);
    assert_eq!(stacked, Tensor::new(&[5.0, 6.0, 7.0], &[3, 1]));

    // 3. 向量的堆叠
    let t1 = Tensor::new(&[1.0, 2.0], &[2]);
    let t2 = Tensor::new(&[3.0, 4.0], &[2]);
    let stacked = Tensor::stack(&[&t1, &t2], 0);
    assert_eq!(stacked, Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
    // (形状必须严格一致，若不同则会报错)
    let t1 = Tensor::new(&[1., 2.], &[2, 1]);
    let t2 = Tensor::new(&[5.0, 6.0, 7.0], &[3, 1]);
    assert_panic!(
        Tensor::stack(&[&t1, &t2], 0),
        "stack: 张量 1 的形状 [3, 1] 与第一个张量的形状 [2, 1] 不一致"
    );

    // 4. 矩阵的堆叠
    let t1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let t2 = Tensor::new(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3]);
    let stacked = Tensor::stack(&[&t1, &t2], 0);
    assert_eq!(
        stacked,
        Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ],
            &[2, 2, 3]
        )
    );
    // (形状必须严格一致，若不同则会报错)
    let t1 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    let t2 = Tensor::new(&[5.0, 6.0], &[2, 1]);
    assert_panic!(
        Tensor::stack(&[&t1, &t2], 0),
        "stack: 张量 1 的形状 [2, 1] 与第一个张量的形状 [2, 2] 不一致"
    );

    // 5.高维张量的堆叠
    let t1 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 3, 2],
    );
    let t2 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 3, 2],
    );
    let stacked = Tensor::stack(&[&t1, &t2], 0);
    assert_eq!(
        stacked,
        Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, //
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ],
            &[2, 2, 3, 2]
        )
    );
    // (形状必须严格一致，若不同则会报错)
    let t1 = Tensor::new(&[1., 2., 3., 4.], &[2, 2, 1, 1]);
    let t2 = Tensor::new(&[1., 2., 3., 4.], &[2, 2, 1]);
    assert_panic!(
        Tensor::stack(&[&t1, &t2], 0),
        "stack: 张量 1 的形状 [2, 2, 1] 与第一个张量的形状 [2, 2, 1, 1] 不一致"
    );
}

/// 测试沿非首维度（axis != 0）的 stack 操作
#[test]
fn test_stack_with_axis() {
    // 1. 沿 axis=1 堆叠（插入新维度）
    let t1 = Tensor::new(&[1.0, 2.0], &[2]); // [2]
    let t2 = Tensor::new(&[3.0, 4.0], &[2]); // [2]
    let stacked = Tensor::stack(&[&t1, &t2], 1);
    assert_eq!(stacked.shape(), &[2, 2]);
    assert_eq!(stacked, Tensor::new(&[1.0, 3.0, 2.0, 4.0], &[2, 2]));

    // 2. 沿 axis=2 堆叠 2D 张量
    let t1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]); // [2, 2]
    let t2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]); // [2, 2]
    let stacked = Tensor::stack(&[&t1, &t2], 2);
    assert_eq!(stacked.shape(), &[2, 2, 2]);
    assert_eq!(
        stacked,
        Tensor::new(&[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], &[2, 2, 2])
    );

    // 3. 三个张量堆叠
    let t1 = Tensor::new(&[1.0, 2.0], &[2]);
    let t2 = Tensor::new(&[3.0, 4.0], &[2]);
    let t3 = Tensor::new(&[5.0, 6.0], &[2]);
    let stacked = Tensor::stack(&[&t1, &t2, &t3], 0);
    assert_eq!(stacked.shape(), &[3, 2]);
    assert_eq!(
        stacked,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
    );

    // 4. [1,1] 形状张量沿 axis=1 堆叠应保留 3D 维度
    //    回归测试：确保不被误判为标量而产生错误的 2D 结果
    let u1 = Tensor::new(&[1.0], &[1, 1]);
    let u2 = Tensor::new(&[2.0], &[1, 1]);
    let u3 = Tensor::new(&[3.0], &[1, 1]);
    let stacked = Tensor::stack(&[&u1, &u2, &u3], 1);
    assert_eq!(stacked.shape(), &[1, 3, 1]);
    assert_eq!(stacked, Tensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1]));

    // 5. 8 个 [1,1] 张量沿 axis=1 堆叠（模拟 RNN forward_seq 场景）
    let tensors: Vec<Tensor> = (0..8).map(|i| Tensor::new(&[i as f32], &[1, 1])).collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let stacked = Tensor::stack(&refs, 1);
    assert_eq!(stacked.shape(), &[1, 8, 1]);

    // 6. [1,1] 沿 axis=0 堆叠
    let stacked = Tensor::stack(&[&u1, &u2], 0);
    assert_eq!(stacked.shape(), &[2, 1, 1]);
}
