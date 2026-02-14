/*
 * @Author       : 老董
 * @Description  : Narrow 节点单元测试
 *
 * 测试策略（6 类标准测试）：
 * 1. 前向传播测试（高层 Graph + Var API）→ basic forward + edge cases + cannot_set_value
 * 2. VJP 单元测试（calc_grad_to_parent_index）→ 底层 NodeInner
 * 3. 端到端反向传播测试（高层 Graph + Var API）
 * 4. 梯度累积测试（高层 Graph + Var API）
 * 5. 动态形状测试
 * 6. 节点创建 API 测试
 *
 * 梯度公式：
 *   forward: output = input.narrow(axis, start, length)
 *   backward: 创建 parent_shape 大小零张量，在 [start..start+length] 位置放入 upstream_grad
 *   VJP: grad = zeros(parent_shape); grad.scatter_range(axis, start, upstream)
 *
 * Python 对照脚本: tests/python/calc_jacobi_by_pytorch/node_narrow.py
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试 ====================

/// 基本 narrow：[2, 4] narrow(1, 1, 2) → [2, 2]
///
/// input:
///   [[1, 2, 3, 4],
///    [5, 6, 7, 8]]
/// narrow(axis=1, start=1, length=2) → 取列 1..3:
///   [[2, 3],
///    [6, 7]]
#[test]
fn test_narrow_forward_basic() {
    let graph = Graph::new();

    let input_data = Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
    );
    let x = graph.input(&input_data).unwrap();
    let result = x.narrow(1, 1, 2).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 7.0, epsilon = 1e-6);
}

/// axis=0 narrow：[4, 3] narrow(0, 1, 2) → [2, 3]
///
/// input:
///   [[1, 2, 3],
///    [4, 5, 6],
///    [7, 8, 9],
///    [10, 11, 12]]
/// narrow(axis=0, start=1, length=2) → 取行 1..3:
///   [[4, 5, 6],
///    [7, 8, 9]]
#[test]
fn test_narrow_forward_axis0() {
    let graph = Graph::new();

    let input_data = Tensor::new(
        &[
            1.0, 2.0, 3.0,   // row 0
            4.0, 5.0, 6.0,   // row 1
            7.0, 8.0, 9.0,   // row 2
            10.0, 11.0, 12.0, // row 3
        ],
        &[4, 3],
    );
    let x = graph.input(&input_data).unwrap();
    let result = x.narrow(0, 1, 2).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 3]);
    assert_abs_diff_eq!(output[[0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 7.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 2]], 9.0, epsilon = 1e-6);
}

/// start=0 narrow：[2, 4] narrow(1, 0, 2) → [2, 2]（从头部开始）
///
/// input:
///   [[1, 2, 3, 4],
///    [5, 6, 7, 8]]
/// narrow(axis=1, start=0, length=2) → 取列 0..2:
///   [[1, 2],
///    [5, 6]]
#[test]
fn test_narrow_forward_start_zero() {
    let graph = Graph::new();

    let input_data = Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
    );
    let x = graph.input(&input_data).unwrap();
    let result = x.narrow(1, 0, 2).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 6.0, epsilon = 1e-6);
}

/// length=1 narrow：[2, 4] narrow(1, 2, 1) → [2, 1]
///
/// input:
///   [[1, 2, 3, 4],
///    [5, 6, 7, 8]]
/// narrow(axis=1, start=2, length=1) → 取列 2:
///   [[3],
///    [7]]
#[test]
fn test_narrow_forward_length_one() {
    let graph = Graph::new();

    let input_data = Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
    );
    let x = graph.input(&input_data).unwrap();
    let result = x.narrow(1, 2, 1).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 1]);
    assert_abs_diff_eq!(output[[0, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 7.0, epsilon = 1e-6);
}

/// Narrow 节点不能直接设置值
#[test]
fn test_narrow_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.narrow(1, 0, 1).unwrap();

    let test_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Narrow 节点不应支持直接设值");
}

// ==================== 2. VJP 单元测试 ====================

/// VJP scatter：unit upstream（全 1.0 上游梯度）
///
/// input [2, 4], narrow(axis=1, start=1, length=2) → output [2, 2]
/// upstream [2, 2] = 全 1.0
/// grad [2, 4]: 在 [:, 1..3] 填入 1.0，其余 = 0.0
/// 期望: [[0, 1, 1, 0], [0, 1, 1, 0]]
#[test]
fn test_narrow_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 4], Some("input"))
        .unwrap();
    let narrowed = inner
        .borrow_mut()
        .create_narrow_node(input.clone(), 1, 1, 2, Some("narrow"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
        )))
        .unwrap();
    narrowed.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[2, 2]);
    let grad = narrowed.calc_grad_to_parent_index(0, &upstream)?;

    assert_eq!(grad.shape(), &[2, 4]);

    // [:, 0] = 0, [:, 1] = 1, [:, 2] = 1, [:, 3] = 0
    for i in 0..2 {
        for j in 0..4 {
            let expected = if j == 1 || j == 2 { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(grad[[i, j]], expected, epsilon = 1e-6);
        }
    }

    Ok(())
}

/// VJP scatter：非 unit upstream（缩放梯度）
///
/// input [2, 4], narrow(axis=1, start=1, length=2) → output [2, 2]
/// upstream [2, 2] = [[2, 3], [4, 5]]
/// grad [2, 4]: 在 [:, 1..3] 填入 upstream 值，其余 = 0.0
/// 期望: [[0, 2, 3, 0], [0, 4, 5, 0]]
#[test]
fn test_narrow_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 4], Some("input"))
        .unwrap();
    let narrowed = inner
        .borrow_mut()
        .create_narrow_node(input.clone(), 1, 1, 2, Some("narrow"))
        .unwrap();

    input
        .set_value(Some(&Tensor::zeros(&[2, 4])))
        .unwrap();
    narrowed.forward_recursive(1, false).unwrap();

    let upstream = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = narrowed.calc_grad_to_parent_index(0, &upstream)?;

    assert_eq!(grad.shape(), &[2, 4]);

    // 期望: [[0, 2, 3, 0], [0, 4, 5, 0]]
    let expected = [
        [0.0, 2.0, 3.0, 0.0],
        [0.0, 4.0, 5.0, 0.0],
    ];
    for i in 0..2 {
        for j in 0..4 {
            assert_abs_diff_eq!(grad[[i, j]], expected[i][j], epsilon = 1e-6);
        }
    }

    Ok(())
}

// ==================== 3. 端到端反向传播测试 ====================

/// narrow(x) -> MSE loss -> backward，验证梯度正确流回
///
/// x [2, 4] → narrow(1, 1, 2) → [2, 2] → MSE(target=[2,2])
/// 梯度应只出现在 [:, 1..3] 位置，[:, 0] 和 [:, 3] 应为零
#[test]
fn test_narrow_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 4], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
    ))?;

    let narrowed = x.narrow(1, 1, 2)?;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = narrowed.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    // loss 应为有限正数
    assert!(loss_val > 0.0);
    assert!(loss_val.is_finite());

    let input_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 4]);

    // [:, 0] 和 [:, 3] 位置不参与计算，梯度应为 0
    for i in 0..2 {
        assert_abs_diff_eq!(input_grad[[i, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(input_grad[[i, 3]], 0.0, epsilon = 1e-6);
    }

    // [:, 1..3] 位置应有非零梯度
    let mut has_nonzero = false;
    for i in 0..2 {
        for j in 1..3 {
            if input_grad[[i, j]].abs() > 1e-10 {
                has_nonzero = true;
            }
        }
    }
    assert!(has_nonzero, "narrow 范围内应有非零梯度");

    Ok(())
}

// ==================== 4. 梯度累积测试 ====================

/// 测试 Narrow 梯度累积 + zero_grad
#[test]
fn test_narrow_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 4], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
    ))?;

    let narrowed = x.narrow(1, 1, 2)?;
    let target = graph.input(&Tensor::ones(&[2, 2]))?;
    let loss = narrowed.mse_loss(&target)?;

    // 第 1 次反向传播
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = x.grad()?.unwrap().clone();

    // 第 2 次反向传播（梯度累积）
    loss.backward()?;
    let grad_second = x.grad()?.unwrap();
    assert_eq!(&grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    loss.backward()?;
    let grad_after_clear = x.grad()?.unwrap();
    assert_eq!(&grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 5. 动态形状测试 ====================

/// 测试 Narrow 节点的动态形状传播
///
/// Input [4, 8] → narrow(1, 2, 3) → [?, 3]
/// batch 维度 (dim 0) 应为动态，特征维度 (dim 1) 应为固定 3
#[test]
fn test_narrow_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let result = x.narrow(1, 2, 3).unwrap();

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(3));
    assert_eq!(dyn_shape.dims().len(), 2);
}

/// 测试 Narrow 节点在不同 batch_size 下的前向计算
#[test]
fn test_narrow_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let result = x.narrow(1, 1, 4).unwrap();

    // 第一次 forward：batch=2
    result.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);

    // 更新输入为不同 batch_size
    x.set_value(&Tensor::zeros(&[6, 8])).unwrap();

    // 第二次 forward：batch=6
    result.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[6, 4]);
}

/// 测试 Narrow 节点在不同 batch_size 下的反向传播
#[test]
fn test_narrow_dynamic_batch_backward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::normal_seeded(0.0, 1.0, &[2, 8], 42))
        .unwrap();
    let result = x.narrow(1, 2, 4).unwrap();
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次：batch=2
    loss.forward().unwrap();
    let loss_val1 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);
    loss.backward().unwrap();

    // 更新为不同 batch_size
    x.set_value(&Tensor::normal_seeded(0.0, 1.0, &[5, 8], 100))
        .unwrap();
    target.set_value(&Tensor::zeros(&[5, 4])).unwrap();

    // 第二次：batch=5
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
    loss.backward().unwrap();
}

// ==================== 6. 节点创建 API 测试 ====================

use std::rc::Rc;

/// 基本创建：验证输出形状
///
/// [2, 4] narrow(axis=1, start=1, length=2) → [2, 2]
#[test]
fn test_create_narrow_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("input"))
        .unwrap();
    let narrowed = inner
        .borrow_mut()
        .create_narrow_node(input.clone(), 1, 1, 2, Some("narrowed"))
        .unwrap();

    assert_eq!(narrowed.shape(), vec![2, 2]);
    assert_eq!(narrowed.name(), Some("narrowed"));
    assert!(!narrowed.is_leaf());
    assert_eq!(narrowed.parents().len(), 1);
}

/// 无效 axis（应报错）
///
/// [2, 4] narrow(axis=2, ...) → axis=2 超出 2 维张量的范围
#[test]
fn test_create_narrow_node_invalid_axis() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_narrow_node(input, 2, 0, 1, None);
    assert!(result.is_err());
}

/// 越界（start + length 超出轴大小，应报错）
///
/// [2, 4] narrow(axis=1, start=3, length=2) → 3+2=5 > 4
#[test]
fn test_create_narrow_node_out_of_bounds() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_narrow_node(input, 1, 3, 2, None);
    assert!(result.is_err());
}

/// drop 释放
#[test]
fn test_create_narrow_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_narrowed;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let narrowed = inner
            .borrow_mut()
            .create_narrow_node(input, 1, 0, 2, None)
            .unwrap();
        weak_narrowed = Rc::downgrade(&narrowed);

        assert!(weak_narrowed.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_narrowed.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
