/*
 * @Author       : 老董
 * @Description  : Abs 节点单元测试（逐元素绝对值）
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）→ 底层 create_* API（文件末尾）
 * 2. 前向传播测试 → 高层 Graph + Var API
 * 3. VJP 单元测试（calc_grad_to_parent_index）→ 底层 NodeInner
 * 4. 端到端反向传播测试 → 高层 Graph + Var API
 * 5. 梯度累积测试 → 高层 Graph + Var API
 * 6. 动态形状测试 → 高层 Graph + Var API
 * 7. 特殊场景测试（幂等性、L1 损失组件）→ 高层 API
 *
 * 梯度公式：
 *   abs(x) 的导数 = sign(x) = { 1 if x > 0, -1 if x < 0, 0 if x = 0 }
 *   （x=0 处梯度为 0，与 PyTorch 行为一致）
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 Abs 前向传播（包含正数、负数、零）
#[test]
fn test_abs_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[0.5, -1.0, 0.0, 2.0, -3.0, 0.0], &[2, 3]))
        .unwrap();
    let result = x.abs();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(&[0.5, 1.0, 0.0, 2.0, 3.0, 0.0], &[2, 3]);
    assert_eq!(output, expected);
}

/// 测试 Abs 前向传播（极端值）
#[test]
fn test_abs_forward_extreme_values() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(
            &[f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX],
            &[2, 2],
        ))
        .unwrap();
    let result = x.abs();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(
        &[f32::INFINITY, f32::INFINITY, f32::MIN.abs(), f32::MAX],
        &[2, 2],
    );
    assert_eq!(output, expected);
}

/// 测试 Abs 节点不能直接设置值
#[test]
fn test_abs_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2]))
        .unwrap();
    let result = x.abs();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Abs 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证梯度计算公式。
// abs'(x) = sign(x)

/// 测试 Abs VJP（全 1 上游梯度）
///
/// abs'(x) = sign(x)，VJP: grad = upstream ⊙ sign(x)
#[test]
fn test_abs_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let abs = inner
        .borrow_mut()
        .create_abs_node(x.clone(), Some("abs"))
        .unwrap();

    // x = [0.5, -1.0, 0.0, 2.0]
    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    abs.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = abs
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    // sign([0.5, -1.0, 0.0, 2.0]) = [1, -1, 0, 1]
    assert_eq!(grad.shape(), &[2, 2]);
    let expected = Tensor::new(&[1.0, -1.0, 0.0, 1.0], &[2, 2]);
    assert_eq!(&grad, &expected);

    Ok(())
}

/// 测试 Abs VJP（非单位上游梯度）
#[test]
fn test_abs_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let abs = inner
        .borrow_mut()
        .create_abs_node(x.clone(), Some("abs"))
        .unwrap();

    // x = [0.5, -1.0, 0.0, 2.0]
    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    abs.forward_recursive(1, false).unwrap();

    // upstream = [2, 3, 4, 5]
    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = abs
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    // grad = upstream ⊙ sign(x) = [2, 3, 4, 5] ⊙ [1, -1, 0, 1] = [2, -3, 0, 5]
    let expected = Tensor::new(&[2.0, -3.0, 0.0, 5.0], &[2, 2]);
    assert_eq!(&grad, &expected);

    Ok(())
}

/// 测试 Abs VJP（全负数输入）
#[test]
fn test_abs_vjp_all_negative() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let abs = inner
        .borrow_mut()
        .create_abs_node(x.clone(), Some("abs"))
        .unwrap();

    // x = [-1, -2, -3, -4]
    x.set_value(Some(&Tensor::new(&[-1.0, -2.0, -3.0, -4.0], &[2, 2])))
        .unwrap();
    abs.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = abs
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    // sign([-1, -2, -3, -4]) = [-1, -1, -1, -1]
    let expected = Tensor::new(&[-1.0, -1.0, -1.0, -1.0], &[2, 2]);
    assert_eq!(&grad, &expected);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 Abs 端到端反向传播：result = abs(x) → loss = MSE(result, target)
#[test]
fn test_abs_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.abs();
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // abs(x) = [0.5, 1.0, 0.0, 2.0]
    // loss = mean([0.25, 1.0, 0.0, 4.0]) = 1.3125
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 1.3125, epsilon = 1e-6);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = result/2 = [0.25, 0.5, 0.0, 1.0]
    // ∂result/∂x = sign(x) = [1, -1, 0, 1]
    // ∂loss/∂x = [0.25, -0.5, 0.0, 1.0]
    assert_abs_diff_eq!(x_grad[[0, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(x_grad[[0, 1]], -0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(x_grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x_grad[[1, 1]], 1.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试梯度累积：多次 backward 不调用 zero_grad，梯度应累加
#[test]
fn test_abs_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.abs();
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // 第 1 次 backward
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = x.grad()?.unwrap().clone();

    // 第 2 次 backward（不 zero_grad → 梯度累积）
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

// ==================== 动态形状测试 ====================

/// 测试 Abs 节点的动态形状传播
#[test]
fn test_abs_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    let result = h0.abs();

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 Abs 节点在不同 batch_size 下的前向和反向计算
#[test]
fn test_abs_dynamic_batch_forward_backward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]

    let result = h0.abs();

    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();
    target.set_value(&Tensor::zeros(&[8, 4])).unwrap();

    // 第二次 forward + backward：batch=8
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[8, 4]);
    loss.backward().unwrap();
}

// ==================== 特殊场景测试 ====================

/// 测试 Abs 的幂等性：abs(abs(x)) == abs(x)
#[test]
fn test_abs_idempotent() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[-2.0, -1.0, 1.0, 2.0], &[2, 2]))
        .unwrap();
    let abs1 = x.abs();
    let abs2 = abs1.abs();

    abs2.forward().unwrap();

    let val1 = abs1.value().unwrap().unwrap();
    let val2 = abs2.value().unwrap().unwrap();
    assert_abs_diff_eq!(&val1, &val2, epsilon = 1e-6);
}

/// 测试 Abs 作为 L1 损失的核心组件
/// L1 Loss ≈ mean(|pred - target|)
#[test]
fn test_abs_as_l1_loss_component() -> Result<(), GraphError> {
    let graph = Graph::new();

    let pred = graph.parameter(&[2, 2], Init::Zeros, "pred")?;
    let target_input = graph.input(&Tensor::new(&[0.5, 2.5, 2.0, 5.0], &[2, 2]))?;

    pred.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;

    // diff = pred - target = [0.5, -0.5, 1.0, -1.0]
    // abs_diff = [0.5, 0.5, 1.0, 1.0]
    let diff = &pred - &target_input;
    let abs_diff = diff.abs();

    abs_diff.forward()?;

    let output = abs_diff.value()?.unwrap();
    let expected = Tensor::new(&[0.5, 0.5, 1.0, 1.0], &[2, 2]);
    assert_abs_diff_eq!(&output, &expected, epsilon = 1e-6);

    Ok(())
}

// ==================== 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_abs_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let abs = inner
        .borrow_mut()
        .create_abs_node(input.clone(), Some("abs"))
        .unwrap();

    assert_eq!(abs.shape(), vec![3, 4]);
    assert_eq!(abs.name(), Some("abs"));
    assert!(!abs.is_leaf());
    assert_eq!(abs.parents().len(), 1);
}

#[test]
fn test_create_abs_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 5, 8], None)
        .unwrap();

    let abs = inner
        .borrow_mut()
        .create_abs_node(input.clone(), None)
        .unwrap();

    assert_eq!(abs.shape(), vec![2, 5, 8]);
}

#[test]
fn test_create_abs_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_abs;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let abs = inner.borrow_mut().create_abs_node(input, None).unwrap();
        weak_abs = Rc::downgrade(&abs);

        assert!(weak_abs.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_abs.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
