/*
 * @Author       : 老董
 * @Description  : Negate 节点单元测试（逐元素取反）
 *
 * 测试策略：
 * 1. 前向传播测试 → 高层 Graph + Var API
 * 2. VJP 单元测试（calc_grad_to_parent_index）→ 底层 NodeInner
 * 3. 端到端反向传播测试 → 高层 Graph + Var API
 * 4. 梯度累积测试 → 高层 Graph + Var API
 * 5. 动态形状测试 → 高层 Graph + Var API
 * 6. 特殊场景测试（双重取反、链式组合）→ 高层 API
 * 7. 节点创建 API 测试 → 底层 create_* API
 *
 * 梯度公式：
 *   negate(x) 的导数 = -1，即 grad = -upstream_grad
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 Negate 前向传播（包含正数、负数、零）
#[test]
fn test_negate_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, -2.0, 0.0, 3.5, -0.5, 0.0], &[2, 3]))
        .unwrap();
    let result = -&x;

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(&[-1.0, 2.0, 0.0, -3.5, 0.5, 0.0], &[2, 3]);
    assert_eq!(output, expected);
}

/// 测试 Negate 前向传播（极端值）
#[test]
fn test_negate_forward_extreme_values() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(
            &[f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN],
            &[2, 2],
        ))
        .unwrap();
    let result = -&x;

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(
        &[f32::NEG_INFINITY, f32::INFINITY, -f32::MAX, -f32::MIN],
        &[2, 2],
    );
    assert_eq!(output, expected);
}

/// 测试 Negate 节点不能直接设置值
#[test]
fn test_negate_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2]))
        .unwrap();
    let result = -&x;

    let test_value = Tensor::new(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Negate 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================
//
// negate'(x) = -1，VJP: grad = -upstream_grad

/// 测试 Negate VJP（全 1 上游梯度）
#[test]
fn test_negate_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let neg = inner
        .borrow_mut()
        .create_negate_node(x.clone(), Some("neg"))
        .unwrap();

    // x = [1.0, -2.0, 0.0, 3.0]
    x.set_value(Some(&Tensor::new(&[1.0, -2.0, 0.0, 3.0], &[2, 2])))
        .unwrap();
    neg.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = neg
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    // grad = -upstream = [-1, -1, -1, -1]
    let expected = Tensor::new(&[-1.0, -1.0, -1.0, -1.0], &[2, 2]);
    assert_eq!(&grad, &expected);

    Ok(())
}

/// 测试 Negate VJP（非单位上游梯度）
#[test]
fn test_negate_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let neg = inner
        .borrow_mut()
        .create_negate_node(x.clone(), Some("neg"))
        .unwrap();

    // x = [1.0, -2.0, 0.0, 3.0]
    x.set_value(Some(&Tensor::new(&[1.0, -2.0, 0.0, 3.0], &[2, 2])))
        .unwrap();
    neg.forward_recursive(1, false).unwrap();

    // upstream = [2, 3, 4, 5]
    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = neg
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    // grad = -upstream = [-2, -3, -4, -5]
    let expected = Tensor::new(&[-2.0, -3.0, -4.0, -5.0], &[2, 2]);
    assert_eq!(&grad, &expected);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 Negate 端到端反向传播：result = -x → loss = MSE(result, target)
#[test]
fn test_negate_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2]))?;

    let result = -&x;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // result = -x = [-1, 2, -3, 4]
    // loss = mean([1, 4, 9, 16]) = 7.5
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 7.5, epsilon = 1e-6);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = result/2 = [-0.5, 1.0, -1.5, 2.0]
    // ∂result/∂x = -1
    // ∂loss/∂x = [-0.5, 1.0, -1.5, 2.0] * (-1) = [0.5, -1.0, 1.5, -2.0]
    assert_abs_diff_eq!(x_grad[[0, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(x_grad[[0, 1]], -1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x_grad[[1, 0]], 1.5, epsilon = 1e-6);
    assert_abs_diff_eq!(x_grad[[1, 1]], -2.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试梯度累积：多次 backward 不调用 zero_grad，梯度应累加
#[test]
fn test_negate_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2]))?;

    let result = -&x;
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

/// 测试 Negate 节点的动态形状传播
#[test]
fn test_negate_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    let result = -&h0;

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

// ==================== 特殊场景测试 ====================

/// 测试双重取反：-(-x) == x
#[test]
fn test_negate_double_negation() {
    let graph = Graph::new();

    let data = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], &[2, 3]);
    let x = graph.input(&data).unwrap();
    let neg1 = -&x;
    let neg2 = -&neg1;

    neg2.forward().unwrap();

    let original = x.value().unwrap().unwrap();
    let double_neg = neg2.value().unwrap().unwrap();
    assert_abs_diff_eq!(&original, &double_neg, epsilon = 1e-6);
}

/// 测试 Negate 只生成 1 个节点（而非旧的 Input + Multiply 两个节点）
#[test]
fn test_negate_single_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let neg = inner
        .borrow_mut()
        .create_negate_node(x.clone(), Some("neg"))
        .unwrap();

    // Negate 只有 1 个父节点
    assert_eq!(neg.parents().len(), 1);
    // 父节点就是 x
    assert_eq!(neg.parents()[0].id(), x.id());
}

// ==================== 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_negate_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let neg = inner
        .borrow_mut()
        .create_negate_node(input.clone(), Some("neg"))
        .unwrap();

    assert_eq!(neg.shape(), vec![3, 4]);
    assert_eq!(neg.name(), Some("neg"));
    assert!(!neg.is_leaf());
    assert_eq!(neg.parents().len(), 1);
}

#[test]
fn test_create_negate_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 5, 8], None)
        .unwrap();

    let neg = inner
        .borrow_mut()
        .create_negate_node(input.clone(), None)
        .unwrap();

    assert_eq!(neg.shape(), vec![2, 5, 8]);
}

#[test]
fn test_create_negate_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_neg;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let neg = inner.borrow_mut().create_negate_node(input, None).unwrap();
        weak_neg = Rc::downgrade(&neg);

        assert!(weak_neg.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_neg.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
