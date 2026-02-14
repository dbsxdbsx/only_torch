/*
 * @Author       : 老董
 * @Description  : Exp（指数函数）节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ basic forward + edge cases + cannot_set_value
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ unit upstream + non-unit
 * 3. 端到端反向传播测试（高层 API）
 * 4. 梯度累积测试（高层 API）
 * 5. 动态形状测试
 * 6. Create API 测试
 *
 * 梯度公式：
 *   y = e^x
 *   dy/dx = e^x = y
 *   VJP: grad_to_parent = upstream_grad * e^x
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 API）====================

/// 测试 Exp 前向传播
///
/// exp([0, 1, 2, -1]) = [1.0, e, e², 1/e]
#[test]
fn test_exp_forward() {
    let graph = Graph::new();

    let e = std::f32::consts::E;
    let x = graph
        .input(&Tensor::new(&[0.0, 1.0, 2.0, -1.0], &[2, 2]))
        .unwrap();
    let result = x.exp();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6); // e^0 = 1
    assert_abs_diff_eq!(output[[0, 1]], e, epsilon = 1e-6); // e^1 = e
    assert_abs_diff_eq!(output[[1, 0]], e * e, epsilon = 1e-5); // e^2
    assert_abs_diff_eq!(output[[1, 1]], 1.0 / e, epsilon = 1e-6); // e^(-1) = 1/e
}

/// 测试 Exp 前向传播（边界值）
///
/// exp(-5)≈0.0067, exp(0)=1, exp(1)≈2.718, exp(5)≈148.41
#[test]
fn test_exp_forward_edge_cases() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[-5.0, 0.0, 1.0, 5.0], &[1, 4]))
        .unwrap();
    let result = x.exp();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 0.006737947, epsilon = 1e-6); // e^(-5)
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-6); // e^0 = 1
    assert_abs_diff_eq!(output[[0, 2]], std::f32::consts::E, epsilon = 1e-6); // e^1
    assert_abs_diff_eq!(output[[0, 3]], 148.41316, epsilon = 1e-3); // e^5
}

/// 测试 Exp 与 Ln 互为反函数：exp(ln(x)) ≈ x
#[test]
fn test_exp_ln_inverse() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[0.5, 1.0, 2.0, 10.0], &[2, 2]))
        .unwrap();
    let result = x.ln().exp();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[1, 1]], 10.0, epsilon = 1e-5);
}

/// 测试 Exp 节点不能直接设置值
#[test]
fn test_exp_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.exp();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Exp 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================

/// 测试 Exp VJP（单位上游梯度）
///
/// grad = e^x → e^[0, 1, -1, 2] = [1, e, 1/e, e²]
#[test]
fn test_exp_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let exp = inner
        .borrow_mut()
        .create_exp_node(x.clone(), Some("exp"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.0, 1.0, -1.0, 2.0], &[2, 2])))
        .unwrap();
    exp.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = exp.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    let e = std::f32::consts::E;
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6); // e^0 = 1
    assert_abs_diff_eq!(grad[[0, 1]], e, epsilon = 1e-6); // e^1 = e
    assert_abs_diff_eq!(grad[[1, 0]], 1.0 / e, epsilon = 1e-6); // e^(-1)
    assert_abs_diff_eq!(grad[[1, 1]], e * e, epsilon = 1e-5); // e^2

    Ok(())
}

/// 测试 Exp VJP（非单位上游梯度）
///
/// grad = upstream * e^x → [2, 3] * e^[0, 1] = [2, 3e]
#[test]
fn test_exp_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("x"))
        .unwrap();
    let exp = inner
        .borrow_mut()
        .create_exp_node(x.clone(), Some("exp"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.0, 1.0], &[1, 2])))
        .unwrap();
    exp.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 3.0], &[1, 2]);
    let grad = exp.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    let e = std::f32::consts::E;
    assert_eq!(grad.shape(), &[1, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0, epsilon = 1e-6); // 2 * e^0 = 2
    assert_abs_diff_eq!(grad[[0, 1]], 3.0 * e, epsilon = 1e-5); // 3 * e^1

    Ok(())
}

// ==================== 端到端反向传播测试（高层 API）====================

/// 测试 Exp 端到端反向传播：result = exp(input) → loss = MSE(result, target)
#[test]
fn test_exp_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.0, 1.0, -1.0, 2.0], &[2, 2]))?;

    let result = x.exp();
    let target = graph.input(&Tensor::new(&[1.0, 2.0, 0.5, 5.0], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);

    Ok(())
}

// ==================== 梯度累积测试（高层 API）====================

/// 测试 Exp 梯度累积
#[test]
fn test_exp_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.0, 1.0, -1.0, 2.0], &[2, 2]))?;

    let result = x.exp();
    let target = graph.input(&Tensor::new(&[1.0, 2.0, 0.5, 5.0], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = x.grad()?.unwrap().clone();

    loss.backward()?;
    let grad_second = x.grad()?.unwrap();
    assert_eq!(&grad_second, &(&grad_first * 2.0));

    graph.zero_grad()?;
    loss.backward()?;
    let grad_after_clear = x.grad()?.unwrap();
    assert_eq!(&grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 Exp 节点的动态形状传播
#[test]
fn test_exp_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::ones(&[4, 8])).unwrap();
    let result = x.exp();

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(8), "特征维度应该是 8");
}

/// 测试 Exp 节点在不同 batch_size 下的前向计算
#[test]
fn test_exp_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let result = x.exp();

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 8], "第一次 forward: batch=2");
    // exp(0) = 1
    assert_abs_diff_eq!(value1[[0, 0]], 1.0, epsilon = 1e-6);

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 8], "第二次 forward: batch=8");
}

/// 测试 Exp 节点在不同 batch_size 下的反向传播
#[test]
fn test_exp_dynamic_batch_backward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(
            &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            &[2, 4],
        ))
        .unwrap();

    let result = x.exp();

    let target = graph.input(&Tensor::ones(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(
        &[
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ],
        &[6, 4],
    ))
    .unwrap();
    target.set_value(&Tensor::ones(&[6, 4])).unwrap();

    // 第二次 forward + backward：batch=6
    loss.forward().unwrap();
    assert_eq!(
        result.value().unwrap().unwrap().shape(),
        &[6, 4],
        "第二次 forward: batch=6"
    );
    loss.backward().unwrap();
}

// ==================== 节点创建 API 测试 ====================

#[test]
fn test_create_exp_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let exp = inner
        .borrow_mut()
        .create_exp_node(input.clone(), Some("exp"))
        .unwrap();

    assert_eq!(exp.shape(), vec![3, 4]);
    assert_eq!(exp.name(), Some("exp"));
}

#[test]
fn test_create_exp_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 5, 8], None)
        .unwrap();

    let exp = inner
        .borrow_mut()
        .create_exp_node(input.clone(), None)
        .unwrap();

    assert_eq!(exp.shape(), vec![2, 5, 8]);
}

#[test]
fn test_create_exp_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_exp;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let exp = inner.borrow_mut().create_exp_node(input, None).unwrap();
        weak_exp = Rc::downgrade(&exp);

        assert!(weak_exp.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_exp.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
