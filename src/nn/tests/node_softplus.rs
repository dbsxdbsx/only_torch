/*
 * SoftPlus 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试 → 高层 Graph + Var API
 * 2. VJP 单元测试（calc_grad_to_parent_index）→ 底层 NodeInner
 * 3. 端到端反向传播测试 → 高层 Graph + Var API
 * 4. 梯度累积测试 → 高层 Graph + Var API
 * 5. 动态形状测试 → 高层 Graph + Var API
 * 6. 节点创建 API 测试 → 底层 create_* API
 *
 * 梯度公式：
 *   softplus(x) = ln(1 + e^x)
 *   d(softplus)/dx = sigmoid(x) = 1/(1+e^(-x))
 *   VJP: grad_to_parent = upstream ⊙ sigmoid(x)
 *
 * 预期值来自 tests/python/calc_jacobi_by_pytorch/node_softplus.py
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 SoftPlus 前向传播（2D 输入）
#[test]
fn test_softplus_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]))
        .unwrap();
    let result = x.softplus();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    // softplus([[-1, 0, 1], [2, -2, 0.5]])
    // = [[0.31326169, 0.69314718, 1.31326163], [2.12692809, 0.12692800, 0.97407699]]
    assert_abs_diff_eq!(output[[0, 0]], 0.31326169, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 1]], 0.69314718, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 2]], 1.31326163, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[1, 0]], 2.12692809, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[1, 1]], 0.12692800, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[1, 2]], 0.97407699, epsilon = 1e-5);
}

/// 测试 SoftPlus 数值稳定性（极端值）
#[test]
fn test_softplus_numerical_stability() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[-50.0, -20.0, 0.0, 20.0, 50.0], &[1, 5]))
        .unwrap();
    let result = x.softplus();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    // softplus(-50) ≈ 0, softplus(-20) ≈ 0
    assert!(output[[0, 0]] < 1e-10, "softplus(-50) should be ≈ 0");
    assert!(output[[0, 1]] < 1e-5, "softplus(-20) should be ≈ 0");
    // softplus(0) = ln(2) ≈ 0.693
    assert_abs_diff_eq!(output[[0, 2]], 0.69314718, epsilon = 1e-5);
    // softplus(20) ≈ 20, softplus(50) ≈ 50
    assert_abs_diff_eq!(output[[0, 3]], 20.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 4]], 50.0, epsilon = 1e-5);
}

/// 测试 SoftPlus 节点不能直接设置值
#[test]
fn test_softplus_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let result = x.softplus();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "SoftPlus 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================

/// 测试 SoftPlus VJP（单位上游梯度）
///
/// VJP: grad = upstream ⊙ sigmoid(x)，单位 upstream 时 grad = sigmoid(x)
#[test]
fn test_softplus_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 5], Some("x"))
        .unwrap();
    let sp = inner
        .borrow_mut()
        .create_softplus_node(x.clone(), Some("sp"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5])))
        .unwrap();
    sp.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[1, 5]);
    let grad = sp
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    // grad = sigmoid(x) = [0.11920292, 0.26894143, 0.5, 0.73105860, 0.88079708]
    let expected = [0.11920292, 0.26894143, 0.5, 0.73105860, 0.88079708];
    assert_eq!(grad.shape(), &[1, 5]);
    for i in 0..5 {
        assert_abs_diff_eq!(grad[[0, i]], expected[i], epsilon = 1e-5);
    }

    Ok(())
}

/// 测试 SoftPlus VJP（非单位上游梯度）
#[test]
fn test_softplus_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let sp = inner
        .borrow_mut()
        .create_softplus_node(x.clone(), Some("sp"))
        .unwrap();

    x.set_value(Some(&Tensor::new(
        &[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5],
        &[2, 3],
    )))
    .unwrap();
    sp.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 3.0, 1.0, 0.5, 4.0, 2.0], &[2, 3]);
    let grad = sp
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    // sigmoid(x) = [0.26894143, 0.5, 0.73105860, 0.88079709, 0.11920292, 0.62245935]
    // grad = upstream ⊙ sigmoid
    let sigmoid_vals = [
        0.26894143, 0.5, 0.73105860, 0.88079709, 0.11920292, 0.62245935,
    ];
    let upstream_vals = [2.0, 3.0, 1.0, 0.5, 4.0, 2.0];
    assert_eq!(grad.shape(), &[2, 3]);
    for i in 0..6 {
        let expected = sigmoid_vals[i] * upstream_vals[i];
        assert_abs_diff_eq!(grad.data_as_slice()[i], expected, epsilon = 1e-5);
    }

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 SoftPlus 端到端反向传播：result = softplus(x) → loss = MSE(result, target)
#[test]
fn test_softplus_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]))?;

    let result = x.softplus();
    let target = graph.input(&Tensor::zeros(&[2, 3]))?;
    let loss = result.mse_loss(&target)?;

    // 前向传播
    loss.forward().unwrap();

    // 反向传播
    graph.zero_grad()?;
    let loss_returned = loss.backward()?;
    assert!(loss_returned > 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 3]);

    // ∂loss/∂x = 2 * softplus * sigmoid / n
    let softplus_vals = [
        0.31326169, 0.69314718, 1.31326163, 2.12692809, 0.12692800, 0.97407699,
    ];
    let sigmoid_vals = [
        0.26894143, 0.5, 0.73105860, 0.88079709, 0.11920292, 0.62245935,
    ];
    let n = 6.0;
    for i in 0..6 {
        let expected = 2.0 * softplus_vals[i] * sigmoid_vals[i] / n;
        assert_abs_diff_eq!(x_grad.data_as_slice()[i], expected, epsilon = 1e-5);
    }

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试 SoftPlus 梯度累积
#[test]
fn test_softplus_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]))?;

    let result = x.softplus();
    let target = graph.input(&Tensor::zeros(&[2, 3]))?;
    let loss = result.mse_loss(&target)?;

    loss.forward().unwrap();

    // 第 1 次反向传播
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = x.grad()?.unwrap().clone();

    // 第 2 次反向传播（梯度累积）
    loss.forward().unwrap();
    loss.backward()?;
    let grad_second = x.grad()?.unwrap();
    assert_eq!(grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    loss.forward().unwrap();
    loss.backward()?;
    let grad_after_clear = x.grad()?.unwrap();
    assert_eq!(grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 SoftPlus 节点的动态形状传播
#[test]
fn test_softplus_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    let result = h0.softplus();

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 SoftPlus 节点在不同 batch_size 下的前向计算
#[test]
fn test_softplus_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    let result = h0.softplus();

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 16], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 16], "第二次 forward: batch=8");
}

/// 测试 SoftPlus 节点在不同 batch_size 下的反向传播
#[test]
fn test_softplus_dynamic_batch_backward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]

    let result = h0.softplus();

    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[6, 8])).unwrap();
    target.set_value(&Tensor::zeros(&[6, 4])).unwrap();

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

use std::rc::Rc;

#[test]
fn test_create_softplus_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();

    let softplus = inner
        .borrow_mut()
        .create_softplus_node(input.clone(), Some("softplus"))
        .unwrap();

    assert_eq!(softplus.shape(), vec![2, 3]);
    assert_eq!(softplus.name(), Some("softplus"));
    assert!(!softplus.is_leaf());
    assert_eq!(softplus.parents().len(), 1);
}

#[test]
fn test_create_softplus_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 测试各种形状都正确保留（节点必须是 2-4 维）
    let input_2d = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 10], None)
        .unwrap();
    let sp_2d = inner
        .borrow_mut()
        .create_softplus_node(input_2d, None)
        .unwrap();
    assert_eq!(sp_2d.shape(), vec![3, 10]);

    let input_4d = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4, 5], None)
        .unwrap();
    let sp_4d = inner
        .borrow_mut()
        .create_softplus_node(input_4d, None)
        .unwrap();
    assert_eq!(sp_4d.shape(), vec![2, 3, 4, 5]);
}

#[test]
fn test_create_softplus_node_chain() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 测试连续两个 softplus
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], None)
        .unwrap();
    let sp1 = inner
        .borrow_mut()
        .create_softplus_node(input, None)
        .unwrap();
    let sp2 = inner
        .borrow_mut()
        .create_softplus_node(sp1.clone(), None)
        .unwrap();

    assert_eq!(sp2.shape(), vec![2, 2]);
    assert_eq!(sp2.parents().len(), 1);
}

#[test]
fn test_create_softplus_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_softplus;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let softplus = inner
            .borrow_mut()
            .create_softplus_node(input, None)
            .unwrap();
        weak_softplus = Rc::downgrade(&softplus);

        assert!(weak_softplus.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_softplus.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
