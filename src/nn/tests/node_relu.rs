/*
 * @Author       : 老董
 * @Description  : ReLU 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 Graph + Var API）
 * 2. VJP 单元测试（calc_grad_to_parent_index）→ 底层 NodeInner
 * 3. 端到端反向传播测试（高层 Graph + Var API）
 * 4. 动态形状测试（高层 Graph + Var API）
 * 5. 节点创建 API 测试（底层 create_* API）
 *
 * 梯度公式：
 *   relu(x) = max(0, x)
 *   dy/dx = 1 if x > 0 else 0
 *   VJP: grad_to_parent = upstream_grad * (1 if x > 0 else 0)
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 ReLU 前向传播
/// input=[0.5, -1.0, 0.0, 2.0] → output=[0.5, 0.0, 0.0, 2.0]
#[test]
fn test_relu_forward() {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x").unwrap();
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))
        .unwrap();

    let result = x.relu();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 2.0, epsilon = 1e-6);
}

/// 测试 ReLU 节点不能直接设置值
#[test]
fn test_relu_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.relu();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "ReLU 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================

/// 测试 ReLU VJP，单位上游梯度
/// input=[0.5, -1.0, 0.0, 2.0] → grad=[1, 0, 0, 1]
#[test]
fn test_relu_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let relu = inner
        .borrow_mut()
        .create_relu_node(x.clone(), Some("relu"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    relu.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = relu.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 1.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 ReLU VJP，非单位上游梯度
/// upstream=[2,3,4,5], input=[0.5,-1.0,0.0,2.0] → grad=[2, 0, 0, 5]
#[test]
fn test_relu_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let relu = inner
        .borrow_mut()
        .create_relu_node(x.clone(), Some("relu"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    relu.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = relu.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    // grad = upstream * (1 if x > 0 else 0) = [2*1, 3*0, 4*0, 5*1] = [2, 0, 0, 5]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 5.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 ReLU VJP：全正值输入 → 梯度全为 1
#[test]
fn test_relu_vjp_all_positive() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let relu = inner
        .borrow_mut()
        .create_relu_node(x.clone(), Some("relu"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    relu.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = relu.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(&grad, &Tensor::ones(&[2, 2]));

    Ok(())
}

/// 测试 ReLU VJP：全负值输入 → 梯度全为 0
#[test]
fn test_relu_vjp_all_negative() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let relu = inner
        .borrow_mut()
        .create_relu_node(x.clone(), Some("relu"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[-1.0, -2.0, -3.0, -4.0], &[2, 2])))
        .unwrap();
    relu.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = relu.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(&grad, &Tensor::zeros(&[2, 2]));

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 ReLU 端到端反向传播
/// result = relu(input) = [0.5, 0.0, 0.0, 2.0], target = zeros
/// loss = mean([0.25, 0.0, 0.0, 4.0]) = 1.0625
#[test]
fn test_relu_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.relu();
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 1.0625, epsilon = 1e-5);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = result/2 = [0.25, 0.0, 0.0, 1.0]
    // ∂loss/∂input = ∂loss/∂result * relu'(x) = [0.25*1, 0.0*0, 0.0*0, 1.0*1] = [0.25, 0.0, 0.0, 1.0]
    assert_abs_diff_eq!(x_grad[[0, 0]], 0.25, epsilon = 1e-5);
    assert_abs_diff_eq!(x_grad[[0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x_grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x_grad[[1, 1]], 1.0, epsilon = 1e-5);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 ReLU 节点的动态形状传播
#[test]
fn test_relu_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    let result = h0.relu();

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 ReLU 节点在不同 batch_size 下的前向计算
#[test]
fn test_relu_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    let result = h0.relu();

    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 16], "第一次 forward: batch=2");

    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();

    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 16], "第二次 forward: batch=8");
}

// ==================== 节点创建 API 测试 ====================

#[test]
fn test_create_relu_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();

    let relu = inner
        .borrow_mut()
        .create_relu_node(input.clone(), Some("relu"))
        .unwrap();

    assert_eq!(relu.shape(), vec![2, 3]);
    assert_eq!(relu.name(), Some("relu"));
    assert!(!relu.is_leaf());
}

#[test]
fn test_create_relu_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input_2d = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 10], None)
        .unwrap();
    let relu_2d = inner
        .borrow_mut()
        .create_relu_node(input_2d, None)
        .unwrap();
    assert_eq!(relu_2d.shape(), vec![3, 10]);

    let input_4d = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4, 5], None)
        .unwrap();
    let relu_4d = inner
        .borrow_mut()
        .create_relu_node(input_4d, None)
        .unwrap();
    assert_eq!(relu_4d.shape(), vec![2, 3, 4, 5]);
}

#[test]
fn test_create_relu_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_relu;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let relu = inner
            .borrow_mut()
            .create_relu_node(input, None)
            .unwrap();
        weak_relu = Rc::downgrade(&relu);

        assert!(weak_relu.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_relu.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
