/*
 * @Author       : 老董
 * @Description  : LeakyReLU 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 Graph + Var API）
 * 2. VJP 单元测试（calc_grad_to_parent_index）→ 底层 NodeInner
 * 3. 端到端反向传播测试（高层 Graph + Var API）
 * 4. 梯度累积测试（高层 Graph + Var API）
 * 5. 动态形状测试（高层 Graph + Var API）
 * 6. 节点创建 API 测试（底层 create_* API）
 *
 * 梯度公式：
 *   leaky_relu(x, slope) = x if x > 0 else slope * x
 *   ReLU 是 slope=0 的特例
 *   dy/dx = 1 if x > 0 else slope
 *   VJP: grad_to_parent = upstream_grad * (1 if x > 0 else slope)
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 ReLU (slope=0) 前向传播
/// input=[0.5, -1.0, 0.0, 2.0] → output=[0.5, 0.0, 0.0, 2.0]
#[test]
fn test_relu_forward() {
    let graph = Graph::new();

    let x = graph
        .parameter(&[2, 2], Init::Zeros, "x")
        .unwrap();
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

/// 测试 LeakyReLU (slope=0.1) 前向传播
/// input=[0.5, -1.0, 0.0, 2.0] → output=[0.5, -0.1, 0.0, 2.0]
#[test]
fn test_leaky_relu_forward() {
    let graph = Graph::new();

    let x = graph
        .parameter(&[2, 2], Init::Zeros, "x")
        .unwrap();
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))
        .unwrap();

    let result = x.leaky_relu(0.1);
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], -0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 2.0, epsilon = 1e-6);
}

/// 测试 LeakyReLU 节点不能直接设置值
#[test]
fn test_leaky_relu_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.leaky_relu(0.1);

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "LeakyReLU 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证梯度计算公式。
// dy/dx = 1 if x > 0 else slope

/// 测试 ReLU VJP (slope=0)
/// input=[0.5, -1.0, 0.0, 2.0] → grad=[1, 0, 0, 1]
#[test]
fn test_relu_vjp() -> Result<(), GraphError> {
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
    let grad = relu.calc_grad_to_parent_index(0, &upstream_grad)?;

    // grad = upstream * (1 if x > 0 else 0) = [1, 0, 0, 1]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 1.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 LeakyReLU VJP (slope=0.1) with unit upstream
/// input=[0.5, -1.0, 0.0, 2.0] → grad=[1.0, 0.1, 0.1, 1.0]
#[test]
fn test_leaky_relu_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let relu = inner
        .borrow_mut()
        .create_leaky_relu_node(x.clone(), 0.1, Some("relu"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    relu.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = relu.calc_grad_to_parent_index(0, &upstream_grad)?;

    // grad = upstream * (1 if x > 0 else 0.1) = [1, 0.1, 0.1, 1]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 1.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 LeakyReLU VJP with non-unit upstream
/// upstream=[2,3,4,5], input=[0.5,-1.0,0.0,2.0] → grad=[2, 0.3, 0.4, 5]
#[test]
fn test_leaky_relu_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let relu = inner
        .borrow_mut()
        .create_leaky_relu_node(x.clone(), 0.1, Some("relu"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    relu.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = relu.calc_grad_to_parent_index(0, &upstream_grad)?;

    // grad = upstream * (1 if x > 0 else 0.1) = [2*1, 3*0.1, 4*0.1, 5*1] = [2, 0.3, 0.4, 5]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.3, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 0.4, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 5.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 LeakyReLU VJP：全正值输入 → 梯度全为 1
#[test]
fn test_leaky_relu_vjp_all_positive() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let relu = inner
        .borrow_mut()
        .create_leaky_relu_node(x.clone(), 0.1, Some("relu"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    relu.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = relu.calc_grad_to_parent_index(0, &upstream_grad)?;

    assert_eq!(&grad, &Tensor::ones(&[2, 2]));

    Ok(())
}

/// 测试 LeakyReLU VJP：全负值输入 → 梯度全为 slope
#[test]
fn test_leaky_relu_vjp_all_negative() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let relu = inner
        .borrow_mut()
        .create_leaky_relu_node(x.clone(), 0.1, Some("relu"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[-1.0, -2.0, -3.0, -4.0], &[2, 2])))
        .unwrap();
    relu.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = relu.calc_grad_to_parent_index(0, &upstream_grad)?;

    let expected = Tensor::new(&[0.1, 0.1, 0.1, 0.1], &[2, 2]);
    assert_eq!(&grad, &expected);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 LeakyReLU 端到端反向传播
/// result = leaky_relu(input) = [0.5, -0.1, 0.0, 2.0], target = zeros
/// loss = mean([0.25, 0.01, 0.0, 4.0]) = 1.065
#[test]
fn test_leaky_relu_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.leaky_relu(0.1);
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 1.065, epsilon = 1e-5);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = result/2 = [0.25, -0.05, 0.0, 1.0]
    // ∂loss/∂input = ∂loss/∂result * leaky_relu'(x) = [0.25, -0.005, 0.0, 1.0]
    assert_abs_diff_eq!(x_grad[[0, 0]], 0.25, epsilon = 1e-5);
    assert_abs_diff_eq!(x_grad[[0, 1]], -0.005, epsilon = 1e-5);
    assert_abs_diff_eq!(x_grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x_grad[[1, 1]], 1.0, epsilon = 1e-5);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试 LeakyReLU 梯度累积
#[test]
fn test_leaky_relu_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.leaky_relu(0.1);
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
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

/// 测试 LeakyReLU 节点的动态形状传播
#[test]
fn test_leaky_relu_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    let result = h0.leaky_relu(0.1);

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 LeakyReLU 节点在不同 batch_size 下的前向计算
#[test]
fn test_leaky_relu_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    let result = h0.leaky_relu(0.1);

    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 16], "第一次 forward: batch=2");

    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();

    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 16], "第二次 forward: batch=8");
}

/// 测试 LeakyReLU 节点在不同 batch_size 下的反向传播
#[test]
fn test_leaky_relu_dynamic_batch_backward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]

    let result = h0.leaky_relu(0.1);

    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    x.set_value(&Tensor::zeros(&[6, 8])).unwrap();
    target.set_value(&Tensor::zeros(&[6, 4])).unwrap();

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
fn test_create_leaky_relu_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();

    let leaky_relu = inner
        .borrow_mut()
        .create_leaky_relu_node(input.clone(), 0.1, Some("leaky_relu"))
        .unwrap();

    assert_eq!(leaky_relu.shape(), vec![2, 3]);
    assert_eq!(leaky_relu.name(), Some("leaky_relu"));
    assert!(!leaky_relu.is_leaf());
    assert_eq!(leaky_relu.parents().len(), 1);
}

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
fn test_create_leaky_relu_node_negative_slope_error() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_leaky_relu_node(input, -0.1, None);

    assert!(result.is_err());
    match result.unwrap_err() {
        GraphError::InvalidOperation(msg) => {
            assert!(msg.contains("negative_slope应为非负数"));
        }
        _ => panic!("应该返回 InvalidOperation 错误"),
    }
}

#[test]
fn test_create_leaky_relu_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input_2d = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 10], None)
        .unwrap();
    let relu_2d = inner
        .borrow_mut()
        .create_leaky_relu_node(input_2d, 0.01, None)
        .unwrap();
    assert_eq!(relu_2d.shape(), vec![3, 10]);

    let input_4d = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4, 5], None)
        .unwrap();
    let relu_4d = inner
        .borrow_mut()
        .create_leaky_relu_node(input_4d, 0.2, None)
        .unwrap();
    assert_eq!(relu_4d.shape(), vec![2, 3, 4, 5]);
}

#[test]
fn test_create_leaky_relu_node_drop_releases() {
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
            .create_leaky_relu_node(input, 0.1, None)
            .unwrap();
        weak_relu = Rc::downgrade(&relu);

        assert!(weak_relu.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_relu.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
