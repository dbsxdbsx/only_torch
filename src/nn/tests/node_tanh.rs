/*
 * @Author       : 老董
 * @Description  : Tanh 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ basic forward + edge cases + cannot_set_value
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ unit upstream + non-unit + saturation
 * 3. 端到端反向传播测试（高层 API）→ MSE loss + 数值验证
 * 4. 梯度累积测试（高层 API）
 * 5. 动态形状测试（KEEP AS-IS）
 * 6. Create API 测试（KEEP AS-IS）
 *
 * 梯度公式：
 *   tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
 *   dy/dx = 1 - tanh²(x) = 1 - y²
 *   VJP: grad_to_parent = upstream_grad * (1 - y²)
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（高层 API）====================

/// 测试 Tanh 前向传播
#[test]
fn test_tanh_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))
        .unwrap();
    let result = x.tanh();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.46211716, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], -0.76159418, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 0.96402758, epsilon = 1e-6);
}

/// 测试 Tanh 前向传播（边界值）
#[test]
fn test_tanh_forward_edge_cases() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[0.0, 10.0, -10.0, 0.001], &[1, 4]))
        .unwrap();
    let result = x.tanh();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-6); // tanh(0) = 0
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-6); // tanh(10) ≈ 1
    assert_abs_diff_eq!(output[[0, 2]], -1.0, epsilon = 1e-6); // tanh(-10) ≈ -1
    assert_abs_diff_eq!(output[[0, 3]], 0.001, epsilon = 1e-5); // tanh(x) ≈ x for small x
}

/// 测试 Tanh 节点不能直接设置值
#[test]
fn test_tanh_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))
        .unwrap();
    let result = x.tanh();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Tanh 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================

/// 测试 Tanh VJP（单位上游梯度）
///
/// grad = upstream_grad * (1 - tanh²)
/// tanh([0.5, -1.0, 0.0, 2.0]) = [0.46211716, -0.76159418, 0.0, 0.96402758]
/// 1 - tanh² = [0.78644770, 0.41997433, 1.0, 0.07065082]
#[test]
fn test_tanh_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let tanh_node = inner
        .borrow_mut()
        .create_tanh_node(x.clone(), Some("tanh"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    tanh_node.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = tanh_node
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 0.78644770, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.41997433, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 0.07065082, epsilon = 1e-6);

    Ok(())
}

/// 测试 Tanh VJP（非单位上游梯度）
#[test]
fn test_tanh_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let tanh_node = inner
        .borrow_mut()
        .create_tanh_node(x.clone(), Some("tanh"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    tanh_node.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = tanh_node
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    // grad = upstream_grad * (1 - tanh²)
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0 * 0.78644770, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 3.0 * 0.41997433, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0]], 4.0 * 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1]], 5.0 * 0.07065082, epsilon = 1e-5);

    Ok(())
}

/// 测试 Tanh VJP（饱和区：大 |x| → 梯度 ≈ 0）
#[test]
fn test_tanh_vjp_saturation() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("x"))
        .unwrap();
    let tanh_node = inner
        .borrow_mut()
        .create_tanh_node(x.clone(), Some("tanh"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[5.0, -5.0], &[1, 2])))
        .unwrap();
    tanh_node.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[1, 2]);
    let grad = tanh_node
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-3);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 API）====================

/// 测试 Tanh 端到端反向传播：result = tanh(input) → loss = MSE(result, target)
///
/// input = [0.5, -1.0, 0.0, 2.0], target = zeros
/// result = tanh(input) = [0.46211716, -0.76159418, 0.0, 0.96402758]
/// loss = mean(result²) = mean([0.2135, 0.5800, 0.0, 0.9293])
#[test]
fn test_tanh_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.tanh();
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    let expected_loss = (0.46211716_f32.powi(2)
        + 0.76159418_f32.powi(2)
        + 0.0_f32.powi(2)
        + 0.96402758_f32.powi(2))
        / 4.0;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, expected_loss, epsilon = 1e-5);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = result/2
    // ∂loss/∂input = ∂loss/∂result * (1 - tanh²)
    assert_abs_diff_eq!(x_grad[[0, 0]], 0.18165, epsilon = 1e-4);
    assert_abs_diff_eq!(x_grad[[0, 1]], -0.15993, epsilon = 1e-4);
    assert_abs_diff_eq!(x_grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x_grad[[1, 1]], 0.03409, epsilon = 1e-4);

    Ok(())
}

// ==================== 梯度累积测试（高层 API）====================

/// 测试 Tanh 梯度累积
#[test]
fn test_tanh_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.tanh();
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

/// 测试 Tanh 节点的动态形状传播
#[test]
fn test_tanh_dynamic_shape_propagation() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建 Tanh: h0 -> tanh(h0) -> [?, 16]
    use crate::nn::var::ops::VarActivationOps;
    let result = h0.tanh();

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 Tanh 节点在不同 batch_size 下的前向计算
#[test]
fn test_tanh_dynamic_batch_forward() {
    use crate::nn::Graph;
    use crate::nn::var::ops::VarActivationOps;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // Tanh: h0 -> tanh(h0)
    let result = h0.tanh();

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

/// 测试 Tanh 节点在不同 batch_size 下的反向传播
#[test]
fn test_tanh_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var::ops::{VarActivationOps, VarLossOps};

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]

    // Tanh: h0 -> tanh(h0) -> [?, 4]
    let result = h0.tanh();

    // 创建目标和损失
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
fn test_create_tanh_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();

    let tanh = inner
        .borrow_mut()
        .create_tanh_node(input.clone(), Some("tanh"))
        .unwrap();

    assert_eq!(tanh.shape(), vec![2, 3]);
    assert_eq!(tanh.name(), Some("tanh"));
    assert!(!tanh.is_leaf());
    assert_eq!(tanh.parents().len(), 1);
}

#[test]
fn test_create_tanh_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 测试各种形状都正确保留（节点必须是 2-4 维）
    let input_2d = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 10], None)
        .unwrap();
    let tanh_2d = inner.borrow_mut().create_tanh_node(input_2d, None).unwrap();
    assert_eq!(tanh_2d.shape(), vec![3, 10]);

    let input_4d = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4, 5], None)
        .unwrap();
    let tanh_4d = inner.borrow_mut().create_tanh_node(input_4d, None).unwrap();
    assert_eq!(tanh_4d.shape(), vec![2, 3, 4, 5]);
}

#[test]
fn test_create_tanh_node_chain() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 测试连续两个 tanh
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], None)
        .unwrap();
    let t1 = inner.borrow_mut().create_tanh_node(input, None).unwrap();
    let t2 = inner
        .borrow_mut()
        .create_tanh_node(t1.clone(), None)
        .unwrap();

    assert_eq!(t2.shape(), vec![2, 2]);
    assert_eq!(t2.parents().len(), 1);
}

#[test]
fn test_create_tanh_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_tanh;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let tanh = inner.borrow_mut().create_tanh_node(input, None).unwrap();
        weak_tanh = Rc::downgrade(&tanh);

        assert!(weak_tanh.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_tanh.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
