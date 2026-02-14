/*
 * @Author       : 老董
 * @Description  : Swish/SiLU 节点单元测试
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
 *   swish(x) = x * sigmoid(x)
 *   swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
 *            = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
 *   VJP: grad_to_parent = upstream_grad ⊙ swish'(x)
 *
 * Python 对照值（PyTorch torch.nn.functional.silu）：
 *   swish([0.5, -1.0, 0.0, 2.0]) = [0.3112, -0.2689, 0.0, 1.7616]
 *   swish'([0.5, -1.0, 0.0, 2.0]) = [0.7400, 0.0723, 0.5000, 1.0908]
 *
 * Python 对照脚本: tests/python/calc_jacobi_by_pytorch/node_swish.py
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试 ====================

/// 测试 Swish 前向传播（PyTorch 对照值）
///
/// swish([0.5, -1.0, 0.0, 2.0]) ≈ [0.3112, -0.2689, 0.0, 1.7616]
#[test]
fn test_swish_forward() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))
        .unwrap();
    let result = x.swish();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.3112, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[0, 1]], -0.2689, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 1.7616, epsilon = 1e-4);
}

/// 测试 Swish 前向传播（边界值）
#[test]
fn test_swish_forward_edge_cases() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[0.0, 10.0, -10.0, 0.001], &[1, 4]))
        .unwrap();
    let result = x.swish();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-6); // swish(0) = 0
    assert_abs_diff_eq!(output[[0, 1]], 10.0, epsilon = 1e-3); // swish(10) ≈ 10
    assert_abs_diff_eq!(output[[0, 2]], 0.0, epsilon = 1e-3); // swish(-10) ≈ 0
}

/// 测试 Swish 节点不能直接设置值
#[test]
fn test_swish_cannot_set_value() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.swish();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Swish 节点不应支持直接设值");
}

/// 测试 SiLU 是 Swish 的别名，结果一致
#[test]
fn test_silu_equals_swish() {
    let graph = Graph::new();
    let x_swish = graph
        .input(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))
        .unwrap();
    let result_swish = x_swish.swish();
    result_swish.forward().unwrap();
    let out_swish = result_swish.value().unwrap().unwrap();

    let graph2 = Graph::new();
    let x_silu = graph2
        .input(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))
        .unwrap();
    let result_silu = x_silu.silu();
    result_silu.forward().unwrap();
    let out_silu = result_silu.value().unwrap().unwrap();

    assert_eq!(out_swish.shape(), out_silu.shape());
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(out_swish[[i, j]], out_silu[[i, j]], epsilon = 1e-6);
        }
    }
}

// ==================== VJP 单元测试 ====================

/// 测试 Swish VJP（全 1 上游梯度，PyTorch 对照值）
///
/// swish'([0.5, -1.0, 0.0, 2.0]) ≈ [0.7400, 0.0723, 0.5000, 1.0908]
#[test]
fn test_swish_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let swish = inner
        .borrow_mut()
        .create_swish_node(x.clone(), Some("swish"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    swish.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = swish.calc_grad_to_parent_index(0, &upstream_grad)?;

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 0.7400, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0723, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[1, 0]], 0.5000, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[1, 1]], 1.0908, epsilon = 1e-3);

    Ok(())
}

/// 测试 Swish VJP（非单位上游梯度）
#[test]
fn test_swish_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let swish = inner
        .borrow_mut()
        .create_swish_node(x.clone(), Some("swish"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    swish.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = swish.calc_grad_to_parent_index(0, &upstream_grad)?;

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0 * 0.7400, epsilon = 1e-2);
    assert_abs_diff_eq!(grad[[0, 1]], 3.0 * 0.0723, epsilon = 1e-2);
    assert_abs_diff_eq!(grad[[1, 0]], 4.0 * 0.5000, epsilon = 1e-2);
    assert_abs_diff_eq!(grad[[1, 1]], 5.0 * 1.0908, epsilon = 1e-2);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 Swish 端到端反向传播：result = swish(x) -> loss = MSE(result, target)
#[test]
fn test_swish_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.swish();
    let target = graph.input(&Tensor::new(&[0.5, 0.0, 0.0, 2.0], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    // loss 应为有限正数
    assert!(loss_val >= 0.0);
    assert!(loss_val.is_finite());

    let input_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 2]);

    // swish(0) = 0，target = 0，diff = 0，对应 grad 应为 0
    assert_abs_diff_eq!(input_grad[[1, 0]], 0.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 梯度累积测试 ====================

/// 测试 Swish 梯度累积 + zero_grad
#[test]
fn test_swish_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.swish();
    let target = graph.input(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

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

// ==================== 动态形状测试 ====================

/// 测试 Swish 节点的动态形状传播
#[test]
fn test_swish_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap();

    use crate::nn::var::ops::VarActivationOps;
    let result = h0.swish();

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16));
}

/// 测试 Swish 节点在不同 batch_size 下的前向计算
#[test]
fn test_swish_dynamic_batch_forward() {
    use crate::nn::var::ops::VarActivationOps;

    let graph = Graph::new();
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap();
    let result = h0.swish();

    result.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 16]);

    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();
    result.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[8, 16]);
}

/// 测试 Swish 节点在不同 batch_size 下的反向传播
#[test]
fn test_swish_dynamic_batch_backward() {
    use crate::nn::var::ops::{VarActivationOps, VarLossOps};

    let graph = Graph::new();
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap();
    let result = h0.swish();
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    x.set_value(&Tensor::zeros(&[6, 8])).unwrap();
    target.set_value(&Tensor::zeros(&[6, 4])).unwrap();
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[6, 4]);
    loss.backward().unwrap();
}

// ==================== 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_swish_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();
    let swish = inner
        .borrow_mut()
        .create_swish_node(input.clone(), Some("swish"))
        .unwrap();

    assert_eq!(swish.shape(), vec![2, 3]);
    assert_eq!(swish.name(), Some("swish"));
    assert!(!swish.is_leaf());
    assert_eq!(swish.parents().len(), 1);
}

#[test]
fn test_create_swish_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input_2d = inner.borrow_mut().create_basic_input_node(&[3, 10], None).unwrap();
    let swish_2d = inner.borrow_mut().create_swish_node(input_2d, None).unwrap();
    assert_eq!(swish_2d.shape(), vec![3, 10]);

    let input_3d = inner.borrow_mut().create_basic_input_node(&[2, 3, 4], None).unwrap();
    let swish_3d = inner.borrow_mut().create_swish_node(input_3d, None).unwrap();
    assert_eq!(swish_3d.shape(), vec![2, 3, 4]);
}

#[test]
fn test_create_swish_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_swish;
    let weak_input;
    {
        let input = inner.borrow_mut().create_basic_input_node(&[2, 3], None).unwrap();
        weak_input = Rc::downgrade(&input);
        let swish = inner.borrow_mut().create_swish_node(input, None).unwrap();
        weak_swish = Rc::downgrade(&swish);
        assert!(weak_swish.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_swish.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
