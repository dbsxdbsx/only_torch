/*
 * @Author       : 老董
 * @Description  : SELU 节点单元测试
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
 *   LAMBDA = 1.0507009873554804934193349852946, ALPHA = 1.6732632423543772848170429916717
 *   selu(x) = LAMBDA * x if x > 0, else LAMBDA * ALPHA * (exp(x) - 1)
 *   selu'(x) = LAMBDA if x > 0, else LAMBDA * ALPHA * exp(x)
 *   VJP: grad_to_parent = upstream_grad ⊙ selu'(x)
 *
 * Python 对照值（PyTorch torch.nn.functional.selu）：
 *   selu([0.5, -1.0, 0.0, 2.0]) = [0.5254, -1.1113, 0.0, 2.1014]
 *   selu'([0.5, -1.0, 0.0, 2.0]) = [1.0507, 0.6466, 1.7581, 1.0507]
 *
 * Python 对照脚本: tests/python/calc_jacobi_by_pytorch/node_selu.py
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试 ====================

/// 测试 SELU 前向传播（PyTorch 对照值）
///
/// selu([0.5, -1.0, 0.0, 2.0]) ≈ [0.5254, -1.1113, 0.0, 2.1014]
#[test]
fn test_selu_forward() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))
        .unwrap();
    let result = x.selu();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.5254, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[0, 1]], -1.1113, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 2.1014, epsilon = 1e-4);
}

/// 测试 SELU 前向传播（边界值）
#[test]
fn test_selu_forward_edge_cases() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[0.0, 10.0, -10.0, 0.001], &[1, 4]))
        .unwrap();
    let result = x.selu();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-6); // selu(0) = 0
    assert_abs_diff_eq!(output[[0, 1]], 10.507, epsilon = 1e-3); // selu(10) ≈ LAMBDA * 10
    assert_abs_diff_eq!(output[[0, 2]], -1.7580, epsilon = 1e-3); // selu(-10) ≈ -LAMBDA*ALPHA
}

/// 测试 SELU 节点不能直接设置值
#[test]
fn test_selu_cannot_set_value() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.selu();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "SELU 节点不应支持直接设值");
}

// ==================== VJP 单元测试 ====================

/// 测试 SELU VJP（全 1 上游梯度，PyTorch 对照值）
///
/// selu'([0.5, -1.0, 0.0, 2.0]) ≈ [1.0507, 0.6466, 1.7581, 1.0507]
#[test]
fn test_selu_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let selu = inner
        .borrow_mut()
        .create_selu_node(x.clone(), Some("selu"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    selu.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = selu
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0507, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[0, 1]], 0.6466, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[1, 0]], 1.7581, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[1, 1]], 1.0507, epsilon = 1e-3);

    Ok(())
}

/// 测试 SELU VJP（非单位上游梯度）
#[test]
fn test_selu_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let selu = inner
        .borrow_mut()
        .create_selu_node(x.clone(), Some("selu"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    selu.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = selu
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0 * 1.0507, epsilon = 1e-2);
    assert_abs_diff_eq!(grad[[0, 1]], 3.0 * 0.6466, epsilon = 1e-2);
    assert_abs_diff_eq!(grad[[1, 0]], 4.0 * 1.7581, epsilon = 1e-2);
    assert_abs_diff_eq!(grad[[1, 1]], 5.0 * 1.0507, epsilon = 1e-2);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 SELU 端到端反向传播：result = selu(x) -> loss = MSE(result, target)
#[test]
fn test_selu_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.selu();
    let target = graph.input(&Tensor::new(&[0.5, 0.0, 0.0, 2.0], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    // loss 应为有限正数
    assert!(loss_val >= 0.0);
    assert!(loss_val.is_finite());

    let input_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 2]);

    // selu(0) = 0，target = 0，diff = 0，对应 grad 应为 0
    assert_abs_diff_eq!(input_grad[[1, 0]], 0.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 梯度累积测试 ====================

/// 测试 SELU 梯度累积 + zero_grad
#[test]
fn test_selu_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.selu();
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

/// 测试 SELU 节点的动态形状传播
#[test]
fn test_selu_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap();

    use crate::nn::var::ops::VarActivationOps;
    let result = h0.selu();

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16));
}

/// 测试 SELU 节点在不同 batch_size 下的前向计算
#[test]
fn test_selu_dynamic_batch_forward() {
    use crate::nn::var::ops::VarActivationOps;

    let graph = Graph::new();
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap();
    let result = h0.selu();

    result.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 16]);

    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();
    result.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[8, 16]);
}

/// 测试 SELU 节点在不同 batch_size 下的反向传播
#[test]
fn test_selu_dynamic_batch_backward() {
    use crate::nn::var::ops::{VarActivationOps, VarLossOps};

    let graph = Graph::new();
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap();
    let result = h0.selu();
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
fn test_create_selu_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();
    let selu = inner
        .borrow_mut()
        .create_selu_node(input.clone(), Some("selu"))
        .unwrap();

    assert_eq!(selu.shape(), vec![2, 3]);
    assert_eq!(selu.name(), Some("selu"));
    assert!(!selu.is_leaf());
    assert_eq!(selu.parents().len(), 1);
}

#[test]
fn test_create_selu_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input_2d = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 10], None)
        .unwrap();
    let selu_2d = inner.borrow_mut().create_selu_node(input_2d, None).unwrap();
    assert_eq!(selu_2d.shape(), vec![3, 10]);

    let input_3d = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4], None)
        .unwrap();
    let selu_3d = inner.borrow_mut().create_selu_node(input_3d, None).unwrap();
    assert_eq!(selu_3d.shape(), vec![2, 3, 4]);
}

#[test]
fn test_create_selu_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_selu;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);
        let selu = inner.borrow_mut().create_selu_node(input, None).unwrap();
        weak_selu = Rc::downgrade(&selu);
        assert!(weak_selu.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_selu.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
