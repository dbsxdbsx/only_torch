/*
 * @Author       : 老董
 * @Description  : LogSoftmax 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 Graph + Var API）→ basic [2,3] + 数值稳定性 + cannot_set_value
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ unit upstream 行和为 0；[0,0,1] 非零 grad
 * 3. 端到端反向传播测试（高层 Graph + Var API）
 * 4. 梯度累积测试（高层 Graph + Var API）
 * 5. 动态形状测试（已有）
 * 6. 新节点创建 API 测试（已有）
 *
 * 关键公式：
 *   LogSoftmax 需要 2D 输入。log_softmax(x) = log(softmax(x))。输出恒为负，exp(output) 每行和为 1。
 *   VJP: dL/dx_i = upstream_i - softmax_i * Σ(upstream)。当 upstream 均匀时，行梯度和为 0。
 *   高层 API: 使用 VarActivationOps 的 .log_softmax()
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 LogSoftmax 前向传播
///
/// [2,3] 输入，验证所有输出为负且 exp 每行和为 1
#[test]
fn test_log_softmax_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]))
        .unwrap();
    let result = x.log_softmax();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 3]);

    // log_softmax 输出应全为负数
    for i in 0..2 {
        for j in 0..3 {
            assert!(output[[i, j]] < 0.0, "log_softmax 输出应为负数");
        }
    }

    // exp(log_softmax) 每行和应为 1
    for i in 0..2 {
        let sum = output[[i, 0]].exp() + output[[i, 1]].exp() + output[[i, 2]].exp();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }
}

/// 测试 LogSoftmax 数值稳定性
///
/// 大数值 [1000, 1001, 1002] 不应溢出
#[test]
fn test_log_softmax_numerical_stability() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1000.0, 1001.0, 1002.0], &[1, 3]))
        .unwrap();
    let result = x.log_softmax();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();

    // 输出应为有限值
    for j in 0..3 {
        assert!(output[[0, j]].is_finite(), "输出应为有限值");
    }

    // exp 后和应为 1
    let sum = output[[0, 0]].exp() + output[[0, 1]].exp() + output[[0, 2]].exp();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
}

/// 测试 LogSoftmax 节点不能直接设置值
#[test]
fn test_log_softmax_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let result = x.log_softmax();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "LogSoftmax 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================
//
// 使用底层 create_log_softmax_node，通过 calc_grad_to_parent_index 直接验证梯度公式。
// VJP: dL/dx_i = upstream_i - softmax_i * Σ(upstream). 当 upstream 均匀时，行梯度和为 0。

/// 测试 LogSoftmax VJP（全 1 上游梯度）
///
/// unit upstream → 每行梯度和应为 0
#[test]
fn test_log_softmax_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let ls = inner
        .borrow_mut()
        .create_log_softmax_node(x.clone(), Some("ls"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3])))
        .unwrap();
    ls.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 3]);
    let grad = ls.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 3]);

    // 当 upstream 全为 1 时，sum(upstream_grad) = num_classes = 3
    // dL/dx_i = 1 - softmax_i * 3，sum(grad) = 3 - 3*sum(softmax) = 0
    let row0_sum = grad[[0, 0]] + grad[[0, 1]] + grad[[0, 2]];
    let row1_sum = grad[[1, 0]] + grad[[1, 1]] + grad[[1, 2]];
    assert_abs_diff_eq!(row0_sum, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(row1_sum, 0.0, epsilon = 1e-5);

    Ok(())
}

/// 测试 LogSoftmax VJP（非单位上游梯度）
///
/// [0, 0, 1] upstream → 梯度非零
#[test]
fn test_log_softmax_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("x"))
        .unwrap();
    let ls = inner
        .borrow_mut()
        .create_log_softmax_node(x.clone(), Some("ls"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    ls.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[0.0, 0.0, 1.0], &[1, 3]);
    let grad = ls.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[1, 3]);
    assert!(grad.data_as_slice().iter().any(|&g| g.abs() > 1e-6));

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 LogSoftmax 端到端反向传播：result = log_softmax(input) → loss = MSE(result, target)
#[test]
fn test_log_softmax_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "input")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))?;

    let result = x.log_softmax();
    let target = graph.input(&Tensor::new(&[-2.0, -1.0, 0.0, -1.1, -1.1, -1.1], &[2, 3]))?;
    let loss = result.mse_loss(&target)?;

    loss.forward().unwrap();
    graph.zero_grad()?;
    let loss_returned = loss.backward()?;
    assert!(loss_returned >= 0.0);

    let input_grad = x.grad()?.expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);
    assert!(input_grad.data_as_slice().iter().any(|&g| g.abs() > 1e-6));

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试 LogSoftmax 梯度累积
#[test]
fn test_log_softmax_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "input")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))?;

    let result = x.log_softmax();
    let target = graph.input(&Tensor::new(&[-2.0, -1.0, 0.0, -1.1, -1.1, -1.1], &[2, 3]))?;
    let loss = result.mse_loss(&target)?;

    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward().unwrap();
    let grad_first = x.grad().unwrap().unwrap().clone();

    loss.forward().unwrap();
    loss.backward().unwrap();
    let grad_second = x.grad().unwrap().unwrap();
    assert_eq!(&grad_second, &(&grad_first * 2.0));

    graph.zero_grad()?;
    loss.forward().unwrap();
    loss.backward().unwrap();
    let grad_after_clear = x.grad().unwrap().unwrap();
    assert_eq!(&grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 LogSoftmax 节点的动态形状传播
#[test]
fn test_log_softmax_dynamic_shape_propagation() {
    use crate::nn::Graph;
    use crate::nn::var::ops::VarActivationOps;

    let graph = Graph::new();

    // 创建 2D 输入
    let x = graph.input(&Tensor::zeros(&[4, 10])).unwrap();

    // LogSoftmax
    let log_probs = x.log_softmax();

    // 验证动态形状传播
    let dyn_shape = log_probs.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "num_classes 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(10), "num_classes 应该是 10");
}

/// 测试 LogSoftmax 节点在不同 batch_size 下的前向计算
#[test]
fn test_log_softmax_dynamic_batch_forward() {
    use crate::nn::Graph;
    use crate::nn::var::ops::VarActivationOps;

    let graph = Graph::new();

    // 创建 2D 输入
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))
        .unwrap();

    // LogSoftmax
    let log_probs = x.log_softmax();

    // 第一次 forward：batch=2
    log_probs.forward().unwrap();
    let value1 = log_probs.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 3], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        &[4, 3],
    ))
    .unwrap();

    // 第二次 forward：batch=4
    log_probs.forward().unwrap();
    let value2 = log_probs.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[4, 3], "第二次 forward: batch=4");
}

/// 测试 LogSoftmax 节点在不同 batch_size 下的反向传播
#[test]
fn test_log_softmax_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var::ops::{VarActivationOps, VarLossOps};

    let graph = Graph::new();

    // 创建参数和目标
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[-2.0, -1.0, 0.0, -1.1, -1.1, -1.1], &[2, 3]))
        .unwrap();

    // LogSoftmax + MSE
    let log_probs = x.log_softmax();
    let loss = log_probs.mse_loss(&target).unwrap();

    // 第一次训练：batch=2
    loss.forward().unwrap();
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::normal_seeded(0.0, 1.0, &[5, 3], 42))
        .unwrap();
    target.set_value(&Tensor::zeros(&[5, 3])).unwrap();

    // 第二次训练：batch=5
    loss.forward().unwrap();
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
}

// ==================== 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_log_softmax_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();

    let log_softmax = inner
        .borrow_mut()
        .create_log_softmax_node(input.clone(), Some("log_softmax"))
        .unwrap();

    assert_eq!(log_softmax.shape(), vec![2, 3]);
    assert_eq!(log_softmax.name(), Some("log_softmax"));
    assert!(!log_softmax.is_leaf());
    assert_eq!(log_softmax.parents().len(), 1);
}

#[test]
fn test_create_log_softmax_node_requires_2d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 3D 输入应该失败
    let input_3d = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4], None)
        .unwrap();

    let result = inner.borrow_mut().create_log_softmax_node(input_3d, None);

    assert!(result.is_err());
    match result.unwrap_err() {
        GraphError::InvalidOperation(msg) => {
            assert!(msg.contains("2D"));
        }
        _ => panic!("应该返回 InvalidOperation 错误"),
    }
}

#[test]
fn test_create_log_softmax_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let log_softmax = inner
        .borrow_mut()
        .create_log_softmax_node(input, None)
        .unwrap();
    assert_eq!(log_softmax.shape(), vec![5, 10]);
}

#[test]
fn test_create_log_softmax_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_log_softmax;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let log_softmax = inner
            .borrow_mut()
            .create_log_softmax_node(input, None)
            .unwrap();
        weak_log_softmax = Rc::downgrade(&log_softmax);

        assert!(weak_log_softmax.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_log_softmax.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
