/*
 * @Author       : 老董
 * @Description  : Pow（幂运算）节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ 平方 + 立方 + 分数幂 + 不能直接设值
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ 单位/非单位上游梯度
 * 3. 端到端反向传播测试（高层 API）
 * 4. 梯度累积测试（高层 API）
 * 5. 动态形状测试
 * 6. Create API 测试
 *
 * 梯度公式：
 *   y = x^p
 *   dy/dx = p * x^(p-1)
 *   VJP: grad_to_parent = upstream_grad * p * x^(p-1)
 *
 * Python 对照值 (numpy):
 *   [1, 2, 3, 4]^2 = [1, 4, 9, 16]
 *   [1, 2, 3, 4]^3 = [1, 8, 27, 64]
 *   [1, 4, 9, 16]^0.5 = [1, 2, 3, 4]
 *   grad(x^2) = 2*x → 2*[1,2,3,4] = [2,4,6,8]
 *   grad(x^3) = 3*x^2 → 3*[1,4,9,16] = [3,12,27,48]
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 API）====================

/// 测试 Pow 前向传播（平方）
///
/// [1, 2, 3, 4]^2 = [1, 4, 9, 16]
#[test]
fn test_pow_forward_square() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.pow(2.0);

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 9.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 16.0, epsilon = 1e-6);
}

/// 测试 Pow 前向传播（立方）
///
/// [1, 2, 3, 4]^3 = [1, 8, 27, 64]
#[test]
fn test_pow_forward_cube() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.pow(3.0);

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 27.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[1, 1]], 64.0, epsilon = 1e-5);
}

/// 测试 Pow 前向传播（分数幂 = 开方）
///
/// [1, 4, 9, 16]^0.5 = [1, 2, 3, 4]
#[test]
fn test_pow_forward_sqrt() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 4.0, 9.0, 16.0], &[2, 2]))
        .unwrap();
    let result = x.pow(0.5);

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[1, 1]], 4.0, epsilon = 1e-5);
}

/// 测试 Pow 节点不能直接设置值
#[test]
fn test_pow_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.pow(2.0);

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Pow 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================

/// 测试 Pow VJP：x^2 的梯度（单位上游梯度）
///
/// grad = 2 * x → 2 * [1, 2, 3, 4] = [2, 4, 6, 8]
#[test]
fn test_pow_vjp_square_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let pow = inner
        .borrow_mut()
        .create_pow_node(x.clone(), 2.0, Some("pow"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    pow.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = pow
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0, epsilon = 1e-6); // 2*1
    assert_abs_diff_eq!(grad[[0, 1]], 4.0, epsilon = 1e-6); // 2*2
    assert_abs_diff_eq!(grad[[1, 0]], 6.0, epsilon = 1e-6); // 2*3
    assert_abs_diff_eq!(grad[[1, 1]], 8.0, epsilon = 1e-6); // 2*4

    Ok(())
}

/// 测试 Pow VJP：x^3 的梯度（非单位上游梯度）
///
/// grad = upstream * 3 * x^2
/// upstream=[2, 1, 0.5, 3], x=[1, 2, 3, 4]
/// → [2*3*1, 1*3*4, 0.5*3*9, 3*3*16] = [6, 12, 13.5, 144]
#[test]
fn test_pow_vjp_cube_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let pow = inner
        .borrow_mut()
        .create_pow_node(x.clone(), 3.0, Some("pow"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    pow.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 1.0, 0.5, 3.0], &[2, 2]);
    let grad = pow
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 6.0, epsilon = 1e-5); // 2 * 3 * 1^2
    assert_abs_diff_eq!(grad[[0, 1]], 12.0, epsilon = 1e-5); // 1 * 3 * 2^2
    assert_abs_diff_eq!(grad[[1, 0]], 13.5, epsilon = 1e-5); // 0.5 * 3 * 3^2
    assert_abs_diff_eq!(grad[[1, 1]], 144.0, epsilon = 1e-4); // 3 * 3 * 4^2

    Ok(())
}

// ==================== 端到端反向传播测试（高层 API）====================

/// 测试 Pow 端到端反向传播：result = x^2 → loss = MSE(result, target)
#[test]
fn test_pow_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;

    let result = x.pow(2.0);
    let target = graph.input(&Tensor::new(&[1.0, 3.0, 8.0, 15.0], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);

    Ok(())
}

// ==================== 梯度累积测试（高层 API）====================

/// 测试 Pow 梯度累积
#[test]
fn test_pow_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;

    let result = x.pow(2.0);
    let target = graph.input(&Tensor::new(&[1.0, 3.0, 8.0, 15.0], &[2, 2]))?;
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

/// 测试 Pow 节点的动态形状传播
#[test]
fn test_pow_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::ones(&[4, 8])).unwrap();
    let result = x.pow(2.0);

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(8), "特征维度应该是 8");
}

/// 测试 Pow 节点在不同 batch_size 下的前向计算
#[test]
fn test_pow_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[2.0; 16], &[2, 8]))
        .unwrap();
    let result = x.pow(2.0);

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 8]);
    assert_abs_diff_eq!(value1[[0, 0]], 4.0, epsilon = 1e-6);

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(&[3.0; 48], &[6, 8])).unwrap();

    // 第二次 forward：batch=6
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[6, 8]);
    assert_abs_diff_eq!(value2[[0, 0]], 9.0, epsilon = 1e-6);
}

// ==================== 节点创建 API 测试 ====================

#[test]
fn test_create_pow_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let pow = inner
        .borrow_mut()
        .create_pow_node(input.clone(), 2.0, Some("pow"))
        .unwrap();

    assert_eq!(pow.shape(), vec![3, 4]);
    assert_eq!(pow.name(), Some("pow"));
}

#[test]
fn test_create_pow_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 5, 8], None)
        .unwrap();

    let pow = inner
        .borrow_mut()
        .create_pow_node(input.clone(), 3.0, None)
        .unwrap();

    assert_eq!(pow.shape(), vec![2, 5, 8]);
}

#[test]
fn test_create_pow_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_pow;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let pow = inner
            .borrow_mut()
            .create_pow_node(input, 2.0, None)
            .unwrap();
        weak_pow = Rc::downgrade(&pow);

        assert!(weak_pow.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_pow.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
