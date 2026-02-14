/*
 * @Author       : 老董
 * @Description  : Sqrt（平方根）节点单元测试
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
 *   y = √x
 *   dy/dx = 0.5 / √x = 0.5 / y
 *   VJP: grad_to_parent = upstream_grad * 0.5 / √x
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 API）====================

/// 测试 Sqrt 前向传播
///
/// sqrt([0, 1, 4, 9]) = [0, 1, 2, 3]
#[test]
fn test_sqrt_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[0.0, 1.0, 4.0, 9.0], &[2, 2]))
        .unwrap();
    let result = x.sqrt();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 3.0, epsilon = 1e-6);
}

/// 测试 Sqrt 前向传播（非整数平方根）
///
/// sqrt([0.25, 2, 0.01, 100]) = [0.5, 1.4142, 0.1, 10]
#[test]
fn test_sqrt_forward_non_perfect() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[0.25, 2.0, 0.01, 100.0], &[1, 4]))
        .unwrap();
    let result = x.sqrt();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], std::f32::consts::SQRT_2, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 3]], 10.0, epsilon = 1e-5);
}

/// 测试 Sqrt 节点不能直接设置值
#[test]
fn test_sqrt_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 4.0, 9.0, 16.0], &[2, 2]))
        .unwrap();
    let result = x.sqrt();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Sqrt 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================

/// 测试 Sqrt VJP（单位上游梯度）
///
/// grad = 0.5 / √x → 0.5 / √[1, 4, 9, 16] = [0.5, 0.25, 1/6, 0.125]
#[test]
fn test_sqrt_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let sqrt = inner
        .borrow_mut()
        .create_sqrt_node(x.clone(), Some("sqrt"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 4.0, 9.0, 16.0], &[2, 2])))
        .unwrap();
    sqrt.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = sqrt.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 0.5, epsilon = 1e-6); // 0.5/√1 = 0.5
    assert_abs_diff_eq!(grad[[0, 1]], 0.25, epsilon = 1e-6); // 0.5/√4 = 0.25
    assert_abs_diff_eq!(grad[[1, 0]], 1.0 / 6.0, epsilon = 1e-6); // 0.5/√9 = 1/6
    assert_abs_diff_eq!(grad[[1, 1]], 0.125, epsilon = 1e-6); // 0.5/√16 = 0.125

    Ok(())
}

/// 测试 Sqrt VJP（非单位上游梯度）
///
/// grad = upstream * 0.5 / √x → [2, 6] * [0.5, 0.25] = [1.0, 1.5]
#[test]
fn test_sqrt_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("x"))
        .unwrap();
    let sqrt = inner
        .borrow_mut()
        .create_sqrt_node(x.clone(), Some("sqrt"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 4.0], &[1, 2])))
        .unwrap();
    sqrt.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 6.0], &[1, 2]);
    let grad = sqrt.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[1, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6); // 2 * 0.5/√1
    assert_abs_diff_eq!(grad[[0, 1]], 1.5, epsilon = 1e-6); // 6 * 0.5/√4

    Ok(())
}

// ==================== 端到端反向传播测试（高层 API）====================

/// 测试 Sqrt 端到端反向传播
#[test]
fn test_sqrt_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 4.0, 9.0, 16.0], &[2, 2]))?;

    let result = x.sqrt();
    let target = graph.input(&Tensor::new(&[0.5, 1.5, 2.5, 3.5], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);

    Ok(())
}

// ==================== 梯度累积测试（高层 API）====================

/// 测试 Sqrt 梯度累积
#[test]
fn test_sqrt_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 4.0, 9.0, 16.0], &[2, 2]))?;

    let result = x.sqrt();
    let target = graph.input(&Tensor::new(&[0.5, 1.5, 2.5, 3.5], &[2, 2]))?;
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

/// 测试 Sqrt 节点的动态形状传播
#[test]
fn test_sqrt_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::ones(&[4, 8])).unwrap();
    let result = x.sqrt();

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(8), "特征维度应该是 8");
}

/// 测试 Sqrt 节点在不同 batch_size 下的前向计算
#[test]
fn test_sqrt_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[4.0; 16], &[2, 8]))
        .unwrap();
    let result = x.sqrt();

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 8]);
    assert_abs_diff_eq!(value1[[0, 0]], 2.0, epsilon = 1e-6);

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(&[9.0; 48], &[6, 8])).unwrap();

    // 第二次 forward：batch=6
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[6, 8]);
    assert_abs_diff_eq!(value2[[0, 0]], 3.0, epsilon = 1e-6);
}

// ==================== 节点创建 API 测试 ====================

#[test]
fn test_create_sqrt_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let sqrt = inner
        .borrow_mut()
        .create_sqrt_node(input.clone(), Some("sqrt"))
        .unwrap();

    assert_eq!(sqrt.shape(), vec![3, 4]);
    assert_eq!(sqrt.name(), Some("sqrt"));
}

#[test]
fn test_create_sqrt_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 5, 8], None)
        .unwrap();

    let sqrt = inner
        .borrow_mut()
        .create_sqrt_node(input.clone(), None)
        .unwrap();

    assert_eq!(sqrt.shape(), vec![2, 5, 8]);
}

#[test]
fn test_create_sqrt_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_sqrt;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let sqrt = inner
            .borrow_mut()
            .create_sqrt_node(input, None)
            .unwrap();
        weak_sqrt = Rc::downgrade(&sqrt);

        assert!(weak_sqrt.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_sqrt.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
