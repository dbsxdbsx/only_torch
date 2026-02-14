/*
 * @Author       : 老董
 * @Description  : Ln（自然对数）节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ basic forward + edge cases + cannot_set_value
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ unit upstream + non-unit + small input
 * 3. 端到端反向传播测试（高层 API）
 * 4. 梯度累积测试（高层 API）
 * 5. 动态形状测试（KEEP AS-IS）
 * 6. Create API 测试（KEEP AS-IS）
 *
 * 梯度公式：
 *   ln(x)，要求 x > 0
 *   dy/dx = 1/x
 *   VJP: grad_to_parent = upstream_grad / x
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 API）====================

/// 测试 Ln 前向传播
///
/// ln([1.0, e, e², 0.5]) = [0.0, 1.0, 2.0, -0.6931472]
#[test]
fn test_ln_forward() {
    let graph = Graph::new();

    let e = std::f32::consts::E;
    let x = graph
        .input(&Tensor::new(&[1.0, e, e * e, 0.5], &[2, 2]))
        .unwrap();
    let result = x.ln();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-6); // ln(1) = 0
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-6); // ln(e) = 1
    assert_abs_diff_eq!(output[[1, 0]], 2.0, epsilon = 1e-5); // ln(e²) = 2
    assert_abs_diff_eq!(output[[1, 1]], -0.6931472, epsilon = 1e-6); // ln(0.5) ≈ -0.693
}

/// 测试 Ln 前向传播（边界值）
///
/// ln(0.001)≈-6.908, ln(1)=0, ln(10)≈2.303, ln(100)≈4.605
#[test]
fn test_ln_forward_edge_cases() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[0.001, 1.0, 10.0, 100.0], &[1, 4]))
        .unwrap();
    let result = x.ln();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], -6.907755, epsilon = 1e-5); // ln(0.001)
    assert_abs_diff_eq!(output[[0, 1]], 0.0, epsilon = 1e-6); // ln(1) = 0
    assert_abs_diff_eq!(output[[0, 2]], 2.302585, epsilon = 1e-5); // ln(10)
    assert_abs_diff_eq!(output[[0, 3]], 4.60517, epsilon = 1e-4); // ln(100)
}

/// 测试 Ln 节点不能直接设置值
#[test]
fn test_ln_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.ln();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Ln 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================

/// 测试 Ln VJP（单位上游梯度）
///
/// grad = 1/x → 1/[1, 2, 4, 0.5] = [1, 0.5, 0.25, 2]
#[test]
fn test_ln_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let ln = inner
        .borrow_mut()
        .create_ln_node(x.clone(), Some("ln"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 4.0, 0.5], &[2, 2])))
        .unwrap();
    ln.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = ln.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 2.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Ln VJP（非单位上游梯度）
///
/// grad = upstream/x → [2, 3, 4, 5]/[1, 2, 4, 0.5] = [2, 1.5, 1, 10]
#[test]
fn test_ln_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let ln = inner
        .borrow_mut()
        .create_ln_node(x.clone(), Some("ln"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 4.0, 0.5], &[2, 2])))
        .unwrap();
    ln.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = ln.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 1.5, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 10.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Ln 梯度计算（小输入 → 梯度放大）
///
/// grad = 1/x → 1/[0.1, 0.01] = [10, 100]
#[test]
fn test_ln_vjp_small_input() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("x"))
        .unwrap();
    let ln = inner
        .borrow_mut()
        .create_ln_node(x.clone(), Some("ln"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.1, 0.01], &[1, 2])))
        .unwrap();
    ln.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[1, 2]);
    let grad = ln.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_abs_diff_eq!(grad[[0, 0]], 10.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 100.0, epsilon = 1e-4);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 API）====================

/// 测试 Ln 端到端反向传播：result = ln(input) → loss = MSE(result, target)
#[test]
fn test_ln_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 4.0, 0.5], &[2, 2]))?;

    let result = x.ln();
    let target = graph.input(&Tensor::new(&[0.0, 0.5, 1.5, -0.5], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);

    Ok(())
}

// ==================== 梯度累积测试（高层 API）====================

/// 测试 Ln 梯度累积
#[test]
fn test_ln_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 4.0, 0.5], &[2, 2]))?;

    let result = x.ln();
    let target = graph.input(&Tensor::new(&[0.0, 0.5, 1.5, -0.5], &[2, 2]))?;
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

/// 测试 Ln 节点的动态形状传播
#[test]
fn test_ln_dynamic_shape_propagation() {
    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用正数）
    let x = graph.input(&Tensor::ones(&[4, 8])).unwrap();

    // 直接用 x 来测试，因为 x 是正数
    let result = x.ln();

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(8), "特征维度应该是 8");
}

/// 测试 Ln 节点在不同 batch_size 下的前向计算
#[test]
fn test_ln_dynamic_batch_forward() {
    let graph = Graph::new();

    // 创建支持动态 batch 的节点（正数输入）
    let x = graph.input(&Tensor::ones(&[2, 8])).unwrap();

    // Ln: x -> ln(x)
    let result = x.ln();

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 8], "第一次 forward: batch=2");
    // ln(1) = 0
    assert_abs_diff_eq!(value1[[0, 0]], 0.0, epsilon = 1e-6);

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::ones(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 8], "第二次 forward: batch=8");
}

/// 测试 Ln 节点在不同 batch_size 下的反向传播
#[test]
fn test_ln_dynamic_batch_backward() {
    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    // 使用 e 的幂作为输入，确保输出在合理范围
    let x = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            &[2, 4],
        ))
        .unwrap();

    // Ln: x -> ln(x) -> [?, 4]
    let result = x.ln();

    // 创建目标和损失
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(
        &[
            1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0,
            2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
        ],
        &[6, 4],
    ))
    .unwrap();
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

#[test]
fn test_create_ln_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let ln = inner
        .borrow_mut()
        .create_ln_node(input.clone(), Some("ln"))
        .unwrap();

    assert_eq!(ln.shape(), vec![3, 4]);
    assert_eq!(ln.name(), Some("ln"));
}

#[test]
fn test_create_ln_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 5, 8], None)
        .unwrap();

    let ln = inner
        .borrow_mut()
        .create_ln_node(input.clone(), None)
        .unwrap();

    assert_eq!(ln.shape(), vec![2, 5, 8]);
}

#[test]
fn test_create_ln_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_ln;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let ln = inner.borrow_mut().create_ln_node(input, None).unwrap();
        weak_ln = Rc::downgrade(&ln);

        assert!(weak_ln.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_ln.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
