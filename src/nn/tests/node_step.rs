/*
 * @Author       : 老董
 * @Description  : Step 节点单元测试（Heaviside 阶跃函数）
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）→ 底层 create_* API（文件末尾）
 * 2. 前向传播测试 → 高层 Graph + Var API
 * 3. VJP 单元测试（calc_grad_to_parent_index）→ 底层 NodeInner
 * 4. 端到端反向传播测试 → 高层 Graph + Var API
 * 5. 动态形状测试 → 高层 Graph + Var API
 *
 * 梯度公式：
 *   step(x) = { 1 if x >= 0, 0 if x < 0 }（Heaviside 阶跃函数）
 *   step'(x) 处处不可微（VJP 返回 0），与 PyTorch 行为一致。
 *   因此无梯度累积测试（0 + 0 = 0 没有意义）。
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 Step 前向传播（x >= 0 → 1, x < 0 → 0）
#[test]
fn test_step_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(
            &[-2.0, -0.5, 0.0, 0.5, 2.0, 0.0],
            &[2, 3],
        ))
        .unwrap();
    let result = x.step();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    // 注意：0.0 >= 0 为 true，输出 1.0
    let expected = Tensor::new(&[0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]);
    assert_eq!(output, expected);
}

/// 测试 Step 前向传播（极端值）
#[test]
fn test_step_forward_extreme_values() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(
            &[f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX],
            &[2, 2],
        ))
        .unwrap();
    let result = x.step();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    // INFINITY >= 0 → 1, NEG_INFINITY < 0 → 0, MIN < 0 → 0, MAX >= 0 → 1
    let expected = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert_eq!(output, expected);
}

/// 测试 Step 节点不能直接设置值
#[test]
fn test_step_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2]))
        .unwrap();
    let result = x.step();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Step 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================
//
// Step 是不可微函数，VJP 返回全零梯度（与 PyTorch 行为一致）。

/// 测试 Step VJP：梯度恒为 0
#[test]
fn test_step_vjp_always_zero() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let step = inner
        .borrow_mut()
        .create_step_node(x.clone(), Some("step"))
        .unwrap();

    // x = [0.5, -1.0, 0.0, 2.0]
    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    step.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = step.calc_grad_to_parent_index(0, &upstream_grad)?;

    // Step 不可微，梯度恒为 0
    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &Tensor::zeros(&[2, 2]));

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 Step 端到端反向传播：梯度不流经 Step 节点
#[test]
fn test_step_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.step();
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // step(x) = [1, 0, 1, 1]
    // loss = mean([1, 0, 1, 1]) = 0.75
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 0.75, epsilon = 1e-6);

    // Step 不可微，参数梯度应全为 0
    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(&x_grad, &Tensor::zeros(&[2, 2]));

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 Step 节点的动态形状传播
#[test]
fn test_step_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap();

    let result = h0.step();

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 Step 节点在不同 batch_size 下的前向计算
/// 注：Step 不可微，梯度恒为 0，因此不测试 backward
#[test]
fn test_step_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap();

    let result = h0.step();

    // 第一次 forward：batch=2
    result.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 16]);

    // 更新 batch_size
    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    result.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[8, 16]);
}

// ==================== 方案 C：新节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_step_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let step = inner
        .borrow_mut()
        .create_step_node(input.clone(), Some("step"))
        .unwrap();

    assert_eq!(step.shape(), vec![3, 4]);
    assert_eq!(step.name(), Some("step"));
    assert!(!step.is_leaf());
    assert_eq!(step.parents().len(), 1);
}

#[test]
fn test_create_step_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 5, 8], None)
        .unwrap();

    let step = inner
        .borrow_mut()
        .create_step_node(input.clone(), None)
        .unwrap();

    assert_eq!(step.shape(), vec![2, 5, 8]);
}

#[test]
fn test_create_step_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_step;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let step = inner.borrow_mut().create_step_node(input, None).unwrap();
        weak_step = Rc::downgrade(&step);

        assert!(weak_step.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_step.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
