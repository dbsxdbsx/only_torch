/*
 * @Author       : 老董
 * @Description  : TopK 节点单元测试
 *
 * 测试策略（6 类标准测试）：
 * 1. 前向传播测试（高层 Graph + Var API）→ basic forward + edge cases + cannot_set_value
 * 2. VJP 单元测试（calc_grad_to_parent_index）→ 底层 NodeInner
 * 3. 端到端反向传播测试（高层 Graph + Var API）
 * 4. 梯度累积测试（高层 Graph + Var API）
 * 5. 节点创建 API 测试
 *
 * 梯度公式：
 *   forward: values = input.topk(k, axis, sorted).0
 *   backward: grad = zeros(parent_shape); 对每个 topk 位置, grad[orig_index] += upstream
 *
 * Python 对照脚本: tests/python/calc_jacobi_by_pytorch/node_topk.py
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarSelectionOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试 ====================

/// 基本 topk：[2, 4] topk(k=2, axis=1, sorted=true) → [2, 2]
///
/// input:
///   [[1, 4, 2, 3],
///    [8, 5, 7, 6]]
/// topk(k=2, axis=1, sorted=true) →
///   values: [[4, 3], [8, 7]]（每行前 2 大，降序）
#[test]
fn test_topk_forward_basic() {
    let graph = Graph::new();

    let input_data = Tensor::new(&[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0], &[2, 4]);
    let x = graph.input(&input_data).unwrap();
    let result = x.topk(2, 1, true).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    // 第一行前 2 大：4.0, 3.0
    assert_abs_diff_eq!(output[[0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 3.0, epsilon = 1e-6);
    // 第二行前 2 大：8.0, 7.0
    assert_abs_diff_eq!(output[[1, 0]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 7.0, epsilon = 1e-6);
}

/// k=1 topk：[2, 4] topk(k=1, axis=1, sorted=true) → [2, 1]（等价于 max）
///
/// input:
///   [[1, 4, 2, 3],
///    [8, 5, 7, 6]]
/// topk(k=1, axis=1) → [[4], [8]]
#[test]
fn test_topk_forward_k1_max() {
    let graph = Graph::new();

    let input_data = Tensor::new(&[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0], &[2, 4]);
    let x = graph.input(&input_data).unwrap();
    let result = x.topk(1, 1, true).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 1]);
    assert_abs_diff_eq!(output[[0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 8.0, epsilon = 1e-6);
}

/// axis=0 topk：[4, 2] topk(k=2, axis=0, sorted=true) → [2, 2]
///
/// input:
///   [[1, 2],
///    [5, 6],
///    [3, 4],
///    [7, 8]]
/// topk(k=2, axis=0, sorted=true):
///   对每列取前 2 大 →
///   列 0: [7, 5], 列 1: [8, 6] → [[7, 8], [5, 6]]
#[test]
fn test_topk_forward_axis0() {
    let graph = Graph::new();

    let input_data = Tensor::new(&[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0], &[4, 2]);
    let x = graph.input(&input_data).unwrap();
    let result = x.topk(2, 0, true).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    // 列 0 前 2 大：7.0, 5.0; 列 1 前 2 大：8.0, 6.0
    assert_abs_diff_eq!(output[[0, 0]], 7.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 6.0, epsilon = 1e-6);
}

/// TopK 节点不能直接设置值
#[test]
fn test_topk_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.topk(1, 1, true).unwrap();

    let test_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "TopK 节点不应支持直接设值");
}

// ==================== 2. VJP 单元测试 ====================

/// VJP scatter：unit upstream（全 1.0 上游梯度）
///
/// input [2, 4], topk(k=2, axis=1, sorted=true)
/// input: [[1, 4, 2, 3], [8, 5, 7, 6]]
/// indices: [[1, 3], [0, 2]]（前 2 大的原始列索引）
/// upstream [2, 2] = 全 1.0
/// grad [2, 4]: 在选中位置填入 1.0，其余 = 0.0
/// 期望: [[0, 1, 0, 1], [1, 0, 1, 0]]
#[test]
fn test_topk_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 4], Some("input"))
        .unwrap();
    let topk = inner
        .borrow_mut()
        .create_topk_node(input.clone(), 2, 1, true, Some("topk"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(
            &[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0],
            &[2, 4],
        )))
        .unwrap();
    topk.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[2, 2]);
    let grad = topk
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_eq!(grad.shape(), &[2, 4]);

    // 第一行：index 1 (4.0) 和 index 3 (3.0) 被选中
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 2]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 3]], 1.0, epsilon = 1e-6);

    // 第二行：index 0 (8.0) 和 index 2 (7.0) 被选中
    assert_abs_diff_eq!(grad[[1, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 2]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 3]], 0.0, epsilon = 1e-6);

    Ok(())
}

/// VJP scatter：非 unit upstream（缩放梯度）
///
/// input [2, 4], topk(k=2, axis=1, sorted=true)
/// input: [[1, 4, 2, 3], [8, 5, 7, 6]]
/// upstream [2, 2] = [[2, 3], [4, 5]]
/// grad [2, 4]:
///   第一行：pos 1 ← 2.0, pos 3 ← 3.0 → [0, 2, 0, 3]
///   第二行：pos 0 ← 4.0, pos 2 ← 5.0 → [4, 0, 5, 0]
#[test]
fn test_topk_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 4], Some("input"))
        .unwrap();
    let topk = inner
        .borrow_mut()
        .create_topk_node(input.clone(), 2, 1, true, Some("topk"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(
            &[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0],
            &[2, 4],
        )))
        .unwrap();
    topk.forward_recursive(1, false).unwrap();

    let upstream = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = topk
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_eq!(grad.shape(), &[2, 4]);

    // 第一行
    let expected_row0 = [0.0, 2.0, 0.0, 3.0];
    for j in 0..4 {
        assert_abs_diff_eq!(grad[[0, j]], expected_row0[j], epsilon = 1e-6);
    }

    // 第二行
    let expected_row1 = [4.0, 0.0, 5.0, 0.0];
    for j in 0..4 {
        assert_abs_diff_eq!(grad[[1, j]], expected_row1[j], epsilon = 1e-6);
    }

    Ok(())
}

// ==================== 3. 端到端反向传播测试 ====================

/// topk(x) -> MSE loss -> backward，验证梯度正确流回
///
/// x [2, 4] → topk(k=2, axis=1) → [2, 2] → MSE(target=[2,2])
/// 梯度应只出现在被选中的位置，未被选中的位置梯度为零
#[test]
fn test_topk_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 4], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(
        &[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0],
        &[2, 4],
    ))?;

    let topk_vals = x.topk(2, 1, true)?;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = topk_vals.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    // loss 应为有限正数
    assert!(loss_val > 0.0);
    assert!(loss_val.is_finite());

    let input_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 4]);

    // 未选中的位置梯度应为 0
    // 第一行：index 0 (1.0) 和 index 2 (2.0) 未选中
    assert_abs_diff_eq!(input_grad[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(input_grad[[0, 2]], 0.0, epsilon = 1e-6);

    // 第二行：index 1 (5.0) 和 index 3 (6.0) 未选中
    assert_abs_diff_eq!(input_grad[[1, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(input_grad[[1, 3]], 0.0, epsilon = 1e-6);

    // 选中位置应有非零梯度
    assert!(input_grad[[0, 1]].abs() > 1e-10, "选中位置应有非零梯度");
    assert!(input_grad[[0, 3]].abs() > 1e-10, "选中位置应有非零梯度");
    assert!(input_grad[[1, 0]].abs() > 1e-10, "选中位置应有非零梯度");
    assert!(input_grad[[1, 2]].abs() > 1e-10, "选中位置应有非零梯度");

    Ok(())
}

// ==================== 4. 梯度累积测试 ====================

/// 测试 TopK 梯度累积 + zero_grad
#[test]
fn test_topk_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 4], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(
        &[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0],
        &[2, 4],
    ))?;

    let topk_vals = x.topk(2, 1, true)?;
    let target = graph.input(&Tensor::ones(&[2, 2]))?;
    let loss = topk_vals.mse_loss(&target)?;

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

// ==================== 5. 节点创建 API 测试 ====================

use std::rc::Rc;

/// 基本创建：验证输出形状
///
/// [2, 4] topk(k=2, axis=1) → [2, 2]
#[test]
fn test_create_topk_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("input"))
        .unwrap();
    let topk = inner
        .borrow_mut()
        .create_topk_node(input.clone(), 2, 1, true, Some("topk"))
        .unwrap();

    assert_eq!(topk.shape(), vec![2, 2]);
    assert_eq!(topk.name(), Some("topk"));
    assert!(!topk.is_leaf());
    assert_eq!(topk.parents().len(), 1);
}

/// 无效 axis（应报错）
///
/// [2, 4] topk(axis=2, ...) → axis=2 超出 2 维张量的范围
#[test]
fn test_create_topk_node_invalid_axis() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None)
        .unwrap();

    let result = inner.borrow_mut().create_topk_node(input, 2, 2, true, None);
    assert!(result.is_err());
}

/// k=0（应报错）
#[test]
fn test_create_topk_node_k_zero() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None)
        .unwrap();

    let result = inner.borrow_mut().create_topk_node(input, 0, 1, true, None);
    assert!(result.is_err());
}

/// k 超出轴大小（应报错）
///
/// [2, 4] topk(k=5, axis=1) → k=5 > axis_len=4
#[test]
fn test_create_topk_node_k_exceeds() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None)
        .unwrap();

    let result = inner.borrow_mut().create_topk_node(input, 5, 1, true, None);
    assert!(result.is_err());
}

/// drop 释放
#[test]
fn test_create_topk_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_topk;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let topk = inner
            .borrow_mut()
            .create_topk_node(input, 2, 1, true, None)
            .unwrap();
        weak_topk = Rc::downgrade(&topk);

        assert!(weak_topk.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_topk.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
