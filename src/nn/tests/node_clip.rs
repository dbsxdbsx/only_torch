/*
 * @Author       : 老董
 * @Description  : Clip（值域裁剪）节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ basic forward + boundary + cannot_set_value
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ unit upstream + non-unit + all-clipped
 * 3. 端到端反向传播测试（高层 API）
 * 4. 梯度累积测试（高层 API）
 * 5. 动态形状测试
 * 6. Create API 测试 + 参数验证
 *
 * 梯度公式：
 *   y = clip(x, min, max)
 *   dy/dx = 1 if min < x < max, else 0
 *   VJP: grad_to_parent = upstream_grad * mask
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 API）====================

/// 测试 Clip 前向传播
///
/// clip([-3, -1, 0, 1, 3], -2, 2) = [-2, -1, 0, 1, 2]
#[test]
fn test_clip_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[-3.0, -1.0, 0.0, 1.0, 3.0, 5.0], &[2, 3]))
        .unwrap();
    let result = x.clip(-2.0, 2.0);

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 3]);
    assert_abs_diff_eq!(output[[0, 0]], -2.0, epsilon = 1e-6); // -3 → -2（被裁剪）
    assert_abs_diff_eq!(output[[0, 1]], -1.0, epsilon = 1e-6); // -1（在范围内）
    assert_abs_diff_eq!(output[[0, 2]], 0.0, epsilon = 1e-6); // 0（在范围内）
    assert_abs_diff_eq!(output[[1, 0]], 1.0, epsilon = 1e-6); // 1（在范围内）
    assert_abs_diff_eq!(output[[1, 1]], 2.0, epsilon = 1e-6); // 3 → 2（被裁剪）
    assert_abs_diff_eq!(output[[1, 2]], 2.0, epsilon = 1e-6); // 5 → 2（被裁剪）
}

/// 测试 Clip 边界值
///
/// clip([-20, -2, 0, 2, 20], -2, 2)
/// 恰好等于边界的值不变
#[test]
fn test_clip_forward_boundary() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[-20.0, -2.0, 0.0, 2.0, 20.0], &[1, 5]))
        .unwrap();
    let result = x.clip(-2.0, 2.0);

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], -2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], -2.0, epsilon = 1e-6); // 恰好等于 min
    assert_abs_diff_eq!(output[[0, 2]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 3]], 2.0, epsilon = 1e-6); // 恰好等于 max
    assert_abs_diff_eq!(output[[0, 4]], 2.0, epsilon = 1e-6);
}

/// 测试 SAC 典型用例：log_std 裁剪到 [-20, 2]
#[test]
fn test_clip_sac_log_std() {
    let graph = Graph::new();

    let log_std = graph
        .input(&Tensor::new(&[-25.0, -10.0, 0.0, 5.0], &[1, 4]))
        .unwrap();
    let clamped = log_std.clip(-20.0, 2.0);

    clamped.forward().unwrap();

    let output = clamped.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], -20.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], -10.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 3]], 2.0, epsilon = 1e-6);
}

/// 测试 Clip 节点不能直接设置值
#[test]
fn test_clip_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.clip(-1.0, 1.0);

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Clip 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================

/// 测试 Clip VJP（单位上游梯度）
///
/// 对于 clip(x, -2, 2)：
/// x = [-3, -1, 0, 1, 3, 2]
/// mask = [0, 1, 1, 1, 0, 0]  （边界处梯度为 0）
/// grad = [0, 1, 1, 1, 0, 0]
#[test]
fn test_clip_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let clip = inner
        .borrow_mut()
        .create_clip_node(x.clone(), -2.0, 2.0, Some("clip"))
        .unwrap();

    x.set_value(Some(&Tensor::new(
        &[-3.0, -1.0, 0.0, 1.0, 3.0, 2.0],
        &[2, 3],
    )))
    .unwrap();
    clip.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 3]);
    let grad = clip.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 3]);
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-6); // x=-3 < min → 0
    assert_abs_diff_eq!(grad[[0, 1]], 1.0, epsilon = 1e-6); // x=-1 在范围内 → 1
    assert_abs_diff_eq!(grad[[0, 2]], 1.0, epsilon = 1e-6); // x=0 在范围内 → 1
    assert_abs_diff_eq!(grad[[1, 0]], 1.0, epsilon = 1e-6); // x=1 在范围内 → 1
    assert_abs_diff_eq!(grad[[1, 1]], 0.0, epsilon = 1e-6); // x=3 > max → 0
    assert_abs_diff_eq!(grad[[1, 2]], 0.0, epsilon = 1e-6); // x=2 == max → 0

    Ok(())
}

/// 测试 Clip VJP（非单位上游梯度）
///
/// grad = upstream * mask
#[test]
fn test_clip_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("x"))
        .unwrap();
    let clip = inner
        .borrow_mut()
        .create_clip_node(x.clone(), -1.0, 1.0, Some("clip"))
        .unwrap();

    // x = [-2, 0, 0.5, 2], mask = [0, 1, 1, 0]
    x.set_value(Some(&Tensor::new(&[-2.0, 0.0, 0.5, 2.0], &[1, 4])))
        .unwrap();
    clip.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[3.0, 5.0, 7.0, 9.0], &[1, 4]);
    let grad = clip.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[1, 4]);
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-6); // 3 * 0 = 0
    assert_abs_diff_eq!(grad[[0, 1]], 5.0, epsilon = 1e-6); // 5 * 1 = 5
    assert_abs_diff_eq!(grad[[0, 2]], 7.0, epsilon = 1e-6); // 7 * 1 = 7
    assert_abs_diff_eq!(grad[[0, 3]], 0.0, epsilon = 1e-6); // 9 * 0 = 0

    Ok(())
}

/// 测试所有元素都被裁剪的情况（梯度全为 0）
#[test]
fn test_clip_vjp_all_clipped() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("x"))
        .unwrap();
    let clip = inner
        .borrow_mut()
        .create_clip_node(x.clone(), -1.0, 1.0, Some("clip"))
        .unwrap();

    // 所有值都在范围外
    x.set_value(Some(&Tensor::new(&[-5.0, 5.0, -1.0], &[1, 3])))
        .unwrap();
    clip.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[1, 3]);
    let grad = clip.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    // x=-5 被裁剪, x=5 被裁剪, x=-1==min 边界处也为 0
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 2]], 0.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 API）====================

/// 测试 Clip 端到端反向传播
#[test]
fn test_clip_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[-3.0, -0.5, 0.5, 3.0], &[2, 2]))?;

    let result = x.clip(-1.0, 1.0);
    let target = graph.input(&Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);

    // x=-3 被裁剪到 -1，梯度应为 0（裁剪区域无梯度）
    assert_abs_diff_eq!(x_grad[[0, 0]], 0.0, epsilon = 1e-6);
    // x=-0.5 在范围内，梯度应非零
    assert!(x_grad[[0, 1]].abs() > 1e-6);
    // x=0.5 在范围内，梯度应非零
    assert!(x_grad[[1, 0]].abs() > 1e-6);
    // x=3 被裁剪到 1，梯度应为 0
    assert_abs_diff_eq!(x_grad[[1, 1]], 0.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 梯度累积测试（高层 API）====================

/// 测试 Clip 梯度累积
#[test]
fn test_clip_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[-0.5, 0.0, 0.5, 1.5], &[2, 2]))?;

    let result = x.clip(-1.0, 1.0);
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

/// 测试 Clip 节点的动态形状传播
#[test]
fn test_clip_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::ones(&[4, 8])).unwrap();
    let result = x.clip(-1.0, 1.0);

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(8), "特征维度应该是 8");
}

/// 测试 Clip 节点在不同 batch_size 下的前向计算
#[test]
fn test_clip_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[5.0; 16], &[2, 8]))
        .unwrap();
    let result = x.clip(-1.0, 1.0);

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 8]);
    assert_abs_diff_eq!(value1[[0, 0]], 1.0, epsilon = 1e-6); // 5 被裁剪到 1

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(&[-5.0; 48], &[6, 8])).unwrap();

    // 第二次 forward：batch=6
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[6, 8]);
    assert_abs_diff_eq!(value2[[0, 0]], -1.0, epsilon = 1e-6); // -5 被裁剪到 -1
}

// ==================== 节点创建 API 测试 ====================

#[test]
fn test_create_clip_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let clip = inner
        .borrow_mut()
        .create_clip_node(input.clone(), -1.0, 1.0, Some("clip"))
        .unwrap();

    assert_eq!(clip.shape(), vec![3, 4]);
    assert_eq!(clip.name(), Some("clip"));
}

#[test]
fn test_create_clip_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 5, 8], None)
        .unwrap();

    let clip = inner
        .borrow_mut()
        .create_clip_node(input.clone(), 0.0, 1.0, None)
        .unwrap();

    assert_eq!(clip.shape(), vec![2, 5, 8]);
}

/// 测试 min > max 时应报错
#[test]
fn test_create_clip_node_invalid_range() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_clip_node(input.clone(), 2.0, -2.0, None);

    assert!(result.is_err(), "min > max 应返回错误");
}

#[test]
fn test_create_clip_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_clip;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let clip = inner
            .borrow_mut()
            .create_clip_node(input, -1.0, 1.0, None)
            .unwrap();
        weak_clip = Rc::downgrade(&clip);

        assert!(weak_clip.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_clip.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
