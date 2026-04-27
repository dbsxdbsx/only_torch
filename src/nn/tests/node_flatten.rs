/*
 * @Author       : 老董
 * @Description  : Flatten 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层+底层混合）→ keep_first 2D; 完全展平; 方形; batch; 组合; 链式; cannot_set_value
 * 2. VJP 单元测试（底层）→ unit upstream; 非 unit upstream
 * 3. E2E 反向传播（高层）
 * 4. 梯度累积
 * 5. 动态形状（KEEP AS-IS）
 * 6. Create API（KEEP AS-IS）
 *
 * Flatten 将高维张量展平。keep_first_dim=true 保留 batch 维。VJP: grad = reshape(upstream, input_shape)
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps, VarMatrixOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试（高层+底层混合）====================

/// 测试 keep_first_dim=true（高层 API 默认）— 2D 张量保持不变
///
/// [3, 4] → flatten(keep_first=true) → [3, 4]（已经是 2D，无需展平）
#[test]
fn test_flatten_keep_first_dim_2d() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input_data = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );
    let x = graph.input(&input_data)?;
    let flat = x.flatten()?;

    flat.forward()?;

    let output = flat.value()?.unwrap();
    // 2D 张量保持不变
    assert_eq!(output.shape(), &[3, 4]);
    assert_eq!(&output, &input_data);

    Ok(())
}

/// 测试 keep_first_dim=false（完全展平为行向量）— 底层 API
///
/// [2, 3] → flatten(keep_first=false) → [1, 6]
#[test]
fn test_flatten_to_row_vector() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();
    let flat = inner
        .borrow_mut()
        .create_flatten_node(input.clone(), false, Some("flat"))
        .unwrap();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    input.set_value(Some(&input_data)).unwrap();
    flat.forward_recursive(1, false).unwrap();

    let output = flat.value().unwrap();
    assert_eq!(output.shape(), &[1, 6]);

    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    assert_eq!(&output, &expected);

    Ok(())
}

/// 测试方形矩阵完全展平 — 底层 API（keep_first=false）
///
/// [4, 4] → flatten(keep_first=false) → [1, 16]
#[test]
fn test_flatten_square_matrix() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 4], Some("input"))
        .unwrap();
    let flat = inner
        .borrow_mut()
        .create_flatten_node(input.clone(), false, Some("flat"))
        .unwrap();

    let input_data = Tensor::normal_seeded(0.0, 1.0, &[4, 4], 42);
    input.set_value(Some(&input_data)).unwrap();
    flat.forward_recursive(1, false).unwrap();

    let output = flat.value().unwrap();
    assert_eq!(output.shape(), &[1, 16]);

    // 验证元素一致（行优先顺序不变）
    for i in 0..16 {
        let in_row = i / 4;
        let in_col = i % 4;
        assert_abs_diff_eq!(input_data[[in_row, in_col]], output[[0, i]], epsilon = 1e-6);
    }

    Ok(())
}

/// 测试批量输入的前向传播（高层 API）
///
/// [4, 6] → flatten(keep_first=true) → [4, 6]（2D 不变）
#[test]
fn test_flatten_batch_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input_data = Tensor::normal_seeded(0.0, 1.0, &[4, 6], 42);
    let x = graph.input(&input_data)?;
    let flat = x.flatten()?;

    flat.forward()?;

    let output = flat.value()?.unwrap();
    assert_eq!(output.shape(), &[4, 6]);
    assert_eq!(&output, &input_data);

    Ok(())
}

/// 测试 Flatten + MatMul（典型 CNN→FC 场景，高层 API）
///
/// cnn_out [2, 8] → flatten → matmul(w [8, 4]) → [2, 4]
#[test]
fn test_flatten_with_matmul() -> Result<(), GraphError> {
    let graph = Graph::new();

    let batch_size = 2;
    let cnn_features = 8;
    let hidden_size = 4;

    let x = graph.input(&Tensor::normal_seeded(
        0.0,
        1.0,
        &[batch_size, cnn_features],
        100,
    ))?;
    let flat = x.flatten()?;
    let w = graph.parameter(
        &[cnn_features, hidden_size],
        Init::Normal {
            mean: 0.0,
            std: 0.1,
        },
        "w",
    )?;
    let h = flat.matmul(&w)?;

    h.forward()?;

    let output = h.value()?.unwrap();
    assert_eq!(output.shape(), &[batch_size, hidden_size]);

    Ok(())
}

/// 测试 Flatten → Reshape 链（底层 API，keep_first=false）
///
/// [3, 4] → flatten(keep_first=false) → [1, 12] → reshape → [4, 3]
#[test]
fn test_flatten_reshape_chain() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();
    let flat = inner
        .borrow_mut()
        .create_flatten_node(input.clone(), false, Some("flat"))
        .unwrap();
    let reshaped = inner
        .borrow_mut()
        .create_reshape_node(flat.clone(), &[4, 3], Some("reshaped"))
        .unwrap();

    let input_data = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );
    input.set_value(Some(&input_data)).unwrap();
    reshaped.forward_recursive(1, false).unwrap();

    let output = reshaped.value().unwrap();
    assert_eq!(output.shape(), &[4, 3]);

    // 验证元素顺序不变（行优先）
    for i in 0..12 {
        let in_row = i / 4;
        let in_col = i % 4;
        let out_row = i / 3;
        let out_col = i % 3;
        assert_abs_diff_eq!(
            input_data[[in_row, in_col]],
            output[[out_row, out_col]],
            epsilon = 1e-6
        );
    }

    Ok(())
}

/// 测试 Flatten 节点不能直接设置值
#[test]
fn test_flatten_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let flat = x.flatten().unwrap();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let err = flat.set_value(&test_value);
    assert!(err.is_err(), "Flatten 节点不应支持直接设值");
}

// ==================== 2. VJP 单元测试（底层 calc_grad_to_parent_index）====================
//
// Flatten VJP: grad = reshape(upstream, input_shape)
// 梯度只做形状变换，数值直接透传。

/// 测试 Flatten VJP（全 1 上游梯度）
///
/// input [2, 3] → flatten(keep_first=false) → [1, 6]
/// upstream = ones([1, 6]) → grad = ones([2, 3])
#[test]
fn test_flatten_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("input"))
        .unwrap();
    let flat = inner
        .borrow_mut()
        .create_flatten_node(input.clone(), false, Some("flat"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))
        .unwrap();
    flat.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[1, 6]);
    let grad = flat
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    // 梯度只是形状变化，数值直接透传
    assert_eq!(grad.shape(), &[2, 3]);
    assert_eq!(&grad, &Tensor::ones(&[2, 3]));

    Ok(())
}

/// 测试 Flatten VJP（非单位上游梯度）
///
/// upstream = [1,2,3,4,5,6] shape [1, 6] → grad = [[1,2,3],[4,5,6]] shape [2, 3]
#[test]
fn test_flatten_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("input"))
        .unwrap();
    let flat = inner
        .borrow_mut()
        .create_flatten_node(input.clone(), false, Some("flat"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))
        .unwrap();
    flat.forward_recursive(1, false).unwrap();

    // 非单位上游梯度
    let upstream = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    let grad = flat
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    // 梯度被 reshape 回输入形状，数值保持不变
    assert_eq!(grad.shape(), &[2, 3]);
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(&grad, &expected);

    Ok(())
}

// ==================== 3. E2E 反向传播测试（高层 Graph + Var API）====================

/// 测试 Flatten 端到端反向传播：sigmoid chain
///
/// x [2, 3] → flatten → sigmoid → MSE(target=zeros)
#[test]
fn test_flatten_backward_e2e_sigmoid_chain() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]))?;

    let flat = x.flatten()?; // [2, 6]... 实际 2D 不变 → [2, 3]
    let activated = flat.sigmoid();
    let target = graph.input(&Tensor::zeros(&[2, 3]))?;
    let loss = activated.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val > 0.0);

    // 梯度存在且形状正确
    let input_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    // 梯度非零（sigmoid(>0) > 0.5，loss > 0）
    assert!(input_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10));

    Ok(())
}

/// 测试 single sample 反向传播（底层 keep_first=false → 高层 matmul chain）
///
/// x [2, 3] → flatten(keep_first=false) → [1, 6] → matmul(w [6, 1]) → MSE
#[test]
fn test_flatten_single_sample_backward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 用底层 API 构建 flatten(keep_first=false) 部分
    let x_node = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("x"))
        .unwrap();
    let flat_node = inner
        .borrow_mut()
        .create_flatten_node(x_node.clone(), false, Some("flat"))
        .unwrap();
    let w_node = inner
        .borrow_mut()
        .create_parameter_node_seeded(&[6, 1], Some("w"), 100)
        .unwrap();
    let y_node = inner
        .borrow_mut()
        .create_mat_mul_node(vec![flat_node, w_node.clone()], Some("y"))
        .unwrap();
    let target_node = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1], Some("target"))
        .unwrap();
    let loss_node = inner
        .borrow_mut()
        .create_mse_mean_node(y_node, target_node.clone(), Some("loss"))
        .unwrap();

    // 设置值
    x_node
        .set_value(Some(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3])))
        .unwrap();
    target_node
        .set_value(Some(&Tensor::zeros(&[1, 1])))
        .unwrap();

    // 前向 + 反向
    let mut g = inner.borrow_mut();
    g.forward_via_node_inner(&loss_node)?;
    g.zero_grad()?;
    let loss_val = g.backward_via_node_inner(&loss_node)?;
    drop(g);

    assert!(loss_val >= 0.0);

    // 验证 w 的梯度形状正确
    let grad_w = w_node.grad().expect("w 应有 grad");
    assert_eq!(grad_w.shape(), &[6, 1]);

    Ok(())
}

// ==================== 4. 梯度累积测试（高层 Graph + Var API）====================

/// 测试 Flatten 梯度累积
///
/// 验证：多次 backward 之间 grad 累积，zero_grad 后重新计算应与首次一致。
#[test]
fn test_flatten_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]))?;

    let flat = x.flatten()?;
    let activated = flat.sigmoid();
    let target = graph.input(&Tensor::zeros(&[2, 3]))?;
    let loss = activated.mse_loss(&target)?;

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

// ==================== 5. 动态形状测试（KEEP AS-IS）====================

/// 测试 Flatten 节点的动态形状传播
///
/// Flatten 在 keep_first_dim=true 时支持动态 batch：
/// - 输入: [batch, c, h, w] 或 [batch, features]
/// - 输出: [batch, c*h*w] 或 [batch, features]
#[test]
fn test_flatten_dynamic_shape_propagation() {
    use crate::nn::Graph;
    use crate::nn::var::ops::VarShapeOps;

    let graph = Graph::new();

    // 创建 4D 输入：[batch, channels, height, width]
    // Input 节点默认支持动态 batch
    let x = graph.input(&Tensor::zeros(&[2, 3, 4, 4])).unwrap();

    // Flatten (keep_first_dim=true by default): [batch, 3, 4, 4] -> [batch, 48]
    let flat = x.flatten().unwrap();

    // 验证动态形状传播
    let dyn_shape = flat.dynamic_expected_shape();
    assert!(
        dyn_shape.is_dynamic(0),
        "batch 维度应该是动态的（因为输入是动态的且 keep_first_dim=true）"
    );
    assert!(!dyn_shape.is_dynamic(1), "第二维应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(48), "第二维应该是 3*4*4=48");
}

/// 测试 Flatten 在不同 batch_size 下的前向计算
#[test]
fn test_flatten_dynamic_batch_forward() {
    use crate::nn::Graph;
    use crate::nn::var::ops::VarShapeOps;

    let graph = Graph::new();

    // 创建 4D 输入：[batch, channels, height, width]
    let x = graph.input(&Tensor::zeros(&[2, 3, 4, 4])).unwrap();

    // Flatten (keep_first_dim=true by default)
    let flat = x.flatten().unwrap();

    // 第一次 forward：batch=2
    flat.forward().unwrap();
    let value1 = flat.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 48], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[5, 3, 4, 4])).unwrap();

    // 第二次 forward：batch=5
    flat.forward().unwrap();
    let value2 = flat.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[5, 48], "第二次 forward: batch=5");
}

/// 测试 Flatten 在不同 batch_size 下的反向传播
#[test]
fn test_flatten_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var::ops::{VarLossOps, VarShapeOps};

    let graph = Graph::new();

    // 创建 4D 输入
    let x = graph
        .input(&Tensor::normal_seeded(0.0, 1.0, &[2, 3, 4, 4], 42))
        .unwrap();

    // Flatten (keep_first_dim=true by default) -> MSE
    let flat = x.flatten().unwrap();
    let target = graph.input(&Tensor::zeros(&[2, 48])).unwrap();
    let loss = flat.mse_loss(&target).unwrap();

    // 第一次训练：batch=2
    loss.forward().unwrap();
    let loss_val1 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::normal_seeded(0.0, 1.0, &[5, 3, 4, 4], 100))
        .unwrap();
    target.set_value(&Tensor::zeros(&[5, 48])).unwrap();

    // 第二次训练：batch=5
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
}

// ==================== 6. Create API 测试（KEEP AS-IS）====================

use std::rc::Rc;

#[test]
fn test_create_flatten_node_keep_first() {
    use crate::nn::Graph;

    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 4D 输入: [2, 3, 4, 4] -> 展平为 [2, 48]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4, 4], Some("input"))
        .unwrap();

    let flat = inner
        .borrow_mut()
        .create_flatten_node(input.clone(), true, Some("flat"))
        .unwrap();

    assert_eq!(flat.shape(), vec![2, 48]);
    assert_eq!(flat.name(), Some("flat"));
    assert!(!flat.is_leaf());
    assert_eq!(flat.parents().len(), 1);
}

#[test]
fn test_create_flatten_node_no_keep_first() {
    use crate::nn::Graph;

    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 4D 输入: [2, 3, 4, 4] -> 完全展平为 [1, 96]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4, 4], None)
        .unwrap();

    let flat = inner
        .borrow_mut()
        .create_flatten_node(input, false, None)
        .unwrap();

    assert_eq!(flat.shape(), vec![1, 96]);
}

#[test]
fn test_create_flatten_node_2d_keep_first() {
    use crate::nn::Graph;

    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 2D 输入: [3, 4] -> keep_first_dim=true 时形状不变
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();

    let flat = inner
        .borrow_mut()
        .create_flatten_node(input, true, None)
        .unwrap();

    assert_eq!(flat.shape(), vec![3, 4]);
}

#[test]
fn test_create_flatten_node_2d_no_keep_first() {
    use crate::nn::Graph;

    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 2D 输入: [3, 4] -> keep_first_dim=false 时展平为 [1, 12]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();

    let flat = inner
        .borrow_mut()
        .create_flatten_node(input, false, None)
        .unwrap();

    assert_eq!(flat.shape(), vec![1, 12]);
}

#[test]
fn test_create_flatten_node_drop_releases() {
    use crate::nn::Graph;

    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_flat;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3, 4, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let flat = inner
            .borrow_mut()
            .create_flatten_node(input, true, None)
            .unwrap();
        weak_flat = Rc::downgrade(&flat);

        assert!(weak_flat.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_flat.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
