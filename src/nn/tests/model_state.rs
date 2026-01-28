//! ModelState（模型状态管理器）单元测试
//!
//! 测试智能缓存机制：
//! - Tensor 输入：按形状缓存
//! - detached Var 输入：按形状缓存（只需要值，不需要梯度流）
//! - 非 detached Var 输入：不缓存，每次创建新路径（需要梯度流）

use crate::nn::{
    CrossEntropyLoss, Graph, Linear, ModelState, Module, Rnn, Var, VarActivationOps, VarLossOps,
    VarMatrixOps,
};
use crate::tensor::Tensor;

#[test]
fn test_model_state_basic() {
    let graph = Graph::new_with_seed(42);
    let fc1 = Linear::new(&graph, 2, 4, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 4, 2, true, "fc2").unwrap();
    let state = ModelState::new(&graph);
    let criterion = CrossEntropyLoss::new();

    // 首次调用
    let x1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let target1 = Tensor::new(&[1.0, 0.0], &[1, 2]);
    let output1 = state
        .forward(&x1, |input| Ok(fc2.forward(&fc1.forward(input).tanh())))
        .unwrap();

    assert!(state.is_initialized());
    assert_eq!(state.cache_size(), 1);

    let loss1 = criterion.forward(&output1, &target1).unwrap();
    let val1 = loss1.backward().unwrap();
    assert!(val1 > 0.0);

    // 第二次调用（相同形状，复用）
    let x2 = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let output2 = state
        .forward(&x2, |input| Ok(fc2.forward(&fc1.forward(input).tanh())))
        .unwrap();

    // 验证是同一个节点（复用）
    assert_eq!(output1.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 1);
}

/// 测试: 不同 batch_size 复用同一个缓存（类似 Keras）
///
/// 新机制：缓存键只用特征维度，忽略 batch（第一维）。
/// `[1, 2]` 和 `[2, 2]` 的特征维度都是 `[2]`，所以复用同一个缓存。
#[test]
fn test_model_state_multi_shape() {
    let graph = Graph::new_with_seed(42);
    let fc1 = Linear::new(&graph, 2, 4, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 4, 2, true, "fc2").unwrap();
    let state = ModelState::new(&graph);

    // 第一种形状 [1, 2]
    let x1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let output1 = state
        .forward(&x1, |input| Ok(fc2.forward(&fc1.forward(input).tanh())))
        .unwrap();

    // 第二种形状 [2, 2]（不同 batch_size，但特征维度相同）
    let x2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let output2 = state
        .forward(&x2, |input| Ok(fc2.forward(&fc1.forward(input).tanh())))
        .unwrap();

    // 新机制：相同特征维度 → 复用同一个节点！
    assert_eq!(output1.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 1); // 只有 1 个缓存（特征维度 [2]）

    // 再次使用第一种形状（复用）
    let x3 = Tensor::new(&[5.0, 6.0], &[1, 2]);
    let output3 = state
        .forward(&x3, |input| Ok(fc2.forward(&fc1.forward(input).tanh())))
        .unwrap();

    assert_eq!(output1.node_id(), output3.node_id());
    assert_eq!(state.cache_size(), 1); // 仍然只有 1 个缓存

    // 测试不同特征维度确实会创建不同缓存
    let fc1_3 = Linear::new(&graph, 3, 4, true, "fc1_3").unwrap();
    let fc2_3 = Linear::new(&graph, 4, 2, true, "fc2_3").unwrap();
    let state2 = ModelState::new(&graph);

    let y1 = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let out1 = state2
        .forward(&y1, |input| Ok(fc2_3.forward(&fc1_3.forward(input).tanh())))
        .unwrap();

    let y2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let out2 = state2
        .forward(&y2, |input| Ok(fc2_3.forward(&fc1_3.forward(input).tanh())))
        .unwrap();

    // 相同特征维度 [3] → 复用
    assert_eq!(out1.node_id(), out2.node_id());
    assert_eq!(state2.cache_size(), 1);
}

/// 测试: RNN 变长序列缓存
///
/// 对于 RNN 输入 [batch, seq_len, features]：
/// - batch（第一维）被忽略
/// - 缓存键是 [seq_len, features]
/// - 不同 seq_len 会创建不同缓存（因为展开次数不同）
///
/// 注意：由于 RNN 展开会创建依赖 batch 维度的中间节点，
/// 目前不同 batch_size 复用同一缓存的功能暂不支持 RNN。
/// 这个限制只影响 RNN 等展开式网络，普通 MLP 完全支持。
#[test]
fn test_model_state_var_len_rnn() {
    let graph = Graph::new_with_seed(42);
    let rnn = Rnn::new(&graph, 1, 8, "rnn").unwrap();
    let fc = Linear::new(&graph, 8, 2, true, "fc").unwrap();
    let state = ModelState::new(&graph);

    // seq_len = 5, batch = 2
    let x1 = Tensor::new(&vec![0.1f32; 10], &[2, 5, 1]);
    let output1 = state
        .forward(&x1, |input| {
            let h = rnn.forward(input)?;
            Ok(fc.forward(&h))
        })
        .unwrap();

    // seq_len = 8（不同长度）
    let x2 = Tensor::new(&vec![0.1f32; 16], &[2, 8, 1]);
    let output2 = state
        .forward(&x2, |input| {
            let h = rnn.forward(input)?;
            Ok(fc.forward(&h))
        })
        .unwrap();

    // 不同 seq_len → 不同缓存（因为 RNN 展开次数不同）
    assert_ne!(output1.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 2);

    // 检查缓存的特征形状（不含 batch）
    // 单输入时，缓存键是 [[feature_shape]]
    let shapes = state.cached_shapes();
    assert!(shapes.contains(&vec![vec![5, 1]]), "应有特征形状 [[5, 1]]");
    assert!(shapes.contains(&vec![vec![8, 1]]), "应有特征形状 [[8, 1]]");
}

#[test]
fn test_model_state_clear_cache() {
    let graph = Graph::new_with_seed(42);
    let fc = Linear::new(&graph, 2, 2, true, "fc").unwrap();
    let state = ModelState::new(&graph);

    let x = Tensor::new(&[1.0, 2.0], &[1, 2]);
    state.forward(&x, |input| Ok(fc.forward(input))).unwrap();
    assert_eq!(state.cache_size(), 1);

    state.clear_cache();
    assert_eq!(state.cache_size(), 0);
    assert!(!state.is_initialized());
}

// ============================================================================
// ForwardInput Var 输入测试（GradientRouter 统一缓存）
// ============================================================================

/// 测试: Var 输入使用 GradientRouter 统一缓存
///
/// 新机制：无论是 Tensor、detached Var 还是非 detached Var，
/// 都使用相同的缓存策略（按形状缓存），区别在于梯度路由。
#[test]
fn test_model_state_var_input_no_cache() {
    let graph = Graph::new_with_seed(42);
    let fc = Linear::new(&graph, 2, 2, true, "fc").unwrap();
    let state = ModelState::new(&graph);

    // 创建一个 Var 作为输入
    let x_tensor = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let x_var = graph.input(&x_tensor).unwrap();

    // 使用 Var 输入
    let output1 = state
        .forward(&x_var, |input| Ok(fc.forward(input)))
        .unwrap();

    // GradientRouter 机制：所有输入都使用缓存
    assert_eq!(state.cache_size(), 1);

    // 再次使用相同 Var
    let output2 = state
        .forward(&x_var, |input| Ok(fc.forward(input)))
        .unwrap();

    // 相同形状 → 复用缓存 → 相同的输出节点
    assert_eq!(output1.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 1);
}

/// 测试: 混合使用 Tensor 和 Var 输入（GradientRouter 统一缓存）
///
/// 新机制：Tensor 和 Var 使用相同的缓存，只要形状相同就复用。
#[test]
fn test_model_state_mixed_tensor_var_input() {
    let graph = Graph::new_with_seed(42);
    let fc = Linear::new(&graph, 2, 2, true, "fc").unwrap();
    let state = ModelState::new(&graph);

    // 先使用 Tensor 输入（会缓存）
    let x_tensor = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let output_tensor1 = state
        .forward(&x_tensor, |input| Ok(fc.forward(input)))
        .unwrap();
    assert_eq!(state.cache_size(), 1);

    // 再使用相同形状的 Tensor（复用缓存）
    let x_tensor2 = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let output_tensor2 = state
        .forward(&x_tensor2, |input| Ok(fc.forward(input)))
        .unwrap();
    assert_eq!(output_tensor1.node_id(), output_tensor2.node_id());
    assert_eq!(state.cache_size(), 1);

    // 使用相同形状的 Var 输入（GradientRouter 统一缓存 → 复用）
    let x_var = graph.input(&x_tensor).unwrap();
    let output_var = state
        .forward(&x_var, |input| Ok(fc.forward(input)))
        .unwrap();

    // GradientRouter 统一缓存：相同形状 → 相同输出节点
    assert_eq!(output_tensor1.node_id(), output_var.node_id());
    // 缓存大小不变
    assert_eq!(state.cache_size(), 1);
}

/// 测试: GAN 风格训练场景（GradientRouter 统一缓存 + 梯度路由）
///
/// 新机制：
/// - detached Var 和非 detached Var 都复用同一个缓存
/// - 区别在于 GradientRouter 的梯度路由设置
/// - detached: 无梯度路由（梯度不传回源 Var）
/// - 非 detached: 有梯度路由（梯度传回源 Var）
#[test]
fn test_model_state_gan_style_training() {
    let graph = Graph::new_with_seed(42);

    // 简化的 G 和 D
    let g_w = graph
        .parameter(&[1, 2], crate::nn::var::Init::Ones, "g_w")
        .unwrap();
    let d_fc = Linear::new(&graph, 2, 1, true, "d_fc").unwrap();
    let d_state = ModelState::new(&graph);

    // G 的输出
    let z = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let fake = z.matmul(&g_w).unwrap(); // [1,1] @ [1,2] -> [1,2]

    // === 训练 D ===
    // 使用 detach 后的 fake（无梯度路由）
    let fake_detached = fake.detach();
    assert!(
        fake_detached.is_detached(),
        "detach() 返回的 Var 应该是 detached"
    );

    let d_out_for_d = d_state
        .forward(&fake_detached, |input| Ok(d_fc.forward(input)))
        .unwrap();

    // 验证 d_state 有缓存
    assert_eq!(d_state.cache_size(), 1, "应该有一个缓存条目");

    // 计算 D 损失并反向传播
    let d_target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let d_loss = d_out_for_d.mse_loss(&d_target).unwrap();
    d_loss.backward().unwrap();

    // d_fc 应该有梯度
    // g_w 不应该有梯度（fake 被 detach，GradientRouter 无梯度路由）
    assert!(
        g_w.grad().unwrap().is_none(),
        "g_w 不应有梯度（fake 被 detach）"
    );

    // === 训练 G ===
    graph.zero_grad().unwrap();

    // 使用未 detach 的 fake（有梯度路由）
    assert!(!fake.is_detached(), "原始 fake 不应该是 detached");

    let d_out_for_g = d_state
        .forward(&fake, |input| Ok(d_fc.forward(input)))
        .unwrap();

    // GradientRouter 统一缓存：相同形状 → 相同输出节点
    assert_eq!(d_out_for_d.node_id(), d_out_for_g.node_id());
    // 缓存大小不变
    assert_eq!(d_state.cache_size(), 1);

    // 计算 G 损失并反向传播
    let g_target = graph.input(&Tensor::ones(&[1, 1])).unwrap();
    let g_loss = d_out_for_g.mse_loss(&g_target).unwrap();
    g_loss.backward().unwrap();

    // g_w 现在应该有梯度（GradientRouter 将梯度路由到 fake → g_w）
    assert!(g_w.grad().unwrap().is_some(), "g_w 应该有梯度（训练 G）");
}

/// 测试: Var 输入的值正确传递（通过 GradientRouter）
#[test]
fn test_model_state_var_input_value_passthrough() {
    let graph = Graph::new_with_seed(42);
    let fc = Linear::new(&graph, 2, 2, true, "fc").unwrap();
    let state = ModelState::new(&graph);

    // 创建 Var 输入
    let x_tensor = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let x_var = graph.input(&x_tensor).unwrap();

    // 使用 Var 输入（GradientRouter 会复制值）
    let output = state
        .forward(&x_var, |input| Ok(fc.forward(input)))
        .unwrap();

    // 验证输出值存在且形状正确
    let output_val = output.value().unwrap().unwrap();
    assert_eq!(output_val.shape(), &[1, 2]);

    // 验证缓存已创建
    assert_eq!(state.cache_size(), 1);
}

// ============================================================================
// detached Var 缓存行为测试
// ============================================================================

/// 测试: detached Var 使用缓存
#[test]
fn test_model_state_detached_var_uses_cache() {
    let graph = Graph::new_with_seed(42);
    let fc = Linear::new(&graph, 2, 2, true, "fc").unwrap();
    let state = ModelState::new(&graph);

    // 创建一个 Var 并 detach
    let x_tensor = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let x_var = graph.input(&x_tensor).unwrap();
    let x_detached = x_var.detach();

    // 验证 detach 状态
    assert!(x_detached.is_detached());
    assert!(!x_var.is_detached());

    // 使用 detached Var 输入
    let output1 = state
        .forward(&x_detached, |input| Ok(fc.forward(input)))
        .unwrap();

    // 缓存应该有 1 个条目（detached Var 使用缓存）
    assert_eq!(state.cache_size(), 1, "detached Var 应该使用缓存");

    // 再次使用相同形状的 detached Var
    let x_tensor2 = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let x_var2 = graph.input(&x_tensor2).unwrap();
    let x_detached2 = x_var2.detach();

    let output2 = state
        .forward(&x_detached2, |input| Ok(fc.forward(input)))
        .unwrap();

    // 应该是同一个节点（复用缓存）
    assert_eq!(
        output1.node_id(),
        output2.node_id(),
        "detached Var 应复用缓存"
    );
    assert_eq!(state.cache_size(), 1);
}

/// 测试: 非 detached Var 使用 GradientRouter 缓存并正确路由梯度
///
/// 新机制：非 detached Var 也使用缓存，但通过 GradientRouter 的梯度路由
/// 将梯度传回源 Var。
#[test]
fn test_model_state_non_detached_var_no_cache() {
    let graph = Graph::new_with_seed(42);
    let fc = Linear::new(&graph, 2, 2, true, "fc").unwrap();
    let state = ModelState::new(&graph);

    // 创建一个普通 Var（非 detached）
    let x_tensor = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let x_var = graph.input(&x_tensor).unwrap();
    assert!(!x_var.is_detached());

    // 使用非 detached Var 输入
    let output1 = state
        .forward(&x_var, |input| Ok(fc.forward(input)))
        .unwrap();

    // GradientRouter 机制：所有输入都使用缓存
    assert_eq!(state.cache_size(), 1);

    // 再次使用相同 Var
    let output2 = state
        .forward(&x_var, |input| Ok(fc.forward(input)))
        .unwrap();

    // 相同形状 → 复用缓存 → 相同的输出节点
    assert_eq!(output1.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 1);
}

/// 测试: detached Var 值正确复制到缓存
#[test]
fn test_model_state_detached_var_value_copy() {
    let graph = Graph::new_with_seed(42);

    // 使用简单 FC 层验证值传递（不能用 identity，因为 Input 节点不能被 forward）
    let fc = Linear::new(&graph, 2, 2, false, "fc").unwrap(); // 无 bias，简化验证
    let state = ModelState::new(&graph);

    // 第一次调用
    let x1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let var1 = graph.input(&x1).unwrap();
    let detached1 = var1.detach();

    let output1 = state
        .forward(&detached1, |input| Ok(fc.forward(input)))
        .unwrap();
    let val1 = output1.value().unwrap().unwrap();

    // 验证形状正确
    assert_eq!(val1.shape(), &[1, 2]);
    let val1_00 = val1[[0, 0]];
    let val1_01 = val1[[0, 1]];

    // 第二次调用（不同的值，相同形状）
    let x2 = Tensor::new(&[5.0, 6.0], &[1, 2]);
    let var2 = graph.input(&x2).unwrap();
    let detached2 = var2.detach();

    let output2 = state
        .forward(&detached2, |input| Ok(fc.forward(input)))
        .unwrap();

    // 应该是同一个节点（缓存复用）
    assert_eq!(output1.node_id(), output2.node_id());

    // 验证值已更新（不同于第一次）
    let val2 = output2.value().unwrap().unwrap();
    assert_eq!(val2.shape(), &[1, 2]);
    let val2_00 = val2[[0, 0]];
    let val2_01 = val2[[0, 1]];

    // 由于输入不同，输出也应该不同
    assert!((val2_00 - val1_00).abs() > 1e-6 || (val2_01 - val1_01).abs() > 1e-6);
}

/// 测试: GAN 训练多批次效率
#[test]
fn test_model_state_gan_multi_batch_efficiency() {
    let graph = Graph::new_with_seed(42);
    let d_fc = Linear::new(&graph, 2, 1, true, "d_fc").unwrap();
    let d_state = ModelState::new(&graph);

    // 模拟多批次 GAN 训练
    for batch_idx in 0..5 {
        // 创建当前批次的 fake 输出
        let z = Tensor::new(&[batch_idx as f32, (batch_idx + 1) as f32], &[1, 2]);
        let z_var = graph.input(&z).unwrap();
        let fake = z_var.clone(); // 模拟 G.forward()

        // D 训练（使用 detached）
        let fake_detached = fake.detach();
        let d_out = d_state
            .forward(&fake_detached, |input| Ok(d_fc.forward(input)))
            .unwrap();

        // 验证输出形状正确
        let d_val = d_out.value().unwrap().unwrap();
        assert_eq!(d_val.shape(), &[1, 1]);
    }

    // 关键验证：5 批次后，缓存仍然只有 1 个条目
    // （detached Var 复用缓存，而非每批次创建新路径）
    assert_eq!(
        d_state.cache_size(),
        1,
        "多批次 GAN 训练应复用缓存，不应膨胀"
    );
}

// ============================================================================
// 多输入 forward2/forward3 测试
// ============================================================================

/// 测试: forward2 基本功能
#[test]
fn test_model_state_forward2_basic() {
    let graph = Graph::new_with_seed(42);
    let fc1 = Linear::new(&graph, 2, 4, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 3, 4, true, "fc2").unwrap();
    let fc_out = Linear::new(&graph, 8, 2, true, "fc_out").unwrap();
    let state = ModelState::new(&graph);

    // 双输入
    let x1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let x2 = Tensor::new(&[3.0, 4.0, 5.0], &[1, 3]);

    let output = state
        .forward2(&x1, &x2, |a, b| {
            let h1 = fc1.forward(a).relu();
            let h2 = fc2.forward(b).relu();
            // 拼接两个特征（使用 Var::stack concat 模式）
            let combined = Var::stack(&[&h1, &h2], 1, false)?;
            Ok(fc_out.forward(&combined))
        })
        .unwrap();

    assert!(state.is_initialized());
    assert_eq!(state.cache_size(), 1);

    // 验证输出形状
    let output_val = output.value().unwrap().unwrap();
    assert_eq!(output_val.shape(), &[1, 2]);

    // 第二次调用（复用缓存）
    let x1_2 = Tensor::new(&[5.0, 6.0], &[1, 2]);
    let x2_2 = Tensor::new(&[7.0, 8.0, 9.0], &[1, 3]);

    let output2 = state
        .forward2(&x1_2, &x2_2, |a, b| {
            let h1 = fc1.forward(a).relu();
            let h2 = fc2.forward(b).relu();
            let combined = Var::stack(&[&h1, &h2], 1, false)?;
            Ok(fc_out.forward(&combined))
        })
        .unwrap();

    // 应复用相同节点
    assert_eq!(output.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 1);
}

/// 测试: forward2 缓存键区分不同特征形状
#[test]
fn test_model_state_forward2_different_shapes() {
    let graph = Graph::new_with_seed(42);
    let fc1_a = Linear::new(&graph, 2, 4, true, "fc1_a").unwrap();
    let fc2_a = Linear::new(&graph, 3, 4, true, "fc2_a").unwrap();
    let fc1_b = Linear::new(&graph, 4, 4, true, "fc1_b").unwrap();
    let fc2_b = Linear::new(&graph, 5, 4, true, "fc2_b").unwrap();
    let state = ModelState::new(&graph);

    // 第一种输入形状组合 (2, 3)
    let x1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let x2 = Tensor::new(&[3.0, 4.0, 5.0], &[1, 3]);

    let output1 = state
        .forward2(&x1, &x2, |a, b| {
            let h1 = fc1_a.forward(a);
            let h2 = fc2_a.forward(b);
            Ok(h1 + &h2)
        })
        .unwrap();

    assert_eq!(state.cache_size(), 1);

    // 第二种输入形状组合 (4, 5)
    let y1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0], &[1, 5]);

    let output2 = state
        .forward2(&y1, &y2, |a, b| {
            let h1 = fc1_b.forward(a);
            let h2 = fc2_b.forward(b);
            Ok(h1 + &h2)
        })
        .unwrap();

    // 不同形状 → 不同缓存
    assert_ne!(output1.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 2);

    // 验证缓存键
    let shapes = state.cached_shapes();
    assert!(shapes.contains(&vec![vec![2], vec![3]]));
    assert!(shapes.contains(&vec![vec![4], vec![5]]));
}

/// 测试: forward2 混合输入类型（Tensor + Var）
#[test]
fn test_model_state_forward2_mixed_input_types() {
    let graph = Graph::new_with_seed(42);
    let fc1 = Linear::new(&graph, 2, 4, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 2, 4, true, "fc2").unwrap();
    let state = ModelState::new(&graph);

    // Tensor + Var 输入
    let x_tensor = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let y_tensor = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let y_var = graph.input(&y_tensor).unwrap();

    let output1 = state
        .forward2(&x_tensor, &y_var, |a, b| {
            let h1 = fc1.forward(a);
            let h2 = fc2.forward(b);
            Ok(h1 + &h2)
        })
        .unwrap();

    assert_eq!(state.cache_size(), 1);

    // 再次使用相同形状（Tensor + detached Var）
    let x_tensor2 = Tensor::new(&[5.0, 6.0], &[1, 2]);
    let y_var2 = graph.input(&Tensor::new(&[7.0, 8.0], &[1, 2])).unwrap();
    let y_detached = y_var2.detach();

    let output2 = state
        .forward2(&x_tensor2, &y_detached, |a, b| {
            let h1 = fc1.forward(a);
            let h2 = fc2.forward(b);
            Ok(h1 + &h2)
        })
        .unwrap();

    // 相同特征形状 → 复用缓存
    assert_eq!(output1.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 1);
}

/// 测试: forward2 梯度路由（双输入）
///
/// 使用 Tensor 作为输入，验证参数能正确获得梯度。
#[test]
fn test_model_state_forward2_gradient_routing() {
    let graph = Graph::new_with_seed(42);

    // 创建两个可训练参数
    let w1 = graph
        .parameter(&[2, 4], crate::nn::var::Init::Ones, "w1")
        .unwrap();
    let w2 = graph
        .parameter(&[3, 4], crate::nn::var::Init::Ones, "w2")
        .unwrap();
    let fc_out = Linear::new(&graph, 8, 1, true, "fc_out").unwrap();
    let state = ModelState::new(&graph);

    // 使用 Tensor 作为双输入
    let x1_tensor = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let x2_tensor = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);

    let output = state
        .forward2(&x1_tensor, &x2_tensor, |a, b| {
            let h1 = a.matmul(&w1)?;
            let h2 = b.matmul(&w2)?;
            let combined = Var::stack(&[&h1, &h2], 1, false)?;
            Ok(fc_out.forward(&combined))
        })
        .unwrap();

    // 反向传播
    let target = Tensor::ones(&[1, 1]);
    let target_var = graph.input(&target).unwrap();
    let loss = output.mse_loss(&target_var).unwrap();
    loss.backward().unwrap();

    // 两个权重都应该有梯度
    assert!(w1.grad().unwrap().is_some(), "w1 应有梯度");
    assert!(w2.grad().unwrap().is_some(), "w2 应有梯度");
}

/// 测试: forward2 部分 detach（Tensor + detached Var）
///
/// 验证混合输入类型时参数仍能获得梯度。
#[test]
fn test_model_state_forward2_partial_detach() {
    let graph = Graph::new_with_seed(42);

    let w1 = graph
        .parameter(&[2, 4], crate::nn::var::Init::Ones, "w1")
        .unwrap();
    let w2 = graph
        .parameter(&[2, 4], crate::nn::var::Init::Ones, "w2")
        .unwrap();
    let state = ModelState::new(&graph);

    // x1 是 Tensor，x2 是 detached Var（模拟 GAN 场景）
    let x1_tensor = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let x2_tensor = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let x2_var = graph.input(&x2_tensor).unwrap();
    let x2_detached = x2_var.detach();

    let output = state
        .forward2(&x1_tensor, &x2_detached, |a, b| {
            let h1 = a.matmul(&w1)?;
            let h2 = b.matmul(&w2)?;
            Ok(h1 + &h2)
        })
        .unwrap();

    let target = Tensor::zeros(&[1, 4]);
    let target_var = graph.input(&target).unwrap();
    let loss = output.mse_loss(&target_var).unwrap();
    loss.backward().unwrap();

    // w1 和 w2 都应该有梯度（它们本身是参数）
    assert!(w1.grad().unwrap().is_some(), "w1 应有梯度");
    assert!(w2.grad().unwrap().is_some(), "w2 应有梯度（作为参数）");
}

/// 测试: forward3 基本功能
#[test]
fn test_model_state_forward3_basic() {
    let graph = Graph::new_with_seed(42);
    let fc1 = Linear::new(&graph, 2, 4, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 3, 4, true, "fc2").unwrap();
    let fc3 = Linear::new(&graph, 4, 4, true, "fc3").unwrap();
    let fc_out = Linear::new(&graph, 12, 2, true, "fc_out").unwrap();
    let state = ModelState::new(&graph);

    // 三输入
    let x1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let x2 = Tensor::new(&[3.0, 4.0, 5.0], &[1, 3]);
    let x3 = Tensor::new(&[6.0, 7.0, 8.0, 9.0], &[1, 4]);

    let output = state
        .forward3(&x1, &x2, &x3, |a, b, c| {
            let h1 = fc1.forward(a).relu();
            let h2 = fc2.forward(b).relu();
            let h3 = fc3.forward(c).relu();
            // 拼接三个特征（使用 Var::stack concat 模式）
            let combined = Var::stack(&[&h1, &h2, &h3], 1, false)?;
            Ok(fc_out.forward(&combined))
        })
        .unwrap();

    assert!(state.is_initialized());
    assert_eq!(state.cache_size(), 1);

    // 验证输出形状
    let output_val = output.value().unwrap().unwrap();
    assert_eq!(output_val.shape(), &[1, 2]);

    // 第二次调用（复用缓存）
    let y1 = Tensor::new(&[10.0, 11.0], &[1, 2]);
    let y2 = Tensor::new(&[12.0, 13.0, 14.0], &[1, 3]);
    let y3 = Tensor::new(&[15.0, 16.0, 17.0, 18.0], &[1, 4]);

    let output2 = state
        .forward3(&y1, &y2, &y3, |a, b, c| {
            let h1 = fc1.forward(a).relu();
            let h2 = fc2.forward(b).relu();
            let h3 = fc3.forward(c).relu();
            let combined = Var::stack(&[&h1, &h2, &h3], 1, false)?;
            Ok(fc_out.forward(&combined))
        })
        .unwrap();

    // 应复用相同节点
    assert_eq!(output.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 1);

    // 验证缓存键
    let shapes = state.cached_shapes();
    assert!(shapes.contains(&vec![vec![2], vec![3], vec![4]]));
}

/// 测试: forward3 梯度路由
///
/// 使用 Tensor 作为输入，验证所有参数能正确获得梯度。
#[test]
fn test_model_state_forward3_gradient_routing() {
    let graph = Graph::new_with_seed(42);

    let w1 = graph
        .parameter(&[2, 4], crate::nn::var::Init::Ones, "w1")
        .unwrap();
    let w2 = graph
        .parameter(&[2, 4], crate::nn::var::Init::Ones, "w2")
        .unwrap();
    let w3 = graph
        .parameter(&[2, 4], crate::nn::var::Init::Ones, "w3")
        .unwrap();
    let state = ModelState::new(&graph);

    // 三个 Tensor 输入
    let x1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let x2 = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let x3 = Tensor::new(&[5.0, 6.0], &[1, 2]);

    let output = state
        .forward3(&x1, &x2, &x3, |a, b, c| {
            let h1 = a.matmul(&w1)?;
            let h2 = b.matmul(&w2)?;
            let h3 = c.matmul(&w3)?;
            Ok((h1 + &h2) + &h3)
        })
        .unwrap();

    let target = Tensor::zeros(&[1, 4]);
    let target_var = graph.input(&target).unwrap();
    let loss = output.mse_loss(&target_var).unwrap();
    loss.backward().unwrap();

    // 三个权重都应该有梯度
    assert!(w1.grad().unwrap().is_some(), "w1 应有梯度");
    assert!(w2.grad().unwrap().is_some(), "w2 应有梯度");
    assert!(w3.grad().unwrap().is_some(), "w3 应有梯度");
}

// ============================================================================
// 多输出测试
// ============================================================================

/// 测试: 多输出基本功能（返回元组）
///
/// 验证 forward 闭包可以返回元组类型。
#[test]
fn test_model_state_multi_output_basic() {
    let graph = Graph::new_with_seed(42);

    // 共享层 + 两个输出头
    let shared = Linear::new(&graph, 2, 8, true, "shared").unwrap();
    let head1 = Linear::new(&graph, 8, 1, true, "head1").unwrap();
    let head2 = Linear::new(&graph, 8, 1, true, "head2").unwrap();
    let state = ModelState::new(&graph);

    let x = Tensor::new(&[1.0, 2.0], &[1, 2]);

    // 返回元组 (Var, Var)
    let (out1, out2) = state
        .forward(&x, |input| {
            let feat = shared.forward(input).relu();
            let o1 = head1.forward(&feat);
            let o2 = head2.forward(&feat);
            Ok((o1, o2))
        })
        .unwrap();

    assert!(state.is_initialized());
    assert_eq!(state.cache_size(), 1);

    // 验证两个输出形状
    let out1_val = out1.value().unwrap().unwrap();
    let out2_val = out2.value().unwrap().unwrap();
    assert_eq!(out1_val.shape(), &[1, 1]);
    assert_eq!(out2_val.shape(), &[1, 1]);

    // 第二次调用（复用缓存）
    let y = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let (out1_2, out2_2) = state
        .forward(&y, |input| {
            let feat = shared.forward(input).relu();
            Ok((head1.forward(&feat), head2.forward(&feat)))
        })
        .unwrap();

    // 应复用相同节点
    assert_eq!(out1.node_id(), out1_2.node_id());
    assert_eq!(out2.node_id(), out2_2.node_id());
    assert_eq!(state.cache_size(), 1);
}

/// 测试: 多输出 + 两个回归头（MSE + MSE）
///
/// 验证多任务学习场景：共享特征层 + 两个回归任务。
#[test]
fn test_model_state_multi_output_dual_regression() {
    let graph = Graph::new_with_seed(42);

    // 共享层 + 两个回归头
    let shared = Linear::new(&graph, 2, 8, true, "shared").unwrap();
    let reg_head1 = Linear::new(&graph, 8, 2, true, "reg_head1").unwrap();
    let reg_head2 = Linear::new(&graph, 8, 1, true, "reg_head2").unwrap();
    let state = ModelState::new(&graph);

    let x = Tensor::new(&[1.0, -0.5], &[1, 2]);

    // 双输出 forward
    let (out1, out2) = state
        .forward(&x, |input| {
            let feat = shared.forward(input).relu();
            let o1 = reg_head1.forward(&feat);
            let o2 = reg_head2.forward(&feat);
            Ok((o1, o2))
        })
        .unwrap();

    // 两个回归目标
    let target1 = graph.input(&Tensor::zeros(&[1, 2])).unwrap();
    let target2 = graph.input(&Tensor::zeros(&[1, 1])).unwrap();

    // 计算两个 MSE loss
    let loss1 = out1.mse_loss(&target1).unwrap();
    let loss2 = out2.mse_loss(&target2).unwrap();

    // 组合 loss
    let total_loss = &loss1 + &loss2;
    total_loss.backward().unwrap();

    // 验证所有参数都有梯度
    for param in shared.parameters() {
        assert!(param.grad().unwrap().is_some(), "shared 层参数应有梯度");
    }
    for param in reg_head1.parameters() {
        assert!(param.grad().unwrap().is_some(), "reg_head1 参数应有梯度");
    }
    for param in reg_head2.parameters() {
        assert!(param.grad().unwrap().is_some(), "reg_head2 参数应有梯度");
    }
}

/// 测试: 多输出 + 两个分类头（CrossEntropy + CrossEntropy）
///
/// 验证多任务学习场景：共享特征层 + 两个分类任务。
#[test]
fn test_model_state_multi_output_dual_classification() {
    let graph = Graph::new_with_seed(42);

    // 共享层 + 两个分类头（不同类别数）
    let shared = Linear::new(&graph, 4, 8, true, "shared").unwrap();
    let cls_head1 = Linear::new(&graph, 8, 3, true, "cls_head1").unwrap(); // 3 分类
    let cls_head2 = Linear::new(&graph, 8, 5, true, "cls_head2").unwrap(); // 5 分类
    let state = ModelState::new(&graph);

    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);

    // 双输出 forward
    let (logits1, logits2) = state
        .forward(&x, |input| {
            let feat = shared.forward(input).relu();
            let l1 = cls_head1.forward(&feat);
            let l2 = cls_head2.forward(&feat);
            Ok((l1, l2))
        })
        .unwrap();

    // 两个分类目标（one-hot 编码）
    // 第一个任务：类别 1（3 分类）
    let label1 = graph
        .input(&Tensor::new(&[0.0, 1.0, 0.0], &[1, 3]))
        .unwrap();
    // 第二个任务：类别 3（5 分类）
    let label2 = graph
        .input(&Tensor::new(&[0.0, 0.0, 0.0, 1.0, 0.0], &[1, 5]))
        .unwrap();

    // 计算两个 CrossEntropy loss
    let loss1 = logits1.cross_entropy(&label1).unwrap();
    let loss2 = logits2.cross_entropy(&label2).unwrap();

    // 组合 loss
    let total_loss = &loss1 + &loss2;
    total_loss.backward().unwrap();

    // 验证所有参数都有梯度
    for param in shared.parameters() {
        assert!(param.grad().unwrap().is_some(), "shared 层参数应有梯度");
    }
    for param in cls_head1.parameters() {
        assert!(param.grad().unwrap().is_some(), "cls_head1 参数应有梯度");
    }
    for param in cls_head2.parameters() {
        assert!(param.grad().unwrap().is_some(), "cls_head2 参数应有梯度");
    }
}

/// 测试: 多输出 + 混合头（CrossEntropy + MSE）
///
/// 验证多任务学习场景：共享特征层 + 分类任务 + 回归任务。
/// 这是实际中常见的场景，如目标检测（分类 + 回归）。
#[test]
fn test_model_state_multi_output_mixed_cls_reg() {
    let graph = Graph::new_with_seed(42);

    // 共享层 + 分类头 + 回归头
    let shared = Linear::new(&graph, 4, 8, true, "shared").unwrap();
    let cls_head = Linear::new(&graph, 8, 3, true, "cls_head").unwrap(); // 3 分类
    let reg_head = Linear::new(&graph, 8, 2, true, "reg_head").unwrap(); // 2 维回归
    let state = ModelState::new(&graph);

    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);

    // 双输出 forward
    let (cls_logits, reg_pred) = state
        .forward(&x, |input| {
            let feat = shared.forward(input).relu();
            let cls = cls_head.forward(&feat);
            let reg = reg_head.forward(&feat);
            Ok((cls, reg))
        })
        .unwrap();

    // 分类目标（one-hot）和回归目标
    let cls_label = graph
        .input(&Tensor::new(&[1.0, 0.0, 0.0], &[1, 3]))
        .unwrap();
    let reg_target = graph.input(&Tensor::new(&[0.5, -0.5], &[1, 2])).unwrap();

    // 计算两种不同类型的 loss
    let cls_loss = cls_logits.cross_entropy(&cls_label).unwrap();
    let reg_loss = reg_pred.mse_loss(&reg_target).unwrap();

    // 组合 loss（可以加权，这里简单相加）
    let total_loss = &cls_loss + &reg_loss;
    total_loss.backward().unwrap();

    // 验证所有参数都有梯度
    for param in shared.parameters() {
        assert!(param.grad().unwrap().is_some(), "shared 层参数应有梯度");
    }
    for param in cls_head.parameters() {
        assert!(param.grad().unwrap().is_some(), "cls_head 参数应有梯度");
    }
    for param in reg_head.parameters() {
        assert!(param.grad().unwrap().is_some(), "reg_head 参数应有梯度");
    }
}

/// 测试: 多输出 + 三元组
///
/// 验证返回三个输出的场景（如 VAE: recon, mu, log_var）。
#[test]
fn test_model_state_multi_output_triple() {
    let graph = Graph::new_with_seed(42);

    let encoder = Linear::new(&graph, 4, 8, true, "encoder").unwrap();
    let mu_layer = Linear::new(&graph, 8, 2, true, "mu").unwrap();
    let logvar_layer = Linear::new(&graph, 8, 2, true, "logvar").unwrap();
    let decoder = Linear::new(&graph, 2, 4, true, "decoder").unwrap();
    let state = ModelState::new(&graph);

    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);

    // 返回三元组
    let (recon, mu, log_var) = state
        .forward(&x, |input| {
            let h = encoder.forward(input).relu();
            let mu = mu_layer.forward(&h);
            let log_var = logvar_layer.forward(&h);
            // 简化：直接用 mu 作为 z（省略重参数化）
            let recon = decoder.forward(&mu);
            Ok((recon, mu, log_var))
        })
        .unwrap();

    // 验证三个输出形状
    assert_eq!(recon.value().unwrap().unwrap().shape(), &[1, 4]);
    assert_eq!(mu.value().unwrap().unwrap().shape(), &[1, 2]);
    assert_eq!(log_var.value().unwrap().unwrap().shape(), &[1, 2]);

    // 验证可以对任意输出计算 loss
    let target = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]))
        .unwrap();
    let recon_loss = recon.mse_loss(&target).unwrap();
    recon_loss.backward().unwrap();

    // encoder 应有梯度（通过 mu -> recon 传播）
    for param in encoder.parameters() {
        assert!(param.grad().unwrap().is_some(), "encoder 参数应有梯度");
    }
}

/// 测试: forward2 多输出
///
/// 验证双输入方法也支持多输出返回。
#[test]
fn test_model_state_forward2_multi_output() {
    let graph = Graph::new_with_seed(42);

    // Siamese 风格：双输入共享编码器，返回两个特征向量
    let encoder = Linear::new(&graph, 2, 4, true, "encoder").unwrap();
    let state = ModelState::new(&graph);

    let x1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let x2 = Tensor::new(&[3.0, 4.0], &[1, 2]);

    // 双输入双输出
    let (feat1, feat2) = state
        .forward2(&x1, &x2, |a, b| {
            let f1 = encoder.forward(a).relu();
            let f2 = encoder.forward(b).relu();
            Ok((f1, f2))
        })
        .unwrap();

    // 验证两个输出形状
    assert_eq!(feat1.value().unwrap().unwrap().shape(), &[1, 4]);
    assert_eq!(feat2.value().unwrap().unwrap().shape(), &[1, 4]);

    // 验证缓存复用
    let y1 = Tensor::new(&[5.0, 6.0], &[1, 2]);
    let y2 = Tensor::new(&[7.0, 8.0], &[1, 2]);

    let (feat1_2, feat2_2) = state
        .forward2(&y1, &y2, |a, b| {
            Ok((encoder.forward(a).relu(), encoder.forward(b).relu()))
        })
        .unwrap();

    assert_eq!(feat1.node_id(), feat1_2.node_id());
    assert_eq!(feat2.node_id(), feat2_2.node_id());
    assert_eq!(state.cache_size(), 1);
}
