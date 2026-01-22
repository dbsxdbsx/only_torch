//! ModelState（模型状态管理器）单元测试
//!
//! 测试智能缓存机制：
//! - Tensor 输入：按形状缓存
//! - detached Var 输入：按形状缓存（只需要值，不需要梯度流）
//! - 非 detached Var 输入：不缓存，每次创建新路径（需要梯度流）

use crate::nn::{
    CrossEntropyLoss, Graph, Linear, ModelState, Rnn, VarActivationOps, VarLossOps, VarMatrixOps,
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
    let shapes = state.cached_shapes();
    assert!(shapes.contains(&vec![5, 1]), "应有特征形状 [5, 1]");
    assert!(shapes.contains(&vec![8, 1]), "应有特征形状 [8, 1]");
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
