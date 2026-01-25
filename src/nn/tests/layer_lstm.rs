/*
 * @Author       : 老董
 * @Date         : 2026-01-21
 * @Description  : Lstm Layer 单元测试（展开式设计）
 *
 * 测试覆盖：
 * - 基础功能：创建、参数形状、Module trait
 * - 前向传播：形状验证、数值计算
 * - 反向传播：梯度流动
 * - ModelState 集成：智能缓存
 */

use crate::nn::layer::{Linear, Lstm};
use crate::nn::{CrossEntropyLoss, Graph, GraphError, ModelState, Module};
use crate::tensor::Tensor;

// ==================== 基础功能测试 ====================

/// 测试 Lstm 层创建
#[test]
fn test_lstm_new() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 10, 20, "lstm1")?;

    assert_eq!(lstm.input_size(), 10);
    assert_eq!(lstm.hidden_size(), 20);

    // 验证参数存在
    assert!(lstm.w_ii().value()?.is_some());
    assert!(lstm.w_hi().value()?.is_some());
    assert!(lstm.b_i().value()?.is_some());
    assert!(lstm.w_if().value()?.is_some());
    assert!(lstm.w_hf().value()?.is_some());
    assert!(lstm.b_f().value()?.is_some());
    assert!(lstm.w_ig().value()?.is_some());
    assert!(lstm.w_hg().value()?.is_some());
    assert!(lstm.b_g().value()?.is_some());
    assert!(lstm.w_io().value()?.is_some());
    assert!(lstm.w_ho().value()?.is_some());
    assert!(lstm.b_o().value()?.is_some());

    Ok(())
}

/// 测试参数形状
#[test]
fn test_lstm_parameters() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let input_size = 4;
    let hidden_size = 6;
    let lstm = Lstm::new(&graph, input_size, hidden_size, "lstm1")?;

    // 验证输入门权重形状
    assert_eq!(
        lstm.w_ii().value()?.unwrap().shape(),
        &[input_size, hidden_size]
    );
    assert_eq!(
        lstm.w_hi().value()?.unwrap().shape(),
        &[hidden_size, hidden_size]
    );
    assert_eq!(lstm.b_i().value()?.unwrap().shape(), &[1, hidden_size]);

    // 验证遗忘门偏置初始化为 1
    let b_f = lstm.b_f().value()?.unwrap();
    assert!((b_f[[0, 0]] - 1.0).abs() < 1e-6);

    Ok(())
}

/// 测试 Module trait
#[test]
fn test_lstm_module_trait() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 10, 20, "lstm")?;

    let params = lstm.parameters();
    assert_eq!(params.len(), 12); // 4 门 × 3 参数

    Ok(())
}

// ==================== 前向传播测试 ====================

/// 测试前向传播输出形状
#[test]
fn test_lstm_forward_shape() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 5, 10, "lstm")?;

    // 输入: [batch=2, seq_len=8, input=5]
    let x = graph.zeros(&[2, 8, 5])?;
    let h = lstm.forward(&x)?;
    h.forward()?;

    // 输出: [batch=2, hidden=10]
    let output = h.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 10]);

    Ok(())
}

/// 测试不同序列长度
#[test]
fn test_lstm_various_seq_len() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 3, 8, "lstm")?;

    for seq_len in [1, 5, 10, 20] {
        let x = graph.zeros(&[4, seq_len, 3])?;
        let h = lstm.forward(&x)?;
        h.forward()?;

        let output = h.value()?.unwrap();
        assert_eq!(output.shape(), &[4, 8]);
    }

    Ok(())
}

/// 测试输入维度验证
#[test]
fn test_lstm_forward_invalid_dim() {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 5, 10, "lstm").unwrap();

    // 2D 输入应该报错
    let x = graph.zeros(&[2, 5]).unwrap();
    let result = lstm.forward(&x);
    assert!(result.is_err());
}

/// 测试 input_size 不匹配
#[test]
fn test_lstm_forward_input_size_mismatch() {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 5, 10, "lstm").unwrap();

    // input_size=3 不匹配期望的 5
    let x = graph.zeros(&[2, 8, 3]).unwrap();
    let result = lstm.forward(&x);
    assert!(result.is_err());
}

// ==================== 反向传播测试 ====================

/// 测试反向传播梯度流动
#[test]
fn test_lstm_backward() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 3, 4, "lstm")?;
    let fc = Linear::new(&graph, 4, 2, true, "fc")?;

    // 构建图
    let x = graph.randn(&[2, 5, 3])?;
    let h = lstm.forward(&x)?;
    let logits = fc.forward(&h);

    // 创建标签
    let labels = graph.input(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]))?;

    // 计算 loss
    use crate::nn::VarLossOps;
    let loss = logits.cross_entropy(&labels)?;

    // 反向传播
    loss.backward()?;

    // 验证所有参数都有梯度
    assert!(lstm.w_ii().grad()?.is_some());
    assert!(lstm.w_hi().grad()?.is_some());
    assert!(lstm.b_i().grad()?.is_some());
    assert!(lstm.w_if().grad()?.is_some());
    assert!(lstm.w_hf().grad()?.is_some());
    assert!(lstm.b_f().grad()?.is_some());
    assert!(lstm.w_ig().grad()?.is_some());
    assert!(lstm.w_hg().grad()?.is_some());
    assert!(lstm.b_g().grad()?.is_some());
    assert!(lstm.w_io().grad()?.is_some());
    assert!(lstm.w_ho().grad()?.is_some());
    assert!(lstm.b_o().grad()?.is_some());
    assert!(fc.weights().grad()?.is_some());

    Ok(())
}

/// 测试梯度传播到输入（通过 ModelState 的 input 节点）
#[test]
fn test_lstm_gradient_to_input() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 2, 3, "lstm")?;
    let state = ModelState::new(&graph);

    let x = Tensor::new(&vec![0.1f32; 8], &[1, 4, 2]);
    let h = state.forward(&x, |input| lstm.forward(input))?;

    // 简单的 L2 loss
    use crate::nn::VarLossOps;
    let target = graph.zeros(&[1, 3])?;
    let loss = h.mse_loss(&target)?;
    loss.backward()?;

    // LSTM 参数应该有梯度
    assert!(lstm.w_ii().grad()?.is_some());

    Ok(())
}

// ==================== 数值正确性测试 ====================

/// 测试单时间步计算
#[test]
fn test_lstm_single_timestep_value() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 2, 3, "lstm")?;

    // 设置简单权重
    let zero_w = Tensor::zeros(&[2, 3]);
    let zero_hw = Tensor::zeros(&[3, 3]);
    let zero_b = Tensor::zeros(&[1, 3]);

    lstm.w_ii().set_value(&zero_w)?;
    lstm.w_hi().set_value(&zero_hw)?;
    lstm.b_i().set_value(&zero_b)?;
    lstm.w_if().set_value(&zero_w)?;
    lstm.w_hf().set_value(&zero_hw)?;
    lstm.b_f().set_value(&Tensor::ones(&[1, 3]))?; // 遗忘门偏置
    lstm.w_ig().set_value(&zero_w)?;
    lstm.w_hg().set_value(&zero_hw)?;
    lstm.b_g().set_value(&zero_b)?;
    lstm.w_io().set_value(&zero_w)?;
    lstm.w_ho().set_value(&zero_hw)?;
    lstm.b_o().set_value(&zero_b)?;

    // 输入
    let x = graph.zeros(&[1, 1, 2])?;
    let h = lstm.forward(&x)?;
    h.forward()?;

    let output = h.value()?.unwrap();
    // 所有权重为 0 时：
    // i = σ(0) = 0.5, f = σ(1) ≈ 0.73, g = tanh(0) = 0, o = σ(0) = 0.5
    // c = 0.73 * 0 + 0.5 * 0 = 0
    // h = 0.5 * tanh(0) = 0
    for i in 0..3 {
        assert!(output[[0, i]].abs() < 0.1);
    }

    Ok(())
}

/// 测试隐藏状态传递
#[test]
fn test_lstm_hidden_propagation() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 2, 4, "lstm")?;

    // 短序列和长序列应该产生不同结果
    let x_short = graph.randn(&[1, 2, 2])?;
    let h_short = lstm.forward(&x_short)?;
    h_short.forward()?;
    let val_short = h_short.value()?.unwrap().clone();

    let x_long = graph.randn(&[1, 10, 2])?;
    let h_long = lstm.forward(&x_long)?;
    h_long.forward()?;
    let val_long = h_long.value()?.unwrap();

    // 应该是不同的值
    let diff: f32 = (0..4)
        .map(|i| (val_short[[0, i]] - val_long[[0, i]]).abs())
        .sum();
    assert!(diff > 0.01);

    Ok(())
}

// ==================== ModelState 集成测试 ====================

/// 测试与 ModelState 配合使用
#[test]
fn test_lstm_with_model_state() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 3, 8, "lstm")?;
    let fc = Linear::new(&graph, 8, 2, true, "fc")?;
    let state = ModelState::new(&graph);
    let criterion = CrossEntropyLoss::new();

    // 定义 forward 逻辑
    let forward = |x: &Tensor| -> Result<_, GraphError> {
        state.forward(x, |input| {
            let h = lstm.forward(input)?;
            Ok(fc.forward(&h))
        })
    };

    // 第一次调用
    let x1 = Tensor::new(&vec![0.1f32; 12], &[2, 2, 3]);
    let y1 = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let output1 = forward(&x1)?;
    let loss1 = criterion.forward(&output1, &y1)?;
    loss1.backward()?;

    // 第二次调用（复用）
    let x2 = Tensor::new(&vec![0.2f32; 12], &[2, 2, 3]);
    let output2 = forward(&x2)?;

    // 应该是同一个节点
    assert_eq!(output1.node_id(), output2.node_id());

    Ok(())
}

/// 测试智能缓存支持变长
#[test]
fn test_lstm_model_state_cache() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 2, 4, "lstm")?;
    let fc = Linear::new(&graph, 4, 2, true, "fc")?;
    let state = ModelState::new(&graph);

    let forward = |x: &Tensor| -> Result<_, GraphError> {
        state.forward(x, |input| {
            let h = lstm.forward(input)?;
            Ok(fc.forward(&h))
        })
    };

    // seq_len = 3
    let x1 = Tensor::new(&vec![0.1f32; 12], &[2, 3, 2]);
    let out1 = forward(&x1)?;

    // seq_len = 5（不同长度，应该创建新缓存）
    let x2 = Tensor::new(&vec![0.1f32; 20], &[2, 5, 2]);
    let out2 = forward(&x2)?;

    // 应该是不同的节点
    assert_ne!(out1.node_id(), out2.node_id());
    assert_eq!(state.cache_size(), 2);

    Ok(())
}

// ==================== 边界情况测试 ====================

/// 测试 seq_len = 1
#[test]
fn test_lstm_seq_len_one() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 5, 8, "lstm")?;

    let x = graph.randn(&[3, 1, 5])?;
    let h = lstm.forward(&x)?;
    h.forward()?;

    let output = h.value()?.unwrap();
    assert_eq!(output.shape(), &[3, 8]);

    Ok(())
}

/// 测试 batch_size = 1
#[test]
fn test_lstm_batch_size_one() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 3, 6, "lstm")?;

    let x = graph.randn(&[1, 10, 3])?;
    let h = lstm.forward(&x)?;
    h.forward()?;

    let output = h.value()?.unwrap();
    assert_eq!(output.shape(), &[1, 6]);

    Ok(())
}

/// 测试长序列
#[test]
fn test_lstm_long_sequence() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 2, 4, "lstm")?;

    let x = graph.randn(&[1, 50, 2])?;
    let h = lstm.forward(&x)?;
    h.forward()?;

    let output = h.value()?.unwrap();
    assert_eq!(output.shape(), &[1, 4]);

    Ok(())
}

/// 测试 LSTM 缓存：不同 batch_size 使用相同 seq_len 时，应该创建不同的计算图
///
/// 这是 Bug 修复测试：之前没有缓存机制，每次都重新创建计算图。
/// 现在添加缓存后，需要确保缓存 key 是 (batch_size, seq_len) 而不仅仅是 seq_len。
#[test]
fn test_lstm_different_batch_size_same_seq_len() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 2, 4, "lstm")?;

    // 第一个输入：batch_size=2, seq_len=3
    let x1 = graph.zeros(&[2, 3, 2])?;
    x1.set_value(&Tensor::new(&vec![0.1f32; 12], &[2, 3, 2]))?;

    // 第二个输入：batch_size=4, seq_len=3（相同 seq_len，不同 batch_size）
    let x2 = graph.zeros(&[4, 3, 2])?;
    x2.set_value(&Tensor::new(&vec![0.1f32; 24], &[4, 3, 2]))?;

    // 第三个输入：batch_size=2, seq_len=3（回到第一个配置）
    let x3 = graph.zeros(&[2, 3, 2])?;
    x3.set_value(&Tensor::new(&vec![0.2f32; 12], &[2, 3, 2]))?;

    // 三次 forward
    let h1 = lstm.forward(&x1)?;
    let id1 = h1.node_id();
    let shape1 = h1.value()?.unwrap().shape().to_vec();

    let h2 = lstm.forward(&x2)?;
    let id2 = h2.node_id();
    let shape2 = h2.value()?.unwrap().shape().to_vec();

    let h3 = lstm.forward(&x3)?;
    let id3 = h3.node_id();
    let shape3 = h3.value()?.unwrap().shape().to_vec();

    // 验证：不同 batch_size 应该返回不同的 node_id
    assert_ne!(
        id1, id2,
        "batch_size=2 和 batch_size=4 应该返回不同的 node_id（即使 seq_len 相同）"
    );

    // 验证：相同 (batch_size, seq_len) 应该返回相同的 node_id（复用缓存）
    assert_eq!(
        id1, id3,
        "相同的 (batch_size=2, seq_len=3) 应该返回相同的 node_id"
    );

    // 验证输出形状正确
    assert_eq!(shape1, vec![2, 4], "batch_size=2 的输出形状应该是 [2, 4]");
    assert_eq!(shape2, vec![4, 4], "batch_size=4 的输出形状应该是 [4, 4]");
    assert_eq!(
        shape3,
        vec![2, 4],
        "再次 batch_size=2 的输出形状应该是 [2, 4]"
    );

    Ok(())
}

/// 测试种子可重复性
#[test]
fn test_lstm_seeded_reproducibility() -> Result<(), GraphError> {
    // 相同种子应该产生相同权重初始化，因此相同输入应产生相同输出
    let run = |seed: u64| -> Result<Vec<f32>, GraphError> {
        let graph = Graph::new_with_seed(seed);
        let lstm = Lstm::new(&graph, 2, 3, "lstm")?;

        // 使用固定输入
        let x_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let x = graph.input(&Tensor::new(&x_data, &[1, 4, 2]))?;
        let h = lstm.forward(&x)?;
        h.forward()?;

        let output = h.value()?.unwrap();
        Ok((0..3).map(|i| output[[0, i]]).collect())
    };

    let result1 = run(42)?;
    let result2 = run(42)?;

    for i in 0..3 {
        assert!((result1[i] - result2[i]).abs() < 1e-6);
    }

    Ok(())
}
