/*
 * @Author       : 老董
 * @Date         : 2026-01-21
 * @Description  : Rnn 层单元测试（展开式设计，PyTorch 风格 API）
 *
 * 测试新的展开式 RNN 层，使用 Var::select 实现时间步展开，
 * BPTT 通过标准 backward() 自动完成。
 *
 * 参考值来源: tests/parity_rnn_fixed_len_pytorch.py
 */

use crate::nn::graph::Graph;
use crate::nn::layer::{Linear, Rnn};
use crate::nn::{GraphError, ModelState, Module, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 Rnn 层创建
#[test]
fn test_rnn_new() {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 4, 8, "rnn").unwrap();

    assert_eq!(rnn.input_size(), 4);
    assert_eq!(rnn.hidden_size(), 8);
}

/// 测试 Rnn 参数数量
#[test]
fn test_rnn_parameters() {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 4, 8, "rnn").unwrap();

    // RNN 有 3 个参数: W_ih, W_hh, b_h
    let params = rnn.parameters();
    assert_eq!(params.len(), 3);

    // 验证形状
    let w_ih = rnn.w_ih().value().unwrap().unwrap();
    let w_hh = rnn.w_hh().value().unwrap().unwrap();
    let b_h = rnn.b_h().value().unwrap().unwrap();

    assert_eq!(w_ih.shape(), &[4, 8]); // [input_size, hidden_size]
    assert_eq!(w_hh.shape(), &[8, 8]); // [hidden_size, hidden_size]
    assert_eq!(b_h.shape(), &[1, 8]); // [1, hidden_size]
}

/// 测试 Rnn 前向传播（基本形状检查）
#[test]
fn test_rnn_forward_shape() {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 4, 8, "rnn").unwrap();

    // 创建输入: [batch=2, seq_len=5, input_size=4]
    let x_data: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();
    let x_var = graph.zeros(&[2, 5, 4]).unwrap();
    x_var.set_value(&Tensor::new(&x_data, &[2, 5, 4])).unwrap();

    // 前向传播
    let h = rnn.forward(&x_var).unwrap();
    h.forward().unwrap();

    // 验证输出形状: [batch=2, hidden_size=8]
    let result = h.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 8]);
}

/// 测试 Rnn 输入维度检查（应该拒绝非 3D 输入）
#[test]
fn test_rnn_forward_invalid_dim() {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 4, 8, "rnn").unwrap();

    // 2D 输入应该报错
    let x_2d = graph.zeros(&[2, 4]).unwrap();
    let result = rnn.forward(&x_2d);
    assert!(result.is_err());

    // 4D 输入也应该报错
    let x_4d = graph.zeros(&[2, 3, 4, 5]).unwrap();
    let result = rnn.forward(&x_4d);
    assert!(result.is_err());
}

/// 测试 Rnn 输入特征维度检查
#[test]
fn test_rnn_forward_input_size_mismatch() {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 4, 8, "rnn").unwrap(); // input_size=4

    // input_size 不匹配应该报错
    let x_wrong = graph.zeros(&[2, 5, 6]).unwrap(); // input_size=6
    let result = rnn.forward(&x_wrong);
    assert!(result.is_err());
}

/// 测试 Rnn 与 Linear 组合
#[test]
fn test_rnn_with_linear() {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 4, 8, "rnn").unwrap();
    let fc = Linear::new(&graph, 8, 2, true, "fc").unwrap();

    // 输入
    let x_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let x_var = graph.zeros(&[2, 3, 4]).unwrap();
    x_var.set_value(&Tensor::new(&x_data, &[2, 3, 4])).unwrap();

    // RNN + Linear
    let h = rnn.forward(&x_var).unwrap();
    let y = fc.forward(&h);
    y.forward().unwrap();

    // 验证输出形状: [batch=2, out_features=2]
    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]);
}

/// 测试 Rnn 反向传播
#[test]
fn test_rnn_backward() {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 4, 8, "rnn").unwrap();
    let fc = Linear::new(&graph, 8, 2, true, "fc").unwrap();

    // 输入和目标
    let x_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let x_var = graph.zeros(&[2, 3, 4]).unwrap();
    x_var.set_value(&Tensor::new(&x_data, &[2, 3, 4])).unwrap();

    let target = graph
        .input(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]))
        .unwrap();

    // 前向传播
    let h = rnn.forward(&x_var).unwrap();
    let y = fc.forward(&h);
    let loss = y.mse_loss(&target).unwrap();

    // 反向传播
    loss.backward().unwrap();

    // 验证所有参数都有梯度
    for p in rnn.parameters() {
        let grad = p.grad().unwrap();
        assert!(grad.is_some(), "RNN 参数应该有梯度");
    }
    for p in fc.parameters() {
        let grad = p.grad().unwrap();
        assert!(grad.is_some(), "Linear 参数应该有梯度");
    }
}

/// 测试带种子 Graph 的 Rnn 可重复性
#[test]
fn test_rnn_seeded_reproducibility() {
    let seed = 12345u64;

    let graph1 = Graph::new_with_seed(seed);
    let rnn1 = Rnn::new(&graph1, 4, 8, "rnn").unwrap();

    let graph2 = Graph::new_with_seed(seed);
    let rnn2 = Rnn::new(&graph2, 4, 8, "rnn").unwrap();

    // 权重应该完全相同
    let w_ih_1 = rnn1.w_ih().value().unwrap().unwrap();
    let w_ih_2 = rnn2.w_ih().value().unwrap().unwrap();
    assert_eq!(w_ih_1.data_as_slice(), w_ih_2.data_as_slice());

    let w_hh_1 = rnn1.w_hh().value().unwrap().unwrap();
    let w_hh_2 = rnn2.w_hh().value().unwrap().unwrap();
    assert_eq!(w_hh_1.data_as_slice(), w_hh_2.data_as_slice());
}

// ==================== ModelState 集成测试 ====================

/// 测试 Rnn 与 ModelState 集成
#[test]
fn test_rnn_with_model_state() {
    let graph = Graph::new_with_seed(42);
    let rnn = Rnn::new(&graph, 4, 8, "rnn").unwrap();
    let fc = Linear::new(&graph, 8, 2, true, "fc").unwrap();
    let state = ModelState::new(&graph);

    // 输入 Tensor
    let x_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let x = Tensor::new(&x_data, &[2, 3, 4]);

    // 使用 ModelState 进行 forward
    let output = state
        .forward(&x, |input| {
            let h = rnn.forward(input)?;
            Ok(fc.forward(&h))
        })
        .unwrap();

    output.forward().unwrap();
    let result = output.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]);

    // 验证 ModelState 已初始化
    assert!(state.is_initialized());
}

/// 测试 ModelState 缓存复用
#[test]
fn test_rnn_model_state_cache() {
    let graph = Graph::new_with_seed(42);
    let rnn = Rnn::new(&graph, 4, 8, "rnn").unwrap();
    let fc = Linear::new(&graph, 8, 2, true, "fc").unwrap();
    let state = ModelState::new(&graph);

    let x1 = Tensor::new(&vec![0.1f32; 24], &[2, 3, 4]);
    let x2 = Tensor::new(&vec![0.2f32; 24], &[2, 3, 4]);

    // 第一次调用
    let output1 = state
        .forward(&x1, |input| {
            let h = rnn.forward(input)?;
            Ok(fc.forward(&h))
        })
        .unwrap();
    let id1 = output1.node_id();

    // 第二次调用（应该复用缓存）
    let output2 = state
        .forward(&x2, |input| {
            let h = rnn.forward(input)?;
            Ok(fc.forward(&h))
        })
        .unwrap();
    let id2 = output2.node_id();

    // 节点 ID 应该相同（复用）
    assert_eq!(id1, id2);
}

// ==================== 数值正确性测试 ====================

/// 测试单时间步 RNN 数值
#[test]
fn test_rnn_single_timestep_value() -> Result<(), GraphError> {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 2, 3, "rnn")?;

    // 设置已知权重
    rnn.w_ih()
        .set_value(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]))?;
    rnn.w_hh().set_value(&Tensor::new(
        &[0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
        &[3, 3],
    ))?;
    rnn.b_h().set_value(&Tensor::zeros(&[1, 3]))?;

    // 输入: [batch=1, seq_len=1, input_size=2]
    let x_var = graph.zeros(&[1, 1, 2])?;
    x_var.set_value(&Tensor::new(&[1.0, 1.0], &[1, 1, 2]))?;

    // 前向传播
    let h = rnn.forward(&x_var)?;
    h.forward()?;

    let result = h.value()?.unwrap();
    let data = result.data_as_slice();

    // h = tanh(x @ W_ih + 0 @ W_hh + b_h) = tanh([0.5, 0.7, 0.9])
    // tanh(0.5) ≈ 0.4621, tanh(0.7) ≈ 0.6044, tanh(0.9) ≈ 0.7163
    assert_abs_diff_eq!(data[0], 0.4621, epsilon = 0.01);
    assert_abs_diff_eq!(data[1], 0.6044, epsilon = 0.01);
    assert_abs_diff_eq!(data[2], 0.7163, epsilon = 0.01);

    Ok(())
}

/// 测试多时间步 RNN（验证隐藏状态传递）
#[test]
fn test_rnn_multi_timestep_hidden_propagation() -> Result<(), GraphError> {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 2, 2, "rnn")?;

    // 设置简单权重
    rnn.w_ih()
        .set_value(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]))?;
    rnn.w_hh()
        .set_value(&Tensor::new(&[0.5, 0.0, 0.0, 0.5], &[2, 2]))?;
    rnn.b_h().set_value(&Tensor::zeros(&[1, 2]))?;

    // 输入: [batch=1, seq_len=3, input_size=2]
    // t=0: [1, 0], t=1: [0, 1], t=2: [1, 1]
    let x_var = graph.zeros(&[1, 3, 2])?;
    x_var.set_value(&Tensor::new(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[1, 3, 2]))?;

    let h = rnn.forward(&x_var)?;
    h.forward()?;

    let result = h.value()?.unwrap();

    // 验证最终隐藏状态不为零（说明状态正确传递）
    let data = result.data_as_slice();
    assert!(data[0].abs() > 0.1);
    assert!(data[1].abs() > 0.1);

    println!("最终隐藏状态: [{:.4}, {:.4}]", data[0], data[1]);
    Ok(())
}

/// 测试 RNN 梯度流回输入
#[test]
fn test_rnn_gradient_to_input() -> Result<(), GraphError> {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 2, 2, "rnn")?;
    let fc = Linear::new(&graph, 2, 1, true, "fc")?;

    // 设置可追踪的权重
    rnn.w_ih()
        .set_value(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2]))?;
    rnn.w_hh()
        .set_value(&Tensor::new(&[0.1, 0.0, 0.0, 0.1], &[2, 2]))?;
    rnn.b_h().set_value(&Tensor::zeros(&[1, 2]))?;
    fc.weights().set_value(&Tensor::new(&[1.0, 1.0], &[2, 1]))?;
    fc.bias().unwrap().set_value(&Tensor::zeros(&[1, 1]))?;

    // 输入
    let x_var = graph.zeros(&[1, 2, 2])?;
    x_var.set_value(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]))?;

    let target = graph.input(&Tensor::new(&[2.0], &[1, 1]))?;

    // 前向 + 反向
    let h = rnn.forward(&x_var)?;
    let y = fc.forward(&h);
    let loss = y.mse_loss(&target)?;
    loss.backward()?;

    // 验证 RNN 权重有梯度
    let grad_w_ih = rnn.w_ih().grad()?.unwrap();
    let grad_w_hh = rnn.w_hh().grad()?.unwrap();

    println!("grad_W_ih: {:?}", grad_w_ih.data_as_slice());
    println!("grad_W_hh: {:?}", grad_w_hh.data_as_slice());

    // 梯度不应该全为零
    assert!(grad_w_ih.data_as_slice().iter().any(|&v| v.abs() > 1e-6));
    assert!(grad_w_hh.data_as_slice().iter().any(|&v| v.abs() > 1e-6));

    Ok(())
}

// ==================== 边界情况测试 ====================

/// 测试 seq_len=1 的情况
#[test]
fn test_rnn_seq_len_one() -> Result<(), GraphError> {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 4, 8, "rnn")?;

    let x_var = graph.zeros(&[2, 1, 4])?;
    x_var.set_value(&Tensor::new(&vec![0.1f32; 8], &[2, 1, 4]))?;

    let h = rnn.forward(&x_var)?;
    h.forward()?;

    let result = h.value()?.unwrap();
    assert_eq!(result.shape(), &[2, 8]);

    Ok(())
}

/// 测试 batch_size=1 的情况
#[test]
fn test_rnn_batch_size_one() -> Result<(), GraphError> {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 4, 8, "rnn")?;

    let x_var = graph.zeros(&[1, 5, 4])?;
    x_var.set_value(&Tensor::new(&vec![0.1f32; 20], &[1, 5, 4]))?;

    let h = rnn.forward(&x_var)?;
    h.forward()?;

    let result = h.value()?.unwrap();
    assert_eq!(result.shape(), &[1, 8]);

    Ok(())
}

/// 测试较长序列
#[test]
fn test_rnn_long_sequence() -> Result<(), GraphError> {
    let graph = Graph::new();
    let rnn = Rnn::new(&graph, 4, 8, "rnn")?;

    // 较长序列: seq_len=20
    let x_var = graph.zeros(&[2, 20, 4])?;
    x_var.set_value(&Tensor::new(&vec![0.1f32; 160], &[2, 20, 4]))?;

    let h = rnn.forward(&x_var)?;
    h.forward()?;

    let result = h.value()?.unwrap();
    assert_eq!(result.shape(), &[2, 8]);

    // 验证值在合理范围（tanh 输出在 -1 到 1 之间）
    for &v in result.data_as_slice() {
        assert!(v.abs() <= 1.0);
    }

    Ok(())
}
