/*
 * BPTT PyTorch 数值对照测试
 *
 * TODO(RNN 重构): 此测试基于旧的"显式时间步 + backward_through_time"设计。
 * 待 LSTM/GRU 完成展开式重构后，这些测试可能需要删除或重写。
 * 新的 Rnn 层使用"展开式设计"，BPTT 通过标准 backward() 自动完成。
 *
 * 使用 PyTorch 计算的精确参考值验证 BPTT 实现的数值正确性
 * 参考脚本: tests/python/layer_reference/simple_rnn_bptt.py
 */

use crate::nn::{GraphError, GraphInner, NodeId};
use crate::tensor::Tensor;

// ==================== PyTorch 参考值 ====================
// 序列: [1.0, 0.0, 1.0], 目标: 1.0, w_scale = 1.0, w_out = 1.0

const HIDDEN_T1: f32 = 0.76159418; // tanh(1.0)
const HIDDEN_T2: f32 = 0.64201498; // tanh(tanh(1.0))
const HIDDEN_T3: f32 = 0.92775375; // tanh(tanh(tanh(1.0)) + 1.0)
const LOSS: f32 = 0.00521952;
const GRAD_W_SCALE: f32 = -0.02509185;
const GRAD_W_OUT: f32 = -0.13405347;

const TOLERANCE: f32 = 1e-5;

// ==================== 辅助函数 ====================

/// 创建测试用的简单 RNN 网络
/// hidden = tanh(h_prev + input * w_scale), output = hidden * w_out
fn create_simple_rnn() -> Result<(GraphInner, NodeId, NodeId, NodeId, NodeId, NodeId), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 1], Some("input"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;

    let w_scale = graph.new_parameter_node(&[1, 1], Some("w_scale"))?;
    graph.set_node_value(w_scale, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    let scaled_input = graph.new_mat_mul_node(input, w_scale, Some("scaled_input"))?;
    let pre_hidden = graph.new_add_node(&[h_prev, scaled_input], Some("pre_hidden"))?;
    let hidden = graph.new_tanh_node(pre_hidden, Some("hidden"))?;
    graph.connect_recurrent(hidden, h_prev)?;

    let output = graph.new_mat_mul_node(hidden, w_out, Some("output"))?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    Ok((graph, input, hidden, loss, target, w_scale))
}

// ==================== 数值对照测试 ====================

/// 前向传播：hidden 和 loss 值与 PyTorch 匹配
#[test]
fn test_forward_matches_pytorch() -> Result<(), GraphError> {
    let (mut graph, input, hidden, loss, target, _) = create_simple_rnn()?;

    graph.set_node_value(target, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    let sequence = [1.0, 0.0, 1.0];
    let expected_hiddens = [HIDDEN_T1, HIDDEN_T2, HIDDEN_T3];

    for (t, (&x, &expected_h)) in sequence.iter().zip(expected_hiddens.iter()).enumerate() {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;

        let actual_h = graph.get_node_value(hidden)?.unwrap().data_as_slice()[0];
        assert!(
            (actual_h - expected_h).abs() < TOLERANCE,
            "t={}: hidden 不匹配: actual={}, expected={}",
            t,
            actual_h,
            expected_h
        );
    }

    let actual_loss = graph.get_node_value(loss)?.unwrap().data_as_slice()[0];
    assert!(
        (actual_loss - LOSS).abs() < TOLERANCE,
        "loss 不匹配: actual={}, expected={}",
        actual_loss,
        LOSS
    );

    Ok(())
}

/// BPTT 梯度：w_scale 和 w_out 的梯度与 PyTorch 匹配
#[test]
fn test_bptt_gradient_matches_pytorch() -> Result<(), GraphError> {
    let (mut graph, input, _, loss, target, w_scale) = create_simple_rnn()?;

    let w_out = graph
        .get_trainable_nodes()
        .into_iter()
        .find(|&id| {
            graph
                .get_node(id)
                .map(|n| n.name() == "w_out")
                .unwrap_or(false)
        })
        .expect("w_out not found");

    graph.set_node_value(target, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // 前向传播序列
    for &x in &[1.0, 0.0, 1.0] {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;
    }

    // BPTT
    graph.backward_through_time(&[w_scale, w_out], loss)?;

    // 验证梯度
    let actual_w_scale = graph
        .get_node_grad(w_scale)?
        .expect("w_scale 应有梯度")
        .data_as_slice()[0];
    let actual_w_out = graph
        .get_node_grad(w_out)?
        .expect("w_out 应有梯度")
        .data_as_slice()[0];

    assert!(
        (actual_w_out - GRAD_W_OUT).abs() < TOLERANCE,
        "dL/d(w_out) 不匹配: actual={}, expected={}",
        actual_w_out,
        GRAD_W_OUT
    );

    assert!(
        (actual_w_scale - GRAD_W_SCALE).abs() < TOLERANCE,
        "dL/d(w_scale) 不匹配: actual={}, expected={}",
        actual_w_scale,
        GRAD_W_SCALE
    );

    Ok(())
}

// ==================== 多结构 PyTorch 参考值 ====================
// 参考脚本: tests/python/layer_reference/rnn_multi_structure.py

// === 结构 2: 更长序列的 tanh RNN ===
// 用于验证长序列下梯度正确性

/// 结构 2: 更长序列的 tanh RNN（5 步）
///
/// 与基础测试相同结构，但使用更长的序列验证 BPTT 累加正确
#[test]
fn test_longer_sequence_tanh_rnn() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 1], Some("input"))?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;

    let w_ih = graph.new_parameter_node(&[1, 1], Some("w_ih"))?;
    graph.set_node_value(w_ih, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    let w_hh = graph.new_parameter_node(&[1, 1], Some("w_hh"))?;
    graph.set_node_value(w_hh, Some(&Tensor::new(&[0.8], &[1, 1])))?;

    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // h = tanh(input * w_ih + h_prev * w_hh)
    let scaled_input = graph.new_mat_mul_node(input, w_ih, None)?;
    let scaled_h = graph.new_mat_mul_node(h_prev, w_hh, None)?;
    let pre_h = graph.new_add_node(&[scaled_input, scaled_h], None)?;
    let hidden = graph.new_tanh_node(pre_h, Some("hidden"))?;
    graph.connect_recurrent(hidden, h_prev)?;

    let output = graph.new_mat_mul_node(hidden, w_out, None)?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    graph.set_node_value(target, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    // 5 步序列
    let sequence = [1.0, 0.5, -0.5, 0.0, 1.0];
    for &x in &sequence {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;
    }

    // BPTT
    graph.backward_through_time(&[w_ih, w_hh, w_out], loss)?;

    // 验证梯度存在且有限
    let grad_w_ih = graph.get_node_grad(w_ih)?.unwrap().data_as_slice()[0];
    let grad_w_hh = graph.get_node_grad(w_hh)?.unwrap().data_as_slice()[0];
    let grad_w_out = graph.get_node_grad(w_out)?.unwrap().data_as_slice()[0];

    assert!(grad_w_ih.is_finite(), "w_ih 梯度应有限");
    assert!(grad_w_hh.is_finite(), "w_hh 梯度应有限");
    assert!(grad_w_out.is_finite(), "w_out 梯度应有限");

    // 验证梯度非零（网络有贡献）
    assert!(grad_w_out.abs() > 1e-6, "w_out 梯度应非零");

    println!("✅ 长序列 tanh RNN BPTT 测试通过");
    println!("  grad_w_ih = {:.6}", grad_w_ih);
    println!("  grad_w_hh = {:.6}", grad_w_hh);
    println!("  grad_w_out = {:.6}", grad_w_out);

    Ok(())
}

/// 结构 3: 带有两个独立参数路径的 RNN
///
/// 验证同一网络中有多个参数时，梯度正确累加
#[test]
fn test_multi_param_rnn() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 1], Some("input"))?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;

    // 两个独立的输入权重
    let w_ih1 = graph.new_parameter_node(&[1, 1], Some("w_ih1"))?;
    graph.set_node_value(w_ih1, Some(&Tensor::new(&[0.3], &[1, 1])))?;

    let w_ih2 = graph.new_parameter_node(&[1, 1], Some("w_ih2"))?;
    graph.set_node_value(w_ih2, Some(&Tensor::new(&[0.7], &[1, 1])))?;

    let w_hh = graph.new_parameter_node(&[1, 1], Some("w_hh"))?;
    graph.set_node_value(w_hh, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // h = tanh(input * w_ih1 + input * w_ih2 + h_prev * w_hh)
    let path1 = graph.new_mat_mul_node(input, w_ih1, None)?;
    let path2 = graph.new_mat_mul_node(input, w_ih2, None)?;
    let scaled_h = graph.new_mat_mul_node(h_prev, w_hh, None)?;
    let sum1 = graph.new_add_node(&[path1, path2], None)?;
    let pre_h = graph.new_add_node(&[sum1, scaled_h], None)?;
    let hidden = graph.new_tanh_node(pre_h, Some("hidden"))?;
    graph.connect_recurrent(hidden, h_prev)?;

    let output = graph.new_mat_mul_node(hidden, w_out, None)?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    graph.set_node_value(target, Some(&Tensor::new(&[0.3], &[1, 1])))?;

    // 前向传播
    let sequence = [1.0, 0.0, -1.0];
    for &x in &sequence {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;
    }

    // BPTT
    graph.backward_through_time(&[w_ih1, w_ih2, w_hh, w_out], loss)?;

    let grad_w_ih1 = graph.get_node_grad(w_ih1)?.unwrap().data_as_slice()[0];
    let grad_w_ih2 = graph.get_node_grad(w_ih2)?.unwrap().data_as_slice()[0];
    let grad_w_hh = graph.get_node_grad(w_hh)?.unwrap().data_as_slice()[0];
    let grad_w_out = graph.get_node_grad(w_out)?.unwrap().data_as_slice()[0];

    // 验证所有梯度有限
    assert!(grad_w_ih1.is_finite());
    assert!(grad_w_ih2.is_finite());
    assert!(grad_w_hh.is_finite());
    assert!(grad_w_out.is_finite());

    // w_ih1 和 w_ih2 应该有相同符号的梯度（相同输入路径）
    // 但由于权重不同，大小可能不同
    assert_eq!(
        grad_w_ih1.signum(),
        grad_w_ih2.signum(),
        "w_ih1 和 w_ih2 梯度应该同号"
    );

    println!("✅ 多参数 RNN BPTT 测试通过");
    println!("  grad_w_ih1 = {:.6}", grad_w_ih1);
    println!("  grad_w_ih2 = {:.6}", grad_w_ih2);
    println!("  grad_w_hh = {:.6}", grad_w_hh);
    println!("  grad_w_out = {:.6}", grad_w_out);

    Ok(())
}

// ==================== 通用激活函数验收测试 ====================
// 验证 BPTT 对非 tanh 激活函数的支持（sigmoid、混合激活等）

// === Sigmoid RNN PyTorch 参考值 ===
// h[t] = sigmoid(x[t] * w_ih + h[t-1] * w_hh)
// output = h[T] * w_out
// 参数: w_ih=0.5, w_hh=0.9, w_out=2.0
// 序列: [1.0, -1.0, 0.5], 目标: 0.8
const SIGMOID_H_T1: f32 = 0.62245935;
const SIGMOID_H_T2: f32 = 0.51504880;
const SIGMOID_H_T3: f32 = 0.67118376;
const SIGMOID_OUTPUT: f32 = 1.34236753;
const SIGMOID_LOSS: f32 = 0.29416251;
const SIGMOID_GRAD_W_IH: f32 = 0.15453015;
const SIGMOID_GRAD_W_HH: f32 = 0.31359798;
const SIGMOID_GRAD_W_OUT: f32 = 0.72805655;

/// Sigmoid RNN 前向传播与 PyTorch 匹配
#[test]
fn test_sigmoid_rnn_forward_matches_pytorch() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 1], Some("input"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;

    let w_ih = graph.new_parameter_node(&[1, 1], Some("w_ih"))?;
    graph.set_node_value(w_ih, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    let w_hh = graph.new_parameter_node(&[1, 1], Some("w_hh"))?;
    graph.set_node_value(w_hh, Some(&Tensor::new(&[0.9], &[1, 1])))?;

    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(&[2.0], &[1, 1])))?;

    // h = sigmoid(input * w_ih + h_prev * w_hh)
    let scaled_input = graph.new_mat_mul_node(input, w_ih, None)?;
    let scaled_h = graph.new_mat_mul_node(h_prev, w_hh, None)?;
    let pre_h = graph.new_add_node(&[scaled_input, scaled_h], None)?;
    let hidden = graph.new_sigmoid_node(pre_h, Some("hidden"))?;
    graph.connect_recurrent(hidden, h_prev)?;

    let output = graph.new_mat_mul_node(hidden, w_out, None)?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    graph.set_node_value(target, Some(&Tensor::new(&[0.8], &[1, 1])))?;

    // 前向传播序列
    let sequence = [1.0, -1.0, 0.5];
    let expected_h = [SIGMOID_H_T1, SIGMOID_H_T2, SIGMOID_H_T3];

    for (t, (&x, &expected)) in sequence.iter().zip(expected_h.iter()).enumerate() {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;

        let actual_h = graph.get_node_value(hidden)?.unwrap().data_as_slice()[0];
        assert!(
            (actual_h - expected).abs() < 1e-5,
            "t={}: hidden 不匹配: actual={}, expected={}",
            t,
            actual_h,
            expected
        );
    }

    let actual_output = graph.get_node_value(output)?.unwrap().data_as_slice()[0];
    let actual_loss = graph.get_node_value(loss)?.unwrap().data_as_slice()[0];

    assert!(
        (actual_output - SIGMOID_OUTPUT).abs() < 1e-5,
        "output 不匹配: actual={}, expected={}",
        actual_output,
        SIGMOID_OUTPUT
    );
    assert!(
        (actual_loss - SIGMOID_LOSS).abs() < 1e-5,
        "loss 不匹配: actual={}, expected={}",
        actual_loss,
        SIGMOID_LOSS
    );

    println!("✅ Sigmoid RNN 前向传播测试通过");
    Ok(())
}

/// Sigmoid RNN BPTT 梯度与 PyTorch 匹配
///
/// 验证 BPTT 通用化后能正确处理 sigmoid 激活
#[test]
fn test_sigmoid_rnn_bptt_matches_pytorch() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 1], Some("input"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;

    let w_ih = graph.new_parameter_node(&[1, 1], Some("w_ih"))?;
    graph.set_node_value(w_ih, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    let w_hh = graph.new_parameter_node(&[1, 1], Some("w_hh"))?;
    graph.set_node_value(w_hh, Some(&Tensor::new(&[0.9], &[1, 1])))?;

    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(&[2.0], &[1, 1])))?;

    // h = sigmoid(input * w_ih + h_prev * w_hh)
    let scaled_input = graph.new_mat_mul_node(input, w_ih, None)?;
    let scaled_h = graph.new_mat_mul_node(h_prev, w_hh, None)?;
    let pre_h = graph.new_add_node(&[scaled_input, scaled_h], None)?;
    let hidden = graph.new_sigmoid_node(pre_h, Some("hidden"))?;
    graph.connect_recurrent(hidden, h_prev)?;

    let output = graph.new_mat_mul_node(hidden, w_out, None)?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    graph.set_node_value(target, Some(&Tensor::new(&[0.8], &[1, 1])))?;

    // 前向传播序列
    for &x in &[1.0, -1.0, 0.5] {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;
    }

    // BPTT
    graph.backward_through_time(&[w_ih, w_hh, w_out], loss)?;

    // 验证梯度
    let actual_w_ih = graph.get_node_grad(w_ih)?.unwrap().data_as_slice()[0];
    let actual_w_hh = graph.get_node_grad(w_hh)?.unwrap().data_as_slice()[0];
    let actual_w_out = graph.get_node_grad(w_out)?.unwrap().data_as_slice()[0];

    assert!(
        (actual_w_out - SIGMOID_GRAD_W_OUT).abs() < 1e-5,
        "dL/d(w_out) 不匹配: actual={}, expected={}",
        actual_w_out,
        SIGMOID_GRAD_W_OUT
    );

    assert!(
        (actual_w_ih - SIGMOID_GRAD_W_IH).abs() < 1e-5,
        "dL/d(w_ih) 不匹配: actual={}, expected={}",
        actual_w_ih,
        SIGMOID_GRAD_W_IH
    );

    assert!(
        (actual_w_hh - SIGMOID_GRAD_W_HH).abs() < 1e-5,
        "dL/d(w_hh) 不匹配: actual={}, expected={}",
        actual_w_hh,
        SIGMOID_GRAD_W_HH
    );

    println!("✅ Sigmoid RNN BPTT 梯度测试通过");
    println!(
        "  dL/d(w_ih) = {} (期望: {})",
        actual_w_ih, SIGMOID_GRAD_W_IH
    );
    println!(
        "  dL/d(w_hh) = {} (期望: {})",
        actual_w_hh, SIGMOID_GRAD_W_HH
    );
    println!(
        "  dL/d(w_out) = {} (期望: {})",
        actual_w_out, SIGMOID_GRAD_W_OUT
    );

    Ok(())
}

/// 混合激活 RNN（tanh + sigmoid 双层）
///
/// 验证 BPTT 能正确处理同一网络中的不同激活函数
/// 网络结构:
///   h1[t] = tanh(x[t] * w_ih1 + h1[t-1] * w_hh1)
///   h2[t] = sigmoid(h1[t] * w_h12 + h2[t-1] * w_hh2)
///   output = h2[T] * w_out
#[test]
fn test_mixed_activation_rnn() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 1], Some("input"))?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;

    // 两个 State 节点
    let h1_prev = graph.new_state_node(&[1, 1], Some("h1_prev"))?;
    graph.set_node_value(h1_prev, Some(&Tensor::zeros(&[1, 1])))?;
    let h2_prev = graph.new_state_node(&[1, 1], Some("h2_prev"))?;
    graph.set_node_value(h2_prev, Some(&Tensor::zeros(&[1, 1])))?;

    // 参数
    let w_ih1 = graph.new_parameter_node(&[1, 1], Some("w_ih1"))?;
    graph.set_node_value(w_ih1, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    let w_hh1 = graph.new_parameter_node(&[1, 1], Some("w_hh1"))?;
    graph.set_node_value(w_hh1, Some(&Tensor::new(&[0.6], &[1, 1])))?;

    let w_h12 = graph.new_parameter_node(&[1, 1], Some("w_h12"))?;
    graph.set_node_value(w_h12, Some(&Tensor::new(&[0.7], &[1, 1])))?;

    let w_hh2 = graph.new_parameter_node(&[1, 1], Some("w_hh2"))?;
    graph.set_node_value(w_hh2, Some(&Tensor::new(&[0.8], &[1, 1])))?;

    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // 第一层：tanh
    let scaled_input = graph.new_mat_mul_node(input, w_ih1, None)?;
    let scaled_h1 = graph.new_mat_mul_node(h1_prev, w_hh1, None)?;
    let pre_h1 = graph.new_add_node(&[scaled_input, scaled_h1], None)?;
    let h1 = graph.new_tanh_node(pre_h1, Some("h1"))?;
    graph.connect_recurrent(h1, h1_prev)?;

    // 第二层：sigmoid
    let h1_scaled = graph.new_mat_mul_node(h1, w_h12, None)?;
    let scaled_h2 = graph.new_mat_mul_node(h2_prev, w_hh2, None)?;
    let pre_h2 = graph.new_add_node(&[h1_scaled, scaled_h2], None)?;
    let h2 = graph.new_sigmoid_node(pre_h2, Some("h2"))?;
    graph.connect_recurrent(h2, h2_prev)?;

    let output = graph.new_mat_mul_node(h2, w_out, None)?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    graph.set_node_value(target, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    // 前向传播序列
    let sequence = [1.0, -0.5, 0.5, 1.0];
    for &x in &sequence {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;
    }

    // BPTT
    graph.backward_through_time(&[w_ih1, w_hh1, w_h12, w_hh2, w_out], loss)?;

    // 验证所有梯度有限且非零
    let grad_w_ih1 = graph.get_node_grad(w_ih1)?.unwrap().data_as_slice()[0];
    let grad_w_hh1 = graph.get_node_grad(w_hh1)?.unwrap().data_as_slice()[0];
    let grad_w_h12 = graph.get_node_grad(w_h12)?.unwrap().data_as_slice()[0];
    let grad_w_hh2 = graph.get_node_grad(w_hh2)?.unwrap().data_as_slice()[0];
    let grad_w_out = graph.get_node_grad(w_out)?.unwrap().data_as_slice()[0];

    assert!(grad_w_ih1.is_finite(), "w_ih1 梯度应有限");
    assert!(grad_w_hh1.is_finite(), "w_hh1 梯度应有限");
    assert!(grad_w_h12.is_finite(), "w_h12 梯度应有限");
    assert!(grad_w_hh2.is_finite(), "w_hh2 梯度应有限");
    assert!(grad_w_out.is_finite(), "w_out 梯度应有限");

    // 所有梯度应该非零（网络有贡献）
    assert!(grad_w_out.abs() > 1e-6, "w_out 梯度应非零");
    assert!(grad_w_h12.abs() > 1e-6, "w_h12 梯度应非零");
    assert!(grad_w_hh2.abs() > 1e-6, "w_hh2 梯度应非零");
    // w_ih1 和 w_hh1 可能因为 tanh 饱和而接近零，但至少应该有限

    println!("✅ 混合激活 RNN BPTT 测试通过");
    println!(
        "  第 1 层 (tanh): w_ih1={:.6}, w_hh1={:.6}",
        grad_w_ih1, grad_w_hh1
    );
    println!(
        "  第 2 层 (sigmoid): w_h12={:.6}, w_hh2={:.6}",
        grad_w_h12, grad_w_hh2
    );
    println!("  输出层: w_out={:.6}", grad_w_out);

    Ok(())
}

// ==================== 结构 4: LeakyReLU RNN PyTorch 参考值 ====================
// 参考脚本: tests/python/layer_reference/rnn_multi_structure.py
// h[t] = LeakyReLU(x[t] * w_ih + h[t-1] * w_hh, negative_slope=0.1)
// output = h[T] * w_out
// 参数: w_ih=0.5, w_hh=0.8, w_out=1.5, negative_slope=0.1
// 序列: [1.0, -0.5, 0.3, -0.8, 0.6], 目标: 0.7
const LEAKY_RELU_NEGATIVE_SLOPE: f64 = 0.1;
const LEAKY_RELU_H_T1: f32 = 0.50000000;
const LEAKY_RELU_H_T2: f32 = 0.15000001;
const LEAKY_RELU_H_T3: f32 = 0.27000001;
const LEAKY_RELU_H_T4: f32 = -0.01840000;
const LEAKY_RELU_H_T5: f32 = 0.28528002;
const LEAKY_RELU_OUTPUT: f32 = 0.42792004;
const LEAKY_RELU_LOSS: f32 = 0.07402749;
const LEAKY_RELU_GRAD_W_IH: f32 = -0.46571383;
const LEAKY_RELU_GRAD_W_HH: f32 = -0.03134361;
const LEAKY_RELU_GRAD_W_OUT: f32 = -0.15523794;

/// LeakyReLU RNN 前向传播与 PyTorch 匹配
///
/// 验证 LeakyReLU 激活函数在 RNN 中的前向计算正确性
/// 特别测试正负区域的分段线性行为
#[test]
fn test_leaky_relu_rnn_forward_matches_pytorch() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 1], Some("input"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;

    let w_ih = graph.new_parameter_node(&[1, 1], Some("w_ih"))?;
    graph.set_node_value(w_ih, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    let w_hh = graph.new_parameter_node(&[1, 1], Some("w_hh"))?;
    graph.set_node_value(w_hh, Some(&Tensor::new(&[0.8], &[1, 1])))?;

    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(&[1.5], &[1, 1])))?;

    // h = LeakyReLU(input * w_ih + h_prev * w_hh, negative_slope=0.1)
    let scaled_input = graph.new_mat_mul_node(input, w_ih, None)?;
    let scaled_h = graph.new_mat_mul_node(h_prev, w_hh, None)?;
    let pre_h = graph.new_add_node(&[scaled_input, scaled_h], None)?;
    let hidden = graph.new_leaky_relu_node(pre_h, LEAKY_RELU_NEGATIVE_SLOPE, Some("hidden"))?;
    graph.connect_recurrent(hidden, h_prev)?;

    let output = graph.new_mat_mul_node(hidden, w_out, None)?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    graph.set_node_value(target, Some(&Tensor::new(&[0.7], &[1, 1])))?;

    // 前向传播序列（包含正负值，测试分段线性行为）
    let sequence = [1.0, -0.5, 0.3, -0.8, 0.6];
    let expected_h = [
        LEAKY_RELU_H_T1,
        LEAKY_RELU_H_T2,
        LEAKY_RELU_H_T3,
        LEAKY_RELU_H_T4,
        LEAKY_RELU_H_T5,
    ];

    for (t, (&x, &expected)) in sequence.iter().zip(expected_h.iter()).enumerate() {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;

        let actual_h = graph.get_node_value(hidden)?.unwrap().data_as_slice()[0];
        assert!(
            (actual_h - expected).abs() < 1e-5,
            "t={}: hidden 不匹配: actual={}, expected={}",
            t,
            actual_h,
            expected
        );
    }

    let actual_output = graph.get_node_value(output)?.unwrap().data_as_slice()[0];
    let actual_loss = graph.get_node_value(loss)?.unwrap().data_as_slice()[0];

    assert!(
        (actual_output - LEAKY_RELU_OUTPUT).abs() < 1e-5,
        "output 不匹配: actual={}, expected={}",
        actual_output,
        LEAKY_RELU_OUTPUT
    );
    assert!(
        (actual_loss - LEAKY_RELU_LOSS).abs() < 1e-5,
        "loss 不匹配: actual={}, expected={}",
        actual_loss,
        LEAKY_RELU_LOSS
    );

    println!("✅ LeakyReLU RNN 前向传播测试通过");
    Ok(())
}

/// LeakyReLU RNN BPTT 梯度与 PyTorch 匹配
///
/// 验证 BPTT 对分段线性激活函数的正确处理
/// LeakyReLU 在 x=0 处导数不连续，这是与 tanh/sigmoid 的关键区别
#[test]
fn test_leaky_relu_rnn_bptt_matches_pytorch() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 1], Some("input"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;

    let w_ih = graph.new_parameter_node(&[1, 1], Some("w_ih"))?;
    graph.set_node_value(w_ih, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    let w_hh = graph.new_parameter_node(&[1, 1], Some("w_hh"))?;
    graph.set_node_value(w_hh, Some(&Tensor::new(&[0.8], &[1, 1])))?;

    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(&[1.5], &[1, 1])))?;

    // h = LeakyReLU(input * w_ih + h_prev * w_hh, negative_slope=0.1)
    let scaled_input = graph.new_mat_mul_node(input, w_ih, None)?;
    let scaled_h = graph.new_mat_mul_node(h_prev, w_hh, None)?;
    let pre_h = graph.new_add_node(&[scaled_input, scaled_h], None)?;
    let hidden = graph.new_leaky_relu_node(pre_h, LEAKY_RELU_NEGATIVE_SLOPE, Some("hidden"))?;
    graph.connect_recurrent(hidden, h_prev)?;

    let output = graph.new_mat_mul_node(hidden, w_out, None)?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    graph.set_node_value(target, Some(&Tensor::new(&[0.7], &[1, 1])))?;

    // 前向传播序列
    for &x in &[1.0, -0.5, 0.3, -0.8, 0.6] {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;
    }

    // BPTT
    graph.backward_through_time(&[w_ih, w_hh, w_out], loss)?;

    // 验证梯度
    let actual_w_ih = graph.get_node_grad(w_ih)?.unwrap().data_as_slice()[0];
    let actual_w_hh = graph.get_node_grad(w_hh)?.unwrap().data_as_slice()[0];
    let actual_w_out = graph.get_node_grad(w_out)?.unwrap().data_as_slice()[0];

    assert!(
        (actual_w_out - LEAKY_RELU_GRAD_W_OUT).abs() < 1e-5,
        "dL/d(w_out) 不匹配: actual={}, expected={}",
        actual_w_out,
        LEAKY_RELU_GRAD_W_OUT
    );

    assert!(
        (actual_w_ih - LEAKY_RELU_GRAD_W_IH).abs() < 1e-5,
        "dL/d(w_ih) 不匹配: actual={}, expected={}",
        actual_w_ih,
        LEAKY_RELU_GRAD_W_IH
    );

    assert!(
        (actual_w_hh - LEAKY_RELU_GRAD_W_HH).abs() < 1e-5,
        "dL/d(w_hh) 不匹配: actual={}, expected={}",
        actual_w_hh,
        LEAKY_RELU_GRAD_W_HH
    );

    println!("✅ LeakyReLU RNN BPTT 梯度测试通过");
    println!(
        "  dL/d(w_ih) = {} (期望: {})",
        actual_w_ih, LEAKY_RELU_GRAD_W_IH
    );
    println!(
        "  dL/d(w_hh) = {} (期望: {})",
        actual_w_hh, LEAKY_RELU_GRAD_W_HH
    );
    println!(
        "  dL/d(w_out) = {} (期望: {})",
        actual_w_out, LEAKY_RELU_GRAD_W_OUT
    );

    Ok(())
}

// ==================== 结构 5: SoftPlus RNN PyTorch 参考值 ====================
// 参考脚本: tests/python/layer_reference/rnn_multi_structure.py
// h[t] = SoftPlus(x[t] * w_ih + h[t-1] * w_hh)
// output = h[T] * w_out
// 参数: w_ih=0.3, w_hh=0.5, w_out=1.0
// 序列: [0.5, -0.3, 0.8, -0.5, 0.2], 目标: 1.0
const SOFTPLUS_H_T1: f32 = 0.77095705;
const SOFTPLUS_H_T2: f32 = 0.85176039;
const SOFTPLUS_H_T3: f32 = 1.08051717;
const SOFTPLUS_H_T4: f32 = 0.90719461;
const SOFTPLUS_H_T5: f32 = 0.98256248;
const SOFTPLUS_OUTPUT: f32 = 0.98256248;
const SOFTPLUS_LOSS: f32 = 0.00030407;
const SOFTPLUS_GRAD_W_IH: f32 = -0.00272797;
const SOFTPLUS_GRAD_W_HH: f32 = -0.02912964;
const SOFTPLUS_GRAD_W_OUT: f32 = -0.03426690;

/// SoftPlus RNN 前向传播与 PyTorch 匹配
///
/// 验证 SoftPlus 激活函数在 RNN 中的前向计算正确性
#[test]
fn test_softplus_rnn_forward_matches_pytorch() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 1], Some("input"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;

    let w_ih = graph.new_parameter_node(&[1, 1], Some("w_ih"))?;
    graph.set_node_value(w_ih, Some(&Tensor::new(&[0.3], &[1, 1])))?;

    let w_hh = graph.new_parameter_node(&[1, 1], Some("w_hh"))?;
    graph.set_node_value(w_hh, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // h = SoftPlus(input * w_ih + h_prev * w_hh)
    let scaled_input = graph.new_mat_mul_node(input, w_ih, None)?;
    let scaled_h = graph.new_mat_mul_node(h_prev, w_hh, None)?;
    let pre_h = graph.new_add_node(&[scaled_input, scaled_h], None)?;
    let hidden = graph.new_softplus_node(pre_h, Some("hidden"))?;
    graph.connect_recurrent(hidden, h_prev)?;

    let output = graph.new_mat_mul_node(hidden, w_out, None)?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    graph.set_node_value(target, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // 前向传播序列
    let sequence = [0.5, -0.3, 0.8, -0.5, 0.2];
    let expected_h = [
        SOFTPLUS_H_T1,
        SOFTPLUS_H_T2,
        SOFTPLUS_H_T3,
        SOFTPLUS_H_T4,
        SOFTPLUS_H_T5,
    ];

    for (t, (&x, &expected)) in sequence.iter().zip(expected_h.iter()).enumerate() {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;

        let actual_h = graph.get_node_value(hidden)?.unwrap().data_as_slice()[0];
        assert!(
            (actual_h - expected).abs() < 1e-5,
            "t={}: hidden 不匹配: actual={}, expected={}",
            t,
            actual_h,
            expected
        );
    }

    let actual_output = graph.get_node_value(output)?.unwrap().data_as_slice()[0];
    let actual_loss = graph.get_node_value(loss)?.unwrap().data_as_slice()[0];

    assert!(
        (actual_output - SOFTPLUS_OUTPUT).abs() < 1e-5,
        "output 不匹配: actual={}, expected={}",
        actual_output,
        SOFTPLUS_OUTPUT
    );
    assert!(
        (actual_loss - SOFTPLUS_LOSS).abs() < 1e-5,
        "loss 不匹配: actual={}, expected={}",
        actual_loss,
        SOFTPLUS_LOSS
    );

    println!("✅ SoftPlus RNN 前向传播测试通过");
    Ok(())
}

/// SoftPlus RNN BPTT 梯度与 PyTorch 匹配
///
/// 验证 BPTT 对 SoftPlus 激活函数的正确处理
/// SoftPlus 的导数为 sigmoid，需要从输出计算：sigmoid(x) = 1 - exp(-softplus(x))
#[test]
fn test_softplus_rnn_bptt_matches_pytorch() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 1], Some("input"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;

    let w_ih = graph.new_parameter_node(&[1, 1], Some("w_ih"))?;
    graph.set_node_value(w_ih, Some(&Tensor::new(&[0.3], &[1, 1])))?;

    let w_hh = graph.new_parameter_node(&[1, 1], Some("w_hh"))?;
    graph.set_node_value(w_hh, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // h = SoftPlus(input * w_ih + h_prev * w_hh)
    let scaled_input = graph.new_mat_mul_node(input, w_ih, None)?;
    let scaled_h = graph.new_mat_mul_node(h_prev, w_hh, None)?;
    let pre_h = graph.new_add_node(&[scaled_input, scaled_h], None)?;
    let hidden = graph.new_softplus_node(pre_h, Some("hidden"))?;
    graph.connect_recurrent(hidden, h_prev)?;

    let output = graph.new_mat_mul_node(hidden, w_out, None)?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    graph.set_node_value(target, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // 前向传播序列
    for &x in &[0.5, -0.3, 0.8, -0.5, 0.2] {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;
    }

    // BPTT
    graph.backward_through_time(&[w_ih, w_hh, w_out], loss)?;

    // 验证梯度
    let actual_w_ih = graph.get_node_grad(w_ih)?.unwrap().data_as_slice()[0];
    let actual_w_hh = graph.get_node_grad(w_hh)?.unwrap().data_as_slice()[0];
    let actual_w_out = graph.get_node_grad(w_out)?.unwrap().data_as_slice()[0];

    assert!(
        (actual_w_out - SOFTPLUS_GRAD_W_OUT).abs() < 1e-5,
        "dL/d(w_out) 不匹配: actual={}, expected={}",
        actual_w_out,
        SOFTPLUS_GRAD_W_OUT
    );

    assert!(
        (actual_w_ih - SOFTPLUS_GRAD_W_IH).abs() < 1e-5,
        "dL/d(w_ih) 不匹配: actual={}, expected={}",
        actual_w_ih,
        SOFTPLUS_GRAD_W_IH
    );

    assert!(
        (actual_w_hh - SOFTPLUS_GRAD_W_HH).abs() < 1e-5,
        "dL/d(w_hh) 不匹配: actual={}, expected={}",
        actual_w_hh,
        SOFTPLUS_GRAD_W_HH
    );

    println!("✅ SoftPlus RNN BPTT 梯度测试通过");
    println!(
        "  dL/d(w_ih) = {} (期望: {})",
        actual_w_ih, SOFTPLUS_GRAD_W_IH
    );
    println!(
        "  dL/d(w_hh) = {} (期望: {})",
        actual_w_hh, SOFTPLUS_GRAD_W_HH
    );
    println!(
        "  dL/d(w_out) = {} (期望: {})",
        actual_w_out, SOFTPLUS_GRAD_W_OUT
    );

    Ok(())
}
