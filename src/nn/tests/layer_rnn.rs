/*
 * @Author       : 老董
 * @Date         : 2025-12-30
 * @Description  : RNN Layer 单元测试（与 PyTorch 数值对照）
 *
 * 参考值来源: tests/python/layer_reference/rnn_layer_reference.py
 */

use crate::nn::layer::rnn;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== PyTorch 参考常量 ====================

// 测试 1: 简单前向传播 (batch=2, input=3, hidden=4)
const TEST1_X: &[f32] = &[
    1.0, 2.0, 3.0, // batch 0
    0.5, 1.0, 1.5, // batch 1
];
const TEST1_W_IH: &[f32] = &[
    0.1, 0.2, 0.3, 0.4, // input[0] -> hidden
    0.5, 0.6, 0.7, 0.8, // input[1] -> hidden
    0.9, 1.0, 1.1, 1.2, // input[2] -> hidden
];
const TEST1_W_HH: &[f32] = &[
    0.1, 0.0, 0.0, 0.0, //
    0.0, 0.2, 0.0, 0.0, //
    0.0, 0.0, 0.3, 0.0, //
    0.0, 0.0, 0.0, 0.4, //
];
const TEST1_B_H: &[f32] = &[0.1, 0.2, 0.3, 0.4];
const TEST1_HIDDEN: &[f32] = &[
    0.99918085, 0.99979794, 0.99995017, 0.99998772, // batch 0
    0.96402758, 0.98367488, 0.99263149, 0.99668241, // batch 1
];

// 测试 2: 多时间步前向传播 (batch=1, input=2, hidden=3, seq_len=3)
const TEST2_W_IH: &[f32] = &[0.5, 0.3, 0.1, 0.2, 0.4, 0.6];
const TEST2_W_HH: &[f32] = &[0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3];
const TEST2_H_T0: &[f32] = &[0.46211717, 0.29131263, 0.09966800];
const TEST2_H_T1: &[f32] = &[0.24135433, 0.42866707, 0.55798364];
const TEST2_H_T2: &[f32] = &[0.61946434, 0.65598524, 0.70004827];

// 测试 3: 反向传播梯度 (batch=1, input=2, hidden=2, seq_len=2)
const TEST3_W_IH: &[f32] = &[0.5, 0.3, 0.2, 0.4];
const TEST3_W_HH: &[f32] = &[0.1, 0.0, 0.0, 0.2];
const TEST3_B_H: &[f32] = &[0.1, 0.2];
const TEST3_W_OUT: &[f32] = &[1.0, 1.0];
const TEST3_SEQ_0: &[f32] = &[1.0, 0.5];
const TEST3_SEQ_1: &[f32] = &[0.5, 1.0];
const TEST3_TARGET: f32 = 0.8;
const TEST3_OUTPUT: f32 = 1.24625218;
const TEST3_LOSS: f32 = 0.19914100;
const TEST3_GRAD_W_IH: &[f32] = &[0.35383806, 0.28394660, 0.64788759, 0.48165056];
const TEST3_GRAD_W_HH: &[f32] = &[0.37951767, 0.27372000, 0.37951767, 0.27372000];
const TEST3_GRAD_B_H: &[f32] = &[0.66781712, 0.51039809];
const TEST3_GRAD_W_OUT: &[f32] = &[0.48591015, 0.62637532];

// ==================== 基础功能测试 ====================

/// 测试 RNN 层创建
#[test]
fn test_rnn_creation() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 16;
    let input_size = 10;
    let hidden_size = 20;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let rnn_out = rnn(&mut graph, input, input_size, hidden_size, batch_size, Some("rnn1"))?;

    // 验证节点创建成功
    assert!(graph.get_node_value(rnn_out.hidden).is_ok());
    assert!(graph.get_node_value(rnn_out.h_prev).is_ok());
    assert!(graph.get_node_value(rnn_out.w_ih).is_ok());
    assert!(graph.get_node_value(rnn_out.w_hh).is_ok());
    assert!(graph.get_node_value(rnn_out.b_h).is_ok());

    Ok(())
}

/// 测试 RNN 参数形状
#[test]
fn test_rnn_shapes() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 8;
    let input_size = 4;
    let hidden_size = 6;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let rnn_out = rnn(&mut graph, input, input_size, hidden_size, batch_size, Some("rnn1"))?;

    // 验证权重形状
    let w_ih_shape = graph.get_node(rnn_out.w_ih)?.value_expected_shape();
    assert_eq!(w_ih_shape, &[input_size, hidden_size]);

    let w_hh_shape = graph.get_node(rnn_out.w_hh)?.value_expected_shape();
    assert_eq!(w_hh_shape, &[hidden_size, hidden_size]);

    let b_h_shape = graph.get_node(rnn_out.b_h)?.value_expected_shape();
    assert_eq!(b_h_shape, &[1, hidden_size]);

    // 验证状态形状
    let h_prev_shape = graph.get_node(rnn_out.h_prev)?.value_expected_shape();
    assert_eq!(h_prev_shape, &[batch_size, hidden_size]);

    Ok(())
}

// ==================== PyTorch 数值对照测试 ====================

/// 测试 1: 简单前向传播（与 PyTorch 对照）
#[test]
fn test_rnn_forward_pytorch_comparison() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 2;
    let input_size = 3;
    let hidden_size = 4;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let rnn_out = rnn(&mut graph, input, input_size, hidden_size, batch_size, Some("rnn1"))?;

    // 设置与 PyTorch 相同的权重
    graph.set_node_value(input, Some(&Tensor::new(TEST1_X, &[batch_size, input_size])))?;
    graph.set_node_value(
        rnn_out.w_ih,
        Some(&Tensor::new(TEST1_W_IH, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        rnn_out.w_hh,
        Some(&Tensor::new(TEST1_W_HH, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(rnn_out.b_h, Some(&Tensor::new(TEST1_B_H, &[1, hidden_size])))?;

    // 前向传播（单时间步）
    graph.step(rnn_out.hidden)?;

    // 验证隐藏状态
    let hidden = graph.get_node_value(rnn_out.hidden)?.unwrap();
    assert_eq!(hidden.shape(), &[batch_size, hidden_size]);

    let hidden_data = hidden.data_as_slice();
    for (i, (&actual, &expected)) in hidden_data.iter().zip(TEST1_HIDDEN.iter()).enumerate() {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        if i < 4 {
            println!("batch[0][{}]: actual={:.6}, expected={:.6}", i, actual, expected);
        }
    }

    println!("✅ 测试 1 通过：前向传播与 PyTorch 一致");
    Ok(())
}

/// 测试 2: 多时间步前向传播（与 PyTorch 对照）
#[test]
fn test_rnn_multi_step_forward_pytorch_comparison() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 3;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let rnn_out = rnn(&mut graph, input, input_size, hidden_size, batch_size, Some("rnn1"))?;

    // 设置权重
    graph.set_node_value(
        rnn_out.w_ih,
        Some(&Tensor::new(TEST2_W_IH, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        rnn_out.w_hh,
        Some(&Tensor::new(TEST2_W_HH, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(
        rnn_out.b_h,
        Some(&Tensor::zeros(&[1, hidden_size])),
    )?;

    // 输入序列
    let sequence = vec![
        vec![1.0, 0.0], // t=0
        vec![0.0, 1.0], // t=1
        vec![1.0, 1.0], // t=2
    ];
    let expected_hidden = vec![TEST2_H_T0, TEST2_H_T1, TEST2_H_T2];

    // 执行多时间步
    for (t, (x_t, expected)) in sequence.iter().zip(expected_hidden.iter()).enumerate() {
        graph.set_node_value(input, Some(&Tensor::new(x_t, &[batch_size, input_size])))?;
        graph.step(rnn_out.hidden)?;

        let hidden = graph.get_node_value(rnn_out.hidden)?.unwrap();
        let hidden_data = hidden.data_as_slice();

        println!("t={}: actual={:?}", t, hidden_data);
        for (&actual, &exp) in hidden_data.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, exp, epsilon = 1e-5);
        }
    }

    println!("✅ 测试 2 通过：多时间步前向传播与 PyTorch 一致");
    Ok(())
}

/// 测试 3: BPTT 反向传播梯度（与 PyTorch 对照）
#[test]
fn test_rnn_bptt_gradient_pytorch_comparison() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 2;

    // 创建 RNN 层
    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let rnn_out = rnn(&mut graph, input, input_size, hidden_size, batch_size, Some("rnn1"))?;

    // 设置权重
    graph.set_node_value(
        rnn_out.w_ih,
        Some(&Tensor::new(TEST3_W_IH, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        rnn_out.w_hh,
        Some(&Tensor::new(TEST3_W_HH, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(rnn_out.b_h, Some(&Tensor::new(TEST3_B_H, &[1, hidden_size])))?;

    // 创建输出层
    let w_out = graph.new_parameter_node(&[hidden_size, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(TEST3_W_OUT, &[hidden_size, 1])))?;

    let output = graph.new_mat_mul_node(rnn_out.hidden, w_out, Some("output"))?;

    // 创建 loss
    let target = graph.new_input_node(&[batch_size, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 输入序列
    let sequence = vec![TEST3_SEQ_0, TEST3_SEQ_1];

    // 设置目标值（必须在 step 之前设置，否则 loss 节点无法计算）
    graph.set_node_value(target, Some(&Tensor::new(&[TEST3_TARGET], &[batch_size, 1])))?;

    // 前向传播整个序列
    for x_t in &sequence {
        graph.set_node_value(input, Some(&Tensor::new(*x_t, &[batch_size, input_size])))?;
        graph.step(loss)?;
    }

    // 验证输出和 loss
    let output_val = graph.get_node_value(output)?.unwrap();
    assert_abs_diff_eq!(output_val[[0, 0]], TEST3_OUTPUT, epsilon = 1e-5);

    let loss_val = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], TEST3_LOSS, epsilon = 1e-5);

    // 反向传播
    graph.backward_through_time(&[rnn_out.w_ih, rnn_out.w_hh, rnn_out.b_h, w_out], loss)?;

    // 验证梯度
    let grad_w_ih = graph.get_node_jacobi(rnn_out.w_ih)?.unwrap();
    let grad_w_ih_flat: Vec<f32> = grad_w_ih.data_as_slice().to_vec();
    println!("grad_w_ih: {:?}", grad_w_ih_flat);
    for (&actual, &expected) in grad_w_ih_flat.iter().zip(TEST3_GRAD_W_IH.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    let grad_w_hh = graph.get_node_jacobi(rnn_out.w_hh)?.unwrap();
    let grad_w_hh_flat: Vec<f32> = grad_w_hh.data_as_slice().to_vec();
    println!("grad_w_hh: {:?}", grad_w_hh_flat);
    for (&actual, &expected) in grad_w_hh_flat.iter().zip(TEST3_GRAD_W_HH.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    let grad_b_h = graph.get_node_jacobi(rnn_out.b_h)?.unwrap();
    let grad_b_h_flat: Vec<f32> = grad_b_h.data_as_slice().to_vec();
    println!("grad_b_h: {:?}", grad_b_h_flat);
    for (&actual, &expected) in grad_b_h_flat.iter().zip(TEST3_GRAD_B_H.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    let grad_w_out = graph.get_node_jacobi(w_out)?.unwrap();
    let grad_w_out_flat: Vec<f32> = grad_w_out.data_as_slice().to_vec();
    println!("grad_w_out: {:?}", grad_w_out_flat);
    for (&actual, &expected) in grad_w_out_flat.iter().zip(TEST3_GRAD_W_OUT.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    println!("✅ 测试 3 通过：BPTT 梯度与 PyTorch 一致");
    Ok(())
}

// ==================== 节点名称测试 ====================

/// 测试 RNN 层节点命名
#[test]
fn test_rnn_node_naming() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 4;
    let input_size = 8;
    let hidden_size = 16;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let rnn_out = rnn(&mut graph, input, input_size, hidden_size, batch_size, Some("encoder"))?;

    // 验证节点名称
    let w_ih_name = graph.get_node(rnn_out.w_ih)?.name();
    assert!(w_ih_name.contains("encoder") && w_ih_name.contains("W_ih"));

    let w_hh_name = graph.get_node(rnn_out.w_hh)?.name();
    assert!(w_hh_name.contains("encoder") && w_hh_name.contains("W_hh"));

    let b_h_name = graph.get_node(rnn_out.b_h)?.name();
    assert!(b_h_name.contains("encoder") && b_h_name.contains("b_h"));

    let h_prev_name = graph.get_node(rnn_out.h_prev)?.name();
    assert!(h_prev_name.contains("encoder") && h_prev_name.contains("h_prev"));

    Ok(())
}

// ==================== reset() 测试 ====================

/// 测试 reset() 清除隐藏状态
#[test]
fn test_rnn_reset() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 2;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let rnn_out = rnn(&mut graph, input, input_size, hidden_size, batch_size, Some("rnn"))?;

    // 设置权重
    graph.set_node_value(rnn_out.w_ih, Some(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2])))?;
    graph.set_node_value(rnn_out.w_hh, Some(&Tensor::new(&[0.1, 0.0, 0.0, 0.1], &[2, 2])))?;

    // 运行几步
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;
    graph.step(rnn_out.hidden)?;
    graph.step(rnn_out.hidden)?;

    // 获取当前隐藏状态
    let h_before_reset = graph.get_node_value(rnn_out.hidden)?.unwrap().clone();
    assert!(h_before_reset[[0, 0]].abs() > 0.1); // 确保不是 0

    // reset
    graph.reset();

    // 再运行一步
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;
    graph.step(rnn_out.hidden)?;

    // 获取 reset 后的隐藏状态（应该和第一步相同，因为 h_prev 被重置为 0）
    let h_after_reset = graph.get_node_value(rnn_out.hidden)?.unwrap().clone();

    // 重新从头开始
    graph.reset();
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;
    graph.step(rnn_out.hidden)?;
    let h_fresh = graph.get_node_value(rnn_out.hidden)?.unwrap();

    // reset 后第一步应该和全新开始的第一步相同
    assert_abs_diff_eq!(h_after_reset[[0, 0]], h_fresh[[0, 0]], epsilon = 1e-6);
    assert_abs_diff_eq!(h_after_reset[[0, 1]], h_fresh[[0, 1]], epsilon = 1e-6);

    println!("✅ reset() 正确清除隐藏状态");
    Ok(())
}

