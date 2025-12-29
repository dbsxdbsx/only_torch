/*
 * @Author       : 老董
 * @Date         : 2025-12-30
 * @Description  : LSTM Layer 单元测试（与 PyTorch 数值对照）
 *
 * 参考值来源: tests/pytorch_reference/lstm_layer_reference.py
 */

use crate::nn::layer::lstm;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== PyTorch 参考常量 ====================

// 测试 1: 简单前向传播 (batch=2, input=3, hidden=2)
const TEST1_X: &[f32] = &[1.0, 0.5, 0.2, 0.3, 0.8, 0.1];
const TEST1_W_II: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
const TEST1_W_HI: &[f32] = &[0.1, 0.0, 0.0, 0.1];
const TEST1_B_I: &[f32] = &[0.0, 0.0];
const TEST1_W_IF: &[f32] = &[0.2, 0.1, 0.4, 0.3, 0.6, 0.5];
const TEST1_W_HF: &[f32] = &[0.2, 0.0, 0.0, 0.2];
const TEST1_B_F: &[f32] = &[1.0, 1.0];
const TEST1_W_IG: &[f32] = &[0.3, 0.2, 0.5, 0.4, 0.7, 0.6];
const TEST1_W_HG: &[f32] = &[0.3, 0.0, 0.0, 0.3];
const TEST1_B_G: &[f32] = &[0.0, 0.0];
const TEST1_W_IO: &[f32] = &[0.4, 0.3, 0.6, 0.5, 0.8, 0.7];
const TEST1_W_HO: &[f32] = &[0.4, 0.0, 0.0, 0.4];
const TEST1_B_O: &[f32] = &[0.0, 0.0];
const TEST1_HIDDEN: &[f32] = &[0.23684801, 0.19375373, 0.18987750, 0.15683776];
const TEST1_CELL: &[f32] = &[0.35078675, 0.29958847, 0.29428363, 0.25160298];

// 测试 2: 多时间步 (batch=1, input=2, hidden=2, seq_len=3)
const TEST2_W_II: &[f32] = &[0.5, 0.3, 0.2, 0.4];
const TEST2_W_HI: &[f32] = &[0.1, 0.0, 0.0, 0.1];
const TEST2_W_IF: &[f32] = &[0.3, 0.5, 0.4, 0.2];
const TEST2_W_HF: &[f32] = &[0.1, 0.0, 0.0, 0.1];
const TEST2_W_IG: &[f32] = &[0.4, 0.2, 0.3, 0.5];
const TEST2_W_HG: &[f32] = &[0.2, 0.0, 0.0, 0.2];
const TEST2_W_IO: &[f32] = &[0.2, 0.4, 0.5, 0.3];
const TEST2_W_HO: &[f32] = &[0.1, 0.0, 0.0, 0.1];
const TEST2_H_T0: &[f32] = &[0.12766583, 0.06759029];
const TEST2_C_T0: &[f32] = &[0.23650280, 0.11338077];
const TEST2_H_T1: &[f32] = &[0.21817833, 0.20445023];
const TEST2_C_T1: &[f32] = &[0.36411276, 0.37102786];
const TEST2_H_T2: &[f32] = &[0.42088586, 0.42253390];
const TEST2_C_T2: &[f32] = &[0.73379391, 0.73829132];

// 测试 3: 反向传播 (batch=1, input=2, hidden=2, seq_len=2)
const TEST3_OUTPUT: f32 = 0.65790576;
const TEST3_LOSS: f32 = 0.02019078;
const TEST3_GRAD_W_II: &[f32] = &[-0.02190232, -0.01997453, -0.02279367, -0.02265729];
const TEST3_GRAD_W_OUT: &[f32] = &[-0.09518241, -0.09178685];

// ==================== 基础功能测试 ====================

/// 测试 LSTM 层创建
#[test]
fn test_lstm_creation() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 16;
    let input_size = 10;
    let hidden_size = 20;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(&mut graph, input, input_size, hidden_size, batch_size, Some("lstm1"))?;

    // 验证节点创建成功
    assert!(graph.get_node_value(lstm_out.hidden).is_ok());
    assert!(graph.get_node_value(lstm_out.cell).is_ok());
    assert!(graph.get_node_value(lstm_out.h_prev).is_ok());
    assert!(graph.get_node_value(lstm_out.c_prev).is_ok());

    // 验证门参数
    assert!(graph.get_node_value(lstm_out.w_ii).is_ok());
    assert!(graph.get_node_value(lstm_out.w_if).is_ok());
    assert!(graph.get_node_value(lstm_out.w_ig).is_ok());
    assert!(graph.get_node_value(lstm_out.w_io).is_ok());

    Ok(())
}

/// 测试 LSTM 参数形状
#[test]
fn test_lstm_shapes() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 8;
    let input_size = 4;
    let hidden_size = 6;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(&mut graph, input, input_size, hidden_size, batch_size, Some("lstm1"))?;

    // 验证输入门权重形状
    assert_eq!(
        graph.get_node(lstm_out.w_ii)?.value_expected_shape(),
        &[input_size, hidden_size]
    );
    assert_eq!(
        graph.get_node(lstm_out.w_hi)?.value_expected_shape(),
        &[hidden_size, hidden_size]
    );
    assert_eq!(
        graph.get_node(lstm_out.b_i)?.value_expected_shape(),
        &[1, hidden_size]
    );

    // 验证状态形状
    assert_eq!(
        graph.get_node(lstm_out.h_prev)?.value_expected_shape(),
        &[batch_size, hidden_size]
    );
    assert_eq!(
        graph.get_node(lstm_out.c_prev)?.value_expected_shape(),
        &[batch_size, hidden_size]
    );

    Ok(())
}

// ==================== PyTorch 数值对照测试 ====================

/// 测试 1: 简单前向传播（与 PyTorch 对照）
#[test]
fn test_lstm_forward_pytorch_comparison() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 2;
    let input_size = 3;
    let hidden_size = 2;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(&mut graph, input, input_size, hidden_size, batch_size, Some("lstm"))?;

    // 设置与 PyTorch 相同的权重
    graph.set_node_value(input, Some(&Tensor::new(TEST1_X, &[batch_size, input_size])))?;

    // 输入门
    graph.set_node_value(lstm_out.w_ii, Some(&Tensor::new(TEST1_W_II, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_hi, Some(&Tensor::new(TEST1_W_HI, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_i, Some(&Tensor::new(TEST1_B_I, &[1, hidden_size])))?;

    // 遗忘门
    graph.set_node_value(lstm_out.w_if, Some(&Tensor::new(TEST1_W_IF, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_hf, Some(&Tensor::new(TEST1_W_HF, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_f, Some(&Tensor::new(TEST1_B_F, &[1, hidden_size])))?;

    // 候选细胞
    graph.set_node_value(lstm_out.w_ig, Some(&Tensor::new(TEST1_W_IG, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_hg, Some(&Tensor::new(TEST1_W_HG, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_g, Some(&Tensor::new(TEST1_B_G, &[1, hidden_size])))?;

    // 输出门
    graph.set_node_value(lstm_out.w_io, Some(&Tensor::new(TEST1_W_IO, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_ho, Some(&Tensor::new(TEST1_W_HO, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_o, Some(&Tensor::new(TEST1_B_O, &[1, hidden_size])))?;

    // 前向传播
    graph.step(lstm_out.hidden)?;

    // 验证隐藏状态
    let hidden = graph.get_node_value(lstm_out.hidden)?.unwrap();
    let hidden_data = hidden.data_as_slice();
    println!("Hidden: {:?}", hidden_data);
    for (&actual, &expected) in hidden_data.iter().zip(TEST1_HIDDEN.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
    }

    // 验证细胞状态
    let cell = graph.get_node_value(lstm_out.cell)?.unwrap();
    let cell_data = cell.data_as_slice();
    println!("Cell: {:?}", cell_data);
    for (&actual, &expected) in cell_data.iter().zip(TEST1_CELL.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
    }

    println!("✅ 测试 1 通过：LSTM 前向传播与 PyTorch 一致");
    Ok(())
}

/// 测试 2: 多时间步前向传播（与 PyTorch 对照）
#[test]
fn test_lstm_multi_step_forward_pytorch_comparison() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 2;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(&mut graph, input, input_size, hidden_size, batch_size, Some("lstm"))?;

    // 设置权重
    graph.set_node_value(lstm_out.w_ii, Some(&Tensor::new(TEST2_W_II, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_hi, Some(&Tensor::new(TEST2_W_HI, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_i, Some(&Tensor::zeros(&[1, hidden_size])))?;

    graph.set_node_value(lstm_out.w_if, Some(&Tensor::new(TEST2_W_IF, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_hf, Some(&Tensor::new(TEST2_W_HF, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_f, Some(&Tensor::ones(&[1, hidden_size])))?;

    graph.set_node_value(lstm_out.w_ig, Some(&Tensor::new(TEST2_W_IG, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_hg, Some(&Tensor::new(TEST2_W_HG, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_g, Some(&Tensor::zeros(&[1, hidden_size])))?;

    graph.set_node_value(lstm_out.w_io, Some(&Tensor::new(TEST2_W_IO, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_ho, Some(&Tensor::new(TEST2_W_HO, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_o, Some(&Tensor::zeros(&[1, hidden_size])))?;

    // 输入序列
    let sequence = vec![
        vec![1.0, 0.0], // t=0
        vec![0.0, 1.0], // t=1
        vec![1.0, 1.0], // t=2
    ];
    let expected_h = vec![TEST2_H_T0, TEST2_H_T1, TEST2_H_T2];
    let expected_c = vec![TEST2_C_T0, TEST2_C_T1, TEST2_C_T2];

    // 执行多时间步
    for (t, (x_t, (exp_h, exp_c))) in sequence
        .iter()
        .zip(expected_h.iter().zip(expected_c.iter()))
        .enumerate()
    {
        graph.set_node_value(input, Some(&Tensor::new(x_t, &[batch_size, input_size])))?;
        graph.step(lstm_out.hidden)?;

        let hidden = graph.get_node_value(lstm_out.hidden)?.unwrap();
        let cell = graph.get_node_value(lstm_out.cell)?.unwrap();

        println!(
            "t={}: h={:?}, c={:?}",
            t,
            hidden.data_as_slice(),
            cell.data_as_slice()
        );

        for (&actual, &expected) in hidden.data_as_slice().iter().zip(exp_h.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        }
        for (&actual, &expected) in cell.data_as_slice().iter().zip(exp_c.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        }
    }

    println!("✅ 测试 2 通过：LSTM 多时间步前向传播与 PyTorch 一致");
    Ok(())
}

/// 测试 3: BPTT 反向传播梯度（与 PyTorch 对照）
#[test]
fn test_lstm_bptt_gradient_pytorch_comparison() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 2;

    // 创建 LSTM 层
    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(&mut graph, input, input_size, hidden_size, batch_size, Some("lstm"))?;

    // 设置与 PyTorch 相同的权重
    graph.set_node_value(lstm_out.w_ii, Some(&Tensor::new(TEST2_W_II, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_hi, Some(&Tensor::new(TEST2_W_HI, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_i, Some(&Tensor::zeros(&[1, hidden_size])))?;

    graph.set_node_value(lstm_out.w_if, Some(&Tensor::new(TEST2_W_IF, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_hf, Some(&Tensor::new(TEST2_W_HF, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_f, Some(&Tensor::ones(&[1, hidden_size])))?;

    graph.set_node_value(lstm_out.w_ig, Some(&Tensor::new(TEST2_W_IG, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_hg, Some(&Tensor::new(TEST2_W_HG, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_g, Some(&Tensor::zeros(&[1, hidden_size])))?;

    graph.set_node_value(lstm_out.w_io, Some(&Tensor::new(TEST2_W_IO, &[input_size, hidden_size])))?;
    graph.set_node_value(lstm_out.w_ho, Some(&Tensor::new(TEST2_W_HO, &[hidden_size, hidden_size])))?;
    graph.set_node_value(lstm_out.b_o, Some(&Tensor::zeros(&[1, hidden_size])))?;

    // 创建输出层
    let w_out = graph.new_parameter_node(&[hidden_size, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::new(&[1.0, 1.0], &[hidden_size, 1])))?;

    let output = graph.new_mat_mul_node(lstm_out.hidden, w_out, Some("output"))?;

    // 创建 loss
    let target = graph.new_input_node(&[batch_size, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 设置目标
    graph.set_node_value(target, Some(&Tensor::new(&[0.8], &[batch_size, 1])))?;

    // 输入序列
    let sequence = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

    // 前向传播
    for x_t in &sequence {
        graph.set_node_value(input, Some(&Tensor::new(x_t, &[batch_size, input_size])))?;
        graph.step(loss)?;
    }

    // 验证输出和 loss
    let output_val = graph.get_node_value(output)?.unwrap();
    println!("Output: {:.8}", output_val[[0, 0]]);
    assert_abs_diff_eq!(output_val[[0, 0]], TEST3_OUTPUT, epsilon = 1e-5);

    let loss_val = graph.get_node_value(loss)?.unwrap();
    println!("Loss: {:.8}", loss_val[[0, 0]]);
    assert_abs_diff_eq!(loss_val[[0, 0]], TEST3_LOSS, epsilon = 1e-5);

    // 反向传播
    let params = vec![
        lstm_out.w_ii,
        lstm_out.w_hi,
        lstm_out.b_i,
        lstm_out.w_if,
        lstm_out.w_hf,
        lstm_out.b_f,
        lstm_out.w_ig,
        lstm_out.w_hg,
        lstm_out.b_g,
        lstm_out.w_io,
        lstm_out.w_ho,
        lstm_out.b_o,
        w_out,
    ];
    graph.backward_through_time(&params, loss)?;

    // 验证 w_ii 梯度
    let grad_w_ii = graph.get_node_jacobi(lstm_out.w_ii)?.unwrap();
    let grad_w_ii_flat: Vec<f32> = grad_w_ii.data_as_slice().to_vec();
    println!("grad_w_ii: {:?}", grad_w_ii_flat);
    for (&actual, &expected) in grad_w_ii_flat.iter().zip(TEST3_GRAD_W_II.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    // 验证 w_out 梯度
    let grad_w_out = graph.get_node_jacobi(w_out)?.unwrap();
    let grad_w_out_flat: Vec<f32> = grad_w_out.data_as_slice().to_vec();
    println!("grad_w_out: {:?}", grad_w_out_flat);
    for (&actual, &expected) in grad_w_out_flat.iter().zip(TEST3_GRAD_W_OUT.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    println!("✅ 测试 3 通过：LSTM BPTT 梯度与 PyTorch 一致");
    Ok(())
}

// ==================== 节点名称测试 ====================

/// 测试 LSTM 层节点命名
#[test]
fn test_lstm_node_naming() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 4;
    let input_size = 8;
    let hidden_size = 16;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(&mut graph, input, input_size, hidden_size, batch_size, Some("encoder"))?;

    // 验证节点名称
    let w_ii_name = graph.get_node(lstm_out.w_ii)?.name();
    assert!(w_ii_name.contains("encoder") && w_ii_name.contains("W_ii"));

    let h_prev_name = graph.get_node(lstm_out.h_prev)?.name();
    assert!(h_prev_name.contains("encoder") && h_prev_name.contains("h_prev"));

    let c_prev_name = graph.get_node(lstm_out.c_prev)?.name();
    assert!(c_prev_name.contains("encoder") && c_prev_name.contains("c_prev"));

    Ok(())
}

// ==================== reset() 测试 ====================

/// 测试 reset() 清除 LSTM 状态
#[test]
fn test_lstm_reset() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 2;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(&mut graph, input, input_size, hidden_size, batch_size, Some("lstm"))?;

    // 设置固定权重，确保输出不为 0
    graph.set_node_value(lstm_out.w_ii, Some(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2])))?;
    graph.set_node_value(lstm_out.w_ig, Some(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2])))?;
    graph.set_node_value(lstm_out.w_io, Some(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2])))?;

    // 运行几步
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;
    graph.step(lstm_out.hidden)?;
    graph.step(lstm_out.hidden)?;

    // 获取当前状态（运行两步后应该有非零值）
    let h_before = graph.get_node_value(lstm_out.hidden)?.unwrap().clone();
    let c_before = graph.get_node_value(lstm_out.cell)?.unwrap().clone();
    println!("Before reset: h={:?}, c={:?}", h_before.data_as_slice(), c_before.data_as_slice());
    assert!(h_before[[0, 0]].abs() > 0.001, "h should be non-zero after 2 steps");
    assert!(c_before[[0, 0]].abs() > 0.001, "c should be non-zero after 2 steps");

    // reset
    graph.reset();

    // 再运行一步
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;
    graph.step(lstm_out.hidden)?;
    let h_after = graph.get_node_value(lstm_out.hidden)?.unwrap().clone();

    // 重新从头开始
    graph.reset();
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;
    graph.step(lstm_out.hidden)?;
    let h_fresh = graph.get_node_value(lstm_out.hidden)?.unwrap();

    // reset 后第一步应该和全新开始的第一步相同
    assert_abs_diff_eq!(h_after[[0, 0]], h_fresh[[0, 0]], epsilon = 1e-6);
    assert_abs_diff_eq!(h_after[[0, 1]], h_fresh[[0, 1]], epsilon = 1e-6);

    println!("✅ reset() 正确清除 LSTM 状态");
    Ok(())
}

