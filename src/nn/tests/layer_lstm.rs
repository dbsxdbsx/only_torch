/*
 * @Author       : 老董
 * @Date         : 2025-12-30
 * @Description  : LSTM Layer 单元测试（与 PyTorch 数值对照）
 *
 * 参考值来源: tests/python/layer_reference/lstm_layer_reference.py
 */

use crate::nn::layer::lstm;
use crate::nn::{GraphInner, GraphError};
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
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 16;
    let input_size = 10;
    let hidden_size = 20;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm1"),
    )?;

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
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 8;
    let input_size = 4;
    let hidden_size = 6;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm1"),
    )?;

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
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 2;
    let input_size = 3;
    let hidden_size = 2;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm"),
    )?;

    // 设置与 PyTorch 相同的权重
    graph.set_node_value(
        input,
        Some(&Tensor::new(TEST1_X, &[batch_size, input_size])),
    )?;

    // 输入门
    graph.set_node_value(
        lstm_out.w_ii,
        Some(&Tensor::new(TEST1_W_II, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_hi,
        Some(&Tensor::new(TEST1_W_HI, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.b_i,
        Some(&Tensor::new(TEST1_B_I, &[1, hidden_size])),
    )?;

    // 遗忘门
    graph.set_node_value(
        lstm_out.w_if,
        Some(&Tensor::new(TEST1_W_IF, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_hf,
        Some(&Tensor::new(TEST1_W_HF, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.b_f,
        Some(&Tensor::new(TEST1_B_F, &[1, hidden_size])),
    )?;

    // 候选细胞
    graph.set_node_value(
        lstm_out.w_ig,
        Some(&Tensor::new(TEST1_W_IG, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_hg,
        Some(&Tensor::new(TEST1_W_HG, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.b_g,
        Some(&Tensor::new(TEST1_B_G, &[1, hidden_size])),
    )?;

    // 输出门
    graph.set_node_value(
        lstm_out.w_io,
        Some(&Tensor::new(TEST1_W_IO, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_ho,
        Some(&Tensor::new(TEST1_W_HO, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.b_o,
        Some(&Tensor::new(TEST1_B_O, &[1, hidden_size])),
    )?;

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
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 2;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm"),
    )?;

    // 设置权重
    graph.set_node_value(
        lstm_out.w_ii,
        Some(&Tensor::new(TEST2_W_II, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_hi,
        Some(&Tensor::new(TEST2_W_HI, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(lstm_out.b_i, Some(&Tensor::zeros(&[1, hidden_size])))?;

    graph.set_node_value(
        lstm_out.w_if,
        Some(&Tensor::new(TEST2_W_IF, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_hf,
        Some(&Tensor::new(TEST2_W_HF, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(lstm_out.b_f, Some(&Tensor::ones(&[1, hidden_size])))?;

    graph.set_node_value(
        lstm_out.w_ig,
        Some(&Tensor::new(TEST2_W_IG, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_hg,
        Some(&Tensor::new(TEST2_W_HG, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(lstm_out.b_g, Some(&Tensor::zeros(&[1, hidden_size])))?;

    graph.set_node_value(
        lstm_out.w_io,
        Some(&Tensor::new(TEST2_W_IO, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_ho,
        Some(&Tensor::new(TEST2_W_HO, &[hidden_size, hidden_size])),
    )?;
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
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 2;

    // 创建 LSTM 层
    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm"),
    )?;

    // 设置与 PyTorch 相同的权重
    graph.set_node_value(
        lstm_out.w_ii,
        Some(&Tensor::new(TEST2_W_II, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_hi,
        Some(&Tensor::new(TEST2_W_HI, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(lstm_out.b_i, Some(&Tensor::zeros(&[1, hidden_size])))?;

    graph.set_node_value(
        lstm_out.w_if,
        Some(&Tensor::new(TEST2_W_IF, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_hf,
        Some(&Tensor::new(TEST2_W_HF, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(lstm_out.b_f, Some(&Tensor::ones(&[1, hidden_size])))?;

    graph.set_node_value(
        lstm_out.w_ig,
        Some(&Tensor::new(TEST2_W_IG, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_hg,
        Some(&Tensor::new(TEST2_W_HG, &[hidden_size, hidden_size])),
    )?;
    graph.set_node_value(lstm_out.b_g, Some(&Tensor::zeros(&[1, hidden_size])))?;

    graph.set_node_value(
        lstm_out.w_io,
        Some(&Tensor::new(TEST2_W_IO, &[input_size, hidden_size])),
    )?;
    graph.set_node_value(
        lstm_out.w_ho,
        Some(&Tensor::new(TEST2_W_HO, &[hidden_size, hidden_size])),
    )?;
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
    let grad_w_ii = graph.get_node_grad(lstm_out.w_ii)?.unwrap();
    let grad_w_ii_flat: Vec<f32> = grad_w_ii.data_as_slice().to_vec();
    println!("grad_w_ii: {:?}", grad_w_ii_flat);
    for (&actual, &expected) in grad_w_ii_flat.iter().zip(TEST3_GRAD_W_II.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    // 验证 w_out 梯度
    let grad_w_out = graph.get_node_grad(w_out)?.unwrap();
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
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;
    let input_size = 8;
    let hidden_size = 16;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("encoder"),
    )?;

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
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 2;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm"),
    )?;

    // 设置固定权重，确保输出不为 0
    graph.set_node_value(
        lstm_out.w_ii,
        Some(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2])),
    )?;
    graph.set_node_value(
        lstm_out.w_ig,
        Some(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2])),
    )?;
    graph.set_node_value(
        lstm_out.w_io,
        Some(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2])),
    )?;

    // 运行几步
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;
    graph.step(lstm_out.hidden)?;
    graph.step(lstm_out.hidden)?;

    // 获取当前状态（运行两步后应该有非零值）
    let h_before = graph.get_node_value(lstm_out.hidden)?.unwrap().clone();
    let c_before = graph.get_node_value(lstm_out.cell)?.unwrap().clone();
    println!(
        "Before reset: h={:?}, c={:?}",
        h_before.data_as_slice(),
        c_before.data_as_slice()
    );
    assert!(
        h_before[[0, 0]].abs() > 0.001,
        "h should be non-zero after 2 steps"
    );
    assert!(
        c_before[[0, 0]].abs() > 0.001,
        "c should be non-zero after 2 steps"
    );

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

// ==================== 默认命名测试 ====================

/// 测试 LSTM 无名称（使用默认前缀）
#[test]
fn test_lstm_without_name() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;
    let input_size = 8;
    let hidden_size = 16;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(&mut graph, input, input_size, hidden_size, batch_size, None)?;

    // 验证使用默认前缀 "lstm"
    let w_ii_name = graph.get_node(lstm_out.w_ii)?.name();
    assert!(w_ii_name.contains("lstm") && w_ii_name.contains("W_ii"));

    let h_prev_name = graph.get_node(lstm_out.h_prev)?.name();
    assert!(h_prev_name.contains("lstm") && h_prev_name.contains("h_prev"));

    Ok(())
}

// ==================== 名称冲突测试 ====================

/// 测试重复名称应该报错
#[test]
fn test_lstm_duplicate_name_error() {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;
    let input_size = 4;
    let hidden_size = 8;

    let input = graph
        .new_input_node(&[batch_size, input_size], Some("input"))
        .unwrap();

    // 第一个 LSTM 成功
    let lstm1 = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("encoder"),
    );
    assert!(lstm1.is_ok());

    // 第二个 LSTM 使用相同名称，应该失败
    let lstm2 = lstm(
        &mut graph,
        lstm1.unwrap().hidden,
        hidden_size,
        hidden_size,
        batch_size,
        Some("encoder"),
    );
    assert!(lstm2.is_err());

    // 验证错误类型
    if let Err(e) = lstm2 {
        let err_msg = format!("{:?}", e);
        assert!(
            err_msg.contains("Duplicate") || err_msg.contains("重复"),
            "错误信息应包含重复名称提示: {}",
            err_msg
        );
    }
}

/// 测试多个 LSTM 使用不同名称
#[test]
fn test_lstm_multiple_layers_different_names() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;
    let input_size = 4;
    let hidden_size = 8;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;

    let lstm1 = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm1"),
    )?;
    let lstm2 = lstm(
        &mut graph,
        lstm1.hidden,
        hidden_size,
        hidden_size,
        batch_size,
        Some("lstm2"),
    )?;
    let lstm3 = lstm(
        &mut graph,
        lstm2.hidden,
        hidden_size,
        hidden_size,
        batch_size,
        Some("lstm3"),
    )?;

    // 验证各层节点独立存在
    assert!(graph.get_node_value(lstm1.w_ii).is_ok());
    assert!(graph.get_node_value(lstm2.w_ii).is_ok());
    assert!(graph.get_node_value(lstm3.w_ii).is_ok());

    // 验证节点名称正确
    assert!(graph.get_node(lstm1.w_ii)?.name().contains("lstm1"));
    assert!(graph.get_node(lstm2.w_ii)?.name().contains("lstm2"));
    assert!(graph.get_node(lstm3.w_ii)?.name().contains("lstm3"));

    Ok(())
}

/// 测试多个无名称层会冲突（预期行为）
#[test]
fn test_lstm_multiple_unnamed_layers_conflict() {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;
    let input_size = 4;
    let hidden_size = 8;

    let input = graph
        .new_input_node(&[batch_size, input_size], Some("input"))
        .unwrap();

    // 第一个无名称层成功
    let lstm1 = lstm(&mut graph, input, input_size, hidden_size, batch_size, None);
    assert!(lstm1.is_ok());

    // 第二个无名称层应该失败（名称冲突）
    let lstm2 = lstm(
        &mut graph,
        lstm1.unwrap().hidden,
        hidden_size,
        hidden_size,
        batch_size,
        None,
    );
    assert!(lstm2.is_err(), "多个无名称 LSTM 层应该因名称冲突而失败");
}

// ==================== 链式连接测试 ====================

/// 测试多层 LSTM 链式连接
#[test]
fn test_lstm_chain() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 2;
    let input_size = 4;
    let hidden_size = 6;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;

    // 3 层堆叠 LSTM
    let lstm1 = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm1"),
    )?;
    let lstm2 = lstm(
        &mut graph,
        lstm1.hidden,
        hidden_size,
        hidden_size,
        batch_size,
        Some("lstm2"),
    )?;
    let lstm3 = lstm(
        &mut graph,
        lstm2.hidden,
        hidden_size,
        hidden_size,
        batch_size,
        Some("lstm3"),
    )?;

    // 设置输入
    let x = Tensor::normal(0.0, 1.0, &[batch_size, input_size]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.step(lstm3.hidden)?;

    // 验证输出存在且形状正确
    let output = graph.get_node_value(lstm3.hidden)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[batch_size, hidden_size]);

    // 验证 cell 状态也正确
    let cell = graph.get_node_value(lstm3.cell)?;
    assert!(cell.is_some());
    assert_eq!(cell.unwrap().shape(), &[batch_size, hidden_size]);

    println!("✅ 多层 LSTM 链式连接成功");
    Ok(())
}

// ==================== 边界维度测试 ====================

/// 测试单特征输入
#[test]
fn test_lstm_single_input_feature() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 2;
    let input_size = 1;
    let hidden_size = 4;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm"),
    )?;

    // 设置输入
    let x = Tensor::new(&[1.0, 2.0], &[batch_size, input_size]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.step(lstm_out.hidden)?;

    let output = graph.get_node_value(lstm_out.hidden)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[batch_size, hidden_size]);

    Ok(())
}

/// 测试单隐藏单元
#[test]
fn test_lstm_single_hidden_unit() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 2;
    let input_size = 4;
    let hidden_size = 1;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm"),
    )?;

    // 设置输入
    let x = Tensor::normal(0.0, 1.0, &[batch_size, input_size]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.step(lstm_out.hidden)?;

    let output = graph.get_node_value(lstm_out.hidden)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[batch_size, hidden_size]);

    Ok(())
}

/// 测试大维度 LSTM（典型 NLP 配置）
#[test]
fn test_lstm_large_dimensions() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 32;
    let input_size = 128;
    let hidden_size = 256;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm"),
    )?;

    // 验证参数形状
    let w_ii_shape = graph.get_node(lstm_out.w_ii)?.value_expected_shape();
    let w_hi_shape = graph.get_node(lstm_out.w_hi)?.value_expected_shape();

    assert_eq!(w_ii_shape, &[input_size, hidden_size]);
    assert_eq!(w_hi_shape, &[hidden_size, hidden_size]);

    Ok(())
}

// ==================== 参数访问测试 ====================

/// 测试访问 LSTM 内部参数
#[test]
fn test_lstm_access_internal_params() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 2;
    let input_size = 2;
    let hidden_size = 2;

    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm"),
    )?;

    // 应该能访问并修改输入门权重
    let custom_w_ii = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[input_size, hidden_size]);
    graph.set_node_value(lstm_out.w_ii, Some(&custom_w_ii))?;

    let w_ii = graph.get_node_value(lstm_out.w_ii)?.unwrap();
    assert_abs_diff_eq!(w_ii[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(w_ii[[1, 1]], 4.0, epsilon = 1e-6);

    // 修改遗忘门权重
    let custom_w_if = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[input_size, hidden_size]);
    graph.set_node_value(lstm_out.w_if, Some(&custom_w_if))?;

    let w_if = graph.get_node_value(lstm_out.w_if)?.unwrap();
    assert_abs_diff_eq!(w_if[[0, 0]], 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(w_if[[1, 1]], 0.4, epsilon = 1e-6);

    // 修改 bias
    let custom_b_i = Tensor::new(&[0.5, 0.5], &[1, hidden_size]);
    graph.set_node_value(lstm_out.b_i, Some(&custom_b_i))?;

    let b_i = graph.get_node_value(lstm_out.b_i)?.unwrap();
    assert_abs_diff_eq!(b_i[[0, 0]], 0.5, epsilon = 1e-6);

    Ok(())
}

// ==================== Batch 反向传播测试 ====================

/// 测试 LSTM 与 Batch 反向传播（非 BPTT）
#[test]
fn test_lstm_batch_backward() -> Result<(), GraphError> {
    use crate::nn::layer::linear;

    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;
    let input_size = 8;
    let hidden_size = 6;
    let output_size = 3;

    // 构建网络: input -> lstm -> linear -> softmax_ce
    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm"),
    )?;
    let fc = linear(
        &mut graph,
        lstm_out.hidden,
        hidden_size,
        output_size,
        batch_size,
        Some("fc"),
    )?;

    // SoftmaxCrossEntropy Loss
    let labels = graph.new_input_node(&[batch_size, output_size], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(fc.output, labels, Some("loss"))?;

    // 设置数据
    let x = Tensor::normal(0.0, 1.0, &[batch_size, input_size]);
    let y = Tensor::new(
        &[
            1.0, 0.0, 0.0, // batch 0
            0.0, 1.0, 0.0, // batch 1
            0.0, 0.0, 1.0, // batch 2
            1.0, 0.0, 0.0, // batch 3
        ],
        &[batch_size, output_size],
    );

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(labels, Some(&y))?;

    // Batch 前向 + 反向
    graph.forward(loss)?;
    graph.backward(loss)?;

    // 验证 LSTM 输入门权重有梯度
    let w_ii_grad = graph.get_node_grad_ref(lstm_out.w_ii)?;
    assert!(w_ii_grad.is_some());
    assert_eq!(w_ii_grad.unwrap().shape(), &[input_size, hidden_size]);

    // 验证隐藏层权重有梯度
    let w_hi_grad = graph.get_node_grad_ref(lstm_out.w_hi)?;
    assert!(w_hi_grad.is_some());
    assert_eq!(w_hi_grad.unwrap().shape(), &[hidden_size, hidden_size]);

    println!("✅ LSTM batch_backward 正确传播梯度");
    Ok(())
}

// ==================== Chain Batch Training 测试 ====================

/// 测试多层 LSTM 链式批量训练
#[test]
fn test_lstm_chain_batch_training() -> Result<(), GraphError> {
    use crate::nn::layer::linear;

    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;
    let input_size = 8;
    let hidden_size = 6;
    let output_size = 3;

    // 构建网络: input -> lstm1 -> lstm2 -> linear -> softmax_ce
    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm1 = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm1"),
    )?;
    let lstm2 = lstm(
        &mut graph,
        lstm1.hidden,
        hidden_size,
        hidden_size,
        batch_size,
        Some("lstm2"),
    )?;
    let fc = linear(
        &mut graph,
        lstm2.hidden,
        hidden_size,
        output_size,
        batch_size,
        Some("fc"),
    )?;

    // SoftmaxCrossEntropy Loss
    let labels = graph.new_input_node(&[batch_size, output_size], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(fc.output, labels, Some("loss"))?;

    // 设置数据
    let x = Tensor::normal(0.0, 1.0, &[batch_size, input_size]);
    let y = Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        &[batch_size, output_size],
    );

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(labels, Some(&y))?;

    // 前向
    graph.forward(loss)?;
    let loss_before = graph.get_node_value(loss)?.unwrap()[[0, 0]];

    // 反向
    graph.backward(loss)?;

    // 验证两层 LSTM 都有梯度
    let lstm1_grad = graph.get_node_grad_ref(lstm1.w_ii)?;
    assert!(lstm1_grad.is_some());

    let lstm2_grad = graph.get_node_grad_ref(lstm2.w_ii)?;
    assert!(lstm2_grad.is_some());

    println!(
        "✅ LSTM chain_batch_training: loss={:.4}, 两层 LSTM 都有梯度",
        loss_before
    );
    Ok(())
}

// ==================== 与其他层集成测试 ====================

/// 测试 LSTM 与 linear 集成
#[test]
fn test_lstm_with_linear_integration() -> Result<(), GraphError> {
    use crate::nn::layer::linear;

    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 2;
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 3;

    // LSTM -> Linear 经典序列分类结构
    let input = graph.new_input_node(&[batch_size, input_size], Some("input"))?;
    let lstm_out = lstm(
        &mut graph,
        input,
        input_size,
        hidden_size,
        batch_size,
        Some("lstm"),
    )?;
    let fc = linear(
        &mut graph,
        lstm_out.hidden,
        hidden_size,
        output_size,
        batch_size,
        Some("fc"),
    )?;

    // 设置输入
    let x = Tensor::normal(0.0, 1.0, &[batch_size, input_size]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.step(fc.output)?;

    // 验证输出形状
    let output = graph.get_node_value(fc.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[batch_size, output_size]);

    println!("✅ LSTM 与 Linear 集成正常");
    Ok(())
}
