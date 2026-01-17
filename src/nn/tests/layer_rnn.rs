/*
 * @Author       : 老董
 * @Date         : 2026-01-17
 * @Description  : Rnn Layer 单元测试（与 PyTorch 数值对照）
 *
 * 参考值来源: tests/python/layer_reference/rnn_layer_reference.py
 */

use crate::nn::layer::Rnn;
use crate::nn::{Graph, GraphError, Module, VarLossOps};
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

/// 测试 Rnn 层创建
#[test]
fn test_rnn_creation() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 16;
    let input_size = 10;
    let hidden_size = 20;

    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn1")?;

    // 验证参数存在
    assert!(rnn.w_ih().value()?.is_some());
    assert!(rnn.w_hh().value()?.is_some());
    assert!(rnn.b_h().value()?.is_some());
    assert!(rnn.hidden().value()?.is_none()); // 未计算前为 None

    Ok(())
}

/// 测试 Rnn 参数形状
#[test]
fn test_rnn_shapes() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 8;
    let input_size = 4;
    let hidden_size = 6;

    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn1")?;

    // 验证权重形状
    let w_ih = rnn.w_ih().value()?.unwrap();
    assert_eq!(w_ih.shape(), &[input_size, hidden_size]);

    let w_hh = rnn.w_hh().value()?.unwrap();
    assert_eq!(w_hh.shape(), &[hidden_size, hidden_size]);

    let b_h = rnn.b_h().value()?.unwrap();
    assert_eq!(b_h.shape(), &[1, hidden_size]);

    // 验证隐藏状态输入形状
    let h_prev = rnn.hidden_input().value()?.unwrap();
    assert_eq!(h_prev.shape(), &[batch_size, hidden_size]);

    Ok(())
}

/// 测试 Module trait
#[test]
fn test_rnn_module_trait() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let rnn = Rnn::new(&graph, 10, 20, 4, "rnn")?;

    let params = rnn.parameters();
    assert_eq!(params.len(), 3); // w_ih, w_hh, b_h

    Ok(())
}

// ==================== PyTorch 数值对照测试 ====================

/// 测试 1: 简单前向传播（与 PyTorch 对照）
#[test]
fn test_rnn_forward_pytorch_comparison() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 2;
    let input_size = 3;
    let hidden_size = 4;

    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn1")?;

    // 设置与 PyTorch 相同的权重
    rnn.w_ih().set_value(&Tensor::new(TEST1_W_IH, &[input_size, hidden_size]))?;
    rnn.w_hh().set_value(&Tensor::new(TEST1_W_HH, &[hidden_size, hidden_size]))?;
    rnn.b_h().set_value(&Tensor::new(TEST1_B_H, &[1, hidden_size]))?;

    // 前向传播（单时间步）
    let x = Tensor::new(TEST1_X, &[batch_size, input_size]);
    rnn.step(&x)?;

    // 验证隐藏状态
    let hidden = rnn.hidden().value()?.unwrap();
    assert_eq!(hidden.shape(), &[batch_size, hidden_size]);

    let hidden_data = hidden.data_as_slice();
    for (i, (&actual, &expected)) in hidden_data.iter().zip(TEST1_HIDDEN.iter()).enumerate() {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        if i < 4 {
            println!(
                "batch[0][{}]: actual={:.6}, expected={:.6}",
                i, actual, expected
            );
        }
    }

    println!("✅ 测试 1 通过：前向传播与 PyTorch 一致");
    Ok(())
}

/// 测试 2: 多时间步前向传播（与 PyTorch 对照）
#[test]
fn test_rnn_multi_step_forward_pytorch_comparison() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 3;

    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn1")?;

    // 设置权重
    rnn.w_ih().set_value(&Tensor::new(TEST2_W_IH, &[input_size, hidden_size]))?;
    rnn.w_hh().set_value(&Tensor::new(TEST2_W_HH, &[hidden_size, hidden_size]))?;
    rnn.b_h().set_value(&Tensor::zeros(&[1, hidden_size]))?;

    // 输入序列
    let sequence = vec![
        vec![1.0, 0.0], // t=0
        vec![0.0, 1.0], // t=1
        vec![1.0, 1.0], // t=2
    ];
    let expected_hidden = vec![TEST2_H_T0, TEST2_H_T1, TEST2_H_T2];

    // 执行多时间步
    for (t, (x_t, expected)) in sequence.iter().zip(expected_hidden.iter()).enumerate() {
        let x = Tensor::new(x_t, &[batch_size, input_size]);
        rnn.step(&x)?;

        let hidden = rnn.hidden().value()?.unwrap();
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
    let graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 2;

    // 创建 RNN 层
    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn1")?;

    // 设置权重
    rnn.w_ih().set_value(&Tensor::new(TEST3_W_IH, &[input_size, hidden_size]))?;
    rnn.w_hh().set_value(&Tensor::new(TEST3_W_HH, &[hidden_size, hidden_size]))?;
    rnn.b_h().set_value(&Tensor::new(TEST3_B_H, &[1, hidden_size]))?;

    // 创建输出层权重
    let w_out = graph.parameter_seeded(&[hidden_size, 1], "w_out", 42)?;
    w_out.set_value(&Tensor::new(TEST3_W_OUT, &[hidden_size, 1]))?;

    // 创建输出节点和 loss 节点（需要底层 API）
    let (output_id, loss_id, target_id) = {
        let mut g = graph.inner_mut();
        let output = g.new_mat_mul_node(rnn.hidden().node_id(), w_out.node_id(), Some("output"))?;
        let target = g.new_input_node(&[batch_size, 1], Some("target"))?;
        let loss = g.new_mse_loss_node(output, target, Some("loss"))?;
        (output, loss, target)
    };

    // 输入序列
    let sequence = vec![TEST3_SEQ_0, TEST3_SEQ_1];

    // 设置目标值
    graph.inner_mut().set_node_value(
        target_id,
        Some(&Tensor::new(&[TEST3_TARGET], &[batch_size, 1])),
    )?;

    // 前向传播整个序列
    for x_t in &sequence {
        let x = Tensor::new(*x_t, &[batch_size, input_size]);
        rnn.input().set_value(&x)?;
        graph.inner_mut().step(loss_id)?;
    }

    // 验证输出和 loss
    {
        let g = graph.inner();
        let output_val = g.get_node_value(output_id)?.unwrap();
        assert_abs_diff_eq!(output_val[[0, 0]], TEST3_OUTPUT, epsilon = 1e-5);
    }

    {
        let g = graph.inner();
        let loss_val = g.get_node_value(loss_id)?.unwrap();
        assert_abs_diff_eq!(loss_val[[0, 0]], TEST3_LOSS, epsilon = 1e-5);
    }

    // 反向传播
    graph.inner_mut().backward_through_time(
        &[rnn.w_ih().node_id(), rnn.w_hh().node_id(), rnn.b_h().node_id(), w_out.node_id()],
        loss_id,
    )?;

    // 验证梯度
    let grad_w_ih = rnn.w_ih().grad()?.unwrap();
    let grad_w_ih_flat: Vec<f32> = grad_w_ih.data_as_slice().to_vec();
    println!("grad_w_ih: {:?}", grad_w_ih_flat);
    for (&actual, &expected) in grad_w_ih_flat.iter().zip(TEST3_GRAD_W_IH.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    let grad_w_hh = rnn.w_hh().grad()?.unwrap();
    let grad_w_hh_flat: Vec<f32> = grad_w_hh.data_as_slice().to_vec();
    println!("grad_w_hh: {:?}", grad_w_hh_flat);
    for (&actual, &expected) in grad_w_hh_flat.iter().zip(TEST3_GRAD_W_HH.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    let grad_b_h = rnn.b_h().grad()?.unwrap();
    let grad_b_h_flat: Vec<f32> = grad_b_h.data_as_slice().to_vec();
    println!("grad_b_h: {:?}", grad_b_h_flat);
    for (&actual, &expected) in grad_b_h_flat.iter().zip(TEST3_GRAD_B_H.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    let grad_w_out = w_out.grad()?.unwrap();
    let grad_w_out_flat: Vec<f32> = grad_w_out.data_as_slice().to_vec();
    println!("grad_w_out: {:?}", grad_w_out_flat);
    for (&actual, &expected) in grad_w_out_flat.iter().zip(TEST3_GRAD_W_OUT.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    println!("✅ 测试 3 通过：BPTT 梯度与 PyTorch 一致");
    Ok(())
}

// ==================== reset() 测试 ====================

/// 测试 reset() 清除隐藏状态和历史快照
#[test]
fn test_rnn_reset() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 2;

    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn")?;

    // 设置权重
    rnn.w_ih().set_value(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2]))?;
    rnn.w_hh().set_value(&Tensor::new(&[0.1, 0.0, 0.0, 0.1], &[2, 2]))?;

    // 运行几步
    rnn.step(&Tensor::new(&[1.0, 1.0], &[1, 2]))?;
    rnn.step(&Tensor::new(&[1.0, 1.0], &[1, 2]))?;

    // 获取当前隐藏状态
    let h_before_reset = rnn.hidden().value()?.unwrap().clone();
    assert!(h_before_reset[[0, 0]].abs() > 0.1); // 确保不是 0

    // 完整重置（包括历史快照）
    rnn.reset();

    // 再运行一步
    rnn.step(&Tensor::new(&[1.0, 1.0], &[1, 2]))?;
    let h_after_reset = rnn.hidden().value()?.unwrap().clone();

    // 重新从头开始
    rnn.reset();
    rnn.step(&Tensor::new(&[1.0, 1.0], &[1, 2]))?;
    let h_fresh = rnn.hidden().value()?.unwrap();

    // reset 后第一步应该和全新开始的第一步相同
    assert_abs_diff_eq!(h_after_reset[[0, 0]], h_fresh[[0, 0]], epsilon = 1e-6);
    assert_abs_diff_eq!(h_after_reset[[0, 1]], h_fresh[[0, 1]], epsilon = 1e-6);

    println!("✅ reset() 正确清除隐藏状态和历史快照");
    Ok(())
}

// ==================== 边界维度测试 ====================

/// 测试单特征输入
#[test]
fn test_rnn_single_input_feature() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 2;
    let input_size = 1;
    let hidden_size = 4;

    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn")?;

    // 设置输入并前向传播
    rnn.step(&Tensor::new(&[1.0, 2.0], &[batch_size, input_size]))?;

    let output = rnn.hidden().value()?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[batch_size, hidden_size]);

    Ok(())
}

/// 测试单隐藏单元
#[test]
fn test_rnn_single_hidden_unit() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 2;
    let input_size = 4;
    let hidden_size = 1;

    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn")?;

    // 设置输入并前向传播
    rnn.step(&Tensor::normal(0.0, 1.0, &[batch_size, input_size]))?;

    let output = rnn.hidden().value()?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[batch_size, hidden_size]);

    Ok(())
}

/// 测试大维度 RNN（典型 NLP 配置）
#[test]
fn test_rnn_large_dimensions() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 32;
    let input_size = 128;
    let hidden_size = 256;

    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn")?;

    // 验证参数形状
    let w_ih = rnn.w_ih().value()?.unwrap();
    let w_hh = rnn.w_hh().value()?.unwrap();

    assert_eq!(w_ih.shape(), &[input_size, hidden_size]);
    assert_eq!(w_hh.shape(), &[hidden_size, hidden_size]);

    Ok(())
}

// ==================== 参数访问测试 ====================

/// 测试访问 Rnn 内部参数
#[test]
fn test_rnn_access_internal_params() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 2;
    let input_size = 2;
    let hidden_size = 2;

    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn")?;

    // 应该能访问并修改权重
    let custom_w_ih = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[input_size, hidden_size]);
    rnn.w_ih().set_value(&custom_w_ih)?;

    let w_ih = rnn.w_ih().value()?.unwrap();
    assert_abs_diff_eq!(w_ih[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(w_ih[[1, 1]], 4.0, epsilon = 1e-6);

    // 修改 W_hh
    let custom_w_hh = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[hidden_size, hidden_size]);
    rnn.w_hh().set_value(&custom_w_hh)?;

    let w_hh = rnn.w_hh().value()?.unwrap();
    assert_abs_diff_eq!(w_hh[[0, 0]], 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(w_hh[[1, 1]], 0.4, epsilon = 1e-6);

    // 修改 bias
    let custom_b = Tensor::new(&[0.5, 0.5], &[1, hidden_size]);
    rnn.b_h().set_value(&custom_b)?;

    let b_h = rnn.b_h().value()?.unwrap();
    assert_abs_diff_eq!(b_h[[0, 0]], 0.5, epsilon = 1e-6);

    Ok(())
}

// ==================== 与 Linear 集成测试 ====================

/// 测试 Rnn 与 Linear 集成
#[test]
fn test_rnn_with_linear_integration() -> Result<(), GraphError> {
    use crate::nn::layer::Linear;

    let graph = Graph::new_with_seed(42);
    let batch_size = 2;
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 3;

    // RNN -> Linear 经典序列分类结构
    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn")?;
    let fc = Linear::new(&graph, hidden_size, output_size, true, "fc")?;

    // 设置输入并前向传播
    let x = Tensor::normal(0.0, 1.0, &[batch_size, input_size]);
    rnn.step(&x)?;

    // RNN 输出 -> Linear
    let fc_out = fc.forward(rnn.hidden());
    fc_out.forward()?;

    // 验证输出形状
    let output = fc_out.value()?.unwrap();
    assert_eq!(output.shape(), &[batch_size, output_size]);

    println!("✅ Rnn 与 Linear 集成正常");
    Ok(())
}

/// 测试 Rnn + Linear + Loss 完整训练流程
#[test]
fn test_rnn_complete_training() -> Result<(), GraphError> {
    use crate::nn::layer::Linear;

    let graph = Graph::new_with_seed(42);
    let batch_size = 4;
    let input_size = 8;
    let hidden_size = 6;
    let output_size = 3;

    // 构建网络
    let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn")?;
    let fc = Linear::new(&graph, hidden_size, output_size, true, "fc")?;

    // 设置输入
    let x = Tensor::normal(0.0, 1.0, &[batch_size, input_size]);
    rnn.step(&x)?;

    // RNN -> Linear -> Loss
    let fc_out = fc.forward(rnn.hidden());
    let labels = graph.input(&Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        &[batch_size, output_size],
    ))?;
    let loss = fc_out.cross_entropy(&labels)?;

    // 前向 + 反向
    loss.forward()?;
    let loss_val = loss.value()?.unwrap()[[0, 0]];
    loss.backward()?;

    // 验证所有参数都有梯度
    assert!(rnn.w_ih().grad()?.is_some());
    assert!(rnn.w_hh().grad()?.is_some());
    assert!(rnn.b_h().grad()?.is_some());
    assert!(fc.weights().grad()?.is_some());

    println!("✅ Rnn 完整训练流程: loss={:.4}, 所有参数都有梯度", loss_val);
    Ok(())
}
