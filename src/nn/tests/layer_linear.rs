/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Linear layer 单元测试（PyTorch 风格 API）
 *
 * 参考值来源: tests/python/layer_reference/linear_layer_reference.py
 */

use crate::nn::graph::Graph;
use crate::nn::layer::Linear;
use crate::nn::{GraphError, Module, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== PyTorch 参考常量 ====================

// 测试: 简单前向传播 (batch=2, in=3, out=4)
const PYTORCH_FWD_X: &[f32] = &[1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
const PYTORCH_FWD_W: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
const PYTORCH_FWD_B: &[f32] = &[0.1, 0.2, 0.3, 0.4];
const PYTORCH_FWD_OUTPUT: &[f32] = &[3.9, 4.6, 5.3, 6.0, 3.15, 3.7, 4.25, 4.8];

// 测试: 反向传播梯度 (batch=2, in=3, out=2) + MSE Loss
const PYTORCH_BWD_X: &[f32] = &[1.0, 2.0, 3.0, 0.5, 1.0, 1.5];
const PYTORCH_BWD_W: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
const PYTORCH_BWD_B: &[f32] = &[0.1, 0.2];
const PYTORCH_BWD_TARGET: &[f32] = &[1.0, 0.5, 0.5, 1.0];
const PYTORCH_BWD_OUTPUT: &[f32] = &[2.3, 3.0, 1.2, 1.6];
const PYTORCH_BWD_LOSS: f32 = 2.1975;
const PYTORCH_BWD_GRAD_W: &[f32] = &[0.825, 1.4, 1.65, 2.8, 2.475, 4.2];
const PYTORCH_BWD_GRAD_B: &[f32] = &[1.0, 1.55];

// 测试: 两层 Linear + ReLU + SoftmaxCrossEntropy
const PYTORCH_CHAIN_X: &[f32] = &[1.0, 0.5, -0.5, 0.2, 0.3, -0.2, 0.8, -0.1];
const PYTORCH_CHAIN_W1: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.2, 0.3, 0.4];
const PYTORCH_CHAIN_B1: &[f32] = &[0.1, 0.1, 0.1];
const PYTORCH_CHAIN_W2: &[f32] = &[0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const PYTORCH_CHAIN_B2: &[f32] = &[0.1, 0.2];
const PYTORCH_CHAIN_TARGET: &[f32] = &[1.0, 0.0, 0.0, 1.0];
const PYTORCH_CHAIN_H1_RELU: &[f32] = &[0.09, 0.21, 0.33, 0.59, 0.67, 0.75];
const PYTORCH_CHAIN_LOGITS: &[f32] = &[0.589, 0.752, 1.539, 1.84];
const PYTORCH_CHAIN_LOSS: f32 = 0.6659472;
const PYTORCH_CHAIN_GRAD_W1: &[f32] = &[
    0.02065329,
    0.02065331,
    0.02065329,
    0.01776964,
    0.01776965,
    0.01776964,
    -0.03052906,
    -0.03052907,
    -0.03052907,
    0.00753317,
    0.00753317,
    0.00753317,
];
const PYTORCH_CHAIN_GRAD_B1: &[f32] = &[0.00576731, 0.00576732, 0.00576729];
const PYTORCH_CHAIN_GRAD_W2: &[f32] = &[
    0.10113763,
    -0.10113766,
    0.08571057,
    -0.0857106,
    0.07028344,
    -0.07028348,
];
const PYTORCH_CHAIN_GRAD_B2: &[f32] = &[-0.0576735, 0.05767344];

// ==================== 基础功能测试 ====================

/// 测试 Linear 结构体前向传播
#[test]
fn test_linear_forward() {
    let graph = Graph::new();

    // 创建 Linear 层：3 -> 2
    let fc = Linear::new(&graph, 3, 2, true, "fc").unwrap();

    // 创建输入：[2, 3] (batch=2, in_features=3)
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    // 前向传播
    let y = fc.forward(&x);
    y.forward().unwrap();

    // 验证输出形状
    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]);
}

/// 测试 Linear 无 bias
#[test]
fn test_linear_no_bias() {
    let graph = Graph::new();

    // 创建无 bias 的 Linear 层
    let fc = Linear::new(&graph, 3, 2, false, "fc_no_bias").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();

    let y = fc.forward(&x);
    y.forward().unwrap();

    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[1, 2]);
}

/// 测试 Linear 参数数量
#[test]
fn test_linear_parameters() {
    let graph = Graph::new();

    // 有 bias 的 Linear
    let fc_with_bias = Linear::new(&graph, 4, 3, true, "fc1").unwrap();
    assert_eq!(fc_with_bias.parameters().len(), 2);
    assert_eq!(fc_with_bias.num_params(), 2);

    // 无 bias 的 Linear
    let fc_no_bias = Linear::new(&graph, 4, 3, false, "fc2").unwrap();
    assert_eq!(fc_no_bias.parameters().len(), 1);
    assert_eq!(fc_no_bias.num_params(), 1);
}

/// 测试 Linear 链式调用激活函数
#[test]
fn test_linear_chained_with_activation() {
    let graph = Graph::new();

    let fc = Linear::new(&graph, 3, 2, true, "fc").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();

    // 链式调用：fc.forward(x).relu()
    let y = fc.forward(&x).relu();
    y.forward().unwrap();

    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[1, 2]);
    // ReLU 后所有值应该 >= 0
    assert!(result.data_as_slice().iter().all(|&v| v >= 0.0));
}

/// 测试 Linear 反向传播
#[test]
fn test_linear_backward() {
    let graph = Graph::new();

    let fc = Linear::new(&graph, 3, 2, true, "fc").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();

    let y = fc.forward(&x);
    let loss = y.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 权重和偏置都应该有梯度
    let params = fc.parameters();
    for p in params {
        let grad = p.grad().unwrap();
        assert!(grad.is_some(), "参数应该有梯度");
    }
}

/// 测试两层 MLP
#[test]
fn test_linear_mlp_two_layers() {
    let graph = Graph::new();

    // 简单 MLP：3 -> 4 -> 2
    let fc1 = Linear::new(&graph, 3, 4, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 4, 2, true, "fc2").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();

    // 前向：fc1 -> relu -> fc2
    let h = fc1.forward(&x).relu();
    let y = fc2.forward(&h);
    let loss = y.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 两层的参数都应该有梯度
    for p in fc1.parameters() {
        assert!(p.grad().unwrap().is_some());
    }
    for p in fc2.parameters() {
        assert!(p.grad().unwrap().is_some());
    }
}

/// 测试带种子 Graph 的 Linear 可重复性
#[test]
fn test_linear_seeded_reproducibility() {
    let seed = 12345u64;

    // 创建两个使用相同 seed 的 Graph，Linear 层会继承种子
    let graph1 = Graph::new_with_seed(seed);
    let fc1 = Linear::new(&graph1, 4, 3, true, "fc").unwrap();

    let graph2 = Graph::new_with_seed(seed);
    let fc2 = Linear::new(&graph2, 4, 3, true, "fc").unwrap();

    // 权重应该完全相同
    let w1 = fc1.weights().value().unwrap().unwrap();
    let w2 = fc2.weights().value().unwrap().unwrap();
    assert_eq!(w1.data_as_slice(), w2.data_as_slice());

    // bias 都是零初始化，也应该相同
    let b1 = fc1.bias().unwrap().value().unwrap().unwrap();
    let b2 = fc2.bias().unwrap().value().unwrap().unwrap();
    assert_eq!(b1.data_as_slice(), b2.data_as_slice());
}

/// 测试不同 seed 产生不同权重
#[test]
fn test_linear_different_seeds() {
    let graph1 = Graph::new_with_seed(111);
    let fc1 = Linear::new(&graph1, 4, 3, true, "fc").unwrap();

    let graph2 = Graph::new_with_seed(222);
    let fc2 = Linear::new(&graph2, 4, 3, true, "fc").unwrap();

    // 不同 seed 应该产生不同权重
    let w1 = fc1.weights().value().unwrap().unwrap();
    let w2 = fc2.weights().value().unwrap().unwrap();
    assert_ne!(w1.data_as_slice(), w2.data_as_slice());
}

// ==================== PyTorch 数值对照测试 ====================

/// 测试前向传播数值（与 PyTorch 对照）
#[test]
fn test_linear_forward_pytorch_comparison() -> Result<(), GraphError> {
    let graph = Graph::new();
    let batch_size = 2;
    let in_features = 3;
    let out_features = 4;

    // 创建 Linear 层
    let fc = Linear::new(&graph, in_features, out_features, true, "fc")?;

    // 设置与 PyTorch 相同的参数
    fc.weights()
        .set_value(&Tensor::new(PYTORCH_FWD_W, &[in_features, out_features]))?;
    fc.bias()
        .unwrap()
        .set_value(&Tensor::new(PYTORCH_FWD_B, &[1, out_features]))?;

    // 创建输入
    let x = graph.input(&Tensor::new(PYTORCH_FWD_X, &[batch_size, in_features]))?;

    // 前向传播
    let y = fc.forward(&x);
    y.forward()?;

    // 验证输出
    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[batch_size, out_features]);

    let output_data = output.data_as_slice();
    for (i, (&actual, &expected)) in output_data
        .iter()
        .zip(PYTORCH_FWD_OUTPUT.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
        println!(
            "output[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    println!("✅ Linear 前向传播与 PyTorch 一致");
    Ok(())
}

/// 测试反向传播梯度数值（与 PyTorch 对照）
#[test]
fn test_linear_backward_pytorch_comparison() -> Result<(), GraphError> {
    let graph = Graph::new();
    let batch_size = 2;
    let in_features = 3;
    let out_features = 2;

    // 创建 Linear 层
    let fc = Linear::new(&graph, in_features, out_features, true, "fc")?;

    // 设置与 PyTorch 相同的参数
    fc.weights()
        .set_value(&Tensor::new(PYTORCH_BWD_W, &[in_features, out_features]))?;
    fc.bias()
        .unwrap()
        .set_value(&Tensor::new(PYTORCH_BWD_B, &[1, out_features]))?;

    // 创建输入和目标
    let x = graph.input(&Tensor::new(PYTORCH_BWD_X, &[batch_size, in_features]))?;
    let target = graph.input(&Tensor::new(
        PYTORCH_BWD_TARGET,
        &[batch_size, out_features],
    ))?;

    // 前向传播
    let y = fc.forward(&x);
    let loss = y.mse_loss(&target)?;
    loss.forward()?;

    // 验证输出
    let output = y.value()?.unwrap();
    let output_data = output.data_as_slice();
    println!("输出:");
    for (i, (&actual, &expected)) in output_data
        .iter()
        .zip(PYTORCH_BWD_OUTPUT.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
        println!(
            "  output[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    // 验证 loss
    let loss_val = loss.value()?.unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], PYTORCH_BWD_LOSS, epsilon = 1e-3);
    println!(
        "loss: actual={:.6}, expected={:.6}",
        loss_val[[0, 0]],
        PYTORCH_BWD_LOSS
    );

    // 反向传播
    loss.backward()?;

    // 验证权重梯度
    let grad_w = fc.weights().grad()?.unwrap();
    let grad_w_data = grad_w.data_as_slice();
    println!("\n权重梯度:");
    for (i, (&actual, &expected)) in grad_w_data
        .iter()
        .zip(PYTORCH_BWD_GRAD_W.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_w[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    // 验证偏置梯度
    let grad_b = fc.bias().unwrap().grad()?.unwrap();
    let grad_b_data = grad_b.data_as_slice();
    println!("\n偏置梯度:");
    for (i, (&actual, &expected)) in grad_b_data
        .iter()
        .zip(PYTORCH_BWD_GRAD_B.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_b[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    println!("\n✅ Linear 反向传播梯度与 PyTorch 一致");
    Ok(())
}

/// 测试两层网络反向传播（与 PyTorch 对照）
#[test]
fn test_linear_chain_backward_pytorch_comparison() -> Result<(), GraphError> {
    let graph = Graph::new();
    let batch_size = 2;
    let in_features = 4;
    let hidden_features = 3;
    let out_features = 2;

    // 创建两层 Linear
    let fc1 = Linear::new(&graph, in_features, hidden_features, true, "fc1")?;
    let fc2 = Linear::new(&graph, hidden_features, out_features, true, "fc2")?;

    // 设置与 PyTorch 相同的参数
    fc1.weights().set_value(&Tensor::new(
        PYTORCH_CHAIN_W1,
        &[in_features, hidden_features],
    ))?;
    fc1.bias()
        .unwrap()
        .set_value(&Tensor::new(PYTORCH_CHAIN_B1, &[1, hidden_features]))?;
    fc2.weights().set_value(&Tensor::new(
        PYTORCH_CHAIN_W2,
        &[hidden_features, out_features],
    ))?;
    fc2.bias()
        .unwrap()
        .set_value(&Tensor::new(PYTORCH_CHAIN_B2, &[1, out_features]))?;

    // 创建输入和目标
    let x = graph.input(&Tensor::new(PYTORCH_CHAIN_X, &[batch_size, in_features]))?;
    let target = graph.input(&Tensor::new(
        PYTORCH_CHAIN_TARGET,
        &[batch_size, out_features],
    ))?;

    // 前向传播：fc1 -> relu -> fc2 -> softmax_cross_entropy
    let h1 = fc1.forward(&x).relu();
    let logits = fc2.forward(&h1);
    let loss = logits.cross_entropy(&target)?;
    loss.forward()?;

    // 验证隐藏层 (ReLU 输出)
    let h1_val = h1.value()?.unwrap();
    let h1_data = h1_val.data_as_slice();
    println!("隐藏层 ReLU 输出:");
    for (i, (&actual, &expected)) in h1_data.iter().zip(PYTORCH_CHAIN_H1_RELU.iter()).enumerate() {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
        println!(
            "  h1[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    // 验证 logits
    let logits_val = logits.value()?.unwrap();
    let logits_data = logits_val.data_as_slice();
    println!("\nLogits:");
    for (i, (&actual, &expected)) in logits_data
        .iter()
        .zip(PYTORCH_CHAIN_LOGITS.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  logits[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    // 验证 loss
    let loss_val = loss.value()?.unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], PYTORCH_CHAIN_LOSS, epsilon = 1e-3);
    println!(
        "\nloss: actual={:.6}, expected={:.6}",
        loss_val[[0, 0]],
        PYTORCH_CHAIN_LOSS
    );

    // 反向传播
    loss.backward()?;

    // 验证 fc1 权重梯度
    let grad_w1 = fc1.weights().grad()?.unwrap();
    let grad_w1_data = grad_w1.data_as_slice();
    println!("\nfc1 权重梯度:");
    for (i, (&actual, &expected)) in grad_w1_data
        .iter()
        .zip(PYTORCH_CHAIN_GRAD_W1.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_w1[{}]: actual={:.8}, expected={:.8}",
            i, actual, expected
        );
    }

    // 验证 fc1 偏置梯度
    let grad_b1 = fc1.bias().unwrap().grad()?.unwrap();
    let grad_b1_data = grad_b1.data_as_slice();
    println!("\nfc1 偏置梯度:");
    for (i, (&actual, &expected)) in grad_b1_data
        .iter()
        .zip(PYTORCH_CHAIN_GRAD_B1.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_b1[{}]: actual={:.8}, expected={:.8}",
            i, actual, expected
        );
    }

    // 验证 fc2 权重梯度
    let grad_w2 = fc2.weights().grad()?.unwrap();
    let grad_w2_data = grad_w2.data_as_slice();
    println!("\nfc2 权重梯度:");
    for (i, (&actual, &expected)) in grad_w2_data
        .iter()
        .zip(PYTORCH_CHAIN_GRAD_W2.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_w2[{}]: actual={:.8}, expected={:.8}",
            i, actual, expected
        );
    }

    // 验证 fc2 偏置梯度
    let grad_b2 = fc2.bias().unwrap().grad()?.unwrap();
    let grad_b2_data = grad_b2.data_as_slice();
    println!("\nfc2 偏置梯度:");
    for (i, (&actual, &expected)) in grad_b2_data
        .iter()
        .zip(PYTORCH_CHAIN_GRAD_B2.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_b2[{}]: actual={:.8}, expected={:.8}",
            i, actual, expected
        );
    }

    println!("\n✅ 两层 Linear 网络反向传播与 PyTorch 一致");
    Ok(())
}
