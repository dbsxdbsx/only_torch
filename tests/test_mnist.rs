/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : MNIST 手写数字识别 MVP 集成测试
 *                 验证：数据加载 + 网络构建 + 训练循环 的基本逻辑正确性
 * @LastEditors  : 老董
 * @LastEditTime : 2025-12-21
 */

use only_torch::data::MnistDataset;
use only_torch::nn::layer::linear;
use only_torch::nn::optimizer::{Optimizer, SGD};
use only_torch::nn::{Graph, GraphError};
use only_torch::tensor_slice;

/// MNIST MVP 集成测试
///
/// 验证 loss 确实在下降（不是随机波动）
/// - 200 `样本，batch_size=20，共` 10 个 batch
/// - 使用新统一 API：forward + backward + step
/// - 验证：后半段平均 loss < 前半段平均 loss
#[test]
fn test_mnist_mlp() -> Result<(), GraphError> {
    println!("=== MNIST MVP 集成测试 ===\n");

    // 1. 加载数据
    let train_data = MnistDataset::train()
        .expect("加载 MNIST 训练集失败")
        .flatten();

    // 2. 构建 MLP：784 -> 64 (Sigmoid) -> 10 (SoftmaxCrossEntropy)
    let batch_size = 20;
    let mut graph = Graph::new_with_seed(42);
    let x = graph.new_input_node(&[batch_size, 784], Some("x"))?;
    let y = graph.new_input_node(&[batch_size, 10], Some("y"))?;

    // 第一层：784 -> 64 + Sigmoid
    let fc1 = linear(&mut graph, x, 784, 64, batch_size, Some("fc1"))?;
    let a1 = graph.new_sigmoid_node(fc1.output, Some("a1"))?;

    // 第二层：64 -> 10 + SoftmaxCrossEntropy
    let fc2 = linear(&mut graph, a1, 64, 10, batch_size, Some("fc2"))?;
    let loss = graph.new_softmax_cross_entropy_node(fc2.output, y, Some("loss"))?;

    // 3. 训练配置
    let mut optimizer = SGD::new(&graph, 0.5)?;
    let train_samples = 200; // 10 个 batch
    let num_batches = train_samples / batch_size;

    // 获取所有训练数据（labels 已经是 one-hot 编码）
    let all_images = train_data.images();
    let all_labels = train_data.labels();

    // 记录每个 batch 的 loss
    let mut batch_losses: Vec<f32> = Vec::new();

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = start + batch_size;

        let batch_images = tensor_slice!(all_images, start..end, ..);
        let batch_labels = tensor_slice!(all_labels, start..end, ..);

        graph.set_node_value(x, Some(&batch_images))?;
        graph.set_node_value(y, Some(&batch_labels))?;

        graph.zero_grad()?;
        graph.forward(loss)?;
        let loss_val = graph.backward(loss)?; // backward 返回 loss 值
        optimizer.step(&mut graph)?;

        batch_losses.push(loss_val);
    }

    // 4. 打印每个 batch 的 loss
    println!("各 batch 平均 loss:");
    for (i, loss) in batch_losses.iter().enumerate() {
        println!("  Batch {}: {:.4}", i + 1, loss);
    }

    // 5. 验证：后半段 loss < 前半段 loss
    let mid = num_batches / 2;
    let first_half_avg: f32 = batch_losses[..mid].iter().sum::<f32>() / mid as f32;
    let second_half_avg: f32 = batch_losses[mid..].iter().sum::<f32>() / (num_batches - mid) as f32;

    println!("\n前半段平均 loss: {first_half_avg:.4}");
    println!("后半段平均 loss: {second_half_avg:.4}");

    if second_half_avg < first_half_avg {
        println!("\n✅ MNIST MVP 测试通过！loss 趋势下降");
        Ok(())
    } else {
        Err(GraphError::ComputationError(format!(
            "loss 未下降：前半段 {first_half_avg:.4} -> 后半段 {second_half_avg:.4}"
        )))
    }
}
