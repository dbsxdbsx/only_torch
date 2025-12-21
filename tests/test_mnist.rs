/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : MNIST 手写数字识别 MVP 集成测试
 *                 验证：数据加载 + 网络构建 + 训练循环 的基本逻辑正确性
 * @LastEditors  : 老董
 * @LastEditTime : 2025-12-21
 */

use only_torch::data::MnistDataset;
use only_torch::nn::optimizer::{Optimizer, SGD};
use only_torch::nn::{Graph, GraphError};

/// MNIST MVP 集成测试
///
/// 验证 loss 确实在下降（不是随机波动）
/// - 200 样本，batch_size=20，共 10 个 batch
/// - 验证：后半段平均 loss < 前半段平均 loss
#[test]
fn test_mnist_mlp() -> Result<(), GraphError> {
    println!("=== MNIST MVP 集成测试 ===\n");

    // 1. 加载数据
    let train_data = MnistDataset::train()
        .expect("加载 MNIST 训练集失败")
        .flatten();

    // 2. 构建 MLP：784 -> 64 (Sigmoid) -> 10 (SoftmaxCrossEntropy)
    let mut graph = Graph::new();
    let x = graph.new_input_node(&[1, 784], Some("x"))?;
    let y = graph.new_input_node(&[1, 10], Some("y"))?;

    let w1 = graph.new_parameter_node_seeded(&[784, 64], Some("w1"), 42)?;
    let b1 = graph.new_parameter_node_seeded(&[1, 64], Some("b1"), 43)?;
    let z1 = graph.new_mat_mul_node(x, w1, None)?;
    let h1 = graph.new_add_node(&[z1, b1], None)?;
    let a1 = graph.new_sigmoid_node(h1, None)?;

    let w2 = graph.new_parameter_node_seeded(&[64, 10], Some("w2"), 44)?;
    let b2 = graph.new_parameter_node_seeded(&[1, 10], Some("b2"), 45)?;
    let z2 = graph.new_mat_mul_node(a1, w2, None)?;
    let logits = graph.new_add_node(&[z2, b2], None)?;
    let loss = graph.new_softmax_cross_entropy_node(logits, y, Some("loss"))?;

    // 3. 训练配置
    let mut optimizer = SGD::new(&graph, 0.5)?;
    let batch_size = 20;
    let train_samples = 200; // 10 个 batch
    let num_batches = train_samples / batch_size;

    // 记录每个 batch 结束后的平均 loss
    let mut batch_losses: Vec<f32> = Vec::new();
    let mut batch_loss_sum = 0.0;
    let mut batch_count = 0;

    for i in 0..train_samples {
        let (image, label) = train_data.get(i).expect("获取样本失败");
        graph.set_node_value(x, Some(&image.reshape(&[1, 784])))?;
        graph.set_node_value(y, Some(&label.reshape(&[1, 10])))?;

        optimizer.one_step(&mut graph, loss)?;

        let loss_val = graph.get_node_value(loss)?.unwrap()[[0, 0]];
        batch_loss_sum += loss_val;
        batch_count += 1;

        if batch_count >= batch_size {
            let avg_loss = batch_loss_sum / batch_size as f32;
            batch_losses.push(avg_loss);
            optimizer.update(&mut graph)?;
            batch_loss_sum = 0.0;
            batch_count = 0;
        }
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

    println!("\n前半段平均 loss: {:.4}", first_half_avg);
    println!("后半段平均 loss: {:.4}", second_half_avg);

    if second_half_avg < first_half_avg {
        println!("\n✅ MNIST MVP 测试通过！loss 趋势下降");
        Ok(())
    } else {
        Err(GraphError::ComputationError(format!(
            "loss 未下降：前半段 {:.4} -> 后半段 {:.4}",
            first_half_avg, second_half_avg
        )))
    }
}
