//! 奇偶性检测训练示例（固定长度输入）
//!
//! 验证 RNN 层 + BPTT 在序列任务上的能力。
//!
//! ## 数据格式
//! - 输入：`[batch, seq_len, input_size]`（batch_first=True）
//! - 标签：`[batch, 2]`（one-hot：偶数=[1,0]，奇数=[0,1]）
//!
//! ## 运行方式
//! ```bash
//! cargo run --example parity_fixed_len
//! ```
//!
//! ## PyTorch 对照实现
//! 参见 `tests/parity_fixed_len_pytorch.py`

mod model;

use model::{generate_parity_data, ParityRNN};
use only_torch::data::{DataLoader, TensorDataset};
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;

fn main() -> Result<(), GraphError> {
    println!("=== 奇偶性检测 (Parity Detection) ===\n");

    // ========== 超参数 ==========
    let seed = 42u64;
    let seq_len = 8;
    let hidden_size = 16;
    let batch_size = 32;
    let train_samples = 1000;
    let test_samples = 200;
    let max_epochs = 150;
    let lr = 0.01;
    let target_accuracy = 95.0;

    println!("超参数:");
    println!("  序列长度: {seq_len}");
    println!("  隐藏层大小: {hidden_size}");
    println!("  批大小: {batch_size}");
    println!("  学习率: {lr}");
    println!("  优化器: Adam");
    println!("  损失函数: CrossEntropy");
    println!();

    // ========== 数据准备（使用 DataLoader！）==========
    let (train_x, train_y) = generate_parity_data(train_samples, seq_len, seed);
    let (test_x, test_y) = generate_parity_data(test_samples, seq_len, seed + 1000);

    // 创建 DataLoader（PyTorch 风格）
    let train_dataset = TensorDataset::new(train_x, train_y.clone());
    let train_loader = DataLoader::new(train_dataset, batch_size).drop_last(true);

    let test_dataset = TensorDataset::new(test_x, test_y.clone());
    let test_loader = DataLoader::new(test_dataset, batch_size).drop_last(true);

    println!(
        "数据集: 训练 {} 样本, 测试 {} 样本",
        train_samples, test_samples
    );

    let train_odd = (0..train_samples)
        .filter(|&i| train_y[[i, 1]] > 0.5)
        .count();
    let test_odd = (0..test_samples)
        .filter(|&i| test_y[[i, 1]] > 0.5)
        .count();
    println!(
        "标签分布: 训练 {}/{} 奇数, 测试 {}/{} 奇数\n",
        train_odd, train_samples, test_odd, test_samples
    );

    // ========== 模型构建 ==========
    let graph = Graph::new_with_seed(seed);
    let model = ParityRNN::new(&graph, hidden_size, batch_size)?;

    // 标签输入节点
    let labels_node = graph.input(&Tensor::zeros(&[batch_size, 2]))?;

    // 构建 CrossEntropy loss
    let loss = model.output().cross_entropy(&labels_node)?;

    // 创建 Adam 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), lr);

    // ========== 训练循环（完全 PyTorch 风格！）==========
    let mut best_accuracy = 0.0f32;
    println!("开始训练...\n");

    for epoch in 0..max_epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        // 使用 DataLoader 迭代！
        for (x_batch, y_batch) in train_loader.iter() {
            // 1. 设置标签
            labels_node.set_value(&y_batch)?;

            // 2. 前向传播
            model.forward(&x_batch)?;

            // 3. 获取 loss 值
            epoch_loss += loss.value()?.unwrap()[[0, 0]];
            num_batches += 1;

            // 4. 反向传播 + 参数更新
            loss.backward()?;
            optimizer.step()?;
        }

        // 每 10 个 epoch 评估
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            let avg_loss = epoch_loss / num_batches as f32;
            let accuracy = evaluate(&model, &graph, &test_loader)?;
            best_accuracy = best_accuracy.max(accuracy);

            println!(
                "Epoch {:3}/{}: loss={:.4}, test_acc={:.1}%",
                epoch + 1, max_epochs, avg_loss, accuracy
            );

            if accuracy >= target_accuracy {
                println!("\n✅ 达到目标准确率 {target_accuracy}%，提前停止训练");
                break;
            }
        }
    }

    // ========== 最终评估 ==========
    let final_accuracy = evaluate(&model, &graph, &test_loader)?;
    println!("\n========== 最终结果 ==========");
    println!("测试准确率: {:.1}%", final_accuracy);
    println!("最佳准确率: {:.1}%", best_accuracy);

    if final_accuracy >= target_accuracy {
        println!("✅ 奇偶性检测任务成功！");
        Ok(())
    } else {
        Err(GraphError::ComputationError(format!(
            "准确率 {:.1}% 未达到目标 {target_accuracy}%",
            final_accuracy
        )))
    }
}

/// 评估模型准确率
fn evaluate(model: &ParityRNN, graph: &Graph, test_loader: &DataLoader) -> Result<f32, GraphError> {
    graph.eval();

    let mut correct = 0;
    let mut total = 0;
    let output_id = model.output().node_id();

    for (x_batch, y_batch) in test_loader.iter() {
        // 前向传播
        model.forward(&x_batch)?;

        // 计算准确率
        let logits = graph.inner().get_node_value(output_id)?.unwrap().clone();
        let batch_size = x_batch.shape()[0];

        for i in 0..batch_size {
            let pred_class = if logits[[i, 0]] > logits[[i, 1]] { 0 } else { 1 };
            let true_class = if y_batch[[i, 0]] > 0.5 { 0 } else { 1 };
            if pred_class == true_class {
                correct += 1;
            }
            total += 1;
        }
    }

    graph.train();

    Ok(100.0 * correct as f32 / total as f32)
}
