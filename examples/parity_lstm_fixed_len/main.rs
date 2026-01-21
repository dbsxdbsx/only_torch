//! 固定长度奇偶性检测训练示例（LSTM，PyTorch 风格）
//!
//! 验证 LSTM 展开式设计在序列任务上的能力。
//!
//! ## 运行方式
//! ```bash
//! cargo run --example parity_lstm_fixed_len
//! ```

mod model;

use model::ParityLSTM;
use only_torch::data::{DataLoader, TensorDataset};
use only_torch::nn::{Adam, CrossEntropyLoss, Graph, GraphError, Module, Optimizer};
use only_torch::tensor::Tensor;

fn main() -> Result<(), GraphError> {
    println!("=== 奇偶性检测（LSTM，PyTorch 风格）===\n");

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
    println!("  模型: LSTM");
    println!();

    // ========== 数据准备 ==========
    let (train_x, train_y) = generate_parity_data(train_samples, seq_len, seed);
    let (test_x, test_y) = generate_parity_data(test_samples, seq_len, seed + 1000);

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
    let model = ParityLSTM::new(&graph, hidden_size)?;
    let criterion = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(&graph, &model.parameters(), lr);

    // ========== 训练循环 ==========
    let mut best_accuracy = 0.0f32;
    println!("开始训练...\n");

    for epoch in 0..max_epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        for (x_batch, y_batch) in train_loader.iter() {
            optimizer.zero_grad()?;
            let output = model.forward(&x_batch)?;
            let loss = criterion.forward(&output, &y_batch)?;
            let loss_val = loss.backward()?;
            optimizer.step()?;

            epoch_loss += loss_val;
            num_batches += 1;
        }

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
        println!("✅ LSTM 奇偶性检测任务成功！");
        Ok(())
    } else {
        Err(GraphError::ComputationError(format!(
            "准确率 {:.1}% 未达到目标 {target_accuracy}%",
            final_accuracy
        )))
    }
}

fn evaluate(model: &ParityLSTM, graph: &Graph, test_loader: &DataLoader) -> Result<f32, GraphError> {
    graph.eval();

    let mut correct = 0;
    let mut total = 0;

    for (x_batch, y_batch) in test_loader.iter() {
        let output = model.forward(&x_batch)?;
        let logits = output.value()?.unwrap();

        let pred = logits.argmax(1);
        let true_labels = y_batch.argmax(1);

        let batch_size = x_batch.shape()[0];
        correct += (0..batch_size)
            .filter(|&i| pred[[i]] == true_labels[[i]])
            .count();
        total += batch_size;
    }

    graph.train();
    Ok(100.0 * correct as f32 / total as f32)
}

fn generate_parity_data(num_samples: usize, seq_len: usize, seed: u64) -> (Tensor, Tensor) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut seq_data = Vec::with_capacity(num_samples * seq_len);
    let mut label_data = Vec::with_capacity(num_samples * 2);

    for i in 0..num_samples {
        let mut hasher = DefaultHasher::new();
        (seed, i as u64).hash(&mut hasher);
        let mut hash = hasher.finish();

        let mut count_ones = 0u32;

        for j in 0..seq_len {
            if hash == 0 {
                hasher = DefaultHasher::new();
                (seed, i as u64, j).hash(&mut hasher);
                hash = hasher.finish();
            }
            let bit = (hash & 1) as f32;
            seq_data.push(bit);
            count_ones += bit as u32;
            hash >>= 1;
        }

        let is_odd = count_ones % 2 == 1;
        if is_odd {
            label_data.push(0.0);
            label_data.push(1.0);
        } else {
            label_data.push(1.0);
            label_data.push(0.0);
        }
    }

    let sequences = Tensor::new(&seq_data, &[num_samples, seq_len, 1]);
    let labels = Tensor::new(&label_data, &[num_samples, 2]);

    (sequences, labels)
}
