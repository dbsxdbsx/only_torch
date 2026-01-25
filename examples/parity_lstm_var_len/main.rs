//! 变长奇偶性检测训练示例（LSTM，PyTorch 风格）
//!
//! 验证智能缓存在变长序列任务上的能力。
//!
//! ## 运行方式
//! ```bash
//! cargo run --example parity_lstm_var_len
//! ```

mod model;

use model::ParityLSTM;
use only_torch::data::{BucketedDataLoader, VarLenDataset, VarLenSample};
use only_torch::nn::{Adam, CrossEntropyLoss, Graph, GraphError, Module, Optimizer};

fn main() -> Result<(), GraphError> {
    println!("=== 变长奇偶性检测（LSTM，智能缓存 + 分桶批处理）===\n");

    // ========== 超参数 ==========
    let seed = 42u64;
    let min_len = 4;
    let max_len = 12;
    let hidden_size = 16;
    let train_samples = 1000;
    let test_samples = 200;
    let max_epochs = 150;
    let lr = 0.01;
    let target_accuracy = 90.0;

    println!("超参数:");
    println!("  序列长度范围: [{min_len}, {max_len}]");
    println!("  隐藏层大小: {hidden_size}");
    println!("  学习率: {lr}");
    println!("  模型: LSTM");
    println!();

    // ========== 数据准备 ==========
    let train_dataset = generate_var_len_dataset(train_samples, min_len, max_len, seed);
    let test_dataset = generate_var_len_dataset(test_samples, min_len, max_len, seed + 1000);

    let train_loader = BucketedDataLoader::new(&train_dataset)
        .shuffle(true)
        .seed(seed);
    let test_loader = BucketedDataLoader::new(&test_dataset);

    println!(
        "数据集: 训练 {} 样本 ({} 种长度), 测试 {} 样本 ({} 种长度)",
        train_dataset.len(),
        train_loader.num_buckets(),
        test_dataset.len(),
        test_loader.num_buckets()
    );

    let train_odd = train_dataset
        .samples()
        .iter()
        .filter(|s| s.label[1] > 0.5)
        .count();
    let test_odd = test_dataset
        .samples()
        .iter()
        .filter(|s| s.label[1] > 0.5)
        .count();
    println!(
        "标签分布: 训练 {train_odd}/{train_samples} 奇数, 测试 {test_odd}/{test_samples} 奇数\n"
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
        let mut num_samples = 0;

        for (x_batch, y_batch) in train_loader.iter() {
            let batch_size = x_batch.shape()[0];

            optimizer.zero_grad()?;
            let output = model.forward(&x_batch)?;
            let loss = criterion.forward(&output, &y_batch)?;
            let loss_val = loss.backward()?;
            optimizer.step()?;

            epoch_loss += loss_val * batch_size as f32;
            num_samples += batch_size;
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            let avg_loss = epoch_loss / num_samples as f32;
            let accuracy = evaluate(&model, &test_loader)?;
            best_accuracy = best_accuracy.max(accuracy);

            println!(
                "Epoch {:3}/{}: loss={:.4}, test_acc={:.1}%",
                epoch + 1,
                max_epochs,
                avg_loss,
                accuracy
            );

            if accuracy >= target_accuracy {
                println!("\n✅ 达到目标准确率 {target_accuracy}%，提前停止训练");
                break;
            }
        }
    }

    // ========== 最终评估 ==========
    let final_accuracy = evaluate(&model, &test_loader)?;
    println!("\n========== 最终结果 ==========");
    println!("测试准确率: {final_accuracy:.1}%");
    println!("最佳准确率: {best_accuracy:.1}%");
    println!("模型缓存形状数: {}", model.cache_size());
    println!("Criterion 缓存数: {}", criterion.cache_size());

    // 保存可视化
    let vis_result = graph
        .save_visualization("examples/parity_lstm_var_len/parity_lstm_var_len", None)?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    if final_accuracy >= target_accuracy {
        println!("\n✅ LSTM 变长奇偶性检测任务成功！");
        Ok(())
    } else {
        Err(GraphError::ComputationError(format!(
            "准确率 {final_accuracy:.1}% 未达到目标 {target_accuracy}%"
        )))
    }
}

fn evaluate(model: &ParityLSTM, test_loader: &BucketedDataLoader<'_>) -> Result<f32, GraphError> {
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

    Ok(100.0 * correct as f32 / total as f32)
}

fn generate_var_len_dataset(
    num_samples: usize,
    min_len: usize,
    max_len: usize,
    seed: u64,
) -> VarLenDataset {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut dataset = VarLenDataset::new(1, 2);

    for i in 0..num_samples {
        let mut hasher = DefaultHasher::new();
        (seed, i as u64, "len").hash(&mut hasher);
        let len_hash = hasher.finish();
        let seq_len = min_len + (len_hash as usize % (max_len - min_len + 1));

        hasher = DefaultHasher::new();
        (seed, i as u64, "seq").hash(&mut hasher);
        let mut hash = hasher.finish();

        let mut features = Vec::with_capacity(seq_len);
        let mut count_ones = 0u32;

        for j in 0..seq_len {
            if hash == 0 {
                hasher = DefaultHasher::new();
                (seed, i as u64, j, "bit").hash(&mut hasher);
                hash = hasher.finish();
            }
            let bit = (hash & 1) as f32;
            features.push(bit);
            count_ones += bit as u32;
            hash >>= 1;
        }

        let is_odd = count_ones % 2 == 1;
        let label = if is_odd {
            vec![0.0, 1.0]
        } else {
            vec![1.0, 0.0]
        };

        dataset.push(VarLenSample::new(features, seq_len, 1, label));
    }

    dataset
}
