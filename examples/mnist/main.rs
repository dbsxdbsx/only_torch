//! # MNIST 手写数字识别示例
//!
//! 展示在真实图像数据上的深度学习：
//! - MNIST 数据集（28x28 灰度图 → 10 类数字）
//! - 两层 MLP (784 -> 128 -> 10)
//! - CrossEntropy 损失，Adam 优化器
//!
//! ## 运行
//! ```bash
//! cargo run --example mnist
//! ```
//!
//! ## 数据集
//! 首次运行会自动下载 MNIST 数据集到 `~/.cache/only_torch/mnist/`

mod model;

use model::MnistMLP;
use only_torch::data::MnistDataset;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor_slice;
use std::time::Instant;

fn main() -> Result<(), GraphError> {
    println!("=== MNIST 手写数字识别示例 ===\n");

    // 1. 加载数据
    println!("[1/3] 加载 MNIST 数据集...");
    let load_start = Instant::now();

    let train_data = MnistDataset::train()
        .expect("加载 MNIST 训练集失败（首次运行会自动下载）")
        .flatten();
    let test_data = MnistDataset::test()
        .expect("加载 MNIST 测试集失败")
        .flatten();

    println!(
        "  ✓ 训练集: {} 样本，测试集: {} 样本 ({:.1}s)",
        train_data.len(),
        test_data.len(),
        load_start.elapsed().as_secs_f32()
    );

    // 2. 配置
    let batch_size = 256;
    let train_samples = 5000; // 使用部分训练集（加快演示）
    let test_samples = 1000;
    let max_epochs = 10;
    let learning_rate = 0.01;

    println!("\n[2/3] 配置：");
    println!("  - Batch: {}", batch_size);
    println!("  - 训练样本: {} (共 {} 个 batch)", train_samples, train_samples / batch_size);
    println!("  - 测试样本: {}", test_samples);
    println!("  - Epochs: {}", max_epochs);
    println!("  - 学习率: {}", learning_rate);

    // 3. 构建网络
    let graph = Graph::new();
    let model = MnistMLP::new(&graph)?;

    let x = graph.zeros(&[batch_size, 784])?;
    let y = graph.zeros(&[batch_size, 10])?;

    let logits = model.forward(&x);
    let loss = logits.cross_entropy(&y)?;

    let mut optimizer = Adam::new(&graph, &model.parameters(), learning_rate);

    println!("\n  网络: 784 -> 128 (Softplus) -> 10");
    println!("  参数: {} + {} = {} 个", 784 * 128 + 128, 128 * 10 + 10, 784 * 128 + 128 + 128 * 10 + 10);

    // 4. 训练
    println!("\n[3/3] 开始训练...\n");

    let all_train_images = train_data.images();
    let all_train_labels = train_data.labels();
    let all_test_images = test_data.images();
    let all_test_labels = test_data.labels();

    let num_batches = train_samples / batch_size;
    let test_batches = test_samples / batch_size;

    let mut best_acc = 0.0f32;

    for epoch in 0..max_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;

        // 训练
        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            let batch_x = tensor_slice!(all_train_images, start..end, ..);
            let batch_y = tensor_slice!(all_train_labels, start..end, ..);

            x.set_value(&batch_x)?;
            y.set_value(&batch_y)?;

            optimizer.zero_grad()?;
            let loss_val = loss.backward()?;
            optimizer.step()?;

            epoch_loss += loss_val;
        }

        // 测试
        let mut correct = 0;
        for batch_idx in 0..test_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            let batch_x = tensor_slice!(all_test_images, start..end, ..);
            let batch_y = tensor_slice!(all_test_labels, start..end, ..);

            x.set_value(&batch_x)?;
            logits.forward()?;

            let preds = logits.value()?.unwrap();
            for i in 0..batch_size {
                let pred_class = (0..10)
                    .max_by(|&a, &b| preds[[i, a]].partial_cmp(&preds[[i, b]]).unwrap())
                    .unwrap();
                let true_class = (0..10).find(|&j| batch_y[[i, j]] > 0.5).unwrap();
                if pred_class == true_class {
                    correct += 1;
                }
            }
        }

        let acc = correct as f32 / (test_batches * batch_size) as f32 * 100.0;
        best_acc = best_acc.max(acc);

        println!(
            "Epoch {:2}: loss = {:.4}, 准确率 = {:.1}% ({}/{}), {:.1}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            acc,
            correct,
            test_batches * batch_size,
            epoch_start.elapsed().as_secs_f32()
        );
    }

    // 5. 结果
    println!("\n最佳准确率: {:.1}%", best_acc);

    if best_acc >= 85.0 {
        println!("✅ MNIST 示例成功！");
        Ok(())
    } else {
        println!("❌ 准确率不足 85%");
        Err(GraphError::ComputationError("MNIST 准确率不足".to_string()))
    }
}
