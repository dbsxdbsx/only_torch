//! # MNIST 手写数字识别示例（PyTorch 风格）
//!
//! 展示在真实图像数据上的深度学习：
//! - MNIST 数据集（28x28 灰度图 → 10 类数字）
//! - 两层 MLP (784 -> 128 -> 10)
//! - `CrossEntropyLoss` 损失（PyTorch 风格），Adam 优化器
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
use only_torch::data::{DataLoader, MnistDataset, TensorDataset};
use only_torch::nn::{Adam, CrossEntropyLoss, Graph, GraphError, Module, Optimizer};
use only_torch::tensor_slice;
use std::time::Instant;

fn main() -> Result<(), GraphError> {
    println!("=== MNIST 手写数字识别示例（PyTorch 风格）===\n");

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
    let train_samples = 15000; // 使用部分训练集（加快演示）
    let test_samples = 1000;
    let max_epochs = 20;
    let learning_rate = 0.01;

    println!("\n[2/3] 配置：");
    println!("  - Batch: {batch_size}");
    println!(
        "  - 训练样本: {} (共 {} 个 batch)",
        train_samples,
        train_samples / batch_size
    );
    println!("  - 测试样本: {test_samples}");
    println!("  - Epochs: {max_epochs}");
    println!("  - 学习率: {learning_rate}");

    // 3. 准备数据（使用 DataLoader）
    let all_train_images = train_data.images();
    let all_train_labels = train_data.labels();
    let all_test_images = test_data.images();
    let all_test_labels = test_data.labels();

    // 提取部分数据用于训练/测试
    let train_x = tensor_slice!(all_train_images, 0usize..train_samples, ..);
    let train_y = tensor_slice!(all_train_labels, 0usize..train_samples, ..);
    let test_x = tensor_slice!(all_test_images, 0usize..test_samples, ..);
    let test_y = tensor_slice!(all_test_labels, 0usize..test_samples, ..);

    // 创建 DataLoader（drop_last=true 丢弃不完整的最后一个 batch）
    let train_loader =
        DataLoader::new(TensorDataset::new(train_x, train_y), batch_size).drop_last(true);
    let test_loader =
        DataLoader::new(TensorDataset::new(test_x, test_y), batch_size).drop_last(true);

    // 4. 构建网络（PyTorch 风格）
    let graph = Graph::new_with_seed(42);
    let model = MnistMLP::new(&graph)?;

    // 损失函数（PyTorch 风格）
    let criterion = CrossEntropyLoss::new();

    // 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), learning_rate);

    println!("\n  网络: 784 -> 128 (Softplus) -> 10");
    println!(
        "  参数: {} + {} = {} 个",
        784 * 128 + 128,
        128 * 10 + 10,
        784 * 128 + 128 + 128 * 10 + 10
    );

    // 5. 训练（PyTorch 风格！）
    println!("\n[3/3] 开始训练...\n");

    let mut best_acc = 0.0f32;

    for epoch in 0..max_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        // 训练循环（完全 PyTorch 风格！）
        for (batch_x, batch_y) in train_loader.iter() {
            // PyTorch 风格：直接传 Tensor
            let output = model.forward(&batch_x)?;
            let loss = criterion.forward(&output, &batch_y)?;

            // 反向传播 + 参数更新
            optimizer.zero_grad()?;
            let loss_val = loss.backward()?;
            optimizer.step()?;

            epoch_loss += loss_val;
            num_batches += 1;
        }

        // 测试循环
        let mut correct = 0;
        let mut total = 0;

        for (batch_x, batch_y) in test_loader.iter() {
            // PyTorch 风格：直接传 Tensor
            let output = model.forward(&batch_x)?;

            let preds = output.value()?.unwrap();
            let pred_classes = preds.argmax(1); // [batch] 预测类别
            let true_classes = batch_y.argmax(1); // [batch] 真实类别

            // 统计正确预测数
            let batch_size = batch_x.shape()[0];
            correct += (0..batch_size)
                .filter(|&i| pred_classes[[i]] == true_classes[[i]])
                .count();
            total += batch_size;
        }

        let acc = correct as f32 / total as f32 * 100.0;
        best_acc = best_acc.max(acc);

        println!(
            "Epoch {:2}: loss = {:.4}, 准确率 = {:.1}% ({}/{}), {:.1}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            acc,
            correct,
            total,
            epoch_start.elapsed().as_secs_f32()
        );

        // 早停：达到目标准确率即停止
        if acc >= 95.0 {
            println!("\n✅ 达到目标准确率 {acc:.1}%，提前停止训练");
            break;
        }
    }

    // 6. 保存可视化
    let vis_result = graph.save_visualization("examples/mnist/mnist", None)?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    // 7. 结果
    println!("\n最佳准确率: {best_acc:.1}%");

    if best_acc >= 95.0 {
        println!("✅ MNIST 示例成功！");
        Ok(())
    } else {
        println!("❌ 准确率不足 95%");
        Err(GraphError::ComputationError("MNIST 准确率不足".to_string()))
    }
}
