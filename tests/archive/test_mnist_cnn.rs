/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : MNIST CNN 集成测试（对应 MatrixSlow Chapter 8）
 *                 验证：Conv2d + MaxPool2d + AvgPool2d + Linear Layer + Batch 训练 + Adam 优化器
 *                 构建 LeNet 风格 CNN 进行手写数字分类
 *
 * 架构说明：
 *   本测试基于经典 LeNet-5 架构，但有以下调整：
 *   - LeNet-5 原始设计（1989, Yann LeCun）使用 **平均池化 (AvgPool)**
 *   - 现代 CNN 实践中常用 **最大池化 (MaxPool)** 以获得更好的特征提取
 *   - 本测试同时使用两种池化：pool1 用 AvgPool（经典），pool2 用 MaxPool（现代）
 *   - 这样设计既致敬经典，又验证了两种池化层的正确性
 *
 * LeNet-5 原始结构参考：
 *   C1(6@5x5) → S2(AvgPool 2x2) → C3(16@5x5) → S4(AvgPool 2x2) → FC(120) → FC(84) → Output(10)
 */

use only_torch::data::MnistDataset;
use only_torch::nn::layer::{AvgPool2d, Conv2d, Linear, MaxPool2d};
use only_torch::nn::optimizer::{Adam, Optimizer};
use only_torch::nn::{Graph, GraphError, Module, VarActivationOps, VarLossOps, VarShapeOps};
use only_torch::tensor::Tensor;
use std::fs;
use std::time::Instant;

/// MNIST CNN 集成测试
///
/// 使用 Conv2d + AvgPool2d + MaxPool2d + Linear 构建 LeNet 风格 CNN
/// 验证所有 CNN Layer API 的正确性
///
/// 网络结构（基于 LeNet-5，同时测试两种池化）：
/// ```text
/// Input [batch, 1, 28, 28]
///     ↓
/// conv1 (1→8, 5x5, pad=2) → ReLU → [batch, 8, 28, 28]
///     ↓
/// avg_pool1 (2x2, stride=2) → [batch, 8, 14, 14]    ← 经典 LeNet 风格 (AvgPool)
///     ↓
/// conv2 (8→16, 3x3, pad=1) → ReLU → [batch, 16, 14, 14]
///     ↓
/// max_pool2 (2x2, stride=2) → [batch, 16, 7, 7]    ← 现代 CNN 风格 (MaxPool)
///     ↓
/// flatten → [batch, 784]
///     ↓
/// fc1 (784 → 64) → ReLU
///     ↓
/// fc2 (64 → 10) → SoftmaxCrossEntropy
/// ```
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn test_mnist_cnn() -> Result<(), GraphError> {
    let start_time = Instant::now();

    println!("\n{}", "=".repeat(60));
    println!("=== MNIST CNN 集成测试（LeNet 风格）===");
    println!("{}\n", "=".repeat(60));

    // ========== 1. 加载数据 ==========
    println!("[1/4] 加载 MNIST 数据集...");
    let load_start = Instant::now();

    let train_data = MnistDataset::train().expect("加载 MNIST 训练集失败");
    let test_data = MnistDataset::test().expect("加载 MNIST 测试集失败");
    // 注意：CNN 需要 [N, C, H, W] 格式，不 flatten

    println!(
        "  ✓ 训练集: {} 样本，测试集: {} 样本，耗时 {:.2}s",
        train_data.len(),
        test_data.len(),
        load_start.elapsed().as_secs_f32()
    );

    // ========== 2. 训练配置（与 test_mnist_batch.rs 保持一致）==========
    let batch_size = 512;
    let train_samples = 5000;
    let test_samples = 1000;
    let max_epochs = 15;
    let num_batches = train_samples / batch_size;
    let learning_rate = 0.008; // 线性缩放：batch_size ×8，lr ×8
    let target_accuracy = 0.90; // 90% 准确率目标
    let consecutive_success_required = 2;

    println!("\n[2/4] 训练配置：");
    println!("  - Batch Size: {batch_size}");
    println!("  - 训练样本: {train_samples} (共 {num_batches} 个 batch)");
    println!("  - 测试样本: {test_samples}");
    println!("  - 最大 Epochs: {max_epochs}");
    println!("  - 学习率: {learning_rate}");
    println!("  - 目标准确率: {:.0}%", target_accuracy * 100.0);

    // ========== 3. 构建 CNN 网络 ==========
    println!("\n[3/4] 构建 LeNet 风格 CNN...");
    let build_start = Instant::now();

    let graph = Graph::new_with_seed(42);

    // 输入节点: [batch, 1, 28, 28]
    let x = graph.zeros(&[batch_size, 1, 28, 28])?;
    // 标签节点: [batch, 10]
    let y = graph.zeros(&[batch_size, 10])?;

    // ========== 卷积层 1 ==========
    // conv1: 1→8 通道, 5x5 核, padding=2 (same padding)
    let conv1 = Conv2d::new_seeded(&graph, 1, 8, (5, 5), (1, 1), (2, 2), true, "conv1", 42)?;
    // conv1 输出: [batch, 8, 28, 28]
    let h1 = conv1.forward(&x).leaky_relu(0.0);

    // pool1: 2x2, stride=2 —— 使用 AvgPool（经典 LeNet-5 风格）
    let pool1 = AvgPool2d::new((2, 2), Some((2, 2)), "avg_pool1");
    // pool1 输出: [batch, 8, 14, 14]
    let h2 = pool1.forward(&h1);

    // ========== 卷积层 2 ==========
    // conv2: 8→16 通道, 3x3 核, padding=1 (same padding)
    let conv2 = Conv2d::new_seeded(&graph, 8, 16, (3, 3), (1, 1), (1, 1), true, "conv2", 43)?;
    // conv2 输出: [batch, 16, 14, 14]
    let h3 = conv2.forward(&h2).leaky_relu(0.0);

    // pool2: 2x2, stride=2 —— 使用 MaxPool（现代 CNN 风格）
    let pool2 = MaxPool2d::new((2, 2), Some((2, 2)), "max_pool2");
    // pool2 输出: [batch, 16, 7, 7]
    let h4 = pool2.forward(&h3);

    // ========== 展平 + 全连接层 ==========
    // flatten: [batch, 16, 7, 7] → [batch, 784]
    let flat = h4.flatten()?;

    // fc1: 784 → 64
    let fc1 = Linear::new(&graph, 784, 64, true, "fc1")?;
    let h5 = fc1.forward(&flat).leaky_relu(0.0);

    // fc2: 64 → 10 (输出层)
    let fc2 = Linear::new(&graph, 64, 10, true, "fc2")?;
    let logits = fc2.forward(&h5);

    // 损失函数
    let loss = logits.cross_entropy(&y)?;

    // 收集参数
    let mut params = conv1.parameters();
    params.extend(conv2.parameters());
    params.extend(fc1.parameters());
    params.extend(fc2.parameters());

    println!(
        "  ✓ CNN 构建完成，耗时 {:.2}s",
        build_start.elapsed().as_secs_f32()
    );
    println!("  网络结构（基于 LeNet-5，混合两种池化）：");
    println!("    Input [batch, 1, 28, 28]");
    println!("      → Conv1 (1→8, 5x5, bias) → ReLU → AvgPool (2x2)  [经典]");
    println!("      → Conv2 (8→16, 3x3, bias) → ReLU → MaxPool (2x2) [现代]");
    println!("      → Flatten → FC1 (784→64) → ReLU → FC2 (64→10)");
    println!("      → SoftmaxCrossEntropy");

    // 保存网络结构可视化（训练前）
    let output_dir = "tests/outputs";
    fs::create_dir_all(output_dir).ok();
    graph
        .inner()
        .save_visualization(format!("{output_dir}/mnist_cnn"), None)?;
    graph
        .inner()
        .save_summary(format!("{output_dir}/mnist_cnn_summary.md"))?;
    println!("  ✓ 网络结构已保存: {output_dir}/mnist_cnn.png");

    // ========== 4. 训练循环 ==========
    println!("\n[4/4] 开始训练...\n");

    let mut optimizer = Adam::new(&graph, &params, learning_rate);

    // 获取图像数据（保持 [N, 1, 28, 28] 格式）
    let all_train_images = train_data.images(); // [N, 1, 28, 28]
    let all_train_labels = train_data.labels(); // [N, 10]
    let all_test_images = test_data.images();
    let all_test_labels = test_data.labels();

    let mut consecutive_success_count = 0;
    let mut test_passed = false;
    let test_batches = test_samples / batch_size;

    for epoch in 0..max_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss_sum = 0.0;

        // 训练
        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            // 提取 batch 数据
            let batch_images = extract_batch_4d(all_train_images, start, end, batch_size);
            let batch_labels = extract_batch_2d(all_train_labels, start, end, batch_size);

            x.set_value(&batch_images)?;
            y.set_value(&batch_labels)?;

            let loss_val = optimizer.minimize(&loss)?;
            epoch_loss_sum += loss_val;
        }

        let epoch_avg_loss = epoch_loss_sum / num_batches as f32;

        // 测试精度
        graph.inference();
        let mut correct = 0;

        for batch_idx in 0..test_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            let batch_images = extract_batch_4d(all_test_images, start, end, batch_size);
            let batch_labels = extract_batch_2d(all_test_labels, start, end, batch_size);

            x.set_value(&batch_images)?;
            y.set_value(&batch_labels)?;

            logits.forward()?;

            let predictions = logits.value()?.unwrap();

            for i in 0..batch_size {
                let mut pred_class = 0;
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..10 {
                    let val = predictions[[i, j]];
                    if val > max_val {
                        max_val = val;
                        pred_class = j;
                    }
                }

                let mut true_class = 0;
                for j in 0..10 {
                    if batch_labels[[i, j]] > 0.5 {
                        true_class = j;
                        break;
                    }
                }

                if pred_class == true_class {
                    correct += 1;
                }
            }
        }

        graph.train();

        let total_tested = test_batches * batch_size;
        let accuracy = correct as f32 / total_tested as f32;

        println!(
            "Epoch {:2}/{}: loss = {:.4}, 准确率 = {:.1}% ({}/{}), 耗时 {:.2}s",
            epoch + 1,
            max_epochs,
            epoch_avg_loss,
            accuracy * 100.0,
            correct,
            total_tested,
            epoch_start.elapsed().as_secs_f32()
        );

        if accuracy >= target_accuracy {
            consecutive_success_count += 1;
            if consecutive_success_count >= consecutive_success_required {
                test_passed = true;
                println!(
                    "\n🎉 连续 {} 次达到 {:.0}% 以上准确率！",
                    consecutive_success_required,
                    target_accuracy * 100.0
                );
                break;
            }
        } else {
            consecutive_success_count = 0;
        }
    }

    let total_duration = start_time.elapsed();
    println!("\n总耗时: {:.2}s", total_duration.as_secs_f32());

    // 打印模型摘要
    println!("\n模型摘要：");
    graph.inner().summary();

    if test_passed {
        println!("\n{}", "=".repeat(60));
        println!("✅ MNIST CNN 测试通过！");
        println!("{}\n", "=".repeat(60));
        Ok(())
    } else {
        println!("\n{}", "=".repeat(60));
        println!(
            "❌ 测试失败：在 {} 个 epoch 内未能连续 {} 次达到 {:.0}% 准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        );
        println!("{}\n", "=".repeat(60));
        Err(GraphError::ComputationError(format!(
            "MNIST CNN 测试失败：在 {} 个 epoch 内未能连续 {} 次达到 {:.0}% 准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        )))
    }
}

/// 从 4D 张量中提取 batch（手动实现，避免宏依赖问题）
fn extract_batch_4d(tensor: &Tensor, start: usize, end: usize, batch_size: usize) -> Tensor {
    let shape = tensor.shape();
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];

    let mut data = Vec::with_capacity(batch_size * c * h * w);

    for n in start..end {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    data.push(tensor[[n, ci, hi, wi]]);
                }
            }
        }
    }

    Tensor::new(&data, &[batch_size, c, h, w])
}

/// 从 2D 张量中提取 batch
fn extract_batch_2d(tensor: &Tensor, start: usize, end: usize, batch_size: usize) -> Tensor {
    let shape = tensor.shape();
    let cols = shape[1];

    let mut data = Vec::with_capacity(batch_size * cols);

    for n in start..end {
        for j in 0..cols {
            data.push(tensor[[n, j]]);
        }
    }

    Tensor::new(&data, &[batch_size, cols])
}
