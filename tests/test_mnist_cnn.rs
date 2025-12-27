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
use only_torch::nn::layer::{avg_pool2d, conv2d, linear, max_pool2d};
use only_torch::nn::optimizer::{Adam, Optimizer};
use only_torch::nn::{Graph, GraphError};
use only_torch::tensor::Tensor;
use std::fs;
use std::time::Instant;

/// MNIST CNN 集成测试
///
/// 使用 conv2d + avg_pool2d + max_pool2d + linear 构建 LeNet 风格 CNN
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
    println!("  - Batch Size: {}", batch_size);
    println!(
        "  - 训练样本: {} (共 {} 个 batch)",
        train_samples, num_batches
    );
    println!("  - 测试样本: {}", test_samples);
    println!("  - 最大 Epochs: {}", max_epochs);
    println!("  - 学习率: {}", learning_rate);
    println!("  - 目标准确率: {:.0}%", target_accuracy * 100.0);

    // ========== 3. 构建 CNN 网络 ==========
    println!("\n[3/4] 构建 LeNet 风格 CNN...");
    let build_start = Instant::now();

    let mut graph = Graph::new_with_seed(42);

    // 输入节点: [batch, 1, 28, 28]
    let x = graph.new_input_node(&[batch_size, 1, 28, 28], Some("x"))?;
    // 标签节点: [batch, 10]
    let y = graph.new_input_node(&[batch_size, 10], Some("y"))?;

    // ========== 卷积层 1 ==========
    // conv1: 1→8 通道, 5x5 核, padding=2 (same padding)
    let conv1 = conv2d(&mut graph, x, 1, 8, (5, 5), (1, 1), (2, 2), Some("conv1"))?;
    // conv1 输出: [batch, 8, 28, 28]
    let relu1 = graph.new_leaky_relu_node(conv1.output, 0.0, Some("relu1"))?;

    // pool1: 2x2, stride=2 —— 使用 AvgPool（经典 LeNet-5 风格）
    let pool1 = avg_pool2d(&mut graph, relu1, (2, 2), Some((2, 2)), Some("avg_pool1"))?;
    // pool1 输出: [batch, 8, 14, 14]

    // ========== 卷积层 2 ==========
    // conv2: 8→16 通道, 3x3 核, padding=1 (same padding)
    let conv2 = conv2d(
        &mut graph,
        pool1.output,
        8,
        16,
        (3, 3),
        (1, 1),
        (1, 1),
        Some("conv2"),
    )?;
    // conv2 输出: [batch, 16, 14, 14]
    let relu2 = graph.new_leaky_relu_node(conv2.output, 0.0, Some("relu2"))?;

    // pool2: 2x2, stride=2 —— 使用 MaxPool（现代 CNN 风格）
    let pool2 = max_pool2d(&mut graph, relu2, (2, 2), Some((2, 2)), Some("max_pool2"))?;
    // pool2 输出: [batch, 16, 7, 7]

    // ========== 展平 + 全连接层 ==========
    // flatten: [batch, 16, 7, 7] → [batch, 784]
    let flat = graph.new_flatten_node(pool2.output, true, Some("flatten"))?;

    // fc1: 784 → 64
    let fc1 = linear(&mut graph, flat, 784, 64, batch_size, Some("fc1"))?;
    let relu3 = graph.new_leaky_relu_node(fc1.output, 0.0, Some("relu3"))?;

    // fc2: 64 → 10 (输出层)
    let fc2 = linear(&mut graph, relu3, 64, 10, batch_size, Some("fc2"))?;
    let logits = fc2.output;

    // 损失函数
    let loss = graph.new_softmax_cross_entropy_node(logits, y, Some("loss"))?;

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
    graph.save_visualization_grouped(&format!("{}/mnist_cnn", output_dir), None)?;
    graph.save_summary(&format!("{}/mnist_cnn_summary.md", output_dir))?;
    println!("  ✓ 网络结构已保存: {}/mnist_cnn.png", output_dir);

    // ========== 4. 训练循环 ==========
    println!("\n[4/4] 开始训练...\n");

    let mut optimizer = Adam::new(&graph, learning_rate, 0.9, 0.999, 1e-8)?;

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
            let batch_images = extract_batch_4d(&all_train_images, start, end, batch_size);
            let batch_labels = extract_batch_2d(&all_train_labels, start, end, batch_size);

            graph.set_node_value(x, Some(&batch_images))?;
            graph.set_node_value(y, Some(&batch_labels))?;

            optimizer.one_step_batch(&mut graph, loss)?;
            optimizer.update_batch(&mut graph)?;

            let loss_val = graph.get_node_value(loss)?.unwrap()[[0, 0]];
            epoch_loss_sum += loss_val;
        }

        let epoch_avg_loss = epoch_loss_sum / num_batches as f32;

        // 测试精度
        graph.set_eval_mode();
        let mut correct = 0;

        for batch_idx in 0..test_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            let batch_images = extract_batch_4d(&all_test_images, start, end, batch_size);
            let batch_labels = extract_batch_2d(&all_test_labels, start, end, batch_size);

            graph.set_node_value(x, Some(&batch_images))?;
            graph.set_node_value(y, Some(&batch_labels))?;

            graph.forward_batch(loss)?;

            let predictions = graph.get_node_value(logits)?.unwrap();

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

        graph.set_train_mode();

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
    graph.summary();

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
