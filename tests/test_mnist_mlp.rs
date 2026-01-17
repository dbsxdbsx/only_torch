/*
 * @Author       : 老董
 * @Date         : 2026-01-17
 * @Description  : MNIST MLP 集成测试
 *
 * 展示理想的 PyTorch 风格 API：
 * - Graph + Var（用户无需接触 NodeId）
 * - Linear 层（内部自动处理 bias 广播）
 * - Optimizer（PyTorch 风格 API）
 * - 链式调用
 */

use only_torch::data::MnistDataset;
use only_torch::nn::layer::Linear;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarActivationOps, VarLossOps};
use only_torch::tensor_slice;
use std::fs;
use std::time::Instant;

/// MNIST MLP 集成测试
///
/// 展示理想的 PyTorch 风格 API 用法
#[test]
fn test_mnist_mlp() -> Result<(), GraphError> {
    let start_time = Instant::now();

    println!("\n{}", "=".repeat(60));
    println!("=== MNIST MLP 集成测试 ===");
    println!("{}\n", "=".repeat(60));

    // ========== 1. 加载数据 ==========
    println!("[1/4] 加载 MNIST 数据集...");
    let load_start = Instant::now();

    let train_data = MnistDataset::train()
        .expect("加载 MNIST 训练集失败")
        .flatten();
    let test_data = MnistDataset::test()
        .expect("加载 MNIST 测试集失败")
        .flatten();

    println!(
        "  ✓ 训练集: {} 样本，测试集: {} 样本，耗时 {:.2}s",
        train_data.len(),
        test_data.len(),
        load_start.elapsed().as_secs_f32()
    );

    // ========== 2. 训练配置 ==========
    let batch_size = 512;
    let train_samples = 5000;
    let test_samples = 1000;
    let max_epochs = 15;
    let num_batches = train_samples / batch_size;
    let learning_rate = 0.008;
    let target_accuracy = 0.90;
    let consecutive_success_required = 2;

    println!("\n[2/4] 训练配置：");
    println!("  - Batch Size: {batch_size}");
    println!("  - 训练样本: {train_samples} (共 {num_batches} 个 batch)");
    println!("  - 测试样本: {test_samples}");
    println!("  - 最大 Epochs: {max_epochs}");
    println!("  - 学习率: {learning_rate}");
    println!("  - 目标准确率: {:.0}%", target_accuracy * 100.0);

    // ========== 3. 构建网络 ==========
    println!("\n[3/4] 构建 MLP: 784 -> 128 (Softplus) -> 10...");

    let graph = Graph::new_with_seed(42);

    // 输入变量（使用 zeros 占位，稍后通过 set_value 设置真实数据）
    let x = graph.zeros(&[batch_size, 784])?;
    let y = graph.zeros(&[batch_size, 10])?;

    // ========== 使用 Linear 层构建网络 ==========
    // 隐藏层: 784 -> 128（使用 seeded 确保可重复性）
    let fc1 = Linear::new_seeded(&graph, 784, 128, true, "fc1", 100)?;
    // 输出层: 128 -> 10
    let fc2 = Linear::new_seeded(&graph, 128, 10, true, "fc2", 200)?;

    // 前向传播链（PyTorch 风格链式调用）
    // fc1: [batch, 784] -> [batch, 128]
    let h1 = fc1.forward(&x);
    let a1 = h1.softplus(); // Softplus 激活

    // fc2: [batch, 128] -> [batch, 10]
    let logits = fc2.forward(&a1);

    // 损失函数：Softmax + Cross Entropy（cross_entropy 内含 Softmax）
    let loss = logits.cross_entropy(&y)?;

    println!("  ✓ 网络构建完成：784 -> 128 -> 10（2层 MLP）");
    println!(
        "  ✓ 参数节点：fc1_W({} params), fc1_b, fc2_W, fc2_b",
        784 * 128
    );

    // 收集所有参数（使用 Module trait）
    let mut all_params: Vec<_> = fc1.parameters();
    all_params.extend(fc2.parameters());
    println!("  ✓ 总参数数量：{} 个 Var", all_params.len());

    // 保存网络结构可视化
    let output_dir = "tests/outputs";
    fs::create_dir_all(output_dir).ok();

    // ========== 4. 创建优化器 ==========
    let mut optimizer = Adam::new(&graph, &all_params, learning_rate);

    println!("  ✓ 优化器：Adam (lr={learning_rate})");

    // ========== 5. 训练循环 ==========
    println!("\n[4/4] 开始训练...\n");

    let all_train_images = train_data.images();
    let all_train_labels = train_data.labels();
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

            let batch_images = tensor_slice!(all_train_images, start..end, ..);
            let batch_labels = tensor_slice!(all_train_labels, start..end, ..);

            // 设置输入
            x.set_value(&batch_images)?;
            y.set_value(&batch_labels)?;

            // 清空梯度
            optimizer.zero_grad()?;

            // 反向传播（backward 会自动 forward）
            let loss_val = loss.backward()?;

            // 更新参数
            optimizer.step()?;

            epoch_loss_sum += loss_val;
        }

        let epoch_avg_loss = epoch_loss_sum / num_batches as f32;

        // 测试精度
        let mut correct = 0;

        for batch_idx in 0..test_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            let batch_images = tensor_slice!(all_test_images, start..end, ..);
            let batch_labels = tensor_slice!(all_test_labels, start..end, ..);

            x.set_value(&batch_images)?;
            y.set_value(&batch_labels)?;

            // 前向传播
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

    if test_passed {
        println!("\n{}", "=".repeat(60));
        println!("✅ MNIST MLP 测试通过！");
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
            "MNIST MLP 测试失败：在 {} 个 epoch 内未能连续 {} 次达到 {:.0}% 准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        )))
    }
}
