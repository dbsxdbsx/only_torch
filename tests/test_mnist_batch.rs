/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : MNIST Batch 机制集成测试
 *                 验证：batch forward/backward + Adam 优化器的高效训练
 *                 使用 MatMul 实现 bias 广播（ones @ bias）
 * @LastEditors  : 老董
 * @LastEditTime : 2025-12-21
 */

use only_torch::data::MnistDataset;
use only_torch::nn::optimizer::{Adam, Optimizer};
use only_torch::nn::{Graph, GraphError};
use only_torch::tensor::Tensor;
use only_torch::tensor_slice;
use std::time::Instant;

/// MNIST Batch 集成测试
///
/// 使用批量机制训练 MLP（含 bias），验证准确率达到目标
#[test]
fn test_mnist_batch() -> Result<(), GraphError> {
    let start_time = Instant::now();

    println!("\n{}", "=".repeat(60));
    println!("=== MNIST Batch 集成测试（含 bias）===");
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
    // batch_size=512 是最佳平衡点：比原始 64 快 70%，且无需增加数据量
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

    // ========== 3. 构建网络 ==========
    println!("\n[3/4] 构建 MLP 网络: 784 -> 128 (Sigmoid+bias) -> 10 (bias)...");

    let mut graph = Graph::new_with_seed(42);

    // 输入/标签节点（batch 维度）
    let x = graph.new_input_node(&[batch_size, 784], Some("x"))?;
    let y = graph.new_input_node(&[batch_size, 10], Some("y"))?;

    // 用于 bias 广播的 ones 矩阵 [batch_size, 1]
    let ones = graph.new_input_node(&[batch_size, 1], Some("ones"))?;

    // 隐藏层：784 -> 128（使用 ones @ b1 实现 bias 广播）
    let w1 = graph.new_parameter_node_seeded(&[784, 128], Some("w1"), 42)?;
    let b1 = graph.new_parameter_node_seeded(&[1, 128], Some("b1"), 43)?;
    let z1 = graph.new_mat_mul_node(x, w1, None)?; // [batch, 784] @ [784, 128] = [batch, 128]
    let b1_broadcast = graph.new_mat_mul_node(ones, b1, None)?; // [batch, 1] @ [1, 128] = [batch, 128]
    let h1 = graph.new_add_node(&[z1, b1_broadcast], None)?; // [batch, 128] + [batch, 128]
    let a1 = graph.new_sigmoid_node(h1, None)?;

    // 输出层：128 -> 10
    let w2 = graph.new_parameter_node_seeded(&[128, 10], Some("w2"), 44)?;
    let b2 = graph.new_parameter_node_seeded(&[1, 10], Some("b2"), 45)?;
    let z2 = graph.new_mat_mul_node(a1, w2, None)?; // [batch, 128] @ [128, 10] = [batch, 10]
    let b2_broadcast = graph.new_mat_mul_node(ones, b2, None)?; // [batch, 1] @ [1, 10] = [batch, 10]
    let logits = graph.new_add_node(&[z2, b2_broadcast], None)?; // [batch, 10] + [batch, 10]

    // 损失函数
    let loss = graph.new_softmax_cross_entropy_node(logits, y, Some("loss"))?;

    println!("  ✓ 网络构建完成（含 bias 广播）");

    // ========== 4. 训练循环 ==========
    println!("\n[4/4] 开始训练...\n");

    let mut optimizer = Adam::new(&graph, learning_rate, 0.9, 0.999, 1e-8)?;

    // 设置 ones 矩阵（全 1）
    let ones_tensor = Tensor::ones(&[batch_size, 1]);
    graph.set_node_value(ones, Some(&ones_tensor))?;

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

            graph.set_node_value(x, Some(&batch_images))?;
            graph.set_node_value(y, Some(&batch_labels))?;

            graph.zero_grad()?;
            graph.forward(loss)?;
            let loss_val = graph.backward(loss)?; // backward 返回 loss 值
            optimizer.step(&mut graph)?;

            epoch_loss_sum += loss_val;
        }

        let epoch_avg_loss = epoch_loss_sum / num_batches as f32;

        // 测试精度
        graph.set_eval_mode();
        let mut correct = 0;

        for batch_idx in 0..test_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            let batch_images = tensor_slice!(all_test_images, start..end, ..);
            let batch_labels = tensor_slice!(all_test_labels, start..end, ..);

            graph.set_node_value(x, Some(&batch_images))?;
            graph.set_node_value(y, Some(&batch_labels))?;

            graph.forward(loss)?;

            let predictions = graph.get_node_value(logits)?.unwrap();

            for i in 0..batch_size {
                // 预测类别
                let mut pred_class = 0;
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..10 {
                    let val = predictions[[i, j]];
                    if val > max_val {
                        max_val = val;
                        pred_class = j;
                    }
                }

                // 实际类别
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

        // 检查是否达到目标准确率
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
        println!("✅ MNIST Batch 测试通过！");
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
            "MNIST Batch 测试失败：在 {} 个 epoch 内未能连续 {} 次达到 {:.0}% 准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        )))
    }
}
