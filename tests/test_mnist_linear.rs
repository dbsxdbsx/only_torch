/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : MNIST Linear 集成测试（MLP 架构）
 *                 验证：Linear Layer API + batch forward/backward + Adam 优化器
 *                 使用 linear() 构建纯全连接网络，对比 test_mnist_cnn.rs 的 CNN 架构
 */

use only_torch::data::MnistDataset;
use only_torch::nn::optimizer::{Adam, Optimizer};
use only_torch::nn::{Graph, GraphError, linear};
use only_torch::tensor::Tensor;
use only_torch::tensor_slice;
use std::time::Instant;

/// MNIST Linear 集成测试（MLP 架构）
///
/// 使用 Linear layer 构建 MLP，验证 Layer API 的正确性和易用性
#[test]
fn test_mnist_linear() -> Result<(), GraphError> {
    let start_time = Instant::now();

    println!("\n{}", "=".repeat(60));
    println!("=== MNIST Linear 集成测试（MLP 架构）===");
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
    println!("  - Batch Size: {}", batch_size);
    println!(
        "  - 训练样本: {} (共 {} 个 batch)",
        train_samples, num_batches
    );
    println!("  - 测试样本: {}", test_samples);
    println!("  - 最大 Epochs: {}", max_epochs);
    println!("  - 学习率: {}", learning_rate);
    println!("  - 目标准确率: {:.0}%", target_accuracy * 100.0);

    // ========== 3. 构建网络（使用 Layer API）==========
    println!("\n[3/4] 使用 linear() 构建 MLP: 784 -> 128 (Softplus) -> 10...");

    let mut graph = Graph::new_with_seed(42);

    // 输入/标签节点
    let x = graph.new_input_node(&[batch_size, 784], Some("x"))?;
    let y = graph.new_input_node(&[batch_size, 10], Some("y"))?;

    // ========== 使用 linear() 构建网络（Batch-First）==========
    // 隐藏层: 784 -> 128，使用 Softplus 激活（比 Sigmoid 更适合隐藏层）
    let fc1 = linear(&mut graph, x, 784, 128, batch_size, Some("fc1"))?;
    let a1 = graph.new_softplus_node(fc1.output, Some("fc1_act"))?;

    // 输出层: 128 -> 10
    let fc2 = linear(&mut graph, a1, 128, 10, batch_size, Some("fc2"))?;
    let logits = fc2.output;

    // 损失函数
    let loss = graph.new_softmax_cross_entropy_node(logits, y, Some("loss"))?;

    println!("  ✓ 网络构建完成：784 -> 128 -> 10（2层 MLP）");
    println!("  ✓ 参数节点：fc1_W, fc1_b, fc2_W, fc2_b");

    // ========== 4. 训练循环 ==========
    println!("\n[4/4] 开始训练...\n");

    let mut optimizer = Adam::new(&graph, learning_rate, 0.9, 0.999, 1e-8)?;

    // 设置 ones 矩阵（用于 bias 广播）
    let ones_tensor = Tensor::ones(&[batch_size, 1]);
    graph.set_node_value(fc1.ones, Some(&ones_tensor))?;
    graph.set_node_value(fc2.ones, Some(&ones_tensor))?;

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

            let batch_images = tensor_slice!(all_test_images, start..end, ..);
            let batch_labels = tensor_slice!(all_test_labels, start..end, ..);

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

    if test_passed {
        println!("\n{}", "=".repeat(60));
        println!("✅ MNIST Linear 测试通过！");
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
            "MNIST Linear 测试失败：在 {} 个 epoch 内未能连续 {} 次达到 {:.0}% 准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        )))
    }
}
