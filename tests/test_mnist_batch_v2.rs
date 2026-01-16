/*
 * @Author       : 老董
 * @Date         : 2026-01-17
 * @Description  : MNIST Batch V2 集成测试（使用 V2 API）
 *
 * 对应原 test_mnist_batch.rs，改用 V2 API：
 * - Graph + Var（不再用 Graph + NodeId）
 * - Optimizer（PyTorch 风格 API）
 * - 手动构建网络（不使用 Linear 层，验证底层 matmul/add 链式调用）
 */

use only_torch::data::MnistDataset;
use only_torch::nn::{
    Adam, GraphError, Graph, Init, Optimizer, VarActivationOps, VarLossOps, VarMatrixOps,
};
use only_torch::tensor::Tensor;
use only_torch::tensor_slice;
use std::time::Instant;

/// MNIST Batch V2 集成测试
///
/// 手动构建 MLP（不使用 Linear 层），验证 V2 API 的底层链式调用
#[test]
fn test_mnist_batch_v2() -> Result<(), GraphError> {
    let start_time = Instant::now();

    println!("\n{}", "=".repeat(60));
    println!("=== MNIST Batch V2 集成测试（使用 V2 API）===");
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

    // ========== 3. 构建网络（手动，不使用 Linear 层）==========
    println!("\n[3/4] 使用 V2 API 手动构建 MLP: 784 -> 128 (Sigmoid) -> 10...");

    let graph = Graph::new_with_seed(42);

    // 输入变量（使用 zeros 占位）
    let x = graph.zeros(&[batch_size, 784])?;
    let y = graph.zeros(&[batch_size, 10])?;

    // 用于 bias 广播的 ones 矩阵
    let ones = graph.input(&Tensor::ones(&[batch_size, 1]))?;

    // ========== 手动构建网络（验证 Var 链式调用）==========
    // 隐藏层参数
    let w1 = graph.parameter(&[784, 128], Init::Kaiming, "w1")?;
    let b1 = graph.parameter(&[1, 128], Init::Zeros, "b1")?;

    // 输出层参数
    let w2 = graph.parameter(&[128, 10], Init::Kaiming, "w2")?;
    let b2 = graph.parameter(&[1, 10], Init::Zeros, "b2")?;

    // 隐藏层：z1 = x @ w1 + ones @ b1, a1 = sigmoid(z1)
    let z1 = x.matmul(&w1)?;
    let b1_broadcast = ones.matmul(&b1)?;
    let h1 = &z1 + &b1_broadcast;
    let a1 = h1.sigmoid();

    // 输出层：z2 = a1 @ w2 + ones @ b2
    let z2 = a1.matmul(&w2)?;
    let b2_broadcast = ones.matmul(&b2)?;
    let logits = &z2 + &b2_broadcast;

    // 损失函数：cross_entropy 内含 softmax
    let loss = logits.cross_entropy(&y)?;

    println!("  ✓ 网络构建完成（手动 matmul + bias 广播）");
    println!("  ✓ 参数：w1, b1, w2, b2");

    // 收集所有参数
    let all_params = vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()];

    // ========== 4. 创建 V2 优化器 ==========
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

            // 使用 V2 API 设置输入
            x.set_value(&batch_images)?;
            y.set_value(&batch_labels)?;

            // 清空梯度
            optimizer.zero_grad()?;

            // 反向传播（V2 API：backward 会自动 forward）
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
        println!("✅ MNIST Batch V2 测试通过！");
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
            "MNIST Batch V2 测试失败：在 {} 个 epoch 内未能连续 {} 次达到 {:.0}% 准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        )))
    }
}
