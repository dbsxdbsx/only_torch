//! # MNIST CNN 手写数字识别示例（LeNet 风格）
//!
//! 展示卷积神经网络在图像分类上的应用：
//! - MNIST 数据集（28x28 灰度图 → 10 类数字）
//! - LeNet 风格 CNN: Conv → ReLU → Pool → Conv → ReLU → Pool → FC
//! - 相比 MLP：参数更少、泛化更好（平移不变性）
//!
//! ## 运行
//! ```bash
//! cargo run --example mnist_cnn
//! ```
//!
//! ## 数据集
//! 首次运行会自动下载 MNIST 数据集到 `~/.cache/only_torch/mnist/`
//!
//! ## CNN vs MLP
//! | 维度 | MLP (mnist) | CNN (本示例) |
//! |------|-------------|-------------|
//! | 参数量 | ~102K | ~13K |
//! | 平移不变性 | 无 | 有 |
//! | 适用场景 | 展平输入 | 保持空间结构 |
//!
//! ## 性能参考（debug 模式, Intel 12/13 代 CPU）
//! - 训练 ~16s（2048 样本, 7 epochs）
//! - 推理 batch=90: ~43ms（中国象棋棋盘扫描场景参考）

mod model;

use model::MnistCNN;
use only_torch::data::{DataLoader, MnistDataset, TensorDataset};
use only_torch::metrics::accuracy;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;
use only_torch::tensor_slice;
use std::time::Instant;

fn main() -> Result<(), GraphError> {
    let total_start = Instant::now();
    println!("=== MNIST CNN 手写数字识别示例（LeNet 风格）===\n");

    // 1. 加载数据（保持 [N, 1, 28, 28] 格式，不 flatten）
    println!("[1/4] 加载 MNIST 数据集...");
    let load_start = Instant::now();

    let train_data =
        MnistDataset::train().expect("加载 MNIST 训练集失败（首次运行会自动下载）");
    let test_data = MnistDataset::test().expect("加载 MNIST 测试集失败");
    // 注意: 不调用 .flatten()，CNN 需要 [N, 1, 28, 28] 格式

    println!(
        "  ✓ 训练集: {} 样本，测试集: {} 样本 ({:.1}s)",
        train_data.len(),
        test_data.len(),
        load_start.elapsed().as_secs_f32()
    );

    // 2. 配置（为速度优化：精简网络 + 较大 batch + 适当学习率）
    let batch_size = 128;
    let train_samples = 5000;
    let test_samples = 1000;
    let max_epochs = 30;
    let learning_rate = 0.003;
    let target_accuracy = 95.0;

    println!("\n[2/4] 配置：");
    println!("  - Batch: {batch_size}");
    println!(
        "  - 训练样本: {} (共 {} 个 batch)",
        train_samples,
        train_samples / batch_size
    );
    println!("  - 测试样本: {test_samples}");
    println!("  - Epochs: {max_epochs}");
    println!("  - 学习率: {learning_rate}");

    // 3. 准备数据
    let all_train_images = train_data.images(); // [N, 1, 28, 28]
    let all_train_labels = train_data.labels(); // [N, 10]
    let all_test_images = test_data.images();
    let all_test_labels = test_data.labels();

    // 图像是 4D [N, 1, 28, 28]，标签是 2D [N, 10]
    let train_x = tensor_slice!(all_train_images, 0usize..train_samples, .., .., ..);
    let train_y = tensor_slice!(all_train_labels, 0usize..train_samples, ..);
    let test_x = tensor_slice!(all_test_images, 0usize..test_samples, .., .., ..);
    let test_y = tensor_slice!(all_test_labels, 0usize..test_samples, ..);

    // 为推理基准测试保留一份 test_x 副本
    let test_x_for_bench = test_x.clone();

    let train_loader =
        DataLoader::new(TensorDataset::new(train_x, train_y), batch_size).drop_last(true);
    let test_loader =
        DataLoader::new(TensorDataset::new(test_x, test_y), batch_size).drop_last(true);

    // 4. 构建 CNN（PyTorch 风格）
    let graph = Graph::new_with_seed(42);
    let model = MnistCNN::new(&graph)?;

    let mut optimizer = Adam::new(&graph, &model.parameters(), learning_rate);

    let param_count: usize = model
        .parameters()
        .iter()
        .map(|p| {
            p.value()
                .ok()
                .and_then(|v| v)
                .map(|t| t.shape().iter().product::<usize>())
                .unwrap_or(0)
        })
        .sum();

    println!("\n  网络: Conv(1→4) → Pool → Conv(4→8) → Pool → FC(392→32) → FC(32→10)");
    println!("  参数量: {param_count}");

    // 5. 训练
    println!("\n[3/4] 开始训练...\n");

    let train_start = Instant::now();
    let mut best_acc = 0.0f32;

    for epoch in 0..max_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        // ========== 训练阶段 ==========
        graph.train();

        for (batch_x, batch_y) in train_loader.iter() {
            let output = model.forward(&batch_x)?;
            let loss = output.cross_entropy(&batch_y)?;

            graph.snapshot_once_from(&[&loss]);

            optimizer.zero_grad()?;
            let loss_val = loss.backward()?;
            optimizer.step()?;

            epoch_loss += loss_val;
            num_batches += 1;
        }

        // ========== 测试阶段 ==========
        graph.eval();

        let mut total_correct = 0.0;
        let mut total = 0;

        for (batch_x, batch_y) in test_loader.iter() {
            let output = model.forward(&batch_x)?;
            let preds = output.value()?.unwrap();

            let acc = accuracy(&preds, &batch_y);
            total_correct += acc.weighted();
            total += acc.n_samples();
        }

        let acc = total_correct / total as f32 * 100.0;
        best_acc = best_acc.max(acc);

        let correct = total_correct as usize;
        println!(
            "Epoch {:2}: loss = {:.4}, 准确率 = {:.1}% ({}/{}), {:.1}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            acc,
            correct,
            total,
            epoch_start.elapsed().as_secs_f32()
        );

        if acc >= target_accuracy {
            println!(
                "\n✅ 达到目标准确率 {acc:.1}%，提前停止训练（第 {} 轮）",
                epoch + 1
            );
            break;
        }
    }

    let train_duration = train_start.elapsed();
    println!("\n训练总耗时: {:.1}s", train_duration.as_secs_f32());

    // 6. 推理速度基准测试
    println!("\n[4/4] 推理速度基准测试...\n");
    inference_benchmark(&graph, &model, &test_x_for_bench)?;

    // 7. 保存可视化
    let vis_result = graph.visualize_snapshot("examples/mnist_cnn/mnist_cnn")?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    // 8. 结果
    println!(
        "\n最佳准确率: {best_acc:.1}%，总耗时: {:.1}s",
        total_start.elapsed().as_secs_f32()
    );

    if best_acc >= target_accuracy {
        println!("✅ MNIST CNN 示例成功！");
        Ok(())
    } else {
        println!("❌ 准确率不足 {target_accuracy:.0}%");
        Err(GraphError::ComputationError(
            "MNIST CNN 准确率不足".to_string(),
        ))
    }
}

/// 推理速度基准测试
///
/// 模拟中国象棋场景：batch=90 个 patch 的推理时间
fn inference_benchmark(
    graph: &Graph,
    model: &MnistCNN,
    test_images: &Tensor,
) -> Result<(), GraphError> {
    graph.eval();

    // 测试不同 batch 大小的推理速度
    let batch_sizes = [1, 10, 90, 256];

    for &bs in &batch_sizes {
        let batch = tensor_slice!(test_images, 0usize..bs, .., .., ..);

        // 预热（首次运行可能触发图构建/JIT 开销）
        let _ = model.forward(&batch)?;

        // 正式计时：多次运行取平均
        let runs = 5;
        let start = Instant::now();
        for _ in 0..runs {
            let output = model.forward(&batch)?;
            // 确保实际计算完成（读取输出值）
            let _ = output.value()?;
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / runs as f64;
        let per_sample_us = avg_ms * 1000.0 / bs as f64;

        println!(
            "  batch={:>3}: {:.1}ms 总计, {:.0}μs/样本",
            bs, avg_ms, per_sample_us
        );
    }

    // 重点指标：90 个 patch 的推理时间（中国象棋棋盘扫描场景）
    println!("\n  → 中国象棋场景（90 格点）：见 batch=90 行");
    println!("  → 目标：<1000ms 可接受，<100ms 理想");

    Ok(())
}
