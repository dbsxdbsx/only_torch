//! # 中国象棋棋子 CNN 分类器示例
//!
//! 使用 CNN 对合成的中国象棋棋子 patch 进行 15 类分类：
//! - 0: 空位
//! - 1-7: 红帅、红仕、红相、红車、红馬、红炮、红兵
//! - 8-14: 黑将、黑士、黑象、黑車、黑馬、黑炮、黑卒
//!
//! ## 数据准备
//! 先运行 Python 脚本生成训练数据：
//! ```bash
//! python scripts/generate_chess_data.py
//! ```
//!
//! ## 运行
//! ```bash
//! cargo run --example chinese_chess
//! ```
//!
//! ## 性能目标
//! - 准确率: ≥95%
//! - 训练时间: <30s
//! - 推理 batch=90: <100ms（棋盘扫描场景）

mod data;
mod model;

use data::{load_chess_data, CLASS_NAMES};
use model::ChessPieceCNN;
use only_torch::data::{DataLoader, TensorDataset};
use only_torch::metrics::accuracy;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;
use only_torch::tensor_slice;
use std::time::Instant;

fn main() -> Result<(), GraphError> {
    let total_start = Instant::now();
    println!("=== 中国象棋棋子 CNN 分类器 ===\n");

    // 1. 加载数据
    println!("[1/4] 加载训练数据...");
    let load_start = Instant::now();

    let (all_images, all_labels) = load_chess_data("data/chinese_chess")
        .map_err(|e| GraphError::ComputationError(e))?;

    let total_samples = all_images.shape()[0];
    println!(
        "  数据形状: images={:?}, labels={:?} ({:.1}s)",
        all_images.shape(),
        all_labels.shape(),
        load_start.elapsed().as_secs_f32()
    );

    // 2. 配置（诊断模式：极少数据 + 高学习率 + 多轮，看能否过拟合）
    let batch_size = 256;
    let train_samples = (total_samples * 8 / 10).min(1024);
    let test_samples = ((total_samples - train_samples) as usize).min(512);
    let max_epochs = 50;
    let learning_rate = 0.01;
    let target_accuracy = 90.0;

    println!("\n[2/4] 配置：");
    println!("  - Batch: {batch_size}");
    println!("  - 训练样本: {train_samples}");
    println!("  - 测试样本: {test_samples}");
    println!("  - Epochs: {max_epochs}");
    println!("  - 学习率: {learning_rate}");
    println!("  - 目标准确率: {target_accuracy}%");

    // 3. 划分数据集
    let train_x = tensor_slice!(all_images, 0usize..train_samples, .., .., ..);
    let train_y = tensor_slice!(all_labels, 0usize..train_samples, ..);

    let test_start = train_samples;
    let test_end = train_samples + test_samples;
    let test_x = tensor_slice!(all_images, test_start..test_end, .., .., ..);
    let test_y = tensor_slice!(all_labels, test_start..test_end, ..);

    // 为推理基准测试保留副本
    let test_x_for_bench = test_x.clone();

    let train_loader =
        DataLoader::new(TensorDataset::new(train_x, train_y), batch_size).drop_last(true);
    let test_loader =
        DataLoader::new(TensorDataset::new(test_x, test_y), batch_size).drop_last(true);

    // 4. 构建 CNN
    let graph = Graph::new_with_seed(42);
    let model = ChessPieceCNN::new(&graph)?;
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

    println!("\n  网络: Conv(3->8) -> Pool -> Conv(8->16) -> Pool -> FC(784->48) -> FC(48->15)");
    println!("  参数量: {param_count}");

    // 5. 训练
    println!("\n[3/4] 开始训练...\n");
    let train_start = Instant::now();
    let mut best_acc = 0.0f32;

    for epoch in 0..max_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

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

        // 测试
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

        println!(
            "Epoch {:2}: loss = {:.4}, 准确率 = {:.1}% ({}/{}), {:.1}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            acc,
            total_correct as usize,
            total,
            epoch_start.elapsed().as_secs_f32()
        );

        if acc >= target_accuracy {
            println!(
                "\n达到目标准确率 {acc:.1}%，提前停止训练（第 {} 轮）",
                epoch + 1
            );
            break;
        }
    }

    let train_duration = train_start.elapsed();
    println!("\n训练总耗时: {:.1}s", train_duration.as_secs_f32());

    // 6. 每类准确率
    println!("\n各类准确率：");
    print_per_class_accuracy(&graph, &model, &test_x_for_bench, &all_labels, train_samples, test_samples)?;

    // 7. 推理基准测试
    println!("\n[4/4] 推理速度基准测试...\n");
    inference_benchmark(&graph, &model, &test_x_for_bench)?;

    // 8. 保存可视化
    let vis_result = graph.visualize_snapshot("examples/chinese_chess/chinese_chess")?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    // 9. 结果
    println!(
        "\n最佳准确率: {best_acc:.1}%，总耗时: {:.1}s",
        total_start.elapsed().as_secs_f32()
    );

    if best_acc >= target_accuracy {
        println!("[OK] 中国象棋棋子分类器训练成功！");
        Ok(())
    } else {
        println!("[FAIL] 准确率不足 {target_accuracy:.0}%");
        Err(GraphError::ComputationError(
            "中国象棋 CNN 准确率不足".to_string(),
        ))
    }
}

/// 打印每类准确率
fn print_per_class_accuracy(
    graph: &Graph,
    model: &ChessPieceCNN,
    test_images: &Tensor,
    all_labels: &Tensor,
    train_samples: usize,
    test_samples: usize,
) -> Result<(), GraphError> {
    graph.eval();

    let batch_size = 256;
    let num_classes = 15;
    let mut class_correct = vec![0usize; num_classes];
    let mut class_total = vec![0usize; num_classes];

    let mut offset = 0;
    while offset < test_samples {
        let end = (offset + batch_size).min(test_samples);
        let bs = end - offset;

        let batch_x = tensor_slice!(test_images, offset..end, .., .., ..);

        let output = model.forward(&batch_x)?;
        let preds = output.value()?.unwrap();

        for i in 0..bs {
            // 找预测类
            let mut pred_class = 0;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..num_classes {
                let val = preds[[i, j]];
                if val > max_val {
                    max_val = val;
                    pred_class = j;
                }
            }

            // 找真实类
            let label_offset = train_samples + offset + i;
            let mut true_class = 0;
            for j in 0..num_classes {
                if all_labels[[label_offset, j]] > 0.5 {
                    true_class = j;
                    break;
                }
            }

            class_total[true_class] += 1;
            if pred_class == true_class {
                class_correct[true_class] += 1;
            }
        }

        offset = end;
    }

    for cid in 0..num_classes {
        let total = class_total[cid];
        if total > 0 {
            let acc = class_correct[cid] as f32 / total as f32 * 100.0;
            println!(
                "  [{:2}] {}: {:.1}% ({}/{})",
                cid, CLASS_NAMES[cid], acc, class_correct[cid], total
            );
        }
    }

    Ok(())
}

/// 推理速度基准测试
fn inference_benchmark(
    graph: &Graph,
    model: &ChessPieceCNN,
    test_images: &Tensor,
) -> Result<(), GraphError> {
    graph.eval();

    let batch_sizes = [1, 10, 90, 256];

    for &bs in &batch_sizes {
        let batch = tensor_slice!(test_images, 0usize..bs, .., .., ..);

        // 预热
        let _ = model.forward(&batch)?;

        let runs = 5;
        let start = Instant::now();
        for _ in 0..runs {
            let output = model.forward(&batch)?;
            let _ = output.value()?;
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / runs as f64;
        let per_sample_us = avg_ms * 1000.0 / bs as f64;

        println!(
            "  batch={:>3}: {:.1}ms 总计, {:.0}us/样本",
            bs, avg_ms, per_sample_us
        );
    }

    println!("\n  -> 棋盘扫描场景（90 格点）：见 batch=90 行");

    Ok(())
}
