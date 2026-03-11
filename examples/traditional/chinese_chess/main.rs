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
//! ## 网络架构 (与 PyTorch 验证版一致)
//! ```text
//! Input [batch, 3, 28, 28]
//!   → Conv1 (3→16, 3x3, pad=1) → BN → ReLU → MaxPool(2x2)   [batch, 16, 14, 14]
//!   → Conv2 (16→32, 3x3, pad=1) → BN → ReLU → MaxPool(2x2)  [batch, 32, 7, 7]
//!   → Flatten                                                   [batch, 1568]
//!   → FC1 (1568→128) → ReLU → Dropout(0.3)
//!   → FC2 (128→15)
//! ```
//!
//! ## 运行时数据增强
//! 训练时对每个样本应用：RandomCrop(±3px) → RandomRotation(±5°) → ColorJitter → RandomErasing(30%)
//!
//! ## 性能目标
//! - 准确率: ≥95%
//! - 推理 batch=90: <100ms（棋盘扫描场景）

mod data;
mod model;

use data::{load_chess_data, CLASS_NAMES};
use model::ChessPieceCNN;
use only_torch::data::{ColorJitter, DataLoader, TensorDataset, Transform};
use only_torch::metrics::accuracy;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;
use only_torch::tensor_slice;
use std::time::Instant;

fn main() -> Result<(), GraphError> {
    let total_start = Instant::now();
    println!("=== 中国象棋棋子 CNN 分类器 ===\n");

    // 1. 加载数据（train/test 已在生成阶段按风格分离）
    println!("[1/4] 加载训练数据...");
    let load_start = Instant::now();

    let ((train_x, train_y), (test_x, test_y)) = load_chess_data("data/chinese_chess")
        .map_err(|e| GraphError::ComputationError(e))?;

    let train_samples = train_x.shape()[0];
    let test_samples = test_x.shape()[0];
    println!(
        "  加载完成 ({:.1}s)",
        load_start.elapsed().as_secs_f32()
    );

    // 2. 配置（与 PyTorch 版一致）
    let batch_size = 128;
    let max_epochs = 50;
    let learning_rate = 0.001;
    let target_accuracy = 95.0;
    let early_stop_patience = 10;

    println!("\n[2/4] 配置：");
    println!("  - Batch: {batch_size}");
    println!("  - 训练样本: {train_samples}");
    println!("  - 测试样本: {test_samples}");
    println!("  - Epochs: {max_epochs} (early stop patience={early_stop_patience})");
    println!("  - 学习率: {learning_rate}");
    println!("  - 目标准确率: {target_accuracy}%");

    // 3. 运行时数据增强（仅训练集）
    // 仅用温和的色彩扰动 — 28x28 小图上叠加裁切/旋转/遮挡太激进会阻碍学习
    let train_transform = ColorJitter::new(0.15, 0.15, 0.1);

    // 为推理基准测试和每类准确率保留副本
    let test_x_for_eval = test_x.clone();
    let test_y_for_eval = test_y.clone();

    let train_loader = DataLoader::new(TensorDataset::new(train_x, train_y), batch_size)
        .shuffle(true)
        .drop_last(true);
    let test_loader = DataLoader::new(TensorDataset::new(test_x, test_y), batch_size);

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

    println!("\n  网络: Conv(3→16) → BN → Pool → Conv(16→32) → BN → Pool → FC(1568→128) → FC(128→15)");
    println!("  参数量: {param_count}");
    println!("  数据增强: ColorJitter(b=0.15, c=0.15, s=0.1)");

    // 5. 训练
    println!("\n[3/4] 开始训练...\n");
    let train_start = Instant::now();
    let mut best_acc = 0.0f32;
    let mut no_improve_count = 0;

    for epoch in 0..max_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        // CosineAnnealingLR: lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * epoch / T_max))
        let cosine_lr =
            0.5 * learning_rate * (1.0 + (std::f32::consts::PI * epoch as f32 / max_epochs as f32).cos());
        optimizer.set_learning_rate(cosine_lr);

        graph.train();
        for (batch_x, batch_y) in train_loader.iter() {
            // 逐样本应用色彩扰动
            let augmented = apply_transform_batch(&batch_x, &train_transform);

            let output = model.forward(&augmented)?;
            let loss = output.cross_entropy(&batch_y)?;
            graph.snapshot_once_from(&[&loss]);

            optimizer.zero_grad()?;
            let loss_val = loss.backward()?;
            optimizer.step()?;

            epoch_loss += loss_val;
            num_batches += 1;
        }

        // 测试（不做增强）
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

        let acc = if total > 0 {
            total_correct / total as f32 * 100.0
        } else {
            0.0
        };

        println!(
            "Epoch {:3}: loss = {:.4}, 准确率 = {:.1}% ({}/{}), lr={:.6}, {:.1}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            acc,
            total_correct as usize,
            total,
            cosine_lr,
            epoch_start.elapsed().as_secs_f32()
        );

        if acc > best_acc {
            best_acc = acc;
            no_improve_count = 0;
        } else {
            no_improve_count += 1;
        }

        if acc >= target_accuracy {
            println!(
                "\n达到目标准确率 {acc:.1}%，提前停止训练（第 {} 轮）",
                epoch + 1
            );
            break;
        }

        if no_improve_count >= early_stop_patience {
            println!(
                "\n连续 {early_stop_patience} 轮无提升，提前停止（最佳 {best_acc:.1}%）"
            );
            break;
        }
    }

    let train_duration = train_start.elapsed();
    println!("\n训练总耗时: {:.1}s", train_duration.as_secs_f32());

    // 6. 每类准确率
    println!("\n各类准确率：");
    print_per_class_accuracy(&graph, &model, &test_x_for_eval, &test_y_for_eval)?;

    // 7. 推理基准测试
    println!("\n[4/4] 推理速度基准测试...\n");
    inference_benchmark(&graph, &model, &test_x_for_eval)?;

    // 8. 保存可视化
    let vis_result = graph.visualize_snapshot("examples/traditional/chinese_chess/chinese_chess")?;
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

/// 对 batch 中的每个样本逐一应用 Transform，再重组为 batch
///
/// Transform 操作的是单样本 [C, H, W]，不支持 batch 维度。
fn apply_transform_batch(batch: &Tensor, transform: &dyn Transform) -> Tensor {
    let shape = batch.shape();
    let n = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    let sample_size = c * h * w;

    let batch_data = batch.data_as_slice();
    let mut augmented_data = Vec::with_capacity(n * sample_size);

    for i in 0..n {
        let start = i * sample_size;
        let end = start + sample_size;
        let sample = Tensor::new(&batch_data[start..end], &[c, h, w]);
        let transformed = transform.apply(&sample);

        let t_shape = transformed.shape();
        assert_eq!(
            t_shape, &[c, h, w],
            "Transform 后形状变化: 期望 [{c},{h},{w}]，得到 {t_shape:?}"
        );
        augmented_data.extend_from_slice(transformed.data_as_slice());
    }

    Tensor::new(&augmented_data, &[n, c, h, w])
}

/// 打印每类准确率
fn print_per_class_accuracy(
    graph: &Graph,
    model: &ChessPieceCNN,
    test_images: &Tensor,
    test_labels: &Tensor,
) -> Result<(), GraphError> {
    graph.eval();

    let batch_size = 256;
    let num_classes = 15;
    let test_samples = test_images.shape()[0];
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
            let mut true_class = 0;
            for j in 0..num_classes {
                if test_labels[[offset + i, j]] > 0.5 {
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
        let actual_bs = bs.min(test_images.shape()[0]);
        let batch = tensor_slice!(test_images, 0usize..actual_bs, .., .., ..);

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
        let per_sample_us = avg_ms * 1000.0 / actual_bs as f64;

        println!(
            "  batch={:>3}: {:.1}ms 总计, {:.0}us/样本",
            actual_bs, avg_ms, per_sample_us
        );
    }

    println!("\n  -> 棋盘扫描场景（90 格点）：见 batch=90 行");

    Ok(())
}
