//! # 中国象棋棋子 CNN 分类器示例
//!
//! 演示 ONNX 模型导入、继续训练、.otm 保存/加载的完整流程。
//!
//! ## 流程
//! 1. 加载由 PyTorch 导出的 ONNX 模型
//! 2. 用合成数据评估基线准确率
//! 3. 继续训练若干 epoch，验证准确率不低于基线
//! 4. 保存为 .otm 格式
//! 5. 重新加载 .otm 并验证推理结果一致
//!
//! ## 数据准备
//! ```bash
//! # 1. 生成合成训练数据
//! python examples/traditional/chinese_chess/generate_data.py
//! # 2. 用 PyTorch 训练并导出 ONNX
//! python examples/traditional/chinese_chess/train_pytorch.py
//! ```
//!
//! ## 运行
//! ```bash
//! cargo run --example chinese_chess
//! ```

mod data;

use data::{load_chess_data, CLASS_NAMES};
use only_torch::data::{DataLoader, TensorDataset};
use only_torch::metrics::accuracy;
use only_torch::nn::{Adam, Graph, GraphError, Optimizer, RebuildResult, VarLossOps};
use only_torch::tensor::Tensor;
use only_torch::tensor_slice;
use std::path::Path;
use std::time::Instant;

const ONNX_PATH: &str = "models/chess_cnn.onnx";
const OTM_PATH: &str = "models/chinese_chess";
const DATA_DIR: &str = "data/chinese_chess";
const FINETUNE_EPOCHS: usize = 5;
const BATCH_SIZE: usize = 256;
const LEARNING_RATE: f32 = 0.0005;

fn main() -> Result<(), GraphError> {
    let total_start = Instant::now();
    println!("=== 中国象棋棋子 CNN 分类器（ONNX 互通示例）===\n");

    // ────────────────────────────────────────────
    // 1. 加载 ONNX 模型
    // ────────────────────────────────────────────
    println!("[1/5] 加载 ONNX 模型: {ONNX_PATH}");
    if !Path::new(ONNX_PATH).exists() {
        eprintln!("  ONNX 文件不存在！请先运行:");
        eprintln!("    python examples/traditional/chinese_chess/generate_data.py");
        eprintln!("    python examples/traditional/chinese_chess/train_pytorch.py");
        return Err(GraphError::ComputationError(
            "ONNX 模型文件不存在".to_string(),
        ));
    }
    let onnx_result = Graph::from_onnx(ONNX_PATH)?;
    let graph = &onnx_result.graph;
    let param_count = graph.parameter_count();
    println!("  参数量: {param_count}");
    println!("  输入节点: {}", onnx_result.inputs.len());
    println!("  输出节点: {}", onnx_result.outputs.len());

    // ────────────────────────────────────────────
    // 2. 加载数据 & 评估基线
    // ────────────────────────────────────────────
    println!("\n[2/5] 加载数据并评估基线准确率...");
    let ((train_x, train_y), (test_x, test_y)) =
        load_chess_data(DATA_DIR).map_err(GraphError::ComputationError)?;

    let test_samples = test_x.shape()[0];
    println!("  训练集: {} 样本", train_x.shape()[0]);
    println!("  测试集: {test_samples} 样本");

    let baseline_acc = evaluate_rebuild(&onnx_result, &test_x, &test_y)?;
    println!("  ONNX 基线准确率: {baseline_acc:.1}%");

    // ────────────────────────────────────────────
    // 3. 直接在 ONNX 图上继续训练
    // ────────────────────────────────────────────
    println!("\n[3/5] 继续训练 {FINETUNE_EPOCHS} 个 epoch...");

    println!("  可训练参数组: {}", onnx_result.parameters.len());

    let train_loader = DataLoader::new(TensorDataset::new(train_x, train_y), BATCH_SIZE)
        .shuffle(true)
        .drop_last(true);

    let mut optimizer = Adam::new(graph, &onnx_result.parameters, LEARNING_RATE);

    let input_var = &onnx_result.inputs[0].1;
    let output_var = &onnx_result.outputs[0];

    for epoch in 0..FINETUNE_EPOCHS {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        graph.train();
        for (batch_x, batch_y) in train_loader.iter() {
            input_var.set_value(&batch_x)?;
            let loss = output_var.cross_entropy(&batch_y)?;
            optimizer.zero_grad()?;
            let loss_val = loss.backward()?;
            optimizer.step()?;
            epoch_loss += loss_val;
            num_batches += 1;
        }

        graph.eval();
        let acc = evaluate_rebuild(&onnx_result, &test_x, &test_y)?;
        println!(
            "  Epoch {:2}: loss={:.4}, acc={:.1}%, {:.1}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            acc,
            epoch_start.elapsed().as_secs_f32()
        );
    }

    let final_acc = evaluate_rebuild(&onnx_result, &test_x, &test_y)?;
    println!("\n  训练后准确率: {final_acc:.1}%");

    if final_acc < baseline_acc {
        println!("  [WARN] 准确率低于基线 ({baseline_acc:.1}%)，但继续保存");
    } else {
        println!("  [OK] 准确率 >= 基线 ({baseline_acc:.1}%)");
    }

    // ────────────────────────────────────────────
    // 4. 保存为 .otm
    // ────────────────────────────────────────────
    println!("\n[4/5] 保存模型为 .otm...");
    graph.save_model(OTM_PATH, &[output_var])?;
    println!("  已保存: {OTM_PATH}.otm");

    // ────────────────────────────────────────────
    // 5. 重新加载 .otm 并验证
    // ────────────────────────────────────────────
    println!("\n[5/5] 重新加载 .otm 并验证...");
    let reload_result = Graph::load_model(OTM_PATH)?;
    let reload_acc = evaluate_rebuild(&reload_result, &test_x, &test_y)?;
    println!("  重载后准确率: {reload_acc:.1}%");

    let acc_diff = (final_acc - reload_acc).abs();
    if acc_diff < 0.1 {
        println!("  [OK] 保存/加载一致性验证通过 (差异 {acc_diff:.2}%)");
    } else {
        println!("  [WARN] 保存/加载后准确率差异 {acc_diff:.1}%");
    }

    // ────────────────────────────────────────────
    // 每类准确率
    // ────────────────────────────────────────────
    println!("\n各类准确率（最终模型）：");
    print_per_class_accuracy(&onnx_result, &test_x, &test_y)?;

    println!(
        "\n总耗时: {:.1}s",
        total_start.elapsed().as_secs_f32()
    );
    println!("[OK] 中国象棋 CNN 示例完成！");
    Ok(())
}

/// 用 RebuildResult（ONNX 或 .otm 加载结果）评估准确率
fn evaluate_rebuild(
    result: &RebuildResult,
    test_images: &Tensor,
    test_labels: &Tensor,
) -> Result<f32, GraphError> {
    result.graph.eval();
    let test_samples = test_images.shape()[0];
    let mut total_correct = 0.0;
    let mut total = 0;

    let mut offset = 0;
    while offset < test_samples {
        let end = (offset + BATCH_SIZE).min(test_samples);
        let batch_x = tensor_slice!(test_images, offset..end, .., .., ..);

        result.inputs[0].1.set_value(&batch_x)?;
        result.outputs[0].forward()?;
        let preds = result.outputs[0].value()?.unwrap();

        let batch_y = tensor_slice!(test_labels, offset..end, ..);
        let acc = accuracy(&preds, &batch_y);
        total_correct += acc.weighted();
        total += acc.n_samples();
        offset = end;
    }

    Ok(if total > 0 {
        total_correct / total as f32 * 100.0
    } else {
        0.0
    })
}

/// 打印每类准确率
fn print_per_class_accuracy(
    result: &RebuildResult,
    test_images: &Tensor,
    test_labels: &Tensor,
) -> Result<(), GraphError> {
    result.graph.eval();
    let num_classes = 15;
    let test_samples = test_images.shape()[0];
    let mut class_correct = vec![0usize; num_classes];
    let mut class_total = vec![0usize; num_classes];

    let mut offset = 0;
    while offset < test_samples {
        let end = (offset + BATCH_SIZE).min(test_samples);
        let bs = end - offset;
        let batch_x = tensor_slice!(test_images, offset..end, .., .., ..);

        result.inputs[0].1.set_value(&batch_x)?;
        result.outputs[0].forward()?;
        let preds = result.outputs[0].value()?.unwrap();

        for i in 0..bs {
            let mut pred_class = 0;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..num_classes {
                let val = preds[[i, j]];
                if val > max_val {
                    max_val = val;
                    pred_class = j;
                }
            }
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
