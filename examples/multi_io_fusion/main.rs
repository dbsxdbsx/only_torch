/*
 * @Author       : 老董
 * @Date         : 2026-01-28
 * @Description  : 多输入多输出融合示例
 *
 * 任务设计（简单直观）：
 * - 输入 A [4]：前两个元素的和 sum_a = a[0] + a[1]
 * - 输入 B [8]：前两个元素的和 sum_b = b[0] + b[1]
 * - 分类目标：sum_a + sum_b > 0 ? 正 : 负
 * - 回归目标：|sum_a + sum_b|
 */

mod model;

use model::MultiIOFusion;
use only_torch::metrics::{accuracy, r2_score};
use only_torch::nn::{Adam, CrossEntropyLoss, Graph, GraphError, Module, MseLoss, Optimizer};
use only_torch::tensor::Tensor;

/// 生成训练数据
fn generate_data(count: usize, seed: u64) -> Vec<(Tensor, Tensor, Tensor, Tensor)> {
    let mut data = Vec::with_capacity(count);
    let mut h = seed;

    for _ in 0..count {
        // 简单 LCG 随机数
        h = h.wrapping_mul(6364136223846793005).wrapping_add(1);

        // 输入 A [1, 4]：随机值 [-2, 2)
        let a: Vec<f32> = (0..4)
            .map(|i| {
                h = h.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
                ((h % 4000) as f32 / 1000.0) - 2.0
            })
            .collect();
        let input_a = Tensor::new(&a, &[1, 4]);

        // 输入 B [1, 8]：随机值 [-2, 2)
        let b: Vec<f32> = (0..8)
            .map(|i| {
                h = h.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
                ((h % 4000) as f32 / 1000.0) - 2.0
            })
            .collect();
        let input_b = Tensor::new(&b, &[1, 8]);

        // 计算 sum_a + sum_b
        let sum_a = a[0] + a[1];
        let sum_b = b[0] + b[1];
        let total = sum_a + sum_b;

        // 分类标签（one-hot）
        let cls_label = if total < 0.0 {
            Tensor::new(&[1.0, 0.0], &[1, 2]) // 负
        } else {
            Tensor::new(&[0.0, 1.0], &[1, 2]) // 正
        };

        // 回归目标：|total|
        let reg_target = Tensor::new(&[total.abs()], &[1, 1]);

        data.push((input_a, input_b, cls_label, reg_target));
    }

    data
}

fn main() -> Result<(), GraphError> {
    println!("=== 多输入多输出融合示例 ===\n");

    println!("网络结构:");
    println!("  输入A [4] -> 编码器A(8) ─┐");
    println!("                           ├─> 融合层(16) ─┬─> 分类头(2)");
    println!("  输入B [8] -> 编码器B(8) ─┘               └─> 回归头(1)");
    println!();
    println!("任务:");
    println!("  - 分类：sum(A[:2]) + sum(B[:2]) 的正负");
    println!("  - 回归：|sum(A[:2]) + sum(B[:2])|");
    println!();

    // 创建图和模型
    let graph = Graph::new_with_seed(42);
    let model = MultiIOFusion::new(&graph)?;

    // 损失函数
    let cls_criterion = CrossEntropyLoss::new();
    let reg_criterion = MseLoss::new();

    // 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.01);

    // 生成数据
    let train_data = generate_data(100, 42);
    let test_data = generate_data(20, 123);

    // 训练
    let epochs = 100;
    for epoch in 0..epochs {
        let mut total_cls_loss = 0.0;
        let mut total_reg_loss = 0.0;

        for (input_a, input_b, cls_label, reg_target) in &train_data {
            let (cls_logits, reg_pred) = model.forward(input_a, input_b)?;

            let cls_loss = cls_criterion.forward(&cls_logits, cls_label)?;
            let reg_loss = reg_criterion.forward(&reg_pred, reg_target)?;

            optimizer.zero_grad()?;
            let cls_val = cls_loss.backward_ex(true)?;
            let reg_val = reg_loss.backward_ex(false)?;
            optimizer.step()?;

            total_cls_loss += cls_val;
            total_reg_loss += reg_val;
        }

        if epoch == 0 || (epoch + 1) % 20 == 0 {
            println!(
                "Epoch {:3}: 分类损失 = {:.4}, 回归损失 = {:.4}",
                epoch + 1,
                total_cls_loss / train_data.len() as f32,
                total_reg_loss / train_data.len() as f32
            );
        }
    }

    // 测试
    println!("\n=== 测试结果 ===");
    let mut pred_classes = Vec::new();
    let mut true_classes = Vec::new();
    let mut reg_predictions = Vec::new();
    let mut reg_actuals = Vec::new();

    for (input_a, input_b, cls_label, reg_target) in &test_data {
        let (cls_logits, reg_pred) = model.forward(input_a, input_b)?;

        // 分类结果
        cls_logits.forward()?;
        let logits = cls_logits.value()?.unwrap();
        let pred_cls = if logits[[0, 0]] > logits[[0, 1]] {
            0
        } else {
            1
        };
        let true_cls = if cls_label[[0, 0]] > 0.5 { 0 } else { 1 };
        pred_classes.push(pred_cls);
        true_classes.push(true_cls);

        // 回归结果（收集用于 R² 计算）
        reg_pred.forward()?;
        let pred_val = reg_pred.value()?.unwrap()[[0, 0]];
        let true_val = reg_target[[0, 0]];
        reg_predictions.push(pred_val);
        reg_actuals.push(true_val);
        let error = (pred_val - true_val).abs();

        // 计算实际的 sum
        let sum_a = input_a[[0, 0]] + input_a[[0, 1]];
        let sum_b = input_b[[0, 0]] + input_b[[0, 1]];
        let total = sum_a + sum_b;

        let cls_str = if pred_cls == 1 { "正" } else { "负" };
        let mark = if pred_cls == true_cls { "✓" } else { "✗" };

        println!(
            "  sum={:+.2}: 分类={} {} | 目标={:.2}, 预测={:.2}, 误差={:.3}",
            total, cls_str, mark, true_val, pred_val, error
        );
    }

    println!("\n=== 最终评估 ===");
    let cls_acc = accuracy(&pred_classes, &true_classes);
    let r2 = r2_score(&reg_predictions, &reg_actuals);
    println!(
        "分类准确率: {:.1}% ({}/{})",
        cls_acc.percent(),
        cls_acc.weighted().round() as usize,
        cls_acc.n_samples()
    );
    println!("回归 R²: {:.4} ({:.1}%)", r2.value(), r2.percent());

    if cls_acc.value() >= 0.9 && r2.value() >= 0.8 {
        println!("\n✅ 多输入多输出融合成功！");
    } else {
        println!("\n⚠️ 可尝试增加 epoch 或调整学习率以提升效果。");
    }

    // 保存可视化
    let vis_result = graph.save_visualization("examples/multi_io_fusion/multi_io_fusion", None)?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    Ok(())
}
