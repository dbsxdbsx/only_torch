//! # 双输出分类示例
//!
//! 展示 `only_torch` 的 **多输出 forward** API：
//! - `model.forward(x)` 返回 `(Var, Var)` 元组
//! - 分类任务 + 回归任务共享特征层
//! - 多 loss 组合训练
//!
//! ## 任务
//! 给定一个数 x ∈ [-5, 5]：
//! - **分类任务**：判断 x 是正数(1)还是负数(0)
//! - **回归任务**：预测 |x|（绝对值）
//!
//! ## 网络结构
//! ```text
//! 输入(1) ─> 共享层(8, ReLU) ─┬─> 分类头(2) ─> CrossEntropy
//!                             │
//!                             └─> 回归头(1) ─> MSE
//! ```
//!
//! ## 运行
//! ```bash
//! cargo run --example dual_output_classify
//! ```

mod model;

use model::DualOutputClassifier;
use only_torch::metrics::{accuracy, r2_score};
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;

/// 生成训练数据：(x, `cls_label`, `reg_target`)
///
/// - x: 输入值
/// - `cls_label`: one-hot 分类标签 [负=0, 正=1]
/// - `reg_target`: 回归目标 |x|
fn generate_data(n: usize, seed: u64) -> Vec<(Tensor, Tensor, Tensor)> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let h = hasher.finish();

        // x ∈ [-5, 5)，避免 x=0（边界情况）
        let x = ((h % 1000) as f32 / 100.0) - 5.0;
        let x = if x.abs() < 0.1 { x + 0.5 } else { x }; // 避免太接近 0

        // 分类标签：负数 -> [1, 0]，正数 -> [0, 1]
        let cls_label = if x < 0.0 {
            Tensor::new(&[1.0, 0.0], &[1, 2])
        } else {
            Tensor::new(&[0.0, 1.0], &[1, 2])
        };

        // 回归目标：|x|
        let reg_target = Tensor::new(&[x.abs()], &[1, 1]);

        data.push((Tensor::new(&[x], &[1, 1]), cls_label, reg_target));
    }
    data
}

fn main() -> Result<(), GraphError> {
    println!("=== 双输出分类示例（多任务学习）===\n");

    // 1. 创建模型
    let graph = Graph::new_with_seed(42);
    let model = DualOutputClassifier::new(&graph)?;

    // 2. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.01);

    // 4. 生成训练和测试数据
    let train_data = generate_data(100, 42);
    let test_data = generate_data(20, 123);

    println!("网络结构:");
    println!("  Input(1) -> Shared(8, ReLU) ─┬─> ClsHead(2) -> CrossEntropy");
    println!("                               └─> RegHead(1) -> MSE");
    println!("\n任务:");
    println!("  - 分类：判断正/负");
    println!("  - 回归：预测绝对值");
    println!("\n优化器: Adam, Loss: CrossEntropy + MSE\n");

    // 5. 训练循环
    // 多任务学习：分别对两个 loss backward，梯度会累积到共享参数
    let epochs = 150;
    for epoch in 0..epochs {
        let mut total_cls_loss = 0.0;
        let mut total_reg_loss = 0.0;

        for (x, cls_label, reg_target) in &train_data {
            // 双输出 forward
            let (cls_logits, reg_pred) = model.forward(x)?;

            // 计算两个 loss（VarLossOps）
            let cls_loss = cls_logits.cross_entropy(cls_label)?;
            let reg_loss = reg_pred.mse_loss(reg_target)?;

            // 多任务 backward：
            // 1. 清零梯度
            // 2. 第一个 loss backward（retain_graph=true 保留图）
            // 3. 第二个 loss backward（梯度累积到共享参数）
            optimizer.zero_grad()?;
            let cls_val = cls_loss.backward_ex(true)?; // retain_graph=true
            let reg_val = reg_loss.backward_ex(false)?; // 第二个不需要保留
            optimizer.step()?;

            total_cls_loss += cls_val;
            total_reg_loss += reg_val;
        }

        if (epoch + 1) % 30 == 0 || epoch == 0 {
            let avg_cls_loss = total_cls_loss / train_data.len() as f32;
            let avg_reg_loss = total_reg_loss / train_data.len() as f32;
            println!(
                "Epoch {:3}: 分类损失 = {:.4}, 回归损失 = {:.4}",
                epoch + 1,
                avg_cls_loss,
                avg_reg_loss
            );
        }
    }

    // 6. 测试
    println!("\n=== 测试结果 ===");
    let mut pred_classes = Vec::new();
    let mut true_classes = Vec::new();
    let mut reg_predictions = Vec::new();
    let mut reg_actuals = Vec::new();

    for (x, cls_label, reg_target) in &test_data {
        let (cls_logits, reg_pred) = model.forward(x)?;

        // 分类结果
        let cls_probs = cls_logits.value()?.unwrap();
        let pred_class = i32::from(cls_probs[[0, 0]] <= cls_probs[[0, 1]]);
        let true_class = i32::from(cls_label[[0, 0]] <= 0.5);
        pred_classes.push(pred_class);
        true_classes.push(true_class);

        // 回归结果（收集用于 R² 计算）
        let reg_val = reg_pred.value()?.unwrap()[[0, 0]];
        let target_val = reg_target[[0, 0]];
        reg_predictions.push(reg_val);
        reg_actuals.push(target_val);

        let x_val = x[[0, 0]];
        let sign_str = if pred_class == 1 { "正" } else { "负" };
        let correct_str = if pred_class == true_class {
            "✓"
        } else {
            "✗"
        };
        let error = (reg_val - target_val).abs();
        println!(
            "  x={x_val:+.2}: 分类={sign_str} {correct_str} | |x|={target_val:.2}, 预测={reg_val:.2}, 误差={error:.3}"
        );
    }

    // 7. 统计结果
    let cls_acc = accuracy(&pred_classes, &true_classes);
    let r2 = r2_score(&reg_predictions, &reg_actuals);

    println!("\n=== 最终评估 ===");
    println!(
        "分类准确率: {:.1}% ({}/{})",
        cls_acc.percent(),
        cls_acc.weighted().round() as usize,
        cls_acc.n_samples()
    );
    println!("回归 R²: {:.4} ({:.1}%)", r2.value(), r2.percent());

    if cls_acc.value() >= 0.9 && r2.value() >= 0.9 {
        println!("\n✅ 多任务学习成功！分类和回归任务都达到良好效果。");
    } else {
        println!("\n⚠️ 可尝试增加 epoch 或调整学习率以提升效果。");
    }

    // 8. 保存计算图可视化（训练后做一次 forward + loss）
    let (x, cls_label, _reg_target) = &train_data[0];
    let (cls_logits, _reg_pred) = model.forward(x)?;
    let loss = cls_logits.cross_entropy(cls_label)?;
    let vis_result = loss.save_visualization("examples/dual_output_classify/dual_output_classify")?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    Ok(())
}
