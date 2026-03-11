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

/// `生成批量数据：(x_batch` [N,1], `cls_batch` [N,2], `reg_batch` [N,1])
fn generate_batch_data(n: usize, seed: u64) -> (Tensor, Tensor, Tensor) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut xs = Vec::with_capacity(n);
    let mut cls = Vec::with_capacity(n * 2);
    let mut reg = Vec::with_capacity(n);

    for i in 0..n {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let h = hasher.finish();

        let x = ((h % 1000) as f32 / 100.0) - 5.0;
        let x = if x.abs() < 0.1 { x + 0.5 } else { x };

        xs.push(x);
        if x < 0.0 {
            cls.extend_from_slice(&[1.0, 0.0]);
        } else {
            cls.extend_from_slice(&[0.0, 1.0]);
        }
        reg.push(x.abs());
    }

    (
        Tensor::new(&xs, &[n, 1]),
        Tensor::new(&cls, &[n, 2]),
        Tensor::new(&reg, &[n, 1]),
    )
}

fn main() -> Result<(), GraphError> {
    println!("=== 双输出分类示例（多任务学习）===\n");

    // 1. 创建模型
    let graph = Graph::new_with_seed(42);
    let model = DualOutputClassifier::new(&graph)?;

    // 2. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.01);

    // 4. 生成训练和测试数据（batch 模式）
    let (train_x, train_cls, train_reg) = generate_batch_data(100, 42);
    let (test_x, test_cls, test_reg) = generate_batch_data(20, 123);

    println!("网络结构:");
    println!("  Input(1) -> Shared(8, ReLU) ─┬─> ClsHead(2) -> CrossEntropy");
    println!("                               └─> RegHead(1) -> MSE");
    println!("\n任务:");
    println!("  - 分类：判断正/负");
    println!("  - 回归：预测绝对值");
    println!("\n优化器: Adam, Loss: CrossEntropy + MSE\n");

    // 5. 训练循环（full-batch，多任务学习）
    let epochs = 150;
    for epoch in 0..epochs {
        let (cls_logits, reg_pred) = model.forward(&train_x)?;

        let cls_loss = cls_logits.cross_entropy(&train_cls)?;
        let reg_loss = reg_pred.mse_loss(&train_reg)?;

        graph.snapshot_once(&[
            ("Classification Loss", &cls_loss),
            ("Regression Loss", &reg_loss),
        ]);

        optimizer.zero_grad()?;
        let cls_val = cls_loss.backward()?;
        let reg_val = reg_loss.backward()?;
        optimizer.step()?;

        if (epoch + 1) % 30 == 0 || epoch == 0 {
            println!(
                "Epoch {:3}: 分类损失 = {:.4}, 回归损失 = {:.4}",
                epoch + 1,
                cls_val,
                reg_val
            );
        }
    }

    // 6. 测试
    println!("\n=== 测试结果 ===");
    let (cls_logits, reg_pred) = model.forward(&test_x)?;
    cls_logits.forward()?;
    reg_pred.forward()?;
    let cls_vals = cls_logits.value()?.unwrap();
    let reg_vals = reg_pred.value()?.unwrap();

    let n_test = test_x.shape()[0];
    let mut pred_classes = Vec::with_capacity(n_test);
    let mut true_classes = Vec::with_capacity(n_test);
    let mut reg_predictions = Vec::with_capacity(n_test);
    let mut reg_actuals = Vec::with_capacity(n_test);

    for i in 0..n_test {
        let pred_class = i32::from(cls_vals[[i, 0]] <= cls_vals[[i, 1]]);
        let true_class = i32::from(test_cls[[i, 0]] <= 0.5);
        pred_classes.push(pred_class);
        true_classes.push(true_class);

        let reg_val = reg_vals[[i, 0]];
        let target_val = test_reg[[i, 0]];
        reg_predictions.push(reg_val);
        reg_actuals.push(target_val);

        let x_val = test_x[[i, 0]];
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

    // 8. 保存计算图可视化（从训练时拍的快照渲染）
    let vis_result =
        graph.visualize_snapshot("examples/traditional/dual_output_classify/dual_output_classify")?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    Ok(())
}
