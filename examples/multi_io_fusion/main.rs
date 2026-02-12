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
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;

/// `生成批量数据：(a_batch` [N,4], `b_batch` [N,8], `cls_batch` [N,2], `reg_batch` [N,1])
fn generate_batch_data(count: usize, seed: u64) -> (Tensor, Tensor, Tensor, Tensor) {
    let mut a_data = Vec::with_capacity(count * 4);
    let mut b_data = Vec::with_capacity(count * 8);
    let mut cls_data = Vec::with_capacity(count * 2);
    let mut reg_data = Vec::with_capacity(count);
    let mut h = seed;

    for _ in 0..count {
        h = h.wrapping_mul(6364136223846793005).wrapping_add(1);

        // 输入 A [4]
        let mut a = [0.0f32; 4];
        for (i, val) in a.iter_mut().enumerate() {
            h = h.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            *val = ((h % 4000) as f32 / 1000.0) - 2.0;
        }
        a_data.extend_from_slice(&a);

        // 输入 B [8]
        let mut b = [0.0f32; 8];
        for (i, val) in b.iter_mut().enumerate() {
            h = h.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            *val = ((h % 4000) as f32 / 1000.0) - 2.0;
        }
        b_data.extend_from_slice(&b);

        let total = a[0] + a[1] + b[0] + b[1];
        if total < 0.0 {
            cls_data.extend_from_slice(&[1.0, 0.0]);
        } else {
            cls_data.extend_from_slice(&[0.0, 1.0]);
        }
        reg_data.push(total.abs());
    }

    (
        Tensor::new(&a_data, &[count, 4]),
        Tensor::new(&b_data, &[count, 8]),
        Tensor::new(&cls_data, &[count, 2]),
        Tensor::new(&reg_data, &[count, 1]),
    )
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

    // 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.01);

    // 生成数据（batch 模式）
    let (train_a, train_b, train_cls, train_reg) = generate_batch_data(300, 42);
    let (test_a, test_b, test_cls, test_reg) = generate_batch_data(20, 123);

    // 训练（full-batch）
    let epochs = 200;
    for epoch in 0..epochs {
        let (cls_logits, reg_pred) = model.forward(&train_a, &train_b)?;

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

        if epoch == 0 || (epoch + 1) % 20 == 0 {
            println!(
                "Epoch {:3}: 分类损失 = {:.4}, 回归损失 = {:.4}",
                epoch + 1,
                cls_val,
                reg_val
            );
        }
    }

    // 测试
    println!("\n=== 测试结果 ===");
    let (cls_logits, reg_pred) = model.forward(&test_a, &test_b)?;
    cls_logits.forward()?;
    reg_pred.forward()?;
    let logits = cls_logits.value()?.unwrap();
    let reg_vals = reg_pred.value()?.unwrap();

    let n_test = test_reg.shape()[0];
    let mut pred_classes = Vec::with_capacity(n_test);
    let mut true_classes = Vec::with_capacity(n_test);
    let mut reg_predictions = Vec::with_capacity(n_test);
    let mut reg_actuals = Vec::with_capacity(n_test);

    for i in 0..n_test {
        let pred_cls = i32::from(logits[[i, 0]] <= logits[[i, 1]]);
        let true_cls = i32::from(test_cls[[i, 0]] <= 0.5);
        pred_classes.push(pred_cls);
        true_classes.push(true_cls);

        let pred_val = reg_vals[[i, 0]];
        let true_val = test_reg[[i, 0]];
        reg_predictions.push(pred_val);
        reg_actuals.push(true_val);
        let error = (pred_val - true_val).abs();

        let total = test_a[[i, 0]] + test_a[[i, 1]] + test_b[[i, 0]] + test_b[[i, 1]];
        let cls_str = if pred_cls == 1 { "正" } else { "负" };
        let mark = if pred_cls == true_cls { "✓" } else { "✗" };

        println!(
            "  sum={total:+.2}: 分类={cls_str} {mark} | 目标={true_val:.2}, 预测={pred_val:.2}, 误差={error:.3}"
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

    // 保存可视化（从训练时拍的快照渲染）
    let vis_result = graph.visualize_snapshot("examples/multi_io_fusion/multi_io_fusion")?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    Ok(())
}
