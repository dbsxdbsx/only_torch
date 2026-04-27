//! # 双输入加法示例
//!
//! 展示 `only_torch` 的 **多输入 forward** API：
//! - `model.forward(x1, x2)` - 双输入 forward
//! - `ModelState::forward2` - 内部处理多输入的缓存和梯度路由
//!
//! ## 任务
//! 给定两个数 x1 和 x2，预测它们的和 x1 + x2。
//!
//! ## 目标
//! R² ≥ 95%（加法是简单线性函数，模型应能近乎完美拟合）
//!
//! ## 运行
//! ```bash
//! cargo run --example dual_input_add
//! ```

mod model;

use model::DualInputAdder;
use only_torch::metrics::r2_score;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;

/// `生成批量数据：(x1_batch` [N,1], `x2_batch` [N,1], `y_batch` [N,1])
fn generate_batch_data(n: usize, seed: u64) -> (Tensor, Tensor, Tensor) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut x1s = Vec::with_capacity(n);
    let mut x2s = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);

    for i in 0..n {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let h = hasher.finish();

        let x1 = ((h % 1000) as f32 / 100.0) - 5.0; // [-5, 5)
        let x2 = (((h >> 16) % 1000) as f32 / 100.0) - 5.0;

        x1s.push(x1);
        x2s.push(x2);
        ys.push(x1 + x2);
    }

    (
        Tensor::new(&x1s, &[n, 1]),
        Tensor::new(&x2s, &[n, 1]),
        Tensor::new(&ys, &[n, 1]),
    )
}

fn main() -> Result<(), GraphError> {
    println!("=== 双输入加法示例（forward2）===\n");

    // 1. 创建模型
    let graph = Graph::new_with_seed(42);
    let model = DualInputAdder::new(&graph)?;

    // 2. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.05);

    // 4. 生成训练和测试数据（batch 模式）
    let (train_x1, train_x2, train_y) = generate_batch_data(50, 42);
    let (test_x1, test_x2, test_y) = generate_batch_data(10, 123);

    // 目标 R² 分数
    let target_r2 = 0.95;

    println!("网络结构:");
    println!("  Input1(1) -> Linear(4, ReLU) ─┐");
    println!("                                ├─> Concat -> Linear(1)");
    println!("  Input2(1) -> Linear(4, ReLU) ─┘");
    println!(
        "\n优化器: Adam, 损失: MSE, 目标 R²: {:.0}%\n",
        target_r2 * 100.0
    );

    // 5. 训练循环（full-batch）
    let epochs = 200;
    for epoch in 0..epochs {
        let output = model.forward(&train_x1, &train_x2)?;
        let loss = output.mse_loss(&train_y)?;

        graph.snapshot_once_from(&[&loss]);

        optimizer.zero_grad()?;
        let loss_val = loss.backward()?;
        optimizer.step()?;

        if (epoch + 1) % 50 == 0 || epoch == 0 {
            println!("Epoch {:3}: 平均损失 = {:.6}", epoch + 1, loss_val);
        }
    }

    // 6. 测试
    println!("\n=== 测试结果 ===");
    let output = model.forward(&test_x1, &test_x2)?;
    let pred_tensor = output.value()?.unwrap();

    let n_test = test_y.shape()[0];
    let mut predictions = Vec::with_capacity(n_test);
    let mut actuals = Vec::with_capacity(n_test);

    for i in 0..n_test {
        let pred_val = pred_tensor[[i, 0]];
        let target_val = test_y[[i, 0]];
        let error = (pred_val - target_val).abs();

        predictions.push(pred_val);
        actuals.push(target_val);

        println!(
            "  {:.2} + {:.2} = {:.2} (预测: {:.2}, 误差: {:.3})",
            test_x1[[i, 0]],
            test_x2[[i, 0]],
            target_val,
            pred_val,
            error
        );
    }

    // 计算 R² 分数
    let r2 = r2_score(&predictions, &actuals);
    println!("\nR² 分数: {:.4} ({:.1}%)", r2.value(), r2.percent());

    if r2.value() >= target_r2 {
        println!(
            "✅ 训练成功！模型学会了加法（R² ≥ {:.0}%）。",
            target_r2 * 100.0
        );
    } else {
        println!(
            "⚠️ 未达到目标 R²（实际: {:.1}% < 目标: {:.0}%），可尝试增加 epoch 或调整学习率。",
            r2.percent(),
            target_r2 * 100.0
        );
    }

    // 7. 保存计算图可视化（从训练时拍的快照渲染）
    let vis_result =
        graph.visualize_snapshot("examples/traditional/dual_input_add/dual_input_add")?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    Ok(())
}
