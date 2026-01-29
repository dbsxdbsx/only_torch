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
use only_torch::nn::{Adam, Graph, GraphError, Module, MseLoss, Optimizer};
use only_torch::tensor::Tensor;

/// 生成训练数据：(x1, x2, x1+x2)
fn generate_data(n: usize, seed: u64) -> Vec<(Tensor, Tensor, Tensor)> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        // 简单的伪随机数生成
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let h = hasher.finish();

        let x1 = ((h % 1000) as f32 / 100.0) - 5.0; // [-5, 5)
        let x2 = (((h >> 16) % 1000) as f32 / 100.0) - 5.0;
        let y = x1 + x2;

        data.push((
            Tensor::new(&[x1], &[1, 1]),
            Tensor::new(&[x2], &[1, 1]),
            Tensor::new(&[y], &[1, 1]),
        ));
    }
    data
}

fn main() -> Result<(), GraphError> {
    println!("=== 双输入加法示例（forward2）===\n");

    // 1. 创建模型
    let graph = Graph::new_with_seed(42);
    let model = DualInputAdder::new(&graph)?;

    // 2. 损失函数（带缓存，复用 loss 节点）
    let criterion = MseLoss::new();

    // 3. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.01);

    // 4. 生成训练数据
    let train_data = generate_data(50, 42);
    let test_data = generate_data(10, 123);

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

    // 5. 训练循环
    let epochs = 200;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (x1, x2, target) in &train_data {
            // 双输入 forward
            let output = model.forward(x1, x2)?;

            // MSE 损失（使用 criterion，自动缓存复用 loss 节点）
            let loss = criterion.forward(&output, target)?;

            // 反向传播 + 参数更新
            optimizer.zero_grad()?;
            let loss_val = loss.backward()?;
            optimizer.step()?;

            total_loss += loss_val;
        }

        if (epoch + 1) % 50 == 0 || epoch == 0 {
            let avg_loss = total_loss / train_data.len() as f32;
            println!("Epoch {:3}: 平均损失 = {:.6}", epoch + 1, avg_loss);
        }
    }

    // 6. 测试（收集预测值和真实值用于计算 R²）
    println!("\n=== 测试结果 ===");
    let mut predictions = Vec::new();
    let mut actuals = Vec::new();

    for (x1, x2, target) in &test_data {
        let output = model.forward(x1, x2)?;
        let pred = output.value()?.unwrap();
        let pred_val = pred[[0, 0]];
        let target_val = target[[0, 0]];
        let error = (pred_val - target_val).abs();

        predictions.push(pred_val);
        actuals.push(target_val);

        println!(
            "  {:.2} + {:.2} = {:.2} (预测: {:.2}, 误差: {:.3})",
            x1[[0, 0]],
            x2[[0, 0]],
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

    // 7. 保存计算图可视化
    let vis_result = graph.save_visualization("examples/dual_input_add/dual_input_add", None)?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    Ok(())
}
