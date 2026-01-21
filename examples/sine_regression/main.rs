//! # 正弦函数拟合示例（MSE 回归，PyTorch 风格）
//!
//! 展示 MSE 损失在回归任务上的使用：
//! - 拟合 y = sin(x)
//! - 使用 Linear 层 + Tanh 激活
//! - 使用 `MseLoss`（PyTorch 风格）
//!
//! ## 运行
//! ```bash
//! cargo run --example sine_regression
//! ```

mod model;

use model::SineMLP;
use only_torch::nn::{Adam, Graph, GraphError, Module, MseLoss, Optimizer};
use only_torch::tensor::Tensor;

/// 生成 batch 数据: y = sin(x), x ∈ [-π, π]
fn generate_data(n: usize) -> (Tensor, Tensor) {
    let mut x_data = Vec::with_capacity(n);
    let mut y_data = Vec::with_capacity(n);

    for i in 0..n {
        let x = -std::f32::consts::PI + 2.0 * std::f32::consts::PI * (i as f32) / (n as f32 - 1.0);
        x_data.push(x);
        y_data.push(x.sin());
    }

    (Tensor::new(&x_data, &[n, 1]), Tensor::new(&y_data, &[n, 1]))
}

fn main() -> Result<(), GraphError> {
    println!("=== 正弦函数拟合示例（PyTorch 风格）===\n");

    let n_samples = 50;
    let (x_train, y_train) = generate_data(n_samples);

    // 1. 模型（使用固定种子确保可复现）
    let graph = Graph::new_with_seed(42);
    let model = SineMLP::new(&graph)?;

    // 2. 损失函数（PyTorch 风格）
    let criterion = MseLoss::new();

    // 3. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.05);

    println!("网络: Input(1) -> Linear(32, Tanh) -> Linear(1)");
    println!("优化器: Adam (lr=0.05), 损失: MseLoss\n");

    // 4. 训练（PyTorch 风格）
    for epoch in 0..500 {
        // PyTorch 风格：直接传 Tensor
        let output = model.forward(&x_train)?;
        let loss = criterion.forward(&output, &y_train)?;

        // 反向传播 + 参数更新
        optimizer.zero_grad()?;
        let loss_val = loss.backward()?;
        optimizer.step()?;

        if (epoch + 1) % 100 == 0 {
            println!("Epoch {:3}: MSE = {:.6}", epoch + 1, loss_val);
        }
    }

    // 5. 评估（在训练数据上）
    let output = model.forward(&x_train)?;
    let predictions = output.value()?.unwrap();

    println!("\n=== 预测结果（部分样本）===");
    let indices = [0, 12, 25, 37, 49]; // 选几个代表点
    let mut max_error: f32 = 0.0;

    for &i in &indices {
        let x_val = x_train[[i, 0]];
        let expected = y_train[[i, 0]];
        let predicted = predictions[[i, 0]];
        let error = (predicted - expected).abs();
        max_error = max_error.max(error);

        println!("  sin({x_val:+.2}) = {expected:+.4} (预测: {predicted:+.4}, 误差: {error:.4})");
    }

    // 计算全部样本的最大误差
    for i in 0..n_samples {
        let error = (predictions[[i, 0]] - y_train[[i, 0]]).abs();
        max_error = max_error.max(error);
    }

    println!("\n全部样本最大误差: {max_error:.4}");

    if max_error < 0.1 {
        println!("✅ MSE 回归成功！");
        Ok(())
    } else {
        println!("❌ 拟合精度不够 (max_error > 0.1)");
        Err(GraphError::ComputationError("回归精度不足".to_string()))
    }
}
