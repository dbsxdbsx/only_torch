//! # 螺旋分类示例（非线性决策边界）
//!
//! 展示 MLP 学习复杂非线性分类边界的能力：
//! - 双螺旋数据集（合成）
//! - 三层 MLP + Tanh 激活
//! - CrossEntropy 损失
//!
//! ## 运行
//! ```bash
//! cargo run --example spiral
//! ```

mod model;

use model::SpiralMLP;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;

/// 生成双螺旋数据集
///
/// 返回: (inputs [n*2, 2], labels [n*2, 2])
/// - 类别 0: 顺时针螺旋
/// - 类别 1: 逆时针螺旋
fn generate_spiral_data(points_per_class: usize) -> (Tensor, Tensor) {
    let n = points_per_class;
    let total = n * 2;

    let mut x_data = Vec::with_capacity(total * 2);
    let mut y_data = Vec::with_capacity(total * 2);

    // 类别 0: 顺时针螺旋
    for i in 0..n {
        let t = (i as f32) / (n as f32) * 4.0 * std::f32::consts::PI;
        let r = t / (4.0 * std::f32::consts::PI); // 半径随角度增加
        let noise = 0.1;
        let x = r * t.cos() + noise * (rand_simple(i * 2) - 0.5);
        let y = r * t.sin() + noise * (rand_simple(i * 2 + 1) - 0.5);
        x_data.push(x);
        x_data.push(y);
        y_data.push(1.0); // one-hot [1, 0]
        y_data.push(0.0);
    }

    // 类别 1: 逆时针螺旋（旋转 π）
    for i in 0..n {
        let t = (i as f32) / (n as f32) * 4.0 * std::f32::consts::PI;
        let r = t / (4.0 * std::f32::consts::PI);
        let noise = 0.1;
        let x = r * (t + std::f32::consts::PI).cos() + noise * (rand_simple(i * 2 + 1000) - 0.5);
        let y = r * (t + std::f32::consts::PI).sin() + noise * (rand_simple(i * 2 + 1001) - 0.5);
        x_data.push(x);
        x_data.push(y);
        y_data.push(0.0); // one-hot [0, 1]
        y_data.push(1.0);
    }

    (
        Tensor::new(&x_data, &[total, 2]),
        Tensor::new(&y_data, &[total, 2]),
    )
}

/// 简单的伪随机数生成（确定性，便于复现）
fn rand_simple(seed: usize) -> f32 {
    let x = (seed as f32 * 12.9898 + 78.233).sin() * 43758.5453;
    x - x.floor()
}

fn main() -> Result<(), GraphError> {
    println!("=== 螺旋分类示例（非线性边界）===\n");

    let points_per_class = 100;
    let (x_train, y_train) = generate_spiral_data(points_per_class);
    let total_samples = points_per_class * 2;

    // 1. 模型
    let graph = Graph::new();
    let model = SpiralMLP::new(&graph)?;

    // 2. 输入/目标（batch 模式）
    let x = graph.input(&x_train)?;
    let target = graph.input(&y_train)?;

    // 3. 前向 + CrossEntropy 损失
    let logits = model.forward(&x);
    let loss = logits.cross_entropy(&target)?;

    // 4. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.05);

    println!("数据: {} 个样本（每类 {} 个）", total_samples, points_per_class);
    println!("网络: Input(2) -> Linear(16, Tanh) -> Linear(16, Tanh) -> Linear(2)");
    println!("优化器: Adam (lr=0.05), 损失: CrossEntropy\n");

    // 5. 训练
    let mut best_acc = 0.0f32;
    for epoch in 0..500 {
        optimizer.zero_grad()?;
        let loss_val = loss.backward()?;
        optimizer.step()?;

        // 每 100 epoch 评估一次
        if (epoch + 1) % 100 == 0 {
            logits.forward()?;
            let preds = logits.value()?.unwrap();

            let mut correct = 0;
            for i in 0..total_samples {
                let pred_class = if preds[[i, 0]] > preds[[i, 1]] { 0 } else { 1 };
                let true_class = if y_train[[i, 0]] > y_train[[i, 1]] { 0 } else { 1 };
                if pred_class == true_class {
                    correct += 1;
                }
            }
            let acc = correct as f32 / total_samples as f32 * 100.0;
            best_acc = best_acc.max(acc);

            println!(
                "Epoch {:3}: loss = {:.4}, accuracy = {:.1}%",
                epoch + 1,
                loss_val,
                acc
            );
        }
    }

    // 6. 最终评估
    logits.forward()?;
    let preds = logits.value()?.unwrap();

    let mut correct = 0;
    for i in 0..total_samples {
        let pred_class = if preds[[i, 0]] > preds[[i, 1]] { 0 } else { 1 };
        let true_class = if y_train[[i, 0]] > y_train[[i, 1]] { 0 } else { 1 };
        if pred_class == true_class {
            correct += 1;
        }
    }
    let final_acc = correct as f32 / total_samples as f32 * 100.0;

    println!("\n最终准确率: {:.1}%", final_acc);

    if final_acc >= 90.0 {
        println!("✅ 螺旋分类成功！MLP 学会了非线性决策边界");
        Ok(())
    } else {
        println!("❌ 准确率不足 90%");
        Err(GraphError::ComputationError("分类精度不足".to_string()))
    }
}
