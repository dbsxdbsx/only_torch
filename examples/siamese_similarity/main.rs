//! # Siamese 相似度网络示例
//!
//! 验证 `only_torch` 的 **共享编码器** 功能：
//! - 两个输入复用同一个 Layer（参数共享）
//! - 梯度正确累积到共享参数
//!
//! ## 任务
//! 判断两个数是否相近：|x1 - x2| < 2.0 → 相似(1)，否则不相似(0)
//!
//! ## 目标
//! 准确率 ≥ 85%
//!
//! ## 运行
//! ```bash
//! cargo run --example siamese_similarity
//! ```

mod model;

use model::SiameseSimilarity;
use only_torch::metrics::accuracy;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;

/// 生成训练数据：(x1, x2, label)
/// label = 1 如果 |x1 - x2| < threshold，否则 0
fn generate_data(n: usize, seed: u64, threshold: f32) -> Vec<(Tensor, Tensor, Tensor)> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let h = hasher.finish();

        // 生成两个数 [-5, 5)
        let x1 = ((h % 1000) as f32 / 100.0) - 5.0;
        let x2 = (((h >> 16) % 1000) as f32 / 100.0) - 5.0;

        // 标签：相近为 1，否则为 0
        let label = if (x1 - x2).abs() < threshold {
            1.0
        } else {
            0.0
        };

        data.push((
            Tensor::new(&[x1], &[1, 1]),
            Tensor::new(&[x2], &[1, 1]),
            Tensor::new(&[label], &[1, 1]),
        ));
    }
    data
}

fn main() -> Result<(), GraphError> {
    println!("=== Siamese 相似度网络示例（共享编码器）===\n");

    // 1. 创建模型
    let graph = Graph::new_with_seed(42);
    let model = SiameseSimilarity::new(&graph)?;

    // 2. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.01);

    // 4. 生成数据
    let threshold = 2.0;
    let train_data = generate_data(200, 42, threshold);
    let test_data = generate_data(30, 123, threshold);

    // 统计数据分布
    let train_positive = train_data
        .iter()
        .filter(|(_, _, t)| t[[0, 0]] > 0.5)
        .count();
    let test_positive = test_data.iter().filter(|(_, _, t)| t[[0, 0]] > 0.5).count();

    println!("网络结构（共享编码器）:");
    println!("  Input1 ─> Encoder(8, ReLU) ─> Feat1 ─┐");
    println!("               ↑                       ├─> Concat ─> Classifier(1) ─> Sigmoid");
    println!("             共享参数                  │");
    println!("               ↓                       │");
    println!("  Input2 ─> Encoder(8, ReLU) ─> Feat2 ─┘");
    println!("\n任务: 判断 |x1 - x2| < {threshold:.1} (相似=1, 不相似=0)");
    println!(
        "数据: 训练 {} 条 (正例 {}), 测试 {} 条 (正例 {})",
        train_data.len(),
        train_positive,
        test_data.len(),
        test_positive
    );
    println!("优化器: Adam, 损失: MSE, 目标准确率: 85%\n");

    // 5. 训练循环
    let epochs = 300;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (x1, x2, target) in &train_data {
            let output = model.forward(x1, x2)?;
            let loss = output.mse_loss(target)?;

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

    // 6. 测试
    println!("\n=== 测试结果 ===");
    let mut pred_labels = Vec::new();
    let mut true_labels = Vec::new();

    for (x1, x2, target) in &test_data {
        let output = model.forward(x1, x2)?;
        let pred = output.value()?.unwrap();
        let pred_val = pred[[0, 0]];
        let target_val = target[[0, 0]];

        // 预测：> 0.5 为相似
        let pred_label = i32::from(pred_val > 0.5);
        let true_label = target_val as i32;
        pred_labels.push(pred_label);
        true_labels.push(true_label);

        let x1_val = x1[[0, 0]];
        let x2_val = x2[[0, 0]];
        let diff = (x1_val - x2_val).abs();
        println!(
            "  x1={:5.2}, x2={:5.2}, |diff|={:.2}, 预测={:.2}, 真实={}, {}",
            x1_val,
            x2_val,
            diff,
            pred_val,
            true_label,
            if pred_label == true_label {
                "✓"
            } else {
                "✗"
            }
        );
    }

    let acc = accuracy(&pred_labels, &true_labels);
    println!(
        "\n准确率: {:.1}% ({}/{})",
        acc.percent(),
        acc.weighted().round() as usize,
        acc.n_samples()
    );

    let target_accuracy = 0.85;
    if acc.value() >= target_accuracy {
        println!("✅ 训练成功！共享编码器验证通过。");
    } else {
        println!(
            "⚠️ 未达到目标准确率（实际: {:.1}% < 目标: {:.0}%）",
            acc.percent(),
            target_accuracy * 100.0
        );
    }

    // 7. 保存计算图可视化（训练后做一次 forward + loss）
    let (x1, x2, target) = &train_data[0];
    let output = model.forward(x1, x2)?;
    let loss = output.mse_loss(target)?;
    let vis_result = loss.save_visualization("examples/siamese_similarity/siamese_similarity")?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    Ok(())
}
