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

/// `生成批量数据：(x1_batch` [N,1], `x2_batch` [N,1], `label_batch` [N,1])
/// label = 1 如果 |x1 - x2| < threshold，否则 0
fn generate_batch_data(n: usize, seed: u64, threshold: f32) -> (Tensor, Tensor, Tensor) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut x1s = Vec::with_capacity(n);
    let mut x2s = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let h = hasher.finish();

        let x1 = ((h % 1000) as f32 / 100.0) - 5.0;
        let x2 = (((h >> 16) % 1000) as f32 / 100.0) - 5.0;

        x1s.push(x1);
        x2s.push(x2);
        labels.push(if (x1 - x2).abs() < threshold {
            1.0
        } else {
            0.0
        });
    }

    (
        Tensor::new(&x1s, &[n, 1]),
        Tensor::new(&x2s, &[n, 1]),
        Tensor::new(&labels, &[n, 1]),
    )
}

fn main() -> Result<(), GraphError> {
    println!("=== Siamese 相似度网络示例（共享编码器）===\n");

    // 1. 创建模型
    let graph = Graph::new_with_seed(42);
    let model = SiameseSimilarity::new(&graph)?;

    // 2. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.01);

    // 4. 生成数据（batch 模式）
    let threshold = 2.0;
    let (train_x1, train_x2, train_labels) = generate_batch_data(200, 42, threshold);
    let (test_x1, test_x2, test_labels) = generate_batch_data(30, 123, threshold);

    // 统计数据分布
    let n_train = train_labels.shape()[0];
    let n_test = test_labels.shape()[0];
    let train_positive = (0..n_train).filter(|&i| train_labels[[i, 0]] > 0.5).count();
    let test_positive = (0..n_test).filter(|&i| test_labels[[i, 0]] > 0.5).count();

    println!("网络结构（共享编码器）:");
    println!("  Input1 ─> Enc(16→8, ReLU) ─> Feat1 ─┐");
    println!("               ↑                       ├─> Concat ─> Cls(8→1, ReLU→Sigmoid)");
    println!("             共享参数                  │");
    println!("               ↓                       │");
    println!("  Input2 ─> Enc(16→8, ReLU) ─> Feat2 ─┘");
    println!("\n任务: 判断 |x1 - x2| < {threshold:.1} (相似=1, 不相似=0)");
    println!(
        "数据: 训练 {n_train} 条 (正例 {train_positive}), 测试 {n_test} 条 (正例 {test_positive})"
    );
    println!("优化器: Adam (lr=0.01), 损失: MSE, 目标准确率: 85%\n");

    // 5. 训练循环（full-batch）
    let epochs = 300;
    for epoch in 0..epochs {
        let output = model.forward(&train_x1, &train_x2)?;
        let loss = output.mse_loss(&train_labels)?;

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

    let mut pred_labels = Vec::with_capacity(n_test);
    let mut true_labels = Vec::with_capacity(n_test);

    for i in 0..n_test {
        let pred_val = pred_tensor[[i, 0]];
        let target_val = test_labels[[i, 0]];

        let pred_label = i32::from(pred_val > 0.5);
        let true_label = target_val as i32;
        pred_labels.push(pred_label);
        true_labels.push(true_label);

        let x1_val = test_x1[[i, 0]];
        let x2_val = test_x2[[i, 0]];
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

    // 7. 保存计算图可视化（从训练时拍的快照渲染）
    let vis_result = graph.visualize_snapshot("examples/traditional/siamese_similarity/siamese_similarity")?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    Ok(())
}
