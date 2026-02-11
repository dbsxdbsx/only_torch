//! # 多标签点分类示例（BCE Loss）
//!
//! 展示 `only_torch` 的 **BCE Loss**（二元交叉熵）用于多标签分类：
//! - 一个样本可以**同时属于多个类别**
//! - 这是 BCE 相对于 Softmax CE 的核心优势
//!
//! ## 任务
//! 给定二维点 (x, y) ∈ [0, 1]²，预测 4 个独立的二值属性：
//! - `is_right`: x > 0.5（点在右半边）
//! - `is_top`: y > 0.5（点在上半边）
//! - `is_diagonal_above`: x + y > 1（点在对角线上方）
//! - `is_center`: (x-0.5)² + (y-0.5)² < 0.15（点在中心圆内）
//!
//! ## 为什么用 BCE 而不是 Softmax CE？
//! - Softmax CE：所有类别概率和 = 1，只能"N 选 1"
//! - BCE：每个输出独立，可以"N 选 M"（多标签）
//!
//! 例如：点 (0.6, 0.7) 同时满足 `is_right=1, is_top=1, is_diagonal_above=1, is_center=1`
//!
//! ## 运行
//! ```bash
//! cargo run --example multi_label_point
//! ```

mod model;

use model::MultiLabelPointClassifier;
use only_torch::metrics::{multilabel_loose_accuracy, multilabel_strict_accuracy};
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;

/// 标签名称
const LABEL_NAMES: [&str; 4] = ["右半边", "上半边", "对角线上", "中心圆"];

/// 计算点的真实标签
fn compute_labels(x: f32, y: f32) -> [f32; 4] {
    let is_right = if x > 0.5 { 1.0 } else { 0.0 };
    let is_top = if y > 0.5 { 1.0 } else { 0.0 };
    let is_diagonal_above = if x + y > 1.0 { 1.0 } else { 0.0 };
    let is_center = if (y - 0.5).mul_add(y - 0.5, (x - 0.5).powi(2)) < 0.15 {
        1.0
    } else {
        0.0
    };
    [is_right, is_top, is_diagonal_above, is_center]
}

/// 生成训练数据
fn generate_data(n: usize, seed: u64) -> Vec<(Tensor, Tensor)> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let h = hasher.finish();

        // (x, y) ∈ [0, 1]²
        let x = (h % 1000) as f32 / 1000.0;
        let y = ((h >> 16) % 1000) as f32 / 1000.0;

        let input = Tensor::new(&[x, y], &[1, 2]);
        let labels = compute_labels(x, y);
        let target = Tensor::new(&labels, &[1, 4]);

        data.push((input, target));
    }
    data
}

fn main() -> Result<(), GraphError> {
    println!("=== 多标签点分类示例（BCE Loss）===\n");

    // 1. 创建模型
    let graph = Graph::new_with_seed(42);
    let model = MultiLabelPointClassifier::new(&graph)?;

    // 2. 优化器（学习率调低以稳定训练）
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.02);

    // 4. 生成数据（增加训练样本以提高泛化能力）
    let train_data = generate_data(500, 42);
    let test_data = generate_data(100, 123);

    println!("网络: Input(2) -> Linear(32, Tanh) -> Linear(32, Tanh) -> Linear(4)");
    println!("损失: BCE Loss（多标签二元交叉熵）");
    println!("优化器: Adam\n");

    println!("标签说明:");
    println!("  - 右半边: x > 0.5");
    println!("  - 上半边: y > 0.5");
    println!("  - 对角线上: x + y > 1");
    println!("  - 中心圆: (x-0.5)² + (y-0.5)² < 0.15\n");

    // 5. 训练循环
    let epochs = 500;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (input, target) in &train_data {
            let logits = model.forward(input)?;
            let loss = logits.bce_loss(target)?;

            optimizer.zero_grad()?;
            let loss_val = loss.backward()?;
            optimizer.step()?;

            total_loss += loss_val;
        }

        if (epoch + 1) % 50 == 0 || epoch == 0 {
            let avg_loss = total_loss / train_data.len() as f32;
            println!("Epoch {:3}: 平均损失 = {:.4}", epoch + 1, avg_loss);
        }
    }

    // 6. 测试
    println!("\n=== 测试结果（部分样本）===");

    // 收集所有预测和真实值用于统计
    let mut all_preds = Vec::with_capacity(test_data.len() * 4);
    let mut all_actuals = Vec::with_capacity(test_data.len() * 4);

    for (i, (input, target)) in test_data.iter().enumerate() {
        let probs = model.predict_probs(input)?;
        let probs_tensor = probs.value()?.unwrap();

        let x = input[[0, 0]];
        let y = input[[0, 1]];

        // 收集预测概率和真实值
        for j in 0..4 {
            all_preds.push(probs_tensor[[0, j]]);
            all_actuals.push(target[[0, j]]);
        }

        // 只显示前 10 个样本
        if i < 10 {
            let pred: Vec<bool> = (0..4).map(|j| probs_tensor[[0, j]] > 0.5).collect();
            let actual: Vec<bool> = (0..4).map(|j| target[[0, j]] > 0.5).collect();

            let pred_str: String = pred.iter().map(|&b| if b { '1' } else { '0' }).collect();
            let actual_str: String = actual.iter().map(|&b| if b { '1' } else { '0' }).collect();
            let match_str = if pred == actual { "✓" } else { "✗" };

            println!("  ({x:.2}, {y:.2}): 预测={pred_str} 实际={actual_str} {match_str}");
        }
    }

    // 7. 使用 metrics 模块计算准确率
    let pred_tensor = Tensor::new(&all_preds, &[test_data.len(), 4]);
    let actual_tensor = Tensor::new(&all_actuals, &[test_data.len(), 4]);

    // 两种多标签评估指标
    let loose = multilabel_loose_accuracy(&pred_tensor, &actual_tensor, 0.5);
    let strict = multilabel_strict_accuracy(&pred_tensor, &actual_tensor, 0.5);

    println!("\n=== 最终评估 ===");
    println!(
        "宽松准确率（标签级）: {:.1}% ({}/{} 个标签预测正确)",
        loose.percent(),
        loose.weighted() as usize,
        loose.n_samples()
    );
    println!(
        "严格准确率（样本级）: {:.1}% ({}/{} 个样本完全匹配)",
        strict.percent(),
        strict.weighted() as usize,
        strict.n_samples()
    );

    // 按标签统计（直接使用 per_label()）
    println!("\n各标签准确率:");
    for (name, label_acc) in LABEL_NAMES.iter().zip(loose.per_label()) {
        println!(
            "  {}: {:.1}% ({}/{})",
            name,
            label_acc.percent(),
            label_acc.weighted() as usize,
            label_acc.n_samples()
        );
    }

    // 成功阈值（仅作示意，验证模型确实在学习即可，不追求极高指标）
    const LOOSE_THRESHOLD: f32 = 85.0;
    const STRICT_THRESHOLD: f32 = 70.0;

    let loose_ok = loose.percent() >= LOOSE_THRESHOLD;
    let strict_ok = strict.percent() >= STRICT_THRESHOLD;

    if loose_ok && strict_ok {
        println!("\n✅ 多标签分类训练成功！（模型已学会区分多标签属性）");
    } else {
        println!("\n⚠️ 训练未达标：");
        if !loose_ok {
            println!("   - 宽松准确率 < {LOOSE_THRESHOLD}%");
        }
        if !strict_ok {
            println!("   - 严格准确率 < {STRICT_THRESHOLD}%");
        }
        println!("   可尝试增加 epoch 或调整学习率。");
    }

    // 8. 保存计算图可视化
    // 训练后做一次 forward + loss 用于生成计算图
    let (input_vis, target_vis) = &train_data[0];
    let logits_vis = model.forward(input_vis)?;
    let loss_vis = logits_vis.bce_loss(target_vis)?;
    let vis_result = loss_vis.save_visualization("examples/multi_label_point/multi_label_point")?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    Ok(())
}
