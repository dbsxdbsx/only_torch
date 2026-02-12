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

/// 生成批量数据（直接返回 batch Tensor）
fn generate_data(n: usize, seed: u64) -> (Tensor, Tensor) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut inputs = Vec::with_capacity(n * 2);
    let mut targets = Vec::with_capacity(n * 4);

    for i in 0..n {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let h = hasher.finish();

        let x = (h % 1000) as f32 / 1000.0;
        let y = ((h >> 16) % 1000) as f32 / 1000.0;

        inputs.push(x);
        inputs.push(y);
        targets.extend_from_slice(&compute_labels(x, y));
    }

    (
        Tensor::new(&inputs, &[n, 2]),
        Tensor::new(&targets, &[n, 4]),
    )
}

fn main() -> Result<(), GraphError> {
    println!("=== 多标签点分类示例（BCE Loss）===\n");

    // 1. 创建模型
    let graph = Graph::new_with_seed(42);
    let model = MultiLabelPointClassifier::new(&graph)?;

    // 2. 优化器（学习率调低以稳定训练）
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.02);

    // 4. 生成数据（batch 模式）
    let n_train = 500;
    let n_test = 100;
    let (x_train, y_train) = generate_data(n_train, 42);
    let (x_test, y_test) = generate_data(n_test, 123);

    println!("网络: Input(2) -> Linear(32, Tanh) -> Linear(32, Tanh) -> Linear(4)");
    println!("损失: BCE Loss（多标签二元交叉熵）");
    println!("训练样本: {n_train}, 测试样本: {n_test}");
    println!("优化器: Adam\n");

    println!("标签说明:");
    println!("  - 右半边: x > 0.5");
    println!("  - 上半边: y > 0.5");
    println!("  - 对角线上: x + y > 1");
    println!("  - 中心圆: (x-0.5)² + (y-0.5)² < 0.15\n");

    // 5. 训练循环（batch 模式，整个训练集一次 forward）
    let epochs = 200;
    for epoch in 0..epochs {
        let logits = model.forward(&x_train)?;
        let loss = logits.bce_loss(&y_train)?;

        graph.snapshot_once_from(&[&loss]);

        optimizer.zero_grad()?;
        let loss_val = loss.backward()?;
        optimizer.step()?;

        if (epoch + 1) % 50 == 0 || epoch == 0 {
            println!("Epoch {:3}: 损失 = {:.4}", epoch + 1, loss_val);
        }
    }

    // 6. 测试
    println!("\n=== 测试结果（部分样本）===");

    let probs = model.predict_probs(&x_test)?;
    let probs_tensor = probs.value()?.unwrap();

    for i in 0..10.min(n_test) {
        let x = x_test[[i, 0]];
        let y = x_test[[i, 1]];

        let pred: Vec<bool> = (0..4).map(|j| probs_tensor[[i, j]] > 0.5).collect();
        let actual: Vec<bool> = (0..4).map(|j| y_test[[i, j]] > 0.5).collect();

        let pred_str: String = pred.iter().map(|&b| if b { '1' } else { '0' }).collect();
        let actual_str: String = actual.iter().map(|&b| if b { '1' } else { '0' }).collect();
        let match_str = if pred == actual { "✓" } else { "✗" };

        println!("  ({x:.2}, {y:.2}): 预测={pred_str} 实际={actual_str} {match_str}");
    }

    // 7. 使用 metrics 模块计算准确率
    let pred_tensor = probs_tensor;
    let actual_tensor = &y_test;

    // 两种多标签评估指标
    let loose = multilabel_loose_accuracy(&pred_tensor, actual_tensor, 0.5);
    let strict = multilabel_strict_accuracy(&pred_tensor, actual_tensor, 0.5);

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

    // 8. 保存计算图可视化（从训练时拍的快照渲染）
    let vis_result = graph.visualize_snapshot("examples/multi_label_point/multi_label_point")?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    Ok(())
}
