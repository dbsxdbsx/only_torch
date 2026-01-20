//! # Iris 鸢尾花分类示例（三分类）
//!
//! 展示多分类任务：
//! - 经典 Iris 数据集（150 样本，4 特征，3 类别）
//! - 三层 MLP + Tanh 激活
//! - CrossEntropy 损失（多分类）
//!
//! ## 运行
//! ```bash
//! cargo run --example iris
//! ```

mod data;
mod model;

use data::{get_labels, load_iris, CLASS_NAMES};
use model::IrisMLP;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};

fn main() -> Result<(), GraphError> {
    println!("=== Iris 鸢尾花分类示例（三分类）===\n");

    // 1. 加载数据
    let (x_train, y_train) = load_iris();
    let labels = get_labels();
    let n_samples = 150;

    // 2. 模型（使用固定种子确保可复现）
    let graph = Graph::new_with_seed(42);
    let model = IrisMLP::new(&graph)?;

    // 3. 输入/目标（batch 模式）
    let x = graph.input(&x_train)?;
    let target = graph.input(&y_train)?;

    // 4. 前向 + CrossEntropy 损失
    let logits = model.forward(&x);
    let loss = logits.cross_entropy(&target)?;

    // 5. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.05);

    println!("数据: {} 个样本，4 个特征，3 个类别", n_samples);
    println!("类别: {:?}", CLASS_NAMES);
    println!("网络: Input(4) -> Linear(10, Tanh) -> Linear(10, Tanh) -> Linear(3)");
    println!("优化器: Adam (lr=0.05), 损失: CrossEntropy\n");

    // 6. 训练
    for epoch in 0..200 {
        optimizer.zero_grad()?;
        let loss_val = loss.backward()?;
        optimizer.step()?;

        if (epoch + 1) % 50 == 0 {
            // 评估
            logits.forward()?;
            let preds = logits.value()?.unwrap();

            let mut correct = 0;
            for i in 0..n_samples {
                // 找最大概率的类别
                let pred_class = (0..3)
                    .max_by(|&a, &b| preds[[i, a]].partial_cmp(&preds[[i, b]]).unwrap())
                    .unwrap();
                if pred_class == labels[i] {
                    correct += 1;
                }
            }
            let acc = correct as f32 / n_samples as f32 * 100.0;

            println!(
                "Epoch {:3}: loss = {:.4}, accuracy = {:.1}%",
                epoch + 1,
                loss_val,
                acc
            );
        }
    }

    // 7. 最终评估
    logits.forward()?;
    let preds = logits.value()?.unwrap();

    let mut correct = 0;
    let mut confusion = [[0usize; 3]; 3]; // confusion[true][pred]

    for i in 0..n_samples {
        let pred_class = (0..3)
            .max_by(|&a, &b| preds[[i, a]].partial_cmp(&preds[[i, b]]).unwrap())
            .unwrap();
        let true_class = labels[i];

        confusion[true_class][pred_class] += 1;
        if pred_class == true_class {
            correct += 1;
        }
    }

    let final_acc = correct as f32 / n_samples as f32 * 100.0;

    println!("\n=== 最终结果 ===");
    println!("准确率: {:.1}% ({}/{})", final_acc, correct, n_samples);

    // 混淆矩阵
    println!("\n混淆矩阵:");
    println!("              Pred");
    println!("         Set  Ver  Vir");
    for (i, name) in ["Set", "Ver", "Vir"].iter().enumerate() {
        println!(
            "True {}: {:3}  {:3}  {:3}",
            name, confusion[i][0], confusion[i][1], confusion[i][2]
        );
    }

    if final_acc >= 95.0 {
        println!("\n✅ Iris 三分类成功！");
        Ok(())
    } else {
        println!("\n❌ 准确率不足 95%");
        Err(GraphError::ComputationError("分类精度不足".to_string()))
    }
}
