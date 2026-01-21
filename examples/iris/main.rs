//! # Iris 鸢尾花分类示例（三分类，PyTorch 风格）
//!
//! 展示多分类任务：
//! - 经典 Iris 数据集（150 样本，4 特征，3 类别）
//! - 三层 MLP + Tanh 激活
//! - `CrossEntropyLoss` 损失（PyTorch 风格）
//!
//! ## 运行
//! ```bash
//! cargo run --example iris
//! ```

mod data;
mod model;

use data::{CLASS_NAMES, get_labels, load_iris};
use model::IrisMLP;
use only_torch::nn::{Adam, CrossEntropyLoss, Graph, GraphError, Module, Optimizer};

fn main() -> Result<(), GraphError> {
    println!("=== Iris 鸢尾花分类示例（PyTorch 风格）===\n");

    // 1. 加载数据
    let (x_train, y_train) = load_iris();
    let labels = get_labels();
    let n_samples = 150;

    // 2. 模型（使用固定种子确保可复现）
    let graph = Graph::new_with_seed(42);
    let model = IrisMLP::new(&graph)?;

    // 3. 损失函数（PyTorch 风格）
    let criterion = CrossEntropyLoss::new();

    // 4. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.05);

    println!("数据: {n_samples} 个样本，4 个特征，3 个类别");
    println!("类别: {CLASS_NAMES:?}");
    println!("网络: Input(4) -> Linear(10, Tanh) -> Linear(10, Tanh) -> Linear(3)");
    println!("优化器: Adam (lr=0.05), 损失: CrossEntropyLoss\n");

    // 5. 训练（PyTorch 风格）
    for epoch in 0..200 {
        // PyTorch 风格：直接传 Tensor
        let output = model.forward(&x_train)?;
        let loss = criterion.forward(&output, &y_train)?;

        // 反向传播 + 参数更新
        optimizer.zero_grad()?;
        let loss_val = loss.backward()?;
        optimizer.step()?;

        if (epoch + 1) % 50 == 0 {
            // 评估（重新 forward 获取最新预测）
            let output = model.forward(&x_train)?;
            let preds = output.value()?.unwrap();
            let pred_classes = preds.argmax(1); // [n_samples] 预测类别

            let correct = (0..n_samples)
                .filter(|&i| pred_classes[[i]] as usize == labels[i])
                .count();
            let acc = correct as f32 / n_samples as f32 * 100.0;

            println!(
                "Epoch {:3}: loss = {:.4}, accuracy = {:.1}%",
                epoch + 1, loss_val, acc
            );
        }
    }

    // 6. 最终评估
    let output = model.forward(&x_train)?;
    let preds = output.value()?.unwrap();
    let pred_classes = preds.argmax(1); // [n_samples] 预测类别

    let mut correct = 0;
    let mut confusion = [[0usize; 3]; 3]; // confusion[true][pred]

    for i in 0..n_samples {
        let pred_class = pred_classes[[i]] as usize;
        let true_class = labels[i];

        confusion[true_class][pred_class] += 1;
        if pred_class == true_class {
            correct += 1;
        }
    }

    let final_acc = correct as f32 / n_samples as f32 * 100.0;

    println!("\n=== 最终结果 ===");
    println!("准确率: {final_acc:.1}% ({correct}/{n_samples})");

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
