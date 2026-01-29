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
use only_torch::metrics::{accuracy, confusion_matrix};
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
    let target_accuracy = 95.0;
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
            // 直接传 Tensor 和 slice，自动 argmax
            let acc = accuracy(&preds, labels) * 100.0;

            println!(
                "Epoch {:3}: loss = {:.4}, accuracy = {:.1}%",
                epoch + 1,
                loss_val,
                acc
            );

            // 早停：达到目标准确率即停止
            if acc >= target_accuracy {
                println!("\n✅ 达到目标准确率 {acc:.1}%，提前停止训练");
                break;
            }
        }
    }

    // 6. 最终评估
    let output = model.forward(&x_train)?;
    let preds = output.value()?.unwrap();

    // 直接用 metrics 函数，自动 argmax
    let final_acc = accuracy(&preds, labels) * 100.0;
    let cm = confusion_matrix(&preds, labels);
    let correct: usize = (0..cm.len()).map(|i| cm[i][i]).sum(); // 对角线之和

    println!("\n=== 最终结果 ===");
    println!("准确率: {final_acc:.1}% ({correct}/{n_samples})");

    // 混淆矩阵
    println!("\n混淆矩阵:");
    println!("              Pred");
    println!("         Set  Ver  Vir");
    for (i, name) in ["Set", "Ver", "Vir"].iter().enumerate() {
        println!(
            "True {}: {:3}  {:3}  {:3}",
            name, cm[i][0], cm[i][1], cm[i][2]
        );
    }

    // 保存可视化
    let vis_result = graph.save_visualization("examples/iris/iris", None)?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    if final_acc >= 95.0 {
        println!("\n✅ Iris 三分类成功！");
        Ok(())
    } else {
        println!("\n❌ 准确率不足 95%");
        Err(GraphError::ComputationError("分类精度不足".to_string()))
    }
}
