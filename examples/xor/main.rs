//! # XOR 异或问题示例
//!
//! 展示 only_torch 的 PyTorch 风格 API：
//! - `Linear` 层（全连接）
//! - `Adam` 优化器
//! - `cross_entropy` 损失函数
//!
//! ## 运行
//! ```bash
//! cargo run --example xor
//! ```

mod model;

use model::XorMLP;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;

/// XOR 训练数据（one-hot 编码）
fn get_xor_data() -> (Vec<Tensor>, Vec<Tensor>) {
    let inputs = vec![
        Tensor::new(&[0.0, 0.0], &[1, 2]),
        Tensor::new(&[0.0, 1.0], &[1, 2]),
        Tensor::new(&[1.0, 0.0], &[1, 2]),
        Tensor::new(&[1.0, 1.0], &[1, 2]),
    ];
    let labels = vec![
        Tensor::new(&[1.0, 0.0], &[1, 2]), // XOR(0,0) = 0
        Tensor::new(&[0.0, 1.0], &[1, 2]), // XOR(0,1) = 1
        Tensor::new(&[0.0, 1.0], &[1, 2]), // XOR(1,0) = 1
        Tensor::new(&[1.0, 0.0], &[1, 2]), // XOR(1,1) = 0
    ];
    (inputs, labels)
}

fn main() -> Result<(), GraphError> {
    println!("=== XOR 异或问题示例 ===\n");

    // 1. 创建模型
    let graph = Graph::new();
    let model = XorMLP::new(&graph)?;

    // 2. 输入/输出占位符（用于迭代更新数据）
    // 也可用 graph.input(&data) 一次性输入 batch 数据，详见 sine_regression 示例
    let x = graph.zeros(&[1, 2])?;
    let target = graph.zeros(&[1, 2])?;

    // 3. 前向传播 + 损失
    let logits = model.forward(&x);
    let loss = logits.cross_entropy(&target)?;

    // 4. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.1);

    // 5. 数据
    let (inputs, labels) = get_xor_data();

    println!("网络: Input(2) -> Linear(4, Tanh) -> Linear(2)");
    println!("优化器: Adam, 损失: CrossEntropy\n");

    // 6. 训练
    for epoch in 0..100 {
        for (input, label) in inputs.iter().zip(labels.iter()) {
            x.set_value(input)?;
            target.set_value(label)?;
            optimizer.zero_grad()?;
            loss.backward()?;
            optimizer.step()?;
        }

        // 评估
        let correct = inputs
            .iter()
            .zip(labels.iter())
            .filter(|(inp, lbl)| {
                x.set_value(inp).ok();
                logits.forward().ok();
                let out = logits.value().ok().flatten().unwrap();
                (out[[0, 0]] > out[[0, 1]]) == (lbl[[0, 0]] > lbl[[0, 1]])
            })
            .count();

        if correct == 4 {
            println!("Epoch {:2}: 准确率 100% ✓", epoch + 1);
            break;
        }
    }

    // 7. 结果
    println!("\n=== 预测结果 ===");
    for (input, label) in inputs.iter().zip(labels.iter()) {
        x.set_value(input)?;
        logits.forward()?;
        let out = logits.value()?.unwrap();
        let pred = if out[[0, 0]] > out[[0, 1]] { 0 } else { 1 };
        let expected = if label[[0, 0]] > label[[0, 1]] { 0 } else { 1 };
        println!(
            "  XOR({}, {}) = {} {}",
            input[[0, 0]] as i32,
            input[[0, 1]] as i32,
            pred,
            if pred == expected { "✓" } else { "✗" }
        );
    }

    println!("\n✅ 训练成功！");
    Ok(())
}
