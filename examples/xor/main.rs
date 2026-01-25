//! # XOR 异或问题示例
//!
//! 展示 `only_torch` 的 **`PyTorch` 风格** API：
//! - `model.forward(&tensor)` - 直接传入 Tensor
//! - `criterion.forward(&output, &target)` - `PyTorch` 风格损失计算
//! - 训练循环与 `PyTorch` 几乎一致
//!
//! ## 运行
//! ```bash
//! cargo run --example xor
//! ```

mod model;

use model::XorMLP;
use only_torch::nn::{Adam, CrossEntropyLoss, Graph, GraphError, Module, Optimizer};
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
    println!("=== XOR 异或问题示例（PyTorch 风格）===\n");

    // 1. 创建模型
    let graph = Graph::new_with_seed(42);
    let model = XorMLP::new(&graph)?;

    // 2. 损失函数（PyTorch 风格）
    let criterion = CrossEntropyLoss::new();

    // 3. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.1);

    // 4. 数据
    let (inputs, labels) = get_xor_data();

    println!("网络: Input(2) -> Linear(4, Tanh) -> Linear(2)");
    println!("优化器: Adam, 损失: CrossEntropyLoss\n");

    // 5. 训练循环（完全 PyTorch 风格！）
    for epoch in 0..100 {
        for (input, label) in inputs.iter().zip(labels.iter()) {
            // PyTorch 风格：直接传 Tensor
            let output = model.forward(input)?;
            let loss = criterion.forward(&output, label)?;

            // 反向传播 + 参数更新
            optimizer.zero_grad()?;
            loss.backward()?;
            optimizer.step()?;
        }

        // 评估
        let correct = inputs
            .iter()
            .zip(labels.iter())
            .filter(|(inp, lbl)| {
                let out = model.forward(inp).unwrap();
                let pred = out.value().ok().flatten().unwrap();
                pred.argmax(1).get_data_number().unwrap()
                    == lbl.argmax(1).get_data_number().unwrap()
            })
            .count();

        if correct == 4 {
            println!("Epoch {:2}: 准确率 100% ✓", epoch + 1);
            break;
        }
    }

    // 6. 结果展示
    println!("\n=== 预测结果 ===");
    for (input, label) in inputs.iter().zip(labels.iter()) {
        let output = model.forward(input)?;
        let pred_tensor = output.value()?.unwrap();
        let pred = pred_tensor.argmax(1).get_data_number().unwrap() as i32;
        let expected = label.argmax(1).get_data_number().unwrap() as i32;
        println!(
            "  XOR({}, {}) = {} {}",
            input[[0, 0]] as i32,
            input[[0, 1]] as i32,
            pred,
            if pred == expected { "✓" } else { "✗" }
        );
    }

    println!("\n✅ 训练成功！");

    // 7. 保存计算图可视化
    let vis_result = graph.save_visualization("examples/xor/xor", None)?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    Ok(())
}
