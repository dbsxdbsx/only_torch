//! # California Housing 房价回归示例（PyTorch 风格）
//!
//! 展示 MSE 损失在真实数据集上的回归任务：
//! - 使用 California Housing 数据集
//! - 使用 Linear 层 + Softplus 激活
//! - 使用 `MseLoss`（`PyTorch` 风格）
//! - 使用 `DataLoader` 批处理
//!
//! ## 运行
//! ```bash
//! cargo run --example california_housing
//! ```
//!
//! ## 目标
//! R² ≥ 70%（解释 70% 的目标变量方差）

mod model;

use model::CaliforniaHousingMLP;
use only_torch::data::{CaliforniaHousingDataset, DataLoader, TensorDataset};
use only_torch::metrics::{IntoFloatValues, r2_score};
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;
use std::time::Instant;

fn main() -> Result<(), GraphError> {
    println!("=== California Housing 房价回归（PyTorch 风格）===\n");

    let start_time = Instant::now();

    // ========== 1. 加载数据 ==========
    println!("[1/4] 加载数据集...");

    let dataset = CaliforniaHousingDataset::load_default()
        .expect("加载 California Housing 数据集失败")
        .standardize();

    let (train_data, test_data) = dataset
        .train_test_split(0.2, Some(42))
        .expect("划分数据集失败");

    println!(
        "  训练集: {} 样本, 测试集: {} 样本",
        train_data.len(),
        test_data.len()
    );

    // 转换为 TensorDataset
    let (train_x, train_y) = to_tensor_dataset(&train_data);
    let (test_x, test_y) = to_tensor_dataset(&test_data);

    let train_dataset = TensorDataset::new(train_x, train_y);
    let test_dataset = TensorDataset::new(test_x, test_y);

    // ========== 2. 训练配置 ==========
    let batch_size = 256;
    let max_epochs = 30;
    let learning_rate = 0.01;
    let target_r2 = 0.70;

    println!("\n[2/4] 训练配置:");
    println!("  Batch Size: {batch_size}");
    println!("  Max Epochs: {max_epochs}");
    println!("  学习率: {learning_rate}");
    println!("  目标 R²: {:.0}%", target_r2 * 100.0);

    // ========== 3. 构建模型 ==========
    println!("\n[3/4] 构建模型: 8 -> 128 -> 64 -> 32 -> 1");

    let graph = Graph::new_with_seed(42);
    let model = CaliforniaHousingMLP::new(&graph)?;

    println!("  参数数量: {} 个 Var", model.parameters().len());

    // 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), learning_rate);

    // DataLoader
    let train_loader = DataLoader::new(train_dataset, batch_size)
        .shuffle(true)
        .drop_last(true);

    let test_loader = DataLoader::new(test_dataset, batch_size)
        .shuffle(false)
        .drop_last(true);

    // ========== 4. 训练循环 ==========
    println!("\n[4/4] 开始训练...\n");

    let mut best_r2 = f32::NEG_INFINITY;

    for epoch in 0..max_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss_sum = 0.0;
        let mut batch_count = 0;

        // 训练
        for (x_batch, y_batch) in train_loader.iter() {
            let output = model.forward(&x_batch)?;
            let loss = output.mse_loss(&y_batch)?;

            graph.snapshot_once_from(&[&loss]);

            optimizer.zero_grad()?;
            let loss_val = loss.backward()?;
            optimizer.step()?;

            epoch_loss_sum += loss_val;
            batch_count += 1;
        }

        let epoch_avg_loss = epoch_loss_sum / batch_count as f32;

        // 测试集评估（计算 R²）
        let r2_score = evaluate_r2(&model, &test_loader)?;
        best_r2 = best_r2.max(r2_score);

        println!(
            "Epoch {:2}/{}: loss = {:.4}, R² = {:.2}% ({:.4}), 耗时 {:.2}s",
            epoch + 1,
            max_epochs,
            epoch_avg_loss,
            r2_score * 100.0,
            r2_score,
            epoch_start.elapsed().as_secs_f32()
        );

        // 提前结束
        if r2_score >= target_r2 {
            println!("\n🎉 达到目标 R² ≥ {:.0}%！", target_r2 * 100.0);
            break;
        }
    }

    // ========== 保存可视化（从训练时拍的快照渲染）==========
    let vis_result =
        graph.visualize_snapshot("examples/traditional/california_housing/california_housing")?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    // ========== 结果总结 ==========
    let total_duration = start_time.elapsed();
    println!("\n总耗时: {:.2}s", total_duration.as_secs_f32());
    println!("最佳 R²: {:.4} ({:.1}%)", best_r2, best_r2 * 100.0);

    if best_r2 >= target_r2 {
        println!("\n✅ California Housing 回归成功！");
        println!("   模型解释了 {:.1}% 的目标变量方差", best_r2 * 100.0);
        Ok(())
    } else {
        println!("\n❌ 未达到目标 R² (实际: {:.1}%)", best_r2 * 100.0);
        Err(GraphError::ComputationError(format!(
            "R² 分数 {:.2}% < 目标 {:.0}%",
            best_r2 * 100.0,
            target_r2 * 100.0
        )))
    }
}

/// 将 `CaliforniaHousingDataset` 转换为 (Tensor, Tensor)
fn to_tensor_dataset(data: &CaliforniaHousingDataset) -> (Tensor, Tensor) {
    let n = data.len();
    let mut x_data = Vec::with_capacity(n * 8);
    let mut y_data = Vec::with_capacity(n);

    for i in 0..n {
        let (features, target) = data.get(i).unwrap();
        x_data.extend(features.flatten_view().iter().copied());
        y_data.push(target[[0]]);
    }

    (Tensor::new(&x_data, &[n, 8]), Tensor::new(&y_data, &[n, 1]))
}

/// 在测试集上计算 R² 分数
fn evaluate_r2(
    model: &CaliforniaHousingMLP,
    loader: &DataLoader<TensorDataset>,
) -> Result<f32, GraphError> {
    let mut predictions = Vec::new();
    let mut actuals = Vec::new();

    for (x_batch, y_batch) in loader.iter() {
        let output = model.forward(&x_batch)?;
        let pred = output.value()?.unwrap();

        // 直接用 trait 方法批量提取数据
        predictions.extend(pred.to_float_values());
        actuals.extend(y_batch.to_float_values());
    }

    Ok(r2_score(&predictions, &actuals).value())
}
