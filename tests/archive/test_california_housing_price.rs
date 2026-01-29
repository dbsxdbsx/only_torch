/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @LastEditTime : 2026-01-17
 * @Description  : California Housing 房价回归集成测试
 *
 * 使用真实数据集验证 MSELoss + MLP 回归任务
 * 展示 PyTorch 风格的高层 API：Linear 层 + Module trait
 */

use approx::assert_abs_diff_eq;
use only_torch::data::CaliforniaHousingDataset;
use only_torch::metrics::r2_score;
use only_torch::nn::layer::Linear;
use only_torch::nn::optimizer::{Adam, Optimizer};
use only_torch::nn::{Graph, GraphError, Module, VarActivationOps, VarLossOps};
use only_torch::tensor::Tensor;
use std::fs;
use std::time::Instant;

/// California Housing 房价回归（PyTorch 风格 API）
///
/// 网络结构：Input(8) → Linear(128, Softplus) → Linear(64, Softplus) → Linear(32, Softplus) → Linear(1)
/// 目标：R² ≥ 0.70 (70%)
#[test]
fn test_california_housing_regression() -> Result<(), GraphError> {
    let start_time = Instant::now();

    println!("\n{}", "=".repeat(60));
    println!("=== California Housing 房价回归测试 ===");
    println!("{}\n", "=".repeat(60));

    // ========== 1. 加载数据 ==========
    println!("[1/4] 加载 California Housing 数据集...");
    let load_start = Instant::now();

    let dataset = CaliforniaHousingDataset::load_default()
        .expect("加载 California Housing 数据集失败")
        .standardize();

    let (train_data, test_data) = dataset
        .train_test_split(0.2, Some(42))
        .expect("划分数据集失败");

    println!(
        "  ✓ 训练集: {} 样本，测试集: {} 样本，耗时 {:.2}s",
        train_data.len(),
        test_data.len(),
        load_start.elapsed().as_secs_f32()
    );

    // ========== 2. 训练配置 ==========
    let batch_size = 256;
    let train_samples = 16000;
    let test_samples = 1024;
    let max_epochs = 30;
    let num_batches = train_samples / batch_size;
    let learning_rate = 0.01;
    let target_r2 = 0.70;

    println!("\n[2/4] 训练配置：");
    println!("  - Batch Size: {batch_size}");
    println!("  - 训练样本: {train_samples} (共 {num_batches} 个 batch)");
    println!("  - 测试样本: {test_samples}");
    println!("  - 最大 Epochs: {max_epochs}");
    println!("  - 学习率: {learning_rate}");
    println!("  - 目标 R²: {:.0}%", target_r2 * 100.0);

    // ========== 3. 构建网络（使用 Linear 层）==========
    println!("\n[3/4] 构建 MLP: 8 -> 128 -> 64 -> 32 -> 1...");

    let graph = Graph::new_with_seed(42);

    // 输入/标签占位符
    let x = graph.zeros(&[batch_size, 8])?;
    let y_true = graph.zeros(&[batch_size, 1])?;

    // 使用 Linear 层构建网络（PyTorch 风格）
    let fc1 = Linear::new_seeded(&graph, 8, 128, true, "fc1", 100)?;
    let fc2 = Linear::new_seeded(&graph, 128, 64, true, "fc2", 200)?;
    let fc3 = Linear::new_seeded(&graph, 64, 32, true, "fc3", 300)?;
    let fc4 = Linear::new_seeded(&graph, 32, 1, true, "fc4", 400)?;

    // Xavier 初始化（适合 Softplus 激活）
    let xavier_init = |fan_in: usize, fan_out: usize, seed: u64| -> Tensor {
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
        Tensor::normal_seeded(0.0, std, &[fan_in, fan_out], seed)
    };
    fc1.weights().set_value(&xavier_init(8, 128, 42))?;
    fc2.weights().set_value(&xavier_init(128, 64, 43))?;
    fc3.weights().set_value(&xavier_init(64, 32, 44))?;
    fc4.weights().set_value(&xavier_init(32, 1, 45))?;

    // 前向传播链
    let a1 = fc1.forward(&x).softplus();
    let a2 = fc2.forward(&a1).softplus();
    let a3 = fc3.forward(&a2).softplus();
    let y_pred = fc4.forward(&a3);

    // 损失函数
    let loss = y_pred.mse_loss(&y_true)?;

    println!("  ✓ 网络构建完成：8 -> 128 -> 64 -> 32 -> 1（4 层 MLP）");

    // 收集所有参数（使用 Module trait）
    let mut params: Vec<_> = fc1.parameters();
    params.extend(fc2.parameters());
    params.extend(fc3.parameters());
    params.extend(fc4.parameters());
    println!("  ✓ 参数节点：{} 个 Var", params.len());

    // 保存网络结构可视化
    let output_dir = "tests/outputs";
    fs::create_dir_all(output_dir).ok();
    graph
        .inner()
        .save_visualization(format!("{output_dir}/california_housing"), None)?;
    graph
        .inner()
        .save_summary(format!("{output_dir}/california_housing_summary.md"))?;
    println!("  ✓ 网络结构已保存: {output_dir}/california_housing.png");

    // ========== 4. 训练循环 ==========
    println!("\n[4/4] 开始训练...\n");

    let mut optimizer = Adam::new(&graph, &params, learning_rate);

    // 预先构建 batch 数据
    let train_batches = build_batches(&train_data, batch_size, num_batches);
    let test_batches_data = build_batches(&test_data, batch_size, test_samples / batch_size);

    let mut best_r2 = f32::NEG_INFINITY;

    for epoch in 0..max_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss_sum = 0.0;

        // 训练
        for (batch_x, batch_y) in &train_batches {
            x.set_value(batch_x)?;
            y_true.set_value(batch_y)?;

            let loss_val = optimizer.minimize(&loss)?;
            epoch_loss_sum += loss_val;
        }

        let epoch_avg_loss = epoch_loss_sum / num_batches as f32;

        // 测试集评估（计算 R²）
        graph.eval();
        let mut predictions: Vec<f32> = Vec::with_capacity(test_samples);
        let mut actuals: Vec<f32> = Vec::with_capacity(test_samples);

        for (batch_x, batch_y) in &test_batches_data {
            x.set_value(batch_x)?;
            y_true.set_value(batch_y)?;

            y_pred.forward()?;

            let pred_tensor = y_pred.value()?.unwrap();
            for i in 0..batch_size {
                predictions.push(pred_tensor[[i, 0]]);
                actuals.push(batch_y[[i, 0]]);
            }
        }

        graph.train();

        // 计算 R²
        let r2_val = r2_score(&predictions, &actuals);
        best_r2 = best_r2.max(r2_val);

        println!(
            "Epoch {:2}/{}: loss = {:.4}, R² = {:.2}% ({:.4}), 耗时 {:.2}s",
            epoch + 1,
            max_epochs,
            epoch_avg_loss,
            r2_val * 100.0,
            r2_val,
            epoch_start.elapsed().as_secs_f32()
        );

        // 提前结束条件
        if r2_val >= target_r2 {
            println!("\n🎉 达到目标 R² ≥ {:.0}%！", target_r2 * 100.0);
            break;
        }
    }

    let total_duration = start_time.elapsed();
    println!("\n总耗时: {:.2}s", total_duration.as_secs_f32());

    // 打印模型摘要
    println!("\n模型摘要：");
    graph.inner().summary();

    // 最终验证
    println!("\n{}", "=".repeat(60));
    println!("结果验证:");
    println!("  最佳 R²: {:.4} ({:.1}%)", best_r2, best_r2 * 100.0);

    assert!(
        best_r2 >= target_r2,
        "R² 分数应 ≥ {:.0}%，实际: {:.2}%",
        target_r2 * 100.0,
        best_r2 * 100.0
    );

    println!("\n✅ California Housing 回归测试通过！");
    println!("   模型解释了 {:.1}% 的目标变量方差", best_r2 * 100.0);
    println!("{}\n", "=".repeat(60));

    if best_r2 >= 0.85 {
        println!("   🎉 达到优秀水平 (R² ≥ 85%)！");
    }

    Ok(())
}

/// 构建 batch 数据
fn build_batches(
    data: &CaliforniaHousingDataset,
    batch_size: usize,
    num_batches: usize,
) -> Vec<(Tensor, Tensor)> {
    let mut batches = Vec::with_capacity(num_batches);

    for batch_idx in 0..num_batches {
        let mut x_data = Vec::with_capacity(batch_size * 8);
        let mut y_data = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let idx = batch_idx * batch_size + i;
            if idx < data.len() {
                let (features, target) = data.get(idx).unwrap();
                x_data.extend(features.flatten_view().iter().copied());
                y_data.push(target[[0]]);
            }
        }

        let x_tensor = Tensor::new(&x_data, &[batch_size, 8]);
        let y_tensor = Tensor::new(&y_data, &[batch_size, 1]);
        batches.push((x_tensor, y_tensor));
    }

    batches
}

/// 简单验证数据集加载
#[test]
fn test_california_housing_data_loading() {
    let dataset = CaliforniaHousingDataset::load_default().expect("加载数据集失败");

    // 验证数据集大小
    assert!(
        dataset.len() > 20000,
        "数据集应有 20000+ 样本，实际: {}",
        dataset.len()
    );

    // 验证特征维度
    assert_eq!(dataset.feature_dim(), 8);

    // 验证可以获取单个样本
    let (features, target) = dataset.get(0).expect("获取样本失败");
    assert_eq!(features.shape(), &[8]);
    assert_eq!(target.shape(), &[1]);

    // 验证标准化
    let standardized = dataset.standardize();
    assert!(standardized.is_standardized());

    println!("✅ 数据加载测试通过！");
}

/// 验证训练集/测试集划分
#[test]
fn test_california_housing_train_test_split() {
    let dataset = CaliforniaHousingDataset::load_default()
        .expect("加载数据集失败")
        .standardize();

    let total_len = dataset.len();
    let (train, test) = dataset.train_test_split(0.2, Some(42)).expect("划分失败");

    // 验证划分比例
    let expected_test_size = (total_len as f32 * 0.2).round() as usize;
    assert_eq!(test.len(), expected_test_size);
    assert_eq!(train.len(), total_len - expected_test_size);

    // 验证划分确定性
    let dataset2 = CaliforniaHousingDataset::load_default()
        .unwrap()
        .standardize();
    let (train2, _) = dataset2.train_test_split(0.2, Some(42)).unwrap();

    // 相同种子应得到相同的训练集第一个样本
    let (f1, t1) = train.get(0).unwrap();
    let (f2, t2) = train2.get(0).unwrap();

    assert_abs_diff_eq!(f1[[0]], f2[[0]], epsilon = 1e-6);
    assert_abs_diff_eq!(t1[[0]], t2[[0]], epsilon = 1e-6);

    println!("✅ 训练集/测试集划分测试通过！");
}
