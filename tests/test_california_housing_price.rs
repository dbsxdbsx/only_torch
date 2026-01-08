/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : California Housing 房价回归集成测试
 *
 * 使用真实数据集验证 MSELoss + MLP 回归任务
 * 类似于 MNIST 在分类任务中的地位
 *
 * 采用 Layer API + Batch 模式，与 MNIST 测试风格一致
 */

use approx::assert_abs_diff_eq;
use only_torch::data::CaliforniaHousingDataset;
use only_torch::nn::optimizer::{Adam, Optimizer};
use only_torch::nn::{Graph, GraphError, linear};
use only_torch::tensor::Tensor;
use std::fs;
use std::time::Instant;

/// California Housing 房价回归（Layer API + Batch 版本）
///
/// 网络结构：Input(8) → Linear(128, Softplus) → Linear(64, Softplus) → Linear(32, Softplus) → Linear(1)
/// 目标：R² ≥ 0.70 (70%)
///
/// 设计特点：
/// 1. 使用 `linear()` Layer API 构建网络（简洁、可维护）
/// 2. 真正的 batch `训练（batch_size=256，高效`）
/// 3. Softplus 激活：平滑梯度，无死神经元问题
/// 4. Xavier 初始化：适配 Softplus
///
/// 注：California Housing + MLP 在 batch 模式下 70% R² 是合理目标
/// scikit-learn `MLPRegressor` 在此数据集上也达到类似水平
#[test]
fn test_california_housing_regression() -> Result<(), GraphError> {
    let start_time = Instant::now();

    println!("\n{}", "=".repeat(60));
    println!("=== California Housing 房价回归测试（Layer API + Batch）===");
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
    // Batch 模式：与 MNIST 测试风格一致，使用较大 batch 和相应学习率
    // 注：batch MSE 梯度会被 batch_size 平均，需要更高学习率补偿
    let batch_size = 256;
    let train_samples = 16000; // 使用更多训练数据
    let test_samples = 1024; // 测试样本数（batch 整除）
    let max_epochs = 30;
    let num_batches = train_samples / batch_size; // 62 batches/epoch
    let learning_rate = 0.01; // batch 模式需要更高学习率
    let target_r2 = 0.70; // California Housing + MLP 的合理目标

    println!("\n[2/4] 训练配置：");
    println!("  - Batch Size: {batch_size}");
    println!("  - 训练样本: {train_samples} (共 {num_batches} 个 batch)");
    println!("  - 测试样本: {test_samples}");
    println!("  - 最大 Epochs: {max_epochs}");
    println!("  - 学习率: {learning_rate}");
    println!("  - 目标 R²: {:.0}%", target_r2 * 100.0);

    // ========== 3. 构建网络（使用 Layer API）==========
    println!("\n[3/4] 使用 linear() 构建 MLP: 8 -> 128 -> 64 -> 32 -> 1...");

    let mut graph = Graph::new_with_seed(42);

    // 输入/标签节点（batch 维度）
    let x = graph.new_input_node(&[batch_size, 8], Some("x"))?;
    let y_true = graph.new_input_node(&[batch_size, 1], Some("y_true"))?;

    // 隐藏层1: 8 -> 128 (Softplus)
    let fc1 = linear(&mut graph, x, 8, 128, batch_size, Some("fc1"))?;
    let a1 = graph.new_softplus_node(fc1.output, Some("fc1_act"))?;

    // 隐藏层2: 128 -> 64 (Softplus)
    let fc2 = linear(&mut graph, a1, 128, 64, batch_size, Some("fc2"))?;
    let a2 = graph.new_softplus_node(fc2.output, Some("fc2_act"))?;

    // 隐藏层3: 64 -> 32 (Softplus)
    let fc3 = linear(&mut graph, a2, 64, 32, batch_size, Some("fc3"))?;
    let a3 = graph.new_softplus_node(fc3.output, Some("fc3_act"))?;

    // 输出层: 32 -> 1 (线性)
    let fc4 = linear(&mut graph, a3, 32, 1, batch_size, Some("fc4"))?;
    let y_pred = fc4.output;

    // 损失函数
    let loss = graph.new_mse_loss_node(y_pred, y_true, Some("loss"))?;

    println!("  ✓ 网络构建完成：8 -> 128 -> 64 -> 32 -> 1（4 层 MLP）");
    println!("  ✓ 参数节点：fc1_W/b, fc2_W/b, fc3_W/b, fc4_W/b");

    // 保存网络结构可视化（训练前）
    let output_dir = "tests/outputs";
    fs::create_dir_all(output_dir).ok();
    graph.save_visualization_grouped(format!("{output_dir}/california_housing"), None)?;
    graph.save_summary(format!("{output_dir}/california_housing_summary.md"))?;
    println!("  ✓ 网络结构已保存: {output_dir}/california_housing.png");

    // ========== Xavier 初始化 ==========
    let xavier_init = |fan_in: usize, fan_out: usize, seed: u64| -> Tensor {
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
        Tensor::normal_seeded(0.0, std, &[fan_in, fan_out], seed)
    };

    graph.set_node_value(fc1.weights, Some(&xavier_init(8, 128, 42)))?;
    graph.set_node_value(fc2.weights, Some(&xavier_init(128, 64, 43)))?;
    graph.set_node_value(fc3.weights, Some(&xavier_init(64, 32, 44)))?;
    graph.set_node_value(fc4.weights, Some(&xavier_init(32, 1, 45)))?;

    // bias 初始化为 0
    graph.set_node_value(fc1.bias, Some(&Tensor::zeros(&[1, 128])))?;
    graph.set_node_value(fc2.bias, Some(&Tensor::zeros(&[1, 64])))?;
    graph.set_node_value(fc3.bias, Some(&Tensor::zeros(&[1, 32])))?;
    graph.set_node_value(fc4.bias, Some(&Tensor::zeros(&[1, 1])))?;

    // ========== 4. 训练循环 ==========
    println!("\n[4/4] 开始训练...\n");

    let mut optimizer = Adam::new(&graph, learning_rate, 0.9, 0.999, 1e-8)?;

    // 预先构建 batch 数据
    let train_batches = build_batches(&train_data, batch_size, num_batches);
    let test_batches_data = build_batches(&test_data, batch_size, test_samples / batch_size);

    let mut best_r2 = f32::NEG_INFINITY;

    for epoch in 0..max_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss_sum = 0.0;

        // 训练
        for (batch_x, batch_y) in &train_batches {
            graph.set_node_value(x, Some(batch_x))?;
            graph.set_node_value(y_true, Some(batch_y))?;

            graph.zero_grad()?;
            graph.forward(loss)?;
            let loss_val = graph.backward(loss)?; // backward 返回 loss 值
            optimizer.step(&mut graph)?;

            epoch_loss_sum += loss_val;
        }

        let epoch_avg_loss = epoch_loss_sum / num_batches as f32;

        // 测试集评估（计算 R²）
        graph.set_eval_mode();
        let mut predictions: Vec<f32> = Vec::with_capacity(test_samples);
        let mut actuals: Vec<f32> = Vec::with_capacity(test_samples);

        for (batch_x, batch_y) in &test_batches_data {
            graph.set_node_value(x, Some(batch_x))?;
            graph.set_node_value(y_true, Some(batch_y))?;

            graph.forward(y_pred)?;

            let pred_tensor = graph.get_node_value(y_pred)?.unwrap();
            for i in 0..batch_size {
                predictions.push(pred_tensor[[i, 0]]);
                actuals.push(batch_y[[i, 0]]);
            }
        }

        graph.set_train_mode();

        // 计算 R²
        let r2_score = compute_r2(&predictions, &actuals);
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

        // 提前结束条件
        if r2_score >= target_r2 {
            println!("\n🎉 达到目标 R² ≥ {:.0}%！", target_r2 * 100.0);
            break;
        }
    }

    let total_duration = start_time.elapsed();
    println!("\n总耗时: {:.2}s", total_duration.as_secs_f32());

    // 打印模型摘要
    println!("\n模型摘要：");
    graph.summary();

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
                // 使用 flatten_view() 获取数据视图并复制
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

/// 计算 R² 分数
fn compute_r2(predictions: &[f32], actuals: &[f32]) -> f32 {
    let mean_actual: f32 = actuals.iter().sum::<f32>() / actuals.len() as f32;

    let ss_res: f32 = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(pred, actual)| (actual - pred).powi(2))
        .sum();

    let ss_tot: f32 = actuals
        .iter()
        .map(|actual| (actual - mean_actual).powi(2))
        .sum();

    1.0 - (ss_res / ss_tot)
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
