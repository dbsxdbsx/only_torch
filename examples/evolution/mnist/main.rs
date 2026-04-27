/*
 * @Author       : 老董
 * @Date         : 2026-03-11
 * @Description  : MNIST 神经架构演化示例（零模型代码）
 *
 * 与 `examples/mnist`（手动定义 MLP）和 `examples/mnist_cnn`（手动定义 LeNet）不同，
 * 本示例展示 **Evolution API**——只提供图像数据和目标，
 * 系统从 `Input(1@28×28) → Conv2d(1→8,k=3) → Pool2d(Max,2,2) → Flatten → [Linear(10)]` 出发，
 * 通过自动变异发现能识别手写数字的架构。
 *
 * 演化可探索 Conv-BN-ReLU 组合、多层卷积、stride 降维等 CNN 架构——
 * 最终由 fitness（准确率）驱动选择。
 *
 * 关键特性：
 * - 空间输入 [1, 28, 28]（灰度图）→ 自动推断空间模式
 * - 十分类 → 自动推断 CrossEntropy loss + argmax accuracy
 * - Conv-BN-ReLU 模板：60% 概率插入带 BatchNorm 的卷积块
 * - Lamarckian 权重继承：每代在上一代权重基础上继续训练
 * - 变异：InsertLayer（Conv2d/Conv-BN-ReLU/Pool2d/Linear/Activation）、MutateStride 等
 *
 * ## 运行
 * ```bash
 * cargo run --example evolution_mnist
 * ```
 *
 * ## 数据集
 * 首次运行会自动下载 MNIST 数据集到 `~/.cache/only_torch/datasets/mnist/`
 *
 * ## 性能参考（debug 模式）
 * - 1000 训练样本 + 500 测试样本
 * - 目标 95% 准确率，最多 100 代
 * - 预计耗时：数分钟（取决于 CPU）
 */

use only_torch::data::MnistDataset;
use only_torch::nn::evolution::gene::TaskMetric;
use only_torch::nn::evolution::{Evolution, EvolutionResult};
use only_torch::tensor::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::path::Path;
use std::time::Instant;

/// 从 MnistDataset 中取前 n 个样本，转为 per-sample Vec<Tensor>
fn collect_samples(dataset: &MnistDataset, n: usize, seed: u64) -> (Vec<Tensor>, Vec<Tensor>) {
    let n = n.min(dataset.len());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..dataset.len()).collect();
    indices.shuffle(&mut rng);
    indices.truncate(n);
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in indices {
        let (image, label) = dataset.get(i).expect("MNIST 样本读取失败");
        inputs.push(image); // [1, 28, 28]
        labels.push(label); // [10]
    }
    (inputs, labels)
}

fn main() {
    let total_start = Instant::now();
    println!("=== MNIST 神经架构演化示例 ===\n");

    // 1. 加载 MNIST 数据集
    println!("[1/3] 加载 MNIST 数据集...");
    let load_start = Instant::now();
    let train_dataset = MnistDataset::train().expect("加载 MNIST 训练集失败（首次运行会自动下载）");
    let test_dataset = MnistDataset::test().expect("加载 MNIST 测试集失败");
    println!(
        "  ✓ 训练集: {} 样本，测试集: {} 样本 ({:.1}s)",
        train_dataset.len(),
        test_dataset.len(),
        load_start.elapsed().as_secs_f32()
    );

    // 2. 准备数据子集
    let train_samples = 1000;
    let test_samples = 500;
    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4);
    let population_size = 12;
    let offspring_batch_size = 16;
    let train_data = collect_samples(&train_dataset, train_samples, 42);
    let test_data = collect_samples(&test_dataset, test_samples, 43);

    println!("\n[2/3] 配置：");
    println!("  - 训练样本: {train_samples}（mini-batch, auto batch_size=64）");
    println!("  - 测试样本: {test_samples}");
    println!("  - 输入: [1, 28, 28]（灰度图）");
    println!("  - 输出: 10 类（数字 0-9）");
    println!("  - 起始结构: Input(1@28×28) → Conv2d(1→8,k=3) → Pool2d → Flatten → [Linear(10)]");
    println!("  - population_size: {population_size}");
    println!("  - offspring_batch_size: {offspring_batch_size}");
    println!("  - parallelism: {parallelism}");
    println!("  - 目标准确率: ≥95%\n");

    // 3. Evolution API：只需提供数据、指标、目标——零模型代码
    println!("[3/3] 开始演化...\n");
    let evo_start = Instant::now();

    let result = Evolution::supervised(train_data, test_data, TaskMetric::Accuracy)
        .with_target_metric(0.95)
        .with_max_generations(60)
        .with_population_size(population_size)
        .with_offspring_batch_size(offspring_batch_size)
        .with_parallelism(parallelism)
        .with_stagnation_patience(8)
        .with_pareto_patience(16)
        .with_batch_size(64)
        .with_seed(42)
        .run()
        .expect("演化过程出错");

    let evo_duration = evo_start.elapsed();

    // 4. 结果
    println!("\n=== 演化结果 ===");
    println!("状态: {:?}", result.status);
    println!("代数: {}", result.generations);
    println!("准确率: {:.1}%", result.fitness.primary * 100.0);
    println!("最终架构: {}", result.architecture());
    println!("演化耗时: {:.1}s", evo_duration.as_secs_f32());

    // 5. 推理验证
    let (sample, _label) = test_dataset.get(0).unwrap();
    let pred = result.predict(&sample).expect("推理失败");
    let pred_vec = pred.to_vec();
    let predicted_digit = pred_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    println!("\n第一张测试图预测: 数字 {predicted_digit}");

    // 6. 可视化
    let vis = result
        .visualize("examples/evolution/mnist/evolution_mnist")
        .expect("可视化失败");
    println!("计算图已保存: {}", vis.dot_path.display());
    if let Some(img) = &vis.image_path {
        println!("可视化图像: {}", img.display());
    }

    // 7. 模型保存/加载
    let model_path = "examples/evolution/mnist/mnist_model";
    result.save(model_path).expect("保存模型失败");
    println!("\n模型已保存: {model_path}.otm");

    let loaded = EvolutionResult::load(model_path).expect("加载模型失败");
    let pred_loaded = loaded.predict(&sample).expect("加载后推理失败");
    let loaded_digit = pred_loaded
        .to_vec()
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    println!("从磁盘加载后预测: 数字 {loaded_digit}");

    // 清理临时模型文件
    let _ = std::fs::remove_file(Path::new(model_path).with_extension("otm"));

    println!(
        "\n✅ 系统自动发现了 MNIST 手写数字识别架构！总耗时: {:.1}s",
        total_start.elapsed().as_secs_f32()
    );
}
