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
 * - Conv / BatchNorm / Activation 层块可由演化自主插入
 * - Lamarckian 权重继承：每代在上一代权重基础上继续训练
 * - 变异：InsertLayer（Conv2d/Pool2d/Linear/Activation/Normalization）、MutateStride 等
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
use only_torch::nn::evolution::{
    CandidateScoringConfig, Evolution, EvolutionResult, FinalRefitConfig, InitialPortfolioConfig,
    TaskMetric,
};
use only_torch::tensor::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::env;
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy)]
struct MnistEvolutionProfile {
    name: &'static str,
    train_samples: usize,
    test_samples: usize,
    target_metric: f32,
    max_generations: usize,
    population_size: usize,
    offspring_batch_size: usize,
    batch_size: usize,
    stagnation_patience: usize,
    pareto_patience: usize,
    max_parallelism: usize,
    initial_burst: usize,
    final_refit: Option<FinalRefitConfig>,
    max_inference_cost: Option<f32>,
    initial_portfolio: Option<InitialPortfolioConfig>,
    candidate_scoring: Option<CandidateScoringConfig>,
    save_artifacts: bool,
}

impl MnistEvolutionProfile {
    fn from_args() -> Self {
        let profile_name = env::args()
            .find_map(|arg| arg.strip_prefix("--profile=").map(str::to_string))
            .or_else(|| env::var("ONLY_TORCH_MNIST_EVOLUTION_PROFILE").ok())
            .unwrap_or_else(|| "quality".to_string());

        match profile_name.as_str() {
            "smoke" => Self::smoke(),
            "quality" => Self::quality(),
            "search" => Self::search(),
            "demo" => Self::demo(),
            other => {
                eprintln!(
                    "未知 profile `{other}`，可选值：smoke / demo / quality / search；已回退到 quality"
                );
                Self::quality()
            }
        }
    }

    fn smoke() -> Self {
        Self {
            name: "smoke",
            train_samples: 128,
            test_samples: 64,
            target_metric: 0.25,
            max_generations: 3,
            population_size: 4,
            offspring_batch_size: 4,
            batch_size: 32,
            stagnation_patience: 3,
            pareto_patience: 4,
            max_parallelism: 4,
            initial_burst: 1,
            final_refit: None,
            max_inference_cost: None,
            initial_portfolio: None,
            candidate_scoring: None,
            save_artifacts: false,
        }
    }

    fn demo() -> Self {
        Self {
            name: "demo",
            train_samples: 512,
            test_samples: 200,
            target_metric: 0.75,
            max_generations: 20,
            population_size: 8,
            offspring_batch_size: 8,
            batch_size: 64,
            stagnation_patience: 6,
            pareto_patience: 8,
            max_parallelism: 6,
            initial_burst: 2,
            final_refit: None,
            max_inference_cost: None,
            initial_portfolio: Some(InitialPortfolioConfig::vision_classification()),
            candidate_scoring: None,
            save_artifacts: true,
        }
    }

    fn quality() -> Self {
        Self {
            name: "quality",
            train_samples: 15000,
            test_samples: 1000,
            target_metric: 0.95,
            max_generations: 2,
            population_size: 1,
            offspring_batch_size: 1,
            batch_size: 256,
            stagnation_patience: 2,
            pareto_patience: 2,
            max_parallelism: 1,
            initial_burst: 0,
            final_refit: None,
            max_inference_cost: Some(3_000_000.0),
            initial_portfolio: Some(InitialPortfolioConfig::flat_mlp_only(128)),
            candidate_scoring: None,
            save_artifacts: true,
        }
    }

    fn search() -> Self {
        Self {
            name: "search",
            train_samples: 5000,
            test_samples: 1000,
            target_metric: 0.95,
            max_generations: 30,
            population_size: 8,
            offspring_batch_size: 8,
            batch_size: 128,
            stagnation_patience: 8,
            pareto_patience: 12,
            max_parallelism: 8,
            initial_burst: 8,
            final_refit: Some(FinalRefitConfig::new(0.02, 2, 12)),
            max_inference_cost: Some(3_000_000.0),
            initial_portfolio: Some(InitialPortfolioConfig::vision_classification()),
            candidate_scoring: Some(CandidateScoringConfig::p5_lite()),
            save_artifacts: true,
        }
    }
}

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

fn describe_initial_portfolio(config: InitialPortfolioConfig) -> String {
    let mut parts = Vec::new();
    if config.include_flat_mlp {
        parts.push(format!("FlatMLP(hidden={})", config.flat_mlp_hidden));
    }
    if config.include_tiny_cnn {
        parts.push("TinyCNN".to_string());
    }
    if config.include_lenet_tiny {
        parts.push("LeNetTiny".to_string());
    }
    if parts.is_empty() {
        "disabled".to_string()
    } else {
        parts.join(" + ")
    }
}

fn main() {
    let total_start = Instant::now();
    println!("=== MNIST 神经架构演化示例 ===\n");
    let profile = MnistEvolutionProfile::from_args();

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
    let train_samples = profile.train_samples;
    let test_samples = profile.test_samples;
    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, profile.max_parallelism))
        .unwrap_or(4);
    let population_size = profile.population_size;
    let offspring_batch_size = profile.offspring_batch_size;
    let train_data = collect_samples(&train_dataset, train_samples, 42);
    let test_data = collect_samples(&test_dataset, test_samples, 43);

    println!("\n[2/3] 配置：");
    println!(
        "  - profile: {}（可用 --profile=smoke|demo|quality|search 或 ONLY_TORCH_MNIST_EVOLUTION_PROFILE 覆盖）",
        profile.name
    );
    println!(
        "  - 训练样本: {train_samples}（mini-batch, batch_size={}）",
        profile.batch_size
    );
    println!("  - 测试样本: {test_samples}");
    println!("  - 输入: [1, 28, 28]（灰度图）");
    println!("  - 输出: 10 类（数字 0-9）");
    println!("  - minimal seed: Conv2d(1→8,k=3) → Pool2d → Flatten → Linear(10)");
    println!("  - population_size: {population_size}");
    println!("  - offspring_batch_size: {offspring_batch_size}");
    println!("  - parallelism: {parallelism}");
    println!("  - max_generations: {}", profile.max_generations);
    println!("  - initial_burst: {}", profile.initial_burst);
    if let Some(refit) = profile.final_refit {
        println!(
            "  - final_refit: top_k={}, epochs={}, trigger=target-{:.1}pp",
            refit.top_k,
            refit.epochs,
            refit.trigger_margin * 100.0
        );
    }
    if let Some(max_cost) = profile.max_inference_cost {
        println!("  - max_inference_cost: {max_cost:.0} FLOPs");
    }
    if let Some(portfolio) = profile.initial_portfolio {
        println!(
            "  - initial_portfolio: {}",
            describe_initial_portfolio(portfolio)
        );
    }
    if let Some(scoring) = profile.candidate_scoring {
        println!(
            "  - candidate_scoring: p5-lite pool_multiplier={} top_k={}",
            scoring.pool_multiplier,
            scoring
                .keep_top_k
                .map(|n| n.to_string())
                .unwrap_or_else(|| "offspring_batch_size".to_string())
        );
    }
    println!("  - 目标准确率: ≥{:.0}%\n", profile.target_metric * 100.0);

    // 3. Evolution API：只需提供数据、指标、目标——零模型代码
    println!("[3/3] 开始演化...\n");
    let evo_start = Instant::now();

    let mut evolution = Evolution::supervised(train_data, test_data, TaskMetric::Accuracy)
        .with_target_metric(profile.target_metric)
        .with_max_generations(profile.max_generations)
        .with_population_size(population_size)
        .with_offspring_batch_size(offspring_batch_size)
        .with_parallelism(parallelism)
        .with_stagnation_patience(profile.stagnation_patience)
        .with_pareto_patience(profile.pareto_patience)
        .with_batch_size(profile.batch_size)
        .with_initial_burst(profile.initial_burst)
        .with_final_refit(profile.final_refit)
        .with_seed(42);
    if let Some(max_cost) = profile.max_inference_cost {
        evolution = evolution.with_max_inference_cost(max_cost);
    }
    evolution = evolution
        .with_initial_portfolio(profile.initial_portfolio)
        .with_candidate_scoring(profile.candidate_scoring);

    let result = evolution.run().expect("演化过程出错");

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

    if profile.save_artifacts {
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
    } else {
        println!("profile=smoke：跳过可视化和模型保存/加载，专注快速回归");
    }

    println!(
        "\n✅ 系统自动发现了 MNIST 手写数字识别架构！总耗时: {:.1}s",
        total_start.elapsed().as_secs_f32()
    );
}
