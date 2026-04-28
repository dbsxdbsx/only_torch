/*
 * @Author       : 老董
 * @Date         : 2026-04-28
 * @Description  : 多输出 / 多头 supervised evolution 示例
 *
 * 本示例展示 P3 第一阶段能力：一个模型共享同一组输入，同时训练两个命名输出 head。
 *
 * - quadrant head：四分类，预测二维点所在象限
 * - radius head：回归，预测点到原点的归一化距离
 *
 * 推理时可以只取需要的 head，避免无意义地读取全部输出。
 *
 * ## 运行
 * ```bash
 * cargo run --example evolution_multi_head_quadrant_radius
 * ```
 */

use only_torch::data::SyntheticRng;
use only_torch::nn::evolution::{
    ConvergenceConfig, Evolution, EvolutionResult, ReportMetric, SupervisedSpec, TaskMetric,
    TrainingBudget,
};
use only_torch::tensor::Tensor;
use std::env;
use std::error::Error;
use std::path::Path;
use std::time::Instant;

const TRAIN_SAMPLES: usize = 64;
const TEST_SAMPLES: usize = 32;
const DEFAULT_EVOLUTION_SEED: u64 = 42;
const TARGET_QUADRANT_ACCURACY: f32 = 0.99;

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let evolution_seed = env_u64(
        "ONLY_TORCH_EVOLUTION_MULTI_HEAD_SEED",
        DEFAULT_EVOLUTION_SEED,
    );
    let target_accuracy = env_f32(
        "ONLY_TORCH_EVOLUTION_MULTI_HEAD_TARGET",
        TARGET_QUADRANT_ACCURACY,
    );
    let save_artifacts = env_bool("ONLY_TORCH_EVOLUTION_MULTI_HEAD_SAVE_ARTIFACTS", false);

    println!("=== Multi-head Quadrant + Radius Evolution 示例 ===\n");
    println!("任务: 共享二维点输入 [x, y]，同时学习分类和回归两个 head");
    println!("quadrant head: 4 类象限分类，primary metric = Accuracy");
    println!("radius head: 归一化半径回归，report metric = R2 / MSE / MAE / RMSE");
    println!("训练样本: {TRAIN_SAMPLES}, 测试样本: {TEST_SAMPLES}");
    println!(
        "演化 seed: {evolution_seed}, target quadrant accuracy: {target_accuracy:.2}（默认偏高，让示例完整展示两个 head 的联合训练）\n"
    );

    let train = generate_dataset(TRAIN_SAMPLES, 2026);
    let test = generate_dataset(TEST_SAMPLES, 4096);

    let spec = SupervisedSpec::new(train.inputs.clone(), test.inputs.clone())
        .head_targets(
            "quadrant",
            train.quadrants.clone(),
            test.quadrants.clone(),
            TaskMetric::Accuracy,
        )
        .head_targets(
            "radius",
            train.radii.clone(),
            test.radii.clone(),
            TaskMetric::R2,
        )
        .primary_head("quadrant")
        .with_head_loss_weight("radius", 2.0)
        .with_head_metric_weight("radius", 0.25)
        .with_head_inference("radius", false);

    let result = Evolution::supervised_task(spec)
        .with_target_metric(target_accuracy)
        .with_report_metrics([
            ReportMetric::Accuracy,
            ReportMetric::R2,
            ReportMetric::MeanSquaredError,
            ReportMetric::MeanAbsoluteError,
            ReportMetric::RootMeanSquaredError,
        ])
        .with_seed(evolution_seed)
        .with_convergence(ConvergenceConfig {
            budget: TrainingBudget::FixedEpochs(80),
            ..Default::default()
        })
        .with_max_generations(3)
        .with_population_size(4)
        .with_offspring_batch_size(4)
        .with_parallelism(1)
        .with_primary_proxy(None)
        .with_asha(None)
        .with_verbose(false)
        .run()?;

    println!("\n=== 演化结果 ===");
    println!("状态: {:?}", result.status);
    println!("代数: {}", result.generations);
    println!(
        "primary quadrant accuracy: {:.1}%",
        result.fitness.primary * 100.0
    );
    println!("最终架构: {}", result.architecture());

    println!("\n=== Per-head metric report ===");
    for head in &result.fitness.head_reports {
        println!(
            "{}: primary={:.3}, report={}",
            head.head_name,
            head.primary,
            head.report.format_compact()
        );
    }

    let sample = Tensor::new(&[0.7, -0.4], &[2]);
    let quadrant_logits = result.predict_head("quadrant", &sample)?;
    let radius_pred = result.predict_head("radius", &sample)?;
    println!("\n=== 单 head 推理 ===");
    println!(
        "point [0.7, -0.4] quadrant={} logits={:?}",
        quadrant_name(argmax(&quadrant_logits.to_vec())),
        quadrant_logits.to_vec()
    );
    println!(
        "point [0.7, -0.4] radius prediction={:.3}, target={:.3}",
        radius_pred.to_vec()[0],
        normalized_radius(0.7, -0.4)
    );

    let named_outputs = result.predict_heads(&["quadrant", "radius"], &sample)?;
    println!("\n=== 多 head 选择性推理 ===");
    for (name, output) in named_outputs {
        println!(
            "{name}: shape={:?}, values={:?}",
            output.shape(),
            output.to_vec()
        );
    }

    let model_path = "examples/evolution/multi_head_quadrant_radius/multi_head_model";
    result.save(model_path)?;
    let loaded = EvolutionResult::load(model_path)?;
    let loaded_radius = loaded.predict_head("radius", &sample)?;
    println!(
        "\n.otm 加载后 radius head 推理: {:.3}",
        loaded_radius.to_vec()[0]
    );
    let _ = std::fs::remove_file(Path::new(model_path).with_extension("otm"));

    if save_artifacts {
        let vis = result.visualize(
            "examples/evolution/multi_head_quadrant_radius/evolution_multi_head_quadrant_radius",
        )?;
        println!("计算图已保存: {}", vis.dot_path.display());
        if let Some(img) = &vis.image_path {
            println!("可视化图像: {}", img.display());
        }
    } else {
        println!(
            "\n已跳过 Graphviz 产物保存，可设置 ONLY_TORCH_EVOLUTION_MULTI_HEAD_SAVE_ARTIFACTS=1 开启"
        );
    }

    println!(
        "\nMulti-head evolution 示例完成，总耗时: {:.1}s",
        start.elapsed().as_secs_f32()
    );
    Ok(())
}

#[derive(Clone)]
struct MultiHeadDataset {
    inputs: Vec<Tensor>,
    quadrants: Vec<Tensor>,
    radii: Vec<Tensor>,
}

fn generate_dataset(n: usize, seed: u64) -> MultiHeadDataset {
    let mut rng = SyntheticRng::new(seed);
    let mut inputs = Vec::with_capacity(n);
    let mut quadrants = Vec::with_capacity(n);
    let mut radii = Vec::with_capacity(n);

    for _ in 0..n {
        let (x, y) = sample_point_away_from_axes(&mut rng);
        inputs.push(Tensor::new(&[x, y], &[2]));
        quadrants.push(Tensor::new(&one_hot_quadrant(x, y), &[4]));
        radii.push(Tensor::new(&[normalized_radius(x, y)], &[1]));
    }

    MultiHeadDataset {
        inputs,
        quadrants,
        radii,
    }
}

fn sample_point_away_from_axes(rng: &mut SyntheticRng) -> (f32, f32) {
    loop {
        let x = rng.f32_range(-1.0..1.0);
        let y = rng.f32_range(-1.0..1.0);
        if x.abs() >= 0.12 && y.abs() >= 0.12 {
            return (x, y);
        }
    }
}

fn one_hot_quadrant(x: f32, y: f32) -> [f32; 4] {
    let mut target = [0.0; 4];
    target[quadrant_index(x, y)] = 1.0;
    target
}

fn quadrant_index(x: f32, y: f32) -> usize {
    match (x >= 0.0, y >= 0.0) {
        (true, true) => 0,
        (false, true) => 1,
        (false, false) => 2,
        (true, false) => 3,
    }
}

fn quadrant_name(index: usize) -> &'static str {
    match index {
        0 => "Q1(+,+)",
        1 => "Q2(-,+)",
        2 => "Q3(-,-)",
        3 => "Q4(+,-)",
        _ => "unknown",
    }
}

fn normalized_radius(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt() / 2.0_f32.sqrt()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}
