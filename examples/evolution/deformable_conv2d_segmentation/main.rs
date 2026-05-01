/*
 * @Author       : 老董
 * @Date         : 2026-04-28
 * @Description  : DeformableConv2d 分割演化 benchmark
 */

use only_torch::data::SyntheticRng;
use only_torch::metrics::{binary_iou, dice_score, pixel_accuracy};
use only_torch::nn::evolution::{
    CandidateScoringConfig, Evolution, EvolutionResult, InitialPortfolioConfig, ReportMetric,
    TaskMetric,
};
use only_torch::tensor::Tensor;
use only_torch::vision::viz::pixel_block_scale;
use std::env;
use std::error::Error;
use std::time::Instant;

const IMAGE_SIZE: usize = 16;
const MAX_OBJECTS: usize = 3;
const TRAIN_SAMPLES: usize = 12;
const TEST_SAMPLES: usize = 4;
const BATCH_SIZE: usize = 4;
const OVERLAY_SCALE: u32 = 10;
const TARGET_BINARY_IOU: f32 = 0.35;
const DEFAULT_EVOLUTION_SEED: u64 = 42;

fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now();
    let evolution_seed = env_u64(
        "ONLY_TORCH_EVOLUTION_DEFORMABLE_SEG_SEED",
        DEFAULT_EVOLUTION_SEED,
    );
    let target_binary_iou = env_f32(
        "ONLY_TORCH_EVOLUTION_DEFORMABLE_SEG_TARGET",
        TARGET_BINARY_IOU,
    );
    let save_artifacts = env_bool("ONLY_TORCH_EVOLUTION_DEFORMABLE_SEG_SAVE_ARTIFACTS", true);
    let audit_mode = env_bool("ONLY_TORCH_EVOLUTION_DEFORMABLE_SEG_AUDIT", false);

    println!("=== DeformableConv2d Semantic Segmentation Evolution 示例 ===\n");
    println!("任务: 16x16 合成图像二值前景分割（1..3 个可重叠形状）");
    println!("输入: [1, {IMAGE_SIZE}, {IMAGE_SIZE}]，标签: [1, {IMAGE_SIZE}, {IMAGE_SIZE}]");
    println!("指标: BinaryIoU，报告指标: PixelAccuracy / BinaryIoU / Dice");
    println!("演化 seed: {evolution_seed}, target Binary IoU: {target_binary_iou:.2}\n");

    if audit_mode {
        run_audit_matrix(evolution_seed, target_binary_iou, save_artifacts)?;
    } else {
        run_smoke_case(evolution_seed, target_binary_iou, save_artifacts)?;
    }

    println!(
        "\nEvolution DeformableConv2d segmentation 示例完成，总耗时: {:.1}s",
        total_start.elapsed().as_secs_f32()
    );

    Ok(())
}

fn run_smoke_case(
    evolution_seed: u64,
    target_binary_iou: f32,
    save_artifacts: bool,
) -> Result<(), Box<dyn Error>> {
    println!(
        "演化策略: 关闭启发式预筛；用 deformable 初始族直接验证 DeformableConv2d 进入演化主流程"
    );
    println!("训练样本: {TRAIN_SAMPLES}, 测试样本: {TEST_SAMPLES}, batch: {BATCH_SIZE}");

    let train = generate_dataset(TRAIN_SAMPLES, 42);
    let test = generate_dataset(TEST_SAMPLES, 2026);
    let case = AuditCase::smoke_deformable_only();
    let (result, report, _elapsed_secs) =
        run_case(&case, &train, &test, evolution_seed, target_binary_iou)?;

    println!("\n=== 演化结果 ===");
    print_result_summary(&result, &report);

    if save_artifacts {
        let pred = result.predict(&test.inputs[report.worst_idx])?;
        save_sample_visualizations(&test, report.worst_idx, &pred, "test")?;
        let vis = result.visualize(
            "examples/evolution/deformable_conv2d_segmentation/evolution_deformable_conv2d_segmentation",
        )?;
        println!("\n计算图已保存: {}", vis.dot_path.display());
        if let Some(img) = &vis.image_path {
            println!("可视化图像: {}", img.display());
        }
    } else {
        println!("\n已跳过图片和 Graphviz 产物保存");
    }

    Ok(())
}

fn run_audit_matrix(
    evolution_seed: u64,
    target_binary_iou: f32,
    save_artifacts: bool,
) -> Result<(), Box<dyn Error>> {
    println!("审计模式: 对比候选族、heuristic 开关和小幅预算提升后的质量指标");
    println!(
        "审计日志会输出 p5-lite-family / eval-family，用于判断候选是否生成、保留并真实评估。\n"
    );

    let mut rows = Vec::new();
    for case in audit_cases() {
        println!("\n=== Audit case: {} ===", case.name);
        println!("{}", case.description);
        println!(
            "portfolio={}, heuristic={}, train={}, test={}, gen={}, pop={}, off={}",
            case.portfolio_label,
            case.heuristic_label(),
            case.train_samples,
            case.test_samples,
            case.max_generations,
            case.population_size,
            case.offspring_batch_size
        );

        let train = generate_dataset(case.train_samples, 42);
        let test = generate_dataset(case.test_samples, 2026);
        let (result, report, elapsed_secs) =
            run_case(&case, &train, &test, evolution_seed, target_binary_iou)?;

        print_result_summary(&result, &report);
        if save_artifacts {
            let pred = result.predict(&test.inputs[report.worst_idx])?;
            save_sample_visualizations(&test, report.worst_idx, &pred, case.name)?;
        }

        rows.push(AuditRow {
            name: case.name,
            portfolio_label: case.portfolio_label,
            heuristic: case.heuristic_label(),
            train_samples: case.train_samples,
            test_samples: case.test_samples,
            max_generations: case.max_generations,
            population_size: case.population_size,
            offspring_batch_size: case.offspring_batch_size,
            status: format!("{:?}", result.status),
            primary: result.fitness.primary,
            iou_min: report.iou_min,
            iou_mean: report.iou_mean,
            iou_max: report.iou_max,
            dice_mean: report.dice_mean,
            pixel_accuracy_mean: report.pixel_accuracy_mean,
            worst_idx: report.worst_idx,
            has_deformable: result.has_deformable_conv2d(),
            elapsed_secs,
        });
    }

    print_audit_matrix(&rows);
    print_heuristic_decision_hint(&rows);
    Ok(())
}

fn run_case(
    case: &AuditCase,
    train: &SegmentationDataset,
    test: &SegmentationDataset,
    evolution_seed: u64,
    target_binary_iou: f32,
) -> Result<(EvolutionResult, TestSetReport, f32), Box<dyn Error>> {
    let started = Instant::now();
    let train_data = (train.inputs.clone(), train.labels.clone());
    let test_data = (test.inputs.clone(), test.labels.clone());

    let mut evolution = Evolution::supervised(train_data, test_data, TaskMetric::BinaryIoU)
        .with_initial_portfolio(case.portfolio)
        .with_initial_burst(case.initial_burst)
        .with_target_metric(target_binary_iou)
        .with_report_metrics([
            ReportMetric::PixelAccuracy,
            ReportMetric::BinaryIoU,
            ReportMetric::Dice,
        ])
        .with_seed(evolution_seed)
        .with_max_generations(case.max_generations)
        .with_population_size(case.population_size)
        .with_offspring_batch_size(case.offspring_batch_size)
        .with_parallelism(1)
        .with_batch_size(BATCH_SIZE)
        .with_pareto_patience(case.pareto_patience)
        .with_stagnation_patience(case.stagnation_patience)
        .with_verbose(case.verbose);

    evolution = if case.use_heuristic {
        evolution.with_candidate_scoring(CandidateScoringConfig::heuristic())
    } else {
        evolution.with_candidate_scoring(None)
    };

    let result = evolution.run()?;
    let report = evaluate_test_set(&result, test)?;
    Ok((result, report, started.elapsed().as_secs_f32()))
}

fn print_result_summary(result: &EvolutionResult, report: &TestSetReport) {
    println!("状态: {:?}", result.status);
    println!("代数: {}", result.generations);
    println!(
        "Binary IoU(primary): {:.1}%",
        result.fitness.primary * 100.0
    );
    if !result.fitness.report.is_empty() {
        println!("报告指标: {}", result.fitness.report.format_compact());
    }
    println!(
        "全测试集: PixelAccuracy mean={:.1}%, BinaryIoU min/mean/max={:.1}%/{:.1}%/{:.1}%, Dice mean={:.1}%",
        report.pixel_accuracy_mean * 100.0,
        report.iou_min * 100.0,
        report.iou_mean * 100.0,
        report.iou_max * 100.0,
        report.dice_mean * 100.0
    );
    println!(
        "最差样本 #{}: PixelAccuracy={:.1}%, BinaryIoU={:.1}%, Dice={:.1}%",
        report.worst_idx,
        report.samples[report.worst_idx].pixel_accuracy * 100.0,
        report.samples[report.worst_idx].binary_iou * 100.0,
        report.samples[report.worst_idx].dice * 100.0
    );
    println!(
        "最终架构包含 DeformableConv2d: {}",
        result.has_deformable_conv2d()
    );
    println!("最终架构: {}", result.architecture());
}

fn print_audit_matrix(rows: &[AuditRow]) {
    println!("\n=== Deformable evolution audit matrix ===");
    println!(
        "| case | portfolio | heuristic | data | budget | status | primary | IoU min/mean/max | Dice mean | PixelAcc mean | worst | deformable | time |"
    );
    println!("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|");
    for row in rows {
        println!(
            "| {} | {} | {} | {}/{} | g{} p{} o{} | {} | {:.1}% | {:.1}/{:.1}/{:.1}% | {:.1}% | {:.1}% | #{} | {} | {:.1}s |",
            row.name,
            row.portfolio_label,
            row.heuristic,
            row.train_samples,
            row.test_samples,
            row.max_generations,
            row.population_size,
            row.offspring_batch_size,
            row.status,
            row.primary * 100.0,
            row.iou_min * 100.0,
            row.iou_mean * 100.0,
            row.iou_max * 100.0,
            row.dice_mean * 100.0,
            row.pixel_accuracy_mean * 100.0,
            row.worst_idx,
            row.has_deformable,
            row.elapsed_secs
        );
    }
}

fn print_heuristic_decision_hint(rows: &[AuditRow]) {
    let best = rows.iter().max_by(|a, b| {
        a.iou_mean
            .partial_cmp(&b.iou_mean)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let deformable_best = rows.iter().filter(|row| row.has_deformable).max_by(|a, b| {
        a.iou_mean
            .partial_cmp(&b.iou_mean)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n=== Heuristic decision hint ===");
    if let Some(best) = best {
        println!(
            "当前矩阵最佳 mean IoU: {} ({:.1}%)",
            best.name,
            best.iou_mean * 100.0
        );
    }
    if let Some(deformable_best) = deformable_best {
        println!(
            "最佳含 DeformableConv2d 候选: {} ({:.1}%)",
            deformable_best.name,
            deformable_best.iou_mean * 100.0
        );
    }
    println!(
        "判断规则: 若含 deformable 的多族 + heuristic 配置稳定优于 dense/U-Net-lite，再考虑把它提升为独立 DeformableSeg family；否则优先保留为显式 opt-in 示例。"
    );
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

#[derive(Clone, Copy)]
struct AuditCase {
    name: &'static str,
    description: &'static str,
    portfolio_label: &'static str,
    portfolio: InitialPortfolioConfig,
    use_heuristic: bool,
    train_samples: usize,
    test_samples: usize,
    max_generations: usize,
    population_size: usize,
    offspring_batch_size: usize,
    initial_burst: usize,
    pareto_patience: usize,
    stagnation_patience: usize,
    verbose: bool,
}

impl AuditCase {
    fn smoke_deformable_only() -> Self {
        Self {
            name: "smoke_deformable_only",
            description: "当前 smoke 路径：只验证 DeformableConv2d seed 进入演化主流程。",
            portfolio_label: "deformable_only",
            portfolio: deformable_only_portfolio(),
            use_heuristic: false,
            train_samples: TRAIN_SAMPLES,
            test_samples: TEST_SAMPLES,
            max_generations: 3,
            population_size: 3,
            offspring_batch_size: 3,
            initial_burst: 0,
            pareto_patience: 5,
            stagnation_patience: 3,
            verbose: false,
        }
    }

    fn heuristic_label(&self) -> &'static str {
        if self.use_heuristic { "on" } else { "off" }
    }
}

fn deformable_only_portfolio() -> InitialPortfolioConfig {
    InitialPortfolioConfig {
        include_flat_mlp: false,
        include_tiny_cnn: false,
        include_lenet_tiny: false,
        include_unet_lite: false,
        include_deformable_tiny: true,
        flat_mlp_hidden: 1,
    }
}

fn audit_cases() -> Vec<AuditCase> {
    vec![
        AuditCase {
            verbose: true,
            ..AuditCase::smoke_deformable_only()
        },
        AuditCase {
            name: "seg_default_heuristic",
            description: "默认 segmentation portfolio：dense head / deep dense / U-Net-lite，不含 deformable。",
            portfolio_label: "seg_default",
            portfolio: InitialPortfolioConfig::vision_segmentation(),
            use_heuristic: true,
            train_samples: TRAIN_SAMPLES,
            test_samples: TEST_SAMPLES,
            max_generations: 3,
            population_size: 3,
            offspring_batch_size: 3,
            initial_burst: 0,
            pareto_patience: 5,
            stagnation_patience: 3,
            verbose: true,
        },
        AuditCase {
            name: "seg_deformable_heuristic",
            description: "多候选族 + deformable：检查 heuristic 是否保留并评估 deformable 相关候选。",
            portfolio_label: "seg_with_deformable",
            portfolio: InitialPortfolioConfig::vision_segmentation_with_deformable(),
            use_heuristic: true,
            train_samples: TRAIN_SAMPLES,
            test_samples: TEST_SAMPLES,
            max_generations: 3,
            population_size: 4,
            offspring_batch_size: 4,
            initial_burst: 0,
            pareto_patience: 5,
            stagnation_patience: 3,
            verbose: true,
        },
        AuditCase {
            name: "seg_deformable_no_heuristic",
            description: "多候选族 + deformable，但关闭 heuristic：对照预筛是否影响候选族进入完整评估。",
            portfolio_label: "seg_with_deformable",
            portfolio: InitialPortfolioConfig::vision_segmentation_with_deformable(),
            use_heuristic: false,
            train_samples: TRAIN_SAMPLES,
            test_samples: TEST_SAMPLES,
            max_generations: 3,
            population_size: 4,
            offspring_batch_size: 4,
            initial_burst: 0,
            pareto_patience: 5,
            stagnation_patience: 3,
            verbose: true,
        },
        AuditCase {
            name: "seg_deformable_budget_plus",
            description: "多候选族 + deformable + heuristic，并小幅增加样本与搜索预算。",
            portfolio_label: "seg_with_deformable",
            portfolio: InitialPortfolioConfig::vision_segmentation_with_deformable(),
            use_heuristic: true,
            train_samples: 24,
            test_samples: 8,
            max_generations: 5,
            population_size: 6,
            offspring_batch_size: 6,
            initial_burst: 1,
            pareto_patience: 8,
            stagnation_patience: 4,
            verbose: true,
        },
    ]
}

struct SampleMetrics {
    pixel_accuracy: f32,
    binary_iou: f32,
    dice: f32,
}

struct TestSetReport {
    samples: Vec<SampleMetrics>,
    pixel_accuracy_mean: f32,
    iou_min: f32,
    iou_mean: f32,
    iou_max: f32,
    dice_mean: f32,
    worst_idx: usize,
}

struct AuditRow {
    name: &'static str,
    portfolio_label: &'static str,
    heuristic: &'static str,
    train_samples: usize,
    test_samples: usize,
    max_generations: usize,
    population_size: usize,
    offspring_batch_size: usize,
    status: String,
    primary: f32,
    iou_min: f32,
    iou_mean: f32,
    iou_max: f32,
    dice_mean: f32,
    pixel_accuracy_mean: f32,
    worst_idx: usize,
    has_deformable: bool,
    elapsed_secs: f32,
}

fn evaluate_test_set(
    result: &EvolutionResult,
    dataset: &SegmentationDataset,
) -> Result<TestSetReport, Box<dyn Error>> {
    let mut samples = Vec::with_capacity(dataset.inputs.len());

    for idx in 0..dataset.inputs.len() {
        let pred = result.predict(&dataset.inputs[idx])?;
        let target = Tensor::stack(&[&dataset.labels[idx]], 0);
        let pred_mask = threshold_logits(&pred);
        samples.push(SampleMetrics {
            pixel_accuracy: pixel_accuracy(&pred_mask, &target, 0.5).value(),
            binary_iou: binary_iou(&pred_mask, &target, 0.5).value(),
            dice: dice_score(&pred_mask, &target, 0.5).value(),
        });
    }

    let worst_idx = samples
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.binary_iou
                .partial_cmp(&b.binary_iou)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    Ok(TestSetReport {
        pixel_accuracy_mean: mean_by(&samples, |sample| sample.pixel_accuracy),
        iou_min: min_by(&samples, |sample| sample.binary_iou),
        iou_mean: mean_by(&samples, |sample| sample.binary_iou),
        iou_max: max_by(&samples, |sample| sample.binary_iou),
        dice_mean: mean_by(&samples, |sample| sample.dice),
        samples,
        worst_idx,
    })
}

fn mean_by(samples: &[SampleMetrics], f: impl Fn(&SampleMetrics) -> f32) -> f32 {
    if samples.is_empty() {
        0.0
    } else {
        samples.iter().map(f).sum::<f32>() / samples.len() as f32
    }
}

fn min_by(samples: &[SampleMetrics], f: impl Fn(&SampleMetrics) -> f32) -> f32 {
    samples.iter().map(f).fold(f32::INFINITY, f32::min)
}

fn max_by(samples: &[SampleMetrics], f: impl Fn(&SampleMetrics) -> f32) -> f32 {
    samples.iter().map(f).fold(f32::NEG_INFINITY, f32::max)
}

struct SegmentationDataset {
    inputs: Vec<Tensor>,
    labels: Vec<Tensor>,
}

fn generate_dataset(n: usize, seed: u64) -> SegmentationDataset {
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for sample_idx in 0..n {
        let objects = generate_objects(sample_idx, seed);
        let mut image = Vec::with_capacity(IMAGE_SIZE * IMAGE_SIZE);
        let mut mask = Vec::with_capacity(IMAGE_SIZE * IMAGE_SIZE);

        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let mut foreground = false;
                for object in &objects {
                    if object.contains(x, y) {
                        foreground = true;
                    }
                }

                let noise = deterministic_noise(seed, sample_idx, x, y);
                let gradient = (x as f32 + y as f32) / (2.0 * (IMAGE_SIZE - 1) as f32);
                let base = if foreground { 0.82 } else { 0.04 };
                image.push((base + 0.08 * gradient + 0.08 * noise).clamp(0.0, 1.0));
                mask.push(if foreground { 1.0 } else { 0.0 });
            }
        }

        inputs.push(Tensor::new(&image, &[1, IMAGE_SIZE, IMAGE_SIZE]));
        labels.push(Tensor::new(&mask, &[1, IMAGE_SIZE, IMAGE_SIZE]));
    }

    SegmentationDataset { inputs, labels }
}

fn generate_objects(sample_idx: usize, seed: u64) -> Vec<ShapeObject> {
    let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64]);
    let count = rng.usize_range(1..MAX_OBJECTS + 1);
    (0..count)
        .map(|idx| {
            let mut obj_rng = rng.fork(idx as u64 + 1);
            let kind = match obj_rng.usize_range(0..3) {
                0 => ShapeKind::Rectangle,
                1 => ShapeKind::Circle,
                _ => ShapeKind::Triangle,
            };
            let margin = 4isize;
            ShapeObject {
                kind,
                cx: obj_rng.isize_range(margin..IMAGE_SIZE as isize - margin),
                cy: obj_rng.isize_range(margin..IMAGE_SIZE as isize - margin),
                half_w: obj_rng.isize_range(3..7),
                half_h: obj_rng.isize_range(3..7),
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum ShapeKind {
    Rectangle,
    Circle,
    Triangle,
}

struct ShapeObject {
    kind: ShapeKind,
    cx: isize,
    cy: isize,
    half_w: isize,
    half_h: isize,
}

impl ShapeObject {
    fn contains(&self, x: usize, y: usize) -> bool {
        let dx = x as isize - self.cx;
        let dy = y as isize - self.cy;
        match self.kind {
            ShapeKind::Rectangle => dx.abs() <= self.half_w && dy.abs() <= self.half_h,
            ShapeKind::Circle => {
                let rx = self.half_w.max(1) as f32;
                let ry = self.half_h.max(1) as f32;
                (dx as f32 / rx).powi(2) + (dy as f32 / ry).powi(2) <= 1.0
            }
            ShapeKind::Triangle => {
                if dy < -self.half_h || dy > self.half_h {
                    return false;
                }
                let t = (dy + self.half_h) as f32 / (2 * self.half_h.max(1)) as f32;
                let half_width_at_y = (self.half_w as f32 * t).max(1.0);
                (dx as f32).abs() <= half_width_at_y
            }
        }
    }
}

fn save_sample_visualizations(
    dataset: &SegmentationDataset,
    sample_idx: usize,
    prediction: &Tensor,
    artifact_stem: &str,
) -> Result<(), Box<dyn Error>> {
    use image::{ImageBuffer, Rgb};

    let panel_size = IMAGE_SIZE as u32 * OVERLAY_SCALE;
    let mut input_img = ImageBuffer::from_pixel(panel_size, panel_size, Rgb([245, 245, 245]));
    let mut target_img = ImageBuffer::from_pixel(panel_size, panel_size, Rgb([245, 245, 245]));
    let mut pred_img = ImageBuffer::from_pixel(panel_size, panel_size, Rgb([245, 245, 245]));

    let input = &dataset.inputs[sample_idx];
    let target = &dataset.labels[sample_idx];
    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let base = (input[[0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            pixel_block_scale(
                &mut input_img,
                x as u32,
                y as u32,
                [base, base, base],
                OVERLAY_SCALE,
            );

            let target_on = target[[0, y, x]] >= 0.5;
            pixel_block_scale(
                &mut target_img,
                x as u32,
                y as u32,
                binary_color(target_on),
                OVERLAY_SCALE,
            );

            let pred_on = prediction[[0, 0, y, x]] >= 0.0;
            pixel_block_scale(
                &mut pred_img,
                x as u32,
                y as u32,
                binary_color(pred_on),
                OVERLAY_SCALE,
            );
        }
    }

    let input_path = artifact_path(artifact_stem, "in");
    let target_path = artifact_path(artifact_stem, "target");
    let output_path = artifact_path(artifact_stem, "out");
    save_rgb_image(&input_img, &input_path)?;
    save_rgb_image(&target_img, &target_path)?;
    save_rgb_image(&pred_img, &output_path)?;
    println!("样本 #{sample_idx} 输入图: {input_path}");
    println!("样本 #{sample_idx} 标签图: {target_path}");
    println!("样本 #{sample_idx} 输出图: {output_path}");

    Ok(())
}

fn artifact_path(stem: &str, kind: &str) -> String {
    let file_name = if stem == "test" {
        format!("test_{kind}.png")
    } else {
        format!("{stem}_{kind}.png")
    };
    format!("examples/evolution/deformable_conv2d_segmentation/{file_name}")
}

fn save_rgb_image(image: &image::RgbImage, path: &str) -> Result<(), image::ImageError> {
    image.save(path)
}

fn binary_color(on: bool) -> [u8; 3] {
    if on { [80, 220, 120] } else { [32, 32, 32] }
}

fn threshold_logits(tensor: &Tensor) -> Tensor {
    let data: Vec<f32> = tensor
        .to_vec()
        .into_iter()
        .map(|value| if value >= 0.0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(&data, tensor.shape())
}

fn deterministic_noise(seed: u64, sample_idx: usize, x: usize, y: usize) -> f32 {
    let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64, x as u64, y as u64]);
    rng.next_f32()
}
