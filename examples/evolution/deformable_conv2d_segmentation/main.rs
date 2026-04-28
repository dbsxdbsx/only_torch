/*
 * @Author       : 老董
 * @Date         : 2026-04-28
 * @Description  : DeformableConv2d 分割演化 benchmark
 */

use only_torch::data::SyntheticRng;
use only_torch::metrics::{binary_iou, pixel_accuracy};
use only_torch::nn::evolution::{Evolution, InitialPortfolioConfig, ReportMetric, TaskMetric};
use only_torch::tensor::Tensor;
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

    println!("=== DeformableConv2d Semantic Segmentation Evolution 示例 ===\n");

    let train = generate_dataset(TRAIN_SAMPLES, 42);
    let test = generate_dataset(TEST_SAMPLES, 2026);
    let train_data = (train.inputs.clone(), train.labels.clone());
    let test_data = (test.inputs.clone(), test.labels.clone());

    println!("任务: 16x16 合成图像二值前景分割（1..3 个可重叠形状）");
    println!("输入: [1, {IMAGE_SIZE}, {IMAGE_SIZE}]，标签: [1, {IMAGE_SIZE}, {IMAGE_SIZE}]");
    println!("指标: BinaryIoU，候选族: DeformableConv2d dense segmentation seed");
    println!(
        "演化策略: 关闭启发式预筛；用 deformable 初始族直接验证 DeformableConv2d 进入演化主流程"
    );
    println!("训练样本: {TRAIN_SAMPLES}, 测试样本: {TEST_SAMPLES}, batch: {BATCH_SIZE}");
    println!("演化 seed: {evolution_seed}, target Binary IoU: {target_binary_iou:.2}\n");

    let deformable_only = InitialPortfolioConfig {
        include_flat_mlp: false,
        include_tiny_cnn: false,
        include_lenet_tiny: false,
        include_unet_lite: false,
        include_deformable_tiny: true,
        flat_mlp_hidden: 1,
    };

    let result = Evolution::supervised(train_data, test_data, TaskMetric::BinaryIoU)
        .with_initial_portfolio(deformable_only)
        .with_initial_burst(0)
        .with_candidate_scoring(None)
        .with_target_metric(target_binary_iou)
        .with_report_metrics([
            ReportMetric::PixelAccuracy,
            ReportMetric::BinaryIoU,
            ReportMetric::Dice,
        ])
        .with_seed(evolution_seed)
        .with_max_generations(3)
        .with_population_size(3)
        .with_offspring_batch_size(3)
        .with_parallelism(1)
        .with_batch_size(BATCH_SIZE)
        .with_pareto_patience(5)
        .with_stagnation_patience(3)
        .run()?;

    println!("\n=== 演化结果 ===");
    println!("状态: {:?}", result.status);
    println!("代数: {}", result.generations);
    println!("Binary IoU: {:.1}%", result.fitness.primary * 100.0);
    if !result.fitness.report.is_empty() {
        println!("报告指标: {}", result.fitness.report.format_compact());
    }
    println!(
        "最终架构包含 DeformableConv2d: {}",
        result.has_deformable_conv2d()
    );
    println!("最终架构: {}", result.architecture());

    let sample_idx = 0;
    let pred = result.predict(&test.inputs[sample_idx])?;
    let target = Tensor::stack(&[&test.labels[sample_idx]], 0);
    let pred_mask = threshold_logits(&pred);
    println!(
        "代表样本 #{sample_idx} Pixel Accuracy: {:.1}%, Binary IoU: {:.1}%",
        pixel_accuracy(&pred_mask, &target, 0.5).percent(),
        binary_iou(&pred_mask, &target, 0.5).percent()
    );

    if save_artifacts {
        save_sample_visualizations(&test, sample_idx, &pred)?;
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

    println!(
        "\nEvolution DeformableConv2d segmentation 示例完成，总耗时: {:.1}s",
        total_start.elapsed().as_secs_f32()
    );

    Ok(())
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
            fill_scaled_pixel(&mut input_img, x, y, [base, base, base]);

            let target_on = target[[0, y, x]] >= 0.5;
            fill_scaled_pixel(&mut target_img, x, y, binary_color(target_on));

            let pred_on = prediction[[0, 0, y, x]] >= 0.0;
            fill_scaled_pixel(&mut pred_img, x, y, binary_color(pred_on));
        }
    }

    save_rgb_image(
        &input_img,
        "examples/evolution/deformable_conv2d_segmentation/test_in.png",
    )?;
    save_rgb_image(
        &target_img,
        "examples/evolution/deformable_conv2d_segmentation/test_target.png",
    )?;
    save_rgb_image(
        &pred_img,
        "examples/evolution/deformable_conv2d_segmentation/test_out.png",
    )?;
    println!("测试输入图: examples/evolution/deformable_conv2d_segmentation/test_in.png");
    println!("测试标签图: examples/evolution/deformable_conv2d_segmentation/test_target.png");
    println!("测试输出图: examples/evolution/deformable_conv2d_segmentation/test_out.png");

    Ok(())
}

fn save_rgb_image(image: &image::RgbImage, path: &str) -> Result<(), image::ImageError> {
    image.save(path)
}

fn fill_scaled_pixel(canvas: &mut image::RgbImage, x: usize, y: usize, color: [u8; 3]) {
    let x0 = x as u32 * OVERLAY_SCALE;
    let y0 = y as u32 * OVERLAY_SCALE;
    for dy in 0..OVERLAY_SCALE {
        for dx in 0..OVERLAY_SCALE {
            canvas.put_pixel(x0 + dx, y0 + dy, image::Rgb(color));
        }
    }
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
