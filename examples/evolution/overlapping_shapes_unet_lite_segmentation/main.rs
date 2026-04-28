/*
 * @Author       : 老董
 * @Date         : 2026-04-28
 * @Description  : 对齐 U-Net-lite 强基线的重叠形状语义分割 Evolution benchmark
 */

use only_torch::data::SyntheticRng;
use only_torch::metrics::{mean_iou, per_class_iou, semantic_pixel_accuracy};
use only_torch::nn::evolution::{Evolution, ReportMetric, TaskMetric};
use only_torch::tensor::Tensor;
use std::env;
use std::error::Error;
use std::time::Instant;

const IMAGE_SIZE: usize = 64;
const NUM_CLASSES: usize = 4;
const MAX_OBJECTS: usize = 3;
const TRAIN_SAMPLES: usize = 32;
const TEST_SAMPLES: usize = 12;
const BATCH_SIZE: usize = 8;
const OVERLAY_SCALE: u32 = 5;
const TARGET_MEAN_IOU: f32 = 0.60;
const DEFAULT_EVOLUTION_SEED: u64 = 42;

fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now();
    let evolution_seed = env_u64(
        "ONLY_TORCH_EVOLUTION_UNET_LITE_SEED",
        DEFAULT_EVOLUTION_SEED,
    );
    let target_mean_iou = env_f32("ONLY_TORCH_EVOLUTION_UNET_LITE_TARGET", TARGET_MEAN_IOU);
    let save_artifacts = env_bool("ONLY_TORCH_EVOLUTION_UNET_LITE_SAVE_ARTIFACTS", true);

    println!("=== Overlapping Shapes U-Net-lite Benchmark Evolution 示例 ===\n");

    let train = generate_dataset(TRAIN_SAMPLES, 42);
    let test = generate_dataset(TEST_SAMPLES, 2026);
    let train_data = (train.inputs.clone(), train.labels.clone());
    let test_data = (test.inputs.clone(), test.labels.clone());

    println!("任务: 64x64 合成图像多类别语义分割（0..3 个可重叠形状）");
    println!(
        "输入: [1, {IMAGE_SIZE}, {IMAGE_SIZE}]，标签: [{NUM_CLASSES}, {IMAGE_SIZE}, {IMAGE_SIZE}]"
    );
    println!("指标: MeanIoU，传统对照: overlapping_shapes_unet_lite_segmentation");
    println!("范围: 对齐同一 benchmark，并纳入 U-Net-lite encoder-decoder 初始族");
    println!("演化策略: segmentation portfolio + encoder-decoder family-diverse 启发式预筛");
    println!("训练样本: {TRAIN_SAMPLES}, 测试样本: {TEST_SAMPLES}, batch: {BATCH_SIZE}");
    println!("演化 seed: {evolution_seed}, target Mean IoU: {target_mean_iou:.2}\n");

    let result = Evolution::supervised(train_data, test_data, TaskMetric::MeanIoU)
        .with_target_metric(target_mean_iou)
        .with_report_metrics([ReportMetric::PixelAccuracy, ReportMetric::MeanIoU])
        .with_seed(evolution_seed)
        .with_max_generations(3)
        .with_population_size(4)
        .with_offspring_batch_size(4)
        .with_parallelism(1)
        .with_batch_size(BATCH_SIZE)
        .with_pareto_patience(5)
        .with_stagnation_patience(3)
        .run()?;

    println!("\n=== 演化结果 ===");
    println!("状态: {:?}", result.status);
    println!("代数: {}", result.generations);
    println!("Mean IoU: {:.1}%", result.fitness.primary * 100.0);
    if !result.fitness.report.is_empty() {
        println!("报告指标: {}", result.fitness.report.format_compact());
    }
    println!("最终架构: {}", result.architecture());

    let sample_idx = representative_sample_index(&test);
    let pred = result.predict(&test.inputs[sample_idx])?;
    let target = Tensor::stack(&[&test.labels[sample_idx]], 0);
    println!(
        "代表样本 #{sample_idx}（对象数 {}）Pixel Accuracy: {:.1}%, Mean IoU: {:.1}%",
        test.object_counts[sample_idx],
        semantic_pixel_accuracy(&pred, &target).percent(),
        mean_iou(&pred, &target).percent()
    );

    println!("\n=== 代表样本 Per-class IoU ===");
    for (class_idx, value) in per_class_iou(&pred, &target).iter().enumerate() {
        println!("{}: {:.1}%", class_name(class_idx), value.percent());
    }

    if save_artifacts {
        save_sample_visualizations(&test, sample_idx, &pred)?;

        let vis = result.visualize(
            "examples/evolution/overlapping_shapes_unet_lite_segmentation/evolution_overlapping_shapes_unet_lite_segmentation",
        )?;
        println!("\n计算图已保存: {}", vis.dot_path.display());
        if let Some(img) = &vis.image_path {
            println!("可视化图像: {}", img.display());
        }
    } else {
        println!("\n已跳过图片和 Graphviz 产物保存");
    }

    println!(
        "\nEvolution U-Net-lite benchmark 示例完成，总耗时: {:.1}s",
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
    object_counts: Vec<usize>,
}

fn generate_dataset(n: usize, seed: u64) -> SegmentationDataset {
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    let mut object_counts = Vec::with_capacity(n);

    for sample_idx in 0..n {
        let objects = generate_objects(sample_idx, seed);
        let mut image = Vec::with_capacity(IMAGE_SIZE * IMAGE_SIZE);
        let mut class_map = vec![0usize; IMAGE_SIZE * IMAGE_SIZE];

        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let mut class_id = 0usize;
                for object in &objects {
                    if object.contains(x, y) {
                        class_id = object.class_id;
                    }
                }
                class_map[y * IMAGE_SIZE + x] = class_id;

                let noise = deterministic_noise(seed, sample_idx, x, y);
                let gradient = (x as f32 + y as f32) / (2.0 * (IMAGE_SIZE - 1) as f32);
                let base = match class_id {
                    1 => 0.38,
                    2 => 0.62,
                    3 => 0.86,
                    _ => 0.04,
                };
                image.push((base + 0.08 * gradient + 0.10 * noise).clamp(0.0, 1.0));
            }
        }

        let mut mask = Vec::with_capacity(NUM_CLASSES * IMAGE_SIZE * IMAGE_SIZE);
        for class_idx in 0..NUM_CLASSES {
            for &pixel_class in &class_map {
                mask.push(if pixel_class == class_idx { 1.0 } else { 0.0 });
            }
        }

        object_counts.push(objects.len());
        inputs.push(Tensor::new(&image, &[1, IMAGE_SIZE, IMAGE_SIZE]));
        labels.push(Tensor::new(&mask, &[NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE]));
    }

    SegmentationDataset {
        inputs,
        labels,
        object_counts,
    }
}

fn generate_objects(sample_idx: usize, seed: u64) -> Vec<ShapeObject> {
    let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64]);
    let count = rng.usize_range(0..MAX_OBJECTS + 1);
    (0..count)
        .map(|idx| {
            let mut obj_rng = rng.fork(idx as u64 + 1);
            let kind = match obj_rng.usize_range(0..3) {
                0 => ShapeKind::Rectangle,
                1 => ShapeKind::Circle,
                _ => ShapeKind::Triangle,
            };
            let margin = 12isize;
            ShapeObject {
                kind,
                class_id: kind.class_id(),
                cx: obj_rng.isize_range(margin..IMAGE_SIZE as isize - margin),
                cy: obj_rng.isize_range(margin..IMAGE_SIZE as isize - margin),
                half_w: obj_rng.isize_range(5..16),
                half_h: obj_rng.isize_range(5..16),
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

impl ShapeKind {
    const fn class_id(self) -> usize {
        match self {
            Self::Rectangle => 1,
            Self::Circle => 2,
            Self::Triangle => 3,
        }
    }
}

struct ShapeObject {
    kind: ShapeKind,
    class_id: usize,
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

fn representative_sample_index(dataset: &SegmentationDataset) -> usize {
    dataset
        .object_counts
        .iter()
        .position(|&count| count >= 2)
        .unwrap_or(0)
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

            fill_scaled_pixel(
                &mut target_img,
                x,
                y,
                class_color(argmax_class(target, y, x)),
            );
            fill_scaled_pixel(
                &mut pred_img,
                x,
                y,
                class_color(argmax_prediction_class(prediction, y, x)),
            );
        }
    }

    save_rgb_image(
        &input_img,
        "examples/evolution/overlapping_shapes_unet_lite_segmentation/test_in.png",
    )?;
    save_rgb_image(
        &target_img,
        "examples/evolution/overlapping_shapes_unet_lite_segmentation/test_target.png",
    )?;
    save_rgb_image(
        &pred_img,
        "examples/evolution/overlapping_shapes_unet_lite_segmentation/test_out.png",
    )?;
    println!(
        "测试输入图: examples/evolution/overlapping_shapes_unet_lite_segmentation/test_in.png"
    );
    println!(
        "测试标签图: examples/evolution/overlapping_shapes_unet_lite_segmentation/test_target.png"
    );
    println!(
        "测试输出图: examples/evolution/overlapping_shapes_unet_lite_segmentation/test_out.png"
    );

    Ok(())
}

fn argmax_class(tensor: &Tensor, y: usize, x: usize) -> usize {
    let classes = tensor.shape()[0];
    let mut best_class = 0usize;
    let mut best_value = tensor[[0, y, x]];
    for class_idx in 1..classes {
        let value = tensor[[class_idx, y, x]];
        if value > best_value {
            best_class = class_idx;
            best_value = value;
        }
    }
    best_class
}

fn argmax_prediction_class(tensor: &Tensor, y: usize, x: usize) -> usize {
    let shape = tensor.shape();
    let classes = shape[1];
    let mut best_class = 0usize;
    let mut best_value = tensor[[0, 0, y, x]];
    for class_idx in 1..classes {
        let value = tensor[[0, class_idx, y, x]];
        if value > best_value {
            best_class = class_idx;
            best_value = value;
        }
    }
    best_class
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

fn class_name(class_idx: usize) -> &'static str {
    match class_idx {
        0 => "background",
        1 => "rectangle",
        2 => "circle",
        3 => "triangle",
        _ => "unknown",
    }
}

fn class_color(class_idx: usize) -> [u8; 3] {
    match class_idx {
        0 => [32, 32, 32],
        1 => [240, 76, 76],
        2 => [76, 160, 255],
        3 => [80, 220, 120],
        _ => [245, 245, 245],
    }
}

fn deterministic_noise(seed: u64, sample_idx: usize, x: usize, y: usize) -> f32 {
    let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64, x as u64, y as u64]);
    rng.next_f32()
}
