/*
 * @Author       : 老董
 * @Date         : 2026-04-28
 * @Description  : 重叠形状语义分割的 Evolution 示例
 */

use only_torch::metrics::mean_iou;
use only_torch::nn::evolution::{Evolution, ReportMetric, TaskMetric};
use only_torch::tensor::Tensor;
use std::time::Instant;

const IMAGE_SIZE: usize = 32;
const NUM_CLASSES: usize = 3;
const MAX_OBJECTS: usize = 2;

fn main() {
    let total_start = Instant::now();
    println!("=== Overlapping Shapes Semantic Segmentation Evolution 示例 ===\n");

    let train_samples = 24;
    let test_samples = 8;
    let train_data = generate_dataset(train_samples, 42);
    let test_data = generate_dataset(test_samples, 2026);

    println!("任务: 32x32 合成图像语义分割（0..2 个可重叠形状）");
    println!(
        "输入: [1, {IMAGE_SIZE}, {IMAGE_SIZE}]，标签: [{NUM_CLASSES}, {IMAGE_SIZE}, {IMAGE_SIZE}]"
    );
    println!("指标: MeanIoU，起始结构: Conv2d → 1x1 Conv2d head（不经过 Flatten）");
    println!("训练样本: {train_samples}, 测试样本: {test_samples}\n");

    let result = Evolution::supervised(train_data, test_data.clone(), TaskMetric::MeanIoU)
        .with_target_metric(0.35)
        .with_report_metrics([ReportMetric::PixelAccuracy, ReportMetric::MeanIoU])
        .with_seed(42)
        .with_max_generations(4)
        .with_population_size(4)
        .with_offspring_batch_size(4)
        .with_parallelism(1)
        .with_batch_size(12)
        .with_pareto_patience(6)
        .with_stagnation_patience(4)
        .run()
        .expect("分割演化过程出错");

    println!("\n=== 演化结果 ===");
    println!("状态: {:?}", result.status);
    println!("代数: {}", result.generations);
    println!("Mean IoU: {:.1}%", result.fitness.primary * 100.0);
    if !result.fitness.report.is_empty() {
        println!("报告指标: {}", result.fitness.report.format_compact());
    }
    println!("最终架构: {}", result.architecture());

    let pred = result.predict(&test_data.0[0]).expect("分割推理失败");
    let pred_iou = mean_iou(&pred, &Tensor::stack(&[&test_data.1[0]], 0));
    println!("单样本 Mean IoU: {:.1}%", pred_iou.percent());

    let vis = result
        .visualize(
            "examples/evolution/overlapping_shapes_semantic_segmentation/evolution_overlapping_shapes_semantic_segmentation",
        )
        .expect("可视化失败");
    println!("计算图已保存: {}", vis.dot_path.display());
    if let Some(img) = &vis.image_path {
        println!("可视化图像: {}", img.display());
    }

    println!(
        "\nEvolution semantic segmentation 示例完成，总耗时: {:.1}s",
        total_start.elapsed().as_secs_f32()
    );
}

fn generate_dataset(n: usize, seed: u64) -> (Vec<Tensor>, Vec<Tensor>) {
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
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
                let base = match class_id {
                    1 => 0.45,
                    2 => 0.78,
                    _ => 0.05,
                };
                image.push((base + 0.10 * noise).clamp(0.0, 1.0));
            }
        }

        let mut mask = Vec::with_capacity(NUM_CLASSES * IMAGE_SIZE * IMAGE_SIZE);
        for class_idx in 0..NUM_CLASSES {
            for &pixel_class in &class_map {
                mask.push(if pixel_class == class_idx { 1.0 } else { 0.0 });
            }
        }

        inputs.push(Tensor::new(&image, &[1, IMAGE_SIZE, IMAGE_SIZE]));
        labels.push(Tensor::new(&mask, &[NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE]));
    }
    (inputs, labels)
}

fn generate_objects(sample_idx: usize, seed: u64) -> Vec<ShapeObject> {
    let h = mix(seed ^ sample_idx as u64);
    let count = (h % (MAX_OBJECTS as u64 + 1)) as usize;
    (0..count)
        .map(|idx| {
            let oh = mix(h ^ (idx as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let margin = 7isize;
            let span = IMAGE_SIZE as isize - 2 * margin;
            let cx = margin + ((oh >> 16) % span as u64) as isize;
            let cy = margin + ((oh >> 32) % span as u64) as isize;
            ShapeObject {
                kind: if oh & 1 == 0 {
                    ShapeKind::Rectangle
                } else {
                    ShapeKind::Circle
                },
                class_id: 1 + (oh as usize % (NUM_CLASSES - 1)),
                cx,
                cy,
                half_w: 4 + ((oh >> 40) % 6) as isize,
                half_h: 4 + ((oh >> 48) % 6) as isize,
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum ShapeKind {
    Rectangle,
    Circle,
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
        }
    }
}

fn deterministic_noise(seed: u64, sample_idx: usize, x: usize, y: usize) -> f32 {
    let h = mix(seed ^ ((sample_idx as u64) << 32) ^ ((x as u64) << 16) ^ y as u64);
    (h % 1000) as f32 / 1000.0
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}
