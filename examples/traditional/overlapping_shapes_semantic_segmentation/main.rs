//! # Overlapping Shapes Semantic Segmentation 示例
//!
//! 这个示例是比 16x16 toy 更可信的小型语义分割 benchmark：
//! - 固定 64x64 单通道图像，一个 batch 内尺寸固定。
//! - 每张图随机生成 0 到 3 个矩形、圆形、三角形。
//! - 生成对象允许重叠，但标签是每像素单类别的 visible semantic map。
//! - 输出 `[N, 4, H, W]` logits：背景 / 矩形 / 圆形 / 三角形。
//! - 使用 one-hot mask + BCEWithLogits 训练，报告 Pixel Accuracy / Dice / Mean IoU。
//!
//! ## 运行
//! ```bash
//! cargo run --example overlapping_shapes_semantic_segmentation
//! ```

mod model;

use model::OverlappingShapesSemanticSegmentationNet;
use only_torch::data::{DataLoader, SyntheticRng, TensorDataset};
use only_torch::metrics::{dice_score, mean_iou, per_class_iou, semantic_pixel_accuracy};
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;
use std::time::Instant;

const IMAGE_SIZE: usize = 64;
const NUM_CLASSES: usize = 4;
const MAX_OBJECTS: usize = 3;
const OVERLAY_SCALE: u32 = 5;
const TRAIN_SAMPLES: usize = 96;
const TEST_SAMPLES: usize = 32;
const BATCH_SIZE: usize = 16;
const MAX_EPOCHS: usize = 24;
const LEARNING_RATE: f32 = 0.02;
const TARGET_MEAN_IOU: f32 = 0.55;

fn main() -> Result<(), GraphError> {
    let total_start = Instant::now();
    println!("=== Overlapping Shapes Semantic Segmentation 示例 ===\n");

    let (train_x, train_y) = generate_dataset(TRAIN_SAMPLES, 42);
    let (test_x, test_y) = generate_dataset(TEST_SAMPLES, 2026);
    let train_loader = DataLoader::new(TensorDataset::new(train_x, train_y), BATCH_SIZE)
        .shuffle(true)
        .seed(17);

    let graph = Graph::new_with_seed(42);
    let model = OverlappingShapesSemanticSegmentationNet::new(&graph)?;
    let mut optimizer = Adam::new(&graph, &model.parameters(), LEARNING_RATE);

    let param_count: usize = model
        .parameters()
        .iter()
        .filter_map(|p| p.value().ok().flatten())
        .map(|t| t.shape().iter().product::<usize>())
        .sum();

    println!("任务: 64x64 合成图像多类别语义分割（0..3 个可重叠形状）");
    println!("类别: 0=背景, 1=矩形, 2=圆形, 3=三角形");
    println!("网络: Conv(1→12) → ReLU → Conv(12→16) → ReLU → Conv(16→4)");
    println!("损失: BCEWithLogitsLoss(one-hot)，指标: Pixel Accuracy + Dice + Mean IoU");
    println!("训练样本: {TRAIN_SAMPLES}, 测试样本: {TEST_SAMPLES}, 参数量: {param_count}\n");

    let mut best_mean_iou = 0.0f32;
    let mut best_acc = 0.0f32;

    for epoch in 0..MAX_EPOCHS {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0usize;

        graph.train();
        for (batch_x, batch_y) in train_loader.iter() {
            let logits = model.forward(&batch_x)?;
            let loss = logits.bce_loss(&batch_y)?;

            graph.snapshot_once_from(&[&loss]);

            optimizer.zero_grad()?;
            let loss_val = loss.backward()?;
            optimizer.step()?;

            epoch_loss += loss_val;
            num_batches += 1;
        }

        graph.eval();
        let report = evaluate(&model, &test_x, &test_y)?;
        best_acc = best_acc.max(report.pixel_accuracy);
        best_mean_iou = best_mean_iou.max(report.mean_iou);

        println!(
            "Epoch {:2}: loss={:.4}, pixel_acc={:.1}%, fg_dice={:.1}%, mean_iou={:.1}%, {:.2}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            report.pixel_accuracy * 100.0,
            report.foreground_dice * 100.0,
            report.mean_iou * 100.0,
            epoch_start.elapsed().as_secs_f32()
        );

        if report.mean_iou >= TARGET_MEAN_IOU {
            println!(
                "\n达到目标 Mean IoU {:.1}%，提前停止。",
                report.mean_iou * 100.0
            );
            break;
        }
    }

    let report = evaluate(&model, &test_x, &test_y)?;
    println!("\n=== Per-class IoU ===");
    for (class_idx, value) in report.per_class_iou.iter().enumerate() {
        println!("{}: {:.1}%", class_name(class_idx), value * 100.0);
    }

    save_sample_visualizations(&model, &test_x, 0)?;
    println!(
        "测试输入图: examples/traditional/overlapping_shapes_semantic_segmentation/test_in.png"
    );
    println!(
        "测试输出图: examples/traditional/overlapping_shapes_semantic_segmentation/test_out.png"
    );

    let vis_result = graph.visualize_snapshot(
        "examples/traditional/overlapping_shapes_semantic_segmentation/overlapping_shapes_semantic_segmentation",
    )?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    println!(
        "\n最佳 Pixel Accuracy: {:.1}%，最佳 Mean IoU: {:.1}%，总耗时: {:.1}s",
        best_acc * 100.0,
        best_mean_iou * 100.0,
        total_start.elapsed().as_secs_f32()
    );

    if best_mean_iou >= TARGET_MEAN_IOU {
        println!("Overlapping shapes semantic segmentation 训练成功。");
        Ok(())
    } else {
        Err(GraphError::ComputationError(format!(
            "Overlapping shapes semantic segmentation 未达到目标 Mean IoU {:.1}%，最佳 {:.1}%",
            TARGET_MEAN_IOU * 100.0,
            best_mean_iou * 100.0
        )))
    }
}

struct EvalReport {
    pixel_accuracy: f32,
    foreground_dice: f32,
    mean_iou: f32,
    per_class_iou: Vec<f32>,
}

fn evaluate(
    model: &OverlappingShapesSemanticSegmentationNet,
    inputs: &Tensor,
    targets: &Tensor,
) -> Result<EvalReport, GraphError> {
    let probs = model.predict_probs(inputs)?;
    let probs = probs.value()?.unwrap();
    let pixel_accuracy = semantic_pixel_accuracy(&probs, targets).value();
    let mean_iou_value = mean_iou(&probs, targets).value();
    let per_class_iou_values = per_class_iou(&probs, targets)
        .iter()
        .map(|metric| metric.value())
        .collect();
    let foreground_dice = dice_score(
        &foreground_probability_mask(&probs),
        &foreground_target_mask(targets),
        0.5,
    )
    .value();

    Ok(EvalReport {
        pixel_accuracy,
        foreground_dice,
        mean_iou: mean_iou_value,
        per_class_iou: per_class_iou_values,
    })
}

fn generate_dataset(n: usize, seed: u64) -> (Tensor, Tensor) {
    let mut images = Vec::with_capacity(n * IMAGE_SIZE * IMAGE_SIZE);
    let mut masks = Vec::with_capacity(n * NUM_CLASSES * IMAGE_SIZE * IMAGE_SIZE);

    for sample_idx in 0..n {
        let objects = generate_objects(sample_idx, seed);
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
                let pixel = base + 0.08 * gradient + 0.10 * noise;
                images.push(pixel.clamp(0.0, 1.0));
            }
        }

        for class_idx in 0..NUM_CLASSES {
            for &pixel_class in &class_map {
                masks.push(if pixel_class == class_idx { 1.0 } else { 0.0 });
            }
        }
    }

    (
        Tensor::new(&images, &[n, 1, IMAGE_SIZE, IMAGE_SIZE]),
        Tensor::new(&masks, &[n, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE]),
    )
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

fn foreground_probability_mask(probs: &Tensor) -> Tensor {
    let shape = probs.shape();
    let (n, h, w) = (shape[0], shape[2], shape[3]);
    let mut data = Vec::with_capacity(n * h * w);
    for sample in 0..n {
        for y in 0..h {
            for x in 0..w {
                let mut value = 0.0f32;
                for class_idx in 1..shape[1] {
                    value = value.max(probs[[sample, class_idx, y, x]]);
                }
                data.push(value);
            }
        }
    }
    Tensor::new(&data, &[n, 1, h, w])
}

fn foreground_target_mask(targets: &Tensor) -> Tensor {
    let shape = targets.shape();
    let (n, h, w) = (shape[0], shape[2], shape[3]);
    let mut data = Vec::with_capacity(n * h * w);
    for sample in 0..n {
        for y in 0..h {
            for x in 0..w {
                let mut value = 0.0f32;
                for class_idx in 1..shape[1] {
                    value = value.max(targets[[sample, class_idx, y, x]]);
                }
                data.push(value);
            }
        }
    }
    Tensor::new(&data, &[n, 1, h, w])
}

fn save_sample_visualizations(
    model: &OverlappingShapesSemanticSegmentationNet,
    inputs: &Tensor,
    sample_idx: usize,
) -> Result<(), GraphError> {
    use image::{ImageBuffer, Rgb};

    let probs = model.predict_probs(inputs)?;
    let probs = probs.value()?.unwrap();

    let panel_size = IMAGE_SIZE as u32 * OVERLAY_SCALE;
    let mut input_img = ImageBuffer::from_pixel(panel_size, panel_size, Rgb([245, 245, 245]));
    let mut output_img = ImageBuffer::from_pixel(panel_size, panel_size, Rgb([245, 245, 245]));

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let base = (inputs[[sample_idx, 0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            fill_scaled_pixel(&mut input_img, x, y, [base, base, base]);

            let class_idx = argmax_class(&probs, sample_idx, y, x);
            fill_scaled_pixel(&mut output_img, x, y, class_color(class_idx));
        }
    }

    save_rgb_image(
        &input_img,
        "examples/traditional/overlapping_shapes_semantic_segmentation/test_in.png",
    )?;
    save_rgb_image(
        &output_img,
        "examples/traditional/overlapping_shapes_semantic_segmentation/test_out.png",
    )?;

    Ok(())
}

fn argmax_class(tensor: &Tensor, sample_idx: usize, y: usize, x: usize) -> usize {
    let classes = tensor.shape()[1];
    let mut best_class = 0usize;
    let mut best_value = tensor[[sample_idx, 0, y, x]];
    for class_idx in 1..classes {
        let value = tensor[[sample_idx, class_idx, y, x]];
        if value > best_value {
            best_class = class_idx;
            best_value = value;
        }
    }
    best_class
}

fn save_rgb_image(image: &image::RgbImage, path: &str) -> Result<(), GraphError> {
    image
        .save(path)
        .map_err(|err| GraphError::ComputationError(format!("保存图像失败 {path}: {err}")))
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
