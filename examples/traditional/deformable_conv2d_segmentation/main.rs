//! # DeformableConv2d Semantic Segmentation 示例
//!
//! 这是 P4 Deformable Conv2d 的传统手写网络基线：先把算子作为普通
//! `Layer` 放进二值前景分割网络，确认它能独立训练，再接入演化搜索空间。
//!
//! ```bash
//! cargo run --example deformable_conv2d_segmentation
//! ```

mod model;

use model::DeformableSegmentationNet;
use only_torch::data::{DataLoader, SyntheticRng, TensorDataset};
use only_torch::metrics::{binary_iou, pixel_accuracy};
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;
use std::time::Instant;

const IMAGE_SIZE: usize = 16;
const MAX_OBJECTS: usize = 3;
const TRAIN_SAMPLES: usize = 12;
const TEST_SAMPLES: usize = 4;
const BATCH_SIZE: usize = 4;
const MAX_EPOCHS: usize = 5;
const LEARNING_RATE: f32 = 0.012;
const OVERLAY_SCALE: u32 = 10;

fn main() -> Result<(), GraphError> {
    let total_start = Instant::now();
    println!("=== DeformableConv2d Semantic Segmentation 示例 ===\n");

    let (train_x, train_y) = generate_dataset(TRAIN_SAMPLES, 42);
    let (test_x, test_y) = generate_dataset(TEST_SAMPLES, 2026);
    let train_loader = DataLoader::new(TensorDataset::new(train_x, train_y), BATCH_SIZE)
        .shuffle(true)
        .seed(17);

    let graph = Graph::new_with_seed(42);
    let model = DeformableSegmentationNet::new(&graph)?;
    let mut optimizer = Adam::new(&graph, &model.parameters(), LEARNING_RATE);
    let param_count: usize = model
        .parameters()
        .iter()
        .filter_map(|p| p.value().ok().flatten())
        .map(|t| t.shape().iter().product::<usize>())
        .sum();

    println!("任务: 16x16 合成图像二值前景分割（1..3 个可重叠形状）");
    println!("输出: 1 个 foreground 概率通道，绿色越亮表示前景概率越高");
    println!("网络: Conv -> DeformableConv2d(offset-only) -> Conv -> 1x1 head");
    println!("损失: BCEWithLogitsLoss，指标: Pixel Accuracy + Binary IoU");
    println!("训练样本: {TRAIN_SAMPLES}, 测试样本: {TEST_SAMPLES}, 参数量: {param_count}\n");

    let mut best_acc = 0.0f32;
    let mut best_iou = 0.0f32;
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
        best_iou = best_iou.max(report.binary_iou);
        println!(
            "Epoch {:2}: loss={:.4}, pixel_acc={:.1}%, binary_iou={:.1}%, {:.2}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            report.pixel_accuracy * 100.0,
            report.binary_iou * 100.0,
            epoch_start.elapsed().as_secs_f32()
        );
    }

    let report = evaluate(&model, &test_x, &test_y)?;
    println!("\n=== 最终指标 ===");
    println!("Pixel Accuracy: {:.1}%", report.pixel_accuracy * 100.0);
    println!("Binary IoU: {:.1}%", report.binary_iou * 100.0);

    save_sample_visualizations(&model, &test_x, 0)?;
    println!("测试输入图: examples/traditional/deformable_conv2d_segmentation/test_in.png");
    println!("测试输出图: examples/traditional/deformable_conv2d_segmentation/test_out.png");

    let vis_result = graph.visualize_snapshot(
        "examples/traditional/deformable_conv2d_segmentation/deformable_conv2d_segmentation",
    )?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    } else if let Some(hint) = &vis_result.graphviz_hint {
        println!("Graphviz PNG 生成提示: {hint}");
    }

    println!(
        "\n最佳 Pixel Accuracy: {:.1}%，最佳 Binary IoU: {:.1}%，总耗时: {:.1}s",
        best_acc * 100.0,
        best_iou * 100.0,
        total_start.elapsed().as_secs_f32()
    );
    Ok(())
}

struct EvalReport {
    pixel_accuracy: f32,
    binary_iou: f32,
}

fn evaluate(
    model: &DeformableSegmentationNet,
    inputs: &Tensor,
    targets: &Tensor,
) -> Result<EvalReport, GraphError> {
    let probs = model.predict_probs(inputs)?;
    let probs = probs.value()?.unwrap();
    Ok(EvalReport {
        pixel_accuracy: pixel_accuracy(&probs, targets, 0.5).value(),
        binary_iou: binary_iou(&probs, targets, 0.5).value(),
    })
}

fn generate_dataset(n: usize, seed: u64) -> (Tensor, Tensor) {
    let mut images = Vec::with_capacity(n * IMAGE_SIZE * IMAGE_SIZE);
    let mut masks = Vec::with_capacity(n * IMAGE_SIZE * IMAGE_SIZE);

    for sample_idx in 0..n {
        let objects = generate_objects(sample_idx, seed);
        let mut foreground_map = vec![0.0f32; IMAGE_SIZE * IMAGE_SIZE];

        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let mut foreground = false;
                for object in &objects {
                    if object.contains(x, y) {
                        foreground = true;
                    }
                }
                foreground_map[y * IMAGE_SIZE + x] = if foreground { 1.0 } else { 0.0 };

                let noise = deterministic_noise(seed, sample_idx, x, y);
                let gradient = (x as f32 + y as f32) / (2.0 * (IMAGE_SIZE - 1) as f32);
                let base = if foreground { 0.82 } else { 0.04 };
                images.push((base + 0.08 * gradient + 0.08 * noise).clamp(0.0, 1.0));
            }
        }

        masks.extend(foreground_map);
    }

    (
        Tensor::new(&images, &[n, 1, IMAGE_SIZE, IMAGE_SIZE]),
        Tensor::new(&masks, &[n, 1, IMAGE_SIZE, IMAGE_SIZE]),
    )
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
                half_w: obj_rng.isize_range(2..5),
                half_h: obj_rng.isize_range(2..5),
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
                dy >= -self.half_h
                    && dy <= self.half_h
                    && dx.abs() <= self.half_w * (self.half_h - dy) / (2 * self.half_h).max(1)
            }
        }
    }
}

fn deterministic_noise(seed: u64, sample_idx: usize, x: usize, y: usize) -> f32 {
    let mut value = seed
        ^ ((sample_idx as u64).wrapping_mul(0x9E37_79B9))
        ^ ((x as u64).wrapping_mul(0x85EB_CA6B))
        ^ ((y as u64).wrapping_mul(0xC2B2_AE35));
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^= value >> 33;
    (value as f64 / u64::MAX as f64) as f32
}

fn save_sample_visualizations(
    model: &DeformableSegmentationNet,
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
            fill_scaled_pixel(
                &mut output_img,
                x,
                y,
                foreground_color(probs[[sample_idx, 0, y, x]]),
            );
        }
    }

    save_rgb_image(
        &input_img,
        "examples/traditional/deformable_conv2d_segmentation/test_in.png",
    )?;
    save_rgb_image(
        &output_img,
        "examples/traditional/deformable_conv2d_segmentation/test_out.png",
    )?;
    Ok(())
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

fn foreground_color(value: f32) -> [u8; 3] {
    let v = value.clamp(0.0, 1.0);
    let base = (32.0 * (1.0 - v)) as u8;
    let green = (32.0 + 220.0 * v) as u8;
    [base, green, base]
}
