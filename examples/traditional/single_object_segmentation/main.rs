//! # Single Object Segmentation 示例
//!
//! 用极小的合成图像演示二值语义分割完整闭环：
//! - 内置生成 16x16 矩形 / 圆形 mask，不依赖下载数据
//! - 小型 CNN 保持空间维度，输出 `[N, 1, H, W]` logits
//! - 使用 4D `BCEWithLogits` 训练，使用 Pixel Accuracy / IoU 评估
//!
//! ## 运行
//! ```bash
//! cargo run --example single_object_segmentation
//! ```

mod model;

use model::SingleObjectSegmentationNet;
use only_torch::data::{DataLoader, SyntheticRng, TensorDataset};
use only_torch::metrics::{binary_iou, pixel_accuracy};
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;
use std::time::Instant;

const IMAGE_SIZE: usize = 16;
const OVERLAY_SCALE: u32 = 12;
const TRAIN_SAMPLES: usize = 96;
const TEST_SAMPLES: usize = 32;
const BATCH_SIZE: usize = 24;
const MAX_EPOCHS: usize = 20;
const LEARNING_RATE: f32 = 0.03;
const TARGET_IOU: f32 = 0.90;

fn main() -> Result<(), GraphError> {
    let total_start = Instant::now();
    println!("=== Single Object Segmentation 示例 ===\n");

    let (train_x, train_y) = generate_dataset(TRAIN_SAMPLES, 42);
    let (test_x, test_y) = generate_dataset(TEST_SAMPLES, 2026);
    let train_loader = DataLoader::new(TensorDataset::new(train_x, train_y), BATCH_SIZE)
        .shuffle(true)
        .seed(7);

    let graph = Graph::new_with_seed(42);
    let model = SingleObjectSegmentationNet::new(&graph)?;
    let mut optimizer = Adam::new(&graph, &model.parameters(), LEARNING_RATE);

    let param_count: usize = model
        .parameters()
        .iter()
        .filter_map(|p| p.value().ok().flatten())
        .map(|t| t.shape().iter().product::<usize>())
        .sum();

    println!("任务: 16x16 合成图像单目标二值语义分割（固定 seed 的矩形 / 圆形样本）");
    println!("网络: Conv(1→4) → ReLU → Conv(4→4) → ReLU → Conv(4→1)");
    println!("损失: BCEWithLogitsLoss，指标: Pixel Accuracy + Binary IoU");
    println!("训练样本: {TRAIN_SAMPLES}, 测试样本: {TEST_SAMPLES}, 参数量: {param_count}\n");

    let mut best_iou = 0.0f32;
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
        let (acc, iou) = evaluate(&model, &test_x, &test_y)?;
        best_acc = best_acc.max(acc);
        best_iou = best_iou.max(iou);

        println!(
            "Epoch {:2}: loss={:.4}, pixel_acc={:.1}%, IoU={:.1}%, {:.2}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            acc * 100.0,
            iou * 100.0,
            epoch_start.elapsed().as_secs_f32()
        );

        if iou >= TARGET_IOU {
            println!("\n达到目标 IoU {:.1}%，提前停止。", iou * 100.0);
            break;
        }
    }

    println!("\n=== 预测示例（# = 前景，. = 背景）===");
    print_sample_prediction(&model, &test_x, &test_y, 0)?;
    save_sample_visualizations(&model, &test_x, &test_y, 0)?;
    println!("测试输入图: examples/traditional/single_object_segmentation/test_in.png");
    println!("测试输出图: examples/traditional/single_object_segmentation/test_out.png");

    let vis_result = graph.visualize_snapshot(
        "examples/traditional/single_object_segmentation/single_object_segmentation",
    )?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    println!(
        "\n最佳 Pixel Accuracy: {:.1}%，最佳 IoU: {:.1}%，总耗时: {:.1}s",
        best_acc * 100.0,
        best_iou * 100.0,
        total_start.elapsed().as_secs_f32()
    );

    if best_iou >= TARGET_IOU {
        println!("Single object segmentation 训练成功。");
        Ok(())
    } else {
        Err(GraphError::ComputationError(format!(
            "Single object segmentation 未达到目标 IoU {:.1}%，最佳 {:.1}%",
            TARGET_IOU * 100.0,
            best_iou * 100.0
        )))
    }
}

fn evaluate(
    model: &SingleObjectSegmentationNet,
    inputs: &Tensor,
    targets: &Tensor,
) -> Result<(f32, f32), GraphError> {
    let probs = model.predict_probs(inputs)?;
    let probs = probs.value()?.unwrap();
    let acc = pixel_accuracy(&probs, targets, 0.5);
    let iou = binary_iou(&probs, targets, 0.5);
    Ok((acc.value(), iou.value()))
}

fn print_sample_prediction(
    model: &SingleObjectSegmentationNet,
    inputs: &Tensor,
    targets: &Tensor,
    sample_idx: usize,
) -> Result<(), GraphError> {
    let probs = model.predict_probs(inputs)?;
    let probs = probs.value()?.unwrap();

    println!("目标 mask        预测 mask");
    for y in 0..IMAGE_SIZE {
        let target_row = mask_row(targets, sample_idx, y, 0.5);
        let pred_row = mask_row(&probs, sample_idx, y, 0.5);
        println!("{target_row}    {pred_row}");
    }

    Ok(())
}

fn mask_row(tensor: &Tensor, sample_idx: usize, y: usize, threshold: f32) -> String {
    (0..IMAGE_SIZE)
        .map(|x| {
            if tensor[[sample_idx, 0, y, x]] >= threshold {
                '#'
            } else {
                '.'
            }
        })
        .collect()
}

fn save_sample_visualizations(
    model: &SingleObjectSegmentationNet,
    inputs: &Tensor,
    _targets: &Tensor,
    sample_idx: usize,
) -> Result<(), GraphError> {
    use image::{ImageBuffer, Rgb};

    let probs = model.predict_probs(inputs)?;
    let probs = probs.value()?.unwrap();

    let panel_size = IMAGE_SIZE as u32 * OVERLAY_SCALE;
    let mut input_img = ImageBuffer::from_pixel(panel_size, panel_size, Rgb([245, 245, 245]));
    let mut overlay_img = ImageBuffer::from_pixel(panel_size, panel_size, Rgb([245, 245, 245]));

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let base = (inputs[[sample_idx, 0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            let base_rgb = [base, base, base];
            let pred_positive = probs[[sample_idx, 0, y, x]] >= 0.5;
            let pred_prob = probs[[sample_idx, 0, y, x]].clamp(0.0, 1.0);

            fill_scaled_pixel(&mut input_img, 0, x, y, base_rgb);
            fill_scaled_pixel(
                &mut overlay_img,
                0,
                x,
                y,
                overlay(base_rgb, pred_positive, [255, 64, 64], pred_prob * 0.60),
            );
        }
    }

    save_rgb_image(
        &input_img,
        "examples/traditional/single_object_segmentation/test_in.png",
    )?;
    save_rgb_image(
        &overlay_img,
        "examples/traditional/single_object_segmentation/test_out.png",
    )?;

    Ok(())
}

fn save_rgb_image(image: &image::RgbImage, path: &str) -> Result<(), GraphError> {
    image
        .save(path)
        .map_err(|err| GraphError::ComputationError(format!("保存图像失败 {path}: {err}")))
}

fn fill_scaled_pixel(
    canvas: &mut image::RgbImage,
    x_offset: u32,
    x: usize,
    y: usize,
    color: [u8; 3],
) {
    let x0 = x_offset + x as u32 * OVERLAY_SCALE;
    let y0 = y as u32 * OVERLAY_SCALE;
    for dy in 0..OVERLAY_SCALE {
        for dx in 0..OVERLAY_SCALE {
            canvas.put_pixel(x0 + dx, y0 + dy, image::Rgb(color));
        }
    }
}

fn overlay(base: [u8; 3], enabled: bool, mask_color: [u8; 3], alpha: f32) -> [u8; 3] {
    if !enabled {
        return base;
    }

    [
        blend_channel(base[0], mask_color[0], alpha),
        blend_channel(base[1], mask_color[1], alpha),
        blend_channel(base[2], mask_color[2], alpha),
    ]
}

fn blend_channel(base: u8, overlay: u8, alpha: f32) -> u8 {
    ((base as f32 * (1.0 - alpha)) + (overlay as f32 * alpha)).round() as u8
}

fn generate_dataset(n: usize, seed: u64) -> (Tensor, Tensor) {
    let mut images = Vec::with_capacity(n * IMAGE_SIZE * IMAGE_SIZE);
    let mut masks = Vec::with_capacity(n * IMAGE_SIZE * IMAGE_SIZE);

    for sample_idx in 0..n {
        let config = ShapeConfig::new(sample_idx, seed);
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let inside = config.contains(x, y);
                let noise = deterministic_noise(seed, sample_idx, x, y);
                let pixel = if inside {
                    0.75 + 0.20 * noise
                } else {
                    0.05 + 0.15 * noise
                };

                images.push(pixel);
                masks.push(if inside { 1.0 } else { 0.0 });
            }
        }
    }

    (
        Tensor::new(&images, &[n, 1, IMAGE_SIZE, IMAGE_SIZE]),
        Tensor::new(&masks, &[n, 1, IMAGE_SIZE, IMAGE_SIZE]),
    )
}

#[derive(Clone, Copy)]
struct ShapeConfig {
    kind: ShapeKind,
    cx: isize,
    cy: isize,
    radius: isize,
    half_w: isize,
    half_h: isize,
}

impl ShapeConfig {
    fn new(sample_idx: usize, seed: u64) -> Self {
        let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64]);
        Self {
            kind: if rng.next_bool() {
                ShapeKind::Rectangle
            } else {
                ShapeKind::Circle
            },
            cx: rng.isize_range(4..12),
            cy: rng.isize_range(4..12),
            radius: rng.isize_range(2..5),
            half_w: rng.isize_range(2..5),
            half_h: rng.isize_range(2..5),
        }
    }

    fn contains(&self, x: usize, y: usize) -> bool {
        let x = x as isize;
        let y = y as isize;
        match self.kind {
            ShapeKind::Rectangle => {
                (x - self.cx).abs() <= self.half_w && (y - self.cy).abs() <= self.half_h
            }
            ShapeKind::Circle => {
                let dx = x - self.cx;
                let dy = y - self.cy;
                dx * dx + dy * dy <= self.radius * self.radius
            }
        }
    }
}

#[derive(Clone, Copy)]
enum ShapeKind {
    Rectangle,
    Circle,
}

fn deterministic_noise(seed: u64, sample_idx: usize, x: usize, y: usize) -> f32 {
    let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64, x as u64, y as u64]);
    rng.next_f32()
}
