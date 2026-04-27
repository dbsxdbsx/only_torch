//! # Multi Instance Segmentation 示例
//!
//! 用极小的合成图像演示固定两实例 mask 的训练闭环：
//! - 每张 16x16 图固定恰好 2 个非重叠实例，不是“最多两个”
//! - 输出 `[N, 2, H, W]` logits，两个通道分别是固定 slot 的实例 mask
//! - 不实现类别、confidence、presence head、NMS、matching 或变长实例列表
//! - 使用 4D `BCEWithLogits` 训练，使用 Mean Instance IoU 评估
//!
//! 这个示例是教学用 toy instance segmentation：它只验证“一个输入图对应多个
//! mask 通道”的最小闭环，不代表通用 Mask R-CNN / YOLO-seg 级实例分割系统。
//!
//! ## 运行
//! ```bash
//! cargo run --example multi_instance_segmentation
//! ```

mod model;

use model::MultiInstanceSegmentationNet;
use only_torch::data::{DataLoader, SyntheticRng, TensorDataset};
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;
use std::time::Instant;

const IMAGE_SIZE: usize = 16;
const INSTANCE_SLOTS: usize = 2;
const OVERLAY_SCALE: u32 = 12;
const TRAIN_SAMPLES: usize = 128;
const TEST_SAMPLES: usize = 32;
const BATCH_SIZE: usize = 32;
const MAX_EPOCHS: usize = 30;
const LEARNING_RATE: f32 = 0.03;
const TARGET_MEAN_IOU: f32 = 0.85;
const MASK_THRESHOLD: f32 = 0.5;

fn main() -> Result<(), GraphError> {
    let total_start = Instant::now();
    println!("=== Multi Instance Segmentation 示例 ===\n");

    let (train_x, train_y) = generate_dataset(TRAIN_SAMPLES, 42);
    let (test_x, test_y) = generate_dataset(TEST_SAMPLES, 2026);
    let train_loader = DataLoader::new(TensorDataset::new(train_x, train_y), BATCH_SIZE)
        .shuffle(true)
        .seed(17);

    let graph = Graph::new_with_seed(42);
    let model = MultiInstanceSegmentationNet::new(&graph)?;
    let mut optimizer = Adam::new(&graph, &model.parameters(), LEARNING_RATE);

    let param_count: usize = model
        .parameters()
        .iter()
        .filter_map(|p| p.value().ok().flatten())
        .map(|t| t.shape().iter().product::<usize>())
        .sum();

    println!("任务: 16x16 合成图像固定两实例分割（每图恰好 2 个非重叠矩形）");
    println!("Slot: channel 0 = 左侧实例，channel 1 = 右侧实例；无类别 / confidence / 空 slot");
    println!("网络: Conv(1→8) → ReLU → Conv(8→8) → ReLU → Conv(8→2)");
    println!("损失: BCEWithLogitsLoss，指标: Mean Instance IoU + Slot Pixel Accuracy");
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
            "Epoch {:2}: loss={:.4}, slot_acc={:.1}%, mean_instance_iou={:.1}%, {:.2}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            acc * 100.0,
            iou * 100.0,
            epoch_start.elapsed().as_secs_f32()
        );

        if iou >= TARGET_MEAN_IOU {
            println!(
                "\n达到目标 Mean Instance IoU {:.1}%，提前停止。",
                iou * 100.0
            );
            break;
        }
    }

    println!("\n=== 预测示例（# = 当前 slot 前景，. = 背景）===");
    print_sample_prediction(&model, &test_x, &test_y, 0)?;
    save_sample_visualizations(&model, &test_x, 0)?;
    println!("测试输入图: examples/traditional/multi_instance_segmentation/test_in.png");
    println!("测试输出图: examples/traditional/multi_instance_segmentation/test_out.png");

    let vis_result = graph.visualize_snapshot(
        "examples/traditional/multi_instance_segmentation/multi_instance_segmentation",
    )?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    println!(
        "\n最佳 Slot Pixel Accuracy: {:.1}%，最佳 Mean Instance IoU: {:.1}%，总耗时: {:.1}s",
        best_acc * 100.0,
        best_iou * 100.0,
        total_start.elapsed().as_secs_f32()
    );

    if best_iou >= TARGET_MEAN_IOU {
        println!("Multi instance segmentation 训练成功。");
        Ok(())
    } else {
        Err(GraphError::ComputationError(format!(
            "Multi instance segmentation 未达到目标 Mean Instance IoU {:.1}%，最佳 {:.1}%",
            TARGET_MEAN_IOU * 100.0,
            best_iou * 100.0
        )))
    }
}

fn evaluate(
    model: &MultiInstanceSegmentationNet,
    inputs: &Tensor,
    targets: &Tensor,
) -> Result<(f32, f32), GraphError> {
    let probs = model.predict_probs(inputs)?;
    let probs = probs.value()?.unwrap();
    Ok((
        slot_pixel_accuracy(&probs, targets, MASK_THRESHOLD),
        mean_instance_iou(&probs, targets, MASK_THRESHOLD),
    ))
}

fn slot_pixel_accuracy(predictions: &Tensor, targets: &Tensor, threshold: f32) -> f32 {
    assert_eq!(predictions.shape(), targets.shape());
    let total = predictions.size();
    if total == 0 {
        return 0.0;
    }

    let correct = predictions
        .to_vec()
        .into_iter()
        .zip(targets.to_vec())
        .filter(|(pred, target)| (*pred >= threshold) == (*target >= threshold))
        .count();
    correct as f32 / total as f32
}

fn mean_instance_iou(predictions: &Tensor, targets: &Tensor, threshold: f32) -> f32 {
    assert_eq!(predictions.shape(), targets.shape());
    assert!(
        predictions.shape().len() == 4 && predictions.shape()[1] == INSTANCE_SLOTS,
        "mean_instance_iou: 期望 shape=[N, {INSTANCE_SLOTS}, H, W]，实际 {:?}",
        predictions.shape()
    );

    let n = predictions.shape()[0];
    if n == 0 {
        return 0.0;
    }

    let mut total_iou = 0.0f32;
    for sample_idx in 0..n {
        for slot in 0..INSTANCE_SLOTS {
            total_iou += instance_iou(predictions, targets, sample_idx, slot, threshold);
        }
    }
    total_iou / (n * INSTANCE_SLOTS) as f32
}

fn instance_iou(
    predictions: &Tensor,
    targets: &Tensor,
    sample_idx: usize,
    slot: usize,
    threshold: f32,
) -> f32 {
    let mut intersection = 0usize;
    let mut union = 0usize;

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let pred_positive = predictions[[sample_idx, slot, y, x]] >= threshold;
            let target_positive = targets[[sample_idx, slot, y, x]] >= threshold;
            if pred_positive && target_positive {
                intersection += 1;
            }
            if pred_positive || target_positive {
                union += 1;
            }
        }
    }

    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

fn print_sample_prediction(
    model: &MultiInstanceSegmentationNet,
    inputs: &Tensor,
    targets: &Tensor,
    sample_idx: usize,
) -> Result<(), GraphError> {
    let probs = model.predict_probs(inputs)?;
    let probs = probs.value()?.unwrap();

    for slot in 0..INSTANCE_SLOTS {
        println!("slot {slot} 目标 mask        slot {slot} 预测 mask");
        for y in 0..IMAGE_SIZE {
            let target_row = mask_row(targets, sample_idx, slot, y, MASK_THRESHOLD);
            let pred_row = mask_row(&probs, sample_idx, slot, y, MASK_THRESHOLD);
            println!("{target_row}    {pred_row}");
        }
        println!();
    }

    Ok(())
}

fn mask_row(tensor: &Tensor, sample_idx: usize, slot: usize, y: usize, threshold: f32) -> String {
    (0..IMAGE_SIZE)
        .map(|x| {
            if tensor[[sample_idx, slot, y, x]] >= threshold {
                '#'
            } else {
                '.'
            }
        })
        .collect()
}

fn save_sample_visualizations(
    model: &MultiInstanceSegmentationNet,
    inputs: &Tensor,
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
            let mut out_rgb = base_rgb;

            fill_scaled_pixel(&mut input_img, x, y, base_rgb);
            for slot in 0..INSTANCE_SLOTS {
                let prob = probs[[sample_idx, slot, y, x]].clamp(0.0, 1.0);
                let enabled = prob >= MASK_THRESHOLD;
                out_rgb = overlay(out_rgb, enabled, slot_color(slot), prob * 0.65);
            }
            fill_scaled_pixel(&mut overlay_img, x, y, out_rgb);
        }
    }

    save_rgb_image(
        &input_img,
        "examples/traditional/multi_instance_segmentation/test_in.png",
    )?;
    save_rgb_image(
        &overlay_img,
        "examples/traditional/multi_instance_segmentation/test_out.png",
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

fn slot_color(slot: usize) -> [u8; 3] {
    match slot {
        0 => [255, 64, 64],
        1 => [64, 128, 255],
        _ => [64, 220, 64],
    }
}

/// 生成固定两实例数据。
///
/// 每张图恰好包含两个非重叠矩形：slot 0 固定在左半区，slot 1 固定在右半区。
/// 这种固定 slot 约定是为了让 toy 示例稳定收敛；真实实例分割通常还需要
/// proposal、matching 或 presence/objectness 之类机制来处理可变数量实例。
fn generate_dataset(n: usize, seed: u64) -> (Tensor, Tensor) {
    let mut images = Vec::with_capacity(n * IMAGE_SIZE * IMAGE_SIZE);
    let mut masks = Vec::with_capacity(n * INSTANCE_SLOTS * IMAGE_SIZE * IMAGE_SIZE);

    for sample_idx in 0..n {
        let instances = InstancePair::new(sample_idx, seed);

        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let slot0 = instances.rects[0].contains(x, y);
                let slot1 = instances.rects[1].contains(x, y);
                let noise = deterministic_noise(seed, sample_idx, x, y);
                let x_hint = x as f32 / (IMAGE_SIZE - 1) as f32;

                let pixel = if slot0 {
                    0.68 + 0.06 * x_hint + 0.18 * noise
                } else if slot1 {
                    0.78 + 0.06 * x_hint + 0.16 * noise
                } else {
                    0.04 + 0.10 * x_hint + 0.12 * noise
                };
                images.push(pixel.clamp(0.0, 1.0));
            }
        }

        for slot in 0..INSTANCE_SLOTS {
            for y in 0..IMAGE_SIZE {
                for x in 0..IMAGE_SIZE {
                    masks.push(if instances.rects[slot].contains(x, y) {
                        1.0
                    } else {
                        0.0
                    });
                }
            }
        }
    }

    (
        Tensor::new(&images, &[n, 1, IMAGE_SIZE, IMAGE_SIZE]),
        Tensor::new(&masks, &[n, INSTANCE_SLOTS, IMAGE_SIZE, IMAGE_SIZE]),
    )
}

struct InstancePair {
    rects: [InstanceRect; INSTANCE_SLOTS],
}

impl InstancePair {
    fn new(sample_idx: usize, seed: u64) -> Self {
        let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64]);
        let left = InstanceRect {
            cx: rng.isize_range(3..6),
            cy: rng.isize_range(4..12),
            half_w: rng.isize_range(1..3),
            half_h: rng.isize_range(1..4),
        };
        let right = InstanceRect {
            cx: rng.isize_range(10..13),
            cy: rng.isize_range(4..12),
            half_w: rng.isize_range(1..3),
            half_h: rng.isize_range(1..4),
        };

        Self {
            rects: [left, right],
        }
    }
}

#[derive(Clone, Copy)]
struct InstanceRect {
    cx: isize,
    cy: isize,
    half_w: isize,
    half_h: isize,
}

impl InstanceRect {
    fn contains(&self, x: usize, y: usize) -> bool {
        let x = x as isize;
        let y = y as isize;
        (x - self.cx).abs() <= self.half_w && (y - self.cy).abs() <= self.half_h
    }
}

fn deterministic_noise(seed: u64, sample_idx: usize, x: usize, y: usize) -> f32 {
    let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64, x as u64, y as u64]);
    rng.next_f32()
}
