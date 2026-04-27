//! # Overlapping Fixed Slot Instance Segmentation 示例
//!
//! 这个示例是 fixed-slot instance segmentation lite：
//! - 固定 64x64 单通道图像，一个 batch 内尺寸固定。
//! - 每张图随机生成 1 到 3 个实例，实例允许重叠。
//! - 输出 `[N, 3, H, W]` logits，三个通道代表按中心 x 坐标排序后的实例 slot。
//! - 重叠区域只训练 topmost 可见实例的 visible mask，不做 amodal mask。
//! - 空 slot 目标全 0，额外报告 empty-slot accuracy。
//!
//! 它比 legacy `multi_instance_segmentation` 更难，但仍不是通用 Mask R-CNN / YOLO-seg。
//!
//! ## 运行
//! ```bash
//! cargo run --example overlapping_fixed_slot_instance_segmentation
//! ```

mod model;

use model::OverlappingFixedSlotInstanceSegmentationNet;
use only_torch::data::{DataLoader, TensorDataset};
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;
use std::time::Instant;

const IMAGE_SIZE: usize = 64;
const INSTANCE_SLOTS: usize = 3;
const OVERLAY_SCALE: u32 = 5;
const TRAIN_SAMPLES: usize = 96;
const TEST_SAMPLES: usize = 32;
const BATCH_SIZE: usize = 16;
const MAX_EPOCHS: usize = 24;
const LEARNING_RATE: f32 = 0.02;
const TARGET_MEAN_IOU: f32 = 0.45;
const MASK_THRESHOLD: f32 = 0.5;

fn main() -> Result<(), GraphError> {
    let total_start = Instant::now();
    println!("=== Overlapping Fixed Slot Instance Segmentation 示例 ===\n");

    let (train_x, train_y) = generate_dataset(TRAIN_SAMPLES, 42);
    let (test_x, test_y) = generate_dataset(TEST_SAMPLES, 2026);
    let train_loader = DataLoader::new(TensorDataset::new(train_x, train_y), BATCH_SIZE)
        .shuffle(true)
        .seed(17);

    let graph = Graph::new_with_seed(42);
    let model = OverlappingFixedSlotInstanceSegmentationNet::new(&graph)?;
    let mut optimizer = Adam::new(&graph, &model.parameters(), LEARNING_RATE);

    let param_count: usize = model
        .parameters()
        .iter()
        .filter_map(|p| p.value().ok().flatten())
        .map(|t| t.shape().iter().product::<usize>())
        .sum();

    println!("任务: 64x64 合成图像固定 slot 实例分割（1..3 个可重叠实例）");
    println!("Slot: 按实例中心 x 坐标从左到右排序；空 slot 目标全 0");
    println!("重叠: 只预测 topmost visible mask，不预测 amodal mask");
    println!("网络: Conv(1→12) → ReLU → Conv(12→16) → ReLU → Conv(16→3)");
    println!("损失: BCEWithLogitsLoss，指标: Valid-slot IoU + Empty-slot Accuracy");
    println!("训练样本: {TRAIN_SAMPLES}, 测试样本: {TEST_SAMPLES}, 参数量: {param_count}\n");

    let mut best_iou = 0.0f32;
    let mut best_empty_acc = 0.0f32;

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
        best_iou = best_iou.max(report.mean_valid_slot_iou);
        best_empty_acc = best_empty_acc.max(report.empty_slot_accuracy);

        println!(
            "Epoch {:2}: loss={:.4}, valid_slot_iou={:.1}%, empty_slot_acc={:.1}%, {:.2}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            report.mean_valid_slot_iou * 100.0,
            report.empty_slot_accuracy * 100.0,
            epoch_start.elapsed().as_secs_f32()
        );

        if report.mean_valid_slot_iou >= TARGET_MEAN_IOU {
            println!(
                "\n达到目标 Valid-slot IoU {:.1}%，提前停止。",
                report.mean_valid_slot_iou * 100.0
            );
            break;
        }
    }

    let report = evaluate(&model, &test_x, &test_y)?;
    println!(
        "\n最终 Valid-slot IoU: {:.1}%，Empty-slot Accuracy: {:.1}%",
        report.mean_valid_slot_iou * 100.0,
        report.empty_slot_accuracy * 100.0
    );

    save_sample_visualizations(&model, &test_x, 0)?;
    println!(
        "测试输入图: examples/traditional/overlapping_fixed_slot_instance_segmentation/test_in.png"
    );
    println!(
        "测试输出图: examples/traditional/overlapping_fixed_slot_instance_segmentation/test_out.png"
    );

    let vis_result = graph.visualize_snapshot(
        "examples/traditional/overlapping_fixed_slot_instance_segmentation/overlapping_fixed_slot_instance_segmentation",
    )?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    println!(
        "\n最佳 Valid-slot IoU: {:.1}%，最佳 Empty-slot Accuracy: {:.1}%，总耗时: {:.1}s",
        best_iou * 100.0,
        best_empty_acc * 100.0,
        total_start.elapsed().as_secs_f32()
    );

    if best_iou >= TARGET_MEAN_IOU {
        println!("Overlapping fixed slot instance segmentation 训练成功。");
        Ok(())
    } else {
        Err(GraphError::ComputationError(format!(
            "Overlapping fixed slot instance segmentation 未达到目标 Valid-slot IoU {:.1}%，最佳 {:.1}%",
            TARGET_MEAN_IOU * 100.0,
            best_iou * 100.0
        )))
    }
}

struct EvalReport {
    mean_valid_slot_iou: f32,
    empty_slot_accuracy: f32,
}

fn evaluate(
    model: &OverlappingFixedSlotInstanceSegmentationNet,
    inputs: &Tensor,
    targets: &Tensor,
) -> Result<EvalReport, GraphError> {
    let probs = model.predict_probs(inputs)?;
    let probs = probs.value()?.unwrap();
    Ok(EvalReport {
        mean_valid_slot_iou: mean_valid_slot_iou(&probs, targets, MASK_THRESHOLD),
        empty_slot_accuracy: empty_slot_accuracy(&probs, targets, MASK_THRESHOLD),
    })
}

fn mean_valid_slot_iou(predictions: &Tensor, targets: &Tensor, threshold: f32) -> f32 {
    assert_eq!(predictions.shape(), targets.shape());
    let n = predictions.shape()[0];
    let mut total_iou = 0.0f32;
    let mut valid_slots = 0usize;

    for sample in 0..n {
        for slot in 0..INSTANCE_SLOTS {
            if slot_has_target(targets, sample, slot) {
                total_iou += instance_iou(predictions, targets, sample, slot, threshold);
                valid_slots += 1;
            }
        }
    }

    if valid_slots == 0 {
        0.0
    } else {
        total_iou / valid_slots as f32
    }
}

fn empty_slot_accuracy(predictions: &Tensor, targets: &Tensor, threshold: f32) -> f32 {
    let n = predictions.shape()[0];
    let mut empty_slots = 0usize;
    let mut correct_empty_slots = 0usize;

    for sample in 0..n {
        for slot in 0..INSTANCE_SLOTS {
            if !slot_has_target(targets, sample, slot) {
                empty_slots += 1;
                if !slot_has_prediction(predictions, sample, slot, threshold) {
                    correct_empty_slots += 1;
                }
            }
        }
    }

    if empty_slots == 0 {
        1.0
    } else {
        correct_empty_slots as f32 / empty_slots as f32
    }
}

fn instance_iou(
    predictions: &Tensor,
    targets: &Tensor,
    sample: usize,
    slot: usize,
    threshold: f32,
) -> f32 {
    let mut intersection = 0usize;
    let mut union = 0usize;
    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let pred_positive = predictions[[sample, slot, y, x]] >= threshold;
            let target_positive = targets[[sample, slot, y, x]] >= threshold;
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

fn slot_has_target(targets: &Tensor, sample: usize, slot: usize) -> bool {
    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            if targets[[sample, slot, y, x]] >= 0.5 {
                return true;
            }
        }
    }
    false
}

fn slot_has_prediction(predictions: &Tensor, sample: usize, slot: usize, threshold: f32) -> bool {
    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            if predictions[[sample, slot, y, x]] >= threshold {
                return true;
            }
        }
    }
    false
}

fn generate_dataset(n: usize, seed: u64) -> (Tensor, Tensor) {
    let mut images = Vec::with_capacity(n * IMAGE_SIZE * IMAGE_SIZE);
    let mut masks = Vec::with_capacity(n * INSTANCE_SLOTS * IMAGE_SIZE * IMAGE_SIZE);

    for sample_idx in 0..n {
        let instances = generate_instances(sample_idx, seed);
        let visible_slot_map = visible_slot_map(&instances);

        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let visible_slot = visible_slot_map[y * IMAGE_SIZE + x];
                let noise = deterministic_noise(seed, sample_idx, x, y);
                let gradient = x as f32 / (IMAGE_SIZE - 1) as f32;
                let base = match visible_slot {
                    Some(slot) => 0.35 + slot as f32 * 0.24,
                    None => 0.04,
                };
                let pixel = base + 0.08 * gradient + 0.10 * noise;
                images.push(pixel.clamp(0.0, 1.0));
            }
        }

        for slot in 0..INSTANCE_SLOTS {
            for y in 0..IMAGE_SIZE {
                for x in 0..IMAGE_SIZE {
                    masks.push(if visible_slot_map[y * IMAGE_SIZE + x] == Some(slot) {
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

fn generate_instances(sample_idx: usize, seed: u64) -> Vec<InstanceObject> {
    let h = mix(seed ^ sample_idx as u64);
    let count = 1 + (h % INSTANCE_SLOTS as u64) as usize;
    let mut instances: Vec<InstanceObject> = (0..count)
        .map(|idx| {
            let oh = mix(h ^ (idx as u64 + 1).wrapping_mul(0x517c_c1b7_2722_0a95));
            let half_w = 5 + ((oh >> 8) % 11) as isize;
            let half_h = 5 + ((oh >> 16) % 11) as isize;
            let margin = 12isize;
            let span = IMAGE_SIZE as isize - 2 * margin;
            let cx = margin + ((oh >> 24) % span as u64) as isize;
            let cy = margin + ((oh >> 40) % span as u64) as isize;
            let kind = match oh % 3 {
                0 => InstanceKind::Rectangle,
                1 => InstanceKind::Circle,
                _ => InstanceKind::Triangle,
            };
            InstanceObject {
                kind,
                cx,
                cy,
                half_w,
                half_h,
                draw_order: idx,
                slot: 0,
            }
        })
        .collect();

    instances.sort_by_key(|instance| instance.cx);
    for (slot, instance) in instances.iter_mut().enumerate() {
        instance.slot = slot;
    }
    instances
}

fn visible_slot_map(instances: &[InstanceObject]) -> Vec<Option<usize>> {
    let mut map = vec![None; IMAGE_SIZE * IMAGE_SIZE];
    let mut draw_order: Vec<&InstanceObject> = instances.iter().collect();
    draw_order.sort_by_key(|instance| instance.draw_order);

    for instance in draw_order {
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                if instance.contains(x, y) {
                    map[y * IMAGE_SIZE + x] = Some(instance.slot);
                }
            }
        }
    }

    map
}

#[derive(Clone, Copy)]
enum InstanceKind {
    Rectangle,
    Circle,
    Triangle,
}

struct InstanceObject {
    kind: InstanceKind,
    cx: isize,
    cy: isize,
    half_w: isize,
    half_h: isize,
    draw_order: usize,
    slot: usize,
}

impl InstanceObject {
    fn contains(&self, x: usize, y: usize) -> bool {
        let dx = x as isize - self.cx;
        let dy = y as isize - self.cy;
        match self.kind {
            InstanceKind::Rectangle => dx.abs() <= self.half_w && dy.abs() <= self.half_h,
            InstanceKind::Circle => {
                let rx = self.half_w.max(1) as f32;
                let ry = self.half_h.max(1) as f32;
                (dx as f32 / rx).powi(2) + (dy as f32 / ry).powi(2) <= 1.0
            }
            InstanceKind::Triangle => {
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
    model: &OverlappingFixedSlotInstanceSegmentationNet,
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
            let base_rgb = [base, base, base];
            fill_scaled_pixel(&mut input_img, x, y, base_rgb);

            let mut out_rgb = base_rgb;
            for slot in 0..INSTANCE_SLOTS {
                let prob = probs[[sample_idx, slot, y, x]].clamp(0.0, 1.0);
                out_rgb = overlay(
                    out_rgb,
                    prob >= MASK_THRESHOLD,
                    slot_color(slot),
                    prob * 0.65,
                );
            }
            fill_scaled_pixel(&mut output_img, x, y, out_rgb);
        }
    }

    save_rgb_image(
        &input_img,
        "examples/traditional/overlapping_fixed_slot_instance_segmentation/test_in.png",
    )?;
    save_rgb_image(
        &output_img,
        "examples/traditional/overlapping_fixed_slot_instance_segmentation/test_out.png",
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
        2 => [64, 220, 96],
        _ => [245, 245, 245],
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
