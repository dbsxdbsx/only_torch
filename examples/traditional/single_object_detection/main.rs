//! # Single Object Detection 示例
//!
//! 用极小的合成图像演示单目标 bbox 回归闭环：
//! - 内置生成 16x16 单目标图像，不依赖下载数据
//! - 小型 CNN 输出归一化 `[cx, cy, w, h]` bbox
//! - 使用 Huber loss 训练，使用 Mean Box IoU / MAE 评估
//!
//! ## 运行
//! ```bash
//! cargo run --example single_object_detection
//! ```

mod model;

use model::SingleObjectDetectionNet;
use only_torch::data::{DataLoader, SyntheticRng, TensorDataset};
use only_torch::metrics::mean_box_iou;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;
use only_torch::vision::detection::{BBox, BoxFormat};
use only_torch::vision::draw::draw_bbox;
use only_torch::vision::io::save_rgb_image;
use only_torch::vision::viz::{TinyFont, pixel_block_scale};
use std::time::Instant;

const IMAGE_SIZE: usize = 16;
const OVERLAY_SCALE: u32 = 12;
const TRAIN_SAMPLES: usize = 192;
const TEST_SAMPLES: usize = 32;
const BATCH_SIZE: usize = 32;
const MAX_EPOCHS: usize = 80;
const LEARNING_RATE: f32 = 0.02;
const TARGET_IOU: f32 = 0.83;

fn main() -> Result<(), GraphError> {
    let total_start = Instant::now();
    println!("=== Single Object Detection 示例 ===\n");

    let (train_x, train_y) = generate_dataset(TRAIN_SAMPLES, 42);
    let (test_x, test_y) = generate_dataset(TEST_SAMPLES, 2026);
    let train_loader = DataLoader::new(TensorDataset::new(train_x, train_y), BATCH_SIZE)
        .shuffle(true)
        .seed(11);

    let graph = Graph::new_with_seed(42);
    let model = SingleObjectDetectionNet::new(&graph)?;
    let mut optimizer = Adam::new(&graph, &model.parameters(), LEARNING_RATE);

    let param_count: usize = model
        .parameters()
        .iter()
        .filter_map(|p| p.value().ok().flatten())
        .map(|t| t.shape().iter().product::<usize>())
        .sum();

    println!("任务: 16x16 合成图像单目标检测（固定 seed 的随机位置 / 随机尺寸矩形）");
    println!("网络: Conv(1→8) → Pool → Conv(8→16) → Pool → Flatten → Linear → bbox(cx, cy, w, h)");
    println!("损失: HuberLoss，指标: Mean Box IoU + bbox MAE");
    println!("训练样本: {TRAIN_SAMPLES}, 测试样本: {TEST_SAMPLES}, 参数量: {param_count}\n");

    let mut best_iou = 0.0f32;
    let mut best_mae = f32::MAX;

    for epoch in 0..MAX_EPOCHS {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0usize;

        graph.train();
        for (batch_x, batch_y) in train_loader.iter() {
            let pred_boxes = model.forward(&batch_x)?;
            let loss = pred_boxes.huber_loss(&batch_y)?;

            graph.snapshot_once_from(&[&loss]);

            optimizer.zero_grad()?;
            let loss_val = loss.backward()?;
            optimizer.step()?;

            epoch_loss += loss_val;
            num_batches += 1;
        }

        graph.inference();
        let (mae, iou) = evaluate(&model, &test_x, &test_y)?;
        best_iou = best_iou.max(iou);
        best_mae = best_mae.min(mae);

        println!(
            "Epoch {:2}: loss={:.4}, bbox_mae={:.4}, mean_iou={:.1}%, {:.2}s",
            epoch + 1,
            epoch_loss / num_batches as f32,
            mae,
            iou * 100.0,
            epoch_start.elapsed().as_secs_f32()
        );

        if iou >= TARGET_IOU {
            println!("\n达到目标 mean IoU {:.1}%，提前停止。", iou * 100.0);
            break;
        }
    }

    println!("\n=== 预测示例 ===");
    print_sample_predictions(&model, &test_x, &test_y, 5)?;
    save_sample_visualizations(&model, &test_x, &test_y, 0)?;
    println!("测试输入图: examples/traditional/single_object_detection/test_in.png");
    println!("测试输出图: examples/traditional/single_object_detection/test_out.png");

    let vis_result = graph.visualize_snapshot(
        "examples/traditional/single_object_detection/single_object_detection",
    )?;
    println!("\n计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("可视化图像: {}", img_path.display());
    }

    println!(
        "\n最佳 bbox MAE: {:.4}，最佳 mean IoU: {:.1}%，总耗时: {:.1}s",
        best_mae,
        best_iou * 100.0,
        total_start.elapsed().as_secs_f32()
    );

    if best_iou >= TARGET_IOU {
        println!("Single object detection 训练成功。");
        Ok(())
    } else {
        Err(GraphError::ComputationError(format!(
            "Single object detection 未达到目标 mean IoU {:.1}%，最佳 {:.1}%",
            TARGET_IOU * 100.0,
            best_iou * 100.0
        )))
    }
}

fn evaluate(
    model: &SingleObjectDetectionNet,
    inputs: &Tensor,
    targets: &Tensor,
) -> Result<(f32, f32), GraphError> {
    let boxes = model.predict_boxes(inputs)?;
    let boxes = boxes.value()?.unwrap();
    let mae = bbox_mae(&boxes, targets);
    let pred_bboxes = clipped_bboxes_from_tensor(&boxes);
    let actual_bboxes = clipped_bboxes_from_tensor(targets);
    let iou = mean_box_iou(&pred_bboxes, &actual_bboxes);
    Ok((mae, iou.value()))
}

/// 把 `[N, 4]` cxcywh Tensor 转换为裁剪到 `[0, 1]` 后的 BBox 列表。
///
/// 这里只是一个 example 局部 adapter：复用 `BBox::vec_from_tensor` 拆分 Tensor，
/// 然后按本任务的归一化坐标约定 `clip(0.0, 1.0)`。换成像素坐标的检测器，把
/// 这一行换成 `clip_to_size(w, h)` 即可。
fn clipped_bboxes_from_tensor(tensor: &Tensor) -> Vec<BBox> {
    BBox::vec_from_tensor(tensor, BoxFormat::CxCyWh)
        .into_iter()
        .map(|bbox| bbox.clip(0.0, 1.0))
        .collect()
}

fn bbox_mae(predictions: &Tensor, targets: &Tensor) -> f32 {
    assert_eq!(predictions.shape(), targets.shape());
    if predictions.size() == 0 {
        return 0.0;
    }

    let total_error: f32 = predictions
        .to_vec()
        .into_iter()
        .zip(targets.to_vec())
        .map(|(pred, actual)| (pred - actual).abs())
        .sum();
    total_error / predictions.size() as f32
}

fn print_sample_predictions(
    model: &SingleObjectDetectionNet,
    inputs: &Tensor,
    targets: &Tensor,
    max_samples: usize,
) -> Result<(), GraphError> {
    let boxes = model.predict_boxes(inputs)?;
    let boxes = boxes.value()?.unwrap();

    for i in 0..max_samples.min(inputs.shape()[0]) {
        let pred = bbox_at(&boxes, i);
        let actual = bbox_at(targets, i);
        let iou = single_box_iou(pred, actual);
        println!(
            "样本 {i}: pred=[{:.2}, {:.2}, {:.2}, {:.2}], true=[{:.2}, {:.2}, {:.2}, {:.2}], IoU={:.1}%",
            pred[0],
            pred[1],
            pred[2],
            pred[3],
            actual[0],
            actual[1],
            actual[2],
            actual[3],
            iou * 100.0
        );
    }

    Ok(())
}

fn save_sample_visualizations(
    model: &SingleObjectDetectionNet,
    inputs: &Tensor,
    targets: &Tensor,
    sample_idx: usize,
) -> Result<(), GraphError> {
    use image::{DynamicImage, ImageBuffer, Rgb};

    let boxes = model.predict_boxes(inputs)?.value()?.unwrap();
    let pred_arr = bbox_at(&boxes, sample_idx);
    let actual_arr = bbox_at(targets, sample_idx);
    let iou = single_box_iou(pred_arr, actual_arr);

    let panel_size = IMAGE_SIZE as u32 * OVERLAY_SCALE;
    let mut input_img = ImageBuffer::from_pixel(panel_size, panel_size, Rgb([245, 245, 245]));
    let mut output_img = ImageBuffer::from_pixel(panel_size, panel_size, Rgb([245, 245, 245]));

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let base = (inputs[[sample_idx, 0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            let base_rgb = [base, base, base];
            pixel_block_scale(&mut input_img, x as u32, y as u32, base_rgb, OVERLAY_SCALE);
            pixel_block_scale(&mut output_img, x as u32, y as u32, base_rgb, OVERLAY_SCALE);
        }
    }

    // 把归一化 cxcywh 映射到像素空间，然后用库的 draw_bbox 画 3px 边框。
    // 中转 DynamicImage 是因为 vision::draw 统一接受 DynamicImage。
    let pred_bbox_pixel = BBox::from_array(pred_arr, BoxFormat::CxCyWh)
        .clip(0.0, 1.0)
        .scale_translate(panel_size as f32, panel_size as f32, 0.0, 0.0);
    let mut output_dyn = DynamicImage::ImageRgb8(output_img);
    draw_bbox(&mut output_dyn, pred_bbox_pixel, [32, 220, 64], 3);
    let mut output_img = output_dyn.into_rgb8();

    let label = format!("IoU {}%", (iou * 100.0).round() as usize);
    let label_x = pred_bbox_pixel.x1.max(0.0) as u32;
    let label_y = if pred_bbox_pixel.y1 >= 11.0 {
        (pred_bbox_pixel.y1 - 11.0) as u32
    } else {
        0
    };
    TinyFont::draw_with_box(
        &mut output_img,
        label_x,
        label_y,
        &label,
        [230, 255, 230],
        [20, 20, 20],
    );

    save_rgb_image(
        &input_img,
        "examples/traditional/single_object_detection/test_in.png",
    )
    .map_err(GraphError::ComputationError)?;
    save_rgb_image(
        &output_img,
        "examples/traditional/single_object_detection/test_out.png",
    )
    .map_err(GraphError::ComputationError)?;

    Ok(())
}

fn generate_dataset(n: usize, seed: u64) -> (Tensor, Tensor) {
    let mut images = Vec::with_capacity(n * IMAGE_SIZE * IMAGE_SIZE);
    let mut boxes = Vec::with_capacity(n * 4);

    for sample_idx in 0..n {
        let config = ShapeConfig::new(sample_idx, seed);
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let inside = config.contains(x, y);
                let noise = deterministic_noise(seed, sample_idx, x, y);
                let pixel = if inside {
                    0.76 + 0.18 * noise
                } else {
                    0.04 + 0.14 * noise
                };

                images.push(pixel);
            }
        }
        boxes.extend_from_slice(&config.bbox_cxcywh());
    }

    (
        Tensor::new(&images, &[n, 1, IMAGE_SIZE, IMAGE_SIZE]),
        Tensor::new(&boxes, &[n, 4]),
    )
}

#[derive(Clone, Copy)]
struct ShapeConfig {
    cx: isize,
    cy: isize,
    half_w: isize,
    half_h: isize,
}

impl ShapeConfig {
    fn new(sample_idx: usize, seed: u64) -> Self {
        let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64]);
        Self {
            cx: rng.isize_range(4..12),
            cy: rng.isize_range(4..12),
            half_w: rng.isize_range(2..5),
            half_h: rng.isize_range(2..5),
        }
    }

    fn contains(&self, x: usize, y: usize) -> bool {
        let x = x as isize;
        let y = y as isize;
        (x - self.cx).abs() <= self.half_w && (y - self.cy).abs() <= self.half_h
    }

    fn bbox_cxcywh(&self) -> [f32; 4] {
        let (x1, y1, x2, y2) = (
            self.cx - self.half_w,
            self.cy - self.half_h,
            self.cx + self.half_w,
            self.cy + self.half_h,
        );
        bbox_from_pixels(x1, y1, x2, y2)
    }
}

fn bbox_from_pixels(x1: isize, y1: isize, x2: isize, y2: isize) -> [f32; 4] {
    let size = IMAGE_SIZE as f32;
    let x1 = x1.clamp(0, IMAGE_SIZE as isize - 1) as f32;
    let y1 = y1.clamp(0, IMAGE_SIZE as isize - 1) as f32;
    let x2 = x2.clamp(0, IMAGE_SIZE as isize - 1) as f32;
    let y2 = y2.clamp(0, IMAGE_SIZE as isize - 1) as f32;

    [
        (x1 + x2 + 1.0) / (2.0 * size),
        (y1 + y2 + 1.0) / (2.0 * size),
        (x2 - x1 + 1.0) / size,
        (y2 - y1 + 1.0) / size,
    ]
}

fn bbox_at(tensor: &Tensor, index: usize) -> [f32; 4] {
    [
        tensor[[index, 0]],
        tensor[[index, 1]],
        tensor[[index, 2]],
        tensor[[index, 3]],
    ]
}

fn single_box_iou(pred: [f32; 4], actual: [f32; 4]) -> f32 {
    let pred_box = BBox::from_array(pred, BoxFormat::CxCyWh).clip(0.0, 1.0);
    let actual_box = BBox::from_array(actual, BoxFormat::CxCyWh).clip(0.0, 1.0);
    pred_box.iou(actual_box)
}

fn deterministic_noise(seed: u64, sample_idx: usize, x: usize, y: usize) -> f32 {
    let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64, x as u64, y as u64]);
    rng.next_f32()
}
