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
use only_torch::metrics::mean_box_iou_cxcywh;
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer, VarLossOps};
use only_torch::tensor::Tensor;
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

        graph.eval();
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
    let iou = mean_box_iou_cxcywh(&boxes, targets);
    Ok((mae, iou.value()))
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
    use image::{ImageBuffer, Rgb};

    let boxes = model.predict_boxes(inputs)?;
    let boxes = boxes.value()?.unwrap();
    let pred_box = bbox_at(&boxes, sample_idx);
    let actual_box = bbox_at(targets, sample_idx);
    let iou = single_box_iou(pred_box, actual_box);

    let panel_size = IMAGE_SIZE as u32 * OVERLAY_SCALE;
    let mut input_img = ImageBuffer::from_pixel(panel_size, panel_size, Rgb([245, 245, 245]));
    let mut output_img = ImageBuffer::from_pixel(panel_size, panel_size, Rgb([245, 245, 245]));

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let base = (inputs[[sample_idx, 0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            let base_rgb = [base, base, base];
            fill_scaled_pixel(&mut input_img, x, y, base_rgb);
            fill_scaled_pixel(&mut output_img, x, y, base_rgb);
        }
    }

    draw_bbox(&mut output_img, pred_box, [32, 220, 64]);
    draw_detection_label(&mut output_img, pred_box, iou);

    save_rgb_image(
        &input_img,
        "examples/traditional/single_object_detection/test_in.png",
    )?;
    save_rgb_image(
        &output_img,
        "examples/traditional/single_object_detection/test_out.png",
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

fn draw_bbox(canvas: &mut image::RgbImage, bbox: [f32; 4], color: [u8; 3]) {
    let (x1, y1, x2, y2) = bbox_to_canvas_rect(bbox, canvas.width(), canvas.height());
    for t in 0..3 {
        let t = t as i32;
        draw_hline(canvas, x1, x2, y1 + t, color);
        draw_hline(canvas, x1, x2, y2 - t, color);
        draw_vline(canvas, x1 + t, y1, y2, color);
        draw_vline(canvas, x2 - t, y1, y2, color);
    }
}

fn draw_detection_label(canvas: &mut image::RgbImage, bbox: [f32; 4], iou: f32) {
    let (x1, y1, _, _) = bbox_to_canvas_rect(bbox, canvas.width(), canvas.height());
    let label = format!("IoU {}%", (iou * 100.0).round() as usize);
    let text_w = tiny_text_width(&label);
    let x = x1.max(0) as u32;
    let y = if y1 >= 11 { (y1 - 11) as u32 } else { 0 };

    fill_rect(canvas, x, y, text_w + 4, 9, [20, 20, 20]);
    draw_tiny_text(canvas, x + 2, y + 2, &label, [230, 255, 230]);
}

fn bbox_to_canvas_rect(bbox: [f32; 4], width: u32, height: u32) -> (i32, i32, i32, i32) {
    let [cx, cy, bw, bh] = bbox;
    let x1 = ((cx - bw * 0.5).clamp(0.0, 1.0) * width as f32).round() as i32;
    let y1 = ((cy - bh * 0.5).clamp(0.0, 1.0) * height as f32).round() as i32;
    let x2 = ((cx + bw * 0.5).clamp(0.0, 1.0) * width as f32).round() as i32 - 1;
    let y2 = ((cy + bh * 0.5).clamp(0.0, 1.0) * height as f32).round() as i32 - 1;
    (
        x1.clamp(0, width as i32 - 1),
        y1.clamp(0, height as i32 - 1),
        x2.max(x1).clamp(0, width as i32 - 1),
        y2.max(y1).clamp(0, height as i32 - 1),
    )
}

fn draw_hline(canvas: &mut image::RgbImage, x1: i32, x2: i32, y: i32, color: [u8; 3]) {
    for x in x1..=x2 {
        put_pixel_checked(canvas, x, y, color);
    }
}

fn draw_vline(canvas: &mut image::RgbImage, x: i32, y1: i32, y2: i32, color: [u8; 3]) {
    for y in y1..=y2 {
        put_pixel_checked(canvas, x, y, color);
    }
}

fn fill_rect(
    canvas: &mut image::RgbImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    color: [u8; 3],
) {
    for dy in 0..height {
        for dx in 0..width {
            put_pixel_checked(canvas, (x + dx) as i32, (y + dy) as i32, color);
        }
    }
}

fn put_pixel_checked(canvas: &mut image::RgbImage, x: i32, y: i32, color: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < canvas.width() && (y as u32) < canvas.height() {
        canvas.put_pixel(x as u32, y as u32, image::Rgb(color));
    }
}

fn tiny_text_width(text: &str) -> u32 {
    let char_count = text.chars().count() as u32;
    if char_count == 0 {
        0
    } else {
        char_count * 4 - 1
    }
}

fn draw_tiny_text(canvas: &mut image::RgbImage, x: u32, y: u32, text: &str, color: [u8; 3]) {
    let mut cursor = x;
    for ch in text.chars() {
        draw_tiny_char(canvas, cursor, y, ch, color);
        cursor += 4;
    }
}

fn draw_tiny_char(canvas: &mut image::RgbImage, x: u32, y: u32, ch: char, color: [u8; 3]) {
    let pattern = match ch {
        '0' => ["111", "101", "101", "101", "111"],
        '1' => ["010", "110", "010", "010", "111"],
        '2' => ["111", "001", "111", "100", "111"],
        '3' => ["111", "001", "111", "001", "111"],
        '4' => ["101", "101", "111", "001", "001"],
        '5' => ["111", "100", "111", "001", "111"],
        '6' => ["111", "100", "111", "101", "111"],
        '7' => ["111", "001", "010", "010", "010"],
        '8' => ["111", "101", "111", "101", "111"],
        '9' => ["111", "101", "111", "001", "111"],
        'I' => ["111", "010", "010", "010", "111"],
        'O' | 'o' => ["111", "101", "101", "101", "111"],
        'U' => ["101", "101", "101", "101", "111"],
        '%' => ["101", "001", "010", "100", "101"],
        ' ' => ["000", "000", "000", "000", "000"],
        _ => ["000", "000", "000", "000", "000"],
    };

    for (dy, row) in pattern.iter().enumerate() {
        for (dx, bit) in row.as_bytes().iter().enumerate() {
            if *bit == b'1' {
                put_pixel_checked(
                    canvas,
                    (x + dx as u32) as i32,
                    (y + dy as u32) as i32,
                    color,
                );
            }
        }
    }
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
    let pred_tensor = Tensor::new(&pred, &[1, 4]);
    let actual_tensor = Tensor::new(&actual, &[1, 4]);
    mean_box_iou_cxcywh(&pred_tensor, &actual_tensor).value()
}

fn deterministic_noise(seed: u64, sample_idx: usize, x: usize, y: usize) -> f32 {
    let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64, x as u64, y as u64]);
    rng.next_f32()
}
