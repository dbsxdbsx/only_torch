use approx::assert_abs_diff_eq;

use crate::tensor::Tensor;
use crate::vision::detection::adapter::yolo::v5;

/// 构造 YOLOv5 风格的一行 raw 输出 `[cx, cy, w, h, obj, cls0, cls1, ...]`。
fn yolo_row(cx: f32, cy: f32, w: f32, h: f32, obj: f32, cls_scores: &[f32]) -> Vec<f32> {
    let mut row = vec![cx, cy, w, h, obj];
    row.extend_from_slice(cls_scores);
    row
}

#[test]
fn test_decode_filters_by_combined_confidence() {
    // 1 个 anchor，obj * max(cls) = 0.9 * 0.8 = 0.72 ≥ 0.5
    let row = yolo_row(50.0, 50.0, 20.0, 20.0, 0.9, &[0.8, 0.1]);
    let detections = v5::decode(&row, 2, 0.5);

    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].class_id, 0);
    assert_abs_diff_eq!(detections[0].score, 0.72, epsilon = 1e-6);
    // cxcywh (50, 50, 20, 20) → xyxy (40, 40, 60, 60)
    assert_abs_diff_eq!(detections[0].bbox.x1, 40.0, epsilon = 1e-6);
    assert_abs_diff_eq!(detections[0].bbox.y1, 40.0, epsilon = 1e-6);
    assert_abs_diff_eq!(detections[0].bbox.x2, 60.0, epsilon = 1e-6);
    assert_abs_diff_eq!(detections[0].bbox.y2, 60.0, epsilon = 1e-6);
}

#[test]
fn test_decode_drops_low_obj_conf_early() {
    // obj < conf_thresh，整行直接跳过（不会算 cls）
    let row = yolo_row(0.0, 0.0, 1.0, 1.0, 0.1, &[0.99, 0.99]);
    let detections = v5::decode(&row, 2, 0.5);
    assert!(detections.is_empty());
}

#[test]
fn test_decode_drops_low_combined_conf() {
    // obj 过阈值，但 obj * max(cls) = 0.6 * 0.4 = 0.24 < 0.5
    let row = yolo_row(0.0, 0.0, 1.0, 1.0, 0.6, &[0.4, 0.3]);
    let detections = v5::decode(&row, 2, 0.5);
    assert!(detections.is_empty());
}

#[test]
fn test_decode_picks_highest_class_score() {
    // 3 类，最高 score 在 class 2
    let row = yolo_row(10.0, 10.0, 4.0, 4.0, 1.0, &[0.2, 0.5, 0.9]);
    let detections = v5::decode(&row, 3, 0.5);

    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].class_id, 2);
    assert_abs_diff_eq!(detections[0].score, 0.9, epsilon = 1e-6);
}

#[test]
fn test_decode_handles_multiple_anchors() {
    // 3 个 anchor：第 1 个被 obj 阈值过滤、第 2 个保留、第 3 个被组合阈值过滤
    let mut buf = Vec::new();
    buf.extend(yolo_row(0.0, 0.0, 0.0, 0.0, 0.05, &[0.99])); // obj 太低
    buf.extend(yolo_row(10.0, 10.0, 4.0, 4.0, 0.9, &[0.8])); // 保留
    buf.extend(yolo_row(0.0, 0.0, 0.0, 0.0, 0.6, &[0.1])); // 组合 0.06 太低

    let detections = v5::decode(&buf, 1, 0.5);
    assert_eq!(detections.len(), 1);
    assert_abs_diff_eq!(detections[0].bbox.x1, 8.0, epsilon = 1e-6);
}

#[test]
fn test_decode_misaligned_buffer_returns_empty() {
    // stride = 5 + 2 = 7，但 buf.len() = 5，无法整除
    let buf = vec![0.0; 5];
    let detections = v5::decode(&buf, 2, 0.5);
    assert!(detections.is_empty());
}

#[test]
fn test_detect_end_to_end_runs_nms() {
    // 构造两个完全重叠的检测，class 相同 → NMS 应该只保留分数高的
    let mut buf = Vec::new();
    buf.extend(yolo_row(50.0, 50.0, 20.0, 20.0, 0.9, &[0.9])); // score = 0.81
    buf.extend(yolo_row(50.0, 50.0, 20.0, 20.0, 0.9, &[0.7])); // score = 0.63

    let tensor = Tensor::new(&buf, &[1, 2, 6]);
    let detections = v5::detect(&tensor, 0.5, 0.45).unwrap();

    assert_eq!(detections.len(), 1);
    assert_abs_diff_eq!(detections[0].score, 0.81, epsilon = 1e-6);
}

#[test]
fn test_detect_rejects_invalid_last_dim() {
    // 最后一维 = 4 < 5
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let err = v5::detect(&tensor, 0.5, 0.45).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("YOLOv5"), "错误信息应提及 YOLOv5，得到 {msg}");
}

#[test]
fn test_detect_infers_num_classes_from_shape() {
    // last_dim = 8 → num_classes = 3
    let row = yolo_row(0.0, 0.0, 0.0, 0.0, 0.9, &[0.7, 0.2, 0.1]);
    let tensor = Tensor::new(&row, &[1, 1, 8]);
    let detections = v5::detect(&tensor, 0.5, 0.45).unwrap();

    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].class_id, 0); // class 0 score 最高
}
