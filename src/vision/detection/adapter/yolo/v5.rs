//! YOLOv5 ONNX 输出解码器。
//!
//! 适配 YOLOv5 风格 head 的最终输出形状 `[1, num_anchors, 5+nc]`，每行布局：
//!
//! ```text
//! [cx, cy, w, h, obj_conf, cls_score_0, cls_score_1, ..., cls_score_{nc-1}]
//! ```
//!
//! - `cx, cy, w, h`：letterbox 空间下 bbox 中心坐标 + 宽高
//! - `obj_conf`：objectness（已过 sigmoid）
//! - `cls_score_*`：每类置信度（已过 sigmoid）
//!
//! 解码流程：`obj_conf × max(cls_score) ≥ conf_thresh` 的行保留，按类别分组做 NMS。
//! 输出坐标空间是 head 的输入空间（通常是 letterbox 640x640）。如需还原到原图
//! 坐标，调用方使用 [`crate::vision::preprocess::LetterboxResult::bbox_to_origin`]。
//!
//! # 兼容性
//!
//! 兼容形状为 `[1, num_anchors, 5+nc]` 或扁平等价排布的输出（按行优先解释）。

use crate::nn::GraphError;
use crate::tensor::Tensor;
use crate::vision::detection::{BBox, Detection, NmsOptions, nms};

/// 端到端：YOLOv5 raw output → 已 NMS 的 `Detection` 列表。
///
/// 内部完成：
/// 1. shape 校验（最后一维必须 `>= 5`，`num_classes` 由 `last_dim - 5` 反推）
/// 2. 阈值过滤 + cxcywh → xyxy 解码
/// 3. per-class NMS（[`NmsOptions::class_aware`]）
///
/// # 参数
///
/// - `raw`：head 输出张量，形状 `[1, num_anchors, 5+nc]`
/// - `conf_thresh`：综合置信度阈值（典型 0.25）
/// - `iou_thresh`：NMS IoU 阈值（典型 0.45）
///
/// # 错误
///
/// - 输入张量无维度
/// - 最后一维 `< 5`（无法解析 5+nc 布局）
pub fn detect(
    raw: &Tensor,
    conf_thresh: f32,
    iou_thresh: f32,
) -> Result<Vec<Detection>, GraphError> {
    let last_dim = *raw
        .shape()
        .last()
        .ok_or_else(|| GraphError::ComputationError("YOLOv5 输出张量无维度".to_string()))?;
    if last_dim < 5 {
        return Err(GraphError::ComputationError(format!(
            "YOLOv5 输出最后一维 {last_dim} < 5，无法解析"
        )));
    }
    let num_classes = last_dim - 5;
    let detections = decode(&raw.to_vec(), num_classes, conf_thresh);
    Ok(nms(&detections, NmsOptions::class_aware(iou_thresh)))
}

/// 解码 YOLOv5 输出张量为检测结果列表（**未 NMS**）。
///
/// 通常调用方应该用 [`detect`] 一站式拿到 NMS 后结果；这个低层 API 暴露给
/// 想自己控制 NMS 策略（如 class-agnostic NMS、自定义 score 阈值等）的场景。
///
/// # 参数
///
/// - `output`：扁平化的输出数据（按 `[1, num_anchors, 5+nc]` 行优先）
/// - `num_classes`：类别数
/// - `conf_thresh`：综合置信度阈值（典型 0.25）
///
/// # 返回
///
/// 通过阈值的所有检测，bbox 是 `xyxy` 像素坐标（letterbox 空间）。
///
/// 如果 `output.len() % (5 + num_classes) != 0`，返回空 `Vec`（容错处理）。
pub fn decode(output: &[f32], num_classes: usize, conf_thresh: f32) -> Vec<Detection> {
    let stride = 5 + num_classes;
    if !output.len().is_multiple_of(stride) {
        return Vec::new();
    }
    let num_anchors = output.len() / stride;

    let mut detections = Vec::with_capacity(num_anchors / 16);
    for i in 0..num_anchors {
        let row = &output[i * stride..(i + 1) * stride];
        let obj_conf = row[4];
        if obj_conf < conf_thresh {
            continue;
        }
        let mut best_cls = 0usize;
        let mut best_score = row[5];
        for c in 1..num_classes {
            let s = row[5 + c];
            if s > best_score {
                best_score = s;
                best_cls = c;
            }
        }
        let conf = obj_conf * best_score;
        if conf < conf_thresh {
            continue;
        }

        let cx = row[0];
        let cy = row[1];
        let w = row[2];
        let h = row[3];
        let half_w = w * 0.5;
        let half_h = h * 0.5;
        detections.push(Detection::new(
            BBox::from_xyxy(cx - half_w, cy - half_h, cx + half_w, cy + half_h),
            conf,
            best_cls,
        ));
    }
    detections
}
