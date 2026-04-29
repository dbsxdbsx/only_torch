//! YOLOv5 输出解码 + 纯 Rust NMS
//!
//! VinXiangQi 模型最终输出形状 `[1, 25200, 5+nc]`，每行布局：
//!     [cx, cy, w, h, obj_conf, cls_score_0, cls_score_1, ..., cls_score_{nc-1}]
//!
//! - `cx, cy, w, h`：letterbox 空间下的 bbox 中心坐标 + 宽高
//! - `obj_conf`：objectness（已过 sigmoid）
//! - `cls_score_*`：每类置信度（已过 sigmoid）
//!
//! 解码流程：obj_conf × max(cls_score) ≥ conf_thresh 的行保留，按类别分组做 NMS。

use only_torch::nn::GraphError;
use only_torch::tensor::Tensor;
use only_torch::vision::detection::{BBox, Detection, NmsOptions};

/// 端到端：YOLOv5 raw output → NMS 后的 Detection 列表。
///
/// 内部完成 shape 校验、`num_classes` 反推、decode、per-class NMS。
/// 兼容形状为 `[1, num_anchors, 5+nc]` 或扁平等价排布的输出。
pub fn detect(
    output: &Tensor,
    conf_thresh: f32,
    iou_thresh: f32,
) -> Result<Vec<Detection>, GraphError> {
    let last_dim = *output
        .shape()
        .last()
        .ok_or_else(|| GraphError::ComputationError("YOLOv5 输出张量无维度".to_string()))?;
    if last_dim < 5 {
        return Err(GraphError::ComputationError(format!(
            "YOLOv5 输出最后一维 {last_dim} < 5，无法解析"
        )));
    }
    let num_classes = last_dim - 5;
    let raw = decode(&output.flatten_view().to_vec(), num_classes, conf_thresh);
    Ok(nms(raw, iou_thresh))
}

/// 解码 YOLOv5 输出张量为检测结果列表（未 NMS）
///
/// # 参数
/// - `output`：扁平化的输出数据（按 `[1, 25200, 5+nc]` 行优先）
/// - `num_classes`：类别数（VinXiangQi 模型为 14）
/// - `conf_thresh`：综合置信度阈值（典型 0.25）
///
/// # 返回
/// 通过阈值的所有检测，每行一个 `Detection`
pub fn decode(output: &[f32], num_classes: usize, conf_thresh: f32) -> Vec<Detection> {
    let stride = 5 + num_classes;
    if output.len() % stride != 0 {
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
        // 找到最大类别分数
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

/// Per-class NMS（按类别分组分别 NMS）
///
/// O(N²) 朴素实现，对单张棋盘 N < 100 的场景完全够用。
pub fn nms(dets: Vec<Detection>, iou_thresh: f32) -> Vec<Detection> {
    only_torch::vision::detection::nms(&dets, NmsOptions::class_aware(iou_thresh))
}
