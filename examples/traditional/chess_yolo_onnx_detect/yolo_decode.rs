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

/// 单个检测结果（letterbox 空间下的 bbox）
#[derive(Debug, Clone)]
pub struct Detection {
    /// (x_min, y_min, x_max, y_max) 在 letterbox 坐标系
    pub bbox: [f32; 4],
    /// 综合置信度 = obj_conf × cls_score
    pub conf: f32,
    /// 类别索引（0..nc）
    pub class_id: usize,
}

impl Detection {
    /// bbox 中心坐标 (cx, cy)
    pub fn center(&self) -> (f32, f32) {
        let cx = (self.bbox[0] + self.bbox[2]) * 0.5;
        let cy = (self.bbox[1] + self.bbox[3]) * 0.5;
        (cx, cy)
    }

    /// IoU 计算
    pub fn iou(&self, other: &Detection) -> f32 {
        let x1 = self.bbox[0].max(other.bbox[0]);
        let y1 = self.bbox[1].max(other.bbox[1]);
        let x2 = self.bbox[2].min(other.bbox[2]);
        let y2 = self.bbox[3].min(other.bbox[3]);
        let inter_w = (x2 - x1).max(0.0);
        let inter_h = (y2 - y1).max(0.0);
        let inter = inter_w * inter_h;

        let area_a =
            (self.bbox[2] - self.bbox[0]).max(0.0) * (self.bbox[3] - self.bbox[1]).max(0.0);
        let area_b =
            (other.bbox[2] - other.bbox[0]).max(0.0) * (other.bbox[3] - other.bbox[1]).max(0.0);
        let union = area_a + area_b - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }
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
        detections.push(Detection {
            bbox: [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
            conf,
            class_id: best_cls,
        });
    }
    detections
}

/// Per-class NMS（按类别分组分别 NMS）
///
/// O(N²) 朴素实现，对单张棋盘 N < 100 的场景完全够用。
pub fn nms(mut dets: Vec<Detection>, iou_thresh: f32) -> Vec<Detection> {
    if dets.is_empty() {
        return dets;
    }
    // 按 conf 降序排序
    dets.sort_by(|a, b| {
        b.conf
            .partial_cmp(&a.conf)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep: Vec<Detection> = Vec::with_capacity(dets.len());
    let mut suppressed = vec![false; dets.len()];

    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if suppressed[j] {
                continue;
            }
            // 仅同类之间互相抑制
            if dets[i].class_id == dets[j].class_id && dets[i].iou(&dets[j]) > iou_thresh {
                suppressed[j] = true;
            }
        }
    }
    keep
}
