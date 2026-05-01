//! 通用 2D detection 原语与任务契约。
//!
//! - 数据 / 几何积木：`BBox`、`Detection`、`GroundTruthBox`、IoU family、NMS、
//!   `clip_filter_*`，不绑定具体检测模型族。
//! - 任务契约（[`contract`] 模块）：`Backbone` / `BackboneOutput` /
//!   `DetectionHeadDecode` / `Assigner` / `AssignmentResult`，让 only_torch 内
//!   不同检测器实现能够互换。**契约比实现先行**，本目录暂不提供具体 backbone
//!   或 head 实现。
//! - 第三方 / 外部预训练检测器适配（[`adapter`] 模块）：把 YOLO 等具体模型族
//!   的特殊输出格式翻译成本框架的 [`Detection`] / [`BBox`] 类型。
//!
//! 配套的高层组合：
//! - `vision::preprocess::letterbox` / `image_to_nchw_normalized`：图像侧预处理
//! - `metrics::detection`：mAP / precision / recall

pub mod adapter;
mod contract;
mod io;
pub(crate) mod iou_loss;
mod loss;
mod transform;

pub use contract::{Assigner, AssignmentResult, Backbone, BackboneOutput, DetectionHeadDecode};
pub use io::{parse_yolo_txt_file, parse_yolo_txt_labels};
pub use loss::{DetectionLossComponents, DetectionLossWeights};
pub use transform::{
    DetectionLabelFilter, clip_filter_labels, horizontal_flip_labels, letterbox_labels,
    restore_letterbox_labels,
};

use std::cmp::Ordering;

use crate::tensor::Tensor;

/// bbox 坐标所属空间。
///
/// - `Pixel`：坐标单位是像素，通常属于原图或 letterbox 后图像。
/// - `Normalized`：坐标归一化到 `[0, 1]`，需要结合图像宽高还原为像素。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CoordinateSpace {
    Pixel,
    Normalized,
}

/// 外部 bbox 坐标格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BoxFormat {
    /// `[x_min, y_min, x_max, y_max]`。
    XyXy,
    /// `[center_x, center_y, width, height]`。
    CxCyWh,
}

/// bbox IoU-family 损失类型。
///
/// 该枚举只是 `Var::bbox_loss / giou_loss / diou_loss / ciou_loss` 的数据标签，
/// 实际可微图由 [`crate::vision::detection::iou_loss`] 模块用基础算子拼出。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BBoxLossKind {
    /// `1 - IoU`
    IoU,
    /// `1 - GIoU`
    GIoU,
    /// `1 - DIoU`
    DIoU,
    /// `1 - CIoU`
    CIoU,
}

/// 2D 边界框，内部统一使用 `xyxy` 表示。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub const fn from_xyxy(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    pub fn from_cxcywh(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        let half_w = w.max(0.0) * 0.5;
        let half_h = h.max(0.0) * 0.5;
        Self {
            x1: cx - half_w,
            y1: cy - half_h,
            x2: cx + half_w,
            y2: cy + half_h,
        }
    }

    pub fn from_array(values: [f32; 4], format: BoxFormat) -> Self {
        match format {
            BoxFormat::XyXy => Self::from_xyxy(values[0], values[1], values[2], values[3]),
            BoxFormat::CxCyWh => Self::from_cxcywh(values[0], values[1], values[2], values[3]),
        }
    }

    pub const fn to_xyxy(self) -> [f32; 4] {
        [self.x1, self.y1, self.x2, self.y2]
    }

    pub fn to_cxcywh(self) -> [f32; 4] {
        let w = self.width();
        let h = self.height();
        [self.x1 + w * 0.5, self.y1 + h * 0.5, w, h]
    }

    pub fn width(self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(self) -> f32 {
        self.width() * self.height()
    }

    /// bbox 是否是有限且面积为正的有效框。
    pub fn is_valid(self) -> bool {
        self.x1.is_finite()
            && self.y1.is_finite()
            && self.x2.is_finite()
            && self.y2.is_finite()
            && self.x2 > self.x1
            && self.y2 > self.y1
    }

    /// bbox 面积是否达到指定阈值。
    pub fn has_min_area(self, min_area: f32) -> bool {
        self.is_valid() && self.area() >= min_area.max(0.0)
    }

    pub fn center(self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    pub fn clip(self, min: f32, max: f32) -> Self {
        Self {
            x1: self.x1.clamp(min, max),
            y1: self.y1.clamp(min, max),
            x2: self.x2.clamp(min, max),
            y2: self.y2.clamp(min, max),
        }
    }

    /// 按图像尺寸裁剪 bbox。
    ///
    /// `image_width` / `image_height` 使用当前坐标空间的单位；像素坐标传图像尺寸，
    /// 归一化坐标可传 `1.0, 1.0`。
    pub fn clip_to_size(self, image_width: f32, image_height: f32) -> Self {
        let max_x = image_width.max(0.0);
        let max_y = image_height.max(0.0);
        Self {
            x1: self.x1.clamp(0.0, max_x),
            y1: self.y1.clamp(0.0, max_y),
            x2: self.x2.clamp(0.0, max_x),
            y2: self.y2.clamp(0.0, max_y),
        }
    }

    /// 从归一化坐标转换到像素坐标。
    pub fn to_pixel_space(self, image_width: f32, image_height: f32) -> Self {
        assert!(
            image_width > 0.0 && image_height > 0.0,
            "to_pixel_space: 图像宽高必须大于 0"
        );
        self.scale_translate(image_width, image_height, 0.0, 0.0)
    }

    /// 从像素坐标转换到归一化坐标。
    pub fn to_normalized(self, image_width: f32, image_height: f32) -> Self {
        assert!(
            image_width > 0.0 && image_height > 0.0,
            "to_normalized: 图像宽高必须大于 0"
        );
        self.scale_translate(1.0 / image_width, 1.0 / image_height, 0.0, 0.0)
    }

    /// 对 bbox 做缩放和平移。
    ///
    /// 适用于 letterbox / resize / crop 后同步更新标签坐标。
    pub fn scale_translate(self, scale_x: f32, scale_y: f32, dx: f32, dy: f32) -> Self {
        Self {
            x1: self.x1 * scale_x + dx,
            y1: self.y1 * scale_y + dy,
            x2: self.x2 * scale_x + dx,
            y2: self.y2 * scale_y + dy,
        }
    }

    /// 水平翻转 bbox。
    ///
    /// `image_width` 是当前坐标系的宽度；归一化坐标可传 `1.0`。
    pub fn horizontal_flip(self, image_width: f32) -> Self {
        Self {
            x1: image_width - self.x2,
            y1: self.y1,
            x2: image_width - self.x1,
            y2: self.y2,
        }
    }

    pub fn intersection_area(self, other: Self) -> f32 {
        let w = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let h = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        w * h
    }

    pub fn union_area(self, other: Self) -> f32 {
        self.area() + other.area() - self.intersection_area(other)
    }

    pub fn enclosing_box(self, other: Self) -> Self {
        Self {
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
            x2: self.x2.max(other.x2),
            y2: self.y2.max(other.y2),
        }
    }

    pub fn iou(self, other: Self) -> f32 {
        let union = self.union_area(other);
        if union <= 0.0 {
            0.0
        } else {
            self.intersection_area(other) / union
        }
    }

    pub fn giou(self, other: Self) -> f32 {
        let union = self.union_area(other);
        if union <= 0.0 {
            return 0.0;
        }
        let enclosing = self.enclosing_box(other);
        let enclosing_area = enclosing.area();
        if enclosing_area <= 0.0 {
            return self.iou(other);
        }
        self.iou(other) - (enclosing_area - union) / enclosing_area
    }

    pub fn diou(self, other: Self) -> f32 {
        let enclosing = self.enclosing_box(other);
        let diag_sq = squared(enclosing.width()) + squared(enclosing.height());
        if diag_sq <= 0.0 {
            return self.iou(other);
        }
        let (cx1, cy1) = self.center();
        let (cx2, cy2) = other.center();
        self.iou(other) - (squared(cx1 - cx2) + squared(cy1 - cy2)) / diag_sq
    }

    pub fn ciou(self, other: Self) -> f32 {
        let iou = self.iou(other);
        let diou = self.diou(other);
        let w1 = self.width();
        let h1 = self.height();
        let w2 = other.width();
        let h2 = other.height();
        if w1 <= 0.0 || h1 <= 0.0 || w2 <= 0.0 || h2 <= 0.0 {
            return diou;
        }

        let angle_diff = w2.atan2(h2) - w1.atan2(h1);
        let v = 4.0 / std::f32::consts::PI.powi(2) * angle_diff.powi(2);
        let alpha_den = 1.0 - iou + v;
        let alpha = if alpha_den <= 0.0 { 0.0 } else { v / alpha_den };
        diou - alpha * v
    }

    /// 从 `[N, 4]` Tensor 批量构造 `BBox`，按 `format` 解析每一行的四个数。
    ///
    /// 输入 shape 必须严格是 `[N, 4]`，否则 panic；`N == 0` 时返回空 `Vec`。
    ///
    /// 故意**不附带任何 clip 行为**：归一化坐标要 `clip(0.0, 1.0)`、像素坐标要
    /// `clip_to_size(w, h)` 时，由调用方在链式 `.map()` 里显式处理。这样同一个
    /// API 既能服务归一化输出，也能服务 letterbox / 原图像素坐标的检测器输出。
    pub fn vec_from_tensor(tensor: &Tensor, format: BoxFormat) -> Vec<Self> {
        let shape = tensor.shape();
        assert!(
            shape.len() == 2 && shape[1] == 4,
            "BBox::vec_from_tensor: 期望 shape=[N, 4]，实际 {shape:?}"
        );
        let n = shape[0];
        let data = tensor.to_vec();
        (0..n)
            .map(|i| {
                let offset = i * 4;
                Self::from_array(
                    [
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ],
                    format,
                )
            })
            .collect()
    }

    /// 把 `BBox` 列表序列化为 `[N, 4]` Tensor，按 `format` 写每一行的四个数。
    ///
    /// `boxes` 为空时返回 shape `[0, 4]` 的空 Tensor。
    pub fn vec_to_tensor(boxes: &[Self], format: BoxFormat) -> Tensor {
        let n = boxes.len();
        let mut data = Vec::with_capacity(n * 4);
        for bbox in boxes {
            let row = match format {
                BoxFormat::XyXy => bbox.to_xyxy(),
                BoxFormat::CxCyWh => bbox.to_cxcywh(),
            };
            data.extend_from_slice(&row);
        }
        Tensor::new(&data, &[n, 4])
    }
}

/// 检测预测框。
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub bbox: BBox,
    pub score: f32,
    pub class_id: usize,
}

impl Detection {
    pub const fn new(bbox: BBox, score: f32, class_id: usize) -> Self {
        Self {
            bbox,
            score,
            class_id,
        }
    }
}

/// 检测真值框。
#[derive(Debug, Clone, PartialEq)]
pub struct GroundTruthBox {
    pub bbox: BBox,
    pub class_id: usize,
}

impl GroundTruthBox {
    pub const fn new(bbox: BBox, class_id: usize) -> Self {
        Self { bbox, class_id }
    }
}

/// NMS 策略参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NmsOptions {
    pub iou_threshold: f32,
    /// `true` 表示只抑制同类框；`false` 表示 class-agnostic NMS。
    pub class_aware: bool,
    /// NMS 前按置信度过滤。默认不额外过滤，调用方可显式设置。
    pub score_threshold: f32,
    /// NMS 前保留的最高分候选数。
    pub pre_nms_top_k: Option<usize>,
    /// NMS 后最多返回的检测数。
    pub max_detections: Option<usize>,
}

impl NmsOptions {
    pub const fn class_aware(iou_threshold: f32) -> Self {
        Self {
            iou_threshold,
            class_aware: true,
            score_threshold: f32::NEG_INFINITY,
            pre_nms_top_k: None,
            max_detections: None,
        }
    }

    pub const fn class_agnostic(iou_threshold: f32) -> Self {
        Self {
            iou_threshold,
            class_aware: false,
            score_threshold: f32::NEG_INFINITY,
            pre_nms_top_k: None,
            max_detections: None,
        }
    }

    pub const fn with_score_threshold(mut self, score_threshold: f32) -> Self {
        self.score_threshold = score_threshold;
        self
    }

    pub const fn with_pre_nms_top_k(mut self, pre_nms_top_k: usize) -> Self {
        self.pre_nms_top_k = Some(pre_nms_top_k);
        self
    }

    pub const fn with_max_detections(mut self, max_detections: usize) -> Self {
        self.max_detections = Some(max_detections);
        self
    }
}

/// 对单张图的检测结果执行 NMS。
pub fn nms(detections: &[Detection], options: NmsOptions) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }
    if options.max_detections == Some(0) {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..detections.len())
        .filter(|&idx| {
            let score = detections[idx].score;
            score.is_finite() && score >= options.score_threshold
        })
        .collect();
    order.sort_by(|&a, &b| {
        detections[b]
            .score
            .partial_cmp(&detections[a].score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    if let Some(top_k) = options.pre_nms_top_k {
        order.truncate(top_k);
    }

    let mut suppressed = vec![false; detections.len()];
    let mut keep = Vec::with_capacity(detections.len());

    for (rank, &idx) in order.iter().enumerate() {
        if suppressed[idx] {
            continue;
        }
        let current = &detections[idx];
        keep.push(current.clone());
        if options
            .max_detections
            .is_some_and(|max_detections| keep.len() >= max_detections)
        {
            break;
        }

        for &other_idx in &order[(rank + 1)..] {
            if suppressed[other_idx] {
                continue;
            }
            let other = &detections[other_idx];
            if options.class_aware && current.class_id != other.class_id {
                continue;
            }
            if current.bbox.iou(other.bbox) > options.iou_threshold {
                suppressed[other_idx] = true;
            }
        }
    }

    keep
}

/// 对 batch 中每张图分别执行同一套 NMS。
pub fn batch_nms(batch_detections: &[Vec<Detection>], options: NmsOptions) -> Vec<Vec<Detection>> {
    batch_detections
        .iter()
        .map(|detections| nms(detections, options))
        .collect()
}

/// 按图像边界裁剪并过滤无效检测。
pub fn clip_filter_detections(
    detections: &[Detection],
    image_width: f32,
    image_height: f32,
    min_area: f32,
) -> Vec<Detection> {
    detections
        .iter()
        .filter_map(|detection| {
            let bbox = detection.bbox.clip_to_size(image_width, image_height);
            bbox.has_min_area(min_area)
                .then(|| Detection::new(bbox, detection.score, detection.class_id))
        })
        .collect()
}

/// 按图像边界裁剪并过滤无效真值框。
pub fn clip_filter_ground_truths(
    labels: &[GroundTruthBox],
    image_width: f32,
    image_height: f32,
    min_area: f32,
) -> Vec<GroundTruthBox> {
    labels
        .iter()
        .filter_map(|label| {
            let bbox = label.bbox.clip_to_size(image_width, image_height);
            bbox.has_min_area(min_area)
                .then(|| GroundTruthBox::new(bbox, label.class_id))
        })
        .collect()
}

fn squared(x: f32) -> f32 {
    x * x
}
