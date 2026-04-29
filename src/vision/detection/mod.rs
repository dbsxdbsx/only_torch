//! 通用 2D detection 原语。
//!
//! 这里提供 bbox 几何、IoU family 与 NMS，不绑定具体检测模型族。

use std::cmp::Ordering;

/// 外部 bbox 坐标格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxFormat {
    /// `[x_min, y_min, x_max, y_max]`。
    XyXy,
    /// `[center_x, center_y, width, height]`。
    CxCyWh,
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
}

impl NmsOptions {
    pub const fn class_aware(iou_threshold: f32) -> Self {
        Self {
            iou_threshold,
            class_aware: true,
        }
    }

    pub const fn class_agnostic(iou_threshold: f32) -> Self {
        Self {
            iou_threshold,
            class_aware: false,
        }
    }
}

/// 对单张图的检测结果执行 NMS。
pub fn nms(detections: &[Detection], options: NmsOptions) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..detections.len()).collect();
    order.sort_by(|&a, &b| {
        detections[b]
            .score
            .partial_cmp(&detections[a].score)
            .unwrap_or(Ordering::Equal)
    });

    let mut suppressed = vec![false; detections.len()];
    let mut keep = Vec::with_capacity(detections.len());

    for (rank, &idx) in order.iter().enumerate() {
        if suppressed[idx] {
            continue;
        }
        let current = &detections[idx];
        keep.push(current.clone());

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

fn squared(x: f32) -> f32 {
    x * x
}
