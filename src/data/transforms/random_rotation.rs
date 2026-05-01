//! 随机旋转
//!
//! 在 [-degrees, +degrees] 范围内随机旋转图像，使用双线性插值。
//! 同时为 `ClassificationSample` / `DetectionSample` / `SegmentationSample`
//! 实现 [`SampleTransform`]——detection 路径下自动旋转 bbox 4 个角后取 AABB 并
//! 裁剪到图像边界；segmentation 路径下 mask 使用**最近邻**插值（避免离散类别
//! 被双线性混成非法中间值）。

use super::Transform;
use super::sample_transform::SampleTransform;
use crate::data::DetectionSample;
use crate::data::sample::{ClassificationSample, SegmentationSample};
use crate::tensor::Tensor;
use crate::vision::detection::{BBox, DetectionLabelFilter, GroundTruthBox, clip_filter_labels};
use rand::Rng;

/// 随机旋转
///
/// 对输入张量 [C, H, W] 或 [H, W] 在 `[-degrees, +degrees]` 范围内随机旋转，
/// 使用双线性插值填充，超出边界的像素用 `fill_value` 填充。
///
/// # 示例
///
/// ```ignore
/// let rot = RandomRotation::new(15.0); // ±15°
/// let output = rot.apply(&image_tensor);
/// ```
pub struct RandomRotation {
    degrees: f64,
    fill_value: f32,
    label_filter: DetectionLabelFilter,
}

impl RandomRotation {
    /// 创建随机旋转变换
    ///
    /// # 参数
    /// - `degrees`: 最大旋转角度（绝对值），单位为度
    pub fn new(degrees: f64) -> Self {
        assert!(
            degrees >= 0.0,
            "RandomRotation: degrees 必须 >= 0，得到 {degrees}"
        );
        Self {
            degrees,
            fill_value: 0.0,
            label_filter: DetectionLabelFilter::default(),
        }
    }

    /// 设置超出边界的填充值
    pub fn fill_value(mut self, value: f32) -> Self {
        self.fill_value = value;
        self
    }

    /// 设置 detection label 过滤规则；仅在 `SampleTransform<DetectionSample>`
    /// 路径下生效。旋转后的 bbox 会先 clip 到图像边界，再按此 filter 丢弃面积
    /// 过小的框。
    pub fn with_label_filter(mut self, filter: DetectionLabelFilter) -> Self {
        self.label_filter = filter;
        self
    }

    /// 采样一次旋转角度；`degrees == 0.0` 时固定返回 0.0。
    ///
    /// paired 路径下 image / mask / bbox 必须共用**同一个**采样结果，所以此
    /// 方法独立抽出。
    fn sample_angle(&self) -> f64 {
        if self.degrees == 0.0 {
            return 0.0;
        }
        let mut rng = rand::thread_rng();
        rng.gen_range(-self.degrees..=self.degrees)
    }
}

impl Transform for RandomRotation {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let angle = self.sample_angle();
        if angle == 0.0 {
            return tensor.clone();
        }
        rotate(tensor, angle, self.fill_value)
    }
}

// ============================================================================
// SampleTransform 实现：保持 image / label 几何同步
// ============================================================================

impl SampleTransform<ClassificationSample> for RandomRotation {
    fn apply_to(&self, mut sample: ClassificationSample) -> ClassificationSample {
        let angle = self.sample_angle();
        if angle == 0.0 {
            return sample;
        }
        sample.image = rotate(&sample.image, angle, self.fill_value);
        sample
    }
}

impl SampleTransform<DetectionSample> for RandomRotation {
    fn apply_to(&self, sample: DetectionSample) -> DetectionSample {
        let angle = self.sample_angle();
        if angle == 0.0 {
            return sample;
        }
        let DetectionSample { image, labels } = sample;
        let (h, w) = image_h_w(&image);
        let new_image = rotate(&image, angle, self.fill_value);
        let rotated: Vec<GroundTruthBox> = labels
            .into_iter()
            .map(|gt| {
                GroundTruthBox::new(rotate_bbox(gt.bbox, angle, w as f32, h as f32), gt.class_id)
            })
            .collect();
        let filtered = clip_filter_labels(&rotated, w as f32, h as f32, self.label_filter);
        DetectionSample::new(new_image, filtered)
    }
}

impl SampleTransform<SegmentationSample> for RandomRotation {
    fn apply_to(&self, sample: SegmentationSample) -> SegmentationSample {
        let angle = self.sample_angle();
        if angle == 0.0 {
            return sample;
        }
        let SegmentationSample { image, mask } = sample;
        let new_image = rotate(&image, angle, self.fill_value);
        let new_mask = rotate_nearest(&mask, angle, self.fill_value);
        SegmentationSample::new(new_image, new_mask)
    }
}

// ============================================================================
// 底层旋转 kernel（供 Transform / SampleTransform / tests 共用）
// ============================================================================

/// 旋转图像（双线性插值，适用于连续灰度 / RGB 图像）。
///
/// # 参数
/// - `tensor`: [C, H, W] 或 [H, W]
/// - `angle_deg`: 旋转角度（度）。正方向与 `vision::geom` / PyTorch 一致——
///   逆向映射矩阵为 `[cos, sin; -sin, cos]`，视觉上表现为屏幕坐标系下的顺时针
///   旋转（与 `rotate_bbox` 对齐）。
/// - `fill_value`: 超出边界的填充值
pub(crate) fn rotate(tensor: &Tensor, angle_deg: f64, fill_value: f32) -> Tensor {
    rotate_kernel(tensor, angle_deg, fill_value, InterpKind::Bilinear)
}

/// 旋转图像（最近邻插值，适用于离散类别 mask）。
///
/// 与 [`rotate`] 共用坐标系和中心约定；唯一差别是插值方式——最近邻保证输出
/// 像素值**一定是**输入某个像素值，不会混出非法中间类别。
pub(crate) fn rotate_nearest(tensor: &Tensor, angle_deg: f64, fill_value: f32) -> Tensor {
    rotate_kernel(tensor, angle_deg, fill_value, InterpKind::Nearest)
}

/// 旋转 bbox。
///
/// 把 bbox 的 4 个角点按**正向**旋转矩阵变换，再取其 AABB（axis-aligned
/// bounding box）作为新 bbox。这是标准的"旋转后外接矩形"做法，不对下游做
/// clip / filter——调用方通常配合 `clip_filter_labels` 处理。
///
/// 旋转中心与 [`rotate`] 完全一致：`((image_w - 1) / 2, (image_h - 1) / 2)`，
/// 在像素索引层面对齐，避免 image / bbox 产生亚像素偏移。
pub(crate) fn rotate_bbox(bbox: BBox, angle_deg: f64, image_w: f32, image_h: f32) -> BBox {
    let angle_rad = angle_deg.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    let cx = (image_w as f64 - 1.0) / 2.0;
    let cy = (image_h as f64 - 1.0) / 2.0;

    // 正向旋转矩阵（和 rotate() 反向映射互逆）：
    //   out_x - cx =  cos(a) * (in_x - cx) - sin(a) * (in_y - cy)
    //   out_y - cy =  sin(a) * (in_x - cx) + cos(a) * (in_y - cy)
    let transform = |px: f32, py: f32| -> (f32, f32) {
        let dx = px as f64 - cx;
        let dy = py as f64 - cy;
        let ox = cos_a * dx - sin_a * dy + cx;
        let oy = sin_a * dx + cos_a * dy + cy;
        (ox as f32, oy as f32)
    };

    let [x1, y1, x2, y2] = bbox.to_xyxy();
    let corners = [
        transform(x1, y1),
        transform(x2, y1),
        transform(x1, y2),
        transform(x2, y2),
    ];
    let (mut min_x, mut min_y) = corners[0];
    let (mut max_x, mut max_y) = corners[0];
    for &(x, y) in &corners[1..] {
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }
    BBox::from_xyxy(min_x, min_y, max_x, max_y)
}

/// 推断图像 Tensor 的 `(height, width)`。支持 `[H, W]` 与 `[C, H, W]`。
fn image_h_w(tensor: &Tensor) -> (usize, usize) {
    let shape = tensor.shape();
    match shape.len() {
        2 => (shape[0], shape[1]),
        3 => (shape[1], shape[2]),
        _ => panic!(
            "RandomRotation: 期望图像形状 [H, W] 或 [C, H, W]，得到 {:?}",
            shape
        ),
    }
}

#[derive(Clone, Copy)]
enum InterpKind {
    Bilinear,
    Nearest,
}

/// 旋转的统一实现：根据 `kind` 选择插值策略。
fn rotate_kernel(tensor: &Tensor, angle_deg: f64, fill_value: f32, kind: InterpKind) -> Tensor {
    let shape = tensor.shape();
    let ndim = shape.len();
    assert!(
        ndim == 2 || ndim == 3,
        "RandomRotation: 输入应为 2D [H, W] 或 3D [C, H, W]，得到 {ndim}D"
    );

    let (c, h, w) = if ndim == 2 {
        (1, shape[0], shape[1])
    } else {
        (shape[0], shape[1], shape[2])
    };

    let angle_rad = angle_deg.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let cx = (w as f64 - 1.0) / 2.0;
    let cy = (h as f64 - 1.0) / 2.0;

    let flat: Vec<f32> = tensor.flatten_view().to_vec();
    let mut data = vec![fill_value; c * h * w];

    for ch in 0..c {
        let ch_offset = ch * h * w;
        for out_y in 0..h {
            for out_x in 0..w {
                let dx = out_x as f64 - cx;
                let dy = out_y as f64 - cy;
                let in_x = cos_a * dx + sin_a * dy + cx;
                let in_y = -sin_a * dx + cos_a * dy + cy;

                let sample = match kind {
                    InterpKind::Bilinear => {
                        bilinear_interpolate(&flat, ch_offset, h, w, in_y, in_x)
                    }
                    InterpKind::Nearest => nearest_interpolate(&flat, ch_offset, h, w, in_y, in_x),
                };
                if let Some(val) = sample {
                    data[ch_offset + out_y * w + out_x] = val;
                }
            }
        }
    }

    Tensor::new(&data, shape)
}

/// 双线性插值
///
/// 在 flat 数据中，从 `ch_offset` 开始的 h×w 区域中插值
fn bilinear_interpolate(
    flat: &[f32],
    ch_offset: usize,
    h: usize,
    w: usize,
    y: f64,
    x: f64,
) -> Option<f32> {
    let eps = 1e-6;
    if x < -eps || x > (w - 1) as f64 + eps || y < -eps || y > (h - 1) as f64 + eps {
        return None;
    }

    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let dx = x - x0 as f64;
    let dy = y - y0 as f64;

    let v00 = flat[ch_offset + y0 * w + x0] as f64;
    let v01 = flat[ch_offset + y0 * w + x1] as f64;
    let v10 = flat[ch_offset + y1 * w + x0] as f64;
    let v11 = flat[ch_offset + y1 * w + x1] as f64;

    let val = v00 * (1.0 - dx) * (1.0 - dy)
        + v01 * dx * (1.0 - dy)
        + v10 * (1.0 - dx) * dy
        + v11 * dx * dy;

    Some(val as f32)
}

/// 最近邻插值
///
/// 输入越界时返回 `None`（调用方用 `fill_value` 填充）；否则返回离 `(y, x)`
/// 最近的像素值。用于 mask 等离散数据，避免 bilinear 混出非法中间类别。
fn nearest_interpolate(
    flat: &[f32],
    ch_offset: usize,
    h: usize,
    w: usize,
    y: f64,
    x: f64,
) -> Option<f32> {
    let eps = 1e-6;
    if x < -eps || x > (w - 1) as f64 + eps || y < -eps || y > (h - 1) as f64 + eps {
        return None;
    }
    let xi = x.round().clamp(0.0, (w - 1) as f64) as usize;
    let yi = y.round().clamp(0.0, (h - 1) as f64) as usize;
    Some(flat[ch_offset + yi * w + xi])
}
