//! 通用 2D 仿射变换 kernel。
//!
//! 抽出的目的是让 `RandomAffine` 的 **image-only** 路径和 `SampleTransform`
//! **paired** 路径共用同一组"采样参数 → 执行变换"的组合；image / mask / bbox
//! 必须被同一个 `AffineParams` 同时作用，否则 paired 就不再同步。
//!
//! 坐标约定：
//!
//! - 中心 `(cx, cy) = ((w - 1) / 2, (h - 1) / 2)`，像素索引层面（与
//!   `random_rotation::rotate` 对齐）。
//! - 正向变换（input → output）：
//!
//!   ```text
//!   out_x - cx - tx = s * (cos_a * (in_x - cx) + (cos_a * tan_sh - sin_a) * (in_y - cy))
//!   out_y - cy - ty = s * (sin_a * (in_x - cx) + (sin_a * tan_sh + cos_a) * (in_y - cy))
//!   ```
//!
//!   即矩阵 `A = s * R * Sh_x`（R 为旋转，Sh_x 为水平剪切）；
//!   image 侧用其逆矩阵做反向映射 + 插值；bbox 侧用 `A` 做 4 角点正向变换再
//!   取 AABB。

use crate::tensor::Tensor;
use crate::vision::detection::BBox;

/// 仿射参数。弧度 / 像素单位；由调用方（如 `RandomAffine::sample_params`）
/// 负责从度数 / 比例转换。
#[derive(Clone, Copy, Debug)]
pub(crate) struct AffineParams {
    pub angle_rad: f64,
    pub tx: f64,
    pub ty: f64,
    pub scale: f64,
    pub shear_rad: f64,
}

impl AffineParams {
    /// 是否等价于恒等变换。浮点精度范围内判断；用作 degrees / scale /
    /// translate / shear 全部采样到 identity 值时的短路出口，跳过 kernel
    /// 循环。
    pub(crate) fn is_identity(self) -> bool {
        self.angle_rad == 0.0
            && self.tx == 0.0
            && self.ty == 0.0
            && self.scale == 1.0
            && self.shear_rad == 0.0
    }
}

#[derive(Clone, Copy)]
enum InterpKind {
    Bilinear,
    Nearest,
}

/// 对 image / 连续数据做仿射变换，使用双线性插值。
pub(crate) fn affine_bilinear(tensor: &Tensor, params: AffineParams, fill: f32) -> Tensor {
    affine_kernel(tensor, params, fill, InterpKind::Bilinear)
}

/// 对 mask / 离散类别数据做仿射变换，使用最近邻插值。
pub(crate) fn affine_nearest(tensor: &Tensor, params: AffineParams, fill: f32) -> Tensor {
    affine_kernel(tensor, params, fill, InterpKind::Nearest)
}

/// 把 bbox 的 4 个角点按**正向**仿射变换，再取 AABB。调用方通常再接
/// `clip_filter_labels` 做边界裁剪 + 面积过滤。
pub(crate) fn affine_bbox(bbox: BBox, params: AffineParams, image_w: f32, image_h: f32) -> BBox {
    let cx = (image_w as f64 - 1.0) / 2.0;
    let cy = (image_h as f64 - 1.0) / 2.0;
    let cos_a = params.angle_rad.cos();
    let sin_a = params.angle_rad.sin();
    let tan_sh = params.shear_rad.tan();
    let s = params.scale;

    let a00 = s * cos_a;
    let a01 = s * (cos_a * tan_sh - sin_a);
    let a10 = s * sin_a;
    let a11 = s * (sin_a * tan_sh + cos_a);

    let transform = |px: f32, py: f32| -> (f32, f32) {
        let dx = px as f64 - cx;
        let dy = py as f64 - cy;
        let ox = a00 * dx + a01 * dy + cx + params.tx;
        let oy = a10 * dx + a11 * dy + cy + params.ty;
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

fn affine_kernel(tensor: &Tensor, params: AffineParams, fill: f32, kind: InterpKind) -> Tensor {
    let shape = tensor.shape();
    let ndim = shape.len();
    assert!(
        ndim == 2 || ndim == 3,
        "affine: 输入应为 2D [H, W] 或 3D [C, H, W]，得到 {ndim}D"
    );

    let (c, h, w) = if ndim == 2 {
        (1, shape[0], shape[1])
    } else {
        (shape[0], shape[1], shape[2])
    };

    let cx = (w as f64 - 1.0) / 2.0;
    let cy = (h as f64 - 1.0) / 2.0;

    let cos_a = params.angle_rad.cos();
    let sin_a = params.angle_rad.sin();
    let tan_sh = params.shear_rad.tan();
    let s = params.scale;

    // 正向矩阵 A = s * R * Sh_x
    let a00 = s * cos_a;
    let a01 = s * (cos_a * tan_sh - sin_a);
    let a10 = s * sin_a;
    let a11 = s * (sin_a * tan_sh + cos_a);
    // det(A) = s^2（R、Sh_x 的 det 均为 1）。`RandomAffine::scale` 断言
    // `min > 0`，所以 det 不会为 0。
    let det = s * s;
    let inv00 = a11 / det;
    let inv01 = -a01 / det;
    let inv10 = -a10 / det;
    let inv11 = a00 / det;

    let flat: Vec<f32> = tensor.flatten_view().to_vec();
    let mut out = vec![fill; c * h * w];

    for ch in 0..c {
        let ch_offset = ch * h * w;
        for out_y in 0..h {
            for out_x in 0..w {
                let dx = out_x as f64 - cx - params.tx;
                let dy = out_y as f64 - cy - params.ty;

                let in_x = inv00 * dx + inv01 * dy + cx;
                let in_y = inv10 * dx + inv11 * dy + cy;

                let sample = match kind {
                    InterpKind::Bilinear => bilinear_sample(&flat, ch_offset, h, w, in_y, in_x),
                    InterpKind::Nearest => nearest_sample(&flat, ch_offset, h, w, in_y, in_x),
                };
                if let Some(val) = sample {
                    out[ch_offset + out_y * w + out_x] = val;
                }
            }
        }
    }

    Tensor::new(&out, shape)
}

fn bilinear_sample(
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

fn nearest_sample(
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
