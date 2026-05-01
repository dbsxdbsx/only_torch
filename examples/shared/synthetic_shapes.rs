//! `examples/` 间共享的合成形状数据集 helper。
//!
//! 用于 `overlapping_shapes_*` 系列 example 共用的"R/C/T 三类形状 + class_id
//! 由 kind 派生 + margin=12 + half_w/h ∈ 5..16"标准配置。每个 example 顶部用
//! `#[path]` 跨目录引用即可：
//!
//! ```ignore
//! #[path = "../../shared/synthetic_shapes.rs"]
//! mod synthetic_shapes;
//!
//! use synthetic_shapes::{ShapeObject, generate_objects};
//! ```
//!
//! 其它 example（如 `deformable_conv2d_segmentation` 用 16x16 small range、
//! `evolution/overlapping_shapes_semantic_segmentation` 只 R/C 且 class_id 随机）
//! 因为参数差异大，**不复用本 module**——保留各自的本地 helper 反而更直观。

use only_torch::data::SyntheticRng;

#[derive(Clone, Copy)]
pub enum ShapeKind {
    Rectangle,
    Circle,
    Triangle,
}

impl ShapeKind {
    /// 标准 class_id 映射：Rectangle = 1、Circle = 2、Triangle = 3。0 留给背景。
    pub const fn class_id(self) -> usize {
        match self {
            Self::Rectangle => 1,
            Self::Circle => 2,
            Self::Triangle => 3,
        }
    }
}

pub struct ShapeObject {
    pub kind: ShapeKind,
    pub class_id: usize,
    pub cx: isize,
    pub cy: isize,
    pub half_w: isize,
    pub half_h: isize,
}

impl ShapeObject {
    /// 判断 `(x, y)` 像素是否落在形状内部。Triangle 是底边在下、顶点在上的等腰三角形。
    pub fn contains(&self, x: usize, y: usize) -> bool {
        let dx = x as isize - self.cx;
        let dy = y as isize - self.cy;
        match self.kind {
            ShapeKind::Rectangle => dx.abs() <= self.half_w && dy.abs() <= self.half_h,
            ShapeKind::Circle => {
                let rx = self.half_w.max(1) as f32;
                let ry = self.half_h.max(1) as f32;
                (dx as f32 / rx).powi(2) + (dy as f32 / ry).powi(2) <= 1.0
            }
            ShapeKind::Triangle => {
                if dy < -self.half_h || dy > self.half_h {
                    return false;
                }
                let t = (dy + self.half_h) as f32 / (2 * self.half_h.max(1)) as f32;
                let half_width_at_y = (self.half_w as f32 * t).max(1.0);
                (dx as f32).abs() <= half_width_at_y
            }
        }
    }
}

/// 按"overlapping_shapes 标准配置"生成形状：R/C/T 等概率、margin=12、
/// half_w/h ∈ `5..16`。形状数量在 `0..=max_objects` 之间随机。
///
/// `(seed, sample_idx)` 派生子 RNG，确保同样输入得到同样形状（可复现）。
pub fn generate_objects(
    sample_idx: usize,
    seed: u64,
    image_size: usize,
    max_objects: usize,
) -> Vec<ShapeObject> {
    let mut rng = SyntheticRng::from_seed_parts(seed, &[sample_idx as u64]);
    let count = rng.usize_range(0..max_objects + 1);
    (0..count)
        .map(|idx| {
            let mut obj_rng = rng.fork(idx as u64 + 1);
            let kind = match obj_rng.usize_range(0..3) {
                0 => ShapeKind::Rectangle,
                1 => ShapeKind::Circle,
                _ => ShapeKind::Triangle,
            };
            let margin = 12isize;
            ShapeObject {
                kind,
                class_id: kind.class_id(),
                cx: obj_rng.isize_range(margin..image_size as isize - margin),
                cy: obj_rng.isize_range(margin..image_size as isize - margin),
                half_w: obj_rng.isize_range(5..16),
                half_h: obj_rng.isize_range(5..16),
            }
        })
        .collect()
}
