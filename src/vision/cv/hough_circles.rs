//! Hough 变换圆检测（教学级）。
//!
//! 与 OpenCV `cv2.HoughCircles` 接近的简化版本：基于梯度方向投票、按 min_dist
//! 抑制邻近候选。只用于 toy 示例；性能与精度都不与 OpenCV 对齐。

use image::DynamicImage;
use num_integer::Roots;
use std::collections::HashMap;

/// Hough 圆检测候选。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoughCircle {
    pub center_x: u32,
    pub center_y: u32,
    pub radius: u32,
}

impl HoughCircle {
    pub const fn new(center_x: u32, center_y: u32, radius: u32) -> Self {
        Self {
            center_x,
            center_y,
            radius,
        }
    }
}

/// 在灰度（必要时自动转灰度）图像上检测圆。
///
/// 参数沿用 OpenCV 习惯：
/// - `min_dist`：候选圆心之间的最小欧氏距离，单位像素。
/// - `gradient_threshold`：梯度幅值阈值，越大滤掉越多噪声边缘。
/// - `accumulator_threshold`：投票数阈值，越大保留越自信的圆。
/// - `min_radius` / `max_radius`：候选半径范围（含端点），单位像素。
pub fn detect_circles_hough(
    image: &DynamicImage,
    min_dist: f32,
    gradient_threshold: f32,
    accumulator_threshold: u32,
    min_radius: u32,
    max_radius: u32,
) -> Vec<HoughCircle> {
    assert!(
        min_radius <= max_radius,
        "Hough: min_radius 必须 <= max_radius"
    );
    assert!(min_dist >= 0.0, "Hough: min_dist 必须非负");

    let gray = image.to_luma8();
    let (width, height) = gray.dimensions();
    if width < 3 || height < 3 {
        return Vec::new();
    }

    let mut accumulator: HashMap<(u32, u32, u32), u32> = HashMap::new();
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let gx =
                i32::from(gray.get_pixel(x + 1, y).0[0]) - i32::from(gray.get_pixel(x - 1, y).0[0]);
            let gy =
                i32::from(gray.get_pixel(x, y + 1).0[0]) - i32::from(gray.get_pixel(x, y - 1).0[0]);
            let magnitude = ((gx * gx + gy * gy).max(0) as u32).sqrt() as f32;
            if magnitude <= gradient_threshold {
                continue;
            }
            let angle = (gy as f32).atan2(gx as f32);
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            for radius in min_radius..=max_radius {
                let r = radius as f32;
                let a = (x as f32 + r * cos_a).round();
                let b = (y as f32 + r * sin_a).round();
                if a < 0.0 || b < 0.0 || a >= width as f32 || b >= height as f32 {
                    continue;
                }
                *accumulator.entry((a as u32, b as u32, radius)).or_insert(0) += 1;
            }
        }
    }

    let min_dist_sq = min_dist * min_dist;
    let mut kept: Vec<HoughCircle> = Vec::new();
    for ((cx, cy, radius), votes) in accumulator {
        if votes < accumulator_threshold {
            continue;
        }
        let too_close = kept.iter().any(|c| {
            let dx = (c.center_x as f32) - (cx as f32);
            let dy = (c.center_y as f32) - (cy as f32);
            dx * dx + dy * dy < min_dist_sq
        });
        if !too_close {
            kept.push(HoughCircle::new(cx, cy, radius));
        }
    }
    kept
}
