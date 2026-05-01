//! 在图像上绘制可视化几何（bbox / 矩形 / 圆 / 线）。
//!
//! 参照 `torchvision.utils.draw_bounding_boxes` 的设计：接收 `&mut DynamicImage`
//! 在原图上 in-place 绘制；颜色用 `[u8; 3]` RGB；坐标使用强类型（`BBox` /
//! 像素 i32 等），不依赖"裸 Tensor 是图像"的运行时检查。

use crate::vision::detection::BBox;
use image::{DynamicImage, GenericImage, GenericImageView, Rgb};

/// 在图像上绘制一个 `BBox`。
///
/// `thickness == 0` 表示填充矩形；`>0` 时绘制指定粗细的矩形边框。
/// 落在图像外的部分会被自动裁剪。
pub fn draw_bbox(canvas: &mut DynamicImage, bbox: BBox, color: [u8; 3], thickness: u32) {
    let (img_w, img_h) = canvas.dimensions();
    let x1 = bbox.x1.round().clamp(0.0, img_w as f32 - 1.0) as i32;
    let y1 = bbox.y1.round().clamp(0.0, img_h as f32 - 1.0) as i32;
    let x2 = bbox.x2.round().clamp(0.0, img_w as f32 - 1.0) as i32;
    let y2 = bbox.y2.round().clamp(0.0, img_h as f32 - 1.0) as i32;
    draw_rectangle_xyxy(canvas, x1, y1, x2, y2, color, thickness);
}

/// 按 `(x1, y1, x2, y2)` 像素坐标绘制矩形。
///
/// `thickness == 0` 表示填充；落在图像外的部分会被自动裁剪。
pub fn draw_rectangle_xyxy(
    canvas: &mut DynamicImage,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    color: [u8; 3],
    thickness: u32,
) {
    let (left, right) = (x1.min(x2), x1.max(x2));
    let (top, bottom) = (y1.min(y2), y1.max(y2));

    if thickness == 0 {
        for y in top..=bottom {
            for x in left..=right {
                put_pixel(canvas, x, y, color);
            }
        }
        return;
    }

    let t = thickness as i32;
    for y in top..=bottom {
        for x in left..=right {
            let on_border = x < left + t || x > right - t || y < top + t || y > bottom - t;
            if on_border {
                put_pixel(canvas, x, y, color);
            }
        }
    }
}

/// 在图像上绘制一个圆。
///
/// `thickness == 0` 或 `thickness >= radius` 表示填充；`>0` 时绘制环形带。
/// 落在图像外的部分会被自动裁剪。
pub fn draw_circle(
    canvas: &mut DynamicImage,
    center: (i32, i32),
    radius: u32,
    color: [u8; 3],
    thickness: u32,
) {
    let (cx, cy) = center;
    let r = radius as i32;
    let r_sq = r * r;
    let inner_r = radius.saturating_sub(thickness) as i32;
    let inner_sq = inner_r * inner_r;
    let fill = thickness == 0 || thickness >= radius;

    for dy in -r..=r {
        for dx in -r..=r {
            let dist_sq = dx * dx + dy * dy;
            let in_circle = dist_sq <= r_sq;
            if !in_circle {
                continue;
            }
            if fill || dist_sq > inner_sq {
                put_pixel(canvas, cx + dx, cy + dy, color);
            }
        }
    }
}

fn put_pixel(canvas: &mut DynamicImage, x: i32, y: i32, color: [u8; 3]) {
    let (w, h) = canvas.dimensions();
    if x < 0 || y < 0 {
        return;
    }
    let (xu, yu) = (x as u32, y as u32);
    if xu >= w || yu >= h {
        return;
    }

    match canvas {
        DynamicImage::ImageLuma8(buf) => {
            buf.put_pixel(xu, yu, image::Luma([color[0]]));
        }
        DynamicImage::ImageRgb8(buf) => {
            buf.put_pixel(xu, yu, Rgb(color));
        }
        DynamicImage::ImageRgba8(buf) => {
            buf.put_pixel(xu, yu, image::Rgba([color[0], color[1], color[2], 255]));
        }
        _ => {
            // 其他色彩模式按 RGB 写入；image crate 会自行做转换映射。
            canvas.put_pixel(xu, yu, image::Rgba([color[0], color[1], color[2], 255]));
        }
    }
}
