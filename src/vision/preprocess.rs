//! 通用视觉预处理工具。
//!
//! 本模块只处理图像到张量的通用变换，不绑定具体模型族或业务任务。

use crate::tensor::Tensor;
use crate::vision::detection::BBox;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};

/// Letterbox 预处理结果。
#[derive(Debug, Clone)]
pub struct LetterboxResult {
    /// 处理后的 RGB 图像。
    pub image: DynamicImage,
    /// 等比缩放因子。
    pub scale: f32,
    /// 左上 padding 像素数 `(pad_x, pad_y)`。
    pub pad: (u32, u32),
    /// 原始图像尺寸 `(width, height)`。
    pub original_size: (u32, u32),
    /// 输出图像尺寸 `(width, height)`。
    pub output_size: (u32, u32),
}

impl LetterboxResult {
    /// 把 letterbox 坐标系下的点反映射到原图坐标。
    pub fn to_origin(&self, x_letterbox: f32, y_letterbox: f32) -> (f32, f32) {
        let x_origin = (x_letterbox - self.pad.0 as f32) / self.scale;
        let y_origin = (y_letterbox - self.pad.1 as f32) / self.scale;
        (x_origin, y_origin)
    }

    /// 把 letterbox 坐标系下的 bbox 反映射并裁剪到原图坐标。
    pub fn bbox_to_origin(&self, bbox: BBox) -> BBox {
        bbox.scale_translate(
            1.0 / self.scale,
            1.0 / self.scale,
            -(self.pad.0 as f32) / self.scale,
            -(self.pad.1 as f32) / self.scale,
        )
        .clip_to_size(self.original_size.0 as f32, self.original_size.1 as f32)
    }

    /// 把原图坐标系下的 bbox 映射并裁剪到 letterbox 输出坐标。
    pub fn bbox_to_letterbox(&self, bbox: BBox) -> BBox {
        bbox.scale_translate(self.scale, self.scale, self.pad.0 as f32, self.pad.1 as f32)
            .clip_to_size(self.output_size.0 as f32, self.output_size.1 as f32)
    }
}

/// 等比缩放图像并居中 padding 到 `target x target`。
///
/// 默认填充值 `114` 与 YOLO 系列常用预处理一致，但函数本身不依赖 YOLO 输出格式。
pub fn letterbox(img: &DynamicImage, target: u32) -> LetterboxResult {
    letterbox_with_fill(img, target, [114, 114, 114])
}

/// 等比缩放图像并居中 padding 到指定 `(width, height)`。
pub fn letterbox_to(img: &DynamicImage, output_size: (u32, u32)) -> LetterboxResult {
    letterbox_with_fill_to(img, output_size, [114, 114, 114])
}

/// 等比缩放图像并居中 padding 到 `target x target`，允许指定 RGB 填充值。
pub fn letterbox_with_fill(img: &DynamicImage, target: u32, fill: [u8; 3]) -> LetterboxResult {
    letterbox_with_fill_to(img, (target, target), fill)
}

/// 等比缩放图像并居中 padding 到指定 `(width, height)`，允许指定 RGB 填充值。
pub fn letterbox_with_fill_to(
    img: &DynamicImage,
    output_size: (u32, u32),
    fill: [u8; 3],
) -> LetterboxResult {
    let (orig_w, orig_h) = img.dimensions();
    let (out_w, out_h) = output_size;
    assert!(orig_w > 0 && orig_h > 0, "letterbox: 原图尺寸必须非零");
    assert!(out_w > 0 && out_h > 0, "letterbox: 输出尺寸必须大于 0");

    let scale = (out_w as f32 / orig_w as f32).min(out_h as f32 / orig_h as f32);
    let new_w = ((orig_w as f32 * scale).floor() as u32).max(1);
    let new_h = ((orig_h as f32 * scale).floor() as u32).max(1);

    let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
    let resized_rgb = resized.to_rgb8();

    let pad_x = (out_w.saturating_sub(new_w)) / 2;
    let pad_y = (out_h.saturating_sub(new_h)) / 2;
    let mut canvas: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_pixel(out_w, out_h, Rgb(fill));
    image::imageops::overlay(&mut canvas, &resized_rgb, pad_x as i64, pad_y as i64);

    LetterboxResult {
        image: DynamicImage::ImageRgb8(canvas),
        scale,
        pad: (pad_x, pad_y),
        original_size: (orig_w, orig_h),
        output_size: (out_w, out_h),
    }
}

/// 把 RGB 图像转换为 `[1, 3, target, target]` 的 NCHW 归一化张量。
pub fn image_to_nchw_normalized(img: &DynamicImage, target: u32) -> Tensor {
    let data = image_to_nchw_normalized_data(img, target);
    Tensor::new(&data, &[1, 3, target as usize, target as usize])
}

/// 把 RGB 图像转换为 `[1, 3, height, width]` 的 NCHW 归一化张量。
pub fn image_to_nchw_normalized_size(img: &DynamicImage, width: u32, height: u32) -> Tensor {
    let data = image_to_nchw_normalized_data_size(img, width, height);
    Tensor::new(&data, &[1, 3, height as usize, width as usize])
}

/// 把 RGB 图像转换为 NCHW 排布的 `[0, 1]` 归一化数据。
pub fn image_to_nchw_normalized_data(img: &DynamicImage, target: u32) -> Vec<f32> {
    image_to_nchw_normalized_data_size(img, target, target)
}

/// 把 RGB 图像转换为 NCHW 排布的 `[0, 1]` 归一化数据。
pub fn image_to_nchw_normalized_data_size(img: &DynamicImage, width: u32, height: u32) -> Vec<f32> {
    let rgb = img.to_rgb8();
    assert_eq!(
        rgb.dimensions(),
        (width, height),
        "image_to_nchw_normalized_data_size: 输入图像尺寸必须等于 width x height"
    );

    let mut out = vec![0f32; (3 * width * height) as usize];
    let plane = (width * height) as usize;
    for (x, y, p) in rgb.enumerate_pixels() {
        let idx = (y * width + x) as usize;
        out[idx] = p.0[0] as f32 / 255.0;
        out[plane + idx] = p.0[1] as f32 / 255.0;
        out[2 * plane + idx] = p.0[2] as f32 / 255.0;
    }
    out
}
