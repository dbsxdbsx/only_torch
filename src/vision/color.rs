//! 色彩空间转换。

use image::DynamicImage;

/// 把任意 `DynamicImage` 转换为单通道 Luma8 灰度图。
///
/// 行为与 `image::DynamicImage::to_luma8` 一致——彩色按 ITU-R BT.601 系数
/// 计算亮度，已是灰度的图像保持不变。
pub fn to_luma(image: &DynamicImage) -> DynamicImage {
    DynamicImage::ImageLuma8(image.to_luma8())
}
