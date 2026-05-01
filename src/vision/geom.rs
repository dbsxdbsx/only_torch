//! 通用图像几何变换。
//!
//! 这里只放"图像本体"的几何操作（resize / crop / flip 等）。涉及 detection /
//! segmentation 标签同步的版本见 `vision::detection::transform`。

use image::{DynamicImage, GenericImageView, imageops::FilterType};

/// 不保留宽高比，直接拉伸到目标尺寸。
///
/// 对应 PyTorch `torchvision.transforms.functional.resize` 的非保留宽高比模式。
pub fn resize_exact(image: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    assert!(width > 0 && height > 0, "resize_exact: 目标尺寸必须大于 0");
    image.resize_exact(width, height, FilterType::Triangle)
}

/// 等比缩放图像至完全包含在 `(width, height)` 内（不会留白）。
///
/// 适合"按最长边等比缩到目标尺寸"的场景；如果需要居中 padding 到固定尺寸，
/// 用 `vision::preprocess::letterbox`。
pub fn resize_keep_ratio(image: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    assert!(
        width > 0 && height > 0,
        "resize_keep_ratio: 目标尺寸必须大于 0"
    );
    image.resize(width, height, FilterType::Triangle)
}

/// 中心裁剪到 `(width, height)`。
///
/// 目标尺寸必须不超过原图，否则会 panic（与 PyTorch
/// `torchvision.transforms.functional.center_crop` 在 PIL 模式下的语义一致）。
pub fn center_crop(image: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    let (orig_w, orig_h) = image.dimensions();
    assert!(
        width <= orig_w && height <= orig_h,
        "center_crop: 目标 ({width}, {height}) 不能大于原图 ({orig_w}, {orig_h})"
    );
    let x = (orig_w - width) / 2;
    let y = (orig_h - height) / 2;
    image.crop_imm(x, y, width, height)
}
