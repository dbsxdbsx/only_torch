//! 图像 IO：加载与保存。
//!
//! 接收 / 返回 `image::DynamicImage`，与 image crate 生态接轨。需要 Tensor 形态
//! 时，调用方可以接 `ForDynamicImage::to_tensor` 或
//! `vision::preprocess::image_to_nchw_normalized` 自行转换。

use image::DynamicImage;
use std::path::Path;

/// 加载本地图像文件为 `DynamicImage`。
///
/// 支持 PNG / JPG / BMP / TIFF 等 image crate 默认开启的格式。
pub fn load_image(path: impl AsRef<Path>) -> Result<DynamicImage, String> {
    image::open(path.as_ref()).map_err(|err| format!("加载图像失败: {err}"))
}

/// 把 `DynamicImage` 保存到本地文件，扩展名决定格式。
///
/// 若目标路径已存在，会先删除再写入。
pub fn save_image(image: &DynamicImage, path: impl AsRef<Path>) -> Result<(), String> {
    let path = path.as_ref();
    if path.exists() {
        std::fs::remove_file(path).map_err(|err| format!("删除已有图像文件失败: {err}"))?;
    }
    image
        .save(path)
        .map_err(|err| format!("保存图像失败: {err}"))
}
