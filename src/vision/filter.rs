//! 图像滤波。
//!
//! 基础非线性滤波（中值等）。线性滤波（高斯模糊等）目前由 `image` crate 直接
//! 覆盖，按需在用户侧调用即可。

use image::DynamicImage;

/// 对 RGB / 灰度图像做 ksize×ksize 中值滤波。
///
/// `ksize` 必须 ≥ 2；图像边缘 `ksize/2` 像素保持不变（行为与早期 OpenCV
/// `medianBlur` 在边界的处理一致——不外推）。返回 RGB 或 Luma 图像，与输入
/// 通道数一致；其他色彩模式会先转 RGB8。
pub fn median_blur(image: &DynamicImage, ksize: u32) -> DynamicImage {
    assert!(ksize >= 2, "median_blur: ksize 必须 >= 2，得到 {ksize}");

    match image {
        DynamicImage::ImageLuma8(_) => {
            let buf = image.to_luma8();
            DynamicImage::ImageLuma8(median_blur_luma8(&buf, ksize))
        }
        _ => {
            let buf = image.to_rgb8();
            DynamicImage::ImageRgb8(median_blur_rgb8(&buf, ksize))
        }
    }
}

fn median_blur_rgb8(
    src: &image::RgbImage,
    ksize: u32,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let (w, h) = src.dimensions();
    let mut dst = src.clone();
    let half = ksize / 2;
    if w <= 2 * half || h <= 2 * half {
        return dst;
    }

    let mut window = Vec::with_capacity((ksize * ksize) as usize);
    for y in half..(h - half) {
        for x in half..(w - half) {
            for ch in 0..3usize {
                window.clear();
                for ky in (y - half)..=(y + half) {
                    for kx in (x - half)..=(x + half) {
                        window.push(src.get_pixel(kx, ky).0[ch]);
                    }
                }
                window.sort_unstable();
                let median = window[window.len() / 2];
                dst.get_pixel_mut(x, y).0[ch] = median;
            }
        }
    }
    dst
}

fn median_blur_luma8(
    src: &image::GrayImage,
    ksize: u32,
) -> image::ImageBuffer<image::Luma<u8>, Vec<u8>> {
    let (w, h) = src.dimensions();
    let mut dst = src.clone();
    let half = ksize / 2;
    if w <= 2 * half || h <= 2 * half {
        return dst;
    }

    let mut window = Vec::with_capacity((ksize * ksize) as usize);
    for y in half..(h - half) {
        for x in half..(w - half) {
            window.clear();
            for ky in (y - half)..=(y + half) {
                for kx in (x - half)..=(x + half) {
                    window.push(src.get_pixel(kx, ky).0[0]);
                }
            }
            window.sort_unstable();
            dst.get_pixel_mut(x, y).0[0] = window[window.len() / 2];
        }
    }
    dst
}
