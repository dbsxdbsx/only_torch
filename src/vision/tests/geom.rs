use crate::vision::geom::{center_crop, resize_exact, resize_keep_ratio};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};

fn rgb(w: u32, h: u32, color: [u8; 3]) -> DynamicImage {
    DynamicImage::ImageRgb8(ImageBuffer::from_pixel(w, h, Rgb(color)))
}

#[test]
fn test_resize_exact_dimensions() {
    let img = rgb(40, 20, [10, 20, 30]);
    let out = resize_exact(&img, 20, 10);
    assert_eq!(out.dimensions(), (20, 10));
}

#[test]
fn test_resize_keep_ratio_fits_within_target() {
    let img = rgb(40, 20, [0, 0, 0]);
    let out = resize_keep_ratio(&img, 30, 30);
    let (w, h) = out.dimensions();
    // 输入宽高比 2:1，等比缩到 30×30 应得 30×15
    assert_eq!(w, 30);
    assert_eq!(h, 15);
}

#[test]
fn test_center_crop_dimensions() {
    let img = rgb(10, 10, [0, 0, 0]);
    let out = center_crop(&img, 4, 4);
    assert_eq!(out.dimensions(), (4, 4));
}

#[test]
#[should_panic(expected = "不能大于原图")]
fn test_center_crop_larger_than_input_panics() {
    let img = rgb(4, 4, [0, 0, 0]);
    let _ = center_crop(&img, 10, 10);
}

#[test]
#[should_panic(expected = "目标尺寸必须大于 0")]
fn test_resize_exact_rejects_zero_dimensions() {
    let img = rgb(4, 4, [0, 0, 0]);
    let _ = resize_exact(&img, 0, 4);
}
