use crate::vision::filter::median_blur;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb};

#[test]
fn test_median_blur_removes_isolated_noise_in_rgb() {
    let mut img = ImageBuffer::from_pixel(5, 5, Rgb([0u8, 0, 0]));
    img.put_pixel(2, 2, Rgb([255, 255, 255]));
    let img = DynamicImage::ImageRgb8(img);

    let blurred = median_blur(&img, 3);
    let buf = blurred.to_rgb8();
    // 中心孤立噪点应被周围 0 取代
    assert_eq!(buf.get_pixel(2, 2).0, [0, 0, 0]);
}

#[test]
fn test_median_blur_preserves_uniform_image() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(5, 5, Rgb([100u8, 200, 50])));
    let blurred = median_blur(&img, 3);
    let buf = blurred.to_rgb8();
    assert_eq!(buf.get_pixel(2, 2).0, [100, 200, 50]);
    assert_eq!(blurred.dimensions(), (5, 5));
}

#[test]
fn test_median_blur_luma_input_returns_luma() {
    let img = DynamicImage::ImageLuma8(ImageBuffer::from_pixel(5, 5, Luma([128u8])));
    let blurred = median_blur(&img, 3);
    assert!(matches!(blurred, DynamicImage::ImageLuma8(_)));
}

#[test]
#[should_panic(expected = "ksize 必须 >= 2")]
fn test_median_blur_rejects_small_ksize() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(3, 3, Rgb([0u8, 0, 0])));
    let _ = median_blur(&img, 1);
}
