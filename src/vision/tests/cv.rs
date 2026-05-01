use crate::vision::cv::{HoughCircle, detect_circles_hough};
use image::{DynamicImage, GrayImage, Luma};

#[test]
fn test_hough_circle_struct_construction() {
    let c = HoughCircle::new(10, 20, 30);
    assert_eq!(c.center_x, 10);
    assert_eq!(c.center_y, 20);
    assert_eq!(c.radius, 30);
}

#[test]
fn test_hough_handles_blank_image_without_panic() {
    // 全黑图像没有梯度——应该返回空且不崩溃。
    let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(20, 20, Luma([0])));
    let circles = detect_circles_hough(&img, 5.0, 1.0, 1, 3, 5);
    assert!(circles.is_empty());
}

#[test]
fn test_hough_returns_empty_on_subthreshold_votes() {
    let mut img = GrayImage::from_pixel(20, 20, Luma([0]));
    img.put_pixel(10, 10, Luma([255]));
    let img = DynamicImage::ImageLuma8(img);

    // 极高的 accumulator threshold 应该过滤掉所有候选
    let circles = detect_circles_hough(&img, 5.0, 1.0, 1_000_000, 2, 4);
    assert!(circles.is_empty());
}
