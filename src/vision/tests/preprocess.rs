use approx::assert_abs_diff_eq;
use image::{DynamicImage, ImageBuffer, Rgb};

use crate::vision::preprocess::{
    image_to_nchw_normalized, image_to_nchw_normalized_data, letterbox, letterbox_with_fill,
};

#[test]
fn test_letterbox_wide_image_adds_vertical_padding() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 2, Rgb([10, 20, 30])));

    let result = letterbox(&img, 8);

    assert_eq!(result.original_size, (4, 2));
    assert_eq!(result.output_size, (8, 8));
    assert_abs_diff_eq!(result.scale, 2.0, epsilon = 1e-6);
    assert_eq!(result.pad, (0, 2));
    assert_eq!(result.image.to_rgb8().get_pixel(0, 0).0, [114, 114, 114]);
    assert_eq!(result.image.to_rgb8().get_pixel(0, 2).0, [10, 20, 30]);

    let (x, y) = result.to_origin(4.0, 4.0);
    assert_abs_diff_eq!(x, 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y, 1.0, epsilon = 1e-6);
}

#[test]
fn test_letterbox_tall_image_adds_horizontal_padding() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2, 4, Rgb([1, 2, 3])));

    let result = letterbox_with_fill(&img, 8, [9, 9, 9]);

    assert_abs_diff_eq!(result.scale, 2.0, epsilon = 1e-6);
    assert_eq!(result.pad, (2, 0));
    assert_eq!(result.image.to_rgb8().get_pixel(0, 0).0, [9, 9, 9]);
    assert_eq!(result.image.to_rgb8().get_pixel(2, 0).0, [1, 2, 3]);
}

#[test]
fn test_image_to_nchw_normalized_data_layout() {
    let mut img = ImageBuffer::from_pixel(2, 2, Rgb([0, 0, 0]));
    img.put_pixel(0, 0, Rgb([255, 0, 0]));
    img.put_pixel(1, 0, Rgb([0, 128, 0]));
    img.put_pixel(0, 1, Rgb([0, 0, 64]));
    img.put_pixel(1, 1, Rgb([255, 128, 64]));
    let img = DynamicImage::ImageRgb8(img);

    let data = image_to_nchw_normalized_data(&img, 2);

    assert_abs_diff_eq!(data[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(data[1], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(data[2], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(data[3], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(data[4], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(data[5], 128.0 / 255.0, epsilon = 1e-6);
    assert_abs_diff_eq!(data[8], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(data[10], 64.0 / 255.0, epsilon = 1e-6);

    let tensor = image_to_nchw_normalized(&img, 2);
    assert_eq!(tensor.shape(), &[1, 3, 2, 2]);
}
