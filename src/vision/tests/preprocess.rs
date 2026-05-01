use approx::assert_abs_diff_eq;
use image::{DynamicImage, ImageBuffer, Rgb};

use crate::vision::detection::BBox;
use crate::vision::preprocess::{
    image_to_nchw_normalized, image_to_nchw_normalized_data, image_to_nchw_normalized_size,
    letterbox, letterbox_to, letterbox_with_fill,
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
fn test_letterbox_to_rectangular_output_and_bbox_mapping() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 2, Rgb([10, 20, 30])));

    let result = letterbox_to(&img, (8, 4));

    assert_eq!(result.original_size, (4, 2));
    assert_eq!(result.output_size, (8, 4));
    assert_abs_diff_eq!(result.scale, 2.0, epsilon = 1e-6);
    assert_eq!(result.pad, (0, 0));

    let origin_bbox = BBox::from_xyxy(1.0, 0.5, 3.0, 1.5);
    let letterbox_bbox = result.bbox_to_letterbox(origin_bbox);
    assert_abs_diff_eq!(letterbox_bbox.x1, 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(letterbox_bbox.y1, 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(letterbox_bbox.x2, 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(letterbox_bbox.y2, 3.0, epsilon = 1e-6);

    let restored = result.bbox_to_origin(letterbox_bbox);
    assert_abs_diff_eq!(restored.x1, origin_bbox.x1, epsilon = 1e-6);
    assert_abs_diff_eq!(restored.y1, origin_bbox.y1, epsilon = 1e-6);
    assert_abs_diff_eq!(restored.x2, origin_bbox.x2, epsilon = 1e-6);
    assert_abs_diff_eq!(restored.y2, origin_bbox.y2, epsilon = 1e-6);
}

#[test]
fn test_letterbox_to_rectangular_output_with_horizontal_padding() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2, 4, Rgb([10, 20, 30])));

    let result = letterbox_to(&img, (8, 4));

    assert_eq!(result.original_size, (2, 4));
    assert_eq!(result.output_size, (8, 4));
    assert_abs_diff_eq!(result.scale, 1.0, epsilon = 1e-6);
    assert_eq!(result.pad, (3, 0));

    let origin_bbox = BBox::from_xyxy(0.0, 1.0, 2.0, 3.0);
    let letterbox_bbox = result.bbox_to_letterbox(origin_bbox);
    assert_abs_diff_eq!(letterbox_bbox.x1, 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(letterbox_bbox.y1, 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(letterbox_bbox.x2, 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(letterbox_bbox.y2, 3.0, epsilon = 1e-6);

    let restored = result.bbox_to_origin(BBox::from_xyxy(2.0, -1.0, 6.0, 5.0));
    assert_eq!(restored.to_xyxy(), [0.0, 0.0, 2.0, 4.0]);
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

#[test]
fn test_image_to_nchw_normalized_size_supports_rectangular_input() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(3, 2, Rgb([255, 0, 128])));

    let tensor = image_to_nchw_normalized_size(&img, 3, 2);

    assert_eq!(tensor.shape(), &[1, 3, 2, 3]);
    assert_abs_diff_eq!(tensor[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(tensor[[0, 1, 0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(tensor[[0, 2, 0, 0]], 128.0 / 255.0, epsilon = 1e-6);
}
