use approx::assert_abs_diff_eq;
use image::{DynamicImage, ImageBuffer, Rgb};

use crate::data::{DetectionBatch, DetectionSample};
use crate::tensor::Tensor;
use crate::vision::detection::{
    BBox, DetectionLabelFilter, GroundTruthBox, clip_filter_labels, horizontal_flip_labels,
    letterbox_labels, restore_letterbox_labels,
};
use crate::vision::preprocess::letterbox;

#[test]
fn test_detection_batch_stacks_images_and_keeps_var_len_labels() {
    let sample1 = DetectionSample::new(
        Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]),
        vec![GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 0)],
    );
    let sample2 = DetectionSample::new(
        Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[1, 2, 2]),
        vec![
            GroundTruthBox::new(BBox::from_xyxy(0.1, 0.1, 0.2, 0.2), 1),
            GroundTruthBox::new(BBox::from_xyxy(0.3, 0.3, 0.4, 0.4), 2),
        ],
    );

    let batch = DetectionBatch::from_samples(&[sample1, sample2]).unwrap();

    assert_eq!(batch.images.shape(), &[2, 1, 2, 2]);
    assert_eq!(batch.labels.len(), 2);
    assert_eq!(batch.labels[0].len(), 1);
    assert_eq!(batch.labels[1].len(), 2);
    assert_abs_diff_eq!(batch.images[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(batch.images[[1, 0, 1, 1]], 8.0, epsilon = 1e-6);
}

#[test]
fn test_detection_batch_rejects_shape_mismatch() {
    let sample1 = DetectionSample::new(Tensor::zeros(&[1, 2, 2]), Vec::new());
    let sample2 = DetectionSample::new(Tensor::zeros(&[1, 3, 3]), Vec::new());

    let err = DetectionBatch::from_samples(&[sample1, sample2]).unwrap_err();

    assert!(
        format!("{err}").contains("形状不匹配"),
        "错误信息应说明形状不匹配，实际: {err}"
    );
}

#[test]
fn test_detection_batch_rejects_empty_samples() {
    let err = DetectionBatch::from_samples(&[]).unwrap_err();

    assert!(
        format!("{err}").contains("至少需要 1 个样本"),
        "错误信息应说明空 batch，实际: {err}"
    );
}

#[test]
fn test_letterbox_labels_roundtrip_and_filter_tiny_boxes() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 2, Rgb([0, 0, 0])));
    let lb = letterbox(&img, 8);
    let labels = vec![
        GroundTruthBox::new(BBox::from_xyxy(1.0, 0.5, 3.0, 1.5), 0),
        GroundTruthBox::new(BBox::from_xyxy(3.9, 1.9, 4.2, 2.2), 1),
    ];

    let mapped = letterbox_labels(&labels, &lb, DetectionLabelFilter::new(4.0));
    let restored = restore_letterbox_labels(&mapped, &lb, DetectionLabelFilter::new(0.0));

    assert_eq!(mapped.len(), 1);
    assert_abs_diff_eq!(mapped[0].bbox.x1, 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(mapped[0].bbox.y1, 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(mapped[0].bbox.x2, 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(mapped[0].bbox.y2, 5.0, epsilon = 1e-6);
    assert_eq!(restored.len(), 1);
    assert_abs_diff_eq!(restored[0].bbox.x1, 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(restored[0].bbox.y1, 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(restored[0].bbox.x2, 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(restored[0].bbox.y2, 1.5, epsilon = 1e-6);
}

#[test]
fn test_horizontal_flip_and_clip_filter_labels() {
    let labels = vec![
        GroundTruthBox::new(BBox::from_xyxy(10.0, 5.0, 30.0, 25.0), 0),
        GroundTruthBox::new(BBox::from_xyxy(99.0, 99.0, 120.0, 120.0), 1),
    ];

    let flipped = horizontal_flip_labels(&labels, 100.0);
    let filtered = clip_filter_labels(&flipped, 100.0, 100.0, DetectionLabelFilter::new(10.0));

    assert_eq!(flipped[0].bbox.to_xyxy(), [70.0, 5.0, 90.0, 25.0]);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].class_id, 0);
}
