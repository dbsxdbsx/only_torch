use approx::assert_abs_diff_eq;

use crate::data::{DetectionBatch, DetectionSample};
use crate::tensor::Tensor;
use crate::vision::detection::{BBox, GroundTruthBox};

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
