use approx::assert_abs_diff_eq;

use crate::vision::detection::{
    BBox, BoxFormat, Detection, GroundTruthBox, NmsOptions, batch_nms, clip_filter_detections,
    clip_filter_ground_truths, nms,
};

#[test]
fn test_bbox_format_conversion_roundtrip() {
    let bbox = BBox::from_array([0.5, 0.4, 0.2, 0.6], BoxFormat::CxCyWh);

    assert_abs_diff_eq!(bbox.x1, 0.4, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox.y1, 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox.x2, 0.6, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox.y2, 0.7, epsilon = 1e-6);

    let [cx, cy, w, h] = bbox.to_cxcywh();
    assert_abs_diff_eq!(cx, 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(cy, 0.4, epsilon = 1e-6);
    assert_abs_diff_eq!(w, 0.2, epsilon = 1e-6);
    assert_abs_diff_eq!(h, 0.6, epsilon = 1e-6);
}

#[test]
fn test_iou_family_known_values() {
    let a = BBox::from_xyxy(0.0, 0.0, 2.0, 2.0);
    let b = BBox::from_xyxy(1.0, 0.0, 3.0, 2.0);

    assert_abs_diff_eq!(a.iou(b), 1.0 / 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(a.giou(b), 1.0 / 3.0, epsilon = 1e-6);

    let diou = a.diou(b);
    assert!(
        diou < a.iou(b),
        "中心点不同的 DIoU 应小于 IoU，diou={diou}, iou={}",
        a.iou(b)
    );
    assert_abs_diff_eq!(a.ciou(b), diou, epsilon = 1e-6);
}

#[test]
fn test_bbox_scale_translate_and_horizontal_flip() {
    let bbox = BBox::from_xyxy(10.0, 20.0, 30.0, 60.0);

    let transformed = bbox.scale_translate(0.5, 0.25, 2.0, 3.0);
    assert_abs_diff_eq!(transformed.x1, 7.0, epsilon = 1e-6);
    assert_abs_diff_eq!(transformed.y1, 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(transformed.x2, 17.0, epsilon = 1e-6);
    assert_abs_diff_eq!(transformed.y2, 18.0, epsilon = 1e-6);

    let flipped = bbox.horizontal_flip(100.0);
    assert_abs_diff_eq!(flipped.x1, 70.0, epsilon = 1e-6);
    assert_abs_diff_eq!(flipped.y1, 20.0, epsilon = 1e-6);
    assert_abs_diff_eq!(flipped.x2, 90.0, epsilon = 1e-6);
    assert_abs_diff_eq!(flipped.y2, 60.0, epsilon = 1e-6);
}

#[test]
fn test_bbox_clip_normalize_and_min_area_contract() {
    let bbox = BBox::from_xyxy(-10.0, 5.0, 110.0, 55.0);
    let clipped = bbox.clip_to_size(100.0, 50.0);

    assert_eq!(clipped.to_xyxy(), [0.0, 5.0, 100.0, 50.0]);
    assert!(clipped.is_valid());
    assert!(clipped.has_min_area(4_000.0));
    assert!(!clipped.has_min_area(5_000.0));

    let normalized = clipped.to_normalized(100.0, 50.0);
    assert_abs_diff_eq!(normalized.x1, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(normalized.y1, 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(normalized.x2, 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(normalized.y2, 1.0, epsilon = 1e-6);

    let pixels = normalized.to_pixel_space(100.0, 50.0);
    assert_abs_diff_eq!(pixels.x1, clipped.x1, epsilon = 1e-6);
    assert_abs_diff_eq!(pixels.y1, clipped.y1, epsilon = 1e-6);
    assert_abs_diff_eq!(pixels.x2, clipped.x2, epsilon = 1e-6);
    assert_abs_diff_eq!(pixels.y2, clipped.y2, epsilon = 1e-6);
}

#[test]
fn test_bbox_zero_area_is_invalid() {
    let bbox = BBox::from_xyxy(1.0, 1.0, 1.0, 2.0);

    assert_eq!(bbox.area(), 0.0);
    assert!(!bbox.is_valid());
    assert!(!bbox.has_min_area(0.0));
}

#[test]
fn test_giou_penalizes_non_overlapping_boxes() {
    let a = BBox::from_xyxy(0.0, 0.0, 1.0, 1.0);
    let b = BBox::from_xyxy(2.0, 0.0, 3.0, 1.0);

    assert_eq!(a.iou(b), 0.0);
    assert_abs_diff_eq!(a.giou(b), -1.0 / 3.0, epsilon = 1e-6);
}

#[test]
fn test_nms_class_aware_keeps_overlapping_different_classes() {
    let dets = vec![
        Detection::new(BBox::from_xyxy(0.0, 0.0, 2.0, 2.0), 0.9, 0),
        Detection::new(BBox::from_xyxy(0.2, 0.2, 2.2, 2.2), 0.8, 0),
        Detection::new(BBox::from_xyxy(0.1, 0.1, 2.1, 2.1), 0.7, 1),
    ];

    let kept = nms(&dets, NmsOptions::class_aware(0.5));

    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].class_id, 0);
    assert_eq!(kept[1].class_id, 1);
}

#[test]
fn test_nms_class_agnostic_suppresses_across_classes() {
    let dets = vec![
        Detection::new(BBox::from_xyxy(0.0, 0.0, 2.0, 2.0), 0.9, 0),
        Detection::new(BBox::from_xyxy(0.1, 0.1, 2.1, 2.1), 0.8, 1),
    ];

    let kept = nms(&dets, NmsOptions::class_agnostic(0.5));

    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].class_id, 0);
}

#[test]
fn test_nms_filters_score_and_limits_candidates() {
    let dets = vec![
        Detection::new(BBox::from_xyxy(0.0, 0.0, 2.0, 2.0), 0.95, 0),
        Detection::new(BBox::from_xyxy(4.0, 4.0, 5.0, 5.0), 0.90, 0),
        Detection::new(BBox::from_xyxy(6.0, 6.0, 7.0, 7.0), 0.10, 0),
    ];

    let kept = nms(
        &dets,
        NmsOptions::class_aware(0.5)
            .with_score_threshold(0.5)
            .with_pre_nms_top_k(2)
            .with_max_detections(1),
    );

    assert_eq!(kept.len(), 1);
    assert_abs_diff_eq!(kept[0].score, 0.95, epsilon = 1e-6);
}

#[test]
fn test_nms_zero_limits_and_non_finite_scores() {
    let dets = vec![
        Detection::new(BBox::from_xyxy(0.0, 0.0, 2.0, 2.0), f32::NAN, 0),
        Detection::new(BBox::from_xyxy(3.0, 3.0, 4.0, 4.0), f32::INFINITY, 0),
        Detection::new(BBox::from_xyxy(5.0, 5.0, 6.0, 6.0), 0.5, 0),
    ];

    let max_zero = nms(&dets, NmsOptions::class_aware(0.5).with_max_detections(0));
    let pre_zero = nms(&dets, NmsOptions::class_aware(0.5).with_pre_nms_top_k(0));
    let finite_only = nms(
        &dets,
        NmsOptions::class_aware(0.5).with_score_threshold(0.5),
    );

    assert!(max_zero.is_empty());
    assert!(pre_zero.is_empty());
    assert_eq!(finite_only.len(), 1);
    assert_abs_diff_eq!(finite_only[0].score, 0.5, epsilon = 1e-6);
}

#[test]
fn test_batch_nms_applies_per_image() {
    let batch = vec![
        vec![
            Detection::new(BBox::from_xyxy(0.0, 0.0, 2.0, 2.0), 0.9, 0),
            Detection::new(BBox::from_xyxy(0.1, 0.1, 2.1, 2.1), 0.8, 0),
        ],
        vec![Detection::new(BBox::from_xyxy(4.0, 4.0, 5.0, 5.0), 0.7, 1)],
    ];

    let kept = batch_nms(&batch, NmsOptions::class_aware(0.5));

    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].len(), 1);
    assert_eq!(kept[1].len(), 1);
}

#[test]
fn test_clip_filter_helpers_drop_invalid_boxes() {
    let detections = vec![
        Detection::new(BBox::from_xyxy(-1.0, -1.0, 4.0, 4.0), 0.9, 0),
        Detection::new(BBox::from_xyxy(9.0, 9.0, 20.0, 20.0), 0.8, 1),
    ];
    let labels = vec![
        GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 4.0, 4.0), 0),
        GroundTruthBox::new(BBox::from_xyxy(9.5, 9.5, 20.0, 20.0), 1),
    ];

    let kept_dets = clip_filter_detections(&detections, 10.0, 10.0, 4.0);
    let kept_labels = clip_filter_ground_truths(&labels, 10.0, 10.0, 4.0);

    assert_eq!(kept_dets.len(), 1);
    assert_eq!(kept_labels.len(), 1);
    assert_eq!(kept_dets[0].bbox.to_xyxy(), [0.0, 0.0, 4.0, 4.0]);
}
