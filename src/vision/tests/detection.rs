use approx::assert_abs_diff_eq;

use crate::vision::detection::{BBox, BoxFormat, Detection, NmsOptions, nms};

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
