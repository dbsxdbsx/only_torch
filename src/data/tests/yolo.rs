use approx::assert_abs_diff_eq;

use crate::vision::detection::parse_yolo_txt_labels;

#[test]
fn test_parse_yolo_txt_labels_allows_blank_lines_and_comments() {
    let content = "\n# class cx cy w h\n0 0.5 0.5 0.2 0.4\n2 0.5 0.5 0.3 0.4\n";

    let labels = parse_yolo_txt_labels(content, 3).unwrap();

    assert_eq!(labels.len(), 2);
    assert_eq!(labels[0].class_id, 0);
    assert_eq!(labels[1].class_id, 2);
    let [x1, y1, x2, y2] = labels[0].bbox.to_xyxy();
    assert_abs_diff_eq!(x1, 0.4, epsilon = 1e-6);
    assert_abs_diff_eq!(y1, 0.3, epsilon = 1e-6);
    assert_abs_diff_eq!(x2, 0.6, epsilon = 1e-6);
    assert_abs_diff_eq!(y2, 0.7, epsilon = 1e-6);
}

#[test]
fn test_parse_yolo_txt_labels_rejects_bad_field_count() {
    let err = parse_yolo_txt_labels("0 0.5 0.5 0.2", 1).unwrap_err();

    assert!(
        format!("{err}").contains("期望 5 个字段"),
        "错误信息应说明字段数错误，实际: {err}"
    );
}

#[test]
fn test_parse_yolo_txt_labels_rejects_class_out_of_range() {
    let err = parse_yolo_txt_labels("3 0.5 0.5 0.2 0.2", 3).unwrap_err();

    assert!(
        format!("{err}").contains("类别 ID 3 越界"),
        "错误信息应说明类别越界，实际: {err}"
    );
}

#[test]
fn test_parse_yolo_txt_labels_rejects_coordinate_out_of_range() {
    let err = parse_yolo_txt_labels("0 1.2 0.5 0.2 0.2", 1).unwrap_err();

    assert!(
        format!("{err}").contains("必须在 [0, 1]"),
        "错误信息应说明坐标范围，实际: {err}"
    );
}

#[test]
fn test_parse_yolo_txt_labels_rejects_zero_classes() {
    let err = parse_yolo_txt_labels("0 0.5 0.5 0.2 0.2", 0).unwrap_err();

    assert!(
        format!("{err}").contains("num_classes > 0"),
        "错误信息应说明类别数约束，实际: {err}"
    );
}

#[test]
fn test_parse_yolo_txt_labels_rejects_negative_class_id() {
    let err = parse_yolo_txt_labels("-1 0.5 0.5 0.2 0.2", 1).unwrap_err();

    assert!(
        format!("{err}").contains("类别 ID 不是非负整数"),
        "错误信息应说明类别 ID 非法，实际: {err}"
    );
}

#[test]
fn test_parse_yolo_txt_labels_rejects_zero_area_box() {
    let err = parse_yolo_txt_labels("0 0.5 0.5 0.0 0.2", 1).unwrap_err();

    assert!(
        format!("{err}").contains("宽高必须大于 0"),
        "错误信息应说明 bbox 正面积约束，实际: {err}"
    );
}

#[test]
fn test_parse_yolo_txt_labels_rejects_box_outside_normalized_image() {
    let err = parse_yolo_txt_labels("0 0.1 0.5 0.4 0.2", 1).unwrap_err();

    assert!(
        format!("{err}").contains("xyxy 后必须落在 [0, 1]"),
        "错误信息应说明转换后 bbox 越界，实际: {err}"
    );
}
