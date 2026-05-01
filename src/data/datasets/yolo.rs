//! YOLO txt 检测标签解析。
//!
//! 支持常见 YOLOv5 txt 格式：`<class> <cx> <cy> <w> <h>`，坐标归一化到 `[0, 1]`。

use crate::data::DataError;
use crate::vision::detection::{BBox, BoxFormat, GroundTruthBox};
use std::path::Path;

/// 解析 YOLO txt 标签文件。
pub fn parse_yolo_txt_file(
    path: impl AsRef<Path>,
    num_classes: usize,
) -> Result<Vec<GroundTruthBox>, DataError> {
    let content = std::fs::read_to_string(path)?;
    parse_yolo_txt_labels(&content, num_classes)
}

/// 解析 YOLO txt 标签内容。
pub fn parse_yolo_txt_labels(
    content: &str,
    num_classes: usize,
) -> Result<Vec<GroundTruthBox>, DataError> {
    if num_classes == 0 {
        return Err(DataError::FormatError(
            "YOLO 标签解析需要 num_classes > 0".to_string(),
        ));
    }

    let mut labels = Vec::new();
    for (line_idx, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 5 {
            return Err(yolo_format_error(
                line_idx,
                format!("期望 5 个字段，实际 {}", fields.len()),
            ));
        }

        let class_id = fields[0]
            .parse::<usize>()
            .map_err(|e| yolo_format_error(line_idx, format!("类别 ID 不是非负整数: {e}")))?;
        if class_id >= num_classes {
            return Err(yolo_format_error(
                line_idx,
                format!("类别 ID {class_id} 越界，num_classes={num_classes}"),
            ));
        }

        let mut coords = [0.0f32; 4];
        for i in 0..4 {
            coords[i] = fields[i + 1].parse::<f32>().map_err(|e| {
                yolo_format_error(line_idx, format!("坐标字段 {} 不是 f32: {e}", i + 1))
            })?;
            if !(0.0..=1.0).contains(&coords[i]) {
                return Err(yolo_format_error(
                    line_idx,
                    format!("坐标字段 {} 必须在 [0, 1]，得到 {}", i + 1, coords[i]),
                ));
            }
        }
        if coords[2] <= 0.0 || coords[3] <= 0.0 {
            return Err(yolo_format_error(
                line_idx,
                format!("bbox 宽高必须大于 0，得到 w={}, h={}", coords[2], coords[3]),
            ));
        }

        let bbox = BBox::from_array(coords, BoxFormat::CxCyWh);
        if bbox.x1 < 0.0 || bbox.y1 < 0.0 || bbox.x2 > 1.0 || bbox.y2 > 1.0 {
            return Err(yolo_format_error(
                line_idx,
                format!(
                    "bbox 转换到 xyxy 后必须落在 [0, 1]，得到 {:?}",
                    bbox.to_xyxy()
                ),
            ));
        }

        labels.push(GroundTruthBox::new(bbox, class_id));
    }
    Ok(labels)
}

fn yolo_format_error(line_idx: usize, message: String) -> DataError {
    DataError::FormatError(format!(
        "YOLO 标签第 {} 行格式错误: {message}",
        line_idx + 1
    ))
}
