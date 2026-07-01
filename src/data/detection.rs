//! Detection 训练数据组装。
//!
//! 这里只放"喂给 dataloader 的数据载体"：每张图配一组变长 GT、batch 组装时
//! 图像堆叠、label 保持每图变长列表。
//!
//! label 几何变换（letterbox / hflip / clip-filter）见
//! `vision::detection::transform`；YOLO 等标签格式解析见
//! `vision::detection::io`。

use crate::data::DataError;
use crate::tensor::Tensor;
use crate::vision::detection::GroundTruthBox;

/// 单张图像及其变长检测标签。
#[derive(Clone, Debug)]
pub struct DetectionSample {
    pub image: Tensor,
    pub labels: Vec<GroundTruthBox>,
}

impl DetectionSample {
    pub fn new(image: Tensor, labels: Vec<GroundTruthBox>) -> Self {
        Self { image, labels }
    }
}

/// Detection batch：图像可堆叠为 Tensor，标签保持每图变长列表。
#[derive(Clone, Debug)]
pub struct DetectionBatch {
    pub images: Tensor,
    pub labels: Vec<Vec<GroundTruthBox>>,
}

impl DetectionBatch {
    pub fn from_samples(samples: &[DetectionSample]) -> Result<Self, DataError> {
        let first = samples.first().ok_or_else(|| {
            DataError::FormatError("DetectionBatch 至少需要 1 个样本".to_string())
        })?;
        let sample_shape = first.image.shape().to_vec();
        let sample_size: usize = sample_shape.iter().product();
        let mut data = Vec::with_capacity(samples.len() * sample_size);
        let mut labels = Vec::with_capacity(samples.len());

        for sample in samples {
            if sample.image.shape() != sample_shape {
                return Err(DataError::ShapeMismatch {
                    expected: sample_shape.clone(),
                    got: sample.image.shape().to_vec(),
                });
            }
            // to_vec 按逻辑行主序展开、对任意布局都成立（非连续视图不会 panic）。
            data.extend_from_slice(&sample.image.to_vec());
            labels.push(sample.labels.clone());
        }

        let mut batch_shape = vec![samples.len()];
        batch_shape.extend_from_slice(&sample_shape);
        Ok(Self {
            images: Tensor::new(&data, &batch_shape),
            labels,
        })
    }
}
