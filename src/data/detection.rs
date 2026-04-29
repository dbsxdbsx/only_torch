//! Detection 数据结构与 batch 组装。

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
            data.extend_from_slice(&sample.image.flatten_view().to_vec());
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
