//! DataLoader transforms 集成测试

use crate::data::transforms::{Normalize, Transform};
use crate::data::{DataLoader, TensorDataset};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// 测试用：加法变换
struct AddTransform(f32);

impl Transform for AddTransform {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        tensor + self.0
    }
}

#[test]
fn test_dataloader_without_transform() {
    let features = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let labels = Tensor::new(&[0.0, 1.0], &[2, 1]);
    let dataset = TensorDataset::new(features, labels);

    let loader = DataLoader::new(dataset, 2);
    let mut iter = loader.iter();
    let (x, _y) = iter.next().unwrap();

    assert_eq!(x.shape(), &[2, 2]);
    assert_abs_diff_eq!(x[[0, 0]], 1.0, epsilon = 1e-6);
}

#[test]
fn test_dataloader_with_transform() {
    let features = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let labels = Tensor::new(&[0.0, 1.0], &[2, 1]);
    let dataset = TensorDataset::new(features, labels);

    // 每个样本 +10
    let loader = DataLoader::new(dataset, 2).with_transform(AddTransform(10.0));
    let mut iter = loader.iter();
    let (x, y) = iter.next().unwrap();

    assert_eq!(x.shape(), &[2, 2]);
    assert_abs_diff_eq!(x[[0, 0]], 11.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x[[0, 1]], 12.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x[[1, 0]], 13.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x[[1, 1]], 14.0, epsilon = 1e-6);

    // 标签不应被变换
    assert_abs_diff_eq!(y[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(y[[1, 0]], 1.0, epsilon = 1e-6);
}

#[test]
fn test_dataloader_with_normalize() {
    // [N=4, C=1, H=1, W=2]
    let features = Tensor::new(&[0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.25, 0.75], &[4, 1, 1, 2]);
    let labels = Tensor::new(&[0.0, 1.0, 2.0, 3.0], &[4, 1]);
    let dataset = TensorDataset::new(features, labels);

    // mean=0.5, std=0.5 → (x-0.5)/0.5
    let loader = DataLoader::new(dataset, 2).with_transform(Normalize::new(vec![0.5], vec![0.5]));

    let mut iter = loader.iter();

    // 第一个 batch: 样本 0 和 1
    let (x, _) = iter.next().unwrap();
    assert_eq!(x.shape(), &[2, 1, 1, 2]);
    // 样本 0: [0.0, 1.0] → [-1.0, 1.0]
    assert_abs_diff_eq!(x[[0, 0, 0, 0]], -1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x[[0, 0, 0, 1]], 1.0, epsilon = 1e-6);
    // 样本 1: [0.5, 0.5] → [0.0, 0.0]
    assert_abs_diff_eq!(x[[1, 0, 0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(x[[1, 0, 0, 1]], 0.0, epsilon = 1e-6);
}

#[test]
fn test_dataloader_transform_multiple_batches() {
    let features = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let labels = Tensor::new(&[0.0, 1.0, 2.0], &[3, 1]);
    let dataset = TensorDataset::new(features, labels);

    let loader = DataLoader::new(dataset, 2).with_transform(AddTransform(100.0));

    let batches: Vec<_> = loader.iter().collect();
    assert_eq!(batches.len(), 2); // 3 样本, batch_size=2

    // 第一个 batch: 2 个样本
    assert_eq!(batches[0].0.shape(), &[2, 2]);
    assert_abs_diff_eq!(batches[0].0[[0, 0]], 101.0, epsilon = 1e-6);

    // 第二个 batch: 1 个样本
    assert_eq!(batches[1].0.shape(), &[1, 2]);
    assert_abs_diff_eq!(batches[1].0[[0, 0]], 105.0, epsilon = 1e-6);
}
