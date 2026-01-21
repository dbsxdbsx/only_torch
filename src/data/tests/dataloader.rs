//! DataLoader 单元测试

use crate::data::{DataLoader, TensorDataset};
use crate::tensor::Tensor;

#[test]
fn test_tensor_dataset() {
    let features = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let labels = Tensor::new(&[0.0, 1.0, 0.0], &[3, 1]);
    let dataset = TensorDataset::new(features, labels);

    assert_eq!(dataset.len(), 3);
    assert!(!dataset.is_empty());
}

#[test]
fn test_dataloader_basic() {
    let features = Tensor::new(&(0..20).map(|x| x as f32).collect::<Vec<_>>(), &[10, 2]);
    let labels = Tensor::new(&(0..10).map(|x| x as f32).collect::<Vec<_>>(), &[10, 1]);
    let dataset = TensorDataset::new(features, labels);

    let loader = DataLoader::new(dataset, 3);
    assert_eq!(loader.num_batches(), 4); // 10 / 3 = 3.33, 向上取整 = 4

    let batches: Vec<_> = loader.iter().collect();
    assert_eq!(batches.len(), 4);

    // 前 3 个批次大小为 3
    assert_eq!(batches[0].0.shape()[0], 3);
    assert_eq!(batches[1].0.shape()[0], 3);
    assert_eq!(batches[2].0.shape()[0], 3);
    // 最后一个批次大小为 1
    assert_eq!(batches[3].0.shape()[0], 1);
}

#[test]
fn test_dataloader_drop_last() {
    let features = Tensor::new(&(0..20).map(|x| x as f32).collect::<Vec<_>>(), &[10, 2]);
    let labels = Tensor::new(&(0..10).map(|x| x as f32).collect::<Vec<_>>(), &[10, 1]);
    let dataset = TensorDataset::new(features, labels);

    let loader = DataLoader::new(dataset, 3).drop_last(true);
    assert_eq!(loader.num_batches(), 3); // 10 / 3 = 3

    let batches: Vec<_> = loader.iter().collect();
    assert_eq!(batches.len(), 3);
}

#[test]
fn test_dataloader_shuffle_with_seed() {
    let features = Tensor::new(&(0..20).map(|x| x as f32).collect::<Vec<_>>(), &[10, 2]);
    let labels = Tensor::new(&(0..10).map(|x| x as f32).collect::<Vec<_>>(), &[10, 1]);
    let dataset = TensorDataset::new(features.clone(), labels.clone());

    // 使用相同种子，两次迭代应该产生相同的结果
    let loader1 = DataLoader::new(dataset.clone(), 3).shuffle(true).seed(42);
    let loader2 = DataLoader::new(dataset, 3).shuffle(true).seed(42);

    let batches1: Vec<_> = loader1.iter().collect();
    let batches2: Vec<_> = loader2.iter().collect();

    assert_eq!(batches1.len(), batches2.len());
    for (b1, b2) in batches1.iter().zip(batches2.iter()) {
        assert_eq!(b1.0, b2.0);
        assert_eq!(b1.1, b2.1);
    }
}

#[test]
fn test_dataloader_3d_features() {
    // 模拟 RNN 输入: [samples, seq_len, input_size]
    let features = Tensor::new(
        &(0..24).map(|x| x as f32).collect::<Vec<_>>(),
        &[4, 3, 2], // 4 samples, seq_len=3, input_size=2
    );
    let labels = Tensor::new(&[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], &[4, 2]); // one-hot
    let dataset = TensorDataset::new(features, labels);

    let loader = DataLoader::new(dataset, 2);
    let batches: Vec<_> = loader.iter().collect();

    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].0.shape(), &[2, 3, 2]); // batch=2, seq_len=3, input=2
    assert_eq!(batches[0].1.shape(), &[2, 2]); // batch=2, num_classes=2
}
