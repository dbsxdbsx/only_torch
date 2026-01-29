//! DataLoader 单元测试

use crate::data::{
    BucketedSampling, DataLoader, Dataset, SamplingStrategy, SequentialSampling, TensorDataset,
    VarLenDataset, VarLenSample,
};
use crate::tensor::Tensor;

// ═══════════════════════════════════════════════════════════════
// TensorDataset 测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_tensor_dataset() {
    let features = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let labels = Tensor::new(&[0.0, 1.0, 0.0], &[3, 1]);
    let dataset = TensorDataset::new(features, labels);

    assert_eq!(dataset.len(), 3);
    assert!(!dataset.is_empty());
}

#[test]
#[should_panic(expected = "样本数必须一致")]
fn test_tensor_dataset_mismatched_samples() {
    let features = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let labels = Tensor::new(&[0.0, 1.0, 0.0], &[3, 1]);
    TensorDataset::new(features, labels);
}

#[test]
fn test_tensor_dataset_get_batch() {
    let features = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let labels = Tensor::new(&[0.0, 1.0, 2.0], &[3, 1]);
    let dataset = TensorDataset::new(features, labels);

    let (batch_features, batch_labels) = dataset.get_batch(&[0, 2]);

    assert_eq!(batch_features.shape(), &[2, 2]);
    assert_eq!(batch_labels.shape(), &[2, 1]);

    // 验证数据内容
    let flat = batch_features.flatten_view();
    assert_eq!(flat[0], 1.0); // 第 0 个样本的第一个特征
    assert_eq!(flat[1], 2.0); // 第 0 个样本的第二个特征
    assert_eq!(flat[2], 5.0); // 第 2 个样本的第一个特征
    assert_eq!(flat[3], 6.0); // 第 2 个样本的第二个特征
}

// ═══════════════════════════════════════════════════════════════
// DataLoader 基础测试
// ═══════════════════════════════════════════════════════════════

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
    let dataset = TensorDataset::new(features, labels);

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

// ═══════════════════════════════════════════════════════════════
// SequentialSampling 测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_sequential_sampling_no_shuffle() {
    let mut strategy = SequentialSampling::new();
    let batches = strategy.generate_batches(5, 2, None);

    assert_eq!(batches.len(), 3);
    assert_eq!(batches[0], vec![0, 1]);
    assert_eq!(batches[1], vec![2, 3]);
    assert_eq!(batches[2], vec![4]);
}

#[test]
fn test_sequential_sampling_drop_last() {
    let mut strategy = SequentialSampling::new().drop_last(true);
    let batches = strategy.generate_batches(5, 2, None);

    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0], vec![0, 1]);
    assert_eq!(batches[1], vec![2, 3]);
}

#[test]
fn test_sequential_sampling_shuffle_deterministic() {
    let mut strategy1 = SequentialSampling::new().shuffle(true).seed(123);
    let mut strategy2 = SequentialSampling::new().shuffle(true).seed(123);

    let batches1 = strategy1.generate_batches(10, 3, None);
    let batches2 = strategy2.generate_batches(10, 3, None);

    assert_eq!(batches1, batches2);
}

// ═══════════════════════════════════════════════════════════════
// VarLenDataset 测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_var_len_dataset_creation() {
    let mut dataset = VarLenDataset::new(2, 3);

    dataset.push(VarLenSample::new(
        vec![1.0, 2.0, 3.0, 4.0],
        2,
        2,
        vec![0.0, 1.0, 0.0],
    ));
    dataset.push(VarLenSample::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        3,
        2,
        vec![1.0, 0.0, 0.0],
    ));

    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.feature_size(), 2);
    assert_eq!(dataset.label_size(), 3);
    assert_eq!(dataset.num_buckets(), 2); // 长度 2 和长度 3
}

#[test]
fn test_var_len_dataset_bucket_key() {
    let mut dataset = VarLenDataset::new(1, 2);

    dataset.push(VarLenSample::new(vec![1.0, 2.0, 3.0], 3, 1, vec![0.0, 1.0]));
    dataset.push(VarLenSample::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        5,
        1,
        vec![1.0, 0.0],
    ));
    dataset.push(VarLenSample::new(vec![1.0, 2.0, 3.0], 3, 1, vec![0.0, 1.0]));

    assert_eq!(dataset.bucket_key(0), Some(3));
    assert_eq!(dataset.bucket_key(1), Some(5));
    assert_eq!(dataset.bucket_key(2), Some(3));
}

#[test]
fn test_var_len_dataset_get_batch() {
    let mut dataset = VarLenDataset::new(1, 2);

    // 添加两个长度为 3 的样本
    dataset.push(VarLenSample::new(vec![1.0, 2.0, 3.0], 3, 1, vec![0.0, 1.0]));
    dataset.push(VarLenSample::new(vec![4.0, 5.0, 6.0], 3, 1, vec![1.0, 0.0]));

    let (features, labels) = dataset.get_batch(&[0, 1]);

    assert_eq!(features.shape(), &[2, 3, 1]); // [batch, seq_len, feature_size]
    assert_eq!(labels.shape(), &[2, 2]); // [batch, label_size]

    // 验证特征内容
    let flat = features.flatten_view();
    assert_eq!(flat[0], 1.0);
    assert_eq!(flat[1], 2.0);
    assert_eq!(flat[2], 3.0);
    assert_eq!(flat[3], 4.0);
    assert_eq!(flat[4], 5.0);
    assert_eq!(flat[5], 6.0);
}

// ═══════════════════════════════════════════════════════════════
// BucketedSampling 测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_bucketed_sampling_basic() {
    let bucket_keys = vec![3, 5, 3, 5, 3]; // 3 个长度为 3，2 个长度为 5

    let mut strategy = BucketedSampling::new();
    let batches = strategy.generate_batches(5, 0, Some(&bucket_keys));

    assert_eq!(batches.len(), 2); // 2 个桶

    // 桶按 key 排序，所以长度 3 的桶在前
    assert_eq!(batches[0].len(), 3); // 长度为 3 的样本
    assert_eq!(batches[1].len(), 2); // 长度为 5 的样本

    // 验证索引
    assert!(batches[0].contains(&0));
    assert!(batches[0].contains(&2));
    assert!(batches[0].contains(&4));
    assert!(batches[1].contains(&1));
    assert!(batches[1].contains(&3));
}

#[test]
fn test_bucketed_sampling_shuffle_deterministic() {
    let bucket_keys = vec![3, 5, 3, 5, 3, 5, 3];

    let mut strategy1 = BucketedSampling::new().shuffle(true).seed(42);
    let mut strategy2 = BucketedSampling::new().shuffle(true).seed(42);

    let batches1 = strategy1.generate_batches(7, 0, Some(&bucket_keys));
    let batches2 = strategy2.generate_batches(7, 0, Some(&bucket_keys));

    assert_eq!(batches1, batches2);
}

// ═══════════════════════════════════════════════════════════════
// DataLoader + VarLenDataset 集成测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_dataloader_from_var_len() {
    let mut dataset = VarLenDataset::new(1, 2);

    // 添加不同长度的样本
    dataset.push(VarLenSample::new(vec![1.0, 2.0, 3.0], 3, 1, vec![0.0, 1.0]));
    dataset.push(VarLenSample::new(
        vec![4.0, 5.0, 6.0, 7.0, 8.0],
        5,
        1,
        vec![1.0, 0.0],
    ));
    dataset.push(VarLenSample::new(
        vec![9.0, 10.0, 11.0],
        3,
        1,
        vec![0.0, 1.0],
    ));

    let loader = DataLoader::from_var_len(&dataset);

    let batches: Vec<_> = loader.iter().collect();
    assert_eq!(batches.len(), 2); // 2 个桶

    // 第一个桶（长度 3）应该有 2 个样本
    let (feat1, lab1) = &batches[0];
    assert_eq!(feat1.shape(), &[2, 3, 1]);
    assert_eq!(lab1.shape(), &[2, 2]);

    // 第二个桶（长度 5）应该有 1 个样本
    let (feat2, lab2) = &batches[1];
    assert_eq!(feat2.shape(), &[1, 5, 1]);
    assert_eq!(lab2.shape(), &[1, 2]);
}

#[test]
fn test_dataloader_from_var_len_shuffle() {
    let mut dataset = VarLenDataset::new(1, 2);

    for i in 0..10 {
        let len = if i % 2 == 0 { 3 } else { 5 };
        let features: Vec<f32> = (0..len).map(|j| (i * 10 + j) as f32).collect();
        dataset.push(VarLenSample::new(features, len, 1, vec![0.0, 1.0]));
    }

    let loader1 = DataLoader::from_var_len(&dataset).shuffle(true).seed(99);
    let loader2 = DataLoader::from_var_len(&dataset).shuffle(true).seed(99);

    let batches1: Vec<_> = loader1.iter().collect();
    let batches2: Vec<_> = loader2.iter().collect();

    assert_eq!(batches1.len(), batches2.len());
    for i in 0..batches1.len() {
        assert_eq!(batches1[i].0.flatten_view(), batches2[i].0.flatten_view());
    }
}

// ═══════════════════════════════════════════════════════════════
// 边界情况测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_dataloader_empty_dataset() {
    let features = Tensor::new(&[] as &[f32], &[0, 2]);
    let labels = Tensor::new(&[] as &[f32], &[0, 1]);
    let dataset = TensorDataset::new(features, labels);

    let loader = DataLoader::new(dataset, 2);

    assert!(loader.is_empty());
    assert_eq!(loader.num_batches(), 0);
    assert_eq!(loader.iter().count(), 0);
}

#[test]
fn test_dataloader_batch_size_larger_than_dataset() {
    let features = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let labels = Tensor::new(&[0.0, 1.0], &[2, 1]);
    let dataset = TensorDataset::new(features, labels);

    let loader = DataLoader::new(dataset.clone(), 10);

    assert_eq!(loader.num_batches(), 1);
    let batches: Vec<_> = loader.iter().collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].0.shape()[0], 2);

    // drop_last=true 时应该没有批次
    let loader = DataLoader::new(dataset, 10).drop_last(true);
    assert_eq!(loader.num_batches(), 0);
}

#[test]
fn test_dataloader_exact_batch_size() {
    let features = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let labels = Tensor::new(&[0.0, 1.0, 2.0], &[3, 1]);
    let dataset = TensorDataset::new(features, labels);

    let loader = DataLoader::new(dataset, 3);

    assert_eq!(loader.num_batches(), 1);
    let batches: Vec<_> = loader.iter().collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].0.shape()[0], 3);
}

// ═══════════════════════════════════════════════════════════════
// 多维张量测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_dataloader_4d_features() {
    // 模拟图像数据 [batch, channels, height, width]
    let features = Tensor::new(
        &(0..24).map(|x| x as f32).collect::<Vec<_>>(),
        &[2, 3, 2, 2],
    );
    let labels = Tensor::new(&[0.0, 1.0], &[2, 1]);
    let dataset = TensorDataset::new(features, labels);

    let loader = DataLoader::new(dataset, 1);

    let batches: Vec<_> = loader.iter().collect();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].0.shape(), &[1, 3, 2, 2]);
    assert_eq!(batches[1].0.shape(), &[1, 3, 2, 2]);
}
