/*
 * @Author       : 老董
 * @Date         : 2025-01-21
 * @Description  : DataLoader - PyTorch 风格的数据批量加载器
 *
 * 提供统一的数据迭代 API，支持：
 * - 自动分批 (batch_size)
 * - 随机打乱 (shuffle)
 * - 丢弃不完整批次 (drop_last)
 * - 变长序列分桶 (VarLenDataset + BucketedDataLoader)
 */

use crate::tensor::Tensor;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// `TensorDataset` - 持有特征和标签的数据集
///
/// # 示例
/// ```ignore
/// let dataset = TensorDataset::new(features, labels);
/// println!("样本数: {}", dataset.len());
/// ```
#[derive(Clone)]
pub struct TensorDataset {
    features: Tensor,
    labels: Tensor,
    len: usize,
}

impl TensorDataset {
    /// 创建新的 `TensorDataset`
    ///
    /// # 参数
    /// - `features`: 特征张量，第一维为样本数
    /// - `labels`: 标签张量，第一维为样本数（必须与 features 一致）
    ///
    /// # Panics
    /// 如果 features 和 labels 的样本数不一致
    pub fn new(features: Tensor, labels: Tensor) -> Self {
        let len = features.shape()[0];
        assert_eq!(
            len,
            labels.shape()[0],
            "TensorDataset: features 和 labels 的样本数必须一致，得到 {} vs {}",
            len,
            labels.shape()[0]
        );
        Self {
            features,
            labels,
            len,
        }
    }

    /// 获取样本数量
    pub const fn len(&self) -> usize {
        self.len
    }

    /// 检查数据集是否为空
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 获取特征张量引用
    pub const fn features(&self) -> &Tensor {
        &self.features
    }

    /// 获取标签张量引用
    pub const fn labels(&self) -> &Tensor {
        &self.labels
    }
}

/// `DataLoader` - `PyTorch` 风格的数据批量加载器
///
/// # 示例
/// ```ignore
/// let dataset = TensorDataset::new(train_x, train_y);
/// let loader = DataLoader::new(dataset, 32)
///     .shuffle(true)
///     .drop_last(true);
///
/// for (x_batch, y_batch) in loader.iter() {
///     model.forward(&x_batch)?;
///     loss.backward()?;
///     optimizer.step()?;
/// }
/// ```
pub struct DataLoader {
    dataset: TensorDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    seed: Option<u64>,
}

impl DataLoader {
    /// 创建新的 `DataLoader`
    ///
    /// # 参数
    /// - `dataset`: 数据集
    /// - `batch_size`: 批大小
    pub fn new(dataset: TensorDataset, batch_size: usize) -> Self {
        assert!(batch_size > 0, "DataLoader: batch_size 必须大于 0");
        Self {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
            seed: None,
        }
    }

    /// 设置是否打乱数据
    pub const fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// 设置是否丢弃最后一个不完整的批次
    pub const fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// 设置随机种子（用于 shuffle）
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// 获取批次数量
    pub const fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// 获取数据集大小
    pub const fn len(&self) -> usize {
        self.dataset.len()
    }

    /// 检查是否为空
    pub const fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// 创建迭代器
    pub fn iter(&self) -> DataLoaderIterator<'_> {
        // 生成索引
        let n = self.dataset.len();
        let mut indices: Vec<usize> = (0..n).collect();

        // 如果需要打乱
        if self.shuffle {
            if let Some(seed) = self.seed {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
            } else {
                let mut rng = rand::thread_rng();
                indices.shuffle(&mut rng);
            }
        }

        DataLoaderIterator {
            loader: self,
            indices,
            current_batch: 0,
        }
    }
}

/// `DataLoader` 迭代器
pub struct DataLoaderIterator<'a> {
    loader: &'a DataLoader,
    indices: Vec<usize>,
    current_batch: usize,
}

impl Iterator for DataLoaderIterator<'_> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.loader.dataset.len();
        let batch_size = self.loader.batch_size;
        let start = self.current_batch * batch_size;

        // 检查是否还有数据
        if start >= n {
            return None;
        }

        let end = (start + batch_size).min(n);
        let actual_batch_size = end - start;

        // 如果 drop_last 且批次不完整，则跳过
        if self.loader.drop_last && actual_batch_size < batch_size {
            return None;
        }

        self.current_batch += 1;

        // 提取批次数据
        let batch_indices = &self.indices[start..end];
        let (features_batch, labels_batch) = extract_batch(&self.loader.dataset, batch_indices);

        Some((features_batch, labels_batch))
    }
}

/// 从数据集中按索引提取批次
fn extract_batch(dataset: &TensorDataset, indices: &[usize]) -> (Tensor, Tensor) {
    let features = &dataset.features;
    let labels = &dataset.labels;

    let batch_size = indices.len();
    let feature_shape = features.shape();
    let label_shape = labels.shape();

    // 计算每个样本的特征/标签大小
    let feature_sample_size: usize = feature_shape[1..].iter().product();
    let label_sample_size: usize = label_shape[1..].iter().product();

    // 获取扁平化视图
    let flat_features = features.flatten_view();
    let flat_labels = labels.flatten_view();

    // 提取特征
    let mut feature_data = Vec::with_capacity(batch_size * feature_sample_size);
    for &idx in indices {
        let sample_start = idx * feature_sample_size;
        for i in 0..feature_sample_size {
            feature_data.push(flat_features[sample_start + i]);
        }
    }

    // 提取标签
    let mut label_data = Vec::with_capacity(batch_size * label_sample_size);
    for &idx in indices {
        let sample_start = idx * label_sample_size;
        for i in 0..label_sample_size {
            label_data.push(flat_labels[sample_start + i]);
        }
    }

    // 构建新的形状
    let mut new_feature_shape = vec![batch_size];
    new_feature_shape.extend_from_slice(&feature_shape[1..]);

    let mut new_label_shape = vec![batch_size];
    new_label_shape.extend_from_slice(&label_shape[1..]);

    let features_batch = Tensor::new(&feature_data, &new_feature_shape);
    let labels_batch = Tensor::new(&label_data, &new_label_shape);

    (features_batch, labels_batch)
}

// ==================== 变长序列支持 ====================

/// 变长样本
///
/// 用于存储长度可变的序列数据。
#[derive(Debug, Clone)]
pub struct VarLenSample {
    /// 序列数据（展平为 1D，实际形状由 `seq_len` 和 `feature_size` 决定）
    pub features: Vec<f32>,
    /// 序列长度
    pub seq_len: usize,
    /// 特征维度
    pub feature_size: usize,
    /// 标签（固定大小）
    pub label: Vec<f32>,
}

impl VarLenSample {
    /// 创建新的变长样本
    ///
    /// # 参数
    /// - `features`: 特征数据，长度应为 `seq_len` * `feature_size`
    /// - `seq_len`: 序列长度
    /// - `feature_size`: 特征维度
    /// - `label`: 标签
    pub fn new(features: Vec<f32>, seq_len: usize, feature_size: usize, label: Vec<f32>) -> Self {
        debug_assert_eq!(
            features.len(),
            seq_len * feature_size,
            "features 长度应为 seq_len * feature_size"
        );
        Self {
            features,
            seq_len,
            feature_size,
            label,
        }
    }
}

/// 变长数据集
///
/// 存储长度可变的序列样本，支持按长度分桶。
///
/// # 示例
/// ```ignore
/// let mut dataset = VarLenDataset::new(1, 2);  // feature_size=1, label_size=2
/// dataset.push(VarLenSample::new(vec![1.0, 0.0, 1.0], 3, 1, vec![0.0, 1.0]));
/// dataset.push(VarLenSample::new(vec![1.0, 1.0, 0.0, 1.0, 0.0], 5, 1, vec![1.0, 0.0]));
///
/// let loader = BucketedDataLoader::new(&dataset);
/// for (x_batch, y_batch) in loader.iter() {
///     // x_batch 同一批次内序列长度相同
/// }
/// ```
pub struct VarLenDataset {
    samples: Vec<VarLenSample>,
    feature_size: usize,
    label_size: usize,
}

impl VarLenDataset {
    /// 创建新的变长数据集
    ///
    /// # 参数
    /// - `feature_size`: 特征维度
    /// - `label_size`: 标签维度
    pub const fn new(feature_size: usize, label_size: usize) -> Self {
        Self {
            samples: Vec::new(),
            feature_size,
            label_size,
        }
    }

    /// 添加样本
    pub fn push(&mut self, sample: VarLenSample) {
        debug_assert_eq!(sample.feature_size, self.feature_size);
        debug_assert_eq!(sample.label.len(), self.label_size);
        self.samples.push(sample);
    }

    /// 获取样本数量
    pub const fn len(&self) -> usize {
        self.samples.len()
    }

    /// 检查是否为空
    pub const fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// 获取所有样本的引用
    pub fn samples(&self) -> &[VarLenSample] {
        &self.samples
    }

    /// 获取特征维度
    pub const fn feature_size(&self) -> usize {
        self.feature_size
    }

    /// 获取标签维度
    pub const fn label_size(&self) -> usize {
        self.label_size
    }

    /// 按长度分桶
    fn bucket_by_length(&self) -> HashMap<usize, Vec<usize>> {
        let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, sample) in self.samples.iter().enumerate() {
            buckets.entry(sample.seq_len).or_default().push(idx);
        }
        buckets
    }
}

/// 分桶数据加载器
///
/// 自动将相同长度的序列放在一起批处理。
///
/// # 示例
/// ```ignore
/// let loader = BucketedDataLoader::new(&dataset);
///
/// for (x_batch, y_batch) in loader.iter() {
///     // x_batch: [batch, seq_len, feature_size]
///     // 同一批次内 seq_len 相同，不同批次可能不同
///     let output = model.forward(&x_batch)?;
///     let loss = criterion.forward(&output, &y_batch)?;
/// }
/// ```
pub struct BucketedDataLoader<'a> {
    dataset: &'a VarLenDataset,
    shuffle: bool,
    seed: Option<u64>,
}

impl<'a> BucketedDataLoader<'a> {
    /// 创建分桶数据加载器
    ///
    /// # 参数
    /// - `dataset`: 变长数据集引用
    pub const fn new(dataset: &'a VarLenDataset) -> Self {
        Self {
            dataset,
            shuffle: false,
            seed: None,
        }
    }

    /// 设置是否打乱（在每个桶内打乱）
    pub const fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// 设置随机种子
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// 创建迭代器
    ///
    /// 每次迭代返回同一长度的所有样本组成的批次。
    pub fn iter(&self) -> BucketedDataLoaderIterator<'_> {
        let mut buckets: Vec<(usize, Vec<usize>)> =
            self.dataset.bucket_by_length().into_iter().collect();

        // 可选：按 seq_len 排序（使输出顺序确定）
        buckets.sort_by_key(|(seq_len, _)| *seq_len);

        // 可选：打乱每个桶内的样本顺序
        if self.shuffle {
            if let Some(seed) = self.seed {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                for (_, indices) in &mut buckets {
                    indices.shuffle(&mut rng);
                }
            } else {
                let mut rng = rand::thread_rng();
                for (_, indices) in &mut buckets {
                    indices.shuffle(&mut rng);
                }
            }
        }

        BucketedDataLoaderIterator {
            dataset: self.dataset,
            buckets,
            current_bucket: 0,
        }
    }

    /// 获取桶的数量（即不同长度的数量）
    pub fn num_buckets(&self) -> usize {
        self.dataset.bucket_by_length().len()
    }
}

/// 分桶数据加载器迭代器
pub struct BucketedDataLoaderIterator<'a> {
    dataset: &'a VarLenDataset,
    buckets: Vec<(usize, Vec<usize>)>,
    current_bucket: usize,
}

impl Iterator for BucketedDataLoaderIterator<'_> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_bucket >= self.buckets.len() {
            return None;
        }

        let (seq_len, indices) = &self.buckets[self.current_bucket];
        self.current_bucket += 1;

        let batch_size = indices.len();
        let feature_size = self.dataset.feature_size;
        let label_size = self.dataset.label_size;

        // 收集特征数据
        let mut feature_data = Vec::with_capacity(batch_size * seq_len * feature_size);
        for &idx in indices {
            feature_data.extend(&self.dataset.samples[idx].features);
        }

        // 收集标签数据
        let mut label_data = Vec::with_capacity(batch_size * label_size);
        for &idx in indices {
            label_data.extend(&self.dataset.samples[idx].label);
        }

        let features = Tensor::new(&feature_data, &[batch_size, *seq_len, feature_size]);
        let labels = Tensor::new(&label_data, &[batch_size, label_size]);

        Some((features, labels))
    }
}

impl ExactSizeIterator for BucketedDataLoaderIterator<'_> {
    fn len(&self) -> usize {
        self.buckets.len() - self.current_bucket
    }
}
