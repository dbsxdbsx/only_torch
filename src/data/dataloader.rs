/*
 * @Author       : 老董
 * @Date         : 2025-01-21
 * @LastModified : 2025-01-29
 * @Description  : DataLoader - PyTorch 风格的数据批量加载器
 *
 * 提供统一的数据迭代 API，支持：
 * - 自动分批 (batch_size)
 * - 随机打乱 (shuffle)
 * - 丢弃不完整批次 (drop_last)
 * - 变长序列分桶 (VarLenDataset + BucketedSampling)
 *
 * 架构设计：
 * - Dataset trait: 数据集抽象
 * - SamplingStrategy trait: 采样策略抽象
 * - DataLoader<D, S>: 组合数据集和采样策略
 */

use crate::tensor::Tensor;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// Dataset Trait
// ═══════════════════════════════════════════════════════════════

/// 数据集 trait
///
/// 定义数据集的基本接口，支持固定长度和变长序列数据。
pub trait Dataset {
    /// 获取样本数量
    fn len(&self) -> usize;

    /// 检查数据集是否为空
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 按索引列表提取批次
    ///
    /// 返回 `(features, labels)` 张量对
    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor);

    /// 获取样本的分桶键（用于 BucketedSampling）
    ///
    /// 默认返回 `None`，表示不支持分桶。
    /// `VarLenDataset` 返回序列长度作为分桶键。
    fn bucket_key(&self, _index: usize) -> Option<usize> {
        None
    }
}

// ═══════════════════════════════════════════════════════════════
// SamplingStrategy Trait
// ═══════════════════════════════════════════════════════════════

/// 采样策略 trait
///
/// 定义如何将数据集划分为批次。
pub trait SamplingStrategy: Clone {
    /// 生成批次索引列表
    ///
    /// # 参数
    /// - `dataset_len`: 数据集大小
    /// - `batch_size`: 批大小（某些策略可能忽略，如 BucketedSampling）
    /// - `bucket_keys`: 可选的分桶键数组（用于 BucketedSampling）
    ///
    /// # 返回
    /// 批次索引列表，每个元素是一个批次中的样本索引
    fn generate_batches(
        &mut self,
        dataset_len: usize,
        batch_size: usize,
        bucket_keys: Option<&[usize]>,
    ) -> Vec<Vec<usize>>;
}

// ═══════════════════════════════════════════════════════════════
// SequentialSampling - 顺序/打乱采样
// ═══════════════════════════════════════════════════════════════

/// 顺序采样策略
///
/// 按顺序或随机打乱的方式将数据集划分为固定大小的批次。
#[derive(Clone)]
pub struct SequentialSampling {
    shuffle: bool,
    drop_last: bool,
    seed: Option<u64>,
}

impl Default for SequentialSampling {
    fn default() -> Self {
        Self::new()
    }
}

impl SequentialSampling {
    /// 创建新的顺序采样策略
    pub const fn new() -> Self {
        Self {
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

    /// 设置随机种子
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl SamplingStrategy for SequentialSampling {
    fn generate_batches(
        &mut self,
        dataset_len: usize,
        batch_size: usize,
        _bucket_keys: Option<&[usize]>,
    ) -> Vec<Vec<usize>> {
        // 生成索引
        let mut indices: Vec<usize> = (0..dataset_len).collect();

        // 可选打乱
        if self.shuffle {
            if let Some(seed) = self.seed {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
            } else {
                let mut rng = rand::thread_rng();
                indices.shuffle(&mut rng);
            }
        }

        // 划分批次
        let mut batches = Vec::new();
        for chunk in indices.chunks(batch_size) {
            if self.drop_last && chunk.len() < batch_size {
                break;
            }
            batches.push(chunk.to_vec());
        }

        batches
    }
}

// ═══════════════════════════════════════════════════════════════
// BucketedSampling - 分桶采样
// ═══════════════════════════════════════════════════════════════

/// 分桶采样策略
///
/// 将相同长度的样本分到同一个桶中，每个桶作为一个批次。
/// 适用于变长序列数据（RNN、LSTM 等）。
#[derive(Clone)]
pub struct BucketedSampling {
    shuffle: bool,
    seed: Option<u64>,
}

impl Default for BucketedSampling {
    fn default() -> Self {
        Self::new()
    }
}

impl BucketedSampling {
    /// 创建新的分桶采样策略
    pub const fn new() -> Self {
        Self {
            shuffle: false,
            seed: None,
        }
    }

    /// 设置是否打乱（桶内打乱）
    pub const fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// 设置随机种子
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl SamplingStrategy for BucketedSampling {
    fn generate_batches(
        &mut self,
        dataset_len: usize,
        _batch_size: usize, // 分桶采样忽略 batch_size
        bucket_keys: Option<&[usize]>,
    ) -> Vec<Vec<usize>> {
        let bucket_keys =
            bucket_keys.expect("BucketedSampling 需要 bucket_keys（数据集需实现 bucket_key 方法）");

        // 按分桶键分组
        let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
        for idx in 0..dataset_len {
            let key = bucket_keys[idx];
            buckets.entry(key).or_default().push(idx);
        }

        // 转换为有序列表（按 key 排序，确保输出顺序确定）
        let mut bucket_list: Vec<(usize, Vec<usize>)> = buckets.into_iter().collect();
        bucket_list.sort_by_key(|(key, _)| *key);

        // 可选：打乱每个桶内的样本顺序
        if self.shuffle {
            if let Some(seed) = self.seed {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                for (_, indices) in &mut bucket_list {
                    indices.shuffle(&mut rng);
                }
            } else {
                let mut rng = rand::thread_rng();
                for (_, indices) in &mut bucket_list {
                    indices.shuffle(&mut rng);
                }
            }
        }

        // 提取批次（每个桶一个批次）
        bucket_list
            .into_iter()
            .map(|(_, indices)| indices)
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════
// DataLoader
// ═══════════════════════════════════════════════════════════════

/// `DataLoader` - PyTorch 风格的数据批量加载器
///
/// 泛型参数：
/// - `D`: 数据集类型（实现 `Dataset` trait）
/// - `S`: 采样策略类型（实现 `SamplingStrategy` trait）
///
/// # 示例
///
/// ## 固定长度数据（默认 SequentialSampling）
/// ```ignore
/// let dataset = TensorDataset::new(train_x, train_y);
/// let loader = DataLoader::new(dataset, 32)
///     .shuffle(true)
///     .drop_last(true);
///
/// for (x_batch, y_batch) in loader.iter() {
///     model.forward(&x_batch)?;
/// }
/// ```
///
/// ## 变长序列（切换到 BucketedSampling）
/// ```ignore
/// let dataset = VarLenDataset::new(1, 2);
/// // ... 添加样本 ...
/// let loader = DataLoader::from_var_len(&dataset)
///     .shuffle(true);
///
/// for (x_batch, y_batch) in loader.iter() {
///     // 同一批次内序列长度相同
/// }
/// ```
pub struct DataLoader<D: Dataset, S: SamplingStrategy = SequentialSampling> {
    dataset: D,
    strategy: S,
    batch_size: usize,
}

// ----- DataLoader 通用方法 -----

impl<D: Dataset, S: SamplingStrategy> DataLoader<D, S> {
    /// 获取数据集大小
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// 获取批大小
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// 创建迭代器
    pub fn iter(&self) -> DataLoaderIterator<'_, D> {
        // 收集分桶键（如果数据集支持）
        let bucket_keys: Option<Vec<usize>> =
            if !self.dataset.is_empty() && self.dataset.bucket_key(0).is_some() {
                Some(
                    (0..self.dataset.len())
                        .map(|i| self.dataset.bucket_key(i).unwrap())
                        .collect(),
                )
            } else {
                None
            };

        // 生成批次索引
        let batches = self.strategy.clone().generate_batches(
            self.dataset.len(),
            self.batch_size,
            bucket_keys.as_deref(),
        );

        DataLoaderIterator {
            dataset: &self.dataset,
            batches,
            current_batch: 0,
        }
    }

    /// 获取批次数量
    pub fn num_batches(&self) -> usize {
        self.iter().len()
    }
}

// ----- DataLoader<D, SequentialSampling> 专用方法 -----

impl<D: Dataset> DataLoader<D, SequentialSampling> {
    /// 创建新的 DataLoader（默认使用 SequentialSampling）
    ///
    /// # 参数
    /// - `dataset`: 数据集
    /// - `batch_size`: 批大小
    pub fn new(dataset: D, batch_size: usize) -> Self {
        assert!(batch_size > 0, "DataLoader: batch_size 必须大于 0");
        Self {
            dataset,
            strategy: SequentialSampling::new(),
            batch_size,
        }
    }

    /// 设置是否打乱数据
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.strategy = self.strategy.shuffle(shuffle);
        self
    }

    /// 设置是否丢弃最后一个不完整的批次
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.strategy = self.strategy.drop_last(drop_last);
        self
    }

    /// 设置随机种子
    pub fn seed(mut self, seed: u64) -> Self {
        self.strategy = self.strategy.seed(seed);
        self
    }
}

// ----- DataLoader<&VarLenDataset, BucketedSampling> 专用方法 -----

impl<'a> DataLoader<&'a VarLenDataset, BucketedSampling> {
    /// 从变长数据集创建 DataLoader（使用 BucketedSampling）
    ///
    /// # 参数
    /// - `dataset`: 变长数据集引用
    pub fn from_var_len(dataset: &'a VarLenDataset) -> Self {
        Self {
            dataset,
            strategy: BucketedSampling::new(),
            batch_size: 0, // 分桶采样不使用 batch_size
        }
    }

    /// 设置是否打乱（桶内打乱）
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.strategy = self.strategy.shuffle(shuffle);
        self
    }

    /// 设置随机种子
    pub fn seed(mut self, seed: u64) -> Self {
        self.strategy = self.strategy.seed(seed);
        self
    }
}

// ═══════════════════════════════════════════════════════════════
// DataLoaderIterator
// ═══════════════════════════════════════════════════════════════

/// DataLoader 迭代器
pub struct DataLoaderIterator<'a, D: Dataset> {
    dataset: &'a D,
    batches: Vec<Vec<usize>>,
    current_batch: usize,
}

impl<D: Dataset> Iterator for DataLoaderIterator<'_, D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_batch >= self.batches.len() {
            return None;
        }

        let indices = &self.batches[self.current_batch];
        self.current_batch += 1;

        Some(self.dataset.get_batch(indices))
    }
}

impl<D: Dataset> ExactSizeIterator for DataLoaderIterator<'_, D> {
    fn len(&self) -> usize {
        self.batches.len() - self.current_batch
    }
}

// ═══════════════════════════════════════════════════════════════
// TensorDataset
// ═══════════════════════════════════════════════════════════════

/// `TensorDataset` - 持有特征和标签的数据集
///
/// 用于固定长度数据（MLP、CNN、固定长度 RNN 等）。
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

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.len
    }

    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        extract_tensor_batch(&self.features, &self.labels, indices)
    }
}

/// 从张量中按索引提取批次
fn extract_tensor_batch(features: &Tensor, labels: &Tensor, indices: &[usize]) -> (Tensor, Tensor) {
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

// ═══════════════════════════════════════════════════════════════
// VarLenDataset - 变长序列数据集
// ═══════════════════════════════════════════════════════════════

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
/// let loader = DataLoader::from_var_len(&dataset).shuffle(true);
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

    /// 获取桶数量（不同序列长度的数量）
    pub fn num_buckets(&self) -> usize {
        let mut lengths: Vec<usize> = self.samples.iter().map(|s| s.seq_len).collect();
        lengths.sort();
        lengths.dedup();
        lengths.len()
    }

    /// 获取样本数量
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// 检查数据集是否为空
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl Dataset for VarLenDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        let batch_size = indices.len();
        assert!(batch_size > 0, "批次不能为空");

        // 获取序列长度（同一批次内应该相同，由 BucketedSampling 保证）
        let seq_len = self.samples[indices[0]].seq_len;

        // 收集特征数据
        let mut feature_data = Vec::with_capacity(batch_size * seq_len * self.feature_size);
        for &idx in indices {
            debug_assert_eq!(
                self.samples[idx].seq_len, seq_len,
                "同一批次内的序列长度必须相同"
            );
            feature_data.extend(&self.samples[idx].features);
        }

        // 收集标签数据
        let mut label_data = Vec::with_capacity(batch_size * self.label_size);
        for &idx in indices {
            label_data.extend(&self.samples[idx].label);
        }

        let features = Tensor::new(&feature_data, &[batch_size, seq_len, self.feature_size]);
        let labels = Tensor::new(&label_data, &[batch_size, self.label_size]);

        (features, labels)
    }

    fn bucket_key(&self, index: usize) -> Option<usize> {
        Some(self.samples[index].seq_len)
    }
}

// 为 &VarLenDataset 实现 Dataset trait
impl Dataset for &VarLenDataset {
    fn len(&self) -> usize {
        (*self).len()
    }

    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        (*self).get_batch(indices)
    }

    fn bucket_key(&self, index: usize) -> Option<usize> {
        (*self).bucket_key(index)
    }
}





