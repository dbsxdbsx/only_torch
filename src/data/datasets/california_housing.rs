//! California Housing 房价回归数据集
//!
//! 来源：sklearn / StatLib
//! - 20,640 个样本
//! - 8 个特征：MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
//! - 1 个目标：MedHouseVal（房价中位数，单位：$100,000）
//!
//! 这是回归任务的经典数据集，类似于分类任务中的 MNIST。

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;

use crate::data::error::DataError;
use crate::tensor::Tensor;

use super::default_data_dir;

/// California Housing 数据集下载地址（来自 Hands-on ML 书籍 GitHub）
/// CSV 格式，包含 ocean_proximity 列需要跳过
const CALIFORNIA_HOUSING_URL: &str =
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv";

/// 特征数量
const NUM_FEATURES: usize = 8;

/// California Housing 房价数据集
///
/// # 特征说明
/// - `MedInc`: 街区收入中位数
/// - `HouseAge`: 房屋年龄中位数
/// - `AveRooms`: 平均房间数
/// - `AveBedrms`: 平均卧室数
/// - `Population`: 街区人口
/// - `AveOccup`: 平均入住人数
/// - `Latitude`: 纬度
/// - `Longitude`: 经度
///
/// # 目标
/// - `MedHouseVal`: 房价中位数（单位：$100,000）
#[derive(Debug, Clone)]
pub struct CaliforniaHousingDataset {
    /// 特征数据 [N, 8]
    features: Tensor,
    /// 目标数据 [N, 1]
    targets: Tensor,
    /// 样本数量
    len: usize,
    /// 是否已标准化
    is_standardized: bool,
    /// 特征均值（用于逆标准化）
    feature_means: Option<Vec<f32>>,
    /// 特征标准差（用于逆标准化）
    feature_stds: Option<Vec<f32>>,
    /// 目标均值
    target_mean: Option<f32>,
    /// 目标标准差
    target_std: Option<f32>,
}

impl CaliforniaHousingDataset {
    /// 完整加载 API
    ///
    /// # 参数
    /// - `root`: 数据目录，None 则使用默认目录
    /// - `download`: true=自动下载缺失文件
    ///
    /// # 返回
    /// 加载后的数据集（未标准化）
    pub fn load(root: Option<&str>, download: bool) -> Result<Self, DataError> {
        let data_dir = root
            .map(PathBuf::from)
            .unwrap_or_else(|| default_data_dir().join("california_housing"));

        let data_path = ensure_file(&data_dir, download)?;
        let (features, targets) = parse_csv(&data_path)?;

        let len = targets.shape()[0];

        Ok(Self {
            features,
            targets,
            len,
            is_standardized: false,
            feature_means: None,
            feature_stds: None,
            target_mean: None,
            target_std: None,
        })
    }

    /// 便捷 API：加载数据集（默认路径，自动下载）
    pub fn load_default() -> Result<Self, DataError> {
        Self::load(None, true)
    }

    /// 对特征和目标进行标准化（Z-score）
    ///
    /// 回归任务中标准化非常重要，可以加速收敛
    pub fn standardize(mut self) -> Self {
        if self.is_standardized {
            return self;
        }

        // 计算特征均值和标准差
        let (feature_means, feature_stds) = self.compute_feature_stats();
        let features_standardized = self.standardize_features(&feature_means, &feature_stds);

        // 计算目标均值和标准差
        let (target_mean, target_std) = self.compute_target_stats();
        let targets_standardized = self.standardize_targets(target_mean, target_std);

        self.features = features_standardized;
        self.targets = targets_standardized;
        self.is_standardized = true;
        self.feature_means = Some(feature_means);
        self.feature_stds = Some(feature_stds);
        self.target_mean = Some(target_mean);
        self.target_std = Some(target_std);

        self
    }

    /// 计算特征的均值和标准差
    fn compute_feature_stats(&self) -> (Vec<f32>, Vec<f32>) {
        let n = self.len as f32;
        let mut means = vec![0.0; NUM_FEATURES];
        let mut stds = vec![0.0; NUM_FEATURES];

        // 计算均值
        for i in 0..self.len {
            for j in 0..NUM_FEATURES {
                means[j] += self.features[[i, j]];
            }
        }
        for mean in &mut means {
            *mean /= n;
        }

        // 计算标准差
        for i in 0..self.len {
            for j in 0..NUM_FEATURES {
                let diff = self.features[[i, j]] - means[j];
                stds[j] += diff * diff;
            }
        }
        for std in &mut stds {
            *std = (*std / n).sqrt().max(1e-8); // 避免除零
        }

        (means, stds)
    }

    /// 标准化特征
    fn standardize_features(&self, means: &[f32], stds: &[f32]) -> Tensor {
        let mut data = Vec::with_capacity(self.len * NUM_FEATURES);
        for i in 0..self.len {
            for j in 0..NUM_FEATURES {
                data.push((self.features[[i, j]] - means[j]) / stds[j]);
            }
        }
        Tensor::new(&data, &[self.len, NUM_FEATURES])
    }

    /// 计算目标的均值和标准差
    fn compute_target_stats(&self) -> (f32, f32) {
        let n = self.len as f32;
        let mut mean = 0.0;
        let mut std = 0.0;

        for i in 0..self.len {
            mean += self.targets[[i, 0]];
        }
        mean /= n;

        for i in 0..self.len {
            let diff = self.targets[[i, 0]] - mean;
            std += diff * diff;
        }
        std = (std / n).sqrt().max(1e-8);

        (mean, std)
    }

    /// 标准化目标
    fn standardize_targets(&self, mean: f32, std: f32) -> Tensor {
        let mut data = Vec::with_capacity(self.len);
        for i in 0..self.len {
            data.push((self.targets[[i, 0]] - mean) / std);
        }
        Tensor::new(&data, &[self.len, 1])
    }

    /// 逆标准化预测值（用于评估时还原真实房价）
    pub fn inverse_transform_target(&self, prediction: f32) -> f32 {
        if let (Some(mean), Some(std)) = (self.target_mean, self.target_std) {
            prediction * std + mean
        } else {
            prediction
        }
    }

    /// 数据集样本数量
    pub fn len(&self) -> usize {
        self.len
    }

    /// 数据集是否为空
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 获取第 index 个样本
    ///
    /// # 返回
    /// (features, target) 元组
    /// - features: [8]
    /// - target: [1]
    pub fn get(&self, index: usize) -> Result<(Tensor, Tensor), DataError> {
        if index >= self.len {
            return Err(DataError::IndexOutOfBounds {
                index,
                len: self.len,
            });
        }

        let feature = self.features.slice(&[&index, &(..)]).flatten();
        let target = self.targets.slice(&[&index, &(..)]).flatten();

        Ok((feature, target))
    }

    /// 获取批量样本
    ///
    /// # 参数
    /// - `start`: 起始索引
    /// - `end`: 结束索引（不包含）
    ///
    /// # 返回
    /// (features, targets) 元组
    /// - features: [batch_size, 8]
    /// - targets: [batch_size, 1]
    pub fn get_batch(&self, start: usize, end: usize) -> Result<(Tensor, Tensor), DataError> {
        let end = end.min(self.len);
        if start >= end {
            return Err(DataError::IndexOutOfBounds {
                index: start,
                len: self.len,
            });
        }

        let batch_size = end - start;
        let features = self.features.slice(&[&(start..end), &(..)]);
        let targets = self.targets.slice(&[&(start..end), &(..)]);

        // 确保形状正确
        let features = features.reshape(&[batch_size, NUM_FEATURES]);
        let targets = targets.reshape(&[batch_size, 1]);

        Ok((features, targets))
    }

    /// 划分训练集和测试集
    ///
    /// # 参数
    /// - `test_ratio`: 测试集比例 (0.0 - 1.0)
    /// - `seed`: 随机种子（用于打乱数据）
    ///
    /// # 返回
    /// (train_dataset, test_dataset)
    pub fn train_test_split(
        self,
        test_ratio: f32,
        seed: Option<u64>,
    ) -> Result<(Self, Self), DataError> {
        if !(0.0..=1.0).contains(&test_ratio) {
            return Err(DataError::FormatError(
                "test_ratio 必须在 0.0 到 1.0 之间".to_string(),
            ));
        }

        let test_size = (self.len as f32 * test_ratio).round() as usize;
        let train_size = self.len - test_size;

        // 生成打乱的索引
        let mut indices: Vec<usize> = (0..self.len).collect();
        if let Some(s) = seed {
            // 简单的伪随机打乱（Fisher-Yates）
            let mut rng_state = s;
            for i in (1..indices.len()).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng_state as usize) % (i + 1);
                indices.swap(i, j);
            }
        }

        // 分割数据
        let train_indices = &indices[..train_size];
        let test_indices = &indices[train_size..];

        let train_features = self.gather_features(train_indices);
        let train_targets = self.gather_targets(train_indices);
        let test_features = self.gather_features(test_indices);
        let test_targets = self.gather_targets(test_indices);

        let train_dataset = Self {
            features: train_features,
            targets: train_targets,
            len: train_size,
            is_standardized: self.is_standardized,
            feature_means: self.feature_means.clone(),
            feature_stds: self.feature_stds.clone(),
            target_mean: self.target_mean,
            target_std: self.target_std,
        };

        let test_dataset = Self {
            features: test_features,
            targets: test_targets,
            len: test_size,
            is_standardized: self.is_standardized,
            feature_means: self.feature_means,
            feature_stds: self.feature_stds,
            target_mean: self.target_mean,
            target_std: self.target_std,
        };

        Ok((train_dataset, test_dataset))
    }

    /// 根据索引收集特征
    fn gather_features(&self, indices: &[usize]) -> Tensor {
        let mut data = Vec::with_capacity(indices.len() * NUM_FEATURES);
        for &idx in indices {
            for j in 0..NUM_FEATURES {
                data.push(self.features[[idx, j]]);
            }
        }
        Tensor::new(&data, &[indices.len(), NUM_FEATURES])
    }

    /// 根据索引收集目标
    fn gather_targets(&self, indices: &[usize]) -> Tensor {
        let mut data = Vec::with_capacity(indices.len());
        for &idx in indices {
            data.push(self.targets[[idx, 0]]);
        }
        Tensor::new(&data, &[indices.len(), 1])
    }

    /// 特征维度
    pub fn feature_dim(&self) -> usize {
        NUM_FEATURES
    }

    /// 获取所有特征
    pub fn features(&self) -> &Tensor {
        &self.features
    }

    /// 获取所有目标
    pub fn targets(&self) -> &Tensor {
        &self.targets
    }

    /// 是否已标准化
    pub fn is_standardized(&self) -> bool {
        self.is_standardized
    }
}

/// 确保数据文件存在
fn ensure_file(data_dir: &Path, download: bool) -> Result<PathBuf, DataError> {
    let csv_path = data_dir.join("housing.csv");
    if csv_path.exists() {
        return Ok(csv_path);
    }

    if download {
        std::fs::create_dir_all(data_dir).map_err(DataError::IoError)?;
        download_dataset(&csv_path)?;
        Ok(csv_path)
    } else {
        Err(DataError::FileNotFound(csv_path))
    }
}

/// 下载数据集
fn download_dataset(dest_path: &Path) -> Result<(), DataError> {
    println!("正在下载 California Housing 数据集...");
    println!("URL: {}", CALIFORNIA_HOUSING_URL);

    let response = ureq::get(CALIFORNIA_HOUSING_URL)
        .call()
        .map_err(|e| DataError::DownloadError(format!("HTTP 请求失败: {}", e)))?;

    if response.status() != 200 {
        return Err(DataError::DownloadError(format!(
            "HTTP 状态码: {}",
            response.status()
        )));
    }

    let mut content = String::new();
    response
        .into_reader()
        .read_to_string(&mut content)
        .map_err(|e| DataError::DownloadError(format!("读取响应失败: {}", e)))?;

    std::fs::write(dest_path, &content).map_err(DataError::IoError)?;

    println!("下载完成: {:?}", dest_path);
    Ok(())
}

/// 解析 CSV 文件
///
/// CSV 格式（来自 Hands-on ML）：
/// longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
/// population, households, median_income, median_house_value, ocean_proximity
///
/// 转换为 sklearn 格式特征：
/// MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
fn parse_csv(path: &Path) -> Result<(Tensor, Tensor), DataError> {
    let file = File::open(path).map_err(|_| DataError::FileNotFound(path.to_path_buf()))?;

    let reader: Box<dyn BufRead> = if path.extension().map_or(false, |ext| ext == "gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut features_data: Vec<f32> = Vec::new();
    let mut targets_data: Vec<f32> = Vec::new();
    let mut sample_count = 0;
    let mut is_header = true;

    for line in reader.lines() {
        let line = line.map_err(|e| DataError::FormatError(format!("读取行失败: {}", e)))?;
        let line = line.trim();

        // 跳过空行
        if line.is_empty() {
            continue;
        }

        // 跳过头部
        if is_header {
            is_header = false;
            continue;
        }

        // 解析数据行
        let values: Vec<&str> = line.split(',').collect();
        if values.len() < 10 {
            continue; // 跳过不完整的行
        }

        // 原始列：longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
        //         population, households, median_income, median_house_value, ocean_proximity
        let longitude: f32 = match values[0].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let latitude: f32 = match values[1].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let housing_median_age: f32 = match values[2].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let total_rooms: f32 = match values[3].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let total_bedrooms: f32 = match values[4].trim().parse() {
            Ok(v) => v,
            Err(_) => continue, // 有些行缺少此值
        };
        let population: f32 = match values[5].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let households: f32 = match values[6].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let median_income: f32 = match values[7].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let median_house_value: f32 = match values[8].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        // ocean_proximity (values[9]) 是类别，跳过

        // 防止除零
        if households < 1.0 {
            continue;
        }

        // 转换为 sklearn 格式特征
        // MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
        let med_inc = median_income;
        let house_age = housing_median_age;
        let ave_rooms = total_rooms / households;
        let ave_bedrms = total_bedrooms / households;
        let pop = population;
        let ave_occup = population / households;
        let lat = latitude;
        let lon = longitude;

        features_data.push(med_inc);
        features_data.push(house_age);
        features_data.push(ave_rooms);
        features_data.push(ave_bedrms);
        features_data.push(pop);
        features_data.push(ave_occup);
        features_data.push(lat);
        features_data.push(lon);

        // 目标值：房价中位数，转换为单位 $100,000
        targets_data.push(median_house_value / 100000.0);
        sample_count += 1;
    }

    if sample_count == 0 {
        return Err(DataError::FormatError("未能解析任何有效数据".to_string()));
    }

    let features = Tensor::new(&features_data, &[sample_count, NUM_FEATURES]);
    let targets = Tensor::new(&targets_data, &[sample_count, 1]);

    println!("加载了 {} 个样本", sample_count);

    Ok((features, targets))
}
