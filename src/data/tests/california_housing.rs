//! California Housing 数据集单元测试
//!
//! 注意：首次运行需要网络连接下载数据。
//! 数据下载后缓存在 `~/.cache/only_torch/datasets/california_housing/`

use approx::assert_abs_diff_eq;

use crate::data::CaliforniaHousingDataset;

/// 测试数据集加载
#[test]
fn test_california_housing_load() {
    let dataset = CaliforniaHousingDataset::load_default().expect("加载数据集失败");

    // California Housing 有 20640 个样本
    assert!(
        dataset.len() > 20000,
        "数据集应有 20000+ 样本，实际: {}",
        dataset.len()
    );
    assert_eq!(dataset.feature_dim(), 8);
    assert!(!dataset.is_empty());
    assert!(!dataset.is_standardized());
}

/// 测试获取单个样本
#[test]
fn test_california_housing_get_sample() {
    let dataset = CaliforniaHousingDataset::load_default().expect("加载数据集失败");

    // 获取第一个样本
    let (features, target) = dataset.get(0).expect("获取样本失败");

    assert_eq!(features.shape(), &[8], "特征应为 8 维");
    assert_eq!(target.shape(), &[1], "目标应为标量");

    // 特征值应该是有效数字
    for i in 0..8 {
        assert!(!features[[i]].is_nan(), "特征 {} 不应为 NaN", i);
        assert!(!features[[i]].is_infinite(), "特征 {} 不应为无穷", i);
    }
    assert!(!target[[0]].is_nan(), "目标不应为 NaN");
}

/// 测试获取多个样本
#[test]
fn test_california_housing_get_multiple_samples() {
    let dataset = CaliforniaHousingDataset::load_default().expect("加载数据集失败");

    // 随机获取几个样本
    for i in [0, 100, 1000, 10000] {
        let (features, target) = dataset.get(i).expect(&format!("获取样本 {} 失败", i));
        assert_eq!(features.shape(), &[8]);
        assert_eq!(target.shape(), &[1]);
    }
}

/// 测试批量获取
#[test]
fn test_california_housing_get_batch() {
    let dataset = CaliforniaHousingDataset::load_default().expect("加载数据集失败");

    // 获取前 32 个样本
    let (features, targets) = dataset.get_batch(0, 32).expect("获取批量失败");

    assert_eq!(features.shape(), &[32, 8], "批量特征形状错误");
    assert_eq!(targets.shape(), &[32, 1], "批量目标形状错误");
}

/// 测试批量获取边界
#[test]
fn test_california_housing_get_batch_boundary() {
    let dataset = CaliforniaHousingDataset::load_default().expect("加载数据集失败");
    let len = dataset.len();

    // 超出末尾的批量应该被截断
    let (features, targets) = dataset
        .get_batch(len - 10, len + 100)
        .expect("获取批量失败");
    assert_eq!(features.shape()[0], 10, "应只返回最后 10 个样本");
    assert_eq!(targets.shape()[0], 10);
}

/// 测试标准化
#[test]
fn test_california_housing_standardize() {
    let dataset = CaliforniaHousingDataset::load_default()
        .expect("加载数据集失败")
        .standardize();

    assert!(dataset.is_standardized(), "应该已标准化");

    // 标准化后均值应接近 0，标准差应接近 1
    let (features, _) = dataset.get(0).expect("获取样本失败");
    for i in 0..8 {
        // 单个样本无法验证统计量，但可以验证值在合理范围内
        assert!(
            features[[i]].abs() < 10.0,
            "标准化后特征 {} 值 {} 超出预期范围",
            i,
            features[[i]]
        );
    }
}

/// 测试标准化统计量
#[test]
fn test_california_housing_standardize_stats() {
    let dataset = CaliforniaHousingDataset::load_default()
        .expect("加载数据集失败")
        .standardize();

    // 计算所有特征的均值
    let features = dataset.features();
    let n = dataset.len() as f32;

    // 计算第一个特征的均值（应接近 0）
    let mut mean0 = 0.0;
    for i in 0..dataset.len() {
        mean0 += features[[i, 0]];
    }
    mean0 /= n;

    assert_abs_diff_eq!(mean0, 0.0, epsilon = 0.01);
}

/// 测试训练集/测试集划分
#[test]
fn test_california_housing_train_test_split() {
    let dataset = CaliforniaHousingDataset::load_default()
        .expect("加载数据集失败")
        .standardize();

    let total_len = dataset.len();
    let (train, test) = dataset.train_test_split(0.2, Some(42)).expect("划分失败");

    // 验证划分比例
    let expected_test_size = (total_len as f32 * 0.2).round() as usize;
    assert_eq!(test.len(), expected_test_size, "测试集大小错误");
    assert_eq!(
        train.len(),
        total_len - expected_test_size,
        "训练集大小错误"
    );

    // 总样本数应该不变
    assert_eq!(train.len() + test.len(), total_len);
}

/// 测试划分确定性
#[test]
fn test_california_housing_split_deterministic() {
    let dataset1 = CaliforniaHousingDataset::load_default()
        .unwrap()
        .standardize();
    let dataset2 = CaliforniaHousingDataset::load_default()
        .unwrap()
        .standardize();

    let (train1, _) = dataset1.train_test_split(0.2, Some(42)).unwrap();
    let (train2, _) = dataset2.train_test_split(0.2, Some(42)).unwrap();

    // 相同种子应得到相同的训练集第一个样本
    let (f1, t1) = train1.get(0).unwrap();
    let (f2, t2) = train2.get(0).unwrap();

    assert_abs_diff_eq!(f1[[0]], f2[[0]], epsilon = 1e-6);
    assert_abs_diff_eq!(t1[[0]], t2[[0]], epsilon = 1e-6);
}

/// 测试不同种子产生不同划分
#[test]
fn test_california_housing_split_different_seeds() {
    let dataset1 = CaliforniaHousingDataset::load_default()
        .unwrap()
        .standardize();
    let dataset2 = CaliforniaHousingDataset::load_default()
        .unwrap()
        .standardize();

    let (train1, _) = dataset1.train_test_split(0.2, Some(42)).unwrap();
    let (train2, _) = dataset2.train_test_split(0.2, Some(123)).unwrap();

    // 不同种子应得到不同的训练集（高概率）
    let (f1, _) = train1.get(0).unwrap();
    let (f2, _) = train2.get(0).unwrap();

    // 至少有一个特征应该不同
    let any_different = (0..8).any(|i| (f1[[i]] - f2[[i]]).abs() > 1e-6);
    assert!(any_different, "不同种子应产生不同划分");
}

/// 测试索引越界
#[test]
fn test_california_housing_index_out_of_bounds() {
    let dataset = CaliforniaHousingDataset::load_default().expect("加载数据集失败");
    let len = dataset.len();

    let result = dataset.get(len);
    assert!(result.is_err(), "越界访问应返回错误");

    let result = dataset.get(len + 100);
    assert!(result.is_err(), "越界访问应返回错误");
}

/// 测试逆标准化
#[test]
fn test_california_housing_inverse_transform() {
    let dataset = CaliforniaHousingDataset::load_default()
        .expect("加载数据集失败")
        .standardize();

    // 获取原始目标均值（通过逆变换 0）
    let original_mean = dataset.inverse_transform_target(0.0);

    // California Housing 房价中位数通常在 1-5 范围（单位：$100,000）
    assert!(
        original_mean > 0.5 && original_mean < 10.0,
        "逆变换后的均值 {} 不在预期范围",
        original_mean
    );
}

/// 测试无效的划分比例
#[test]
fn test_california_housing_invalid_split_ratio() {
    let dataset = CaliforniaHousingDataset::load_default().expect("加载数据集失败");

    // 负数比例
    let result = dataset.clone().train_test_split(-0.1, None);
    assert!(result.is_err(), "负数比例应返回错误");

    // 超过 1 的比例
    let result = dataset.train_test_split(1.5, None);
    assert!(result.is_err(), "超过 1 的比例应返回错误");
}

/// 测试特征和目标的访问器
#[test]
fn test_california_housing_accessors() {
    let dataset = CaliforniaHousingDataset::load_default().expect("加载数据集失败");

    let features = dataset.features();
    let targets = dataset.targets();

    assert_eq!(features.shape()[0], dataset.len());
    assert_eq!(features.shape()[1], 8);
    assert_eq!(targets.shape()[0], dataset.len());
    assert_eq!(targets.shape()[1], 1);
}
