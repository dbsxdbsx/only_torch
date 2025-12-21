//! MNIST 数据集单元测试
//!
//! 注意：首次运行需要网络连接下载数据（约 11MB）。
//! 数据下载后缓存在 `~/.cache/only_torch/datasets/mnist/`，后续无需网络。

use approx::assert_abs_diff_eq;

use crate::data::datasets::default_data_dir;
use crate::data::MnistDataset;

#[test]
fn test_default_data_dir() {
    let dir = default_data_dir();
    assert!(dir.to_string_lossy().contains("only_torch"));
    assert!(dir.to_string_lossy().contains("datasets"));
}

#[test]
fn test_mnist_train_load() {
    let dataset = MnistDataset::train().expect("加载 MNIST 训练集失败");

    assert_eq!(dataset.len(), 60000);
    assert_eq!(dataset.input_shape(), vec![1, 28, 28]);
    assert_eq!(dataset.label_shape(), vec![10]);
    assert!(!dataset.is_empty());
}

#[test]
fn test_mnist_test_load() {
    let dataset = MnistDataset::test().expect("加载 MNIST 测试集失败");

    assert_eq!(dataset.len(), 10000);
    assert_eq!(dataset.input_shape(), vec![1, 28, 28]);
    assert_eq!(dataset.label_shape(), vec![10]);
}

#[test]
fn test_mnist_get_sample() {
    let dataset = MnistDataset::train().expect("加载 MNIST 训练集失败");

    // 获取第一个样本
    let (image, label) = dataset.get(0).expect("获取样本失败");

    assert_eq!(image.shape(), &[1, 28, 28]);
    assert_eq!(label.shape(), &[10]);

    // one-hot 标签应该只有一个 1
    let label_sum: f32 = (0..10).map(|i| label[[i]]).sum();
    assert_abs_diff_eq!(label_sum, 1.0, epsilon = 1e-6);
}

#[test]
fn test_mnist_get_multiple_samples() {
    let dataset = MnistDataset::train().expect("加载 MNIST 训练集失败");

    // 获取多个样本
    for i in [0, 100, 1000, 59999] {
        let (image, label) = dataset.get(i).expect(&format!("获取样本 {} 失败", i));
        assert_eq!(image.shape(), &[1, 28, 28]);
        assert_eq!(label.shape(), &[10]);
    }
}

#[test]
fn test_mnist_flatten() {
    let dataset = MnistDataset::train()
        .expect("加载 MNIST 训练集失败")
        .flatten();

    assert_eq!(dataset.input_shape(), vec![784]);

    let (image, label) = dataset.get(0).expect("获取样本失败");
    assert_eq!(image.shape(), &[784]);
    assert_eq!(label.shape(), &[10]);
}

#[test]
fn test_mnist_pixel_normalization() {
    let dataset = MnistDataset::train().expect("加载 MNIST 训练集失败");
    let images = dataset.images();

    // 像素值应该在 [0, 1] 范围内（已归一化）
    let min = images.min_value();
    let max = images.max_value();

    assert!(min >= 0.0, "像素最小值 {} < 0", min);
    assert!(max <= 1.0, "像素最大值 {} > 1", max);
}

#[test]
fn test_mnist_labels_valid() {
    let dataset = MnistDataset::train().expect("加载 MNIST 训练集失败");

    // 检查前 100 个样本的标签
    for i in 0..100 {
        let (_, label) = dataset.get(i).expect(&format!("获取样本 {} 失败", i));

        // 找到 one-hot 编码中为 1 的位置
        let mut class = None;
        for j in 0..10 {
            if label[[j]] > 0.5 {
                assert!(class.is_none(), "one-hot 编码有多个 1");
                class = Some(j);
            }
        }
        assert!(class.is_some(), "one-hot 编码没有 1");
    }
}

#[test]
fn test_mnist_index_out_of_bounds() {
    let dataset = MnistDataset::train().expect("加载 MNIST 训练集失败");

    // 尝试访问越界索引
    let result = dataset.get(60000);
    assert!(result.is_err());
}

#[test]
fn test_mnist_custom_path_not_exist() {
    // 使用不存在的自定义路径，不下载
    let result = MnistDataset::load(Some("./nonexistent_path/mnist"), true, false);

    // 应该返回 FileNotFound 错误
    assert!(result.is_err());
}
