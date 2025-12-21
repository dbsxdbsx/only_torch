//! transforms 模块单元测试

use crate::data::transforms::{flatten_images, normalize_pixels, one_hot};
use crate::tensor::Tensor;

#[test]
fn test_normalize_pixels_basic() {
    let tensor = Tensor::new(&[0.0, 127.5, 255.0, 51.0], &[2, 2]);
    let normalized = normalize_pixels(&tensor);

    assert!((normalized[[0, 0]] - 0.0).abs() < 1e-6);
    assert!((normalized[[0, 1]] - 0.5).abs() < 1e-6);
    assert!((normalized[[1, 0]] - 1.0).abs() < 1e-6);
    assert!((normalized[[1, 1]] - 0.2).abs() < 1e-6);
}

#[test]
fn test_one_hot_basic() {
    // 3 个样本，3 个类别
    let labels = Tensor::new(&[0.0, 2.0, 1.0], &[3]);
    let encoded = one_hot(&labels, 3);

    assert_eq!(encoded.shape(), &[3, 3]);

    // 类别 0 -> [1, 0, 0]
    assert_eq!(encoded[[0, 0]], 1.0);
    assert_eq!(encoded[[0, 1]], 0.0);
    assert_eq!(encoded[[0, 2]], 0.0);

    // 类别 2 -> [0, 0, 1]
    assert_eq!(encoded[[1, 0]], 0.0);
    assert_eq!(encoded[[1, 1]], 0.0);
    assert_eq!(encoded[[1, 2]], 1.0);

    // 类别 1 -> [0, 1, 0]
    assert_eq!(encoded[[2, 0]], 0.0);
    assert_eq!(encoded[[2, 1]], 1.0);
    assert_eq!(encoded[[2, 2]], 0.0);
}

#[test]
fn test_one_hot_mnist_style() {
    // MNIST: 10 个类别
    let labels = Tensor::new(&[0.0, 5.0, 9.0, 3.0], &[4]);
    let encoded = one_hot(&labels, 10);

    assert_eq!(encoded.shape(), &[4, 10]);

    // 验证每行只有一个 1
    for i in 0..4 {
        let row_sum: f32 = (0..10).map(|j| encoded[[i, j]]).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }

    // 验证正确位置
    assert_eq!(encoded[[0, 0]], 1.0); // 类别 0
    assert_eq!(encoded[[1, 5]], 1.0); // 类别 5
    assert_eq!(encoded[[2, 9]], 1.0); // 类别 9
    assert_eq!(encoded[[3, 3]], 1.0); // 类别 3
}

#[test]
fn test_flatten_images_4d() {
    // 模拟 MNIST: [N, C, H, W] = [2, 1, 28, 28]
    let tensor = Tensor::zeros(&[2, 1, 28, 28]);
    let flat = flatten_images(&tensor);

    assert_eq!(flat.shape(), &[2, 784]);
}

#[test]
fn test_flatten_images_3d() {
    // [N, H, W] = [3, 28, 28]
    let tensor = Tensor::zeros(&[3, 28, 28]);
    let flat = flatten_images(&tensor);

    assert_eq!(flat.shape(), &[3, 784]);
}

#[test]
fn test_flatten_preserves_data() {
    // 验证 flatten 不改变数据，只改变形状
    let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
    let tensor = Tensor::new(&data, &[2, 2, 2]);
    let flat = flatten_images(&tensor);

    assert_eq!(flat.shape(), &[2, 4]);
    // 验证数据顺序
    assert_eq!(flat[[0, 0]], 0.0);
    assert_eq!(flat[[0, 1]], 1.0);
    assert_eq!(flat[[0, 2]], 2.0);
    assert_eq!(flat[[0, 3]], 3.0);
    assert_eq!(flat[[1, 0]], 4.0);
    assert_eq!(flat[[1, 1]], 5.0);
    assert_eq!(flat[[1, 2]], 6.0);
    assert_eq!(flat[[1, 3]], 7.0);
}

