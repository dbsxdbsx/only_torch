//! Criterion（损失函数封装）单元测试
//!
//! 测试 CrossEntropyLoss 和 MseLoss 的智能缓存机制。

use crate::nn::{CrossEntropyLoss, Graph, MseLoss};
use crate::tensor::Tensor;

// ==================== CrossEntropyLoss 测试 ====================

#[test]
fn test_cross_entropy_loss_basic() {
    let graph = Graph::new_with_seed(42);

    // 模拟 output: [2, 3]（2 个样本，3 个类别）
    let output = graph.randn(&[2, 3]).unwrap();

    let criterion = CrossEntropyLoss::new();

    // 第一次调用
    let target1 = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
    let loss1 = criterion.forward(&output, &target1).unwrap();
    let val1 = loss1.backward().unwrap();
    assert!(val1 > 0.0);

    // 第二次调用（复用节点）
    let target2 = Tensor::new(&[0.0, 1.0, 0.0, 1.0, 0.0, 0.0], &[2, 3]);
    let loss2 = criterion.forward(&output, &target2).unwrap();
    let val2 = loss2.backward().unwrap();
    assert!(val2 > 0.0);

    // 验证 loss 节点是同一个
    assert_eq!(loss1.node_id(), loss2.node_id());
    assert_eq!(criterion.cache_size(), 1);
}

#[test]
fn test_cross_entropy_multi_output() {
    let graph = Graph::new_with_seed(42);

    // 两个不同的 output 节点
    let output1 = graph.randn(&[2, 3]).unwrap();
    let output2 = graph.randn(&[4, 3]).unwrap();

    let criterion = CrossEntropyLoss::new();

    // 使用 output1
    let target1 = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
    let loss1 = criterion.forward(&output1, &target1).unwrap();

    // 使用 output2（不同节点，自动创建新缓存）
    #[rustfmt::skip]
    let target2 = Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        &[4, 3],
    );
    let loss2 = criterion.forward(&output2, &target2).unwrap();

    // 应该是不同的 loss 节点
    assert_ne!(loss1.node_id(), loss2.node_id());
    assert_eq!(criterion.cache_size(), 2);

    // 再次使用 output1（复用）
    let target3 = Tensor::new(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]);
    let loss3 = criterion.forward(&output1, &target3).unwrap();

    assert_eq!(loss1.node_id(), loss3.node_id());
    assert_eq!(criterion.cache_size(), 2); // 仍然只有 2 个缓存
}

#[test]
fn test_cross_entropy_clear_cache() {
    let graph = Graph::new_with_seed(42);
    let output = graph.randn(&[2, 3]).unwrap();
    let criterion = CrossEntropyLoss::new();

    let target = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
    criterion.forward(&output, &target).unwrap();
    assert_eq!(criterion.cache_size(), 1);

    criterion.clear_cache();
    assert_eq!(criterion.cache_size(), 0);
}

// ==================== MseLoss 测试 ====================

#[test]
fn test_mse_loss_basic() {
    let graph = Graph::new_with_seed(42);

    let output = graph.randn(&[3, 1]).unwrap();
    let criterion = MseLoss::new();

    // 第一次调用
    let target1 = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    let loss1 = criterion.forward(&output, &target1).unwrap();
    let val1 = loss1.backward().unwrap();
    assert!(val1 >= 0.0);

    // 第二次调用（复用节点）
    let target2 = Tensor::new(&[0.5, 1.5, 2.5], &[3, 1]);
    let loss2 = criterion.forward(&output, &target2).unwrap();
    let val2 = loss2.backward().unwrap();
    assert!(val2 >= 0.0);

    assert_eq!(loss1.node_id(), loss2.node_id());
    assert_eq!(criterion.cache_size(), 1);
}

#[test]
fn test_mse_loss_multi_output() {
    let graph = Graph::new_with_seed(42);

    let output1 = graph.randn(&[3, 1]).unwrap();
    let output2 = graph.randn(&[5, 1]).unwrap();

    let criterion = MseLoss::new();

    let target1 = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    let loss1 = criterion.forward(&output1, &target1).unwrap();

    let target2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5, 1]);
    let loss2 = criterion.forward(&output2, &target2).unwrap();

    assert_ne!(loss1.node_id(), loss2.node_id());
    assert_eq!(criterion.cache_size(), 2);
}

#[test]
fn test_mse_loss_clear_cache() {
    let graph = Graph::new_with_seed(42);
    let output = graph.randn(&[3, 1]).unwrap();
    let criterion = MseLoss::new();

    let target = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    criterion.forward(&output, &target).unwrap();
    assert_eq!(criterion.cache_size(), 1);

    criterion.clear_cache();
    assert_eq!(criterion.cache_size(), 0);
}
