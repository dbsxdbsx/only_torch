//! ModelState（模型状态管理器）单元测试
//!
//! 测试智能缓存机制，支持不同形状输入的自动缓存复用。

use crate::nn::{CrossEntropyLoss, Graph, Linear, ModelState, Rnn, VarActivationOps};
use crate::tensor::Tensor;

#[test]
fn test_model_state_basic() {
    let graph = Graph::new_with_seed(42);
    let fc1 = Linear::new(&graph, 2, 4, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 4, 2, true, "fc2").unwrap();
    let state = ModelState::new(&graph);
    let criterion = CrossEntropyLoss::new();

    // 首次调用
    let x1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let target1 = Tensor::new(&[1.0, 0.0], &[1, 2]);
    let output1 = state
        .forward(&x1, |input| Ok(fc2.forward(&fc1.forward(input).tanh())))
        .unwrap();

    assert!(state.is_initialized());
    assert_eq!(state.cache_size(), 1);

    let loss1 = criterion.forward(&output1, &target1).unwrap();
    let val1 = loss1.backward().unwrap();
    assert!(val1 > 0.0);

    // 第二次调用（相同形状，复用）
    let x2 = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let output2 = state
        .forward(&x2, |input| Ok(fc2.forward(&fc1.forward(input).tanh())))
        .unwrap();

    // 验证是同一个节点（复用）
    assert_eq!(output1.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 1);
}

#[test]
fn test_model_state_multi_shape() {
    let graph = Graph::new_with_seed(42);
    let fc1 = Linear::new(&graph, 2, 4, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 4, 2, true, "fc2").unwrap();
    let state = ModelState::new(&graph);

    // 第一种形状 [1, 2]
    let x1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let output1 = state
        .forward(&x1, |input| Ok(fc2.forward(&fc1.forward(input).tanh())))
        .unwrap();

    // 第二种形状 [2, 2]（不同 batch_size）
    let x2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let output2 = state
        .forward(&x2, |input| Ok(fc2.forward(&fc1.forward(input).tanh())))
        .unwrap();

    // 应该是不同的节点
    assert_ne!(output1.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 2);

    // 再次使用第一种形状（复用）
    let x3 = Tensor::new(&[5.0, 6.0], &[1, 2]);
    let output3 = state
        .forward(&x3, |input| Ok(fc2.forward(&fc1.forward(input).tanh())))
        .unwrap();

    assert_eq!(output1.node_id(), output3.node_id());
    assert_eq!(state.cache_size(), 2); // 仍然只有 2 个缓存
}

#[test]
fn test_model_state_var_len_rnn() {
    let graph = Graph::new_with_seed(42);
    let rnn = Rnn::new(&graph, 1, 8, "rnn").unwrap();
    let fc = Linear::new(&graph, 8, 2, true, "fc").unwrap();
    let state = ModelState::new(&graph);

    // seq_len = 5
    let x1 = Tensor::new(&vec![0.1f32; 10], &[2, 5, 1]);
    let output1 = state
        .forward(&x1, |input| {
            let h = rnn.forward(input)?;
            Ok(fc.forward(&h))
        })
        .unwrap();

    // seq_len = 8（不同长度）
    let x2 = Tensor::new(&vec![0.1f32; 16], &[2, 8, 1]);
    let output2 = state
        .forward(&x2, |input| {
            let h = rnn.forward(input)?;
            Ok(fc.forward(&h))
        })
        .unwrap();

    // 应该是不同的节点
    assert_ne!(output1.node_id(), output2.node_id());
    assert_eq!(state.cache_size(), 2);

    // 检查缓存的形状
    let shapes = state.cached_shapes();
    assert!(shapes.contains(&vec![2, 5, 1]));
    assert!(shapes.contains(&vec![2, 8, 1]));
}

#[test]
fn test_model_state_clear_cache() {
    let graph = Graph::new_with_seed(42);
    let fc = Linear::new(&graph, 2, 2, true, "fc").unwrap();
    let state = ModelState::new(&graph);

    let x = Tensor::new(&[1.0, 2.0], &[1, 2]);
    state.forward(&x, |input| Ok(fc.forward(input))).unwrap();
    assert_eq!(state.cache_size(), 1);

    state.clear_cache();
    assert_eq!(state.cache_size(), 0);
    assert!(!state.is_initialized());
}
