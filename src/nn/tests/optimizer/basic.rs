/*
 * @Author       : 老董
 * @Date         : 2025-12-20
 * @Description  : GradientAccumulator 和 OptimizerState 基础组件测试
 */

use crate::nn::optimizer::{GradientAccumulator, OptimizerState};
use crate::nn::{Graph, NodeId};
use crate::tensor::Tensor;

// ============================================================================
// GradientAccumulator 测试
// ============================================================================

#[test]
fn test_gradient_accumulator_creation() {
    let accumulator = GradientAccumulator::new();
    assert_eq!(accumulator.sample_count(), 0);
}

#[test]
fn test_gradient_accumulator_single_accumulate() {
    let mut accumulator = GradientAccumulator::new();
    let node_id = NodeId(1);
    let gradient = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // 累积一个梯度
    accumulator.accumulate(node_id, &gradient).unwrap();
    accumulator.increment_sample_count();

    // 验证样本数
    assert_eq!(accumulator.sample_count(), 1);

    // 验证平均梯度（只有一个样本，平均值等于原值）
    let avg_gradient = accumulator.get_average_gradient(node_id).unwrap();
    assert_eq!(avg_gradient, gradient);
}

#[test]
fn test_gradient_accumulator_multiple_accumulate() {
    let mut accumulator = GradientAccumulator::new();
    let node_id = NodeId(1);

    // 累积多个梯度
    let gradient1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let gradient2 = Tensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);
    let gradient3 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);

    accumulator.accumulate(node_id, &gradient1).unwrap();
    accumulator.increment_sample_count();
    accumulator.accumulate(node_id, &gradient2).unwrap();
    accumulator.increment_sample_count();
    accumulator.accumulate(node_id, &gradient3).unwrap();
    accumulator.increment_sample_count();

    // 验证样本数
    assert_eq!(accumulator.sample_count(), 3);

    // 验证平均梯度：(1+3+2)/3=2, (2+4+3)/3=3, (3+5+4)/3=4, (4+6+5)/3=5
    let avg_gradient = accumulator.get_average_gradient(node_id).unwrap();
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(avg_gradient, expected);
}

#[test]
fn test_gradient_accumulator_multiple_nodes() {
    let mut accumulator = GradientAccumulator::new();
    let node_id_1 = NodeId(1);
    let node_id_2 = NodeId(2);

    let gradient1 = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let gradient2 = Tensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);

    accumulator.accumulate(node_id_1, &gradient1).unwrap();
    accumulator.accumulate(node_id_2, &gradient2).unwrap();
    accumulator.increment_sample_count();

    // 每个节点都有自己的累积梯度
    let avg1 = accumulator.get_average_gradient(node_id_1).unwrap();
    let avg2 = accumulator.get_average_gradient(node_id_2).unwrap();

    assert_eq!(avg1, gradient1);
    assert_eq!(avg2, gradient2);
}

#[test]
fn test_gradient_accumulator_clear() {
    let mut accumulator = GradientAccumulator::new();
    let node_id = NodeId(1);
    let gradient = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    accumulator.accumulate(node_id, &gradient).unwrap();
    accumulator.increment_sample_count();

    // 验证累积状态
    assert_eq!(accumulator.sample_count(), 1);
    assert!(accumulator.get_average_gradient(node_id).is_some());

    // 清除
    accumulator.clear();

    // 验证清除后的状态
    assert_eq!(accumulator.sample_count(), 0);
    assert!(accumulator.get_average_gradient(node_id).is_none());
}

#[test]
fn test_gradient_accumulator_get_nonexistent_node() {
    let accumulator = GradientAccumulator::new();
    let node_id = NodeId(999);

    // 不存在的节点应返回None
    assert!(accumulator.get_average_gradient(node_id).is_none());
}

#[test]
fn test_gradient_accumulator_zero_sample_count() {
    let mut accumulator = GradientAccumulator::new();
    let node_id = NodeId(1);
    let gradient = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // 只累积梯度，不增加样本数
    accumulator.accumulate(node_id, &gradient).unwrap();

    // 样本数为0时，应返回None（避免除以0）
    assert_eq!(accumulator.sample_count(), 0);
    assert!(accumulator.get_average_gradient(node_id).is_none());
}

// ============================================================================
// OptimizerState 测试
// ============================================================================

#[test]
fn test_optimizer_state_creation() {
    let mut graph = Graph::new();
    let _input = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let param1 = graph.new_parameter_node(&[2, 2], Some("param1")).unwrap();
    let param2 = graph.new_parameter_node(&[1, 1], Some("param2")).unwrap();

    let state = OptimizerState::new(&graph, 0.01).unwrap();

    // 验证可训练节点（只有Parameter节点）
    let trainable = state.trainable_nodes();
    assert_eq!(trainable.len(), 2);
    assert!(trainable.contains(&param1));
    assert!(trainable.contains(&param2));

    // 验证学习率
    assert_eq!(state.learning_rate(), 0.01);
}

#[test]
fn test_optimizer_state_learning_rate() {
    let graph = Graph::new();
    let mut state = OptimizerState::new(&graph, 0.01).unwrap();

    // 验证初始学习率
    assert_eq!(state.learning_rate(), 0.01);

    // 修改学习率
    state.set_learning_rate(0.001);
    assert_eq!(state.learning_rate(), 0.001);

    // 设置为0
    state.set_learning_rate(0.0);
    assert_eq!(state.learning_rate(), 0.0);
}

#[test]
fn test_optimizer_state_reset() {
    let mut graph = Graph::new();
    let _param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();
    let mut state = OptimizerState::new(&graph, 0.01).unwrap();

    // 手动向累积器添加一些数据
    let node_id = NodeId(1);
    let gradient = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    state
        .gradient_accumulator_mut()
        .accumulate(node_id, &gradient)
        .unwrap();
    state.gradient_accumulator_mut().increment_sample_count();

    // 验证累积状态
    assert_eq!(state.gradient_accumulator().sample_count(), 1);

    // 重置
    state.reset();

    // 验证重置后状态
    assert_eq!(state.gradient_accumulator().sample_count(), 0);
}
