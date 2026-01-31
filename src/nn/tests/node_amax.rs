/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Amax 节点单元测试
 *
 * 测试覆盖：
 * - 前向传播（不同轴、不同形状）
 * - 反向传播（梯度计算、并列最大值平分）
 * - 强化学习场景（DQN 选最优动作）
 *
 * 注意：框架要求张量至少 2D，所以 reduction 后至少保留 2D
 */

use crate::nn::graph::Graph;
use crate::nn::{Init, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试 ====================

/// 测试 Amax 前向传播 - 3D -> 2D，axis=0
#[test]
fn test_amax_forward_3d_axis0() {
    let graph = Graph::new();

    // [[[1, 2], [3, 4]],
    //  [[5, 6], [7, 8]]]
    let input = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        ))
        .unwrap();

    let max_id = graph
        .inner_mut()
        .new_amax_node(input.node_id(), 0, Some("amax"))
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    max_var.forward().unwrap();

    let result = max_var.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]); // axis=0 被移除
    // amax([[1,2],[3,4]], [[5,6],[7,8]], axis=0) = [[5,6],[7,8]]
    assert_eq!(result.data_as_slice(), &[5.0, 6.0, 7.0, 8.0]);
}

/// 测试 Amax 前向传播 - 3D -> 2D，axis=1
#[test]
fn test_amax_forward_3d_axis1() {
    let graph = Graph::new();

    // [[[1, 2], [3, 4]],
    //  [[5, 6], [7, 8]]]
    let input = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        ))
        .unwrap();

    // 沿 axis=1 求最大值
    let max_id = graph
        .inner_mut()
        .new_amax_node(input.node_id(), 1, None)
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    max_var.forward().unwrap();

    let result = max_var.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]); // axis=1 被移除
    // [[max(1,3), max(2,4)], [max(5,7), max(6,8)]] = [[3, 4], [7, 8]]
    assert_eq!(result.data_as_slice(), &[3.0, 4.0, 7.0, 8.0]);
}

/// 测试 Amax 前向传播 - 3D -> 2D，axis=2
#[test]
fn test_amax_forward_3d_axis2() {
    let graph = Graph::new();

    // [[[1, 2], [3, 4]],
    //  [[5, 6], [7, 8]]]
    let input = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        ))
        .unwrap();

    // 沿 axis=2 求最大值
    let max_id = graph
        .inner_mut()
        .new_amax_node(input.node_id(), 2, None)
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    max_var.forward().unwrap();

    let result = max_var.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]); // axis=2 被移除
    // [[max(1,2), max(3,4)], [max(5,6), max(7,8)]] = [[2, 4], [6, 8]]
    assert_eq!(result.data_as_slice(), &[2.0, 4.0, 6.0, 8.0]);
}

// ==================== 反向传播测试 ====================

/// 测试 Amax 反向传播 - 基本场景
#[test]
fn test_amax_backward_basic() {
    let graph = Graph::new();

    // 3D 输入，axis=2 -> 2D 输出
    let input = graph.parameter(&[2, 2, 3], Init::Zeros, "input").unwrap();
    // [[[1, 2, 3], [4, 5, 6]],
    //  [[7, 8, 9], [10, 11, 12]]]
    input
        .set_value(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[2, 2, 3],
        ))
        .unwrap();

    // axis=2: amax = [[3, 6], [9, 12]]
    let max_id = graph
        .inner_mut()
        .new_amax_node(input.node_id(), 2, None)
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    // MSE loss with target zeros [2, 2]
    let target = Tensor::zeros(&[2, 2]);
    let loss = max_var.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // amax = [[3, 6], [9, 12]], target = [[0, 0], [0, 0]]
    // d(mse)/d(amax) = 2*(amax - 0) / 4 = [[1.5, 3], [4.5, 6]]
    //
    // 梯度只流向最大值位置：
    //   [0,0,:] 最大值在位置 2 -> grad = 1.5
    //   [0,1,:] 最大值在位置 2 -> grad = 3
    //   [1,0,:] 最大值在位置 2 -> grad = 4.5
    //   [1,1,:] 最大值在位置 2 -> grad = 6

    let grad = input.grad().unwrap().unwrap();
    assert_eq!(grad.shape(), &[2, 2, 3]);
    // [0,0,:] = [0, 0, 1.5]
    assert_abs_diff_eq!(grad[[0, 0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 2]], 1.5, epsilon = 1e-5);
    // [0,1,:] = [0, 0, 3]
    assert_abs_diff_eq!(grad[[0, 1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 2]], 3.0, epsilon = 1e-5);
    // [1,0,:] = [0, 0, 4.5]
    assert_abs_diff_eq!(grad[[1, 0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 2]], 4.5, epsilon = 1e-5);
    // [1,1,:] = [0, 0, 6]
    assert_abs_diff_eq!(grad[[1, 1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 2]], 6.0, epsilon = 1e-5);
}

/// 测试 Amax 反向传播 - 并列最大值平分
#[test]
fn test_amax_backward_tie() {
    let graph = Graph::new();

    let input = graph.parameter(&[2, 2, 3], Init::Zeros, "input").unwrap();
    // [[[3, 3, 3], [1, 6, 6]],    <- 第一组全是 3，第二组有两个 6
    //  [[2, 2, 2], [5, 5, 1]]]    <- 第一组全是 2，第二组有两个 5
    input
        .set_value(&Tensor::new(
            &[3.0, 3.0, 3.0, 1.0, 6.0, 6.0, 2.0, 2.0, 2.0, 5.0, 5.0, 1.0],
            &[2, 2, 3],
        ))
        .unwrap();

    // axis=2: amax = [[3, 6], [2, 5]]
    let max_id = graph
        .inner_mut()
        .new_amax_node(input.node_id(), 2, None)
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    // MSE loss with target zeros
    let target = Tensor::zeros(&[2, 2]);
    let loss = max_var.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // d(mse)/d(amax) = 2*amax / 4 = [[1.5, 3], [1, 2.5]]
    //
    // [0,0,:] 有 3 个并列最大值 3，梯度平分：1.5 / 3 = 0.5
    // [0,1,:] 有 2 个并列最大值 6，梯度平分：3 / 2 = 1.5
    // [1,0,:] 有 3 个并列最大值 2，梯度平分：1 / 3 ≈ 0.333
    // [1,1,:] 有 2 个并列最大值 5，梯度平分：2.5 / 2 = 1.25

    let grad = input.grad().unwrap().unwrap();
    // [0,0,:] = [0.5, 0.5, 0.5]
    assert_abs_diff_eq!(grad[[0, 0, 0]], 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 1]], 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 2]], 0.5, epsilon = 1e-5);
    // [0,1,:] = [0, 1.5, 1.5]
    assert_abs_diff_eq!(grad[[0, 1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 1]], 1.5, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 2]], 1.5, epsilon = 1e-5);
    // [1,0,:] = [1/3, 1/3, 1/3]
    assert_abs_diff_eq!(grad[[1, 0, 0]], 1.0 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 1]], 1.0 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 2]], 1.0 / 3.0, epsilon = 1e-5);
    // [1,1,:] = [1.25, 1.25, 0]
    assert_abs_diff_eq!(grad[[1, 1, 0]], 1.25, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 1]], 1.25, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 2]], 0.0, epsilon = 1e-5);
}

/// 测试 Amax 反向传播 - axis=0
#[test]
fn test_amax_backward_axis0() {
    let graph = Graph::new();

    let input = graph.parameter(&[2, 2, 2], Init::Zeros, "input").unwrap();
    // [[[1, 5], [3, 2]],
    //  [[4, 2], [6, 8]]]
    input
        .set_value(&Tensor::new(&[1.0, 5.0, 3.0, 2.0, 4.0, 2.0, 6.0, 8.0], &[2, 2, 2]))
        .unwrap();

    // axis=0: amax = [[4, 5], [6, 8]]
    let max_id = graph
        .inner_mut()
        .new_amax_node(input.node_id(), 0, None)
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    // MSE loss with target zeros
    let target = Tensor::zeros(&[2, 2]);
    let loss = max_var.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // d(mse)/d(amax) = 2*amax / 4 = [[2, 2.5], [3, 4]]
    //
    // [0,0] 最大值在 [1,0,0]，grad = 2
    // [0,1] 最大值在 [0,0,1]，grad = 2.5
    // [1,0] 最大值在 [1,1,0]，grad = 3
    // [1,1] 最大值在 [1,1,1]，grad = 4

    let grad = input.grad().unwrap().unwrap();
    // 输入[0] = [[1,5],[3,2]]
    assert_abs_diff_eq!(grad[[0, 0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 1]], 2.5, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 1]], 0.0, epsilon = 1e-5);
    // 输入[1] = [[4,2],[6,8]]
    assert_abs_diff_eq!(grad[[1, 0, 0]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 0]], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 1]], 4.0, epsilon = 1e-5);
}

// ==================== 强化学习场景测试 ====================

/// 测试 DQN 风格：amax(Q_values, axis=2) 选最优动作的 Q 值
#[test]
fn test_amax_dqn_style() {
    let graph = Graph::new();

    // Q 值表：batch_size=2, seq_len=2, num_actions=4
    let q_values = graph
        .parameter(&[2, 2, 4], Init::Zeros, "q_values")
        .unwrap();
    // 每个时间步有 4 个动作的 Q 值
    q_values
        .set_value(&Tensor::new(
            &[
                // batch 0
                0.1, 0.5, 0.3, 0.2, // t=0: 最大值 0.5 在动作 1
                0.8, 0.2, 0.4, 0.6, // t=1: 最大值 0.8 在动作 0
                // batch 1
                0.3, 0.3, 0.9, 0.1, // t=0: 最大值 0.9 在动作 2
                0.4, 0.7, 0.7, 0.2, // t=1: 最大值 0.7 在动作 1 和 2（并列）
            ],
            &[2, 2, 4],
        ))
        .unwrap();

    // amax Q 值 = [[0.5, 0.8], [0.9, 0.7]]
    let max_q_id = graph
        .inner_mut()
        .new_amax_node(q_values.node_id(), 2, Some("max_q"))
        .unwrap();
    let max_q = graph.wrap_node_id(max_q_id);

    // DQN loss: (amax_Q - target_Q)^2
    let target_q = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let loss = max_q.mse_loss(&target_q).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // amax_Q = [[0.5, 0.8], [0.9, 0.7]], target = [[1, 1], [1, 1]]
    // d(mse)/d(amax_Q) = 2*(amax_Q - target) / 4 = [[-0.25, -0.1], [-0.05, -0.15]]

    let grad = q_values.grad().unwrap().unwrap();

    // batch 0, t=0: 动作 1 -> grad = -0.25
    assert_abs_diff_eq!(grad[[0, 0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 1]], -0.25, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 2]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 3]], 0.0, epsilon = 1e-5);

    // batch 0, t=1: 动作 0 -> grad = -0.1
    assert_abs_diff_eq!(grad[[0, 1, 0]], -0.1, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 2]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 3]], 0.0, epsilon = 1e-5);

    // batch 1, t=0: 动作 2 -> grad = -0.05
    assert_abs_diff_eq!(grad[[1, 0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 2]], -0.05, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 3]], 0.0, epsilon = 1e-5);

    // batch 1, t=1: 动作 1 和 2 并列 -> grad 平分 = -0.15 / 2 = -0.075
    assert_abs_diff_eq!(grad[[1, 1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 1]], -0.075, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 2]], -0.075, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 3]], 0.0, epsilon = 1e-5);
}

// ==================== 错误处理测试 ====================

/// 测试 axis 超出范围应报错
#[test]
fn test_amax_invalid_axis() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        ))
        .unwrap();

    // axis=3 超出 3D 张量的范围
    let result = graph.inner_mut().new_amax_node(input.node_id(), 3, None);

    assert!(result.is_err());
}
