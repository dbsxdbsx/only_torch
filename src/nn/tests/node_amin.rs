/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Amin 节点单元测试
 *
 * 测试覆盖：
 * - 前向传播（不同轴、不同形状）
 * - 反向传播（梯度计算、并列最小值平分）
 * - 强化学习场景（Double DQN 选保守 Q 值）
 *
 * 注意：框架要求张量至少 2D，所以 reduction 后至少保留 2D
 */

use crate::nn::graph::Graph;
use crate::nn::{Init, Var, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试 ====================

/// 测试 Amin 前向传播 - 3D -> 2D，axis=0
#[test]
fn test_amin_forward_3d_axis0() {
    let graph = Graph::new();

    // [[[1, 2], [3, 4]],
    //  [[5, 6], [7, 8]]]
    let input = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        ))
        .unwrap();

    let min_node = graph
        .inner_mut()
        .create_amin_node(Rc::clone(input.node()), 0, Some("amin"))
        .unwrap();
    let min_var = Var::new_with_rc_graph(min_node, &graph.inner_rc());

    min_var.forward().unwrap();

    let result = min_var.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]); // axis=0 被移除
    // amin([[1,2],[3,4]], [[5,6],[7,8]], axis=0) = [[1,2],[3,4]]
    assert_eq!(result.data_as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

/// 测试 Amin 前向传播 - 3D -> 2D，axis=1
#[test]
fn test_amin_forward_3d_axis1() {
    let graph = Graph::new();

    // [[[1, 2], [3, 4]],
    //  [[5, 6], [7, 8]]]
    let input = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        ))
        .unwrap();

    // 沿 axis=1 求最小值
    let min_node = graph
        .inner_mut()
        .create_amin_node(Rc::clone(input.node()), 1, None)
        .unwrap();
    let min_var = Var::new_with_rc_graph(min_node, &graph.inner_rc());

    min_var.forward().unwrap();

    let result = min_var.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]); // axis=1 被移除
    // [[min(1,3), min(2,4)], [min(5,7), min(6,8)]] = [[1, 2], [5, 6]]
    assert_eq!(result.data_as_slice(), &[1.0, 2.0, 5.0, 6.0]);
}

/// 测试 Amin 前向传播 - 3D -> 2D，axis=2
#[test]
fn test_amin_forward_3d_axis2() {
    let graph = Graph::new();

    // [[[1, 2], [3, 4]],
    //  [[5, 6], [7, 8]]]
    let input = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        ))
        .unwrap();

    // 沿 axis=2 求最小值
    let min_node = graph
        .inner_mut()
        .create_amin_node(Rc::clone(input.node()), 2, None)
        .unwrap();
    let min_var = Var::new_with_rc_graph(min_node, &graph.inner_rc());

    min_var.forward().unwrap();

    let result = min_var.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]); // axis=2 被移除
    // [[min(1,2), min(3,4)], [min(5,6), min(7,8)]] = [[1, 3], [5, 7]]
    assert_eq!(result.data_as_slice(), &[1.0, 3.0, 5.0, 7.0]);
}

// ==================== 反向传播测试 ====================

/// 测试 Amin 反向传播 - 基本场景
#[test]
fn test_amin_backward_basic() {
    let graph = Graph::new();

    // 3D 输入，axis=2 -> 2D 输出
    let input = graph.parameter(&[2, 2, 3], Init::Zeros, "input").unwrap();
    // [[[3, 2, 1], [6, 5, 4]],
    //  [[9, 8, 7], [12, 11, 10]]]
    input
        .set_value(&Tensor::new(
            &[
                3.0, 2.0, 1.0, 6.0, 5.0, 4.0, 9.0, 8.0, 7.0, 12.0, 11.0, 10.0,
            ],
            &[2, 2, 3],
        ))
        .unwrap();

    // axis=2: amin = [[1, 4], [7, 10]]
    let min_node = graph
        .inner_mut()
        .create_amin_node(Rc::clone(input.node()), 2, None)
        .unwrap();
    let min_var = Var::new_with_rc_graph(min_node, &graph.inner_rc());

    // MSE loss with target zeros [2, 2]
    let target = Tensor::zeros(&[2, 2]);
    let loss = min_var.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // amin = [[1, 4], [7, 10]], target = [[0, 0], [0, 0]]
    // d(mse)/d(amin) = 2*(amin - 0) / 4 = [[0.5, 2], [3.5, 5]]
    //
    // 梯度只流向最小值位置：
    //   [0,0,:] 最小值在位置 2 -> grad = 0.5
    //   [0,1,:] 最小值在位置 2 -> grad = 2
    //   [1,0,:] 最小值在位置 2 -> grad = 3.5
    //   [1,1,:] 最小值在位置 2 -> grad = 5

    let grad = input.grad().unwrap().unwrap();
    assert_eq!(grad.shape(), &[2, 2, 3]);
    // [0,0,:] = [0, 0, 0.5]
    assert_abs_diff_eq!(grad[[0, 0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 2]], 0.5, epsilon = 1e-5);
    // [0,1,:] = [0, 0, 2]
    assert_abs_diff_eq!(grad[[0, 1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 2]], 2.0, epsilon = 1e-5);
    // [1,0,:] = [0, 0, 3.5]
    assert_abs_diff_eq!(grad[[1, 0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 2]], 3.5, epsilon = 1e-5);
    // [1,1,:] = [0, 0, 5]
    assert_abs_diff_eq!(grad[[1, 1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 2]], 5.0, epsilon = 1e-5);
}

/// 测试 Amin 反向传播 - 并列最小值平分
#[test]
fn test_amin_backward_tie() {
    let graph = Graph::new();

    let input = graph.parameter(&[2, 2, 3], Init::Zeros, "input").unwrap();
    // [[[1, 1, 1], [2, 0, 0]],    <- 第一组全是 1，第二组有两个 0
    //  [[3, 3, 3], [4, 2, 2]]]    <- 第一组全是 3，第二组有两个 2
    input
        .set_value(&Tensor::new(
            &[1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 3.0, 3.0, 3.0, 4.0, 2.0, 2.0],
            &[2, 2, 3],
        ))
        .unwrap();

    // axis=2: amin = [[1, 0], [3, 2]]
    let min_node = graph
        .inner_mut()
        .create_amin_node(Rc::clone(input.node()), 2, None)
        .unwrap();
    let min_var = Var::new_with_rc_graph(min_node, &graph.inner_rc());

    // MSE loss with target zeros
    let target = Tensor::zeros(&[2, 2]);
    let loss = min_var.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // d(mse)/d(amin) = 2*amin / 4 = [[0.5, 0], [1.5, 1]]
    //
    // [0,0,:] 有 3 个并列最小值 1，梯度平分：0.5 / 3 ≈ 0.167
    // [0,1,:] 有 2 个并列最小值 0，梯度平分：0 / 2 = 0
    // [1,0,:] 有 3 个并列最小值 3，梯度平分：1.5 / 3 = 0.5
    // [1,1,:] 有 2 个并列最小值 2，梯度平分：1 / 2 = 0.5

    let grad = input.grad().unwrap().unwrap();
    // [0,0,:] = [0.167, 0.167, 0.167]
    assert_abs_diff_eq!(grad[[0, 0, 0]], 0.5 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 1]], 0.5 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 2]], 0.5 / 3.0, epsilon = 1e-5);
    // [0,1,:] = [0, 0, 0]
    assert_abs_diff_eq!(grad[[0, 1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 2]], 0.0, epsilon = 1e-5);
    // [1,0,:] = [0.5, 0.5, 0.5]
    assert_abs_diff_eq!(grad[[1, 0, 0]], 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 1]], 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 2]], 0.5, epsilon = 1e-5);
    // [1,1,:] = [0, 0.5, 0.5]
    assert_abs_diff_eq!(grad[[1, 1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 1]], 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 2]], 0.5, epsilon = 1e-5);
}

/// 测试 Amin 反向传播 - axis=0
#[test]
fn test_amin_backward_axis0() {
    let graph = Graph::new();

    let input = graph.parameter(&[2, 2, 2], Init::Zeros, "input").unwrap();
    // [[[8, 3], [6, 7]],
    //  [[4, 5], [2, 1]]]
    input
        .set_value(&Tensor::new(
            &[8.0, 3.0, 6.0, 7.0, 4.0, 5.0, 2.0, 1.0],
            &[2, 2, 2],
        ))
        .unwrap();

    // axis=0: amin = [[4, 3], [2, 1]]
    let min_node = graph
        .inner_mut()
        .create_amin_node(Rc::clone(input.node()), 0, None)
        .unwrap();
    let min_var = Var::new_with_rc_graph(min_node, &graph.inner_rc());

    // MSE loss with target zeros
    let target = Tensor::zeros(&[2, 2]);
    let loss = min_var.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // d(mse)/d(amin) = 2*amin / 4 = [[2, 1.5], [1, 0.5]]
    //
    // [0,0] 最小值在 [1,0,0]，grad = 2
    // [0,1] 最小值在 [0,0,1]，grad = 1.5
    // [1,0] 最小值在 [1,1,0]，grad = 1
    // [1,1] 最小值在 [1,1,1]，grad = 0.5

    let grad = input.grad().unwrap().unwrap();
    // 输入[0] = [[8,3],[6,7]]
    assert_abs_diff_eq!(grad[[0, 0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 1]], 1.5, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 1]], 0.0, epsilon = 1e-5);
    // 输入[1] = [[4,5],[2,1]]
    assert_abs_diff_eq!(grad[[1, 0, 0]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 1]], 0.5, epsilon = 1e-5);
}

// ==================== 强化学习场景测试 ====================

/// 测试 Double DQN 风格：amin(Q_values, axis=2) 选保守 Q 值
#[test]
fn test_amin_double_dqn_style() {
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
                0.5, 0.1, 0.3, 0.2, // t=0: 最小值 0.1 在动作 1
                0.2, 0.8, 0.4, 0.6, // t=1: 最小值 0.2 在动作 0
                // batch 1
                0.3, 0.3, 0.1, 0.9, // t=0: 最小值 0.1 在动作 2
                0.4, 0.2, 0.2, 0.7, // t=1: 最小值 0.2 在动作 1 和 2（并列）
            ],
            &[2, 2, 4],
        ))
        .unwrap();

    // amin Q 值 = [[0.1, 0.2], [0.1, 0.2]]
    let min_q_node = graph
        .inner_mut()
        .create_amin_node(Rc::clone(q_values.node()), 2, Some("min_q"))
        .unwrap();
    let min_q = Var::new_with_rc_graph(min_q_node, &graph.inner_rc());

    // Double DQN loss: (amin_Q - target_Q)^2
    let target_q = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
    let loss = min_q.mse_loss(&target_q).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // amin_Q = [[0.1, 0.2], [0.1, 0.2]], target = [[0, 0], [0, 0]]
    // d(mse)/d(amin_Q) = 2*(amin_Q - target) / 4 = [[0.05, 0.1], [0.05, 0.1]]

    let grad = q_values.grad().unwrap().unwrap();

    // batch 0, t=0: 动作 1 -> grad = 0.05
    assert_abs_diff_eq!(grad[[0, 0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 1]], 0.05, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 2]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 0, 3]], 0.0, epsilon = 1e-5);

    // batch 0, t=1: 动作 0 -> grad = 0.1
    assert_abs_diff_eq!(grad[[0, 1, 0]], 0.1, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 2]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1, 3]], 0.0, epsilon = 1e-5);

    // batch 1, t=0: 动作 2 -> grad = 0.05
    assert_abs_diff_eq!(grad[[1, 0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 2]], 0.05, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0, 3]], 0.0, epsilon = 1e-5);

    // batch 1, t=1: 动作 1 和 2 并列 -> grad 平分 = 0.1 / 2 = 0.05
    assert_abs_diff_eq!(grad[[1, 1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 1]], 0.05, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 2]], 0.05, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1, 3]], 0.0, epsilon = 1e-5);
}

// ==================== 错误处理测试 ====================

/// 测试 axis 超出范围应报错
#[test]
fn test_amin_invalid_axis() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        ))
        .unwrap();

    // axis=3 超出 3D 张量的范围
    let result = graph
        .inner_mut()
        .create_amin_node(Rc::clone(input.node()), 3, None);

    assert!(result.is_err());
}

// ==================== 方案 C：新节点创建 API 测试 ====================

#[test]
fn test_create_amin_node_axis0() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let amin = inner
        .borrow_mut()
        .create_amin_node(input.clone(), 0, Some("amin"))
        .unwrap();

    // 沿 axis=0 取 min 后移除该轴，输出 [4]
    assert_eq!(amin.shape(), vec![4]);
    assert_eq!(amin.name(), Some("amin"));
}

#[test]
fn test_create_amin_node_axis1() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();

    let amin = inner
        .borrow_mut()
        .create_amin_node(input.clone(), 1, None)
        .unwrap();

    // 沿 axis=1 取 min 后，输出 [3]
    assert_eq!(amin.shape(), vec![3]);
}

#[test]
fn test_create_amin_node_invalid_axis() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();

    // axis=2 超出范围
    let result = inner.borrow_mut().create_amin_node(input, 2, None);

    assert!(result.is_err());
}

#[test]
fn test_create_amin_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_amin;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let amin = inner.borrow_mut().create_amin_node(input, 0, None).unwrap();
        weak_amin = Rc::downgrade(&amin);

        assert!(weak_amin.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_amin.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
