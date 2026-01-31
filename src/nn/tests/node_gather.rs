//! Gather 节点测试
//!
//! 测试按索引张量从指定维度收集元素的功能，包括前向传播和反向传播。

use approx::assert_abs_diff_eq;

use crate::nn::{Graph, Init, VarShapeOps};
use crate::tensor::Tensor;

// ============================================================================
// 前向传播测试
// ============================================================================

/// 测试 Gather 基本功能（2D，dim=1）- SAC/DQN 核心场景
#[test]
fn test_gather_forward_2d_dim1() {
    let graph = Graph::new();

    // Q 值：[[1.0, 2.0, 3.0],
    //        [4.0, 5.0, 6.0]]
    let q_values = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    // 动作索引：[[1], [2]]
    let actions = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();

    // Gather
    let selected = q_values.gather(1, &actions).unwrap();

    // 前向传播
    selected.forward().unwrap();

    // 验证结果
    let result = selected.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 1]);
    assert_eq!(result[[0, 0]], 2.0); // q_values[0, 1]
    assert_eq!(result[[1, 0]], 6.0); // q_values[1, 2]
}

/// 测试 Gather 2D dim=0
#[test]
fn test_gather_forward_2d_dim0() {
    let graph = Graph::new();

    // input: [[1.0, 2.0],
    //         [3.0, 4.0],
    //         [5.0, 6.0]]
    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]))
        .unwrap();

    // index: [[0, 2],
    //         [1, 0]]
    let index = graph
        .input(&Tensor::new(&[0.0, 2.0, 1.0, 0.0], &[2, 2]))
        .unwrap();

    let result_var = input.gather(0, &index).unwrap();
    result_var.forward().unwrap();

    let result = result_var.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result[[0, 0]], 1.0); // input[0, 0]
    assert_eq!(result[[0, 1]], 6.0); // input[2, 1]
    assert_eq!(result[[1, 0]], 3.0); // input[1, 0]
    assert_eq!(result[[1, 1]], 2.0); // input[0, 1]
}

// ============================================================================
// 反向传播测试
// ============================================================================

/// 测试 Gather 反向传播基本功能
#[test]
fn test_gather_backward_basic() {
    use crate::nn::VarLossOps;

    let graph = Graph::new();

    // 输入：参数节点
    let input = graph.parameter(&[2, 3], Init::Zeros, "input").unwrap();
    input
        .set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    // 索引
    let index = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();

    // Gather
    let selected = input.gather(1, &index).unwrap();

    // 为了测试反向传播，需要一个标量 loss
    // 用 MSE loss 与全 1 目标，这样 d(mse)/d(selected) = 2*(selected - 1)/n
    let target = graph.input(&Tensor::new(&[1.0, 1.0], &[2, 1])).unwrap();
    let loss = selected.mse_loss(&target).unwrap();

    // 前向 + 反向传播
    loss.forward().unwrap();
    loss.backward().unwrap();

    // 验证输入的梯度
    // selected = [2.0, 6.0]
    // target = [1.0, 1.0]
    // d(mse)/d(selected) = 2*(selected - target) / n = [1.0, 5.0]
    // gather 的反向传播是 scatter：
    // - grad_input[0, 1] = 1.0 (从 selected[0])
    // - grad_input[1, 2] = 5.0 (从 selected[1])
    let grad_input = input.grad().unwrap().unwrap();
    assert_eq!(grad_input.shape(), &[2, 3]);
    assert_abs_diff_eq!(grad_input[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_input[[0, 1]], 1.0, epsilon = 1e-6);  // 2*(2-1)/2
    assert_abs_diff_eq!(grad_input[[0, 2]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_input[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_input[[1, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_input[[1, 2]], 5.0, epsilon = 1e-6);  // 2*(6-1)/2
}

/// 测试 Gather 反向传播 - 相同索引累加
#[test]
fn test_gather_backward_same_index_accumulates() {
    use crate::nn::VarLossOps;

    let graph = Graph::new();

    // 输入
    let input = graph.parameter(&[1, 3], Init::Zeros, "input").unwrap();
    input
        .set_value(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();

    // 索引：选择同一个位置两次
    let index = graph.input(&Tensor::new(&[1.0, 1.0], &[1, 2])).unwrap();

    // Gather
    let selected = input.gather(1, &index).unwrap();

    // 前向传播
    selected.forward().unwrap();

    // 验证前向结果
    let result = selected.value().unwrap().unwrap();
    assert_eq!(result[[0, 0]], 2.0);
    assert_eq!(result[[0, 1]], 2.0);

    // 为了测试反向传播，需要一个标量 loss
    let target = graph.input(&Tensor::new(&[0.0, 0.0], &[1, 2])).unwrap();
    let loss = selected.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // 验证梯度累加
    // selected = [2.0, 2.0], target = [0.0, 0.0]
    // d(mse)/d(selected) = 2*(selected - target) / n = [2.0, 2.0]
    // gather 反向传播 scatter：
    // - grad_input[0, 1] += 2.0 (从 selected[0, 0])
    // - grad_input[0, 1] += 2.0 (从 selected[0, 1])
    // 所以 grad_input[0, 1] = 4.0
    let grad_input = input.grad().unwrap().unwrap();
    assert_abs_diff_eq!(grad_input[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_input[[0, 1]], 4.0, epsilon = 1e-6); // 累加
    assert_abs_diff_eq!(grad_input[[0, 2]], 0.0, epsilon = 1e-6);
}

/// 测试 Gather 与 MSE Loss 结合（SAC Critic 更新场景）
#[test]
fn test_gather_with_mse_loss() {
    use crate::nn::VarLossOps;

    let graph = Graph::new();

    // 模拟 Q 网络输出：[batch=2, action_dim=3]
    let q_values = graph.parameter(&[2, 3], Init::Zeros, "q_values").unwrap();
    q_values
        .set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    // 动作索引：[batch=2, 1]
    let actions = graph.input(&Tensor::new(&[1.0, 0.0], &[2, 1])).unwrap();

    // 按动作选择 Q 值
    let q_selected = q_values.gather(1, &actions).unwrap();

    // 目标 Q 值
    let target_q = graph.input(&Tensor::new(&[2.5, 3.5], &[2, 1])).unwrap();

    // MSE Loss
    let loss = q_selected.mse_loss(&target_q).unwrap();

    // 前向传播
    loss.forward().unwrap();

    // 验证 loss 值
    // q_selected = [q[0,1], q[1,0]] = [2.0, 4.0]
    // target = [2.5, 3.5]
    // mse = ((2.0-2.5)² + (4.0-3.5)²) / 2 = (0.25 + 0.25) / 2 = 0.25
    let loss_value = loss.value().unwrap().unwrap();
    assert_abs_diff_eq!(loss_value.get_data_number().unwrap(), 0.25, epsilon = 1e-6);

    // 反向传播
    loss.backward().unwrap();

    // 验证 Q 值梯度
    let grad_q = q_values.grad().unwrap().unwrap();
    assert_eq!(grad_q.shape(), &[2, 3]);

    // 对于 MSE，d(loss)/d(q_selected) = 2*(q_selected - target) / n
    // d(loss)/d(q[0,1]) = 2*(2.0 - 2.5) / 2 = -0.5
    // d(loss)/d(q[1,0]) = 2*(4.0 - 3.5) / 2 = 0.5
    assert_abs_diff_eq!(grad_q[[0, 1]], -0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_q[[1, 0]], 0.5, epsilon = 1e-6);

    // 其他位置梯度为 0
    assert_abs_diff_eq!(grad_q[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_q[[0, 2]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_q[[1, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_q[[1, 2]], 0.0, epsilon = 1e-6);
}

// ============================================================================
// 错误处理测试
// ============================================================================

// ============================================================================
// GatherIndex trait 测试（支持 &Tensor 作为 index）
// ============================================================================

/// 测试 gather 直接接受 &Tensor 作为 index
#[test]
fn test_gather_with_tensor_index() {
    let graph = Graph::new();

    // Q 值：[[1.0, 2.0, 3.0],
    //        [4.0, 5.0, 6.0]]
    let q_values = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    // 动作索引：直接使用 Tensor（不需要 graph.input 转换）
    let actions = Tensor::new(&[1.0, 2.0], &[2, 1]);

    // Gather 直接接受 &Tensor
    let selected = q_values.gather(1, &actions).unwrap();

    // 前向传播
    selected.forward().unwrap();

    // 验证结果
    let result = selected.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 1]);
    assert_eq!(result[[0, 0]], 2.0); // q_values[0, 1]
    assert_eq!(result[[1, 0]], 6.0); // q_values[1, 2]
}

/// 测试 gather 接受 Tensor（非引用）作为 index
#[test]
fn test_gather_with_owned_tensor_index() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();

    // 直接传入 owned Tensor
    let selected = input.gather(1, Tensor::new(&[0.0, 1.0], &[2, 1])).unwrap();

    selected.forward().unwrap();

    let result = selected.value().unwrap().unwrap();
    assert_eq!(result[[0, 0]], 1.0); // input[0, 0]
    assert_eq!(result[[1, 0]], 4.0); // input[1, 1]
}

/// 测试 gather 使用 Tensor index 时的反向传播
#[test]
fn test_gather_tensor_index_backward() {
    use crate::nn::VarLossOps;

    let graph = Graph::new();

    // 输入：参数节点
    let q_values = graph.parameter(&[2, 3], Init::Zeros, "q").unwrap();
    q_values
        .set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    // 动作索引：直接使用 Tensor
    let actions = Tensor::new(&[1.0, 0.0], &[2, 1]);

    // Gather
    let q_selected = q_values.gather(1, &actions).unwrap();

    // MSE Loss
    let target = Tensor::new(&[2.5, 3.5], &[2, 1]);
    let loss = q_selected.mse_loss(&target).unwrap();

    // 前向 + 反向
    loss.forward().unwrap();
    loss.backward().unwrap();

    // 验证梯度
    let grad_q = q_values.grad().unwrap().unwrap();
    assert_abs_diff_eq!(grad_q[[0, 1]], -0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_q[[1, 0]], 0.5, epsilon = 1e-6);
}

/// 测试 RL 场景：完整的 Critic 更新流程（使用 Tensor index）
#[test]
fn test_gather_rl_critic_update_with_tensor_index() {
    use crate::nn::{Adam, Module, Optimizer, VarLossOps};
    use crate::nn::layer::Linear;

    let graph = Graph::new_with_seed(42);

    // 简单的 Q 网络：Linear(4 -> 2)
    let fc = Linear::new(&graph, 4, 2, true, "q_net").unwrap();

    // 优化器
    let mut optimizer = Adam::new(&graph, &fc.parameters(), 0.01);

    // 模拟一个 batch
    let obs = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 4]);
    let actions = Tensor::new(&[0.0, 1.0], &[2, 1]);  // 直接使用 Tensor
    let target_q = Tensor::new(&[1.0, -1.0], &[2, 1]);

    // 前向传播
    let obs_var = graph.input(&obs).unwrap();
    let q_values = fc.forward(&obs_var);
    let q_selected = q_values.gather(1, &actions).unwrap();  // 直接传 &Tensor
    let loss = q_selected.mse_loss(&target_q).unwrap();

    // 反向传播和更新
    optimizer.zero_grad().unwrap();
    loss.backward().unwrap();
    optimizer.step().unwrap();

    // 验证参数有更新（梯度不为零）
    for param in fc.parameters() {
        let grad = param.grad().unwrap();
        assert!(grad.is_some(), "参数应该有梯度");
    }
}

// ============================================================================
// 错误处理测试
// ============================================================================

/// 测试 dim 超出范围
#[test]
fn test_gather_dim_out_of_range() {
    let graph = Graph::new();
    let input = graph.input(&Tensor::ones(&[2, 3])).unwrap();
    let index = graph.input(&Tensor::zeros(&[2, 1])).unwrap();

    let result = input.gather(2, &index);
    assert!(result.is_err());
}

/// 测试 index 维度不匹配（3D vs 2D）
#[test]
fn test_gather_index_dim_mismatch() {
    let graph = Graph::new();
    let input = graph.input(&Tensor::ones(&[2, 3])).unwrap();
    // 3D 索引与 2D 输入不匹配
    let index = graph.input(&Tensor::zeros(&[2, 1, 1])).unwrap();

    let result = input.gather(1, &index);
    assert!(result.is_err());
}

/// 测试 index 形状不匹配（非 gather 维度）
#[test]
fn test_gather_shape_mismatch() {
    let graph = Graph::new();
    let input = graph.input(&Tensor::ones(&[2, 3])).unwrap();
    let index = graph.input(&Tensor::zeros(&[3, 1])).unwrap(); // batch 大小不匹配

    let result = input.gather(1, &index);
    assert!(result.is_err());
}
