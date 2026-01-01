/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : MSELoss 节点单元测试
 */

use crate::nn::{Graph, Reduction};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ========== 基本功能测试 ==========

#[test]
fn test_mse_loss_creation() {
    let mut graph = Graph::new();

    let input_id = graph.new_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    // 验证节点存在且预期形状正确
    assert_eq!(
        graph.get_node_value_expected_shape(loss_id).unwrap(),
        &[1, 1]
    );
}

#[test]
fn test_mse_loss_shape_mismatch() {
    let mut graph = Graph::new();

    let input_id = graph.new_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[1, 4], Some("target")).unwrap(); // 形状不匹配

    let result = graph.new_mse_loss_node(input_id, target_id, None);
    assert!(result.is_err());
}

// ========== Mean Reduction 测试 ==========

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// # loss = 0.25
/// # grad = [-0.333..., -0.333..., -0.333...]
/// ```
#[test]
fn test_mse_loss_forward_mean_basic() {
    let mut graph = Graph::new();

    let input_id = graph.new_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    // 设置输入值
    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3])))
        .unwrap();

    // 前向传播
    graph.forward_node(loss_id).unwrap();

    // 验证损失值: ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.75 / 3 = 0.25
    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    assert_abs_diff_eq!(loss[[0, 0]], 0.25, epsilon = 1e-6);
}

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// # loss = 0.25
/// ```
#[test]
fn test_mse_loss_forward_2d_matrix() {
    let mut graph = Graph::new();

    let input_id = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[2, 2], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(
            input_id,
            Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])),
        )
        .unwrap();
    graph
        .set_node_value(
            target_id,
            Some(&Tensor::new(&[1.5, 2.5, 3.5, 4.5], &[2, 2])),
        )
        .unwrap();

    graph.forward_node(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 4 个元素，每个 diff = 0.5, squared = 0.25
    // sum = 1.0, mean = 0.25
    assert_abs_diff_eq!(loss[[0, 0]], 0.25, epsilon = 1e-6);
}

// ========== Sum Reduction 测试 ==========

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.MSELoss(reduction='sum')(input, target)
/// # loss = 0.75
/// # grad = [-1.0, -1.0, -1.0]
/// ```
#[test]
fn test_mse_loss_forward_sum() {
    let mut graph = Graph::new();

    let input_id = graph.new_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node_with_reduction(input_id, target_id, Reduction::Sum, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3])))
        .unwrap();

    graph.forward_node(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 3 * 0.25 = 0.75
    assert_abs_diff_eq!(loss[[0, 0]], 0.75, epsilon = 1e-6);
}

// ========== Jacobi 测试（单样本模式）==========

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = 2 * (input - target) / N = 2 * (-0.5) / 3 = -0.333...
/// ```
#[test]
fn test_mse_loss_backward_jacobi_mean() {
    let mut graph = Graph::new();

    // 使用 Parameter 作为 input，这样可以计算梯度
    let input_id = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3])))
        .unwrap();

    graph.forward_node(loss_id).unwrap();
    graph.backward_nodes(&[input_id], loss_id).unwrap();

    // 获取雅可比矩阵（单样本模式）
    let jacobi = graph.get_node_jacobi(input_id).unwrap().unwrap();

    // 预期梯度: 2 * (input - target) / N = 2 * [-0.5, -0.5, -0.5] / 3
    let expected_grad = Tensor::new(
        &[-0.333_333_34, -0.333_333_34, -0.333_333_34],
        &[1, 3],
    );

    assert_abs_diff_eq!(jacobi, &expected_grad, epsilon = 1e-5);
}

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.MSELoss(reduction='sum')(input, target)
/// loss.backward()
/// # grad = 2 * (input - target) = 2 * (-0.5) = -1.0
/// ```
#[test]
fn test_mse_loss_backward_jacobi_sum() {
    let mut graph = Graph::new();

    // 使用 Parameter 作为 input，这样可以计算梯度
    let input_id = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node_with_reduction(input_id, target_id, Reduction::Sum, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3])))
        .unwrap();

    graph.forward_node(loss_id).unwrap();
    graph.backward_nodes(&[input_id], loss_id).unwrap();

    let jacobi = graph.get_node_jacobi(input_id).unwrap().unwrap();
    let expected_grad = Tensor::new(&[-1.0, -1.0, -1.0], &[1, 3]);

    assert_abs_diff_eq!(jacobi, &expected_grad, epsilon = 1e-6);
}

/// 测试 2D 矩阵的 Jacobi
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = 2 * (-0.5) / 4 = -0.25
/// ```
#[test]
fn test_mse_loss_backward_jacobi_2d() {
    let mut graph = Graph::new();

    // 使用 Parameter 作为 input，这样可以计算梯度
    let input_id = graph.new_parameter_node(&[2, 2], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[2, 2], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(
            input_id,
            Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])),
        )
        .unwrap();
    graph
        .set_node_value(
            target_id,
            Some(&Tensor::new(&[1.5, 2.5, 3.5, 4.5], &[2, 2])),
        )
        .unwrap();

    graph.forward_node(loss_id).unwrap();
    graph.backward_nodes(&[input_id], loss_id).unwrap();

    // 使用 get_node_grad 获取正确形状的梯度
    let grad = graph.get_node_grad(input_id).unwrap().unwrap();
    let expected_grad = Tensor::new(&[-0.25, -0.25, -0.25, -0.25], &[2, 2]);

    assert_abs_diff_eq!(grad, expected_grad, epsilon = 1e-6);
}

// ========== Batch 模式测试 ==========

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([
///     [1.0, 2.0, 3.0, 4.0],
///     [0.5, 1.5, 2.5, 3.5],
///     [2.0, 3.0, 4.0, 5.0]
/// ], requires_grad=True)
/// target = torch.tensor([
///     [1.2, 2.1, 2.9, 4.1],
///     [0.6, 1.4, 2.6, 3.4],
///     [1.9, 3.1, 4.0, 5.2]
/// ])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// # loss ≈ 0.01417
/// ```
#[test]
fn test_mse_loss_batch_forward() {
    let mut graph = Graph::new();

    let input_id = graph.new_input_node(&[3, 4], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[3, 4], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    #[rustfmt::skip]
    let input_data = Tensor::new(&[
        1.0, 2.0, 3.0, 4.0,
        0.5, 1.5, 2.5, 3.5,
        2.0, 3.0, 4.0, 5.0,
    ], &[3, 4]);

    #[rustfmt::skip]
    let target_data = Tensor::new(&[
        1.2, 2.1, 2.9, 4.1,
        0.6, 1.4, 2.6, 3.4,
        1.9, 3.1, 4.0, 5.2,
    ], &[3, 4]);

    graph.set_node_value(input_id, Some(&input_data)).unwrap();
    graph.set_node_value(target_id, Some(&target_data)).unwrap();

    graph.forward_batch(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // PyTorch: 0.014166653156280518
    assert_abs_diff_eq!(loss[[0, 0]], 0.014_166_653, epsilon = 1e-6);
}

/// Batch 模式的反向传播测试
#[test]
fn test_mse_loss_batch_backward() {
    let mut graph = Graph::new();

    // 使用 Parameter 作为 input，这样可以计算梯度
    let input_id = graph.new_parameter_node(&[3, 4], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[3, 4], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    #[rustfmt::skip]
    let input_data = Tensor::new(&[
        1.0, 2.0, 3.0, 4.0,
        0.5, 1.5, 2.5, 3.5,
        2.0, 3.0, 4.0, 5.0,
    ], &[3, 4]);

    #[rustfmt::skip]
    let target_data = Tensor::new(&[
        1.2, 2.1, 2.9, 4.1,
        0.6, 1.4, 2.6, 3.4,
        1.9, 3.1, 4.0, 5.2,
    ], &[3, 4]);

    graph.set_node_value(input_id, Some(&input_data)).unwrap();
    graph.set_node_value(target_id, Some(&target_data)).unwrap();

    graph.forward_batch(loss_id).unwrap();
    graph.backward_batch(loss_id, None).unwrap();

    let grad = graph.get_node_grad_batch(input_id).unwrap().unwrap();

    // PyTorch 验证的梯度值
    #[rustfmt::skip]
    let expected_grad = Tensor::new(&[
        -0.033_333_34, -0.016_666_65,  0.016_666_65, -0.016_666_65,
        -0.016_666_67,  0.016_666_67, -0.016_666_65,  0.016_666_65,
         0.016_666_67, -0.016_666_65,  0.0,          -0.033_333_30,
    ], &[3, 4]);

    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

// ========== 数值稳定性测试 ==========

#[test]
fn test_mse_loss_large_values() {
    let mut graph = Graph::new();

    let input_id = graph.new_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(
            input_id,
            Some(&Tensor::new(&[1000.0, 2000.0, 3000.0], &[1, 3])),
        )
        .unwrap();
    graph
        .set_node_value(
            target_id,
            Some(&Tensor::new(&[1000.5, 2000.5, 3000.5], &[1, 3])),
        )
        .unwrap();

    graph.forward_node(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 同样是 diff = 0.5，所以 loss = 0.25
    assert_abs_diff_eq!(loss[[0, 0]], 0.25, epsilon = 1e-6);
}

#[test]
fn test_mse_loss_small_values() {
    let mut graph = Graph::new();

    let input_id = graph.new_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(
            input_id,
            Some(&Tensor::new(&[0.001, 0.002, 0.003], &[1, 3])),
        )
        .unwrap();
    graph
        .set_node_value(
            target_id,
            Some(&Tensor::new(&[0.0015, 0.0025, 0.0035], &[1, 3])),
        )
        .unwrap();

    graph.forward_node(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // PyTorch: 2.4999997094710125e-07
    assert_abs_diff_eq!(loss[[0, 0]], 2.5e-7, epsilon = 1e-10);
}

// ========== 梯度清除测试 ==========

#[test]
fn test_mse_loss_clear_jacobi() {
    let mut graph = Graph::new();

    // 使用 Parameter 作为 input，这样可以计算梯度
    let input_id = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3])))
        .unwrap();

    // 第一次前向+反向（retain_graph=true 以便后续继续 backward）
    graph.forward_node(loss_id).unwrap();
    graph.backward_nodes_ex(&[input_id], loss_id, true).unwrap();

    let jacobi_first = graph.get_node_jacobi(input_id).unwrap().unwrap();
    let expected = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);
    assert_abs_diff_eq!(jacobi_first, &expected, epsilon = 1e-5);

    // 清除 Jacobi
    graph.clear_jacobi().unwrap();

    // 再次反向传播（最后一次可以不保留图）
    graph.backward_nodes(&[input_id], loss_id).unwrap();
    let jacobi_after_clear = graph.get_node_jacobi(input_id).unwrap().unwrap();
    assert_abs_diff_eq!(jacobi_after_clear, &expected, epsilon = 1e-5);
}

// ========== 端到端训练测试 ==========

/// 简单的回归训练测试
/// 目标: y = 2x (学习斜率)
#[test]
fn test_mse_loss_simple_regression_training() {
    use crate::nn::optimizer::{Optimizer, SGD};

    let mut graph = Graph::new_with_seed(42);

    // 创建网络: y_pred = x * w
    let x_id = graph.new_input_node(&[1, 1], Some("x")).unwrap();
    let w_id = graph.new_parameter_node(&[1, 1], Some("w")).unwrap();
    let y_pred_id = graph.new_mat_mul_node(x_id, w_id, Some("y_pred")).unwrap();
    let y_true_id = graph.new_input_node(&[1, 1], Some("y_true")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(y_pred_id, y_true_id, Some("loss"))
        .unwrap();

    // 初始化权重为 0.5（目标是 2.0）
    graph
        .set_node_value(w_id, Some(&Tensor::new(&[0.5], &[1, 1])))
        .unwrap();

    let mut optimizer = SGD::new(&graph, 0.1).unwrap();

    // 训练数据: x=1 -> y=2, x=2 -> y=4, x=3 -> y=6
    let training_data = [(1.0_f32, 2.0_f32), (2.0, 4.0), (3.0, 6.0)];

    // 训练 50 个 epoch
    for _ in 0..50 {
        for &(x_val, y_val) in &training_data {
            graph
                .set_node_value(x_id, Some(&Tensor::new(&[x_val], &[1, 1])))
                .unwrap();
            graph
                .set_node_value(y_true_id, Some(&Tensor::new(&[y_val], &[1, 1])))
                .unwrap();

            optimizer.one_step(&mut graph, loss_id).unwrap();
        }
        optimizer.update(&mut graph).unwrap();
    }

    // 验证学习到的权重接近 2.0
    let learned_w = graph.get_node_value(w_id).unwrap().unwrap();
    assert_abs_diff_eq!(learned_w[[0, 0]], 2.0, epsilon = 0.1);
}

