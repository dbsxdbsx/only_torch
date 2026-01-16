/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : MSELoss 节点单元测试
 */

use crate::nn::{GraphInner, Reduction};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ========== 基本功能测试 ==========

#[test]
fn test_mse_loss_creation() {
    let mut graph = GraphInner::new();

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
    let mut graph = GraphInner::new();

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
    let mut graph = GraphInner::new();

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
    graph.forward(loss_id).unwrap();

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
    let mut graph = GraphInner::new();

    let input_id = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[2, 2], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    graph
        .set_node_value(
            target_id,
            Some(&Tensor::new(&[1.5, 2.5, 3.5, 4.5], &[2, 2])),
        )
        .unwrap();

    graph.forward(loss_id).unwrap();

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
    let mut graph = GraphInner::new();

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

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 3 * 0.25 = 0.75
    assert_abs_diff_eq!(loss[[0, 0]], 0.75, epsilon = 1e-6);
}

// ========== 端到端反向传播测试（通过 graph.backward）==========

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = 2 * (input - target) / N = 2 * (-0.5) / 3 = -0.333...
/// ```
#[test]
fn test_mse_loss_backward_e2e_mean() {
    let mut graph = GraphInner::new();

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

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    // 获取梯度
    let grad = graph.get_node(input_id).unwrap().grad().unwrap();

    // 预期梯度: 2 * (input - target) / N = 2 * [-0.5, -0.5, -0.5] / 3
    let expected_grad = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);

    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
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
fn test_mse_loss_backward_e2e_sum() {
    let mut graph = GraphInner::new();

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

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(input_id).unwrap().grad().unwrap();
    let expected_grad = Tensor::new(&[-1.0, -1.0, -1.0], &[1, 3]);

    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-6);
}

/// 测试 2D 矩阵的梯度
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = 2 * (-0.5) / 4 = -0.25
/// ```
#[test]
fn test_mse_loss_backward_e2e_2d() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input")).unwrap();
    let target_id = graph.new_input_node(&[2, 2], Some("target")).unwrap();
    let loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    graph
        .set_node_value(
            target_id,
            Some(&Tensor::new(&[1.5, 2.5, 3.5, 4.5], &[2, 2])),
        )
        .unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(input_id).unwrap().grad().unwrap();
    let expected_grad = Tensor::new(&[-0.25, -0.25, -0.25, -0.25], &[2, 2]);

    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-6);
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
    let mut graph = GraphInner::new();

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

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // PyTorch: 0.014166653156280518
    assert_abs_diff_eq!(loss[[0, 0]], 0.014_166_653, epsilon = 1e-6);
}

/// Batch 模式的反向传播测试
#[test]
fn test_mse_loss_batch_backward() {
    let mut graph = GraphInner::new();

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

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(input_id).unwrap().grad().unwrap();

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
    let mut graph = GraphInner::new();

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

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 同样是 diff = 0.5，所以 loss = 0.25
    assert_abs_diff_eq!(loss[[0, 0]], 0.25, epsilon = 1e-6);
}

#[test]
fn test_mse_loss_small_values() {
    let mut graph = GraphInner::new();

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

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // PyTorch: 2.4999997094710125e-07
    assert_abs_diff_eq!(loss[[0, 0]], 2.5e-7, epsilon = 1e-10);
}

// ========== 梯度累积测试 ==========

#[test]
fn test_mse_loss_gradient_accumulation() {
    let mut graph = GraphInner::new();

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

    // 第一次前向+反向
    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad_first = graph.get_node(input_id).unwrap().grad().unwrap().clone();
    let expected = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);
    assert_abs_diff_eq!(&grad_first, &expected, epsilon = 1e-5);

    // 第二次反向传播（梯度累积）- 需要重新 forward（PyTorch 语义）
    graph.forward(loss_id).unwrap();
    graph.backward(loss_id).unwrap();
    let grad_second = graph.get_node(input_id).unwrap().grad().unwrap();
    assert_eq!(grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad().unwrap();
    graph.forward(loss_id).unwrap();
    graph.backward(loss_id).unwrap();
    let grad_after_clear = graph.get_node(input_id).unwrap().grad().unwrap();
    assert_eq!(grad_after_clear, &grad_first);
}

// ========== 端到端训练测试 ==========

/// 简单的回归训练测试
/// 目标: y = 2x (学习斜率)
#[test]
fn test_mse_loss_simple_regression_training() {
    let mut graph = GraphInner::new_with_seed(42);

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

    let lr = 0.1;

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

            graph.zero_grad().unwrap();
            graph.forward(loss_id).unwrap();
            graph.backward(loss_id).unwrap();

            // 手动 SGD 更新：w = w - lr * grad
            let w_val = graph.get_node_value(w_id).unwrap().unwrap();
            let w_grad = graph.get_node_grad(w_id).unwrap().unwrap();
            let new_w = w_val - lr * &w_grad;
            graph.set_node_value(w_id, Some(&new_w)).unwrap();
        }
    }

    // 验证学习到的权重接近 2.0
    let learned_w = graph.get_node_value(w_id).unwrap().unwrap();
    assert_abs_diff_eq!(learned_w[[0, 0]], 2.0, epsilon = 0.1);
}
