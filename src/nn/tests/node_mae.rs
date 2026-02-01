/*
 * @Author       : 老董
 * @Date         : 2026-01-28
 * @Description  : MAE（Mean Absolute Error）节点单元测试
 */

use crate::nn::{GraphInner, Reduction};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ========== 基本功能测试 ==========

#[test]
fn test_mae_loss_creation() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    // 验证节点存在且预期形状正确
    assert_eq!(
        graph.get_node_value_expected_shape(loss_id).unwrap(),
        &[1, 1]
    );
}

#[test]
fn test_mae_loss_shape_mismatch() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 4], Some("target")).unwrap(); // 形状不匹配

    let result = graph.new_mae_loss_node(input_id, target_id, None);
    assert!(result.is_err());
}

// ========== Mean Reduction 测试 ==========

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.L1Loss(reduction='mean')(input, target)
/// # loss = mean(|[-0.5, -0.5, -0.5]|) = 0.5
/// # grad = sign(input - target) / N = [-1, -1, -1] / 3 = [-0.333..., -0.333..., -0.333...]
/// ```
#[test]
fn test_mae_loss_forward_mean_basic() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
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

    // 验证损失值: mean(|0.5| + |0.5| + |0.5|) = 0.5
    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    assert_abs_diff_eq!(loss[[0, 0]], 0.5, epsilon = 1e-6);
}

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
/// loss = nn.L1Loss(reduction='mean')(input, target)
/// # loss = 0.5
/// ```
#[test]
fn test_mae_loss_forward_2d_matrix() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[2, 2], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[2, 2], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
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
    // 4 个元素，每个 abs_diff = 0.5
    // sum = 2.0, mean = 0.5
    assert_abs_diff_eq!(loss[[0, 0]], 0.5, epsilon = 1e-6);
}

/// 测试混合正负差值
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 3.0, 2.0]], requires_grad=True)
/// target = torch.tensor([[2.0, 1.0, 2.5]])
/// loss = nn.L1Loss(reduction='mean')(input, target)
/// # diff = [-1.0, 2.0, -0.5]
/// # abs_diff = [1.0, 2.0, 0.5]
/// # loss = mean([1.0, 2.0, 0.5]) = 3.5 / 3 = 1.1666...
/// ```
#[test]
fn test_mae_loss_forward_mixed_diff() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 3.0, 2.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[2.0, 1.0, 2.5], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // PyTorch: 1.1666666269302368
    assert_abs_diff_eq!(loss[[0, 0]], 1.166_666_6, epsilon = 1e-5);
}

// ========== Sum Reduction 测试 ==========

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.L1Loss(reduction='sum')(input, target)
/// # loss = 1.5
/// # grad = sign(input - target) = [-1.0, -1.0, -1.0]
/// ```
#[test]
fn test_mae_loss_forward_sum() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node_with_reduction(input_id, target_id, Reduction::Sum, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 3 * 0.5 = 1.5
    assert_abs_diff_eq!(loss[[0, 0]], 1.5, epsilon = 1e-6);
}

// ========== 端到端反向传播测试（通过 graph.backward）==========

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.L1Loss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = sign(input - target) / N = sign([-0.5, -0.5, -0.5]) / 3 = [-0.333..., -0.333..., -0.333...]
/// ```
#[test]
fn test_mae_loss_backward_e2e_mean() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
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

    // 预期梯度: sign(input - target) / N = sign([-0.5, -0.5, -0.5]) / 3 = [-1, -1, -1] / 3
    let expected_grad = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);

    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.L1Loss(reduction='sum')(input, target)
/// loss.backward()
/// # grad = sign(input - target) = [-1.0, -1.0, -1.0]
/// ```
#[test]
fn test_mae_loss_backward_e2e_sum() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node_with_reduction(input_id, target_id, Reduction::Sum, Some("loss"))
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

/// 测试混合正负差值的梯度
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 3.0, 2.0]], requires_grad=True)
/// target = torch.tensor([[2.0, 1.0, 2.5]])
/// loss = nn.L1Loss(reduction='mean')(input, target)
/// loss.backward()
/// # diff = [-1.0, 2.0, -0.5]
/// # grad = sign(diff) / N = [-1, 1, -1] / 3
/// ```
#[test]
fn test_mae_loss_backward_e2e_mixed() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 3.0, 2.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[2.0, 1.0, 2.5], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(input_id).unwrap().grad().unwrap();
    // sign([-1.0, 2.0, -0.5]) / 3 = [-1, 1, -1] / 3
    let expected_grad = Tensor::new(&[-0.333_333_34, 0.333_333_34, -0.333_333_34], &[1, 3]);

    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

/// 测试 2D 矩阵的梯度
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
/// loss = nn.L1Loss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = sign([-0.5]) / 4 = -0.25
/// ```
#[test]
fn test_mae_loss_backward_e2e_2d() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[2, 2], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
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

// ========== 批量输入测试 ==========

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
/// loss = nn.L1Loss(reduction='mean')(input, target)
/// # loss ≈ 0.1083...
/// ```
#[test]
fn test_mae_loss_batch_forward() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[3, 4], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[3, 4], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
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
    // PyTorch: 0.10833333432674408
    assert_abs_diff_eq!(loss[[0, 0]], 0.108_333_33, epsilon = 1e-5);
}

/// 批量输入的反向传播测试
#[test]
fn test_mae_loss_batch_backward() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[3, 4], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[3, 4], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
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
    // diff = input - target:
    // [[-0.2, -0.1, 0.1, -0.1],
    //  [-0.1, 0.1, -0.1, 0.1],
    //  [0.1, -0.1, 0.0, -0.2]]
    // sign(diff) / 12:
    #[rustfmt::skip]
    let expected_grad = Tensor::new(&[
        -0.083_333_336, -0.083_333_336,  0.083_333_336, -0.083_333_336,
        -0.083_333_336,  0.083_333_336, -0.083_333_336,  0.083_333_336,
         0.083_333_336, -0.083_333_336,  0.0,           -0.083_333_336,
    ], &[3, 4]);

    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

// ========== 数值稳定性测试 ==========

#[test]
fn test_mae_loss_large_values() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
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
    // 同样是 diff = 0.5，所以 loss = 0.5
    assert_abs_diff_eq!(loss[[0, 0]], 0.5, epsilon = 1e-6);
}

#[test]
fn test_mae_loss_small_values() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
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
    // mean(|0.0005| * 3) = 0.0005
    assert_abs_diff_eq!(loss[[0, 0]], 0.0005, epsilon = 1e-8);
}

// ========== 梯度累积测试 ==========

#[test]
fn test_mae_loss_gradient_accumulation() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("loss"))
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
fn test_mae_loss_simple_regression_training() {
    let mut graph = GraphInner::new_with_seed(42);

    // 创建网络: y_pred = x * w
    let x_id = graph.new_basic_input_node(&[1, 1], Some("x")).unwrap();
    let w_id = graph.new_parameter_node(&[1, 1], Some("w")).unwrap();
    let y_pred_id = graph.new_mat_mul_node(x_id, w_id, Some("y_pred")).unwrap();
    let y_true_id = graph.new_basic_input_node(&[1, 1], Some("y_true")).unwrap();
    let loss_id = graph
        .new_mae_loss_node(y_pred_id, y_true_id, Some("loss"))
        .unwrap();

    // 初始化权重为 0.5（目标是 2.0）
    graph
        .set_node_value(w_id, Some(&Tensor::new(&[0.5], &[1, 1])))
        .unwrap();

    let lr = 0.1;

    // 训练数据: x=1 -> y=2, x=2 -> y=4, x=3 -> y=6
    let training_data = [(1.0_f32, 2.0_f32), (2.0, 4.0), (3.0, 6.0)];

    // 训练 100 个 epoch（MAE 梯度恒定，可能需要更多迭代）
    for _ in 0..100 {
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

// ==================== 动态形状测试 ====================

use crate::nn::GraphError;

/// 测试 MAE 节点的动态形状（输出固定为标量 [1, 1]）
#[test]
fn test_mae_loss_dynamic_shape_output_fixed() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[4, 8], Some("input"))?;
    let target = graph.new_basic_input_node(&[4, 8], Some("target"))?;
    let loss = graph.new_mae_loss_node(input, target, Some("loss"))?;

    // MAE 输出形状始终是 [1, 1]（标量）
    let node = graph.get_node(loss)?;
    let dyn_shape = node.dynamic_expected_shape();

    // 输出形状固定
    assert!(!dyn_shape.is_dynamic(0), "MAE 输出维度 0 应固定");
    assert!(!dyn_shape.is_dynamic(1), "MAE 输出维度 1 应固定");
    assert_eq!(dyn_shape.dim(0), Some(1));
    assert_eq!(dyn_shape.dim(1), Some(1));

    Ok(())
}

/// 测试 MAE 接受动态 batch 输入
#[test]
fn test_mae_loss_dynamic_batch_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 4], Some("input"))?;
    let target = graph.new_basic_input_node(&[2, 4], Some("target"))?;
    let loss = graph.new_mae_loss_node(input, target, Some("loss"))?;

    // 第一次 forward：batch=2
    graph.set_node_value(input, Some(&Tensor::ones(&[2, 4])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 4])))?;
    graph.forward(loss)?;
    let loss_val1 = graph.get_node_value(loss)?.unwrap();
    assert_eq!(loss_val1.shape(), &[1, 1], "MAE 输出应为标量");
    assert!(loss_val1[[0, 0]] > 0.0);

    // 第二次 forward：batch=6（不同 batch 大小）
    graph.set_node_value(input, Some(&Tensor::ones(&[6, 4])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[6, 4])))?;
    graph.forward(loss)?;
    let loss_val2 = graph.get_node_value(loss)?.unwrap();
    assert_eq!(loss_val2.shape(), &[1, 1], "MAE 输出应始终为标量");

    Ok(())
}

/// 测试 MAE 在不同 batch 大小下的反向传播
#[test]
fn test_mae_loss_dynamic_batch_backward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 使用 Input 节点接收动态 batch 数据
    // 注意：Input 节点支持动态 batch 但没有 grad
    // 为了测试梯度反传，使用 y = x * w 形式，其中 w 是可训练的 Parameter
    let input = graph.new_basic_input_node(&[2, 4], Some("input"))?;
    let weight = graph.new_parameter_node(&[4, 4], Some("weight"))?; // [4, 4] 权重
    let pred = graph.new_mat_mul_node(input, weight, Some("pred"))?;
    let target = graph.new_basic_input_node(&[2, 4], Some("target"))?;
    let loss = graph.new_mae_loss_node(pred, target, Some("loss"))?;

    // 初始化权重
    graph.set_node_value(weight, Some(&Tensor::ones(&[4, 4])))?;

    // 第一次训练：batch=2
    graph.set_node_value(input, Some(&Tensor::ones(&[2, 4])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 4])))?;
    graph.forward(loss)?;
    graph.zero_grad()?;
    graph.backward(loss)?;

    let grad1 = graph.get_node_grad(weight)?.unwrap().clone();
    assert_eq!(grad1.shape(), &[4, 4], "权重梯度形状应保持不变");

    // 更新为不同 batch
    graph.set_node_value(input, Some(&Tensor::ones(&[5, 4])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[5, 4])))?;

    // 第二次训练：batch=5
    graph.forward(loss)?;
    graph.zero_grad()?;
    graph.backward(loss)?;

    let grad2 = graph.get_node_grad(weight)?.unwrap();
    assert_eq!(
        grad2.shape(),
        &[4, 4],
        "权重梯度形状应保持不变（与 batch 大小无关）"
    );

    Ok(())
}

/// 测试 MAE 的动态形状兼容性检查
///
/// MAE 验证 input 和 target 的动态形状兼容性
#[test]
fn test_mae_loss_dynamic_shape_compatibility() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 两个输入都支持动态 batch
    let input = graph.new_basic_input_node(&[4, 8], Some("input"))?;
    let target = graph.new_basic_input_node(&[4, 8], Some("target"))?;

    // 验证动态形状兼容
    let input_node = graph.get_node(input)?;
    let target_node = graph.get_node(target)?;
    let input_dyn = input_node.dynamic_expected_shape();
    let target_dyn = target_node.dynamic_expected_shape();

    assert!(
        input_dyn.is_compatible(&target_dyn),
        "Input 和 Target 的动态形状应兼容"
    );

    // 创建 MAE 节点应该成功
    let loss = graph.new_mae_loss_node(input, target, Some("loss"))?;
    assert!(graph.get_node(loss).is_ok());

    Ok(())
}

// ========== MAE vs MSE 对比测试 ==========

/// 验证 MAE 和 MSE 在相同输入下的不同输出
#[test]
fn test_mae_vs_mse_comparison() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let mae_loss_id = graph
        .new_mae_loss_node(input_id, target_id, Some("mae_loss"))
        .unwrap();
    let mse_loss_id = graph
        .new_mse_loss_node(input_id, target_id, Some("mse_loss"))
        .unwrap();

    // 设置输入值（差值为 0.5）
    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3])))
        .unwrap();

    // 前向传播
    graph.forward(mae_loss_id).unwrap();
    graph.forward(mse_loss_id).unwrap();

    let mae_loss = graph.get_node_value(mae_loss_id).unwrap().unwrap();
    let mse_loss = graph.get_node_value(mse_loss_id).unwrap().unwrap();

    // MAE = mean(|0.5|) = 0.5
    // MSE = mean(0.5^2) = 0.25
    assert_abs_diff_eq!(mae_loss[[0, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(mse_loss[[0, 0]], 0.25, epsilon = 1e-6);

    // MAE > MSE when |diff| < 1
    assert!(mae_loss[[0, 0]] > mse_loss[[0, 0]]);
}

// ==================== 方案 C：新节点创建 API 测试 ====================

use crate::nn::Graph;
use std::rc::Rc;

#[test]
fn test_create_mae_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("target"))
        .unwrap();

    let mae = inner
        .borrow_mut()
        .create_mae_mean_node(input.clone(), target.clone(), Some("mae"))
        .unwrap();

    // MAE 输出形状固定为 [1, 1]
    assert_eq!(mae.shape(), vec![1, 1]);
    assert_eq!(mae.name(), Some("mae"));
    assert!(!mae.is_leaf());
    assert_eq!(mae.parents().len(), 2);
}

#[test]
fn test_create_mae_node_shape_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None) // 形状不匹配
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_mae_mean_node(input, target, None);

    assert!(result.is_err());
}

#[test]
fn test_create_mae_node_output_always_scalar() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let mae = inner
        .borrow_mut()
        .create_mae_mean_node(input, target, None)
        .unwrap();
    assert_eq!(mae.shape(), vec![1, 1]);
}

#[test]
fn test_create_mae_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_mae;
    let weak_input;
    let weak_target;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let target = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_target = Rc::downgrade(&target);

        let mae = inner
            .borrow_mut()
            .create_mae_mean_node(input, target, None)
            .unwrap();
        weak_mae = Rc::downgrade(&mae);

        assert!(weak_mae.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
        assert!(weak_target.upgrade().is_some());
    }
    assert!(weak_mae.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
    assert!(weak_target.upgrade().is_none());
}
