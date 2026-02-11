/*
 * @Author       : 老董
 * @Description  : MSE（Mean Squared Error）损失节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ mean basic [1,3], 2D [2,2], batch [3,4], 大值/小值稳定性, 形状不匹配, 不可设值
 * 2. VJP 单元测试（底层）→ mean [1,3]/[2,2] 梯度验证, sum reduction 前向+梯度
 * 3. 端到端反向传播测试（高层）→ mean [1,3]/[2,2]/batch, 简单回归训练
 * 4. 梯度累积测试（高层）
 * 5. 动态形状测试
 * 6. 新节点创建 API 测试（KEEP AS-IS）
 *
 * 关键点：MSE loss = mean/sum((input - target)^2)。
 * Mean VJP: grad = 2*(input-target)/N。Sum VJP: grad = 2*(input-target)。
 * 高层 API: VarLossOps 的 .mse_loss() 使用 Mean reduction。
 */

use crate::nn::nodes::raw_node::Reduction;
use crate::nn::{Graph, GraphError, Init, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// # loss = 0.25
/// ```
#[test]
fn test_mse_loss_forward_mean_basic() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3]))
        .unwrap();
    let loss = input.mse_loss(&target).unwrap();

    // 验证 MSE 输出形状固定为 [1,1]
    let dyn_shape = loss.dynamic_expected_shape();
    assert_eq!(dyn_shape.dim(0), Some(1));
    assert_eq!(dyn_shape.dim(1), Some(1));

    // 前向传播
    loss.forward().unwrap();

    // 验证损失值: ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.75 / 3 = 0.25
    let loss_val = loss.value().unwrap().unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], 0.25, epsilon = 1e-6);
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
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.5, 2.5, 3.5, 4.5], &[2, 2]))
        .unwrap();
    let loss = input.mse_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.value().unwrap().unwrap();
    // 4 个元素，每个 diff = 0.5, squared = 0.25
    // sum = 1.0, mean = 0.25
    assert_abs_diff_eq!(loss_val[[0, 0]], 0.25, epsilon = 1e-6);
}

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
    let graph = Graph::new();

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

    let input = graph.input(&input_data).unwrap();
    let target = graph.input(&target_data).unwrap();
    let loss = input.mse_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.value().unwrap().unwrap();
    // PyTorch: 0.014166653156280518
    assert_abs_diff_eq!(loss_val[[0, 0]], 0.014_166_653, epsilon = 1e-6);
}

/// 数值稳定性测试（大数值）
#[test]
fn test_mse_loss_large_values() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1000.0, 2000.0, 3000.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1000.5, 2000.5, 3000.5], &[1, 3]))
        .unwrap();
    let loss = input.mse_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.value().unwrap().unwrap();
    // 同样是 diff = 0.5，所以 loss = 0.25
    assert_abs_diff_eq!(loss_val[[0, 0]], 0.25, epsilon = 1e-6);
}

/// 数值稳定性测试（小数值）
#[test]
fn test_mse_loss_small_values() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[0.001, 0.002, 0.003], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.0015, 0.0025, 0.0035], &[1, 3]))
        .unwrap();
    let loss = input.mse_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.value().unwrap().unwrap();
    // PyTorch: 2.4999997094710125e-07
    assert_abs_diff_eq!(loss_val[[0, 0]], 2.5e-7, epsilon = 1e-10);
}

/// 测试 input 和 target 形状不匹配时报错
#[test]
fn test_mse_loss_shape_mismatch() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]))
        .unwrap();

    let result = input.mse_loss(&target);
    assert!(result.is_err());
}

/// 测试 MSE 损失节点不能直接设置值
#[test]
fn test_mse_loss_cannot_set_value() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3]))
        .unwrap();
    let loss = input.mse_loss(&target).unwrap();

    let test_value = Tensor::new(&[0.5], &[1, 1]);
    let err = loss.set_value(&test_value);
    assert!(err.is_err(), "MSE 损失节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证梯度计算公式。
// Mean VJP: grad = 2*(input-target)/N
// Sum  VJP: grad = 2*(input-target)

/// 测试 MSE VJP（Mean reduction）：基础 [1,3]
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = 2 * (input - target) / N = 2 * (-0.5) / 3 = -0.333...
/// ```
#[test]
fn test_mse_vjp_mean_basic() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("target"))
        .unwrap();
    let mse = inner
        .borrow_mut()
        .create_mse_mean_node(input.clone(), target.clone(), Some("mse"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3])))
        .unwrap();
    mse.forward_recursive(1, false).unwrap();

    // 验证前向值
    let mse_val = mse.value().expect("mse 应有值");
    assert_abs_diff_eq!(mse_val[[0, 0]], 0.25, epsilon = 1e-6);

    // VJP: upstream=1 (标量损失)
    let upstream = Tensor::ones(&[1, 1]);
    let grad = mse.calc_grad_to_parent_index(0, &upstream)?;

    // 预期: 2*(input-target)/N = 2*[-0.5,-0.5,-0.5]/3 = [-1/3, -1/3, -1/3]
    assert_eq!(grad.shape(), &[1, 3]);
    let expected = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);
    assert_abs_diff_eq!(grad, &expected, epsilon = 1e-5);

    Ok(())
}

/// 测试 MSE VJP（Mean reduction）：2D 矩阵 [2,2]
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = 2 * (-0.5) / 4 = -0.25
/// ```
#[test]
fn test_mse_vjp_mean_2d() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("target"))
        .unwrap();
    let mse = inner
        .borrow_mut()
        .create_mse_mean_node(input.clone(), target.clone(), Some("mse"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[1.5, 2.5, 3.5, 4.5], &[2, 2])))
        .unwrap();
    mse.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[1, 1]);
    let grad = mse.calc_grad_to_parent_index(0, &upstream)?;

    // 预期: 2*(-0.5)/4 = -0.25
    assert_eq!(grad.shape(), &[2, 2]);
    let expected = Tensor::new(&[-0.25, -0.25, -0.25, -0.25], &[2, 2]);
    assert_abs_diff_eq!(grad, &expected, epsilon = 1e-6);

    Ok(())
}

/// 测试 MSE VJP（Sum reduction）：前向 + 梯度 [1,3]
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.MSELoss(reduction='sum')(input, target)
/// # loss = 0.75
/// loss.backward()
/// # grad = 2 * (input - target) = 2 * (-0.5) = -1.0
/// ```
#[test]
fn test_mse_vjp_sum_forward_and_grad() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("target"))
        .unwrap();
    let mse = inner
        .borrow_mut()
        .create_mse_node(input.clone(), target.clone(), Reduction::Sum, Some("mse"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3])))
        .unwrap();
    mse.forward_recursive(1, false).unwrap();

    // 前向验证: sum([0.25, 0.25, 0.25]) = 0.75
    let mse_val = mse.value().expect("mse 应有值");
    assert_abs_diff_eq!(mse_val[[0, 0]], 0.75, epsilon = 1e-6);

    // VJP: 2*(input-target) = 2*[-0.5,-0.5,-0.5] = [-1, -1, -1]
    let upstream = Tensor::ones(&[1, 1]);
    let grad = mse.calc_grad_to_parent_index(0, &upstream)?;

    assert_eq!(grad.shape(), &[1, 3]);
    let expected = Tensor::new(&[-1.0, -1.0, -1.0], &[1, 3]);
    assert_abs_diff_eq!(grad, &expected, epsilon = 1e-6);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = 2 * (input - target) / N = 2 * (-0.5) / 3 = -0.333...
/// ```
#[test]
fn test_mse_loss_backward_e2e_mean() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input = graph.parameter(&[1, 3], Init::Zeros, "input")?;
    input.set_value(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))?;

    let target = graph.input(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3]))?;
    let loss = input.mse_loss(&target)?;

    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;

    // 获取梯度
    let grad = input.grad()?.expect("应有 grad");

    // 预期梯度: 2 * (input - target) / N = 2 * [-0.5, -0.5, -0.5] / 3
    let expected_grad = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);
    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);

    Ok(())
}

/// 测试 2D 矩阵的端到端梯度
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
/// loss = nn.MSELoss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = 2 * (-0.5) / 4 = -0.25
/// ```
#[test]
fn test_mse_loss_backward_e2e_2d() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input = graph.parameter(&[2, 2], Init::Zeros, "input")?;
    input.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;

    let target = graph.input(&Tensor::new(&[1.5, 2.5, 3.5, 4.5], &[2, 2]))?;
    let loss = input.mse_loss(&target)?;

    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;

    let grad = input.grad()?.expect("应有 grad");
    let expected_grad = Tensor::new(&[-0.25, -0.25, -0.25, -0.25], &[2, 2]);
    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-6);

    Ok(())
}

/// 批量输入的端到端反向传播测试
#[test]
fn test_mse_loss_batch_backward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input = graph.parameter(&[3, 4], Init::Zeros, "input")?;

    #[rustfmt::skip]
    let input_data = Tensor::new(&[
        1.0, 2.0, 3.0, 4.0,
        0.5, 1.5, 2.5, 3.5,
        2.0, 3.0, 4.0, 5.0,
    ], &[3, 4]);
    input.set_value(&input_data)?;

    #[rustfmt::skip]
    let target_data = Tensor::new(&[
        1.2, 2.1, 2.9, 4.1,
        0.6, 1.4, 2.6, 3.4,
        1.9, 3.1, 4.0, 5.2,
    ], &[3, 4]);
    let target = graph.input(&target_data)?;
    let loss = input.mse_loss(&target)?;

    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;

    let grad = input.grad()?.expect("应有 grad");

    // PyTorch 验证的梯度值
    #[rustfmt::skip]
    let expected_grad = Tensor::new(&[
        -0.033_333_34, -0.016_666_65,  0.016_666_65, -0.016_666_65,
        -0.016_666_67,  0.016_666_67, -0.016_666_65,  0.016_666_65,
         0.016_666_67, -0.016_666_65,  0.0,          -0.033_333_30,
    ], &[3, 4]);

    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);

    Ok(())
}

/// 简单回归训练测试
/// 目标: y = 2x（学习斜率）
#[test]
fn test_mse_loss_simple_regression_training() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 创建网络: y_pred = x * w
    let x = graph.input(&Tensor::new(&[1.0], &[1, 1]))?;
    let w = graph.parameter(&[1, 1], Init::Zeros, "w")?;
    w.set_value(&Tensor::new(&[0.5], &[1, 1]))?; // 初始化权重 0.5（目标 2.0）

    let y_pred = x.matmul(&w)?;
    let y_true = graph.input(&Tensor::new(&[2.0], &[1, 1]))?;
    let loss = y_pred.mse_loss(&y_true)?;

    let lr = 0.1_f32;
    let training_data = [(1.0_f32, 2.0_f32), (2.0, 4.0), (3.0, 6.0)];

    // 训练 50 个 epoch
    for _ in 0..50 {
        for &(x_val, y_val) in &training_data {
            x.set_value(&Tensor::new(&[x_val], &[1, 1]))?;
            y_true.set_value(&Tensor::new(&[y_val], &[1, 1]))?;

            graph.zero_grad()?;
            loss.forward().unwrap();
            loss.backward()?;

            // 手动 SGD 更新：w = w - lr * grad
            let w_val = w.value().unwrap().unwrap();
            let w_grad = w.grad()?.unwrap();
            let scaled_grad = &w_grad * lr;
            let new_w = &w_val - &scaled_grad;
            w.set_value(&new_w)?;
        }
    }

    // 验证学习到的权重接近 2.0
    let learned_w = w.value().unwrap().unwrap();
    assert_abs_diff_eq!(learned_w[[0, 0]], 2.0, epsilon = 0.1);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试 MSE 梯度累积
#[test]
fn test_mse_loss_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input = graph.parameter(&[1, 3], Init::Zeros, "input")?;
    input.set_value(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))?;

    let target = graph.input(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3]))?;
    let loss = input.mse_loss(&target)?;

    // 第一次前向+反向
    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;

    let grad_first = input.grad()?.unwrap().clone();
    let expected = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);
    assert_abs_diff_eq!(&grad_first, &expected, epsilon = 1e-5);

    // 第二次反向传播（梯度累积）
    loss.forward().unwrap();
    loss.backward()?;
    let grad_second = input.grad()?.unwrap();
    assert_eq!(&grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    loss.forward().unwrap();
    loss.backward()?;
    let grad_after_clear = input.grad()?.unwrap();
    assert_eq!(&grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 MSE 节点的动态形状（输出固定为标量 [1, 1]）
#[test]
fn test_mse_loss_dynamic_shape_output_fixed() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input = graph.input(&Tensor::zeros(&[4, 8]))?;
    let target = graph.input(&Tensor::zeros(&[4, 8]))?;
    let loss = input.mse_loss(&target)?;

    // MSE 输出形状始终是 [1, 1]（标量）
    let dyn_shape = loss.dynamic_expected_shape();
    assert!(!dyn_shape.is_dynamic(0), "MSE 输出维度 0 应固定");
    assert!(!dyn_shape.is_dynamic(1), "MSE 输出维度 1 应固定");
    assert_eq!(dyn_shape.dim(0), Some(1));
    assert_eq!(dyn_shape.dim(1), Some(1));

    Ok(())
}

/// 测试 MSE 接受动态 batch 输入
#[test]
fn test_mse_loss_dynamic_batch_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input = graph.input(&Tensor::ones(&[2, 4]))?;
    let target = graph.input(&Tensor::zeros(&[2, 4]))?;
    let loss = input.mse_loss(&target)?;

    // 第一次 forward：batch=2
    loss.forward().unwrap();
    let loss_val1 = loss.value().unwrap().unwrap();
    assert_eq!(loss_val1.shape(), &[1, 1], "MSE 输出应为标量");
    assert!(loss_val1[[0, 0]] > 0.0);

    // 第二次 forward：batch=6（不同 batch 大小）
    input.set_value(&Tensor::ones(&[6, 4]))?;
    target.set_value(&Tensor::zeros(&[6, 4]))?;
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap();
    assert_eq!(loss_val2.shape(), &[1, 1], "MSE 输出应始终为标量");

    Ok(())
}

/// 测试 MSE 在不同 batch 大小下的反向传播
#[test]
fn test_mse_loss_dynamic_batch_backward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input = graph.input(&Tensor::ones(&[2, 4]))?;
    let weight = graph.parameter(&[4, 4], Init::Zeros, "weight")?;
    weight.set_value(&Tensor::ones(&[4, 4]))?;

    let pred = input.matmul(&weight)?;
    let target = graph.input(&Tensor::zeros(&[2, 4]))?;
    let loss = pred.mse_loss(&target)?;

    // 第一次训练：batch=2
    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;

    let grad1 = weight.grad()?.unwrap().clone();
    assert_eq!(grad1.shape(), &[4, 4], "权重梯度形状应保持不变");

    // 更新为不同 batch
    input.set_value(&Tensor::ones(&[5, 4]))?;
    target.set_value(&Tensor::zeros(&[5, 4]))?;

    // 第二次训练：batch=5
    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;

    let grad2 = weight.grad()?.unwrap();
    assert_eq!(
        grad2.shape(),
        &[4, 4],
        "权重梯度形状应保持不变（与 batch 大小无关）"
    );

    Ok(())
}

/// 测试 MSE 的动态形状兼容性检查
#[test]
fn test_mse_loss_dynamic_shape_compatibility() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 两个输入都支持动态 batch
    let input = graph.input(&Tensor::zeros(&[4, 8]))?;
    let target = graph.input(&Tensor::zeros(&[4, 8]))?;

    // 验证动态形状兼容
    let input_dyn = input.dynamic_expected_shape();
    let target_dyn = target.dynamic_expected_shape();
    assert!(
        input_dyn.is_compatible(&target_dyn),
        "Input 和 Target 的动态形状应兼容"
    );

    // 创建 MSE 节点应该成功
    let loss = input.mse_loss(&target)?;
    loss.forward().unwrap();

    Ok(())
}

// ==================== 方案 C：新节点创建 API 测试 ====================

#[test]
fn test_create_mse_node() {
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

    let mse = inner
        .borrow_mut()
        .create_mse_mean_node(input.clone(), target.clone(), Some("mse"))
        .unwrap();

    // MSE 输出形状固定为 [1, 1]
    assert_eq!(mse.shape(), vec![1, 1]);
    assert_eq!(mse.name(), Some("mse"));
    assert!(!mse.is_leaf());
    assert_eq!(mse.parents().len(), 2);
}

#[test]
fn test_create_mse_node_shape_mismatch() {
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
        .create_mse_mean_node(input, target, None);

    assert!(result.is_err());
}

#[test]
fn test_create_mse_node_output_always_scalar() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 测试不同输入形状，输出始终为 [1, 1]
    let input_2d = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let target_2d = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let mse_2d = inner
        .borrow_mut()
        .create_mse_mean_node(input_2d, target_2d, None)
        .unwrap();
    assert_eq!(mse_2d.shape(), vec![1, 1]);

    let input_4d = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4, 5], None)
        .unwrap();
    let target_4d = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4, 5], None)
        .unwrap();
    let mse_4d = inner
        .borrow_mut()
        .create_mse_mean_node(input_4d, target_4d, None)
        .unwrap();
    assert_eq!(mse_4d.shape(), vec![1, 1]);
}

#[test]
fn test_create_mse_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_mse;
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

        let mse = inner
            .borrow_mut()
            .create_mse_mean_node(input, target, None)
            .unwrap();
        weak_mse = Rc::downgrade(&mse);

        assert!(weak_mse.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
        assert!(weak_target.upgrade().is_some());
    }
    assert!(weak_mse.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
    assert!(weak_target.upgrade().is_none());
}
