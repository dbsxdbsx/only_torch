/*
 * @Author       : 老董
 * @Description  : MAE（Mean Absolute Error）损失节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ basic mean=0.5; 2D matrix mean=0.5; 混合正负 1.167; cannot_set_value
 * 2. VJP 单元测试（底层）→ Mean VJP; Sum VJP
 * 3. 端到端反向传播测试（高层）→ mean/2D/mixed
 * 4. 梯度累积测试
 * 5. 动态形状测试
 * 6. 新节点创建 API 测试（KEEP AS-IS）
 */

use crate::nn::{Graph, GraphError, Init, Reduction, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.L1Loss(reduction='mean')(input, target)
/// # loss = mean(|[-0.5, -0.5, -0.5]|) = 0.5
/// ```
#[test]
fn test_mae_forward_mean_basic() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3]))
        .unwrap();
    let loss = input.mae_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.value().unwrap().unwrap();
    // 验证损失值: mean(|0.5| + |0.5| + |0.5|) = 0.5
    assert_abs_diff_eq!(loss_val[[0, 0]], 0.5, epsilon = 1e-6);
}

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
/// loss = nn.L1Loss(reduction='mean')(input, target)
/// # loss = 0.5
/// ```
#[test]
fn test_mae_forward_2d_matrix() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.5, 2.5, 3.5, 4.5], &[2, 2]))
        .unwrap();
    let loss = input.mae_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.value().unwrap().unwrap();
    // 4 个元素，每个 abs_diff = 0.5
    // sum = 2.0, mean = 0.5
    assert_abs_diff_eq!(loss_val[[0, 0]], 0.5, epsilon = 1e-6);
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
fn test_mae_forward_mixed_diff() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1.0, 3.0, 2.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[2.0, 1.0, 2.5], &[1, 3]))
        .unwrap();
    let loss = input.mae_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.value().unwrap().unwrap();
    // PyTorch: 1.1666666269302368
    assert_abs_diff_eq!(loss_val[[0, 0]], 1.166_666_6, epsilon = 1e-5);
}

/// 测试批量输入的前向传播
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
fn test_mae_forward_batch() {
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
    let loss = input.mae_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.value().unwrap().unwrap();
    // PyTorch: 0.10833333432674408
    assert_abs_diff_eq!(loss_val[[0, 0]], 0.108_333_33, epsilon = 1e-5);
}

/// 测试大数值输入的数值稳定性
#[test]
fn test_mae_forward_large_values() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1000.0, 2000.0, 3000.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1000.5, 2000.5, 3000.5], &[1, 3]))
        .unwrap();
    let loss = input.mae_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.value().unwrap().unwrap();
    // 同样是 diff = 0.5，所以 loss = 0.5
    assert_abs_diff_eq!(loss_val[[0, 0]], 0.5, epsilon = 1e-6);
}

/// 测试小数值输入的数值稳定性
#[test]
fn test_mae_forward_small_values() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[0.001, 0.002, 0.003], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.0015, 0.0025, 0.0035], &[1, 3]))
        .unwrap();
    let loss = input.mae_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.value().unwrap().unwrap();
    // mean(|0.0005| * 3) = 0.0005
    assert_abs_diff_eq!(loss_val[[0, 0]], 0.0005, epsilon = 1e-8);
}

/// 测试 MAE 节点不能直接设置值
#[test]
fn test_mae_cannot_set_value() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3]))
        .unwrap();
    let loss = input.mae_loss(&target).unwrap();

    let test_value = Tensor::new(&[0.5], &[1, 1]);
    let err = loss.set_value(&test_value);
    assert!(err.is_err(), "MAE 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证梯度计算公式。
// MAE VJP: grad = sign(input - target) / N (Mean) 或 sign(input - target) (Sum)

/// 测试 MAE Mean VJP
///
/// input = [1,2,3], target = [1.5,2.5,3.5]
/// diff = [-0.5, -0.5, -0.5], sign = [-1, -1, -1]
/// Mean grad = sign(diff) / N = [-1/3, -1/3, -1/3]
#[test]
fn test_mae_vjp_mean() -> Result<(), GraphError> {
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
    let mae = inner
        .borrow_mut()
        .create_mae_mean_node(input.clone(), target.clone(), Some("mae"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3])))
        .unwrap();
    mae.forward_recursive(1, false).unwrap();

    // 验证前向值: mean(|0.5|*3) = 0.5
    let mae_val = mae.value().expect("MAE 应有值");
    assert_abs_diff_eq!(mae_val[[0, 0]], 0.5, epsilon = 1e-6);

    // VJP: upstream = 1.0 → grad = sign(diff) / N = [-1/3, -1/3, -1/3]
    let upstream = Tensor::ones(&[1, 1]);
    let grad = mae.calc_grad_to_parent_index(0, &upstream)?;

    assert_eq!(grad.shape(), &[1, 3]);
    let expected_grad = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);
    assert_abs_diff_eq!(&grad, &expected_grad, epsilon = 1e-5);

    Ok(())
}

/// 测试 MAE Sum VJP
///
/// input = [1,2,3], target = [1.5,2.5,3.5]
/// diff = [-0.5, -0.5, -0.5], sign = [-1, -1, -1]
/// Sum forward = sum(|diff|) = 1.5
/// Sum grad = sign(diff) = [-1, -1, -1]
#[test]
fn test_mae_vjp_sum() -> Result<(), GraphError> {
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
    let mae = inner
        .borrow_mut()
        .create_mae_node(input.clone(), target.clone(), Reduction::Sum, Some("mae"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3])))
        .unwrap();
    mae.forward_recursive(1, false).unwrap();

    // 验证前向值: sum(|0.5|*3) = 1.5
    let mae_val = mae.value().expect("MAE Sum 应有值");
    assert_abs_diff_eq!(mae_val[[0, 0]], 1.5, epsilon = 1e-6);

    // VJP: upstream = 1.0 → grad = sign(diff) = [-1, -1, -1]
    let upstream = Tensor::ones(&[1, 1]);
    let grad = mae.calc_grad_to_parent_index(0, &upstream)?;

    assert_eq!(grad.shape(), &[1, 3]);
    let expected_grad = Tensor::new(&[-1.0, -1.0, -1.0], &[1, 3]);
    assert_abs_diff_eq!(&grad, &expected_grad, epsilon = 1e-6);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5, 3.5]])
/// loss = nn.L1Loss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = sign(input - target) / N = sign([-0.5, -0.5, -0.5]) / 3 = [-0.333..., -0.333..., -0.333...]
/// ```
#[test]
fn test_mae_backward_e2e_mean() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input = graph.parameter(&[1, 3], Init::Zeros, "input")?;
    input.set_value(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))?;
    let target = graph.input(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3]))?;
    let loss = input.mae_loss(&target)?;

    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;

    let grad = input.grad()?.expect("应有 grad");
    // 预期梯度: sign(input - target) / N = sign([-0.5, -0.5, -0.5]) / 3 = [-1, -1, -1] / 3
    let expected_grad = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);
    assert_abs_diff_eq!(&grad, &expected_grad, epsilon = 1e-5);

    Ok(())
}

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
/// target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
/// loss = nn.L1Loss(reduction='mean')(input, target)
/// loss.backward()
/// # grad = sign([-0.5]) / 4 = -0.25
/// ```
#[test]
fn test_mae_backward_e2e_2d() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input = graph.parameter(&[2, 2], Init::Zeros, "input")?;
    input.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    let target = graph.input(&Tensor::new(&[1.5, 2.5, 3.5, 4.5], &[2, 2]))?;
    let loss = input.mae_loss(&target)?;

    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;

    let grad = input.grad()?.expect("应有 grad");
    let expected_grad = Tensor::new(&[-0.25, -0.25, -0.25, -0.25], &[2, 2]);
    assert_abs_diff_eq!(&grad, &expected_grad, epsilon = 1e-6);

    Ok(())
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
fn test_mae_backward_e2e_mixed() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input = graph.parameter(&[1, 3], Init::Zeros, "input")?;
    input.set_value(&Tensor::new(&[1.0, 3.0, 2.0], &[1, 3]))?;
    let target = graph.input(&Tensor::new(&[2.0, 1.0, 2.5], &[1, 3]))?;
    let loss = input.mae_loss(&target)?;

    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;

    let grad = input.grad()?.expect("应有 grad");
    // sign([-1.0, 2.0, -0.5]) / 3 = [-1, 1, -1] / 3
    let expected_grad = Tensor::new(&[-0.333_333_34, 0.333_333_34, -0.333_333_34], &[1, 3]);
    assert_abs_diff_eq!(&grad, &expected_grad, epsilon = 1e-5);

    Ok(())
}

/// 批量输入的反向传播测试
#[test]
fn test_mae_backward_e2e_batch() -> Result<(), GraphError> {
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

    let input = graph.parameter(&[3, 4], Init::Zeros, "input")?;
    input.set_value(&input_data)?;
    let target = graph.input(&target_data)?;
    let loss = input.mae_loss(&target)?;

    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;

    let grad = input.grad()?.expect("应有 grad");

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

    assert_abs_diff_eq!(&grad, &expected_grad, epsilon = 1e-5);

    Ok(())
}

/// 简单的回归训练测试
/// 目标: y = 2x（学习斜率）
#[test]
fn test_mae_simple_regression() {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let w = graph.parameter(&[1, 1], Init::Zeros, "w").unwrap();
    w.set_value(&Tensor::new(&[0.5], &[1, 1])).unwrap();
    let y_pred = x.matmul(&w).unwrap();
    let y_true = graph.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();
    let loss = y_pred.mae_loss(&y_true).unwrap();

    // 初始化权重为 0.5（目标是 2.0）
    let lr = 0.1;

    // 训练数据: x=1 -> y=2, x=2 -> y=4, x=3 -> y=6
    let training_data = [(1.0_f32, 2.0_f32), (2.0, 4.0), (3.0, 6.0)];

    // 训练 100 个 epoch（MAE 梯度恒定，可能需要更多迭代）
    for _ in 0..100 {
        for &(x_val, y_val) in &training_data {
            x.set_value(&Tensor::new(&[x_val], &[1, 1])).unwrap();
            y_true.set_value(&Tensor::new(&[y_val], &[1, 1])).unwrap();

            graph.zero_grad().unwrap();
            loss.forward().unwrap();
            loss.backward().unwrap();

            // 手动 SGD 更新：w = w - lr * grad
            let w_val = w.value().unwrap().unwrap();
            let w_grad = w.grad().unwrap().unwrap();
            let new_w = w_val - lr * &w_grad;
            w.set_value(&new_w).unwrap();
        }
    }

    // 验证学习到的权重接近 2.0
    let learned_w = w.value().unwrap().unwrap();
    assert_abs_diff_eq!(learned_w[[0, 0]], 2.0, epsilon = 0.1);
}

/// 验证 MAE 和 MSE 在相同输入下的不同输出
#[test]
fn test_mae_vs_mse_comparison() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3]))
        .unwrap();

    let mae = input.mae_loss(&target).unwrap();
    let mse = input.mse_loss(&target).unwrap();

    mae.forward().unwrap();
    mse.forward().unwrap();

    let mae_val = mae.value().unwrap().unwrap();
    let mse_val = mse.value().unwrap().unwrap();

    // MAE = mean(|0.5|) = 0.5
    // MSE = mean(0.5^2) = 0.25
    assert_abs_diff_eq!(mae_val[[0, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(mse_val[[0, 0]], 0.25, epsilon = 1e-6);

    // MAE > MSE when |diff| < 1
    assert!(mae_val[[0, 0]] > mse_val[[0, 0]]);
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试 MAE 梯度累积
#[test]
fn test_mae_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let input = graph.parameter(&[1, 3], Init::Zeros, "input")?;
    input.set_value(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))?;
    let target = graph.input(&Tensor::new(&[1.5, 2.5, 3.5], &[1, 3]))?;
    let loss = input.mae_loss(&target)?;

    loss.forward().unwrap();

    // 第 1 次反向传播
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = input.grad()?.unwrap().clone();
    let expected = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);
    assert_abs_diff_eq!(&grad_first, &expected, epsilon = 1e-5);

    // 第 2 次反向传播（梯度累积）
    loss.backward()?;
    let grad_second = input.grad()?.unwrap();
    assert_eq!(&grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    loss.backward()?;
    let grad_after_clear = input.grad()?.unwrap();
    assert_eq!(&grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 MAE 节点的动态形状（输出固定为标量 [1, 1]）
#[test]
fn test_mae_dynamic_shape_output_fixed() {
    let graph = Graph::new();

    let input = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let target = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let loss = input.mae_loss(&target).unwrap();

    // MAE 输出形状始终是 [1, 1]（标量）
    let dyn_shape = loss.dynamic_expected_shape();

    // 输出形状固定
    assert!(!dyn_shape.is_dynamic(0), "MAE 输出维度 0 应固定");
    assert!(!dyn_shape.is_dynamic(1), "MAE 输出维度 1 应固定");
    assert_eq!(dyn_shape.dim(0), Some(1));
    assert_eq!(dyn_shape.dim(1), Some(1));
}

/// 测试 MAE 接受动态 batch 输入
#[test]
fn test_mae_dynamic_batch_forward() {
    let graph = Graph::new();

    let input = graph.input(&Tensor::ones(&[2, 4])).unwrap();
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = input.mae_loss(&target).unwrap();

    // 第一次 forward：batch=2
    loss.forward().unwrap();
    let loss_val1 = loss.value().unwrap().unwrap();
    assert_eq!(loss_val1.shape(), &[1, 1], "MAE 输出应为标量");
    assert!(loss_val1[[0, 0]] > 0.0);

    // 第二次 forward：batch=6（不同 batch 大小）
    input.set_value(&Tensor::ones(&[6, 4])).unwrap();
    target.set_value(&Tensor::zeros(&[6, 4])).unwrap();
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap();
    assert_eq!(loss_val2.shape(), &[1, 1], "MAE 输出应始终为标量");
}

/// 测试 MAE 在不同 batch 大小下的反向传播
#[test]
fn test_mae_dynamic_batch_backward() {
    let graph = Graph::new();

    // y = input @ weight 形式，weight 是可训练 Parameter
    let input = graph.input(&Tensor::ones(&[2, 4])).unwrap();
    let weight = graph.parameter(&[4, 4], Init::Zeros, "weight").unwrap();
    weight.set_value(&Tensor::ones(&[4, 4])).unwrap();
    let pred = input.matmul(&weight).unwrap();
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = pred.mae_loss(&target).unwrap();

    // 第一次训练：batch=2
    loss.forward().unwrap();
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    let grad1 = weight.grad().unwrap().unwrap().clone();
    assert_eq!(grad1.shape(), &[4, 4], "权重梯度形状应保持不变");

    // 更新为不同 batch
    input.set_value(&Tensor::ones(&[5, 4])).unwrap();
    target.set_value(&Tensor::zeros(&[5, 4])).unwrap();

    // 第二次训练：batch=5
    loss.forward().unwrap();
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    let grad2 = weight.grad().unwrap().unwrap();
    assert_eq!(
        grad2.shape(),
        &[4, 4],
        "权重梯度形状应保持不变（与 batch 大小无关）"
    );
}

/// 测试 MAE 的动态形状兼容性检查
///
/// MAE 验证 input 和 target 的动态形状兼容性
#[test]
fn test_mae_dynamic_shape_compatibility() {
    let graph = Graph::new();

    // 两个输入都支持动态 batch
    let input = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let target = graph.input(&Tensor::zeros(&[4, 8])).unwrap();

    // 验证动态形状兼容
    let input_dyn = input.dynamic_expected_shape();
    let target_dyn = target.dynamic_expected_shape();

    assert!(
        input_dyn.is_compatible(&target_dyn),
        "Input 和 Target 的动态形状应兼容"
    );

    // 创建 MAE 节点应该成功
    let loss = input.mae_loss(&target).unwrap();
    loss.forward().unwrap();
    let loss_val = loss.value().unwrap().unwrap();
    assert_eq!(loss_val.shape(), &[1, 1]);
}

// ==================== 节点创建 API 测试 ====================

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

    let result = inner.borrow_mut().create_mae_mean_node(input, target, None);

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
