/*
 * @Author       : 老董
 * @Description  : BatchNorm 层单元测试（含 gamma/beta 参数）
 *
 * 测试策略：
 * 1. 前向传播测试（训练模式）→ 默认 gamma=1, beta=0 应与 BatchNormOp 一致
 * 2. 参数收集测试（Module trait）
 * 3. 端到端反向传播 + 参数梯度
 * 4. train/eval 模式切换
 *
 * PyTorch 对照值：
 *   BatchNorm1d(3) 初始 gamma=1, beta=0 → 输出与纯归一化一致
 *   反向传播：gamma_grad 和 beta_grad 应有值
 */

use crate::nn::{BatchNorm, Graph, GraphError, Module, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试 ====================

/// 测试 BatchNorm 层前向传播（gamma=1, beta=0）
///
/// 初始化时 gamma=1, beta=0，所以输出应与 BatchNormOp 完全一致
/// PyTorch 对照：
///   [[-1.3416, -1.3416, -1.3416],
///    [-0.4472, -0.4472, -0.4472],
///    [ 0.4472,  0.4472,  0.4472],
///    [ 1.3416,  1.3416,  1.3416]]
#[test]
fn test_batch_norm_layer_forward_2d() -> Result<(), GraphError> {
    let graph = Graph::new();

    let bn = BatchNorm::new(&graph, 3, 1e-5, 0.1, "bn")?;

    #[rustfmt::skip]
    let x = graph.input(&Tensor::new(&[
        1., 2., 3.,
        4., 5., 6.,
        7., 8., 9.,
        10., 11., 12.,
    ], &[4, 3]))?;

    let y = bn.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[4, 3]);

    // gamma=1, beta=0 → 输出与纯归一化一致
    assert_abs_diff_eq!(output[[0, 0]], -1.3416402, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[1, 0]], -0.4472134, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[2, 0]], 0.4472134, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[3, 0]], 1.3416402, epsilon = 1e-3);

    Ok(())
}

/// 测试 BatchNorm 层前向传播（4D CNN 输入）
#[test]
fn test_batch_norm_layer_forward_4d() -> Result<(), GraphError> {
    let graph = Graph::new();

    let bn = BatchNorm::new(&graph, 2, 1e-5, 0.1, "bn2d")?;

    let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[2, 2, 2, 2]))?;

    let y = bn.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 2, 2, 2]);

    // gamma=1, beta=0 → 应与 BatchNormOp 一致
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], -1.324244, epsilon = 1e-3);

    Ok(())
}

// ==================== 参数测试 ====================

/// 测试 Module::parameters() 返回 gamma 和 beta
#[test]
fn test_batch_norm_layer_parameters() -> Result<(), GraphError> {
    let graph = Graph::new();

    let bn = BatchNorm::new(&graph, 16, 1e-5, 0.1, "bn")?;
    let params = bn.parameters();

    assert_eq!(params.len(), 2, "应有 gamma 和 beta 两个参数");

    // gamma: [1, 16]，初始值全 1
    let gamma_val = params[0].value()?.unwrap();
    assert_eq!(gamma_val.shape(), &[1, 16]);
    assert_abs_diff_eq!(gamma_val[[0, 0]], 1.0, epsilon = 1e-7);

    // beta: [1, 16]，初始值全 0
    let beta_val = params[1].value()?.unwrap();
    assert_eq!(beta_val.shape(), &[1, 16]);
    assert_abs_diff_eq!(beta_val[[0, 0]], 0.0, epsilon = 1e-7);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 BatchNorm 层端到端反向传播（gamma/beta 应有梯度）
///
/// PyTorch 对照：
///   gamma_grad = [0, 0, 0]（因为 sum(x_hat)=0，全局和为 0）
///   beta_grad = [4, 4, 4]（每个 channel 有 4 个样本的 upstream=1）
#[test]
fn test_batch_norm_layer_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let bn = BatchNorm::new(&graph, 3, 1e-5, 0.1, "bn")?;

    #[rustfmt::skip]
    let x = graph.input(&Tensor::new(&[
        1., 2., 3.,
        4., 5., 6.,
        7., 8., 9.,
        10., 11., 12.,
    ], &[4, 3]))?;

    let y = bn.forward(&x);

    // 用 sum loss（每个元素梯度为 1）
    let target = graph.input(&Tensor::zeros(&[4, 3]))?;
    let loss = y.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    // gamma 和 beta 应有梯度
    let gamma_grad = bn.gamma().grad()?.expect("gamma 应有 grad");
    assert_eq!(gamma_grad.shape(), &[1, 3]);

    let beta_grad = bn.beta().grad()?.expect("beta 应有 grad");
    assert_eq!(beta_grad.shape(), &[1, 3]);

    Ok(())
}

// ==================== 4D 非对称维度测试 ====================

/// 测试 BatchNorm 4D 非对称维度前向传播 [N=2, C=3, H=4, W=5]
///
/// 这个测试确保 gamma/beta 的 [1,C] reshape 到 [1,C,1,1] 后能正确广播。
/// 之前的 [2,2,2,2] 测试因为所有维度相同，无法捕捉广播对齐错误。
#[test]
fn test_batch_norm_layer_forward_4d_nonsquare() -> Result<(), GraphError> {
    let graph = Graph::new();

    let bn = BatchNorm::new(&graph, 3, 1e-5, 0.1, "bn_ns")?;

    // [2, 3, 4, 5] = 120 个元素
    let data: Vec<f32> = (1..=120).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[2, 3, 4, 5]))?;

    let y = bn.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 3, 4, 5]);

    // 均值不为 0（确认确实计算了）
    let first = output[[0, 0, 0, 0]];
    assert!(first.abs() > 0.1, "4D 非对称输出不应该为 0: {first}");

    Ok(())
}

/// 测试 BatchNorm 4D 非对称维度反向传播，确认 gamma/beta 梯度正确
#[test]
fn test_batch_norm_layer_backward_4d_nonsquare() -> Result<(), GraphError> {
    let graph = Graph::new();

    let bn = BatchNorm::new(&graph, 3, 1e-5, 0.1, "bn_ns")?;

    let data: Vec<f32> = (1..=120).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[2, 3, 4, 5]))?;

    let y = bn.forward(&x);
    let target = graph.input(&Tensor::ones(&[2, 3, 4, 5]))?;
    let loss = y.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val > 0.0, "loss 应为正数");

    // gamma 和 beta 都应有梯度，形状为 [1, 3]
    let gamma_grad = bn.gamma().grad()?.expect("gamma 应有 grad");
    assert_eq!(gamma_grad.shape(), &[1, 3]);
    // gamma_grad 不应全为 0
    let gamma_grad_sum: f32 = gamma_grad.data_as_slice().iter().map(|v| v.abs()).sum();
    assert!(gamma_grad_sum > 1e-6, "gamma_grad 不应全为 0");

    let beta_grad = bn.beta().grad()?.expect("beta 应有 grad");
    assert_eq!(beta_grad.shape(), &[1, 3]);
    let beta_grad_sum: f32 = beta_grad.data_as_slice().iter().map(|v| v.abs()).sum();
    assert!(beta_grad_sum > 1e-6, "beta_grad 不应全为 0");

    Ok(())
}

// ==================== Train/Eval 切换测试 ====================

/// 测试 train/eval 模式切换
#[test]
fn test_batch_norm_layer_train_eval_switch() -> Result<(), GraphError> {
    let graph = Graph::new();

    let bn = BatchNorm::new(&graph, 3, 1e-5, 0.1, "bn")?;

    #[rustfmt::skip]
    let x = graph.input(&Tensor::new(&[
        1., 2., 3.,
        4., 5., 6.,
        7., 8., 9.,
        10., 11., 12.,
    ], &[4, 3]))?;

    let y = bn.forward(&x);

    // 训练模式
    graph.train();
    y.forward()?;
    let train_output = y.value()?.unwrap().clone();

    // 评估模式
    graph.eval();
    y.forward()?;
    let eval_output = y.value()?.unwrap();

    // 两种模式的输出应不同（running stats != batch stats）
    assert_eq!(train_output.shape(), eval_output.shape());
    // 至少有一个元素不同
    let diff: f32 = train_output
        .data_as_slice()
        .iter()
        .zip(eval_output.data_as_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.1, "训练模式和评估模式的输出应不同");

    Ok(())
}

/// 测试 running stats 跨多次 forward 调用持久化（layer 级别）
///
/// 模拟实际训练场景：多次调用 bn.forward() 后切换 eval 模式，
/// eval 应使用训练期间累积的 running stats，而不是默认值。
///
/// 这个测试能捕捉到之前的 bug：每次 forward 创建新 BatchNormOp 节点，
/// running stats 丢失导致 eval 用默认值。
#[test]
fn test_batch_norm_layer_running_stats_across_forwards() -> Result<(), GraphError> {
    let graph = Graph::new();
    let bn = BatchNorm::new(&graph, 3, 1e-5, 0.1, "bn")?;

    // 训练模式：多次 forward（模拟训练循环中每个 batch 创建新图）
    graph.train();
    for _ in 0..10 {
        let x = graph.input(&Tensor::new(
            &(1..=12).map(|v| v as f32).collect::<Vec<_>>(),
            &[4, 3],
        ))?;
        let y = bn.forward(&x);
        y.forward()?;
    }

    // eval 模式：用新输入 forward
    graph.eval();
    let x_eval = graph.input(&Tensor::new(
        &(1..=12).map(|v| v as f32).collect::<Vec<_>>(),
        &[4, 3],
    ))?;
    let y_eval = bn.forward(&x_eval);
    y_eval.forward()?;
    let eval_output = y_eval.value()?.unwrap();

    // 如果 running stats 正确累积，eval 输出应与 train 输出有差异
    // （running_mean 不是全 0，running_var 不是全 1）
    //
    // 如果 running stats 丢失（用默认 mean=0, var=1），
    // 则 eval 输出≈ gamma * x + beta = x（因为 gamma=1, beta=0）
    // 第一个元素是原始值 1.0。
    //
    // 如果 running stats 正确，eval 输出应是归一化后的值，和 1.0 有显著差异。
    let first_val = eval_output[[0, 0]];
    assert!(
        (first_val - 1.0).abs() > 0.1,
        "eval 输出不应接近原始值 1.0（说明 running stats 丢失），实际值: {first_val}"
    );

    Ok(())
}
