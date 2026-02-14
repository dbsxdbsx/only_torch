/*
 * @Author       : 老董
 * @Description  : BatchNormOp（批归一化运算）节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（训练模式）→ 2D [N,C] + 4D [N,C,H,W]
 * 2. 前向传播测试（评估模式）
 * 3. Running stats 更新验证
 * 4. VJP 单元测试
 * 5. 端到端反向传播测试
 * 6. Create API 测试
 *
 * Python 对照值 (PyTorch):
 *   见 test_batch_norm_reference.py
 *
 * 注意：BatchNormOp 只做归一化（不含 gamma/beta），
 *       gamma/beta 由 BatchNorm Layer 通过 Mul/Add 节点处理。
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（训练模式）====================

/// 测试 BatchNormOp 前向传播（2D 输入 [N, C]）
///
/// 输入 [4, 3]，训练模式：
///   每列均值 = [5.5, 6.5, 7.5]
///   每列方差 = [11.25, 11.25, 11.25]
///   归一化后值应接近 [-1.3416, -0.4472, 0.4472, 1.3416]
///
/// PyTorch 对照（gamma=1, beta=0 时 BatchNorm1d 等价于 BatchNormOp）：
///   [[-1.3416, -1.3416, -1.3416],
///    [-0.4472, -0.4472, -0.4472],
///    [ 0.4472,  0.4472,  0.4472],
///    [ 1.3416,  1.3416,  1.3416]]
#[test]
fn test_batch_norm_op_forward_2d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 3], Some("x"))
        .unwrap();
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(x.clone(), 1e-5, 0.1, Some("bn"))
        .unwrap();

    #[rustfmt::skip]
    x.set_value(Some(&Tensor::new(&[
        1., 2., 3.,
        4., 5., 6.,
        7., 8., 9.,
        10., 11., 12.,
    ], &[4, 3]))).unwrap();

    bn.forward_recursive(1, true).unwrap();

    let output = bn.value().unwrap();
    assert_eq!(output.shape(), &[4, 3]);

    // 第一列的归一化值
    assert_abs_diff_eq!(output[[0, 0]], -1.3416402, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[1, 0]], -0.4472134, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[2, 0]], 0.4472134, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[3, 0]], 1.3416402, epsilon = 1e-4);
}

/// 测试 BatchNormOp 前向传播（4D 输入 [N, C, H, W]）
///
/// 输入 [2, 2, 2, 2]，channel 0: 值 1-4,9-12, channel 1: 值 5-8,13-16
/// PyTorch 对照（gamma=1, beta=0）
#[test]
fn test_batch_norm_op_forward_4d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2, 2, 2], Some("x"))
        .unwrap();
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(x.clone(), 1e-5, 0.1, Some("bn"))
        .unwrap();

    let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    x.set_value(Some(&Tensor::new(&data, &[2, 2, 2, 2])))
        .unwrap();

    bn.forward_recursive(1, true).unwrap();

    let output = bn.value().unwrap();
    assert_eq!(output.shape(), &[2, 2, 2, 2]);

    // 第一个 channel 第一个元素（PyTorch 对照：-1.324244）
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], -1.324244, epsilon = 1e-3);
    // 最后一个 channel 最后一个元素（PyTorch 对照：1.324244）
    assert_abs_diff_eq!(output[[1, 1, 1, 1]], 1.324244, epsilon = 1e-3);
}

// ==================== 评估模式测试 ====================

/// 测试 BatchNormOp 评估模式
///
/// 先训练一次（更新 running stats），然后切换到评估模式
#[test]
fn test_batch_norm_op_eval_mode() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 3], Some("x"))
        .unwrap();
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(x.clone(), 1e-5, 0.1, Some("bn"))
        .unwrap();

    #[rustfmt::skip]
    x.set_value(Some(&Tensor::new(&[
        1., 2., 3.,
        4., 5., 6.,
        7., 8., 9.,
        10., 11., 12.,
    ], &[4, 3]))).unwrap();

    // 训练模式前向（更新 running stats）
    bn.forward_recursive(1, true).unwrap();

    // 切换到评估模式（forward_recursive is_training=false）
    bn.forward_recursive(2, false).unwrap();

    let output = bn.value().unwrap();
    assert_eq!(output.shape(), &[4, 3]);

    // 评估模式使用 running stats，输出不同于训练模式
    // 不需要精确对照，只需确认不是全 0 且形状正确
    let sum: f32 = output.data_as_slice().iter().sum();
    assert!(
        sum.abs() > 0.1,
        "评估模式输出不应全为 0（running stats 已更新）"
    );
}

// ==================== VJP 测试 ====================

/// 测试 BatchNormOp 训练模式 VJP
///
/// PyTorch 对照：BatchNorm1d(gamma=1, beta=0) 对 sum(y) 反向传播
/// x_grad 应全为 0（因为 sum(y) 的梯度均匀分布，BatchNorm 梯度相消）
#[test]
fn test_batch_norm_op_vjp_sum_grad() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 3], Some("x"))
        .unwrap();
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(x.clone(), 1e-5, 0.1, Some("bn"))
        .unwrap();

    #[rustfmt::skip]
    x.set_value(Some(&Tensor::new(&[
        1., 2., 3.,
        4., 5., 6.,
        7., 8., 9.,
        10., 11., 12.,
    ], &[4, 3]))).unwrap();

    bn.forward_recursive(1, true).unwrap();

    // upstream = 全 1（对应 sum loss）
    let upstream_grad = Tensor::ones(&[4, 3]);
    let grad = bn
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[4, 3]);
    // BatchNorm 的性质：当 upstream 为均匀值时，梯度为 0
    for &v in grad.data_as_slice() {
        assert_abs_diff_eq!(v, 0.0, epsilon = 1e-5);
    }

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 BatchNormOp 端到端反向传播
#[test]
fn test_batch_norm_op_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[4, 3], Init::Zeros, "x")?;
    #[rustfmt::skip]
    x.set_value(&Tensor::new(&[
        1., 2., 3.,
        4., 5., 6.,
        7., 8., 9.,
        10., 11., 12.,
    ], &[4, 3]))?;

    // 通过 graph inner 创建 BatchNormOp 节点
    let bn = {
        let node = graph.inner_mut().create_batch_norm_op_node(
            std::rc::Rc::clone(x.node()),
            1e-5,
            0.1,
            Some("bn"),
        )?;
        crate::nn::Var::new_with_rc_graph(node, &graph.inner_rc())
    };

    let target = graph.input(&Tensor::zeros(&[4, 3]))?;
    let loss = bn.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[4, 3]);

    Ok(())
}

// ==================== Create API 测试 ====================

#[test]
fn test_create_batch_norm_op_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 3], Some("input"))
        .unwrap();

    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(input.clone(), 1e-5, 0.1, Some("bn"))
        .unwrap();

    assert_eq!(bn.shape(), vec![4, 3]);
    assert_eq!(bn.name(), Some("bn"));
}

#[test]
fn test_create_batch_norm_op_node_4d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 16, 8, 8], None)
        .unwrap();

    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(input.clone(), 1e-5, 0.1, None)
        .unwrap();

    assert_eq!(bn.shape(), vec![2, 16, 8, 8]);
}

/// 测试 1D 形状应报错（BatchNormOp 至少需要 2 维 [N, C, ...]）
///
/// 注意：create_basic_input_node 本身不接受 1D，
/// 所以这里直接测试 BatchNormOp::new 的维度检查。
#[test]
fn test_batch_norm_op_rejects_1d_input() {
    use crate::nn::nodes::raw_node::BatchNormOp;
    use crate::nn::shape::DynamicShape;

    let result = BatchNormOp::new(&[10], &DynamicShape::new(&[Some(10)]), 1e-5, 0.1);
    assert!(result.is_err(), "1D 输入应被拒绝");
}
