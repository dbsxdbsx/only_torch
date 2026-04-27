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
use std::cell::RefCell;
use std::rc::Rc;

/// 创建默认的共享 running stats（running_mean=0, running_var=1）
fn default_running_stats(num_features: usize) -> (Rc<RefCell<Tensor>>, Rc<RefCell<Tensor>>) {
    (
        Rc::new(RefCell::new(Tensor::zeros(&[num_features]))),
        Rc::new(RefCell::new(Tensor::ones(&[num_features]))),
    )
}

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
    let (rm, rv) = default_running_stats(3);
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(x.clone(), 1e-5, 0.1, rm, rv, Some("bn"))
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
    let (rm, rv) = default_running_stats(2);
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(x.clone(), 1e-5, 0.1, rm, rv, Some("bn"))
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
    let (rm, rv) = default_running_stats(3);
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(x.clone(), 1e-5, 0.1, rm, rv, Some("bn"))
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
    let (rm, rv) = default_running_stats(3);
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(x.clone(), 1e-5, 0.1, rm, rv, Some("bn"))
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
    let (rm, rv) = default_running_stats(3);
    let bn = {
        let node = graph.inner_mut().create_batch_norm_op_node(
            std::rc::Rc::clone(x.node()),
            1e-5,
            0.1,
            rm,
            rv,
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

    let (rm, rv) = default_running_stats(3);
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(input.clone(), 1e-5, 0.1, rm, rv, Some("bn"))
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

    let (rm, rv) = default_running_stats(16);
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(input.clone(), 1e-5, 0.1, rm, rv, None)
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

    let (rm, rv) = default_running_stats(10);
    let result = BatchNormOp::new(&[10], &DynamicShape::new(&[Some(10)]), 1e-5, 0.1, rm, rv);
    assert!(result.is_err(), "1D 输入应被拒绝");
}

// ==================== 跨 forward 调用 running stats 持久化测试 ====================

/// 测试 running stats 通过 Rc<RefCell> 在多个 BatchNormOp 节点间共享
///
/// 模拟实际训练场景：多次 forward 创建新节点，running stats 应持续累积。
/// 最后在 eval 模式下用新节点验证 running stats 已被正确积累。
#[test]
fn test_batch_norm_op_running_stats_persist_across_forwards() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 共享 running stats
    let (rm, rv) = default_running_stats(3);

    // 模拟多次 forward（每次创建新的 BatchNormOp 节点，共享同一 running stats）
    for step in 0..5 {
        let x = inner
            .borrow_mut()
            .create_basic_input_node(&[4, 3], None)
            .unwrap();
        let bn = inner
            .borrow_mut()
            .create_batch_norm_op_node(x.clone(), 1e-5, 0.1, Rc::clone(&rm), Rc::clone(&rv), None)
            .unwrap();

        #[rustfmt::skip]
        x.set_value(Some(&Tensor::new(&[
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.,
            10., 11., 12.,
        ], &[4, 3]))).unwrap();

        bn.forward_recursive((step + 1) as u64, true).unwrap();
    }

    // 验证 running stats 已从默认值改变
    let rm_val = rm.borrow();
    let rv_val = rv.borrow();

    // running_mean 应不再是全 0（初始值）
    let rm_sum: f32 = rm_val.data_as_slice().iter().map(|v| v.abs()).sum();
    assert!(rm_sum > 0.1, "running_mean 应已从 0 更新: {rm_sum}");

    // running_var 应不再是全 1（初始值）
    // 经过多次 EMA 更新，var 会偏离 1.0
    let rv_data = rv_val.data_as_slice();
    let all_one = rv_data.iter().all(|&v| (v - 1.0).abs() < 1e-6);
    assert!(!all_one, "running_var 应已从 1.0 更新");

    // 用新节点在 eval 模式验证 running stats 正确使用
    let x_eval = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 3], None)
        .unwrap();
    drop(rm_val);
    drop(rv_val);
    let bn_eval = inner
        .borrow_mut()
        .create_batch_norm_op_node(
            x_eval.clone(),
            1e-5,
            0.1,
            Rc::clone(&rm),
            Rc::clone(&rv),
            None,
        )
        .unwrap();

    #[rustfmt::skip]
    x_eval.set_value(Some(&Tensor::new(&[
        1., 2., 3.,
        4., 5., 6.,
        7., 8., 9.,
        10., 11., 12.,
    ], &[4, 3]))).unwrap();

    bn_eval.forward_recursive(100, false).unwrap();

    let eval_output = bn_eval.value().unwrap();
    assert_eq!(eval_output.shape(), &[4, 3]);

    // eval 模式应产生非零输出（使用积累的 running stats）
    let sum: f32 = eval_output.data_as_slice().iter().map(|v| v.abs()).sum();
    assert!(
        sum > 0.1,
        "eval 模式应使用积累的 running stats 产生非零输出"
    );
}
