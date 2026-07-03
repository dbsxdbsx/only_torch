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

use crate::nn::Mode;
use crate::nn::{Graph, GraphError, Init, VarLossOps, VarShapeOps};
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

    bn.forward_recursive(1, Mode::Train).unwrap();

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

    bn.forward_recursive(1, Mode::Train).unwrap();

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
    bn.forward_recursive(1, Mode::Train).unwrap();

    // 切换到推理上下文（eval 行为 + 不记录 backward 缓存）
    bn.forward_recursive(2, Mode::Inference).unwrap();

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

    bn.forward_recursive(1, Mode::Train).unwrap();

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

// ==================== 非连续内存（contiguity）回归测试 ====================

/// **回归测试**：BatchNorm 反向拿到非连续 `upstream_grad`（输出接 `permute`）不得 panic/算错。
/// 用 `mse(head, target)` 制造非均匀上游：`mse(bn(x),T)` vs `mse(permute(bn(x)),permute(T))`，
/// 匹配置换下 mse 不变 → loss 与 `x.grad` 逐元素一致（非均匀 upstream 能抓静默错序）。
#[test]
fn test_batch_norm_backward_noncontiguous_upstream() {
    let target_ref = Tensor::new(
        &[
            0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2,
        ],
        &[3, 4],
    );
    let target_perm = target_ref.permute(&[1, 0]).into_contiguous();
    fn run(permute_after: bool, target_ref: &Tensor, target_perm: &Tensor) -> Tensor {
        let graph = Graph::new();
        let x = graph.parameter(&[3, 4], Init::Zeros, "x").unwrap();
        x.set_value(&Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        ))
        .unwrap();
        let (rm, rv) = default_running_stats(4);
        let bn_node = graph
            .inner_mut()
            .create_batch_norm_op_node(Rc::clone(x.node()), 1e-5, 0.1, rm, rv, Some("bn"))
            .unwrap();
        let bn = crate::nn::Var::new_with_rc_graph(bn_node, &graph.inner_rc());
        let (head, target) = if permute_after {
            (
                bn.permute(&[1, 0]).unwrap(),
                graph.input(target_perm).unwrap(),
            )
        } else {
            (bn, graph.input(target_ref).unwrap())
        };
        let loss = head.mse_loss(&target).unwrap();
        graph.zero_grad().unwrap();
        loss.backward().unwrap();
        x.grad().unwrap().unwrap()
    }
    let g_ref = run(false, &target_ref, &target_perm);
    let g = run(true, &target_ref, &target_perm);
    assert_eq!(g.shape(), g_ref.shape());
    for (a, b) in g.to_vec().iter().zip(g_ref.to_vec().iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-5);
    }
}

/// **回归测试**：BatchNorm 前向拿到非连续输入（上游 `permute`）不得 panic/静默算错。
/// 两路 bn 看到**同一逻辑张量** [3,4]（一连续、一为 permute 视图），前向输出逐元素一致
/// （channel_mean/channel_var 曾用手写平铺索引，非连续会静默算错）。
#[test]
fn test_batch_norm_forward_noncontiguous_input() {
    let base = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[4, 3],
    );
    let permuted_contig = base.permute(&[1, 0]).into_contiguous(); // [3,4]，channels=4

    // 参考：leaf 直接为 [3,4] 连续张量
    let out_ref = {
        let graph = Graph::new();
        let inner = graph.inner_rc();
        let leaf = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], Some("leaf"))
            .unwrap();
        let (rm, rv) = default_running_stats(4);
        let bn = inner
            .borrow_mut()
            .create_batch_norm_op_node(leaf.clone(), 1e-5, 0.1, rm, rv, Some("bn"))
            .unwrap();
        leaf.set_value(Some(&permuted_contig)).unwrap();
        bn.forward_recursive(1, Mode::Train).unwrap();
        bn.value().unwrap().clone()
    };

    // 测试：leaf [4,3] → permute → [3,4]（非连续视图）→ bn
    let out_test = {
        let graph = Graph::new();
        let inner = graph.inner_rc();
        let leaf = inner
            .borrow_mut()
            .create_basic_input_node(&[4, 3], Some("leaf"))
            .unwrap();
        let bn_parent = inner
            .borrow_mut()
            .create_permute_node(leaf.clone(), &[1, 0], None)
            .unwrap();
        let (rm, rv) = default_running_stats(4);
        let bn = inner
            .borrow_mut()
            .create_batch_norm_op_node(bn_parent, 1e-5, 0.1, rm, rv, Some("bn"))
            .unwrap();
        leaf.set_value(Some(&base)).unwrap();
        bn.forward_recursive(1, Mode::Train).unwrap();
        bn.value().unwrap().clone()
    };

    assert_eq!(out_ref.shape(), out_test.shape());
    for (a, b) in out_ref.to_vec().iter().zip(out_test.to_vec().iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-5);
    }
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

#[test]
fn test_batch_norm_op_rejects_invalid_config() {
    use crate::nn::nodes::raw_node::BatchNormOp;
    use crate::nn::shape::DynamicShape;

    let (rm, rv) = default_running_stats(3);
    assert!(
        BatchNormOp::new(
            &[2, 3],
            &DynamicShape::new(&[Some(2), Some(3)]),
            0.0,
            0.1,
            Rc::clone(&rm),
            Rc::clone(&rv),
        )
        .is_err(),
        "eps<=0 应被拒绝"
    );
    assert!(
        BatchNormOp::new(
            &[2, 3],
            &DynamicShape::new(&[Some(2), Some(3)]),
            1e-5,
            1.5,
            Rc::clone(&rm),
            Rc::clone(&rv),
        )
        .is_err(),
        "momentum 超出 [0,1] 应被拒绝"
    );

    let bad_rm = Rc::new(RefCell::new(Tensor::zeros(&[2])));
    let bad_rv = Rc::new(RefCell::new(Tensor::ones(&[3])));
    assert!(
        BatchNormOp::new(
            &[2, 3],
            &DynamicShape::new(&[Some(2), Some(3)]),
            1e-5,
            0.1,
            bad_rm,
            bad_rv,
        )
        .is_err(),
        "running stats 长度不等于通道数应被拒绝"
    );
}

#[test]
fn test_batch_norm_op_train_rejects_single_value_per_channel() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("x"))
        .unwrap();
    let (rm, rv) = default_running_stats(3);
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(x.clone(), 1e-5, 0.1, rm, rv, Some("bn"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    let err = bn.forward_recursive(1, Mode::Train).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("至少需要 2 个值"),
        "错误信息应说明单 channel 样本数不足，实际：{msg}"
    );
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

        bn.forward_recursive((step + 1) as u64, Mode::Train)
            .unwrap();
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

    bn_eval.forward_recursive(100, Mode::Inference).unwrap();

    let eval_output = bn_eval.value().unwrap();
    assert_eq!(eval_output.shape(), &[4, 3]);

    // eval 模式应产生非零输出（使用积累的 running stats）
    let sum: f32 = eval_output.data_as_slice().iter().map(|v| v.abs()).sum();
    assert!(
        sum > 0.1,
        "eval 模式应使用积累的 running stats 产生非零输出"
    );
}

// ==================== 反向单趟融合 vs 旧表达式链逐 bit 金测试 ====================
//
// 2026-07-03 优化（优化候选 #7）：训练模式反向从张量表达式链
// `&(&(upstream * n) - &sum_up_t - &(x_hat * &sum_up_xh_t)) / &(std * n)`
// （~6 个全尺寸临时 + [1,C,1,1] 广播减/除慢路径）融合为单趟逐元素循环。
// 参考实现在测试内以与前向完全相同的统计量算法（flat 循环 mean/var、
// `(&var + eps).powf(0.5)`、张量链 x_hat）重建 x_hat/std，再走旧 dx 链，
// 从输入数据出发端到端锁死逐 bit 一致。

/// 与 BatchNormOp::channel_mean 逐 bit 相同的参考实现
fn bn_ref_channel_mean(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let ndim = shape.len();
    let c = shape[1];
    let spatial_size: usize = shape[2..].iter().product();
    let n = shape[0];
    let total = n * spatial_size;

    let flat = x.flatten_view();
    let mut means = vec![0.0f32; c];
    for sample in 0..n {
        for (ch, mean) in means.iter_mut().enumerate() {
            for s in 0..spatial_size {
                let idx = sample * c * spatial_size + ch * spatial_size + s;
                *mean += flat[idx];
            }
        }
    }
    for m in &mut means {
        *m /= total as f32;
    }
    let mut out_shape = vec![1usize; ndim];
    out_shape[1] = c;
    Tensor::new(&means, &out_shape)
}

/// 与 BatchNormOp::channel_var 逐 bit 相同的参考实现
fn bn_ref_channel_var(x: &Tensor, mean: &Tensor) -> Tensor {
    let shape = x.shape();
    let ndim = shape.len();
    let c = shape[1];
    let spatial_size: usize = shape[2..].iter().product();
    let n = shape[0];
    let total = n * spatial_size;

    let flat = x.flatten_view();
    let mean_flat = mean.flatten_view();
    let mut vars = vec![0.0f32; c];
    for sample in 0..n {
        for ch in 0..c {
            let m = mean_flat[ch];
            for s in 0..spatial_size {
                let idx = sample * c * spatial_size + ch * spatial_size + s;
                let diff = flat[idx] - m;
                vars[ch] += diff * diff;
            }
        }
    }
    for v in &mut vars {
        *v /= total as f32;
    }
    let mut out_shape = vec![1usize; ndim];
    out_shape[1] = c;
    Tensor::new(&vars, &out_shape)
}

/// 旧实现参考：dx 表达式链（升级前的原版反向）
fn bn_ref_backward_old_chain(x: &Tensor, upstream: &Tensor, eps: f32) -> Tensor {
    let shape = upstream.shape();
    let ndim = shape.len();
    let c = shape[1];
    let spatial_size: usize = shape[2..].iter().product();
    let batch_size = shape[0];
    let n = (batch_size * spatial_size.max(1)) as f32;

    // 重建与前向逐 bit 相同的 x_hat / std
    let mean = bn_ref_channel_mean(x);
    let var = bn_ref_channel_var(x, &mean);
    let std = (&var + eps).powf(0.5);
    let x_hat = &(x - &mean) / &std;

    // per-channel sum(upstream) / sum(upstream * x_hat)（与旧实现同序）
    let up_flat = upstream.flatten_view();
    let xh_flat = x_hat.flatten_view();
    let sp = spatial_size.max(1);
    let mut sum_up = vec![0.0f32; c];
    let mut sum_up_xh = vec![0.0f32; c];
    for sample in 0..batch_size {
        for ch in 0..c {
            for s in 0..sp {
                let idx = sample * c * sp + ch * sp + s;
                sum_up[ch] += up_flat[idx];
                sum_up_xh[ch] += up_flat[idx] * xh_flat[idx];
            }
        }
    }
    let mut bcast_shape = vec![1usize; ndim];
    bcast_shape[1] = c;
    let sum_up_t = Tensor::new(&sum_up, &bcast_shape);
    let sum_up_xh_t = Tensor::new(&sum_up_xh, &bcast_shape);

    // 旧版表达式链原样
    &(&(upstream * n) - &sum_up_t - &(&x_hat * &sum_up_xh_t)) / &(&std * n)
}

/// 手动档微对照：旧表达式链 vs 单趟融合的隔离耗时（不受全链路 bench 稀释）
///
/// 用法：`cargo test --release --lib --features blas-mkl -- --ignored bn_backward_micro --nocapture`
/// 两侧计时范围对齐（都含 per-channel sum 循环 + dx 计算），仅 dx 构造方式不同。
#[test]
#[ignore = "手动性能对照，需 --release 才有意义"]
fn bn_backward_micro_timing_old_chain_vs_fused() {
    use std::time::Instant;

    let shape: &[usize] = &[32, 16, 28, 28];
    let c = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let batch = shape[0];
    let n = (batch * spatial) as f32;
    let sp = spatial;

    let x_val = Tensor::normal_seeded(0.0, 1.0, shape, 42);
    let upstream = Tensor::normal_seeded(0.0, 1.0, shape, 43);

    // 统计量在计时外重建（两侧共享）
    let mean = bn_ref_channel_mean(&x_val);
    let var = bn_ref_channel_var(&x_val, &mean);
    let std = (&var + 1e-5f32).powf(0.5);
    let x_hat = &(&x_val - &mean) / &std;

    // 旧链：sum 循环 + 表达式链
    let old_run = || {
        let up_flat = upstream.flatten_view();
        let xh_flat = x_hat.flatten_view();
        let mut sum_up = vec![0.0f32; c];
        let mut sum_up_xh = vec![0.0f32; c];
        for sample in 0..batch {
            for ch in 0..c {
                for s in 0..sp {
                    let idx = sample * c * sp + ch * sp + s;
                    sum_up[ch] += up_flat[idx];
                    sum_up_xh[ch] += up_flat[idx] * xh_flat[idx];
                }
            }
        }
        let mut bcast_shape = vec![1usize; shape.len()];
        bcast_shape[1] = c;
        let sum_up_t = Tensor::new(&sum_up, &bcast_shape);
        let sum_up_xh_t = Tensor::new(&sum_up_xh, &bcast_shape);
        &(&(&upstream * n) - &sum_up_t - &(&x_hat * &sum_up_xh_t)) / &(&std * n)
    };

    // 新融合：走真实节点 backward（含相同的 sum 循环 + 单趟融合直写）
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let x = inner
        .borrow_mut()
        .create_basic_input_node(shape, Some("x"))
        .unwrap();
    let (rm, rv) = default_running_stats(c);
    let bn = inner
        .borrow_mut()
        .create_batch_norm_op_node(x.clone(), 1e-5, 0.1, rm, rv, Some("bn"))
        .unwrap();
    x.set_value(Some(&x_val)).unwrap();
    bn.forward_recursive(1, Mode::Train).unwrap();

    let new_run = || {
        bn.calc_grad_to_parent_index(0, &upstream)
            .unwrap()
            .resolve(&upstream)
    };

    let time_median_us = |f: &dyn Fn() -> Tensor| {
        let mut times: Vec<f64> = (0..30)
            .map(|_| {
                let t0 = Instant::now();
                let out = f();
                let dt = t0.elapsed().as_secs_f64() * 1e6;
                std::hint::black_box(out);
                dt
            })
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    };

    let old_us = time_median_us(&old_run);
    let new_us = time_median_us(&new_run);
    println!("BatchNorm 反向 dx 隔离对照（[32,16,28,28]，中位 30 次）：");
    println!("  旧表达式链: {old_us:.1} µs");
    println!("  单趟融合:   {new_us:.1} µs");
    println!("  加速比:     {:.2}x", old_us / new_us);
}

/// 单趟融合反向 vs 旧表达式链逐 bit（2D [N,C] 与 4D [N,C,H,W]）
#[test]
fn test_batch_norm_op_backward_bitwise_matches_old_chain() -> Result<(), GraphError> {
    let cases: &[&[usize]] = &[&[8, 5], &[4, 3, 6, 7], &[2, 16, 14, 14]];

    for (case_idx, &shape) in cases.iter().enumerate() {
        let graph = Graph::new();
        let inner = graph.inner_rc();

        let x = inner
            .borrow_mut()
            .create_basic_input_node(shape, Some("x"))?;
        let (rm, rv) = default_running_stats(shape[1]);
        let bn = inner.borrow_mut().create_batch_norm_op_node(
            x.clone(),
            1e-5,
            0.1,
            rm,
            rv,
            Some("bn"),
        )?;

        let x_val = Tensor::normal_seeded(0.0, 1.0, shape, 20 + case_idx as u64);
        x.set_value(Some(&x_val))?;
        bn.forward_recursive(1, Mode::Train)?;

        let upstream = Tensor::normal_seeded(0.0, 1.0, shape, 500 + case_idx as u64);
        let grad = bn
            .calc_grad_to_parent_index(0, &upstream)?
            .resolve(&upstream);

        let expected = bn_ref_backward_old_chain(&x_val, &upstream, 1e-5);
        assert_eq!(
            grad.shape(),
            expected.shape(),
            "case {case_idx}: 形状不一致"
        );
        let g = grad.data_as_slice();
        let e = expected.data_as_slice();
        for i in 0..g.len() {
            assert_eq!(
                g[i].to_bits(),
                e[i].to_bits(),
                "BatchNorm 反向融合与旧链不逐 bit 一致：case {case_idx} i={i} fused={} old={}",
                g[i],
                e[i]
            );
        }
    }

    Ok(())
}
