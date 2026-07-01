/*
 * @Author       : 老董
 * @Description  : LayerNormOp（层归一化运算）节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试 — 2D [N, D] + 3D [N, T, D]
 * 2. VJP 单元测试（calc_grad_to_parent_index）
 * 3. 端到端反向传播测试（高层 Var API）
 * 4. 节点元数据测试
 *
 * Python 对照值 (PyTorch):
 *   见 tests/layer_norm_reference.py
 */

use crate::nn::Mode;
use crate::nn::{Graph, GraphError, Init, VarLossOps, VarReduceOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试 ====================

/// 测试 LayerNormOp 前向传播（2D 输入 [N, D]）
///
/// PyTorch 对照（LayerNorm(3, elementwise_affine=False)）
#[test]
fn test_layer_norm_op_forward_2d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let ln = inner
        .borrow_mut()
        .create_layer_norm_op_node(x.clone(), 1, 1e-5, Some("ln"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))
        .unwrap();

    ln.forward_recursive(1, Mode::Train).unwrap();
    let val = ln.value().unwrap();

    assert_eq!(val.shape(), &[2, 3]);
    assert_abs_diff_eq!(val[[0, 0]], -1.2247, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[0, 1]], 0.0, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[0, 2]], 1.2247, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[1, 0]], -1.2247, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[1, 1]], 0.0, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[1, 2]], 1.2247, epsilon = 1e-3);
}

/// 测试 LayerNormOp 前向传播（3D 输入 [N, T, D]）
#[test]
fn test_layer_norm_op_forward_3d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2, 4], Some("x"))
        .unwrap();
    let ln = inner
        .borrow_mut()
        .create_layer_norm_op_node(x.clone(), 1, 1e-5, Some("ln"))
        .unwrap();

    #[rustfmt::skip]
    x.set_value(Some(&Tensor::new(&[
        1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,
        0.1, 0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8,
    ], &[2, 2, 4]))).unwrap();

    ln.forward_recursive(1, Mode::Train).unwrap();
    let val = ln.value().unwrap();

    assert_eq!(val.shape(), &[2, 2, 4]);
    assert_abs_diff_eq!(val[[0, 0, 0]], -1.3416, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[0, 0, 1]], -0.4472, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[0, 0, 2]], 0.4472, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[0, 0, 3]], 1.3416, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[1, 0, 0]], -1.3411, epsilon = 1e-3);
}

// ==================== VJP 测试 ====================

/// 测试 LayerNormOp VJP（sum loss → upstream 全 1 → 梯度近零）
#[test]
fn test_layer_norm_op_vjp_sum() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let ln = inner
        .borrow_mut()
        .create_layer_norm_op_node(x.clone(), 1, 1e-5, Some("ln"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))
        .unwrap();

    ln.forward_recursive(1, Mode::Train).unwrap();

    // upstream = 全 1（对应 sum loss）
    let upstream_grad = Tensor::ones(&[2, 3]);
    let grad = ln
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_abs_diff_eq!(grad[[i, j]], 0.0, epsilon = 1e-4);
        }
    }

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 端到端反向传播（高层 Var API）
#[test]
fn test_layer_norm_op_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 4], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4],
        &[2, 4],
    ))?;

    // LayerNormOp + mse_loss
    let ln_out = {
        use std::rc::Rc;
        let ln_node = graph.inner_mut().create_layer_norm_op_node(
            Rc::clone(x.node()),
            1,
            1e-5,
            Some("ln"),
        )?;
        crate::nn::Var::new_with_rc_graph(ln_node, &graph.inner_rc())
    };

    let target = graph.input(&Tensor::zeros(&[2, 4]))?;
    let loss = ln_out.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 4]);
    // 梯度之和应近零
    let grad_sum: f32 = x_grad.data_as_slice().iter().sum();
    assert_abs_diff_eq!(grad_sum, 0.0, epsilon = 1e-3);

    Ok(())
}

// ==================== 非连续内存（contiguity）回归测试 ====================

/// **回归测试**：LayerNorm **反向**拿到非连续 `upstream_grad`（输出接 `permute`）不得 panic/算错。
///
/// 用 `mse(head, target)` 制造**非均匀**上游梯度（比全 1 的 sum 更强，能抓静默错序）：
/// 参考 `mse(ln(x), T)` vs 测试 `mse(permute(ln(x)), permute(T))`。匹配置换下 mse 不变 →
/// loss 与 `x.grad` 逐元素一致；测试路给 ln 反向传入非连续且非均匀的 upstream。
#[test]
fn test_layer_norm_backward_noncontiguous_upstream() {
    use std::rc::Rc;
    let target_ref = Tensor::new(&[0.1, -0.2, 0.3, 0.4, -0.5, 0.6], &[2, 3]);
    let target_perm = target_ref.permute(&[1, 0]).into_contiguous();
    fn run(permute_after: bool, target_ref: &Tensor, target_perm: &Tensor) -> Tensor {
        let graph = Graph::new();
        let x = graph.parameter(&[2, 3], Init::Zeros, "x").unwrap();
        x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 0.5, -1.0, 4.0], &[2, 3]))
            .unwrap();
        let ln_node = graph
            .inner_mut()
            .create_layer_norm_op_node(Rc::clone(x.node()), 1, 1e-5, Some("ln"))
            .unwrap();
        let ln = crate::nn::Var::new_with_rc_graph(ln_node, &graph.inner_rc());
        let (head, target) = if permute_after {
            (
                ln.permute(&[1, 0]).unwrap(),
                graph.input(target_perm).unwrap(),
            )
        } else {
            (ln, graph.input(target_ref).unwrap())
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

/// **回归测试**：LayerNorm **前向**拿到非连续输入（上游 `permute`）不得 panic/静默算错。
///
/// 参考路径喂入 permute 物化后的连续张量；测试路径 `a → permute → ln`。两者 ln 看到**同一逻辑
/// 张量**，前向输出必须逐元素一致（非连续被按内存序误读会得到不同结果，正是 conv2d 曾暴露的坑）。
#[test]
fn test_layer_norm_forward_noncontiguous_input() {
    use std::rc::Rc;
    // [2,3] 原始值；permute[1,0] → [3,2]
    let base = Tensor::new(&[1.0, 2.0, 3.0, 0.5, -1.0, 4.0], &[2, 3]);
    let permuted_contig = base.permute(&[1, 0]).into_contiguous();

    let forward_output = |use_permute: bool| -> Tensor {
        let graph = Graph::new();
        let inner = graph.inner_rc();
        let leaf = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], Some("leaf"))
            .unwrap();
        let ln_parent = if use_permute {
            inner
                .borrow_mut()
                .create_permute_node(leaf.clone(), &[1, 0], None)
                .unwrap()
        } else {
            leaf.clone()
        };
        let ln = inner
            .borrow_mut()
            .create_layer_norm_op_node(ln_parent, 1, 1e-5, Some("ln"))
            .unwrap();
        if use_permute {
            leaf.set_value(Some(&base)).unwrap();
        } else {
            leaf.set_value(Some(&permuted_contig)).unwrap();
        }
        ln.forward_recursive(1, Mode::Train).unwrap();
        ln.value().unwrap().clone()
    };

    let out_ref = forward_output(false);
    let out_test = forward_output(true);
    assert_eq!(out_ref.shape(), out_test.shape());
    for (a, b) in out_ref.to_vec().iter().zip(out_test.to_vec().iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-5);
    }
}
