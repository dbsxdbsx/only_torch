/*
 * Identity 节点测试
 *
 * Identity 节点是恒等映射（y = x），通过 `Var::detach()` 创建。
 * 用于在计算图中建立显式的梯度截断边界。
 *
 * # 与 DetachedVar 的区别
 *
 * | 方法 | 返回类型 | 创建节点 | 用途 |
 * |------|---------|---------|------|
 * | `var.detach()` | `DetachedVar` | ❌ | ModelState.forward()（推荐） |
 * | `var.detach()` | `Var` | ✅ Identity | 直接图操作、可视化调试 |
 *
 * # 测试覆盖
 * 1. 基本功能：前向传播（值透传）
 * 2. 反向传播：梯度透传
 * 3. detach 语义：阻断梯度流（is_detached=true 时）
 * 4. 函数式 detach API
 *
 * # 可视化
 * Identity 节点在 Graphviz 中显示为椭圆形、虚线边框、浅紫色背景。
 */

use crate::nn::{Graph, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;

// ============================================================================
// 1. Identity 节点基本功能测试
// ============================================================================

/// 测试: Identity 节点前向传播（值透传）
///
/// 使用高层 API：var.detach() 创建 Identity 节点
#[test]
fn test_identity_forward_value_passthrough() {
    let graph = Graph::new();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let x = graph.input(&input_data).unwrap();

    // detach() 创建 Identity 节点（detached=true，但前向传播仍透传值）
    let y = x.detach();

    y.forward().unwrap();

    let output = y.value().unwrap().unwrap();
    assert_eq!(output, &input_data, "Identity 应该透传输入值");
}

/// 测试: 非 detached Identity 节点反向传播（梯度透传）
///
/// 使用底层 API 创建非 detached 的 Identity 节点，验证梯度正常流过
#[test]
fn test_identity_backward_gradient_passthrough() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let w = graph
        .parameter(&[1, 2], crate::nn::var::Init::Ones, "w")
        .unwrap();
    let h = w.matmul(&x).unwrap(); // [1,2] @ [2,1] -> [1,1]

    // 使用底层 API 创建非 detached 的 Identity 节点
    let inner = graph.inner_rc();
    let identity_node = inner
        .borrow_mut()
        .create_identity_node(h.node().clone(), Some("identity"), false)
        .unwrap();
    let identity = crate::nn::Var::new_with_rc_graph(identity_node, &inner);

    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = identity.mse_loss(&target).unwrap();

    // 前向 + 反向
    loss.backward().unwrap();

    // w 应该有梯度（梯度通过非 detached Identity 正常传递）
    assert!(
        w.grad().unwrap().is_some(),
        "梯度应该通过 Identity 节点正常传递"
    );
}

/// 测试: Identity 节点形状保持
///
/// 使用底层 API 创建 Identity，验证各种形状的传播
#[test]
fn test_identity_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let shapes: Vec<Vec<usize>> = vec![vec![2, 3], vec![2, 3, 4], vec![1, 2, 3, 4]];

    for shape in shapes {
        let x = inner
            .borrow_mut()
            .create_basic_input_node(&shape, None)
            .unwrap();
        let y = inner
            .borrow_mut()
            .create_identity_node(x.clone(), None, false)
            .unwrap();

        assert_eq!(
            x.value_expected_shape(),
            y.value_expected_shape(),
            "Identity 应该保持形状: {:?}",
            shape
        );
    }
}

// ============================================================================
// 2. Detached Identity 测试
// ============================================================================

/// 测试: detached Identity 阻断梯度流
///
/// 使用高层 API：var.detach() 自然创建 detached Identity
#[test]
fn test_identity_detached_blocks_gradient() {
    let graph = Graph::new();

    // x -> w1 -> h -> detach -> w2 -> y -> loss
    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let w1 = graph
        .parameter(&[2, 2], crate::nn::var::Init::Ones, "w1")
        .unwrap();
    let h = w1.matmul(&x).unwrap(); // [2,2] @ [2,1] -> [2,1]

    let detached = h.detach(); // detached Identity

    let w2 = graph
        .parameter(&[1, 2], crate::nn::var::Init::Ones, "w2")
        .unwrap();
    let y = w2.matmul(&detached).unwrap(); // [1,2] @ [2,1] -> [1,1]

    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // w2 应该有梯度（在 detach 点之后）
    assert!(
        w2.grad().unwrap().is_some(),
        "w2 应该有梯度（在 detach 点之后）"
    );

    // w1 不应该有梯度（被 detached Identity 阻断）
    assert!(
        w1.grad().unwrap().is_none(),
        "w1 不应该有梯度（被 detached Identity 阻断）"
    );
}

/// 测试: detached Identity 不影响前向传播
#[test]
fn test_identity_detached_does_not_affect_forward() {
    let graph = Graph::new();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let x = graph.input(&input_data).unwrap();
    let detached = x.detach();

    detached.forward().unwrap();

    // 即使 detached，值仍然正确透传
    let output = detached.value().unwrap().unwrap();
    assert_eq!(output, &input_data, "detached Identity 仍应透传值");
}

// ============================================================================
// 3. Var::detach() 函数式 API 测试
// ============================================================================

/// 测试: Var::detach() 返回新的 Var
#[test]
fn test_var_detach_returns_new_var() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let w = graph
        .parameter(&[1, 2], crate::nn::var::Init::Ones, "w")
        .unwrap();
    let h = x.matmul(&w).unwrap();

    // detach 返回新的 Var（创建 Identity 节点）
    let h_detached = h.detach();

    // 验证是不同的节点
    assert_ne!(
        h.node_id(),
        h_detached.node_id(),
        "detach() 应该返回新的 Var（不同的节点 ID）"
    );
}

/// 测试: Var::detach() 原节点不受影响
///
/// 使用 Var::is_detached() 替代已移除的 graph.inner().is_node_detached()
#[test]
fn test_var_detach_original_unchanged() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let w = graph
        .parameter(&[1, 2], crate::nn::var::Init::Ones, "w")
        .unwrap();
    let h = (&x).matmul(&w).unwrap();

    // 原节点不是 detached
    assert!(!h.is_detached(), "原节点应该是 attached 状态");

    // 调用 detach（创建新的 Identity 节点）
    let h_detached = h.detach();

    // 原节点状态不变
    assert!(!h.is_detached(), "原节点的 detach 状态不应改变");

    // 新节点是 detached
    assert!(h_detached.is_detached(), "detach 返回的节点应该是 detached");
}

/// 测试: Var::detach() 阻断梯度流（GAN 风格）
#[test]
fn test_var_detach_blocks_gradient_gan_style() {
    let graph = Graph::new();

    // 模拟 GAN: G 输出 -> D
    // z -> g_w -> fake -> d_w -> d_out -> loss
    // 形状: [2,1] @ [1,2] -> [2,2] @ [2,1] -> [2,1]

    let z = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let g_w = graph
        .parameter(&[1, 2], crate::nn::var::Init::Ones, "g_w")
        .unwrap();
    let fake = z.matmul(&g_w).unwrap(); // [2,1] @ [1,2] -> [2,2]

    let d_w = graph
        .parameter(&[2, 1], crate::nn::var::Init::Ones, "d_w")
        .unwrap();

    // === 场景 1: 使用 detach，训练 D ===
    let fake_detached = fake.detach();
    let d_out_for_d = fake_detached.matmul(&d_w).unwrap(); // [2,2] @ [2,1] -> [2,1]
    let target = graph.input(&Tensor::zeros(&[2, 1])).unwrap();
    let d_loss = d_out_for_d.mse_loss(&target).unwrap();

    d_loss.backward().unwrap();

    // d_w 应该有梯度
    assert!(d_w.grad().unwrap().is_some(), "d_w 应该有梯度（训练 D）");

    // g_w 不应该有梯度（fake 被 detach）
    assert!(
        g_w.grad().unwrap().is_none(),
        "g_w 不应该有梯度（fake 被 detach）"
    );
}

/// 测试: 不使用 detach 时梯度正常流动
#[test]
fn test_var_without_detach_gradient_flows() {
    let graph = Graph::new();

    // z -> g_w -> fake -> d_w -> d_out -> loss（不 detach）
    // 形状: [2,1] @ [1,2] -> [2,2] @ [2,1] -> [2,1]

    let z = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let g_w = graph
        .parameter(&[1, 2], crate::nn::var::Init::Ones, "g_w")
        .unwrap();
    let fake = z.matmul(&g_w).unwrap(); // [2,1] @ [1,2] -> [2,2]

    let d_w = graph
        .parameter(&[2, 1], crate::nn::var::Init::Ones, "d_w")
        .unwrap();

    // 不使用 detach
    let d_out = fake.matmul(&d_w).unwrap(); // [2,2] @ [2,1] -> [2,1]
    let target = graph.input(&Tensor::zeros(&[2, 1])).unwrap();
    let g_loss = d_out.mse_loss(&target).unwrap();

    g_loss.backward().unwrap();

    // d_w 和 g_w 都应该有梯度
    assert!(d_w.grad().unwrap().is_some(), "d_w 应该有梯度");
    assert!(
        g_w.grad().unwrap().is_some(),
        "g_w 应该有梯度（没有 detach，梯度正常流动）"
    );
}

/// 测试: 多次 detach 创建多个独立节点
#[test]
fn test_var_detach_multiple_times() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();

    let d1 = x.detach();
    let d2 = x.detach();

    // 每次 detach 都创建新节点
    assert_ne!(d1.node_id(), d2.node_id(), "多次 detach 应该创建不同的节点");
    assert_ne!(d1.node_id(), x.node_id(), "detached 节点应该与原节点不同");
    assert_ne!(d2.node_id(), x.node_id(), "detached 节点应该与原节点不同");
}

/// 测试: detach 后的值正确
#[test]
fn test_var_detach_value_correct() {
    let graph = Graph::new();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let x = graph.input(&input_data).unwrap();
    let x_detached = x.detach();

    // 前向传播
    x_detached.forward().unwrap();

    // 值应该正确
    let output = x_detached.value().unwrap().unwrap();
    assert_eq!(&output, &input_data, "detached Var 的值应该正确");
}

// ==================== 动态形状测试 ====================

/// 测试 Identity 节点的动态形状传播
#[test]
fn test_identity_dynamic_shape_propagation() {
    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建 Identity（detach）
    let result = h0.detach();

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 Identity 节点在不同 batch_size 下的前向计算
#[test]
fn test_identity_dynamic_batch_forward() {
    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // Identity（detach）
    let result = h0.detach();

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 16], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 16], "第二次 forward: batch=8");
}

/// 测试 Identity 节点在不同 batch_size 下的反向传播
#[test]
fn test_identity_dynamic_batch_backward() {
    use crate::nn::var_ops::VarLossOps;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]

    // Identity（非 detached，梯度可以流过）
    let result = h0.detach();

    // 创建目标和损失
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[6, 8])).unwrap();
    target.set_value(&Tensor::zeros(&[6, 4])).unwrap();

    // 第二次 forward + backward：batch=6
    loss.forward().unwrap();
    assert_eq!(
        result.value().unwrap().unwrap().shape(),
        &[6, 4],
        "第二次 forward: batch=6"
    );
    loss.backward().unwrap();
}

// ==================== 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_identity_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let identity = inner
        .borrow_mut()
        .create_identity_node(input.clone(), Some("identity"), false)
        .unwrap();

    assert_eq!(identity.shape(), vec![3, 4]);
    assert_eq!(identity.name(), Some("identity"));
    assert!(!identity.is_leaf());
    assert!(!identity.is_detached());
}

#[test]
fn test_create_identity_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 5, 8], None)
        .unwrap();

    let identity = inner
        .borrow_mut()
        .create_identity_node(input.clone(), None, false)
        .unwrap();

    assert_eq!(identity.shape(), vec![2, 5, 8]);
}

#[test]
fn test_create_identity_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_identity;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let identity = inner
            .borrow_mut()
            .create_identity_node(input, None, false)
            .unwrap();
        weak_identity = Rc::downgrade(&identity);

        assert!(weak_identity.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_identity.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
