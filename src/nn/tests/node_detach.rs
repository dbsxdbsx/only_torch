/*
 * Detach 节点测试（梯度屏障）
 *
 * Detach 节点通过 `Var::detach()` 创建，前向传播透传值，反向传播阻断梯度。
 *
 * # 与 Identity 节点的区别
 *
 * | 节点 | forward | backward | 用途 |
 * |------|---------|----------|------|
 * | Identity | y = x | 透传梯度 | pass-through / NEAT 占位（见 node_identity.rs） |
 * | Detach   | y = x | 阻断梯度 | 梯度截断边界 |
 *
 * # 测试覆盖
 * 1. 基本功能：前向传播（值透传）
 * 2. 核心语义：反向传播阻断梯度
 * 3. Var::detach() 高层 API
 * 4. GAN 风格训练模式
 * 5. 动态形状传播
 * 6. 节点创建 API
 */

use crate::nn::{Graph, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;
use std::rc::Rc;

// ============================================================================
// 1. Detach 基本功能测试
// ============================================================================

/// 测试: Detach 节点前向传播（值透传）
#[test]
fn test_detach_forward_value_passthrough() {
    let graph = Graph::new();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let x = graph.input(&input_data).unwrap();
    let y = x.detach();

    y.forward().unwrap();

    let output = y.value().unwrap().unwrap();
    assert_eq!(output, &input_data, "Detach 应该透传输入值");
}

/// 测试: Detach 节点是 detached 状态
#[test]
fn test_detach_is_detached() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let y = x.detach();

    assert!(y.is_detached(), "Detach 节点应该被识别为 detached");
    assert!(!x.is_detached(), "原节点不应受影响");
}

// ============================================================================
// 2. Detach 核心语义：阻断梯度
// ============================================================================

/// 测试: Detach 节点阻断梯度流
#[test]
fn test_detach_blocks_gradient() {
    let graph = Graph::new();

    // x -> w1 -> h -> detach -> w2 -> y -> loss
    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let w1 = graph
        .parameter(&[2, 2], crate::nn::var::Init::Ones, "w1")
        .unwrap();
    let h = w1.matmul(&x).unwrap(); // [2,2] @ [2,1] -> [2,1]

    let detached = h.detach(); // Detach 节点

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

    // w1 不应该有梯度（被 Detach 节点阻断）
    assert!(
        w1.grad().unwrap().is_none(),
        "w1 不应该有梯度（被 Detach 节点阻断）"
    );
}

/// 测试: 不使用 detach 时梯度正常流动
#[test]
fn test_without_detach_gradient_flows() {
    let graph = Graph::new();

    // z -> g_w -> fake -> d_w -> d_out -> loss（不 detach）
    let z = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let g_w = graph
        .parameter(&[1, 2], crate::nn::var::Init::Ones, "g_w")
        .unwrap();
    let fake = z.matmul(&g_w).unwrap(); // [2,1] @ [1,2] -> [2,2]

    let d_w = graph
        .parameter(&[2, 1], crate::nn::var::Init::Ones, "d_w")
        .unwrap();

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

/// 测试: Detach 不影响前向传播
#[test]
fn test_detach_does_not_affect_forward() {
    let graph = Graph::new();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let x = graph.input(&input_data).unwrap();
    let detached = x.detach();

    detached.forward().unwrap();

    // 即使 detached，值仍然正确透传
    let output = detached.value().unwrap().unwrap();
    assert_eq!(output, &input_data, "Detach 仍应透传值");
}

// ============================================================================
// 3. Var::detach() API 测试
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

    // detach 返回新的 Var（创建 Detach 节点）
    let h_detached = h.detach();

    // 验证是不同的节点
    assert_ne!(
        h.node_id(),
        h_detached.node_id(),
        "detach() 应该返回新的 Var（不同的节点 ID）"
    );
}

/// 测试: Var::detach() 原节点不受影响
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

    // 调用 detach
    let h_detached = h.detach();

    // 原节点状态不变
    assert!(!h.is_detached(), "原节点的 detach 状态不应改变");

    // 新节点是 detached
    assert!(h_detached.is_detached(), "detach 返回的节点应该是 detached");
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

    x_detached.forward().unwrap();

    let output = x_detached.value().unwrap().unwrap();
    assert_eq!(&output, &input_data, "detached Var 的值应该正确");
}

// ============================================================================
// 4. GAN 风格训练模式
// ============================================================================

/// 测试: Var::detach() 阻断梯度流（GAN 风格）
#[test]
fn test_var_detach_blocks_gradient_gan_style() {
    let graph = Graph::new();

    // 模拟 GAN: G 输出 -> D
    // z -> g_w -> fake -> d_w -> d_out -> loss
    let z = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let g_w = graph
        .parameter(&[1, 2], crate::nn::var::Init::Ones, "g_w")
        .unwrap();
    let fake = z.matmul(&g_w).unwrap(); // [2,1] @ [1,2] -> [2,2]

    let d_w = graph
        .parameter(&[2, 1], crate::nn::var::Init::Ones, "d_w")
        .unwrap();

    // 使用 detach，训练 D
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

// ============================================================================
// 5. 动态形状测试
// ============================================================================

/// 测试: Detach 节点的动态形状传播
#[test]
fn test_detach_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    let result = h0.detach();

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试: Detach 节点在不同 batch_size 下的前向计算
#[test]
fn test_detach_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

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

/// 测试: Detach 节点在不同 batch_size 下的反向传播
#[test]
fn test_detach_dynamic_batch_backward() {
    use crate::nn::var_ops::VarLossOps;

    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]

    let result = h0.detach();

    // Detach 节点：反向传播时阻断梯度
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

// ============================================================================
// 6. 节点创建 API 测试
// ============================================================================

#[test]
fn test_create_detach_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let detach = inner
        .borrow_mut()
        .create_detach_node(input.clone(), Some("detach"))
        .unwrap();

    assert_eq!(detach.shape(), vec![3, 4]);
    assert_eq!(detach.name(), Some("detach"));
    assert!(!detach.is_leaf());
    assert_eq!(detach.type_name(), "Detach");
}

#[test]
fn test_create_detach_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 5, 8], None)
        .unwrap();

    let detach = inner
        .borrow_mut()
        .create_detach_node(input.clone(), None)
        .unwrap();

    assert_eq!(detach.shape(), vec![2, 5, 8]);
}

#[test]
fn test_create_detach_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_detach;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let detach = inner.borrow_mut().create_detach_node(input, None).unwrap();
        weak_detach = Rc::downgrade(&detach);

        assert!(weak_detach.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_detach.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
