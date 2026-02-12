/*
 * Identity 节点测试（纯恒等映射）
 *
 * Identity 节点是恒等映射（y = x），前向传播透传值，反向传播透传梯度。
 * 用于 NEAT 占位、skip connection 等场景。
 *
 * # 与 Detach 节点的区别
 *
 * | 节点 | forward | backward | 用途 |
 * |------|---------|----------|------|
 * | Identity | y = x | 透传梯度 | pass-through / NEAT 占位 |
 * | Detach   | y = x | 阻断梯度 | 梯度截断边界（见 node_detach.rs） |
 *
 * # 测试覆盖
 * 1. 基本功能：前向传播（值透传）
 * 2. 反向传播：梯度透传（不阻断）
 * 3. 形状保持
 * 4. 节点创建 API
 * 5. 动态形状传播
 */

use crate::nn::{Graph, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;
use std::rc::Rc;

// ============================================================================
// 1. Identity 节点基本功能测试
// ============================================================================

/// 测试: Identity 节点前向传播（值透传）
#[test]
fn test_identity_forward_value_passthrough() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    x.set_value(Some(&input_data)).unwrap();

    let y = inner
        .borrow_mut()
        .create_identity_node(x.clone(), None)
        .unwrap();

    // 前向传播
    inner.borrow_mut().forward_via_node_inner(&y).unwrap();

    let output = y.value().unwrap();
    assert_eq!(output, input_data, "Identity 应该透传输入值");
}

/// 测试: Identity 节点反向传播（梯度透传，不阻断）
#[test]
fn test_identity_backward_gradient_passthrough() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let w = graph
        .parameter(&[1, 2], crate::nn::var::Init::Ones, "w")
        .unwrap();
    let h = w.matmul(&x).unwrap(); // [1,2] @ [2,1] -> [1,1]

    // 使用底层 API 创建 Identity 节点（梯度应正常流过）
    let inner = graph.inner_rc();
    let identity_node = inner
        .borrow_mut()
        .create_identity_node(h.node().clone(), Some("identity"))
        .unwrap();
    let identity = crate::nn::Var::new_with_rc_graph(identity_node, &inner);

    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = identity.mse_loss(&target).unwrap();

    // 前向 + 反向
    loss.backward().unwrap();

    // w 应该有梯度（梯度通过 Identity 节点正常传递）
    assert!(
        w.grad().unwrap().is_some(),
        "梯度应该通过 Identity 节点正常传递"
    );
}

/// 测试: Identity 节点不是 detached
#[test]
fn test_identity_is_not_detached() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let identity = inner
        .borrow_mut()
        .create_identity_node(x.clone(), None)
        .unwrap();

    // Identity 节点不应该被视为 detached
    assert!(!identity.is_detached());
}

// ============================================================================
// 2. 形状测试
// ============================================================================

/// 测试: Identity 节点保持各种形状
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
            .create_identity_node(x.clone(), None)
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
// 3. 动态形状测试
// ============================================================================

/// 测试: Identity 节点的动态形状传播
#[test]
fn test_identity_dynamic_shape_propagation() {
    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 通过底层 API 创建 Identity 节点
    let inner = graph.inner_rc();
    let identity_node = inner
        .borrow_mut()
        .create_identity_node(h0.node().clone(), None)
        .unwrap();
    let identity = crate::nn::Var::new_with_rc_graph(identity_node, &inner);

    // 验证动态形状传播
    let dyn_shape = identity.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试: Identity 节点在不同 batch_size 下的前向计算
#[test]
fn test_identity_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    let inner = graph.inner_rc();
    let identity_node = inner
        .borrow_mut()
        .create_identity_node(h0.node().clone(), None)
        .unwrap();
    let identity = crate::nn::Var::new_with_rc_graph(identity_node, &inner);

    // 第一次 forward：batch=2
    identity.forward().unwrap();
    let value1 = identity.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 16], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    identity.forward().unwrap();
    let value2 = identity.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 16], "第二次 forward: batch=8");
}

// ============================================================================
// 4. 节点创建 API 测试
// ============================================================================

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
        .create_identity_node(input.clone(), Some("identity"))
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
        .create_identity_node(input.clone(), None)
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
            .create_identity_node(input, None)
            .unwrap();
        weak_identity = Rc::downgrade(&identity);

        assert!(weak_identity.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_identity.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
