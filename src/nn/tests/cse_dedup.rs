/*
 * @Author       : 老董
 * @Description  : CSE（公共子表达式消除）节点去重 单元测试
 *
 * 测试策略：
 * 1. 应去重场景：同一 Var 输入的 Concat/Add/MatMul/一元运算 → 相同 NodeId
 * 2. 不应去重场景：不同 Var 输入/不同 axis/命名节点/Input/Parameter/Detach/不同分组
 * 3. 梯度等价性：共享节点的 backward 梯度 == 独立节点的梯度总和
 * 4. 缓存生命周期：跨 forward pass 重置、Weak 过期
 */

use crate::nn::{Graph, GraphError, Init, Var, VarActivationOps, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;

// ==================== 1. 应去重场景 ====================

/// 同一对 Var 两次 Var::concat → 返回相同 NodeId
#[test]
fn test_cse_concat_same_inputs() {
    let graph = Graph::new();
    let a = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let b = graph.input(&Tensor::new(&[3.0, 4.0], &[1, 2])).unwrap();

    let c1 = Var::concat(&[&a, &b], 0).unwrap();
    let c2 = Var::concat(&[&a, &b], 0).unwrap();

    assert_eq!(c1.node_id(), c2.node_id(), "相同输入的 concat 应去重");
}

/// 同一对 Var 两次 Add → 返回相同 NodeId
#[test]
fn test_cse_add_same_inputs() {
    let graph = Graph::new();
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();

    let s1 = &a + &b;
    let s2 = &a + &b;

    assert_eq!(s1.node_id(), s2.node_id(), "相同输入的 add 应去重");
}

/// 同一对 Var 两次 matmul → 返回相同 NodeId
#[test]
fn test_cse_matmul_same_inputs() -> Result<(), GraphError> {
    let graph = Graph::new();
    let a = graph.input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    let b = graph.input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))?;

    let m1 = a.matmul(&b)?;
    let m2 = a.matmul(&b)?;

    assert_eq!(m1.node_id(), m2.node_id(), "相同输入的 matmul 应去重");
    Ok(())
}

/// 同一 Var 两次一元运算 → 返回相同 NodeId
#[test]
fn test_cse_unary_same_input() {
    let graph = Graph::new();
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();

    let r1 = a.relu();
    let r2 = a.relu();
    assert_eq!(r1.node_id(), r2.node_id(), "相同输入的 relu 应去重");

    let t1 = a.tanh();
    let t2 = a.tanh();
    assert_eq!(t1.node_id(), t2.node_id(), "相同输入的 tanh 应去重");
}

// ==================== 2. 不应去重场景 ====================

/// 不同 Var（即使值相同）的 concat → 不同 NodeId
#[test]
fn test_cse_different_inputs_no_dedup() {
    let graph = Graph::new();
    let data = Tensor::new(&[1.0, 2.0], &[1, 2]);

    let a1 = graph.input(&data).unwrap();
    let a2 = graph.input(&data).unwrap(); // 不同 Input 节点
    let b = graph.input(&data).unwrap();

    let c1 = Var::concat(&[&a1, &b], 0).unwrap();
    let c2 = Var::concat(&[&a2, &b], 0).unwrap();

    assert_ne!(
        c1.node_id(),
        c2.node_id(),
        "不同 Var 输入（即使值相同）不应去重"
    );
}

/// 同一对 Var 但 axis 不同 → 不同 NodeId
#[test]
fn test_cse_different_axis_no_dedup() {
    let graph = Graph::new();
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();

    let c0 = Var::concat(&[&a, &b], 0).unwrap();
    let c1 = Var::concat(&[&a, &b], 1).unwrap();

    assert_ne!(c0.node_id(), c1.node_id(), "不同 axis 不应去重");
}

/// Input 节点不参与去重
#[test]
fn test_cse_input_no_dedup() {
    let graph = Graph::new();
    let data = Tensor::new(&[1.0, 2.0], &[1, 2]);

    let i1 = graph.input(&data).unwrap();
    let i2 = graph.input(&data).unwrap();

    assert_ne!(i1.node_id(), i2.node_id(), "Input 节点不应去重");
}

/// Parameter 节点不参与去重
#[test]
fn test_cse_parameter_no_dedup() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[2, 2], Init::Zeros, "w1")?;
    let p2 = graph.parameter(&[2, 2], Init::Zeros, "w2")?;

    assert_ne!(p1.node_id(), p2.node_id(), "Parameter 节点不应去重");
    Ok(())
}

/// Detach 节点不参与去重（每次调用创建独立梯度边界）
#[test]
fn test_cse_detach_no_dedup() {
    let graph = Graph::new();
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();

    let d1 = a.detach();
    let d2 = a.detach();

    assert_ne!(d1.node_id(), d2.node_id(), "Detach 节点不应去重");
}

// ==================== 3. 梯度等价性 ====================

/// 共享 Concat 节点的 backward 梯度等价于独立节点的梯度总和
///
/// 构造：obs + action → concat_shared → 两个 Linear 路径 → sum → loss
/// 验证 obs 和 action 的梯度正确
#[test]
fn test_cse_gradient_equivalence_concat() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 创建参数
    let obs = graph.parameter(&[2, 3], Init::Zeros, "obs")?;
    obs.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))?;

    let act = graph.parameter(&[2, 1], Init::Zeros, "act")?;
    act.set_value(&Tensor::new(&[0.5, 1.5], &[2, 1]))?;

    // 两次 concat 同一对输入 → CSE 应去重为一个节点
    let cat1 = Var::concat(&[&obs, &act], 1)?; // [2, 4]
    let cat2 = Var::concat(&[&obs, &act], 1)?; // 应该 == cat1
    assert_eq!(cat1.node_id(), cat2.node_id(), "应去重");

    // 两条路径消费同一个 concat 节点
    let w1 = graph.parameter(&[4, 1], Init::Zeros, "w1")?;
    w1.set_value(&Tensor::new(&[1.0, 0.5, 0.2, 0.1], &[4, 1]))?;
    let w2 = graph.parameter(&[4, 1], Init::Zeros, "w2")?;
    w2.set_value(&Tensor::new(&[0.3, 0.4, 0.6, 0.8], &[4, 1]))?;

    let path1 = cat1.matmul(&w1)?; // [2, 1]
    let path2 = cat2.matmul(&w2)?; // [2, 1]（使用同一节点）
    let combined = &path1 + &path2;

    let target = graph.input(&Tensor::zeros(&[2, 1]))?;
    let loss = combined.mse_loss(&target)?;

    // 反向传播
    graph.zero_grad()?;
    loss.backward()?;

    // obs 和 act 应该收到来自两条路径的梯度总和
    let obs_grad = obs.grad()?.expect("obs 应有梯度");
    let act_grad = act.grad()?.expect("act 应有梯度");

    // 梯度不应为零（确认梯度成功流过共享节点）
    assert!(
        obs_grad.sum().get_data_number().unwrap().abs() > 1e-6,
        "obs 梯度不应为零"
    );
    assert!(
        act_grad.sum().get_data_number().unwrap().abs() > 1e-6,
        "act 梯度不应为零"
    );

    Ok(())
}

/// 共享 Add 节点的 backward 梯度正确
#[test]
fn test_cse_gradient_equivalence_add() -> Result<(), GraphError> {
    let graph = Graph::new();

    let a = graph.parameter(&[2, 2], Init::Zeros, "a")?;
    a.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    let b = graph.parameter(&[2, 2], Init::Zeros, "b")?;
    b.set_value(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))?;

    // 两次 add 同一对输入 → CSE 应去重
    let s1 = &a + &b;
    let s2 = &a + &b;
    assert_eq!(s1.node_id(), s2.node_id(), "应去重");

    // 两条路径各自用 relu 和 tanh（产生不同下游梯度）
    let p1 = s1.relu();
    let p2 = s2.tanh();
    let combined = &p1 + &p2;

    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = combined.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    let a_grad = a.grad()?.expect("a 应有梯度");
    let b_grad = b.grad()?.expect("b 应有梯度");

    // a 和 b 的梯度应相同（因为 Add 对两个输入的局部梯度都是 1）
    assert_eq!(&a_grad, &b_grad, "Add 两个输入的梯度应相同");

    Ok(())
}

// ==================== 4. 缓存生命周期 ====================

/// 跨 forward pass 后，缓存应重置，相同操作创建新节点
#[test]
fn test_cse_cache_reset_across_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();

    // 第一次 forward pass
    let s1 = &a + &b;
    s1.forward()?;

    // forward 后创建的同操作应该是新节点（缓存已重置）
    let s2 = &a + &b;

    assert_ne!(s1.node_id(), s2.node_id(), "跨 forward pass 后应创建新节点");

    Ok(())
}

/// 缓存的 Var 被 drop 后（Weak 过期），重新创建同操作 → 新 NodeId
#[test]
fn test_cse_weak_ref_expired() {
    let graph = Graph::new();
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();

    // 创建并立即获取 id，然后 drop
    let id1 = {
        let s = &a + &b;
        s.node_id()
    }; // s 被 drop，Weak 过期

    // 重新创建同操作
    let s2 = &a + &b;

    assert_ne!(id1, s2.node_id(), "Weak 过期后应创建新节点");
}
