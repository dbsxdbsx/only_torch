/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : M4 测试 - 验证 Graph 的动态扩展能力（NEAT 友好性）
 *                 测试在 forward/backward 后动态添加节点的能力
 *                 使用底层 GraphInner API：Graph::new() -> inner_rc() -> borrow_mut()
 * @LastEditors  : 老董
 * @LastEditTime : 2026-02-12
 */

use crate::nn::Graph;
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ============================================================================
// 基础动态添加测试
// ============================================================================

/// 测试: 在 forward 后添加新节点并继续计算
#[test]
fn test_add_node_after_forward() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建初始图: add1 = a + b
    let a = gi
        .create_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    let b = gi
        .create_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();
    let add1 = gi
        .create_add_node(vec![a.clone(), b.clone()], Some("add1"))
        .unwrap();

    // 2. 第 1 次 forward
    gi.forward_via_node_inner(&add1).unwrap();
    let add1_value_before = add1.value().unwrap();
    let first_pass_id = gi.last_forward_pass_id();
    assert_eq!(first_pass_id, 1);

    // 3. 动态添加新节点: add2 = add1 + c
    let c = gi
        .create_parameter_node_seeded(&[2, 1], Some("c"), 44)
        .unwrap();
    let add2 = gi
        .create_add_node(vec![add1.clone(), c.clone()], Some("add2"))
        .unwrap();

    // 4. 新节点的 pass_id 是 0（还未参与计算）
    assert_eq!(add2.last_forward_pass_id(), 0);

    // 5. 对新节点进行 forward
    gi.forward_via_node_inner(&add2).unwrap();
    let second_pass_id = gi.last_forward_pass_id();
    assert_eq!(second_pass_id, 2);

    // 6. 验证计算结果正确
    let a_value = a.value().unwrap();
    let b_value = b.value().unwrap();
    let c_value = c.value().unwrap();
    let add1_value = add1.value().unwrap();
    let add2_value = add2.value().unwrap();

    // add1 = a + b
    let expected_add1 = a_value.clone() + b_value.clone();
    assert_eq!(add1_value, expected_add1);

    // add2 = add1 + c
    let expected_add2 = expected_add1.clone() + c_value.clone();
    assert_eq!(add2_value, expected_add2);

    // 7. 原始节点的值没有被意外修改
    assert_eq!(add1_value_before, add1_value);
}

/// 测试: 在 backward 后添加新节点并继续训练
#[test]
fn test_add_node_after_backward() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建初始图: y = w * x + b，然后创建 loss = MSE(y, target)
    let x = gi.create_basic_input_node(&[2, 1], Some("x")).unwrap();
    let w = gi
        .create_parameter_node_seeded(&[1, 2], Some("w"), 42)
        .unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let b = gi
        .create_parameter_node_seeded(&[1, 1], Some("b"), 43)
        .unwrap();
    gi.register_parameter("b".to_string(), Rc::downgrade(&b))
        .unwrap();
    let wx = gi
        .create_mat_mul_node(vec![w.clone(), x.clone()], Some("wx"))
        .unwrap();
    let y = gi
        .create_add_node(vec![wx.clone(), b.clone()], Some("y"))
        .unwrap();
    let target = gi
        .create_basic_input_node(&[1, 1], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(y.clone(), target.clone(), Some("loss"))
        .unwrap();

    // 2. 设置输入并进行一轮训练
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let target_value = Tensor::new(&[1.0], &[1, 1]);
    x.set_value(Some(&x_value)).unwrap();
    target.set_value(Some(&target_value)).unwrap();
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();

    // 3. 验证反向传播成功
    assert!(w.grad().is_some());
    assert!(b.grad().is_some());
    let w_grad_before = w.grad().unwrap();

    // 4. 动态添加新节点: z = y + c，然后创建新的 loss2 = MSE(z, target2)
    let c = gi
        .create_parameter_node_seeded(&[1, 1], Some("c"), 44)
        .unwrap();
    gi.register_parameter("c".to_string(), Rc::downgrade(&c))
        .unwrap();
    let z = gi
        .create_add_node(vec![y.clone(), c.clone()], Some("z"))
        .unwrap();
    let target2 = gi
        .create_basic_input_node(&[1, 1], Some("target2"))
        .unwrap();
    let loss2 = gi
        .create_mse_mean_node(z.clone(), target2.clone(), Some("loss2"))
        .unwrap();

    // 5. 清除旧梯度（模拟拓扑变化后的重新训练）
    gi.zero_grad().unwrap();

    // 6. 验证梯度已被清除
    assert!(w.grad().is_none());
    assert!(b.grad().is_none());

    // 7. 值仍然保留（Rc 引用持有，不会丢失）
    assert!(y.value().is_some());

    // 8. 对扩展后的图进行新一轮训练
    let target2_value = Tensor::new(&[1.0], &[1, 1]);
    target2.set_value(Some(&target2_value)).unwrap();
    gi.forward_via_node_inner(&loss2).unwrap();
    gi.backward_via_node_inner(&loss2).unwrap();

    // 9. 所有参数都有新的梯度
    assert!(w.grad().is_some());
    assert!(b.grad().is_some());
    assert!(c.grad().is_some());

    // 10. 新梯度形状与参数一致
    let w_grad_after = w.grad().unwrap();
    assert_eq!(w_grad_after.shape(), w_grad_before.shape());
}

// ============================================================================
// 多次拓扑变化测试
// ============================================================================

/// 测试: 连续多次添加节点
#[test]
fn test_multiple_topology_changes() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 初始图: node1 = param + input
    let param = gi
        .create_parameter_node_seeded(&[2, 1], Some("param"), 42)
        .unwrap();
    gi.register_parameter("param".to_string(), Rc::downgrade(&param))
        .unwrap();
    let input = gi
        .create_basic_input_node(&[2, 1], Some("input"))
        .unwrap();
    let node1 = gi
        .create_add_node(vec![param.clone(), input.clone()], Some("node1"))
        .unwrap();
    let target1 = gi
        .create_basic_input_node(&[2, 1], Some("target1"))
        .unwrap();
    let loss1 = gi
        .create_mse_mean_node(node1.clone(), target1.clone(), Some("loss1"))
        .unwrap();

    // 设置输入
    let input_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let target1_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    input.set_value(Some(&input_value)).unwrap();
    target1.set_value(Some(&target1_value)).unwrap();

    // 2. 第 1 轮训练
    gi.forward_via_node_inner(&loss1).unwrap();
    gi.backward_via_node_inner(&loss1).unwrap();
    let value1 = node1.value().unwrap();

    // 3. 第 1 次拓扑变化: 添加 node2 = tanh(node1)
    let node2 = gi
        .create_tanh_node(node1.clone(), Some("node2"))
        .unwrap();
    let target2 = gi
        .create_basic_input_node(&[2, 1], Some("target2"))
        .unwrap();
    let loss2 = gi
        .create_mse_mean_node(node2.clone(), target2.clone(), Some("loss2"))
        .unwrap();
    gi.zero_grad().unwrap();

    // 4. 第 2 轮训练
    let target2_value = Tensor::new(&[0.5, 0.8], &[2, 1]);
    target2.set_value(Some(&target2_value)).unwrap();
    gi.forward_via_node_inner(&loss2).unwrap();
    gi.backward_via_node_inner(&loss2).unwrap();
    let value2 = node2.value().unwrap();

    // 5. 第 2 次拓扑变化: 添加 node3 = node2 + bias
    let bias = gi
        .create_parameter_node_seeded(&[2, 1], Some("bias"), 43)
        .unwrap();
    gi.register_parameter("bias".to_string(), Rc::downgrade(&bias))
        .unwrap();
    let node3 = gi
        .create_add_node(vec![node2.clone(), bias.clone()], Some("node3"))
        .unwrap();
    let target3 = gi
        .create_basic_input_node(&[2, 1], Some("target3"))
        .unwrap();
    let loss3 = gi
        .create_mse_mean_node(node3.clone(), target3.clone(), Some("loss3"))
        .unwrap();
    gi.zero_grad().unwrap();

    // 6. 第 3 轮训练
    let target3_value = Tensor::new(&[0.6, 0.9], &[2, 1]);
    target3.set_value(Some(&target3_value)).unwrap();
    gi.forward_via_node_inner(&loss3).unwrap();
    gi.backward_via_node_inner(&loss3).unwrap();
    let value3 = node3.value().unwrap();

    // 7. 验证计算链正确
    // node1 = param + input（forward 时重新计算，值应与 value1 一致因为参数未更新）
    assert_eq!(node1.value().unwrap(), value1);

    // node2 = tanh(node1)
    let expected_node2 = value1.tanh();
    assert_eq!(value2, expected_node2);

    // node3 = node2 + bias
    let bias_value = bias.value().unwrap();
    let expected_node3 = expected_node2 + bias_value;
    assert_eq!(value3, expected_node3);
}

/// 测试: 在同一个父节点上添加多个子节点（分支）
#[test]
fn test_add_multiple_branches() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 初始图
    let input = gi
        .create_basic_input_node(&[2, 1], Some("input"))
        .unwrap();
    let param = gi
        .create_parameter_node_seeded(&[2, 1], Some("param"), 42)
        .unwrap();
    let base = gi
        .create_add_node(vec![input.clone(), param.clone()], Some("base"))
        .unwrap();

    // 设置输入
    let input_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    input.set_value(Some(&input_value)).unwrap();

    // 2. 初始 forward
    gi.forward_via_node_inner(&base).unwrap();

    // 3. 添加分支1: branch1 = tanh(base)
    let branch1 = gi
        .create_tanh_node(base.clone(), Some("branch1"))
        .unwrap();

    // 4. 添加分支2: branch2 = tanh(base)
    let branch2 = gi
        .create_tanh_node(base.clone(), Some("branch2"))
        .unwrap();

    // 5. 合并分支: merged = branch1 + branch2
    let merged = gi
        .create_add_node(vec![branch1.clone(), branch2.clone()], Some("merged"))
        .unwrap();

    // 6. 验证 branch1/branch2 都以 base 为父节点
    assert_eq!(branch1.parents().len(), 1);
    assert_eq!(branch1.parents()[0].id(), base.id());
    assert_eq!(branch2.parents().len(), 1);
    assert_eq!(branch2.parents()[0].id(), base.id());

    // 7. Forward 合并节点
    gi.forward_via_node_inner(&merged).unwrap();

    // 8. 验证计算正确
    let base_value = base.value().unwrap();
    let branch1_value = branch1.value().unwrap();
    let branch2_value = branch2.value().unwrap();
    let merged_value = merged.value().unwrap();

    let expected_branch = base_value.tanh();
    // branch1 和 branch2 都是 tanh(base)，应该相等
    assert_eq!(branch1_value, expected_branch);
    assert_eq!(branch2_value, expected_branch);

    // merged = branch1 + branch2 = 2 * tanh(base)
    let expected_merged = expected_branch.clone() + expected_branch;
    assert_eq!(merged_value, expected_merged);
}

// ============================================================================
// 边界情况测试
// ============================================================================

/// 测试: 链式添加节点（A -> B -> C -> D）
#[test]
fn test_chain_node_addition() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建初始节点
    let a = gi
        .create_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    gi.register_parameter("a".to_string(), Rc::downgrade(&a))
        .unwrap();
    let b = gi
        .create_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();
    gi.register_parameter("b".to_string(), Rc::downgrade(&b))
        .unwrap();

    // 2. 逐步构建链: result = tanh(tanh(tanh(a + b)))
    let add = gi
        .create_add_node(vec![a.clone(), b.clone()], Some("add"))
        .unwrap();
    gi.forward_via_node_inner(&add).unwrap();

    let tanh1 = gi.create_tanh_node(add.clone(), Some("tanh1")).unwrap();
    gi.forward_via_node_inner(&tanh1).unwrap();

    let tanh2 = gi
        .create_tanh_node(tanh1.clone(), Some("tanh2"))
        .unwrap();
    gi.forward_via_node_inner(&tanh2).unwrap();

    let tanh3 = gi
        .create_tanh_node(tanh2.clone(), Some("tanh3"))
        .unwrap();
    gi.forward_via_node_inner(&tanh3).unwrap();

    // 3. 验证链式计算正确
    let a_value = a.value().unwrap();
    let b_value = b.value().unwrap();

    let expected = (a_value + b_value).tanh().tanh().tanh();
    let actual = tanh3.value().unwrap();
    assert_eq!(actual, expected);

    // 4. 创建 loss 节点并反向传播
    let target = gi
        .create_basic_input_node(&[2, 1], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(tanh3.clone(), target.clone(), Some("loss"))
        .unwrap();
    let target_value = Tensor::new(&[0.5, 0.5], &[2, 1]);
    target.set_value(Some(&target_value)).unwrap();
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();

    // 5. 验证梯度存在
    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
}

/// 测试: 在已有复杂图上添加节点
#[test]
fn test_add_to_complex_graph() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建一个类似 XOR 的网络结构
    let x = gi.create_basic_input_node(&[2, 1], Some("x")).unwrap();
    let w1 = gi
        .create_parameter_node_seeded(&[4, 2], Some("w1"), 42)
        .unwrap();
    gi.register_parameter("w1".to_string(), Rc::downgrade(&w1))
        .unwrap();
    let b1 = gi
        .create_parameter_node_seeded(&[4, 1], Some("b1"), 43)
        .unwrap();
    gi.register_parameter("b1".to_string(), Rc::downgrade(&b1))
        .unwrap();
    let w2 = gi
        .create_parameter_node_seeded(&[1, 4], Some("w2"), 44)
        .unwrap();
    gi.register_parameter("w2".to_string(), Rc::downgrade(&w2))
        .unwrap();
    let b2 = gi
        .create_parameter_node_seeded(&[1, 1], Some("b2"), 45)
        .unwrap();
    gi.register_parameter("b2".to_string(), Rc::downgrade(&b2))
        .unwrap();

    // 隐藏层
    let wx1 = gi
        .create_mat_mul_node(vec![w1.clone(), x.clone()], None)
        .unwrap();
    let z1 = gi
        .create_add_node(vec![wx1.clone(), b1.clone()], None)
        .unwrap();
    let h = gi.create_tanh_node(z1.clone(), Some("hidden")).unwrap();

    // 输出层
    let wx2 = gi
        .create_mat_mul_node(vec![w2.clone(), h.clone()], None)
        .unwrap();
    let output = gi
        .create_add_node(vec![wx2.clone(), b2.clone()], Some("output"))
        .unwrap();

    // Loss 节点
    let target = gi
        .create_basic_input_node(&[1, 1], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(output.clone(), target.clone(), Some("loss"))
        .unwrap();

    // 2. 进行一轮训练
    let x_value = Tensor::new(&[1.0, 0.0], &[2, 1]);
    let target_value = Tensor::new(&[1.0], &[1, 1]);
    x.set_value(Some(&x_value)).unwrap();
    target.set_value(Some(&target_value)).unwrap();
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();

    // 3. 动态添加一个新的隐藏层节点（NEAT 变异：添加节点）
    let w_new = gi
        .create_parameter_node_seeded(&[1, 4], Some("w_new"), 46)
        .unwrap();
    gi.register_parameter("w_new".to_string(), Rc::downgrade(&w_new))
        .unwrap();
    let new_hidden_out = gi
        .create_mat_mul_node(vec![w_new.clone(), h.clone()], Some("new_hidden_out"))
        .unwrap();

    // 4. 创建新的输出（原输出 + 新隐藏层输出）
    let combined = gi
        .create_add_node(
            vec![output.clone(), new_hidden_out.clone()],
            Some("combined"),
        )
        .unwrap();

    // 新的 loss 节点
    let target2 = gi
        .create_basic_input_node(&[1, 1], Some("target2"))
        .unwrap();
    let loss2 = gi
        .create_mse_mean_node(combined.clone(), target2.clone(), Some("loss2"))
        .unwrap();

    // 5. 清除旧梯度
    gi.zero_grad().unwrap();

    // 6. 对新图进行训练
    let target2_value = Tensor::new(&[1.0], &[1, 1]);
    target2.set_value(Some(&target2_value)).unwrap();
    gi.forward_via_node_inner(&loss2).unwrap();
    gi.backward_via_node_inner(&loss2).unwrap();

    // 7. 所有参数都有梯度
    assert!(w1.grad().is_some());
    assert!(b1.grad().is_some());
    assert!(w2.grad().is_some());
    assert!(b2.grad().is_some());
    assert!(w_new.grad().is_some());
}

/// 测试: 多次清零梯度是安全的
#[test]
fn test_multiple_zero_grad_calls() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    let a = gi
        .create_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    gi.register_parameter("a".to_string(), Rc::downgrade(&a))
        .unwrap();
    let b = gi
        .create_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();
    gi.register_parameter("b".to_string(), Rc::downgrade(&b))
        .unwrap();
    let add = gi
        .create_add_node(vec![a.clone(), b.clone()], None)
        .unwrap();
    let target = gi
        .create_basic_input_node(&[2, 1], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(add.clone(), target.clone(), Some("loss"))
        .unwrap();

    // 设置 target 值
    let target_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    target.set_value(Some(&target_value)).unwrap();

    // Forward 和 backward
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();

    // 多次调用 zero_grad 应该是安全的
    gi.zero_grad().unwrap();
    gi.zero_grad().unwrap();
    gi.zero_grad().unwrap();

    // 梯度已被清除
    assert!(a.grad().is_none());
    assert!(b.grad().is_none());

    // 值仍然保留（Rc 持有引用）
    assert!(add.value().is_some());

    // 可以继续训练
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();

    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
}

/// 测试: 添加节点后不调用 zero_grad 的情况
/// 验证即使不显式清梯度，pass_id 机制也能保证 forward 正确性
#[test]
fn test_add_node_without_explicit_zero_grad() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建初始图
    let a = gi
        .create_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    gi.register_parameter("a".to_string(), Rc::downgrade(&a))
        .unwrap();
    let b = gi
        .create_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();
    gi.register_parameter("b".to_string(), Rc::downgrade(&b))
        .unwrap();
    let add1 = gi
        .create_add_node(vec![a.clone(), b.clone()], None)
        .unwrap();
    let target1 = gi
        .create_basic_input_node(&[2, 1], Some("target1"))
        .unwrap();
    let loss1 = gi
        .create_mse_mean_node(add1.clone(), target1.clone(), Some("loss1"))
        .unwrap();

    // 2. 训练
    let target1_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    target1.set_value(Some(&target1_value)).unwrap();
    gi.forward_via_node_inner(&loss1).unwrap();
    gi.backward_via_node_inner(&loss1).unwrap();

    // 3. 添加新节点但不调用 zero_grad
    let c = gi
        .create_parameter_node_seeded(&[2, 1], Some("c"), 44)
        .unwrap();
    gi.register_parameter("c".to_string(), Rc::downgrade(&c))
        .unwrap();
    let add2 = gi
        .create_add_node(vec![add1.clone(), c.clone()], None)
        .unwrap();

    // 4. 直接对新节点进行 forward
    // pass_id 机制会确保重新计算
    gi.forward_via_node_inner(&add2).unwrap();

    // forward 结果正确
    let add2_value = add2.value().unwrap();
    let a_value = a.value().unwrap();
    let b_value = b.value().unwrap();
    let c_value = c.value().unwrap();

    let expected = a_value.clone() + b_value.clone() + c_value.clone();
    assert_eq!(add2_value, expected);

    // 5. 清零后再对新图 backward
    gi.zero_grad().unwrap();
    let target2 = gi
        .create_basic_input_node(&[2, 1], Some("target2"))
        .unwrap();
    let loss2 = gi
        .create_mse_mean_node(add2.clone(), target2.clone(), Some("loss2"))
        .unwrap();
    let target2_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    target2.set_value(Some(&target2_value)).unwrap();
    gi.forward_via_node_inner(&loss2).unwrap();
    gi.backward_via_node_inner(&loss2).unwrap();

    // 所有参数都有梯度
    let a_grad = a.grad().unwrap();
    let b_grad = b.grad().unwrap();
    let c_grad = c.grad().unwrap();

    // 形状应该正确
    assert_eq!(a_grad.shape(), &[2, 1]);
    assert_eq!(b_grad.shape(), &[2, 1]);
    assert_eq!(c_grad.shape(), &[2, 1]);
}

// ============================================================================
// NEAT 典型场景测试
// ============================================================================

/// 测试: 模拟 NEAT 的"添加节点"变异
/// NEAT 中添加节点是在现有连接中间插入一个新节点
/// 原始: A -> B 变成 A -> NEW -> B
/// 注意: 当前 API 不直接支持"插入"，需要通过旁路模拟
#[test]
fn test_neat_add_node_mutation_simulation() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 初始网络: input -> w1*input -> hidden, hidden -> w2*hidden -> output
    let input = gi
        .create_basic_input_node(&[2, 1], Some("input"))
        .unwrap();
    let w1 = gi
        .create_parameter_node_seeded(&[3, 2], Some("w1"), 42)
        .unwrap();
    gi.register_parameter("w1".to_string(), Rc::downgrade(&w1))
        .unwrap();
    let hidden = gi
        .create_mat_mul_node(vec![w1.clone(), input.clone()], Some("hidden"))
        .unwrap();
    let w2 = gi
        .create_parameter_node_seeded(&[1, 3], Some("w2"), 43)
        .unwrap();
    gi.register_parameter("w2".to_string(), Rc::downgrade(&w2))
        .unwrap();
    let output = gi
        .create_mat_mul_node(vec![w2.clone(), hidden.clone()], Some("output"))
        .unwrap();

    // Loss 节点
    let target = gi
        .create_basic_input_node(&[1, 1], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(output.clone(), target.clone(), Some("loss"))
        .unwrap();

    // 2. 初始训练
    let input_value = Tensor::new(&[1.0, 0.5], &[2, 1]);
    let target_value = Tensor::new(&[1.0], &[1, 1]);
    input.set_value(Some(&input_value)).unwrap();
    target.set_value(Some(&target_value)).unwrap();
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();

    // 3. NEAT 变异: 在 hidden 和 output 之间添加新节点（旁路）
    // hidden -> w_new*hidden -> new_hidden -> tanh -> w3*tanh -> new_output
    // final_output = output + new_output
    let w_new = gi
        .create_parameter_node_seeded(&[3, 3], Some("w_new"), 44)
        .unwrap();
    gi.register_parameter("w_new".to_string(), Rc::downgrade(&w_new))
        .unwrap();
    let new_hidden = gi
        .create_mat_mul_node(vec![w_new.clone(), hidden.clone()], Some("new_hidden"))
        .unwrap();
    let tanh_new = gi
        .create_tanh_node(new_hidden.clone(), Some("tanh_new"))
        .unwrap();

    // 将新节点的输出映射到与原输出相同的维度
    let w3 = gi
        .create_parameter_node_seeded(&[1, 3], Some("w3"), 45)
        .unwrap();
    gi.register_parameter("w3".to_string(), Rc::downgrade(&w3))
        .unwrap();
    let new_output = gi
        .create_mat_mul_node(vec![w3.clone(), tanh_new.clone()], Some("new_output"))
        .unwrap();
    let final_output = gi
        .create_add_node(
            vec![output.clone(), new_output.clone()],
            Some("final_output"),
        )
        .unwrap();

    // 新 loss 节点
    let target2 = gi
        .create_basic_input_node(&[1, 1], Some("target2"))
        .unwrap();
    let loss2 = gi
        .create_mse_mean_node(final_output.clone(), target2.clone(), Some("loss2"))
        .unwrap();

    // 4. 清除旧梯度
    gi.zero_grad().unwrap();

    // 5. 对新图进行训练
    let target2_value = Tensor::new(&[1.0], &[1, 1]);
    target2.set_value(Some(&target2_value)).unwrap();
    gi.forward_via_node_inner(&loss2).unwrap();
    gi.backward_via_node_inner(&loss2).unwrap();

    // 6. 所有参数都有梯度
    assert!(w1.grad().is_some());
    assert!(w2.grad().is_some());
    assert!(w_new.grad().is_some());
    assert!(w3.grad().is_some());

    // 7. 验证图结构：final_output 有两个父节点
    assert_eq!(final_output.parents().len(), 2);
    assert_eq!(final_output.parents()[0].id(), output.id());
    assert_eq!(final_output.parents()[1].id(), new_output.id());
}

/// 测试: 模拟 NEAT 的"添加连接"变异
/// 在两个已存在但未连接的节点之间添加连接
#[test]
fn test_neat_add_connection_mutation_simulation() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建一个有多个并行路径的网络
    let input = gi
        .create_basic_input_node(&[2, 1], Some("input"))
        .unwrap();

    // 路径1: input -> w1 -> h1 -> tanh
    let w1 = gi
        .create_parameter_node_seeded(&[2, 2], Some("w1"), 42)
        .unwrap();
    gi.register_parameter("w1".to_string(), Rc::downgrade(&w1))
        .unwrap();
    let h1 = gi
        .create_mat_mul_node(vec![w1.clone(), input.clone()], Some("h1"))
        .unwrap();
    let h1_tanh = gi
        .create_tanh_node(h1.clone(), Some("h1_tanh"))
        .unwrap();

    // 路径2: input -> w2 -> h2 -> tanh
    let w2 = gi
        .create_parameter_node_seeded(&[2, 2], Some("w2"), 43)
        .unwrap();
    gi.register_parameter("w2".to_string(), Rc::downgrade(&w2))
        .unwrap();
    let h2 = gi
        .create_mat_mul_node(vec![w2.clone(), input.clone()], Some("h2"))
        .unwrap();
    let h2_tanh = gi
        .create_tanh_node(h2.clone(), Some("h2_tanh"))
        .unwrap();

    // 输出: h1_tanh + h2_tanh
    let output = gi
        .create_add_node(vec![h1_tanh.clone(), h2_tanh.clone()], Some("output"))
        .unwrap();

    // Loss 节点
    let target = gi
        .create_basic_input_node(&[2, 1], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(output.clone(), target.clone(), Some("loss"))
        .unwrap();

    // 2. 初始训练
    let input_value = Tensor::new(&[1.0, 0.5], &[2, 1]);
    let target_value = Tensor::new(&[0.5, 0.5], &[2, 1]);
    input.set_value(Some(&input_value)).unwrap();
    target.set_value(Some(&target_value)).unwrap();
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();

    // 3. NEAT 变异: 添加从 h1 到 h2 路径的跨层连接
    // 新结构: h2_enhanced = h2 + w3 * h1
    let w3 = gi
        .create_parameter_node_seeded(&[2, 2], Some("w3"), 44)
        .unwrap();
    gi.register_parameter("w3".to_string(), Rc::downgrade(&w3))
        .unwrap();
    let cross_conn = gi
        .create_mat_mul_node(vec![w3.clone(), h1.clone()], Some("cross_conn"))
        .unwrap();
    let h2_enhanced = gi
        .create_add_node(vec![h2.clone(), cross_conn.clone()], Some("h2_enhanced"))
        .unwrap();
    let h2_enhanced_tanh = gi
        .create_tanh_node(h2_enhanced.clone(), Some("h2_enhanced_tanh"))
        .unwrap();

    // 新输出
    let new_output = gi
        .create_add_node(
            vec![h1_tanh.clone(), h2_enhanced_tanh.clone()],
            Some("new_output"),
        )
        .unwrap();

    // 新 loss 节点
    let target2 = gi
        .create_basic_input_node(&[2, 1], Some("target2"))
        .unwrap();
    let loss2 = gi
        .create_mse_mean_node(new_output.clone(), target2.clone(), Some("loss2"))
        .unwrap();

    // 4. 清除旧梯度
    gi.zero_grad().unwrap();

    // 5. 对新图进行训练
    let target2_value = Tensor::new(&[0.5, 0.5], &[2, 1]);
    target2.set_value(Some(&target2_value)).unwrap();
    gi.forward_via_node_inner(&loss2).unwrap();
    gi.backward_via_node_inner(&loss2).unwrap();

    // 6. 所有参数都有梯度
    assert!(w1.grad().is_some());
    assert!(w2.grad().is_some());
    assert!(w3.grad().is_some());
}

// ============================================================================
// 稳定性和正确性测试
// ============================================================================

/// 测试: 验证动态添加后的梯度数值正确性
#[test]
fn test_gradient_correctness_after_dynamic_add() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建初始图: y = a + b，loss = MSE(y, target)
    let a = gi
        .create_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    gi.register_parameter("a".to_string(), Rc::downgrade(&a))
        .unwrap();
    let b = gi
        .create_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();
    gi.register_parameter("b".to_string(), Rc::downgrade(&b))
        .unwrap();
    let y = gi
        .create_add_node(vec![a.clone(), b.clone()], Some("y"))
        .unwrap();
    let target = gi
        .create_basic_input_node(&[2, 1], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(y.clone(), target.clone(), Some("loss"))
        .unwrap();

    // 2. 设置 target = a + b（使得梯度接近 0）
    let a_value = a.value().unwrap();
    let b_value = b.value().unwrap();
    let target_value = a_value.clone() + b_value.clone();
    target.set_value(Some(&target_value)).unwrap();
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();

    // 保存原始梯度
    let a_grad_original = a.grad().unwrap();
    let b_grad_original = b.grad().unwrap();

    // 3. 添加新节点: z = y + c = a + b + c，loss2 = MSE(z, target2)
    let c = gi
        .create_parameter_node_seeded(&[2, 1], Some("c"), 44)
        .unwrap();
    gi.register_parameter("c".to_string(), Rc::downgrade(&c))
        .unwrap();
    let z = gi
        .create_add_node(vec![y.clone(), c.clone()], Some("z"))
        .unwrap();
    let target2 = gi
        .create_basic_input_node(&[2, 1], Some("target2"))
        .unwrap();
    let loss2 = gi
        .create_mse_mean_node(z.clone(), target2.clone(), Some("loss2"))
        .unwrap();

    // 4. 清除旧梯度
    gi.zero_grad().unwrap();

    // 5. 设置 target2 = a + b + c（使得梯度接近 0）
    let c_value = c.value().unwrap();
    let target2_value = a_value.clone() + b_value.clone() + c_value.clone();
    target2.set_value(Some(&target2_value)).unwrap();
    gi.forward_via_node_inner(&loss2).unwrap();
    gi.backward_via_node_inner(&loss2).unwrap();

    // 6. 验证梯度的数值正确性
    // 当 target2 = z 时，梯度应该接近 0
    let a_grad_new = a.grad().unwrap();
    let b_grad_new = b.grad().unwrap();
    let c_grad_new = c.grad().unwrap();

    // 梯度形状正确
    assert_eq!(a_grad_new.shape(), &[2, 1]);
    assert_eq!(b_grad_new.shape(), &[2, 1]);
    assert_eq!(c_grad_new.shape(), &[2, 1]);

    // 当 z = target2 时，梯度应该接近 0
    assert_abs_diff_eq!(a_grad_new[[0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(a_grad_new[[1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(b_grad_new[[0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(b_grad_new[[1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(c_grad_new[[0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(c_grad_new[[1, 0]], 0.0, epsilon = 1e-5);

    // 7. 原始梯度也应该接近 0（因为 target = y）
    assert_abs_diff_eq!(a_grad_original[[0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(b_grad_original[[0, 0]], 0.0, epsilon = 1e-5);
}

/// 测试: 验证 forward pass ID 在动态添加后的行为
#[test]
fn test_pass_id_behavior_after_dynamic_add() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建初始图
    let a = gi
        .create_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    let b = gi
        .create_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();
    let add1 = gi
        .create_add_node(vec![a.clone(), b.clone()], None)
        .unwrap();

    // 2. Forward
    gi.forward_via_node_inner(&add1).unwrap();
    let pass_id_after_first_forward = gi.last_forward_pass_id();
    assert_eq!(pass_id_after_first_forward, 1);

    // 验证节点的 pass_id
    assert_eq!(a.last_forward_pass_id(), 1);
    assert_eq!(b.last_forward_pass_id(), 1);
    assert_eq!(add1.last_forward_pass_id(), 1);

    // 3. 添加新节点
    let c = gi
        .create_parameter_node_seeded(&[2, 1], Some("c"), 44)
        .unwrap();
    let add2 = gi
        .create_add_node(vec![add1.clone(), c.clone()], None)
        .unwrap();

    // 新节点的 pass_id 应该是 0
    assert_eq!(c.last_forward_pass_id(), 0);
    assert_eq!(add2.last_forward_pass_id(), 0);

    // 4. Forward 新节点
    gi.forward_via_node_inner(&add2).unwrap();
    let pass_id_after_second_forward = gi.last_forward_pass_id();
    assert_eq!(pass_id_after_second_forward, 2);

    // 所有相关节点的 pass_id 都更新
    assert_eq!(a.last_forward_pass_id(), 2);
    assert_eq!(b.last_forward_pass_id(), 2);
    assert_eq!(c.last_forward_pass_id(), 2);
    assert_eq!(add1.last_forward_pass_id(), 2);
    assert_eq!(add2.last_forward_pass_id(), 2);
}
