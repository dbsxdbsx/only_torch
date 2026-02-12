/*
 * @Description  : Forward pass ID 机制测试
 *
 * 验证 forward pass ID 的递增、部分前向、错误回滚等行为。
 * 使用底层 GraphInner API：Graph::new() -> inner_rc() -> borrow_mut()。
 */

use crate::nn::Graph;
use crate::tensor::Tensor;

/// 测试部分前向传播时的 pass_id 更新
///
/// 场景：z = x + y, w = x + y，先 forward(add1)，再 forward(final = add1 + add2)
#[test]
fn test_forward_with_partial_forward_propagation() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建计算图：z = x + y, w = x + y (两个节点都依赖同样的 add 操作)
    let x = gi.create_basic_input_node(&[2, 1], Some("x")).unwrap();
    let y = gi.create_basic_input_node(&[2, 1], Some("y")).unwrap();
    let add1 = gi
        .create_add_node(vec![x.clone(), y.clone()], Some("add1"))
        .unwrap();
    let add2 = gi
        .create_add_node(vec![x.clone(), y.clone()], Some("add2"))
        .unwrap();

    // 2. 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let y_value = Tensor::new(&[0.5, 1.5], &[2, 1]);
    x.set_value(Some(&x_value)).unwrap();
    y.set_value(Some(&y_value)).unwrap();

    // 3. 第 1 次前向传播 add1
    gi.forward_via_node_inner(&add1).unwrap();
    let first_pass_id = gi.last_forward_pass_id();

    // 验证参与计算的节点 pass_id 都是第 1 次的
    assert_eq!(x.last_forward_pass_id(), first_pass_id);
    assert_eq!(y.last_forward_pass_id(), first_pass_id);
    assert_eq!(add1.last_forward_pass_id(), first_pass_id);

    // add2 还未被计算，pass_id 应为 0
    assert_eq!(add2.last_forward_pass_id(), 0);

    // 4. 创建菱形依赖：final = add1 + add2
    let final_add = gi
        .create_add_node(vec![add1.clone(), add2.clone()], Some("final"))
        .unwrap();

    // 5. 前向传播 final，会触发 add1 和 add2 的计算
    gi.forward_via_node_inner(&final_add).unwrap();
    let second_pass_id = gi.last_forward_pass_id();
    assert_eq!(second_pass_id, first_pass_id + 1);

    // 验证所有节点都更新到新的 pass_id
    assert_eq!(x.last_forward_pass_id(), second_pass_id);
    assert_eq!(y.last_forward_pass_id(), second_pass_id);
    assert_eq!(add1.last_forward_pass_id(), second_pass_id);
    assert_eq!(add2.last_forward_pass_id(), second_pass_id);
    assert_eq!(final_add.last_forward_pass_id(), second_pass_id);

    // 6. 再次前向传播 final，验证所有节点重新计算
    gi.forward_via_node_inner(&final_add).unwrap();
    let third_pass_id = gi.last_forward_pass_id();
    assert_eq!(third_pass_id, second_pass_id + 1);

    // 验证所有节点的 pass_id 都更新
    assert_eq!(x.last_forward_pass_id(), third_pass_id);
    assert_eq!(y.last_forward_pass_id(), third_pass_id);
    assert_eq!(add1.last_forward_pass_id(), third_pass_id);
    assert_eq!(add2.last_forward_pass_id(), third_pass_id);
    assert_eq!(final_add.last_forward_pass_id(), third_pass_id);
}

/// 测试 pass_id 随每次 forward 递增
#[test]
fn test_forward_pass_id_increment() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建简单计算图：y = x + b
    let x = gi.create_basic_input_node(&[2, 1], Some("x")).unwrap();
    let b = gi.create_parameter_node(&[2, 1], Some("b")).unwrap();
    let y = gi
        .create_add_node(vec![x.clone(), b.clone()], Some("y"))
        .unwrap();

    // 2. 初始状态：pass_id 应为 0
    assert_eq!(gi.last_forward_pass_id(), 0);

    // 3. 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let b_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    x.set_value(Some(&x_value)).unwrap();
    b.set_value(Some(&b_value)).unwrap();

    // 4. 第 1 次前向传播
    gi.forward_via_node_inner(&y).unwrap();
    assert_eq!(gi.last_forward_pass_id(), 1);

    // 5. 第 2 次前向传播
    gi.forward_via_node_inner(&y).unwrap();
    assert_eq!(gi.last_forward_pass_id(), 2);

    // 6. 第 3 次前向传播
    gi.forward_via_node_inner(&y).unwrap();
    assert_eq!(gi.last_forward_pass_id(), 3);
}

/// 测试前向传播失败时 pass_id 回滚
#[test]
fn test_pass_id_rollback_on_forward_error() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建计算图：y = x + b
    let x = gi.create_basic_input_node(&[2, 1], Some("x")).unwrap();
    let b = gi.create_parameter_node(&[2, 1], Some("b")).unwrap();
    let y = gi
        .create_add_node(vec![x.clone(), b.clone()], Some("y"))
        .unwrap();

    // 2. 只设置 b 的值，故意不设置 x
    let b_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    b.set_value(Some(&b_value)).unwrap();

    // 3. 记录初始 pass_id
    let initial_forward_pass_id = gi.last_forward_pass_id();
    assert_eq!(initial_forward_pass_id, 0);

    // 4. 尝试前向传播，应失败（x 无值）
    let forward_result = gi.forward_via_node_inner(&y);
    assert!(forward_result.is_err());

    // 验证失败后 pass_id 未变化
    assert_eq!(gi.last_forward_pass_id(), initial_forward_pass_id);

    // 5. 设置 x 的值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    x.set_value(Some(&x_value)).unwrap();

    // 6. 前向传播应成功
    gi.forward_via_node_inner(&y).unwrap();
    assert_eq!(gi.last_forward_pass_id(), 1);
}
