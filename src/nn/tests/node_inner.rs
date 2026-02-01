/*
 * NodeInner 单元测试
 *
 * 测试方案 C 的核心数据结构：
 * - 基本创建和访问
 * - Rc 引用计数管理
 * - 级联释放机制
 */

use crate::nn::nodes::raw_node::Parameter;
use crate::nn::nodes::NodeInner;
use crate::nn::NodeId;
use crate::tensor::Tensor;
use std::rc::Rc;

/// 辅助函数：创建一个参数节点的 NodeType
fn make_param(shape: &[usize]) -> crate::nn::nodes::NodeType {
    Parameter::new(shape).unwrap().into()
}

#[test]
fn test_node_inner_leaf_creation() {
    // 创建一个参数节点（叶子节点）
    let node = NodeInner::new_leaf(NodeId(1), Some("test_param".to_string()), make_param(&[2, 3]));

    assert_eq!(node.id(), NodeId(1));
    assert_eq!(node.name(), Some("test_param"));
    assert!(node.is_leaf());
    assert!(node.is_parameter());
    assert!(!node.is_detached());
}

#[test]
fn test_node_inner_with_parents() {
    // 创建两个父节点
    let parent1 = Rc::new(NodeInner::new_leaf(
        NodeId(1),
        Some("parent1".to_string()),
        make_param(&[2, 3]),
    ));

    let parent2 = Rc::new(NodeInner::new_leaf(
        NodeId(2),
        Some("parent2".to_string()),
        make_param(&[3, 4]),
    ));

    // 创建子节点，引用两个父节点
    let child = NodeInner::new(
        NodeId(3),
        Some("child".to_string()),
        make_param(&[2, 4]),
        vec![Rc::clone(&parent1), Rc::clone(&parent2)],
    );

    assert_eq!(child.parents().len(), 2);
    assert!(!child.is_leaf());

    // 验证父节点引用正确
    assert_eq!(child.parents()[0].id(), NodeId(1));
    assert_eq!(child.parents()[1].id(), NodeId(2));
}

#[test]
fn test_node_inner_pass_id() {
    let node = NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3]));

    assert_eq!(node.last_forward_pass_id(), 0);
    assert_eq!(node.last_backward_pass_id(), 0);

    node.set_last_forward_pass_id(42);
    node.set_last_backward_pass_id(100);

    assert_eq!(node.last_forward_pass_id(), 42);
    assert_eq!(node.last_backward_pass_id(), 100);
}

#[test]
fn test_node_inner_detach() {
    let node = NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3]));

    assert!(!node.is_detached());
    node.set_detached(true);
    assert!(node.is_detached());
}

#[test]
fn test_node_inner_value_access() {
    let node = NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3]));

    // 获取值（Parameter 会自动初始化）
    let value = node.value().unwrap();
    assert_eq!(value.shape(), &[2, 3]);

    // 设置新值
    let new_value = Tensor::zeros(&[2, 3]);
    node.set_value(Some(&new_value)).unwrap();

    let updated = node.value().unwrap();
    assert_eq!(updated.shape(), &[2, 3]);
    // 验证所有元素都是 0
    let data = updated.data_as_slice();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_node_inner_rc_lifecycle() {
    // 测试 Rc 引用计数正确管理生命周期

    let parent = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3])));

    // 初始引用计数为 1
    assert_eq!(Rc::strong_count(&parent), 1);

    // 创建子节点，引用父节点
    let child = Rc::new(NodeInner::new(
        NodeId(2),
        None,
        make_param(&[2, 3]),
        vec![Rc::clone(&parent)],
    ));

    // 父节点引用计数变为 2（原引用 + child.parents）
    assert_eq!(Rc::strong_count(&parent), 2);
    assert_eq!(Rc::strong_count(&child), 1);

    // 丢弃 child
    drop(child);

    // 父节点引用计数恢复为 1
    assert_eq!(Rc::strong_count(&parent), 1);
}

#[test]
fn test_node_inner_cascade_release() {
    // 测试级联释放：loss -> hidden -> input
    // 当 loss 释放时，如果 hidden 和 input 无其他引用，也会被释放

    let input = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3])));

    let hidden = Rc::new(NodeInner::new(
        NodeId(2),
        None,
        make_param(&[2, 3]),
        vec![Rc::clone(&input)],
    ));

    let loss = Rc::new(NodeInner::new(
        NodeId(3),
        None,
        make_param(&[2, 3]),
        vec![Rc::clone(&hidden)],
    ));

    // 引用计数：input=2, hidden=2, loss=1
    assert_eq!(Rc::strong_count(&input), 2);
    assert_eq!(Rc::strong_count(&hidden), 2);
    assert_eq!(Rc::strong_count(&loss), 1);

    // 释放原始引用
    let loss_weak = Rc::downgrade(&loss);
    let hidden_weak = Rc::downgrade(&hidden);
    let input_weak = Rc::downgrade(&input);

    drop(input);
    drop(hidden);

    // loss 仍然通过 parents 链持有整个图
    assert!(loss_weak.upgrade().is_some());
    assert!(hidden_weak.upgrade().is_some());
    assert!(input_weak.upgrade().is_some());

    // 释放 loss，整个链级联释放
    drop(loss);

    assert!(loss_weak.upgrade().is_none());
    assert!(hidden_weak.upgrade().is_none());
    assert!(input_weak.upgrade().is_none());
}

#[test]
fn test_node_inner_diamond_dag() {
    // 测试菱形 DAG 结构（多路径汇合）
    //
    //      input
    //      /   \
    //   left   right
    //      \   /
    //      output
    //
    // output 同时引用 left 和 right，两者都引用 input

    let input = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3])));

    let left = Rc::new(NodeInner::new(
        NodeId(2),
        Some("left".to_string()),
        make_param(&[2, 3]),
        vec![Rc::clone(&input)],
    ));

    let right = Rc::new(NodeInner::new(
        NodeId(3),
        Some("right".to_string()),
        make_param(&[2, 3]),
        vec![Rc::clone(&input)],
    ));

    let output = Rc::new(NodeInner::new(
        NodeId(4),
        Some("output".to_string()),
        make_param(&[2, 3]),
        vec![Rc::clone(&left), Rc::clone(&right)],
    ));

    // 引用计数：input=3（self + left + right），left=2，right=2，output=1
    assert_eq!(Rc::strong_count(&input), 3);
    assert_eq!(Rc::strong_count(&left), 2);
    assert_eq!(Rc::strong_count(&right), 2);
    assert_eq!(Rc::strong_count(&output), 1);

    // 释放原始引用
    let input_weak = Rc::downgrade(&input);
    drop(input);
    drop(left);
    drop(right);

    // output 仍然持有整个图
    assert!(input_weak.upgrade().is_some());

    // 释放 output，整个图释放
    drop(output);
    assert!(input_weak.upgrade().is_none());
}

#[test]
fn test_node_inner_type_info() {
    let node = NodeInner::new_leaf(NodeId(1), Some("my_param".to_string()), make_param(&[2, 3]));

    assert!(node.is_parameter());
    assert!(!node.is_input());
    assert_eq!(node.shape(), vec![2, 3]);
    assert_eq!(node.type_name(), "Parameter");
}
