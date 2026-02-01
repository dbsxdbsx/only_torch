/*
 * NodeInner 单元测试
 *
 * 测试方案 C 的核心数据结构：
 * - 基本创建和访问
 * - Rc 引用计数管理
 * - 级联释放机制
 */

use crate::nn::NodeId;
use crate::nn::nodes::NodeInner;
use crate::nn::nodes::raw_node::{Add, InputVariant, Parameter};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use std::rc::Rc;

/// 辅助函数：创建一个参数节点的 NodeType
fn make_param(shape: &[usize]) -> crate::nn::nodes::NodeType {
    Parameter::new(shape).unwrap().into()
}

#[test]
fn test_node_inner_leaf_creation() {
    // 创建一个参数节点（叶子节点）
    let node = NodeInner::new_leaf(
        NodeId(1),
        Some("test_param".to_string()),
        make_param(&[2, 3]),
    );

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

// ==================== 前向传播测试（Step 2.5.2）====================

/// 辅助函数：创建 Add 节点的 NodeType
fn make_add(parent_shapes: &[&[usize]]) -> crate::nn::nodes::NodeType {
    let dynamic_shapes: Vec<DynamicShape> = parent_shapes
        .iter()
        .map(|s| DynamicShape::fixed(s))
        .collect();
    Add::new_from_shapes(parent_shapes, &dynamic_shapes)
        .unwrap()
        .into()
}

/// 辅助函数：创建 Input（Data）节点的 NodeType
fn make_input(shape: &[usize]) -> crate::nn::nodes::NodeType {
    InputVariant::new_data(shape).unwrap().into()
}

#[test]
fn test_forward_recursive_basic() {
    // 简单链式结构：input1 + input2 -> add
    let input1 = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3])));
    let input2 = Rc::new(NodeInner::new_leaf(NodeId(2), None, make_param(&[2, 3])));

    // 设置输入值
    input1.set_value(Some(&Tensor::ones(&[2, 3]))).unwrap();
    input2.set_value(Some(&Tensor::ones(&[2, 3]))).unwrap();

    // 创建 Add 节点
    let add = Rc::new(NodeInner::new(
        NodeId(3),
        None,
        make_add(&[&[2, 3], &[2, 3]]),
        vec![Rc::clone(&input1), Rc::clone(&input2)],
    ));

    // 执行前向传播
    add.forward_recursive(1, false).unwrap();

    // 验证结果：1 + 1 = 2
    let result = add.value().unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    assert!(
        result
            .data_as_slice()
            .iter()
            .all(|&x| (x - 2.0).abs() < 1e-6)
    );

    // 验证 pass_id 更新
    assert_eq!(add.last_forward_pass_id(), 1);
    assert_eq!(input1.last_forward_pass_id(), 1);
    assert_eq!(input2.last_forward_pass_id(), 1);
}

#[test]
fn test_forward_recursive_diamond_dag() {
    // 菱形 DAG：测试 pass_id 去重
    //
    //      input
    //      /   \
    //   add1   add2  (both add input to itself)
    //      \   /
    //      output (add1 + add2)
    //
    let input = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3])));
    input.set_value(Some(&Tensor::ones(&[2, 3]))).unwrap();

    // add1 = input + input (由于 Add 支持多个输入，这里模拟两个分支)
    let add1 = Rc::new(NodeInner::new(
        NodeId(2),
        Some("add1".to_string()),
        make_add(&[&[2, 3], &[2, 3]]),
        vec![Rc::clone(&input), Rc::clone(&input)],
    ));

    let add2 = Rc::new(NodeInner::new(
        NodeId(3),
        Some("add2".to_string()),
        make_add(&[&[2, 3], &[2, 3]]),
        vec![Rc::clone(&input), Rc::clone(&input)],
    ));

    // output = add1 + add2
    let output = Rc::new(NodeInner::new(
        NodeId(4),
        Some("output".to_string()),
        make_add(&[&[2, 3], &[2, 3]]),
        vec![Rc::clone(&add1), Rc::clone(&add2)],
    ));

    // 执行前向传播
    output.forward_recursive(1, false).unwrap();

    // 验证结果：(1+1) + (1+1) = 4
    let result = output.value().unwrap();
    assert!(
        result
            .data_as_slice()
            .iter()
            .all(|&x| (x - 4.0).abs() < 1e-6)
    );

    // 验证所有节点的 pass_id 都被更新
    assert_eq!(input.last_forward_pass_id(), 1);
    assert_eq!(add1.last_forward_pass_id(), 1);
    assert_eq!(add2.last_forward_pass_id(), 1);
    assert_eq!(output.last_forward_pass_id(), 1);
}

#[test]
fn test_forward_recursive_skip_computed() {
    // 测试 pass_id 去重：同一 pass_id 不重复计算
    let input = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3])));
    input.set_value(Some(&Tensor::ones(&[2, 3]))).unwrap();

    let add = Rc::new(NodeInner::new(
        NodeId(2),
        None,
        make_add(&[&[2, 3], &[2, 3]]),
        vec![Rc::clone(&input), Rc::clone(&input)],
    ));

    // 第一次前向传播
    add.forward_recursive(1, false).unwrap();
    let first_result = add.value().unwrap();

    // 修改输入值
    input.set_value(Some(&Tensor::zeros(&[2, 3]))).unwrap();

    // 同一 pass_id 再次调用，应跳过计算
    add.forward_recursive(1, false).unwrap();
    let second_result = add.value().unwrap();

    // 结果应该相同（因为被跳过）
    assert_eq!(first_result.data_as_slice(), second_result.data_as_slice());

    // 新 pass_id 会重新计算
    add.forward_recursive(2, false).unwrap();
    let third_result = add.value().unwrap();

    // 结果应该是 0 + 0 = 0
    assert!(third_result.data_as_slice().iter().all(|&x| x.abs() < 1e-6));
}

#[test]
fn test_forward_recursive_leaf_no_value() {
    // 测试叶子节点没有值时的错误
    let input = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_input(&[2, 3])));

    // Input（Data）节点不会自动初始化值
    let result = input.forward_recursive(1, false);

    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("没有值"));
}

// ==================== 反向传播测试（2.6.7）====================

/// 辅助函数：创建 [1,1] 形状的 Add 节点（简化测试代码）
fn make_add_1x1() -> crate::nn::nodes::NodeType {
    make_add(&[&[1, 1], &[1, 1]])
}

/// 测试基础反向传播（简单链式）
///
/// 结构：input -> add (input + input) -> loss
/// 验证梯度正确传播到输入节点
#[test]
fn test_backward_propagate_basic() {
    // input (值=2) -> add -> loss
    let input = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[1, 1])));
    input
        .set_value(Some(&Tensor::new(&[2.0], &[1, 1])))
        .unwrap();

    // add = input + input = 4
    let add = Rc::new(NodeInner::new(
        NodeId(2),
        None,
        make_add_1x1(),
        vec![input.clone(), input.clone()],
    ));

    // 前向传播
    add.forward_recursive(1, false).unwrap();
    assert_eq!(add.value().unwrap()[[0, 0]], 4.0);

    // 设置 loss 梯度为 1（模拟 MSE 等损失函数）
    add.set_grad(Some(&Tensor::ones(&[1, 1]))).unwrap();

    // 反向传播
    add.backward_propagate(1).unwrap();

    // 验证 input 的梯度：d(add)/d(input) = 1 + 1 = 2（因为 input 出现两次）
    let input_grad = input.grad().unwrap();
    assert_eq!(input_grad[[0, 0]], 2.0);
}

/// 测试菱形 DAG 梯度累积
///
/// 结构：
///     input
///    /     \
///  add1   add2  (都是 input + input)
///    \     /
///     output (add1 + add2)
///
/// 验证 input 的梯度正确累积来自两条路径的贡献
#[test]
fn test_backward_propagate_diamond_dag() {
    // input (值=1)
    let input = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[1, 1])));
    input.set_value(Some(&Tensor::ones(&[1, 1]))).unwrap();

    // add1 = input + input = 2
    let add1 = Rc::new(NodeInner::new(
        NodeId(2),
        None,
        make_add_1x1(),
        vec![input.clone(), input.clone()],
    ));

    // add2 = input + input = 2
    let add2 = Rc::new(NodeInner::new(
        NodeId(3),
        None,
        make_add_1x1(),
        vec![input.clone(), input.clone()],
    ));

    // output = add1 + add2 = 4
    let output = Rc::new(NodeInner::new(
        NodeId(4),
        None,
        make_add_1x1(),
        vec![add1.clone(), add2.clone()],
    ));

    // 前向传播
    output.forward_recursive(1, false).unwrap();
    assert_eq!(output.value().unwrap()[[0, 0]], 4.0);

    // 设置 output 梯度为 1
    output.set_grad(Some(&Tensor::ones(&[1, 1]))).unwrap();

    // 反向传播
    output.backward_propagate(1).unwrap();

    // 验证梯度累积：
    // - output 对 add1 的梯度 = 1
    // - output 对 add2 的梯度 = 1
    // - add1 对 input 的梯度 = 2（input 出现两次）
    // - add2 对 input 的梯度 = 2（input 出现两次）
    // - input 总梯度 = 2 + 2 = 4
    let input_grad = input.grad().unwrap();
    assert_eq!(input_grad[[0, 0]], 4.0);

    // 中间节点的梯度
    assert_eq!(add1.grad().unwrap()[[0, 0]], 1.0);
    assert_eq!(add2.grad().unwrap()[[0, 0]], 1.0);
}

/// 测试 detach 行为（梯度不穿透）
///
/// 结构：input -> add (detached) -> output
/// 验证 detach 节点阻止梯度传播
#[test]
fn test_backward_propagate_detach() {
    // input
    let input = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[1, 1])));
    input.set_value(Some(&Tensor::ones(&[1, 1]))).unwrap();

    // add1 = input + input（被 detach）
    let add1 = Rc::new(NodeInner::new(
        NodeId(2),
        None,
        make_add_1x1(),
        vec![input.clone(), input.clone()],
    ));
    add1.set_detached(true); // 设置为 detached

    // output = add1 + add1
    let output = Rc::new(NodeInner::new(
        NodeId(3),
        None,
        make_add_1x1(),
        vec![add1.clone(), add1.clone()],
    ));

    // 前向传播
    output.forward_recursive(1, false).unwrap();

    // 设置 output 梯度
    output.set_grad(Some(&Tensor::ones(&[1, 1]))).unwrap();

    // 反向传播
    output.backward_propagate(1).unwrap();

    // 验证：add1 收到梯度（作为 output 的父节点）
    assert_eq!(add1.grad().unwrap()[[0, 0]], 2.0);

    // 验证：input 没有梯度（因为 add1 被 detach，梯度不再向上传播）
    assert!(input.grad().is_none());
}

/// 测试 pass_id 去重
///
/// 验证同一 pass_id 不会重复处理节点
#[test]
fn test_backward_propagate_pass_id() {
    let input = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[1, 1])));
    input.set_value(Some(&Tensor::ones(&[1, 1]))).unwrap();

    let add = Rc::new(NodeInner::new(
        NodeId(2),
        None,
        make_add_1x1(),
        vec![input.clone(), input.clone()],
    ));

    // 前向传播
    add.forward_recursive(1, false).unwrap();

    // 第一次反向传播
    add.set_grad(Some(&Tensor::ones(&[1, 1]))).unwrap();
    add.backward_propagate(1).unwrap();
    let grad1 = input.grad().unwrap()[[0, 0]];
    assert_eq!(grad1, 2.0);

    // 同一 pass_id 再次调用，不应重复处理
    add.backward_propagate(1).unwrap();
    let grad2 = input.grad().unwrap()[[0, 0]];
    assert_eq!(grad2, 2.0); // 梯度不变

    // 新 pass_id，重新设置梯度
    input.clear_grad().unwrap();
    add.set_grad(Some(&Tensor::ones(&[1, 1]))).unwrap();
    add.backward_propagate(2).unwrap();
    let grad3 = input.grad().unwrap()[[0, 0]];
    assert_eq!(grad3, 2.0); // 新的一轮
}
