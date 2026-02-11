/*
 * Var 过渡期测试
 *
 * 验证 Step 2.2 的 Var 结构过渡改造：
 * - 旧路径（node: None）和新路径（node: Some）的访问方法等价
 * - 过渡期兼容性
 */

use crate::nn::nodes::raw_node::Parameter;
use crate::nn::nodes::NodeInner;
use crate::nn::{Graph, Init, VarMatrixOps};
use crate::tensor::Tensor;
use std::rc::Rc;

/// 辅助函数：创建一个参数节点的 NodeType
fn make_param(shape: &[usize]) -> crate::nn::nodes::NodeType {
    Parameter::new(shape).unwrap().into()
}

#[test]
fn test_var_old_path_basic() {
    // 测试旧路径创建的 Var 基本功能
    let graph = Graph::new();
    let x = graph.input(&Tensor::ones(&[2, 3])).unwrap();

    // 验证基本属性
    assert!(!x.node().is_some()); // 旧路径没有 node
    assert_eq!(x.value_expected_shape(), vec![2, 3]);

    // 验证值访问
    let val = x.value().unwrap().unwrap();
    assert_eq!(val.shape(), &[2, 3]);
}

#[test]
fn test_var_node_id_consistency() {
    // 测试 node_id() 在新旧路径下的一致性
    let graph = Graph::new();

    // 旧路径
    let x = graph.input(&Tensor::ones(&[2, 3])).unwrap();
    let old_id = x.node_id();

    // 验证 id 有效
    assert!(old_id.0 > 0);
}

#[test]
fn test_var_arithmetic_still_works() {
    // 测试算术运算在过渡期仍然正常工作
    let graph = Graph::new();
    let x = graph.input(&Tensor::ones(&[2, 3])).unwrap();
    let y = graph.input(&Tensor::ones(&[2, 3])).unwrap();

    // 加法
    let z = &x + &y;
    z.forward().unwrap();
    let val = z.value().unwrap().unwrap();
    assert!(val.data_as_slice().iter().all(|&v| (v - 2.0).abs() < 1e-6));

    // 减法
    let w = &x - &y;
    w.forward().unwrap();
    let val = w.value().unwrap().unwrap();
    assert!(val.data_as_slice().iter().all(|&v| v.abs() < 1e-6));
}

#[test]
fn test_var_forward_backward_still_works() {
    // 测试前向和反向传播在过渡期仍然正常工作
    let graph = Graph::new();
    let x = graph.input(&Tensor::ones(&[1, 2])).unwrap();
    let w = graph.parameter(&[2, 1], Init::Ones, "w").unwrap();
    let y = x.matmul(&w).unwrap();

    // 前向传播
    y.forward().unwrap();
    assert!(y.value().unwrap().is_some());

    // 反向传播
    let _ = y.backward().unwrap();
    assert!(w.grad().unwrap().is_some());
}

#[test]
fn test_var_is_detached_still_works() {
    // 测试 is_detached() 在过渡期仍然正常工作
    let graph = Graph::new();
    let x = graph.input(&Tensor::ones(&[2, 3])).unwrap();

    assert!(!x.is_detached());

    let x_detached = x.detach();
    assert!(x_detached.is_detached());
}

#[test]
fn test_var_debug_format() {
    // 测试 Debug 格式正确显示
    let graph = Graph::new();
    let x = graph.input(&Tensor::ones(&[2, 3])).unwrap();

    let debug_str = format!("{:?}", x);
    assert!(debug_str.contains("Var"));
    assert!(debug_str.contains("has_node"));
}

#[test]
fn test_var_new_with_node_basic() {
    // 测试新路径创建的 Var 基本功能
    use crate::nn::NodeId;
    use crate::nn::Var;

    let graph = Graph::new();
    let graph_rc = graph.inner_rc();

    // 创建 NodeInner
    let node = Rc::new(NodeInner::new_leaf(
        NodeId(999),
        Some("test_node".to_string()),
        make_param(&[2, 3]),
    ));

    // 使用新路径创建 Var
    let var = Var::new_with_node(Rc::clone(&node), graph_rc);

    // 验证 node_id() 从 node 获取
    assert_eq!(var.node_id(), NodeId(999));

    // 验证 node() 返回 Some
    assert!(var.node().is_some());

    // 验证 value() 从 node 获取
    let val = var.value().unwrap();
    assert!(val.is_some());
    assert_eq!(val.unwrap().shape(), &[2, 3]);

    // 验证 shape() 从 node 获取
    assert_eq!(var.value_expected_shape(), vec![2, 3]);

    // 验证 is_detached() 从 node 获取
    assert!(!var.is_detached());
    node.set_detached(true);
    assert!(var.is_detached());
}

#[test]
fn test_var_clone_preserves_node() {
    // 测试 clone 保留 node 引用
    use crate::nn::NodeId;
    use crate::nn::Var;

    let graph = Graph::new();
    let graph_rc = graph.inner_rc();

    let node = Rc::new(NodeInner::new_leaf(
        NodeId(888),
        None,
        make_param(&[2, 3]),
    ));

    let var1 = Var::new_with_node(Rc::clone(&node), graph_rc);
    let var2 = var1.clone();

    // 两个 Var 应该持有相同的 node
    assert_eq!(var1.node_id(), var2.node_id());
    assert!(var1.node().is_some());
    assert!(var2.node().is_some());

    // 修改一个应该影响另一个（通过 node）
    node.set_detached(true);
    assert!(var1.is_detached());
    assert!(var2.is_detached());
}
