/*
 * GraphInner 参数注册表测试
 *
 * 验证 Step 2.3 的参数注册表功能：
 * - 参数注册和查询
 * - 弱引用行为（参数释放后自动失效）
 * - 清理已失效参数
 */

use crate::nn::nodes::NodeInner;
use crate::nn::nodes::raw_node::Parameter;
use crate::nn::{Graph, NodeId};
use std::rc::Rc;

/// 辅助函数：创建一个参数节点的 NodeType
fn make_param(shape: &[usize]) -> crate::nn::nodes::NodeType {
    Parameter::new(shape).unwrap().into()
}

#[test]
fn test_register_parameter_basic() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 创建 NodeInner
    let node = Rc::new(NodeInner::new_leaf(
        NodeId(1),
        Some("test_param".to_string()),
        make_param(&[2, 3]),
    ));

    // 注册参数
    inner
        .borrow_mut()
        .register_parameter("layer1.weight".to_string(), Rc::downgrade(&node))
        .unwrap();

    // 验证参数已注册
    assert!(inner.borrow().has_parameter("layer1.weight"));
    assert_eq!(inner.borrow().registered_parameters_count(), 1);
    assert_eq!(inner.borrow().valid_parameters_count(), 1);
}

#[test]
fn test_get_parameter() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let node = Rc::new(NodeInner::new_leaf(
        NodeId(1),
        Some("weight".to_string()),
        make_param(&[2, 3]),
    ));

    inner
        .borrow_mut()
        .register_parameter("weight".to_string(), Rc::downgrade(&node))
        .unwrap();

    // 获取参数
    let retrieved = inner.borrow().get_parameter("weight");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id(), NodeId(1));

    // 获取不存在的参数
    let not_found = inner.borrow().get_parameter("nonexistent");
    assert!(not_found.is_none());
}

#[test]
fn test_parameter_weak_reference_behavior() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 在作用域内创建参数
    {
        let node = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3])));

        inner
            .borrow_mut()
            .register_parameter("temp_param".to_string(), Rc::downgrade(&node))
            .unwrap();

        // 在作用域内，参数有效
        assert!(inner.borrow().has_parameter("temp_param"));
        assert_eq!(inner.borrow().valid_parameters_count(), 1);
    }
    // node 离开作用域，被释放

    // 参数注册表中仍有记录，但已失效
    assert_eq!(inner.borrow().registered_parameters_count(), 1);
    assert_eq!(inner.borrow().valid_parameters_count(), 0);
    assert!(!inner.borrow().has_parameter("temp_param"));
    assert!(inner.borrow().get_parameter("temp_param").is_none());
}

#[test]
fn test_cleanup_dead_parameters() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 创建一个会存活的参数
    let alive_node = Rc::new(NodeInner::new_leaf(
        NodeId(1),
        Some("alive".to_string()),
        make_param(&[2, 3]),
    ));

    inner
        .borrow_mut()
        .register_parameter("alive".to_string(), Rc::downgrade(&alive_node))
        .unwrap();

    // 创建一个会被释放的参数
    {
        let dead_node = Rc::new(NodeInner::new_leaf(
            NodeId(2),
            Some("dead".to_string()),
            make_param(&[2, 3]),
        ));
        inner
            .borrow_mut()
            .register_parameter("dead".to_string(), Rc::downgrade(&dead_node))
            .unwrap();
    }

    // 清理前
    assert_eq!(inner.borrow().registered_parameters_count(), 2);
    assert_eq!(inner.borrow().valid_parameters_count(), 1);

    // 清理
    let cleaned = inner.borrow_mut().cleanup_dead_parameters();
    assert_eq!(cleaned, 1);

    // 清理后
    assert_eq!(inner.borrow().registered_parameters_count(), 1);
    assert_eq!(inner.borrow().valid_parameters_count(), 1);
    assert!(inner.borrow().has_parameter("alive"));
    assert!(!inner.borrow().has_parameter("dead"));
}

#[test]
fn test_duplicate_parameter_name_error() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let node1 = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3])));

    let node2 = Rc::new(NodeInner::new_leaf(NodeId(2), None, make_param(&[2, 3])));

    // 第一次注册成功
    inner
        .borrow_mut()
        .register_parameter("param".to_string(), Rc::downgrade(&node1))
        .unwrap();

    // 重复注册同名参数应该失败
    let result = inner
        .borrow_mut()
        .register_parameter("param".to_string(), Rc::downgrade(&node2));
    assert!(result.is_err());
}

#[test]
fn test_replace_dead_parameter() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 注册一个会被释放的参数
    {
        let node1 = Rc::new(NodeInner::new_leaf(NodeId(1), None, make_param(&[2, 3])));
        inner
            .borrow_mut()
            .register_parameter("replaceable".to_string(), Rc::downgrade(&node1))
            .unwrap();
    }
    // node1 已释放

    // 注册同名的新参数应该成功（因为旧的已失效）
    let node2 = Rc::new(NodeInner::new_leaf(NodeId(2), None, make_param(&[2, 3])));
    inner
        .borrow_mut()
        .register_parameter("replaceable".to_string(), Rc::downgrade(&node2))
        .unwrap();

    // 验证新参数已注册
    assert!(inner.borrow().has_parameter("replaceable"));
    let retrieved = inner.borrow().get_parameter("replaceable").unwrap();
    assert_eq!(retrieved.id(), NodeId(2));
}

#[test]
fn test_get_all_parameters() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let node1 = Rc::new(NodeInner::new_leaf(
        NodeId(1),
        Some("w1".to_string()),
        make_param(&[2, 3]),
    ));

    let node2 = Rc::new(NodeInner::new_leaf(
        NodeId(2),
        Some("w2".to_string()),
        make_param(&[3, 4]),
    ));

    inner
        .borrow_mut()
        .register_parameter("layer1.weight".to_string(), Rc::downgrade(&node1))
        .unwrap();
    inner
        .borrow_mut()
        .register_parameter("layer2.weight".to_string(), Rc::downgrade(&node2))
        .unwrap();

    let all_params = inner.borrow().get_all_parameters();
    assert_eq!(all_params.len(), 2);

    // 验证名称正确
    let names: Vec<_> = all_params.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"layer1.weight"));
    assert!(names.contains(&"layer2.weight"));
}
