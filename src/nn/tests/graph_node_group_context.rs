/*
 * @Author       : 老董
 * @Date         : 2026-02-13
 * @Description  : NodeGroupContext 图基础设施测试
 *
 * 测试通用节点分组上下文机制（当前用于分布 cluster 可视化），
 * 不依赖具体分布实现。
 */

use crate::nn::graph::{Graph, NodeGroupContext};
use crate::nn::VarActivationOps;
use crate::tensor::Tensor;

// ==================== 上下文 push/pop ====================

/// Guard 存活期间创建的节点有标签，drop 后创建的节点无标签
#[test]
fn test_context_push_pop() {
    let graph = Graph::new();
    let input = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();

    let instance_id = graph.inner_mut().next_node_group_instance_id();

    let tagged = {
        let _guard = NodeGroupContext::new(&input, "TestGroup", instance_id);
        input.softmax() // 在上下文中创建
    };

    let untagged = input.log_softmax(); // 上下文已 drop

    assert!(tagged.node_group_tag().is_some());
    assert_eq!(tagged.node_group_tag().unwrap().group_type, "TestGroup");
    assert_eq!(tagged.node_group_tag().unwrap().instance_id, 0);
    assert!(untagged.node_group_tag().is_none());
}

// ==================== 自动标记 ====================

/// 上下文激活期间创建的计算节点自动带标签
#[test]
fn test_auto_tagging_compute_nodes() {
    let graph = Graph::new();
    let input = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();

    let instance_id = graph.inner_mut().next_node_group_instance_id();

    let (softmax, log_softmax) = {
        let _guard = NodeGroupContext::new(&input, "MyDist", instance_id);
        (input.softmax(), input.log_softmax())
    };

    // 两个计算节点都应带标签
    assert!(softmax.node_group_tag().is_some());
    assert!(log_softmax.node_group_tag().is_some());
    assert_eq!(softmax.node_group_tag().unwrap().group_type, "MyDist");
    assert_eq!(log_softmax.node_group_tag().unwrap().group_type, "MyDist");
}

// ==================== 上下文隔离 ====================

/// Guard drop 后创建的节点无标签
#[test]
fn test_context_isolation() {
    let graph = Graph::new();
    let input = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();

    let instance_id = graph.inner_mut().next_node_group_instance_id();

    let inside = {
        let _guard = NodeGroupContext::new(&input, "Group", instance_id);
        input.softmax()
    };

    // Guard 已 drop，后续操作创建的节点无标签
    let outside = &inside * &inside;
    assert!(inside.node_group_tag().is_some());
    assert!(outside.node_group_tag().is_none());
}

// ==================== Input 排除 ====================

/// 上下文激活期间，Var * Tensor 创建的 Input 节点不被标记，但 Multiply 节点有标签
#[test]
fn test_input_node_exclusion() {
    let graph = Graph::new();
    let input = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();

    let instance_id = graph.inner_mut().next_node_group_instance_id();

    let multiply = {
        let _guard = NodeGroupContext::new(&input, "Group", instance_id);
        let scalar = Tensor::new(&[3.0, 4.0], &[1, 2]);
        &input * scalar // 内部：创建 Input 节点 + Multiply 节点
    };

    // Multiply 节点有标签
    assert!(multiply.node_group_tag().is_some());
    // 原始 input（在上下文之前创建）无标签
    assert!(input.node_group_tag().is_none());
}

// ==================== 外层优先 ====================

/// 嵌套 Guard 时，内层 push 被跳过，节点标记为外层的 group_type
#[test]
fn test_outer_first_nesting() {
    let graph = Graph::new();
    let input = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();

    let outer_id = graph.inner_mut().next_node_group_instance_id();
    let _inner_id = graph.inner_mut().next_node_group_instance_id();

    let node = {
        let _outer = NodeGroupContext::new(&input, "Outer", outer_id);
        {
            let _inner = NodeGroupContext::new(&input, "Inner", _inner_id);
            input.softmax() // 在嵌套上下文中创建
        }
    };

    // 节点应标记为外层 "Outer"
    let tag = node.node_group_tag().unwrap();
    assert_eq!(tag.group_type, "Outer");
    assert_eq!(tag.instance_id, outer_id);
}

/// 外层 Guard drop 后，内层的后续操作依然无法覆盖
#[test]
fn test_outer_drop_then_inner_clear() {
    let graph = Graph::new();
    let input = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();

    let outer_id = graph.inner_mut().next_node_group_instance_id();
    let inner_id = graph.inner_mut().next_node_group_instance_id();

    let (node_in_nested, node_after_outer_drop) = {
        let _outer = NodeGroupContext::new(&input, "Outer", outer_id);
        let _inner = NodeGroupContext::new(&input, "Inner", inner_id);
        let n1 = input.softmax(); // Outer 标签
        drop(_outer); // 外层 drop
        drop(_inner); // 内层 drop（did_push=false，不清除）
        // 注意：外层 drop 会清除 context
        let n2 = input.log_softmax(); // context 已被外层 drop 清除
        (n1, n2)
    };

    assert_eq!(
        node_in_nested.node_group_tag().unwrap().group_type,
        "Outer"
    );
    assert!(node_after_outer_drop.node_group_tag().is_none());
}

// ==================== 多实例 ID ====================

/// 多次 next_node_group_instance_id() 返回递增值
#[test]
fn test_instance_id_increment() {
    let graph = Graph::new();
    let id0 = graph.inner_mut().next_node_group_instance_id();
    let id1 = graph.inner_mut().next_node_group_instance_id();
    let id2 = graph.inner_mut().next_node_group_instance_id();
    assert_eq!(id0, 0);
    assert_eq!(id1, 1);
    assert_eq!(id2, 2);
}

/// 两个不同实例的同类型 Guard 产生不同 instance_id
#[test]
fn test_multiple_instances_different_tags() {
    let graph = Graph::new();
    let input1 = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let input2 = graph.input(&Tensor::new(&[3.0, 4.0], &[1, 2])).unwrap();

    let id1 = graph.inner_mut().next_node_group_instance_id();
    let node1 = {
        let _guard = NodeGroupContext::new(&input1, "Cat", id1);
        input1.softmax()
    };

    let id2 = graph.inner_mut().next_node_group_instance_id();
    let node2 = {
        let _guard = NodeGroupContext::new(&input2, "Cat", id2);
        input2.softmax()
    };

    let tag1 = node1.node_group_tag().unwrap();
    let tag2 = node2.node_group_tag().unwrap();
    assert_eq!(tag1.group_type, "Cat");
    assert_eq!(tag2.group_type, "Cat");
    assert_ne!(tag1.instance_id, tag2.instance_id);
}
