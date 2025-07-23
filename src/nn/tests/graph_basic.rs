use crate::nn::{Graph, GraphError, NodeId};
use crate::tensor::Tensor;

#[test]
fn test_graph_creation() {
    // 测试默认创建
    let graph = Graph::new();
    assert_eq!(graph.name(), "default_graph");
    assert_eq!(graph.nodes_count(), 0);

    // 测试指定名称创建
    let named_graph = Graph::with_name("custom_graph");
    assert_eq!(named_graph.name(), "custom_graph");
    assert_eq!(named_graph.nodes_count(), 0);
}

#[test]
fn test_graph_mode() {
    // 1. 默认创建模式
    // 默认应该是训练模式
    let mut graph = Graph::new();
    assert!(graph.is_train_mode());

    // 测试切换到评估模式
    graph.set_eval_mode();
    assert!(!graph.is_train_mode());

    // 测试切换回训练模式
    graph.set_train_mode();
    assert!(graph.is_train_mode());

    // 2. 测试指定名称创建
    // 默认应该是训练模式
    let mut named_graph = Graph::with_name("custom_graph");
    assert!(named_graph.is_train_mode());

    // 测试切换到评估模式
    named_graph.set_eval_mode();
    assert!(!named_graph.is_train_mode());

    // 测试切换回训练模式
    named_graph.set_train_mode();
    assert!(named_graph.is_train_mode());
}

#[test]
fn test_new_node_error_handling() {
    let mut graph = Graph::new();

    // 1. 测试节点未找到错误
    let invalid_id = NodeId(999);
    assert_eq!(
        graph.get_node_value(invalid_id),
        Err(GraphError::NodeNotFound(invalid_id))
    );

    // 2. 测试重复节点名称错误
    let _ = graph
        .new_parameter_node(&[2, 2], Some("duplicate"))
        .unwrap();
    assert_eq!(
        graph.new_parameter_node(&[2, 2], Some("duplicate")),
        Err(GraphError::DuplicateNodeName(format!(
            "节点duplicate在图default_graph中重复"
        )))
    );

    // 3. 测试形状不匹配导致的错误
    let param = graph.new_parameter_node(&[2, 2], None).unwrap();
    let wrong_shape = Tensor::new(&[1.0, 2.0], &[2, 1]);
    assert_eq!(
        graph.set_node_value(param, Some(&wrong_shape)),
        Err(GraphError::ShapeMismatch {
            expected: vec![2, 2],
            got: vec![2, 1],
            message: format!(
                "新张量的形状 [2, 1] 与节点 'parameter_1' 现有张量的形状 [2, 2] 不匹配。"
            )
        })
    );
}

#[test]
fn test_node_relationships() {
    let mut graph = Graph::new();

    // 1. 创建节点关系
    let input1 = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let input2 = graph.new_input_node(&[2, 2], Some("input2")).unwrap();
    let add = graph.new_add_node(&[input1, input2], Some("add")).unwrap();

    // 2. 验证父子关系
    let parents = graph.get_node_parents(add).unwrap();
    assert_eq!(parents.len(), 2);
    assert!(parents.contains(&input1));
    assert!(parents.contains(&input2));

    let children1 = graph.get_node_children(input1).unwrap();
    let children2 = graph.get_node_children(input2).unwrap();
    assert_eq!(children1.len(), 1);
    assert_eq!(children2.len(), 1);
    assert!(children1.contains(&add));
    assert!(children2.contains(&add));
}
