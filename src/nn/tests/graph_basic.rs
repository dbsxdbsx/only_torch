use crate::assert_err;
use crate::nn::{GraphInner, GraphError, NodeId};
use crate::tensor::Tensor;

#[test]
fn test_graph_creation() {
    // 测试默认创建
    let graph = GraphInner::new();
    assert_eq!(graph.name(), "default_graph");
    assert_eq!(graph.nodes_count(), 0);

    // 测试指定名称创建
    let named_graph = GraphInner::with_name("custom_graph");
    assert_eq!(named_graph.name(), "custom_graph");
    assert_eq!(named_graph.nodes_count(), 0);
}

#[test]
fn test_graph_mode() {
    // 1. 默认创建模式
    // 默认应该是训练模式
    let mut graph = GraphInner::new();
    assert!(graph.is_train_mode());

    // 测试切换到评估模式
    graph.set_eval_mode();
    assert!(!graph.is_train_mode());

    // 测试切换回训练模式
    graph.set_train_mode();
    assert!(graph.is_train_mode());

    // 2. 测试指定名称创建
    // 默认应该是训练模式
    let mut named_graph = GraphInner::with_name("custom_graph");
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
    let mut graph = GraphInner::new();

    // 1. 测试节点未找到错误
    let invalid_id = NodeId(999);
    assert_err!(
        graph.get_node_value(invalid_id),
        GraphError::NodeNotFound(id) if *id == invalid_id
    );

    // 2. 测试重复节点名称错误
    let _ = graph
        .new_parameter_node(&[2, 2], Some("duplicate"))
        .unwrap();
    assert_err!(
        graph.new_parameter_node(&[2, 2], Some("duplicate")),
        GraphError::DuplicateNodeName("节点duplicate在图default_graph中重复")
    );

    // 3. 测试形状不匹配导致的错误
    let param = graph.new_parameter_node(&[2, 2], None).unwrap();
    let wrong_shape = Tensor::new(&[1.0, 2.0], &[2, 1]);
    assert_err!(
        graph.set_node_value(param, Some(&wrong_shape)),
        GraphError::ShapeMismatch(
            [2, 2],
            [2, 1],
            "新张量的形状 [2, 1] 与节点 'parameter_1' 现有张量的形状 [2, 2] 不匹配。"
        )
    );
}

#[test]
fn test_node_relationships() {
    let mut graph = GraphInner::new();

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

// ============================================================================
// M4b: Graph 级别种子测试
// ============================================================================

/// 测试: GraphInner::new_with_seed 创建确定性图
#[test]
fn test_graph_new_with_seed() {
    // 1. 创建两个相同种子的图
    let mut graph1 = GraphInner::new_with_seed(42);
    let mut graph2 = GraphInner::new_with_seed(42);

    // 2. 在两个图中创建相同结构的参数节点
    let w1_g1 = graph1.new_parameter_node(&[3, 2], Some("w1")).unwrap();
    let b1_g1 = graph1.new_parameter_node(&[3, 1], Some("b1")).unwrap();

    let w1_g2 = graph2.new_parameter_node(&[3, 2], Some("w1")).unwrap();
    let b1_g2 = graph2.new_parameter_node(&[3, 1], Some("b1")).unwrap();

    // 3. 验证相同种子产生相同的参数值
    let w1_value_g1 = graph1.get_node_value(w1_g1).unwrap().unwrap();
    let w1_value_g2 = graph2.get_node_value(w1_g2).unwrap().unwrap();
    assert_eq!(w1_value_g1, w1_value_g2);

    let b1_value_g1 = graph1.get_node_value(b1_g1).unwrap().unwrap();
    let b1_value_g2 = graph2.get_node_value(b1_g2).unwrap().unwrap();
    assert_eq!(b1_value_g1, b1_value_g2);
}

/// 测试: 不同种子产生不同的参数值
#[test]
fn test_graph_different_seeds_produce_different_values() {
    let mut graph1 = GraphInner::new_with_seed(42);
    let mut graph2 = GraphInner::new_with_seed(123);

    let w1 = graph1.new_parameter_node(&[3, 2], Some("w1")).unwrap();
    let w2 = graph2.new_parameter_node(&[3, 2], Some("w1")).unwrap();

    let w1_value = graph1.get_node_value(w1).unwrap().unwrap();
    let w2_value = graph2.get_node_value(w2).unwrap().unwrap();

    // 不同种子应产生不同的值
    assert_ne!(w1_value, w2_value);
}

/// 测试: GraphInner::set_seed 动态设置种子
#[test]
fn test_graph_set_seed() {
    let mut graph1 = GraphInner::new();
    let mut graph2 = GraphInner::new();

    // 动态设置种子
    graph1.set_seed(42);
    graph2.set_seed(42);

    // 验证 has_seed
    assert!(graph1.has_seed());
    assert!(graph2.has_seed());

    // 创建参数节点
    let w1 = graph1.new_parameter_node(&[2, 2], Some("w")).unwrap();
    let w2 = graph2.new_parameter_node(&[2, 2], Some("w")).unwrap();

    // 相同种子应产生相同值
    let w1_value = graph1.get_node_value(w1).unwrap().unwrap();
    let w2_value = graph2.get_node_value(w2).unwrap().unwrap();
    assert_eq!(w1_value, w2_value);
}

/// 测试: 无种子的图是非确定性的（概率性测试）
#[test]
fn test_graph_without_seed_is_non_deterministic() {
    let mut graph1 = GraphInner::new();
    let mut graph2 = GraphInner::new();

    assert!(!graph1.has_seed());
    assert!(!graph2.has_seed());

    // 创建参数节点（使用较大的形状以确保随机性可观察）
    let w1 = graph1.new_parameter_node(&[10, 10], Some("w")).unwrap();
    let w2 = graph2.new_parameter_node(&[10, 10], Some("w")).unwrap();

    // 无种子时，两个图的参数值应该不同（概率上几乎不可能相同）
    let w1_value = graph1.get_node_value(w1).unwrap().unwrap();
    let w2_value = graph2.get_node_value(w2).unwrap().unwrap();
    assert_ne!(w1_value, w2_value);
}

/// 测试: new_parameter_node_seeded 覆盖 Graph 种子
#[test]
fn test_seeded_parameter_overrides_graph_seed() {
    let mut graph1 = GraphInner::new_with_seed(42);
    let mut graph2 = GraphInner::new_with_seed(999); // 不同的 Graph 种子

    // 使用显式种子创建参数（应该覆盖 Graph 种子）
    let w1 = graph1
        .new_parameter_node_seeded(&[3, 2], Some("w"), 123)
        .unwrap();
    let w2 = graph2
        .new_parameter_node_seeded(&[3, 2], Some("w"), 123)
        .unwrap();

    // 显式种子相同，所以参数值应该相同
    let w1_value = graph1.get_node_value(w1).unwrap().unwrap();
    let w2_value = graph2.get_node_value(w2).unwrap().unwrap();
    assert_eq!(w1_value, w2_value);
}

/// 测试: NEAT 兼容性 - 多个 Graph 并行独立运行
#[test]
fn test_neat_compatibility_multiple_graphs() {
    // 创建多个带种子的图（模拟 NEAT 种群）
    let mut graphs: Vec<GraphInner> = (0..5).map(|i| GraphInner::new_with_seed(i as u64)).collect();

    // 每个图独立创建参数
    let params: Vec<NodeId> = graphs
        .iter_mut()
        .map(|g| g.new_parameter_node(&[4, 3], Some("w")).unwrap())
        .collect();

    // 验证：相同种子的图（如果重新创建）产生相同结果
    let mut graph_0_copy = GraphInner::new_with_seed(0);
    let param_0_copy = graph_0_copy.new_parameter_node(&[4, 3], Some("w")).unwrap();

    let original_value = graphs[0].get_node_value(params[0]).unwrap().unwrap();
    let copy_value = graph_0_copy.get_node_value(param_0_copy).unwrap().unwrap();
    assert_eq!(original_value, copy_value);

    // 验证：不同种子的图产生不同结果
    let value_0 = graphs[0].get_node_value(params[0]).unwrap().unwrap();
    let value_1 = graphs[1].get_node_value(params[1]).unwrap().unwrap();
    assert_ne!(value_0, value_1);
}

/// 测试: GraphInner::with_name_and_seed
#[test]
fn test_graph_with_name_and_seed() {
    let mut graph = GraphInner::with_name_and_seed("my_graph", 42);

    assert_eq!(graph.name(), "my_graph");
    assert!(graph.has_seed());

    // 创建参数节点
    let w = graph.new_parameter_node(&[2, 2], None).unwrap();

    // 验证可以正常使用
    assert!(graph.get_node_value(w).unwrap().is_some());
}

/// 测试: 种子设置后的多次参数创建保持确定性
#[test]
fn test_sequential_parameter_creation_determinism() {
    let mut graph1 = GraphInner::new_with_seed(42);
    let mut graph2 = GraphInner::new_with_seed(42);

    // 按顺序创建多个参数
    let w1_g1 = graph1.new_parameter_node(&[3, 2], Some("w1")).unwrap();
    let w2_g1 = graph1.new_parameter_node(&[2, 3], Some("w2")).unwrap();
    let w3_g1 = graph1.new_parameter_node(&[4, 4], Some("w3")).unwrap();

    let w1_g2 = graph2.new_parameter_node(&[3, 2], Some("w1")).unwrap();
    let w2_g2 = graph2.new_parameter_node(&[2, 3], Some("w2")).unwrap();
    let w3_g2 = graph2.new_parameter_node(&[4, 4], Some("w3")).unwrap();

    // 所有对应的参数都应该相同
    assert_eq!(
        graph1.get_node_value(w1_g1).unwrap(),
        graph2.get_node_value(w1_g2).unwrap()
    );
    assert_eq!(
        graph1.get_node_value(w2_g1).unwrap(),
        graph2.get_node_value(w2_g2).unwrap()
    );
    assert_eq!(
        graph1.get_node_value(w3_g1).unwrap(),
        graph2.get_node_value(w3_g2).unwrap()
    );

    // 同一个图中的不同参数应该不同
    assert_ne!(
        graph1.get_node_value(w1_g1).unwrap(),
        graph1.get_node_value(w2_g1).unwrap()
    );
}
