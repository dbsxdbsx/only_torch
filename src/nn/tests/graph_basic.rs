use crate::assert_err;
use crate::nn::{Graph, GraphError, GraphInner};
use crate::tensor::Tensor;
use std::rc::Rc;

/// 测试: 默认创建和命名创建
#[test]
fn test_graph_creation() {
    // 测试默认创建
    let graph = Graph::new();
    assert_eq!(graph.inner().name(), "default_graph");

    // 测试指定名称创建（通过 GraphInner::with_name + from_inner）
    let named_graph = Graph::from_inner(GraphInner::with_name("custom_graph"));
    assert_eq!(named_graph.inner().name(), "custom_graph");
}

/// 测试: 训练/评估模式切换
#[test]
fn test_graph_mode() {
    // 1. 默认创建 — 应该是训练模式
    let graph = Graph::new();
    assert!(graph.inner().is_train_mode());

    // 切换到评估模式
    graph.eval();
    assert!(!graph.inner().is_train_mode());

    // 切换回训练模式
    graph.train();
    assert!(graph.inner().is_train_mode());

    // 2. 指定名称创建 — 同样默认训练模式
    let named_graph = Graph::from_inner(GraphInner::with_name("custom_graph"));
    assert!(named_graph.inner().is_train_mode());

    // 切换到评估模式
    named_graph.eval();
    assert!(!named_graph.inner().is_train_mode());

    // 切换回训练模式
    named_graph.train();
    assert!(named_graph.inner().is_train_mode());
}

/// 测试: 错误处理（新架构适配）
///
/// 新架构下节点由 Rc<NodeInner> 直接持有，不再通过 NodeId 查找，
/// 因此 NodeNotFound 不再适用。此处测试新架构下可能的错误：
/// 1. 重复参数注册
/// 2. 形状不兼容的操作节点创建
#[test]
fn test_new_node_error_handling() {
    let graph = Graph::new();

    // 1. 测试重复参数注册错误
    let param1 = graph
        .inner_mut()
        .create_parameter_node(&[2, 2], Some("duplicate"))
        .unwrap();
    graph
        .inner_mut()
        .register_parameter("duplicate".to_string(), Rc::downgrade(&param1))
        .unwrap();

    let param2 = graph
        .inner_mut()
        .create_parameter_node(&[2, 2], Some("duplicate_2"))
        .unwrap();
    // 同名参数注册应失败
    assert_err!(
        graph
            .inner_mut()
            .register_parameter("duplicate".to_string(), Rc::downgrade(&param2)),
        GraphError::InvalidOperation { .. }
    );

    // 2. 测试形状不兼容的 MatMul 节点创建
    let input_a = graph
        .inner_mut()
        .create_basic_input_node(&[2, 3], Some("a"))
        .unwrap();
    let input_b = graph
        .inner_mut()
        .create_basic_input_node(&[5, 4], Some("b"))
        .unwrap();
    // [2,3] @ [5,4] 形状不兼容
    let result = graph
        .inner_mut()
        .create_mat_mul_node(vec![input_a, input_b], Some("bad_matmul"));
    assert_err!(result);
}

/// 测试: 节点父子关系
///
/// 新架构下只追踪父节点（children 由 Rc 引用关系隐式表达）。
#[test]
fn test_node_relationships() {
    let graph = Graph::new();

    // 1. 创建节点关系
    let input1 = graph
        .inner_mut()
        .create_basic_input_node(&[2, 2], Some("input1"))
        .unwrap();
    input1
        .set_value(Some(&Tensor::zeros(&[2, 2])))
        .unwrap();

    let input2 = graph
        .inner_mut()
        .create_basic_input_node(&[2, 2], Some("input2"))
        .unwrap();
    input2
        .set_value(Some(&Tensor::zeros(&[2, 2])))
        .unwrap();

    let add = graph
        .inner_mut()
        .create_add_node(
            vec![Rc::clone(&input1), Rc::clone(&input2)],
            Some("add"),
        )
        .unwrap();

    // 2. 验证父节点关系
    let parents = add.parents();
    assert_eq!(parents.len(), 2);
    assert!(Rc::ptr_eq(&parents[0], &input1));
    assert!(Rc::ptr_eq(&parents[1], &input2));

    // 3. 验证叶子节点属性
    assert!(input1.is_leaf());
    assert!(input2.is_leaf());
    assert!(!add.is_leaf());
}

// ============================================================================
// M4b: Graph 级别种子测试
// ============================================================================

/// 测试: Graph::new_with_seed 创建确定性图
#[test]
fn test_graph_new_with_seed() {
    // 1. 创建两个相同种子的图
    let graph1 = Graph::new_with_seed(42);
    let graph2 = Graph::new_with_seed(42);

    // 2. 在两个图中创建相同结构的参数节点
    let w1_g1 = graph1
        .inner_mut()
        .create_parameter_node(&[3, 2], Some("w1"))
        .unwrap();
    let b1_g1 = graph1
        .inner_mut()
        .create_parameter_node(&[3, 1], Some("b1"))
        .unwrap();

    let w1_g2 = graph2
        .inner_mut()
        .create_parameter_node(&[3, 2], Some("w1"))
        .unwrap();
    let b1_g2 = graph2
        .inner_mut()
        .create_parameter_node(&[3, 1], Some("b1"))
        .unwrap();

    // 3. 验证相同种子产生相同的参数值
    let w1_value_g1 = w1_g1.value().unwrap();
    let w1_value_g2 = w1_g2.value().unwrap();
    assert_eq!(w1_value_g1, w1_value_g2);

    let b1_value_g1 = b1_g1.value().unwrap();
    let b1_value_g2 = b1_g2.value().unwrap();
    assert_eq!(b1_value_g1, b1_value_g2);
}

/// 测试: 不同种子产生不同的参数值
#[test]
fn test_graph_different_seeds_produce_different_values() {
    let graph1 = Graph::new_with_seed(42);
    let graph2 = Graph::new_with_seed(123);

    let w1 = graph1
        .inner_mut()
        .create_parameter_node(&[3, 2], Some("w1"))
        .unwrap();
    let w2 = graph2
        .inner_mut()
        .create_parameter_node(&[3, 2], Some("w1"))
        .unwrap();

    let w1_value = w1.value().unwrap();
    let w2_value = w2.value().unwrap();

    // 不同种子应产生不同的值
    assert_ne!(w1_value, w2_value);
}

/// 测试: GraphInner::set_seed 动态设置种子
#[test]
fn test_graph_set_seed() {
    let graph1 = Graph::new();
    let graph2 = Graph::new();

    // 动态设置种子
    graph1.inner_mut().set_seed(42);
    graph2.inner_mut().set_seed(42);

    // 验证 has_seed
    assert!(graph1.inner().has_seed());
    assert!(graph2.inner().has_seed());

    // 创建参数节点
    let w1 = graph1
        .inner_mut()
        .create_parameter_node(&[2, 2], Some("w"))
        .unwrap();
    let w2 = graph2
        .inner_mut()
        .create_parameter_node(&[2, 2], Some("w"))
        .unwrap();

    // 相同种子应产生相同值
    let w1_value = w1.value().unwrap();
    let w2_value = w2.value().unwrap();
    assert_eq!(w1_value, w2_value);
}

/// 测试: 无种子的图是非确定性的（概率性测试）
#[test]
fn test_graph_without_seed_is_non_deterministic() {
    let graph1 = Graph::new();
    let graph2 = Graph::new();

    assert!(!graph1.inner().has_seed());
    assert!(!graph2.inner().has_seed());

    // 创建参数节点（使用较大的形状以确保随机性可观察）
    let w1 = graph1
        .inner_mut()
        .create_parameter_node(&[10, 10], Some("w"))
        .unwrap();
    let w2 = graph2
        .inner_mut()
        .create_parameter_node(&[10, 10], Some("w"))
        .unwrap();

    // 无种子时，两个图的参数值应该不同（概率上几乎不可能相同）
    let w1_value = w1.value().unwrap();
    let w2_value = w2.value().unwrap();
    assert_ne!(w1_value, w2_value);
}

/// 测试: create_parameter_node_seeded 覆盖 Graph 种子
#[test]
fn test_seeded_parameter_overrides_graph_seed() {
    let graph1 = Graph::new_with_seed(42);
    let graph2 = Graph::new_with_seed(999); // 不同的 Graph 种子

    // 使用显式种子创建参数（应该覆盖 Graph 种子）
    let w1 = graph1
        .inner_mut()
        .create_parameter_node_seeded(&[3, 2], Some("w"), 123)
        .unwrap();
    let w2 = graph2
        .inner_mut()
        .create_parameter_node_seeded(&[3, 2], Some("w"), 123)
        .unwrap();

    // 显式种子相同，所以参数值应该相同
    let w1_value = w1.value().unwrap();
    let w2_value = w2.value().unwrap();
    assert_eq!(w1_value, w2_value);
}

/// 测试: NEAT 兼容性 - 多个 Graph 并行独立运行
#[test]
fn test_neat_compatibility_multiple_graphs() {
    // 创建多个带种子的图（模拟 NEAT 种群）
    let graphs: Vec<Graph> = (0..5)
        .map(|i| Graph::new_with_seed(i as u64))
        .collect();

    // 每个图独立创建参数
    let params: Vec<_> = graphs
        .iter()
        .map(|g| {
            g.inner_mut()
                .create_parameter_node(&[4, 3], Some("w"))
                .unwrap()
        })
        .collect();

    // 验证：相同种子的图（如果重新创建）产生相同结果
    let graph_0_copy = Graph::new_with_seed(0);
    let param_0_copy = graph_0_copy
        .inner_mut()
        .create_parameter_node(&[4, 3], Some("w"))
        .unwrap();

    let original_value = params[0].value().unwrap();
    let copy_value = param_0_copy.value().unwrap();
    assert_eq!(original_value, copy_value);

    // 验证：不同种子的图产生不同结果
    let value_0 = params[0].value().unwrap();
    let value_1 = params[1].value().unwrap();
    assert_ne!(value_0, value_1);
}

/// 测试: GraphInner::with_name_and_seed
#[test]
fn test_graph_with_name_and_seed() {
    let graph = Graph::from_inner(GraphInner::with_name_and_seed("my_graph", 42));

    assert_eq!(graph.inner().name(), "my_graph");
    assert!(graph.inner().has_seed());

    // 创建参数节点
    let w = graph
        .inner_mut()
        .create_parameter_node(&[2, 2], None)
        .unwrap();

    // 验证可以正常使用
    assert!(w.value().is_some());
}

/// 测试: 种子设置后的多次参数创建保持确定性
#[test]
fn test_sequential_parameter_creation_determinism() {
    let graph1 = Graph::new_with_seed(42);
    let graph2 = Graph::new_with_seed(42);

    // 按顺序创建多个参数
    let w1_g1 = graph1
        .inner_mut()
        .create_parameter_node(&[3, 2], Some("w1"))
        .unwrap();
    let w2_g1 = graph1
        .inner_mut()
        .create_parameter_node(&[2, 3], Some("w2"))
        .unwrap();
    let w3_g1 = graph1
        .inner_mut()
        .create_parameter_node(&[4, 4], Some("w3"))
        .unwrap();

    let w1_g2 = graph2
        .inner_mut()
        .create_parameter_node(&[3, 2], Some("w1"))
        .unwrap();
    let w2_g2 = graph2
        .inner_mut()
        .create_parameter_node(&[2, 3], Some("w2"))
        .unwrap();
    let w3_g2 = graph2
        .inner_mut()
        .create_parameter_node(&[4, 4], Some("w3"))
        .unwrap();

    // 所有对应的参数都应该相同
    assert_eq!(w1_g1.value(), w1_g2.value());
    assert_eq!(w2_g1.value(), w2_g2.value());
    assert_eq!(w3_g1.value(), w3_g2.value());

    // 同一个图中的不同参数应该不同
    assert_ne!(w1_g1.value(), w2_g1.value());
}

// 注意：全局节点复用机制已禁用（需要更细粒度的控制才能正确工作）
// RNN 层使用自己的内部缓存机制（unroll_cache）来复用展开结构
