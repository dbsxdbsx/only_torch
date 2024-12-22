use crate::assert_panic;
use crate::nn::Graph;
use crate::tensor::Tensor;

#[test]
fn test_variable_creation() {
    let mut graph = Graph::new();

    // 测试基本创建
    let var_id = graph
        .new_variable_node(&[2, 3], false, false, Some("test_var"))
        .unwrap();
    let node = graph.get_node(var_id).unwrap();
    assert_eq!(node.name(), "default_graph_test_var");
    assert!(!node.is_inited());
    assert!(node.value().is_none());
    assert!(!node.is_trainable());
    assert!(node.parents_ids().is_empty());

    // 测试带初始化的创建
    let var_init = graph
        .new_variable_node(&[2, 3], true, true, Some("init_var"))
        .unwrap();
    let node_init = graph.get_node(var_init).unwrap();
    assert!(node_init.is_inited());
    assert!(node_init.is_trainable());
    assert!(node_init.parents_ids().is_empty());
    assert_eq!(node_init.value().unwrap().shape(), &[2, 3]);

    // 测试创建时使用错误维度的张量
    assert!(graph
        .new_variable_node(&[2], true, true, Some("wrong_dimension_var"))
        .is_err());
    assert!(graph
        .new_variable_node(&[2, 2, 2], true, true, Some("wrong_dimension_var"))
        .is_err());
}

#[test]
fn test_variable_trainable() {
    let mut graph = Graph::new();

    // 1. 测试初始非可训练节点
    let var_id = graph
        .new_variable_node(&[2, 3], false, false, Some("test_var"))
        .unwrap();
    // 1.1 然后测试设置trainable属性
    let node_mut = graph.get_node_mut(var_id).unwrap();
    node_mut.set_trainable(true).unwrap();
    assert!(node_mut.is_trainable());
    node_mut.set_trainable(false).unwrap();
    assert!(!node_mut.is_trainable());

    // 2. 测试初始可训练节点
    let var_id2 = graph
        .new_variable_node(&[2, 3], false, true, Some("test_var2"))
        .unwrap();

    // 2.1 然后测试设置trainable属性
    let node2 = graph.get_node(var_id2).unwrap();
    assert!(node2.is_trainable());
    let node2_mut = graph.get_node_mut(var_id2).unwrap();
    node2_mut.set_trainable(false).unwrap();
    assert!(!node2_mut.is_trainable());
    node2_mut.set_trainable(true).unwrap();
    assert!(node2_mut.is_trainable());
}

#[test]
fn test_variable_value_operations() {
    let mut graph = Graph::new();
    let var_id = graph
        .new_variable_node(&[2, 2], false, true, Some("test_var"))
        .unwrap();

    // 赋值前
    let node = graph.get_node(var_id).unwrap();
    assert!(node.value().is_none());
    assert!(node.jacobi().is_none());
    assert!(!node.is_inited());

    // 赋值后
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    graph.set_node_value(var_id, Some(&test_value)).unwrap();
    assert!(graph.get_node(var_id).unwrap().is_inited());
    assert_eq!(graph.get_node(var_id).unwrap().value().unwrap(), test_value);

    // 测试形状不匹配
    let wrong_shape = Tensor::new(&[1.0, 2.0], &[2, 1, 1]);
    assert_panic!(graph.set_node_value(var_id, Some(&wrong_shape)));
}

#[test]
fn test_variable_jacobi_operations() {
    let mut graph = Graph::new();
    let var = graph
        .new_variable_node(&[2, 2], true, true, Some("test_var"))
        .unwrap();

    // 初始时雅可比矩阵应为空
    assert!(graph.get_node(var).unwrap().jacobi().is_none());

    // 测试清除雅可比矩阵
    graph.clear_jacobi().unwrap();
    assert!(graph.get_node(var).unwrap().jacobi().is_none());
}

#[test]
fn test_variable_name_generation() {
    let mut graph = Graph::new();

    // 测试显式命名
    let var1 = graph
        .new_variable_node(&[2, 2], false, false, Some("explicit"))
        .unwrap();
    assert_eq!(
        graph.get_node(var1).unwrap().name(),
        "default_graph_explicit"
    );

    // 测试自动命名
    let var2 = graph
        .new_variable_node(&[2, 2], false, false, None)
        .unwrap();
    assert_eq!(graph.get_node(var2).unwrap().name(), "default_graph_var_1");

    // 测试重复名称处理
    let var3 = graph
        .new_variable_node(&[2, 2], false, false, Some("explicit"))
        .unwrap();
    assert_eq!(graph.get_node(var3).unwrap().name(), "default_graph_var_2");
}

#[test]
fn test_variable_invalid_shapes() {
    let mut graph = Graph::new();

    // 测试1维形状(应该失败)
    assert!(graph.new_variable_node(&[3], false, false, None).is_err());

    // 测试3维形状(应该失败)
    assert!(graph
        .new_variable_node(&[2, 2, 2], false, false, None)
        .is_err());

    // 测试空形状(应该失败)
    assert!(graph.new_variable_node(&[], false, false, None).is_err());
}

#[test]
fn test_variable_trainable_flag() {
    let mut graph = Graph::new();

    // 测试可训练变量
    let var1 = graph.new_variable_node(&[2, 2], false, true, None).unwrap();
    assert!(graph.get_node(var1).unwrap().is_trainable());

    // 测试不可训练变量
    let var2 = graph
        .new_variable_node(&[2, 2], false, false, None)
        .unwrap();
    assert!(!graph.get_node(var2).unwrap().is_trainable());
}

#[test]
fn test_variable_initialization() {
    let mut graph = Graph::new();

    // 测试未初始化变量
    let var1 = graph.new_variable_node(&[2, 2], false, true, None).unwrap();
    assert!(graph.get_node(var1).unwrap().value().is_none());

    // 测试已初始化变量
    let var2 = graph.new_variable_node(&[2, 2], true, true, None).unwrap();
    let value = graph.get_node(var2).unwrap().value().unwrap();
    assert_eq!(value.shape(), &[2, 2]);

    // 验证初始化是否接近N(0, 0.001)
    let mean = value.mean();
    // let std = value.std();
    assert!(mean.abs() < 0.1); // 均值应接近0
                               // assert!((std - 0.001).abs() < 0.001); // 标准差应接近0.001
}
