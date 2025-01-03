use crate::nn::{Graph, GraphError, NodeId};
use crate::tensor::Tensor;

#[test]
fn test_graph_creation() {
    // 测试默认创建
    let graph = Graph::new();
    assert_eq!(graph.name(), "default_graph");
    assert_eq!(graph.nodes_count(), 0);
    assert!(graph.is_train_mode());

    // 测试指定名称创建
    let named_graph = Graph::with_name("custom_graph");
    assert_eq!(named_graph.name(), "custom_graph");
    assert_eq!(named_graph.nodes_count(), 0);
}

#[test]
fn test_graph_mode() {
    let mut graph = Graph::new();

    // 默认应该是训练模式
    assert!(graph.is_train_mode());

    // 测试切换到评估模式
    graph.set_eval_mode();
    assert!(!graph.is_train_mode());

    // 测试切换回训练模式
    graph.set_train_mode();
    assert!(graph.is_train_mode());
}

#[test]
fn test_node_management() {
    let mut graph = Graph::new();

    // 1. 创建节点
    let var = graph
        .new_variable_node(&[2, 2], true, true, Some("test_var"))
        .unwrap();

    // 2. 测试节点信息获取
    assert_eq!(graph.get_node_name(var).unwrap(), "test_var");
    assert!(graph.is_node_trainable(var).unwrap());
    assert!(graph.is_node_inited(var).unwrap());
    assert_eq!(graph.get_node_parents(var).unwrap().len(), 0);
    assert_eq!(graph.get_node_children(var).unwrap().len(), 0);

    // 3. 测试节点值操作
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    graph.set_node_value(var, Some(&test_value)).unwrap();
    assert_eq!(graph.get_node_value(var).unwrap().unwrap(), &test_value);

    // 4. 测试清除雅可比矩阵
    graph.clear_jacobi().unwrap();
    assert!(graph.get_node_jacobi(var).unwrap().is_none());
}

#[test]
fn test_new_node_error_handling() {
    let mut graph = Graph::new();

    // 1. 测试节点未找到错误
    let invalid_id = NodeId(999);
    assert!(matches!(
        graph.get_node_value(invalid_id),
        Err(GraphError::NodeNotFound(_))
    ));

    // 2. 测试重复节点名称错误
    let _ = graph
        .new_variable_node(&[2, 2], true, true, Some("duplicate"))
        .unwrap();
    assert!(matches!(
        graph.new_variable_node(&[2, 2], true, true, Some("duplicate")),
        Err(GraphError::DuplicateNodeName(_))
    ));

    // 3. 测试形状不匹配导致的错误
    let var = graph.new_variable_node(&[2, 2], true, true, None).unwrap();
    let wrong_shape = Tensor::new(&[1.0, 2.0], &[2, 1]);
    assert!(matches!(
        graph.set_node_value(var, Some(&wrong_shape)),
        Err(GraphError::ShapeMismatch { .. })
    ));
}

#[test]
fn test_node_relationships() {
    let mut graph = Graph::new();

    // 1. 创建节点关系
    let var1 = graph
        .new_variable_node(&[2, 2], true, true, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[2, 2], true, true, Some("var2"))
        .unwrap();
    let add = graph
        .new_add_node(&[var1, var2], Some("add"), true)
        .unwrap();

    // 2. 验证父子关系
    let parents = graph.get_node_parents(add).unwrap();
    assert_eq!(parents.len(), 2);
    assert!(parents.contains(&var1));
    assert!(parents.contains(&var2));

    let children1 = graph.get_node_children(var1).unwrap();
    let children2 = graph.get_node_children(var2).unwrap();
    assert_eq!(children1.len(), 1);
    assert_eq!(children2.len(), 1);
    assert!(children1.contains(&add));
    assert!(children2.contains(&add));
}

#[test]
fn test_forward_backward_propagation_of_add_node() {
    let mut graph = Graph::new();

    // 1. 创建输入节点
    let var1 = graph
        .new_variable_node(&[2, 2], true, true, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[2, 2], true, true, Some("var2"))
        .unwrap();

    // 2. 设置输入值 (与Python测试tests\calc_jacobi_by_pytorch\node_add.py保持一致)
    let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let value2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    graph.set_node_value(var1, Some(&value1)).unwrap();
    graph.set_node_value(var2, Some(&value2)).unwrap();

    // 3. 创建Add节点并前向传播
    // 检查前向传播前Add节点的值为空
    let add = graph
        .new_add_node(&[var1, var2], Some("add"), true)
        .unwrap();
    assert!(graph.get_node_value(add).unwrap().is_none());
    // 3.1 前向传播
    graph.forward_node(add).unwrap();
    // 3.2 验证前向传播结果
    let result = graph.get_node_value(add).unwrap().unwrap();
    let expected = &value1 + &value2;
    assert_eq!(result, &expected);

    // 4. 对两个输入节点进行反向传播
    // 4.1 检查反向传播前雅可比矩阵为空
    assert!(graph.get_node_jacobi(var1).unwrap().is_none());
    assert!(graph.get_node_jacobi(var2).unwrap().is_none());
    // 4.2 反向传播
    graph.backward_node(var1, add).unwrap();
    graph.backward_node(var2, add).unwrap();
    // 4.3 验证反向传播结果
    // 对于Add操作，雅可比矩阵应该是单位矩阵，因为 ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
    let jacobi_var1 = graph.get_node_jacobi(var1).unwrap().unwrap();
    let jacobi_var2 = graph.get_node_jacobi(var2).unwrap().unwrap();
    // 预期的雅可比矩阵是4x4的单位矩阵（因为输入是2x2展平后是4维向量）
    let expected_jacobi = Tensor::eyes(4);
    assert_eq!(jacobi_var1, &expected_jacobi);
    assert_eq!(jacobi_var2, &expected_jacobi);

    // 5. 验证清除雅可比矩阵
    graph.clear_jacobi().unwrap();
    assert!(graph.get_node_jacobi(var1).unwrap().is_none());
    assert!(graph.get_node_jacobi(var2).unwrap().is_none());
}

#[test]
fn test_variable_node_initialization_and_set_value_for_forward_count_mechanism() {
    let mut graph = Graph::new();

    // 1. 测试变量节点的前向传播次数
    let var1 = graph
        .new_variable_node(&[2, 2], true, true, Some("var1"))
        .unwrap(); // 已初始化
    let var2 = graph
        .new_variable_node(&[2, 2], false, true, Some("var2"))
        .unwrap(); // 未初始化

    // 1.1 未初始化的Variable节点前向传播次数应该为0，已初始化的Variable节点前向传播次数应该为1
    assert_eq!(graph.get_node(var1).unwrap().forward_cnt(), 1);
    assert_eq!(graph.get_node(var2).unwrap().forward_cnt(), 0);
    assert_eq!(graph.forward_cnt(), 0); // 此时图没有执行前向传播，图前向传播次数不变

    // 1.2 无论是否已初始化的Variable节点，也无论被赋予了什么值（包括None），只要图尚未执行前向传播，节点设置值后前向传播次数仍为应该为当前图次数+1
    use rand::Rng;
    let rand_cnt = rand::thread_rng().gen_range(1..=10);
    // 1.2.1 对var1随机设置None值rand_cnt次并验证前向传播次数
    for _ in 0..rand_cnt {
        graph.set_node_value(var1, None).unwrap();
    }
    assert_eq!(graph.get_node(var1).unwrap().forward_cnt(), 1); // 0 + 1

    // 1.2.2 对var2随机设置None值rand_cnt次并验证前向传播次数
    for _ in 0..rand_cnt {
        graph.set_node_value(var2, None).unwrap();
    }
    assert_eq!(graph.get_node(var2).unwrap().forward_cnt(), 1); // 0 + 1

    // 1.2.3 验证图的前向传播次数
    assert_eq!(graph.forward_cnt(), 0); // 此时图仍没有执行前向传播，图前向传播次数仍为0
}

#[test]
fn test_a_complete_case_for_forward_count_mechanism() {
    let mut graph = Graph::new();

    // 1. 测试变量节点的前向传播次数
    let var1 = graph
        .new_variable_node(&[2, 2], true, true, Some("var1"))
        .unwrap(); // 已初始化
    let var2 = graph
        .new_variable_node(&[2, 2], false, true, Some("var2"))
        .unwrap(); // 未初始化

    // 1.1 未初始化的Variable节点前向传播次数应该为0，已初始化的Variable节点前向传播次数应该为1
    assert_eq!(graph.get_node(var1).unwrap().forward_cnt(), 1);
    assert_eq!(graph.get_node(var2).unwrap().forward_cnt(), 0);
    assert_eq!(graph.forward_cnt(), 0); // 此时图没有执行前向传播，图前向传播次数不变

    // 2. 通过Add节点测试整个图的计算节点的前向传播次数
    let add = graph
        .new_add_node(&[var1, var2], Some("add"), true)
        .unwrap();

    // 2.1 初始时前向传播次数应该为0
    assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 0);
    assert_eq!(graph.forward_cnt(), 0); // 此时图仍没有执行前向传播，图前向传播次数仍为0

    // 2.2 尝试前向传播，由于var2未设置值，应该失败
    assert!(matches!(
        graph.forward_node(add),
        Err(GraphError::InvalidOperation(msg)) if msg == "Variable节点[2]不能直接前向传播。问题节点的前向传播次数为0，而图的前向传播次数为1".to_string()
    ));
    assert_eq!(graph.forward_cnt(), 0); // 此时图虽然执行了前向传播，但由于失败了，因此回滚后的图前向传播次数仍不变

    // 2.3 设置var2的值
    let value2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    graph.set_node_value(var2, Some(&value2)).unwrap();
    assert_eq!(graph.get_node(var2).unwrap().forward_cnt(), 1); // 0 + 1
    assert_eq!(graph.forward_cnt(), 0); // 此时图仍没有执行前向传播，图前向传播次数仍为0

    // 2.4 现在前向传播应该成功，add节点的前向传播次数也应该为1
    graph.forward_node(add).unwrap();
    assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 1);
    assert_eq!(graph.forward_cnt(), 1); // 此时图成功执行了前向传播，图前向传播次数+1

    // 2.5 不改变输入值再次前向传播，add节点的“内部”前向传播不应重新计算且不报错
    let old_result = graph.get_node_value(add).unwrap().unwrap().clone();
    graph.forward_node_internal(add).unwrap();
    assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 1); // 没有重新计算，次数不变
    assert_eq!(graph.get_node_value(add).unwrap().unwrap(), &old_result);
    assert_eq!(graph.forward_cnt(), 1); // 此时图没有执行默认的外部前向传播，图前向传播次数不变

    // 2.6 不改变输入值再次前向传播，add节点的前向传播应该报错，因为相应的Variable节点计算次数落后于图的前向传播次数
    assert!(matches!(
        graph.forward_node(add),
        Err(GraphError::InvalidOperation(msg)) if msg == "Variable节点[1]不能直接前向传播。问题节点的前向传播次数为1，而图的前向传播次数为2".to_string()
    ));
    assert_eq!(graph.forward_cnt(), 1); // 此时图虽然执行了前向传播，但由于失败了，因此回滚后的图前向传播次数仍不变

    // 2.7 改变add节点的父节点var1的值后再次对add节点执行前向传播，仍应失败，因为另一个父节点var2的前向传播次数落后于图的前向传播次数
    let new_value1 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    graph.set_node_value(var1, Some(&new_value1)).unwrap();
    assert_eq!(graph.get_node(var1).unwrap().forward_cnt(), 2); // 1 + 1
    assert!(matches!(
        graph.forward_node(add),
        Err(GraphError::InvalidOperation(msg)) if msg == "Variable节点[2]不能直接前向传播。问题节点的前向传播次数为1，而图的前向传播次数为2".to_string()
    ));
    assert_eq!(graph.forward_cnt(), 1); // 此时图虽然执行了前向传播，但由于失败了，因此回滚后的图前向传播次数仍不变

    // 2.8 改变var2的值后再次对add节点执行前向传播，应该成功
    let new_value2 = Tensor::new(&[6.0, 7.0, 8.0, 9.0], &[2, 2]);
    graph.set_node_value(var2, Some(&new_value2)).unwrap();
    graph.forward_node(add).unwrap();
    assert_eq!(graph.get_node(var1).unwrap().forward_cnt(), 2); // 和2.7中一样
    assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 2); // 1 + 1
    assert_eq!(graph.forward_cnt(), 2); // 此时图成功执行了前向传播，图前向传播次数+1
}
