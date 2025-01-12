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

// #[test]
// fn test_variable_node_initialization_and_set_value_for_forward_count_mechanism() {
//     let mut graph = Graph::new();

//     // 1. 测试变量节点的前向传播次数
//     let input1 = graph
//         .new_input_node(&[2, 2], Some("input1"))
//         .unwrap(); // 输入型节点
//     let input2 = graph
//         .new_parameter_node(&[2, 2], Some("input2"))
//         .unwrap(); // 参数型节点

//     // 1.1 未初始化的Variable节点前向传播次数应该为0，已初始化的Variable节点前向传播次数应该为1
//     assert_eq!(graph.get_node(input1).unwrap().forward_cnt(), 1);
//     assert_eq!(graph.get_node(input2).unwrap().forward_cnt(), 0);
//     assert_eq!(graph.forward_cnt(), 0); // 此时图没有执行前向传播，图前向传播次数不变

//     // 1.2 无论是否已初始化的Variable节点，也无论被赋予了什么值（包括None），只要图尚未执行前向传播，节点设置值后前向传播次数仍为应该为当前图次数+1
//     use rand::Rng;
//     let rand_cnt = rand::thread_rng().gen_range(1..=10);
//     // 1.2.1 对input1随机设置None值rand_cnt次并验证前向传播次数
//     for _ in 0..rand_cnt {
//         graph.set_node_value(input1, None).unwrap();
//     }
//     assert_eq!(graph.get_node(input1).unwrap().forward_cnt(), 1); // 0 + 1

//     // 1.2.2 对input2随机设置None值rand_cnt次并验证前向传播次数
//     for _ in 0..rand_cnt {
//         graph.set_node_value(input2, None).unwrap();
//     }
//     assert_eq!(graph.get_node(input2).unwrap().forward_cnt(), 1); // 0 + 1

//     // 1.2.3 验证图的前向传播次数
//     assert_eq!(graph.forward_cnt(), 0); // 此时图仍没有执行前向传播，图前向传播次数仍为0
// }

// #[test]
// fn test_reset_graph_forward_cnt_for_forward_count_mechanism() {
//     let mut graph = Graph::new();

//     // 1. 创建节点并验证初始状态
//     let input1 = graph
//         .new_input_node(&[2, 2], Some("input1"))
//         .unwrap(); // 输入型节点
//     let input2 = graph
//         .new_parameter_node(&[2, 2], Some("input2"))
//         .unwrap(); // 参数型节点
//     let add = graph
//         .new_add_node(&[input1, input2], Some("add"), true)
//         .unwrap();

//     // 1.1 验证初始前向传播次数
//     assert_eq!(graph.get_node(input1).unwrap().forward_cnt(), 1); // 已初始化，次数为1
//     assert_eq!(graph.get_node(input2).unwrap().forward_cnt(), 0); // 未初始化，次数为0
//     assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 0); // 未计算，次数为0
//     assert_eq!(graph.forward_cnt(), 0); // 图未执行前向传播

//     // 2. 设置值并执行前向传播
//     let value2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
//     graph.set_node_value(input2, Some(&value2)).unwrap();
//     graph.forward_node(add).unwrap();

//     // 2.1 验证前向传播后的次数
//     assert_eq!(graph.get_node(input1).unwrap().forward_cnt(), 1);
//     assert_eq!(graph.get_node(input2).unwrap().forward_cnt(), 1);
//     assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 1);
//     assert_eq!(graph.forward_cnt(), 1);

//     // 3. 重置前向传播次数
//     graph.reset_forward_cnt();

//     // 3.1 验证重置后所有节点和图的前向传播次数都应该为0
//     assert_eq!(graph.get_node(input1).unwrap().forward_cnt(), 0);
//     assert_eq!(graph.get_node(input2).unwrap().forward_cnt(), 0);
//     assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 0);
//     assert_eq!(graph.forward_cnt(), 0);

//     // 4. 重置后再次前向传播
//     // 4.1 由于input1和input2的前向传播次数都被重置为0，此时前向传播会失败
//     assert_eq!(
//         graph.forward_node(add),
//         Err(GraphError::InvalidOperation(format!(
//             "节点[id=1, name=input1, type=Variable]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为0，而图的前向传播次数为1",
//         )))
//     );

//     // 4.2 设置节点的值以便后续测试
//     let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
//     graph.set_node_value(input1, Some(&value1)).unwrap();
//     graph.set_node_value(input2, Some(&value2)).unwrap();

//     // 4.2 执行前向传播并验证次数
//     graph.forward_node(add).unwrap();
//     assert_eq!(graph.get_node(input1).unwrap().forward_cnt(), 1);
//     assert_eq!(graph.get_node(input2).unwrap().forward_cnt(), 1);
//     assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 1);
//     assert_eq!(graph.forward_cnt(), 1);
// }

// #[test]
// fn test_a_complete_case_for_forward_count_mechanism() {
//     let mut graph = Graph::new();

//     // 1. 测试变量节点的前向传播次数
//     let input1 = graph
//         .new_input_node(&[2, 2], Some("input1"))
//         .unwrap(); // 输入型节点
//     let input2 = graph
//         .new_parameter_node(&[2, 2], Some("input2"))
//         .unwrap(); // 参数型节点

//     // 1.1 未初始化的Variable节点前向传播次数应该为0，已初始化的Variable节点前向传播次数应该为1
//     assert_eq!(graph.get_node(input1).unwrap().forward_cnt(), 1);
//     assert_eq!(graph.get_node(input2).unwrap().forward_cnt(), 0);
//     assert_eq!(graph.forward_cnt(), 0); // 此时图没有执行前向传播，图前向传播次数不变

//     // 2. 通过Add节点测试整个图的计算节点的前向传播次数
//     let add = graph
//         .new_add_node(&[input1, input2], Some("add"), true)
//         .unwrap();

//     // 2.1 初始时前向传播次数应该为0
//     assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 0);
//     assert_eq!(graph.forward_cnt(), 0); // 此时图仍没有执行前向传播，图前向传播次数仍为0

//     // 2.2 尝试前向传播，由于input2未设置值，应该失败
//     assert_eq!(
//         graph.forward_node(add),
//         Err(GraphError::InvalidOperation(format!(
//             "节点[id=2, name=input2, type=Variable]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为0，而图的前向传播次数为1",
//         )))
//     );
//     assert_eq!(graph.forward_cnt(), 0); // 此时图虽然执行了前向传播，但由于失败了，因此回滚后的图前向传播次数仍不变

//     // 2.3 设置input2的值
//     let value2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
//     graph.set_node_value(input2, Some(&value2)).unwrap();
//     assert_eq!(graph.get_node(input2).unwrap().forward_cnt(), 1); // 0 + 1
//     assert_eq!(graph.forward_cnt(), 0); // 此时图仍没有执行前向传播，图前向传播次数仍为0

//     // 2.4 现在前向传播应该成功，add节点的前向传播次数也应该为1
//     graph.forward_node(add).unwrap();
//     assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 1);
//     assert_eq!(graph.forward_cnt(), 1); // 此时图成功执行了前向传播，图前向传播次数+1

//     // 2.5 不改变输入值再次前向传播，add节点的"内部"前向传播不应重新计算且不报错
//     let old_result = graph.get_node_value(add).unwrap().unwrap().clone();
//     graph.forward_node_internal(add).unwrap();
//     assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 1); // 没有重新计算，次数不变
//     assert_eq!(graph.get_node_value(add).unwrap().unwrap(), &old_result);
//     assert_eq!(graph.forward_cnt(), 1); // 此时图没有执行默认的外部前向传播，图前向传播次数不变

//     // 2.6 不改变输入值再次前向传播，add节点的前向传播应该报错，因为相应的Variable节点计算次数落后于图的前向传播次数
//     assert_eq!(
//         graph.forward_node(add),
//         Err(GraphError::InvalidOperation(format!(
//             "节点[id=1, name=input1, type=Variable]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为1，而图的前向传播次数为2",
//         )))
//     );
//     assert_eq!(graph.forward_cnt(), 1); // 此时图虽然执行了前向传播，但由于失败了，因此回滚后的图前向传播次数仍不变

//     // 2.7 改变add节点的父节点input1的值后再次对add节点执行前向传播，仍应失败，因为另一个父节点input2的前向传播次数落后于图的前向传播次数
//     let new_value1 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
//     graph.set_node_value(input1, Some(&new_value1)).unwrap();
//     assert_eq!(graph.get_node(input1).unwrap().forward_cnt(), 2); // 1 + 1
//     assert_eq!(
//         graph.forward_node(add),
//         Err(GraphError::InvalidOperation(format!(
//             "节点[id=2, name=input2, type=Variable]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为1，而图的前向传播次数为2",
//         )))
//     );
//     assert_eq!(graph.forward_cnt(), 1); // 此时图虽然执行了前向传播，但由于失败了，因此回滚后的图前向传播次数仍不变

//     // 2.8 改变input2的值后再次对add节点执行前向传播，应该成功
//     let new_value2 = Tensor::new(&[6.0, 7.0, 8.0, 9.0], &[2, 2]);
//     graph.set_node_value(input2, Some(&new_value2)).unwrap();
//     graph.forward_node(add).unwrap();
//     assert_eq!(graph.get_node(input1).unwrap().forward_cnt(), 2); // 和2.7中一样
//     assert_eq!(graph.get_node(add).unwrap().forward_cnt(), 2); // 1 + 1
//     assert_eq!(graph.forward_cnt(), 2); // 此时图成功执行了前向传播，图前向传播次数+1
// }
