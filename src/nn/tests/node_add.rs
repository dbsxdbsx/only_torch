use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_new_node_add_with_inited_parents() {
    let mut graph = Graph::new();

    // 1. 测试基本构造（2个父节点）
    let var1 = graph
        .new_variable_node(&[2, 3], true, false, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[2, 3], true, false, Some("var2"))
        .unwrap();
    let add = graph
        .new_add_node(&[var1, var2], Some("add"), true)
        .unwrap();
    // 验证基本属性
    assert_eq!(graph.get_node_parents(add).unwrap().len(), 2);
    assert_eq!(graph.get_node_children(add).unwrap().len(), 0);
    assert_eq!(graph.get_node_name(add).unwrap(), "add");
    assert!(graph.is_node_trainable(add).unwrap());

    // 2. 测试多父节点构造（3个父节点）
    let var3 = graph
        .new_variable_node(&[2, 3], true, false, Some("var3"))
        .unwrap();
    let add_multi = graph
        .new_add_node(&[var1, var2, var3], Some("add_multi"), true)
        .unwrap();
    assert_eq!(graph.get_node_parents(add_multi).unwrap().len(), 3);
    assert_eq!(graph.get_node_children(add_multi).unwrap().len(), 0);
    assert_eq!(graph.get_node_name(add_multi).unwrap(), "add_multi");

    // 3. 测试父节点不足错误
    let result = graph.new_add_node(&[var1], None, true);
    assert_eq!(
        result,
        Err(GraphError::InvalidOperation(
            "Add节点至少需要2个父节点".into()
        ))
    );
}

#[test]
fn test_new_node_add_with_uninited_parents() {
    let mut graph = Graph::new();

    let var1 = graph
        .new_variable_node(&[2, 3], false, false, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[2, 3], false, false, Some("var2"))
        .unwrap();
    let add = graph
        .new_add_node(&[var1, var2], Some("add"), true)
        .unwrap();
    // 验证基本属性
    assert_eq!(graph.get_node_parents(add).unwrap().len(), 2);
    assert_eq!(graph.get_node_children(add).unwrap().len(), 0);
    assert_eq!(graph.get_node_name(add).unwrap(), "add");
    assert!(graph.is_node_trainable(add).unwrap());
}

#[test]
fn test_new_node_add_with_inconsistent_shape() {
    let mut graph = Graph::new();

    // 1. 创建形状不同的父节点（都是2维，但形状不同）
    let var1 = graph
        .new_variable_node(&[2, 2], true, true, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[3, 2], true, true, Some("var2"))
        .unwrap();
    let var3 = graph
        .new_variable_node(&[2, 3], true, true, Some("var3"))
        .unwrap();

    // 2. 设置父节点的值
    let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let value2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let value3 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(var1, Some(&value1)).unwrap();
    graph.set_node_value(var2, Some(&value2)).unwrap();
    graph.set_node_value(var3, Some(&value3)).unwrap();

    // 3. 测试不同形状组合
    // 3.1 测试行数不同
    let result = graph.new_add_node(&[var1, var2], None, true);
    assert_eq!(
        result,
        Err(GraphError::ShapeMismatch {
            expected: vec![2, 2],
            got: vec![3, 2],
            message: "Add节点的所有父节点形状必须相同".to_string()
        })
    );

    // 3.2 测试列数不同
    let result = graph.new_add_node(&[var1, var3], None, true);
    assert_eq!(
        result,
        Err(GraphError::ShapeMismatch {
            expected: vec![2, 2],
            got: vec![2, 3],
            message: "Add节点的所有父节点形状必须相同".to_string()
        })
    );

    // 3.3 测试多个父节点的情况
    let result = graph.new_add_node(&[var1, var2, var3], None, true);
    assert_eq!(
        result,
        Err(GraphError::ShapeMismatch {
            expected: vec![2, 2],
            got: vec![3, 2],
            message: "Add节点的所有父节点形状必须相同".to_string()
        })
    );
}

#[test]
fn test_node_add_trainable_flag() {
    let mut graph = Graph::new();

    // 1. 测试初始为可训练节点
    let var1 = graph
        .new_variable_node(&[2, 3], true, false, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[2, 3], true, false, Some("var2"))
        .unwrap();
    let add = graph
        .new_add_node(&[var1, var2], Some("add"), true)
        .unwrap();
    assert!(graph.is_node_trainable(add).unwrap());
    // 1.1 测试trainable标志的后期修改
    graph.set_node_trainable(add, false).unwrap();
    assert!(!graph.is_node_trainable(add).unwrap());
    graph.set_node_trainable(add, true).unwrap();
    assert!(graph.is_node_trainable(add).unwrap());

    // 2. 测试初始为不可训练节点
    let add_non_trainable = graph
        .new_add_node(&[var1, var2], Some("add_non_trainable"), false)
        .unwrap();
    assert!(!graph.is_node_trainable(add_non_trainable).unwrap());
    // 2.1 测试trainable标志的后期修改
    graph.set_node_trainable(add_non_trainable, true).unwrap();
    assert!(graph.is_node_trainable(add_non_trainable).unwrap());
    graph.set_node_trainable(add_non_trainable, false).unwrap();
    assert!(!graph.is_node_trainable(add_non_trainable).unwrap());
}

#[test]
fn test_node_add_name_generation() {
    // 1. 测试节点显式命名
    // 1.1 图默认命名+节点显式命名
    let mut graph = Graph::new();
    let var1 = graph
        .new_variable_node(&[2, 3], true, false, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[2, 3], true, false, Some("var2"))
        .unwrap();
    let add1 = graph
        .new_add_node(&[var1, var2], Some("explicit_add"), true)
        .unwrap();
    assert_eq!(graph.get_node_name(add1).unwrap(), "explicit_add");

    // 1.2 图显式命名+节点显式命名
    let mut graph_with_name = Graph::with_name("custom_graph");
    let var1 = graph_with_name
        .new_variable_node(&[2, 3], true, false, Some("var1"))
        .unwrap();
    let var2 = graph_with_name
        .new_variable_node(&[2, 3], true, false, Some("var2"))
        .unwrap();
    let add_named = graph_with_name
        .new_add_node(&[var1, var2], Some("explicit_add"), true)
        .unwrap();
    assert_eq!(
        graph_with_name.get_node_name(add_named).unwrap(),
        "explicit_add"
    );

    // 2. 测试节点自动命名
    // 2.1 图默认命名+节点默认命名
    let add2 = graph.new_add_node(&[var1, var2], None, true).unwrap();
    assert_eq!(graph.get_node_name(add2).unwrap(), "add_1");

    // 2.2 图显式命名+节点默认命名
    let add_custom = graph_with_name
        .new_add_node(&[var1, var2], None, true)
        .unwrap();
    assert_eq!(graph_with_name.get_node_name(add_custom).unwrap(), "add_1");

    // 3. 测试重复名称的处理
    // 3.1 测试显式重复名称
    let duplicate_result = graph.new_add_node(&[var1, var2], Some("explicit_add"), true);
    assert_eq!(
        duplicate_result,
        Err(GraphError::DuplicateNodeName(
            "节点explicit_add在图default_graph中重复".to_string()
        ))
    );

    // 3.2 测试在不同图中可以使用相同名称
    let mut another_graph = Graph::with_name("another_graph");
    let var1 = another_graph
        .new_variable_node(&[2, 3], true, false, Some("var1"))
        .unwrap();
    let var2 = another_graph
        .new_variable_node(&[2, 3], true, false, Some("var2"))
        .unwrap();
    let add_another = another_graph
        .new_add_node(&[var1, var2], Some("explicit_add"), true)
        .unwrap();
    assert_eq!(
        another_graph.get_node_name(add_another).unwrap(),
        "explicit_add"
    );
}

#[test]
fn test_node_add_manually_set_value() {
    let mut graph = Graph::new();
    let var1 = graph
        .new_variable_node(&[2, 2], true, true, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[2, 2], true, true, Some("var2"))
        .unwrap();
    let add = graph
        .new_add_node(&[var1, var2], Some("add"), true)
        .unwrap();

    // 1. 测试直接设置Add节点的值（应该失败）
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        graph.set_node_value(add, Some(&test_value)),
        Err(GraphError::InvalidOperation(
            "Add节点的值只能通过前向传播计算得到，不能直接设置".into()
        ))
    );

    // 2. 测试清除Add节点的值（也应该失败）
    assert_eq!(
        graph.set_node_value(add, None),
        Err(GraphError::InvalidOperation(
            "Add节点的值只能通过前向传播计算得到，不能直接设置".into()
        ))
    );
}

#[test]
fn test_node_add_expected_shape() {
    let mut graph = Graph::new();

    // 1. 测试基本的Add节点预期形状
    let var1 = graph
        .new_variable_node(&[2, 3], false, false, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[2, 3], false, false, Some("var2"))
        .unwrap();
    let add = graph
        .new_add_node(&[var1, var2], Some("add"), true)
        .unwrap();

    // 1.1 验证Add节点的预期形状（应该与父节点相同）
    assert_eq!(graph.get_node_value_expected_shape(add).unwrap(), &[2, 3]);
    assert_eq!(graph.get_node_value_shape(add).unwrap(), None); // 实际值形状为None（未计算）

    // 2. 测试前向传播后的形状
    let value1 = Tensor::zeros(&[2, 3]);
    let value2 = Tensor::zeros(&[2, 3]);
    graph.set_node_value(var1, Some(&value1)).unwrap();
    graph.set_node_value(var2, Some(&value2)).unwrap();
    graph.forward_node(add).unwrap();

    // 2.1 验证前向传播后的形状
    assert_eq!(graph.get_node_value_shape(add).unwrap().unwrap(), &[2, 3]); // 实际值形状
    assert_eq!(graph.get_node_value_expected_shape(add).unwrap(), &[2, 3]); // 预期形状保持不变

    // 2.2 测试父节点值在首次前向传播后，再次设置新值后的形状检查
    let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let value2 = Tensor::new(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3]);
    graph.set_node_value(var1, Some(&value1)).unwrap();
    graph.set_node_value(var2, Some(&value2)).unwrap();

    // 验证预期形状和实际形状(应该报错,因为add节点的前向代数落后)
    assert_eq!(graph.get_node_value_expected_shape(add).unwrap(), &[2, 3]);
    assert_eq!(graph.get_node_value_shape(add).unwrap().unwrap(), &[2, 3]); // 虽然值已过期，但由于值仍然存在，所以形状不变

    // 前向传播后再次验证
    graph.forward_node(add).unwrap();
    assert_eq!(graph.get_node_value_expected_shape(add).unwrap(), &[2, 3]);
    assert_eq!(graph.get_node_value_shape(add).unwrap().unwrap(), &[2, 3]);

    // 3. 测试多个父节点的情况
    let mut graph = Graph::new();

    let var1 = graph
        .new_variable_node(&[2, 3], false, false, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[2, 3], false, false, Some("var2"))
        .unwrap();
    let var3 = graph
        .new_variable_node(&[2, 3], false, false, Some("var3"))
        .unwrap();
    let add_multi = graph
        .new_add_node(&[var1, var2, var3], Some("add_multi"), true)
        .unwrap();

    // 3.1 验证多父节点Add的预期形状
    assert_eq!(
        graph.get_node_value_expected_shape(add_multi).unwrap(),
        &[2, 3]
    );
    assert_eq!(graph.get_node_value_shape(add_multi).unwrap(), None);

    // 3.2 设置所有父节点的值并前向传播
    let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let value2 = Tensor::new(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3]);
    let value3 = Tensor::new(&[13.0, 14.0, 15.0, 16.0, 17.0, 18.0], &[2, 3]);
    graph.set_node_value(var1, Some(&value1)).unwrap();
    graph.set_node_value(var2, Some(&value2)).unwrap();
    graph.set_node_value(var3, Some(&value3)).unwrap();
    graph.forward_node(add_multi).unwrap();

    // 3.3 验证前向传播后的形状
    assert_eq!(
        graph.get_node_value_expected_shape(add_multi).unwrap(),
        &[2, 3]
    );
    assert_eq!(
        graph.get_node_value_shape(add_multi).unwrap().unwrap(),
        &[2, 3]
    );
}

// #[test]
// fn test_node_add_forward_propagation() {
//     let mut graph = Graph::new();

//     // 1. 测试基本的前向传播计算
//     let var1 = graph
//         .new_variable_node(&[2, 2], true, true, Some("var1"))
//         .unwrap();
//     let var2 = graph
//         .new_variable_node(&[2, 2], true, true, Some("var2"))
//         .unwrap();
//     let add = graph
//         .new_add_node(&[var1, var2], Some("add"), true)
//         .unwrap();

//     // 1.1 设置父节点的值
//     let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
//     let value2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
//     graph.set_node_value(var1, Some(&value1)).unwrap();
//     graph.set_node_value(var2, Some(&value2)).unwrap();

//     // 1.2 前向传播并验证结果
//     graph.forward_node(add).unwrap();
//     let result = graph.get_node_value(add).unwrap().unwrap();
//     let expected = &value1 + &value2;
//     assert_eq!(result, &expected);

//     // 2. 测试不同初始化和可训练组合的前向传播
//     for var1_trainable in [false, true] {
//         for var2_trainable in [false, true] {
//             for add_trainable in [false, true] {
//                 for var1_inited in [false, true] {
//                     for var2_inited in [false, true] {
//                         let var1 = graph
//                             .new_variable_node(&[2, 2], var1_inited, var1_trainable, None)
//                             .unwrap();
//                         let var2 = graph
//                             .new_variable_node(&[2, 2], var2_inited, var2_trainable, None)
//                             .unwrap();
//                         let add = graph
//                             .new_add_node(&[var1, var2], None, add_trainable)
//                             .unwrap();

//                         if var1_inited && var2_inited {
//                             // 若两个父节点都初始化，前向传播应成功
//                             graph.forward_node(add).unwrap();
//                             assert!(graph.get_node_value(add).unwrap().is_some());
//                         } else {
//                             // 若有未初始化的父节点，前向传播应失败
//                             assert_eq!(
//                                 graph.forward_node(add),
//                                 Err(GraphError::ComputationError("父节点没有值".to_string()))
//                             );
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// #[test]
// fn test_node_add_backward_propagation() {
//     let mut graph = Graph::new();

//     // 1. 创建一个简单的加法图：z = x + y
//     let x = graph
//         .new_variable_node(&[2, 2], true, true, Some("x"))
//         .unwrap();
//     let y = graph
//         .new_variable_node(&[2, 2], true, true, Some("y"))
//         .unwrap();
//     let z = graph.new_add_node(&[x, y], Some("z"), true).unwrap();

//     // 2. 设置输入值
//     let x_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
//     let y_value = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
//     graph.set_node_value(x, Some(&x_value)).unwrap();
//     graph.set_node_value(y, Some(&y_value)).unwrap();

//     // 3. 前向传播
//     graph.forward_node(z).unwrap();

//     // 4. 反向传播
//     // 4.1 对x的反向传播
//     graph.backward_node(x, z).unwrap();
//     let x_jacobi = graph.get_node_jacobi(x).unwrap().unwrap();
//     assert_eq!(x_jacobi, &Tensor::eyes(4)); // ∂z/∂x = I

//     // 4.2 对y的反向传播
//     graph.backward_node(y, z).unwrap();
//     let y_jacobi = graph.get_node_jacobi(y).unwrap().unwrap();
//     assert_eq!(y_jacobi, &Tensor::eyes(4)); // ∂z/∂y = I

//     // 5. 清除雅可比矩阵并验证
//     graph.clear_jacobi().unwrap();
//     assert!(graph.get_node_jacobi(x).unwrap().is_none());
//     assert!(graph.get_node_jacobi(y).unwrap().is_none());
// }
