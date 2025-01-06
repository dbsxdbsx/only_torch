use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_new_node_variable_with_no_initialization() {
    let mut graph = Graph::new();

    // 创建未初始化的变量节点
    let var_id = graph
        .new_variable_node(&[2, 3], false, false, None)
        .unwrap();

    // 验证节点属性
    assert!(!graph.is_node_inited(var_id).unwrap());
}

#[test]
fn test_new_node_variable_with_initialization() {
    let mut graph = Graph::new();

    // 创建带初始化的变量节点
    let var_id = graph.new_variable_node(&[2, 3], true, false, None).unwrap();

    // 验证节点属性
    assert!(graph.is_node_inited(var_id).unwrap());

    // 验证初始化值
    let value = graph.get_node_value(var_id).unwrap().unwrap();
    let mean = value.mean();
    let std_dev = value.std_dev();
    assert!(mean.abs() < 0.1); // 均值应接近0
    assert!((std_dev - 0.001).abs() < 0.001); // 标准差应接近0.001
}

#[test]
fn test_new_node_variable_with_invalid_shape() {
    let mut graph = Graph::new();

    // 测试不同维度的形状(除2维外都应该失败)
    for dims in [0, 1, 3, 4, 5] {
        let shape = match dims {
            0 => vec![],
            1 => vec![2],
            3 => vec![2, 2, 2],
            4 => vec![2, 2, 2, 2],
            5 => vec![2, 2, 2, 2, 2],
            _ => unreachable!(),
        };

        let result = graph.new_variable_node(&shape, false, false, None);
        assert_eq!(
            result,
            Err(GraphError::DimensionMismatch {
                expected: 2,
                got: dims,
                message: format!(
                    "神经网络中的节点张量必须是2维的（矩阵），但收到的维度是{}维。",
                    dims
                ),
            })
        );
    }
}

#[test]
fn test_node_variable_trainable_flag() {
    let mut graph = Graph::new();

    // 1. 测试初始为非可训练节点
    let var_id = graph
        .new_variable_node(&[2, 3], false, false, Some("test_var"))
        .unwrap();
    assert!(!graph.is_node_trainable(var_id).unwrap());
    // 1.1 测试trainable标志的后期设置和检查
    graph.set_node_trainable(var_id, true).unwrap();
    assert!(graph.is_node_trainable(var_id).unwrap());
    graph.set_node_trainable(var_id, false).unwrap();
    assert!(!graph.is_node_trainable(var_id).unwrap());

    // 2. 测试初始为可训练节点
    let trainable_var_id = graph
        .new_variable_node(&[2, 3], false, true, Some("trainable_var"))
        .unwrap();
    assert!(graph.is_node_trainable(trainable_var_id).unwrap());
    // 2.1 测试trainable标志的后期设置和检查
    graph.set_node_trainable(trainable_var_id, false).unwrap();
    assert!(!graph.is_node_trainable(trainable_var_id).unwrap());
    graph.set_node_trainable(trainable_var_id, true).unwrap();
    assert!(graph.is_node_trainable(trainable_var_id).unwrap());
}

#[test]
fn test_node_variable_name_generation() {
    // 1. 测试节点显式命名
    // 1.1 图默认命名+节点显式命名
    let mut graph = Graph::new();
    let var1 = graph
        .new_variable_node(&[2, 2], false, false, Some("explicit_var"))
        .unwrap();
    assert_eq!(graph.get_node_name(var1).unwrap(), "explicit_var");

    // 1.2 图显式命名+节点显式命名
    let mut graph_with_name = Graph::with_name("custom_graph");
    let var_named = graph_with_name
        .new_variable_node(&[2, 2], false, false, Some("explicit_var"))
        .unwrap();
    assert_eq!(
        graph_with_name.get_node_name(var_named).unwrap(),
        "explicit_var"
    );

    // 2. 测试节点自动命名
    // 2.1 图默认命名+节点默认命名
    let var2 = graph
        .new_variable_node(&[2, 2], false, false, None)
        .unwrap();
    assert_eq!(graph.get_node_name(var2).unwrap(), "variable_1");

    // 2.2 图显式命名+节点默认命名
    let var_custom = graph_with_name
        .new_variable_node(&[2, 2], false, false, None)
        .unwrap();
    assert_eq!(
        graph_with_name.get_node_name(var_custom).unwrap(),
        "variable_1"
    );

    // 3. 测试重复名称的处理
    // 3.1 测试显式重复名称
    let duplicate_result = graph.new_variable_node(&[2, 2], false, false, Some("explicit_var"));
    assert_eq!(
        duplicate_result,
        Err(GraphError::DuplicateNodeName(
            "节点explicit_var在图default_graph中重复".to_string()
        ))
    );

    // 3.2 测试在不同图中可以使用相同名称
    let mut another_graph = Graph::with_name("another_graph");
    let var_another = another_graph
        .new_variable_node(&[2, 2], false, false, Some("explicit_var"))
        .unwrap();
    assert_eq!(
        another_graph.get_node_name(var_another).unwrap(),
        "explicit_var"
    );
}

#[test]
fn test_node_variable_manually_set_value() {
    let mut graph = Graph::new();
    let var = graph
        .new_variable_node(&[2, 2], true, true, Some("test_var"))
        .unwrap();

    // 1. 测试有效赋值
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    {
        let cloned_tensor = test_value.clone();
        graph.set_node_value(var, Some(&cloned_tensor)).unwrap();
    } // cloned_tensor在这里被释放

    // 1.1 验证节点状态
    assert!(graph.is_node_inited(var).unwrap());
    assert_eq!(graph.get_node_value(var).unwrap().unwrap(), &test_value);

    // 2. 测试错误形状的赋值
    let invalid_cases = [
        Tensor::new(&[1.0], &[1, 1]),
        Tensor::new(&[1.0, 2.0], &[2, 1]),
        Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]),
    ];
    for value in invalid_cases {
        assert_eq!(
            graph.set_node_value(var, Some(&value)),
            Err(GraphError::ShapeMismatch {
                expected: vec![2, 2],
                got: value.shape().to_vec(),
                message: format!(
                    "新张量的形状 {:?} 与节点 '{}' 现有张量的形状 {:?} 不匹配。",
                    value.shape(),
                    graph.get_node_name(var).unwrap(),
                    &[2, 2]
                ),
            })
        );
    }

    // 3. 测试设置空值（清除值）
    graph.set_node_value(var, None).unwrap();
    assert!(!graph.is_node_inited(var).unwrap());
    assert!(graph.get_node_value(var).unwrap().is_none());
}

#[test]
fn test_node_variable_expected_shape() {
    let mut graph = Graph::new();

    // 创建一个未初始化的Variable节点
    let var = graph
        .new_variable_node(&[2, 3], false, false, Some("var"))
        .unwrap();

    // 1. 初始状态检查
    assert_eq!(graph.get_node_value_shape(var).unwrap(), None); // 实际值形状为None
    assert_eq!(graph.get_node_value_expected_shape(var).unwrap(), &[2, 3]); // 预期形状已确定

    // 2. 设置值后检查
    let value = Tensor::zeros(&[2, 3]);
    graph.set_node_value(var, Some(&value)).unwrap();
    assert_eq!(graph.get_node_value_shape(var).unwrap().unwrap(), &[2, 3]); // 设置值后实际形状
    assert_eq!(graph.get_node_value_expected_shape(var).unwrap(), &[2, 3]); // 预期形状保持不变

    // 3. 清除值后检查
    graph.set_node_value(var, None).unwrap();
    assert_eq!(graph.get_node_value_shape(var).unwrap(), None); // 清除后实际值形状为None
    assert_eq!(graph.get_node_value_expected_shape(var).unwrap(), &[2, 3]); // 预期形状仍然保持
}
#[test]
fn test_node_variable_forward_propagation() {
    let mut graph = Graph::new();

    // 测试所有init、trainable和set_value组合的前向传播
    let mut i = 0;
    for is_init in [false, true] {
        for is_trainable in [false, true] {
            for set_value in [false, true] {
                i += 1;
                let var = graph
                    .new_variable_node(&[2, 2], is_init, is_trainable, None)
                    .unwrap();

                // 如果需要设置值
                if set_value {
                    let value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
                    graph.set_node_value(var, Some(&value)).unwrap();
                }

                // 测试前向传播报错
                assert_eq!(
                    graph.forward_node(var),
                    Err(GraphError::InvalidOperation(
                        format!("节点[id={i}, name=variable_{i}, type=Variable]是输入节点，其值应通过set_value设置，而不是通过父节点前向传播计算", i=i)
                    ))
                );
            }
        }
    }
}

#[test]
fn test_node_variable_backward_propagation() {
    let mut graph = Graph::new();
    let trainable_var = graph.new_variable_node(&[2, 2], true, true, None).unwrap();
    // 1. 初始时雅可比矩阵应为空
    assert!(graph.get_node_jacobi(trainable_var).unwrap().is_none());

    // 2. 基本反向传播
    // 2.1 对自身的反向传播（应生成单位矩阵）
    graph.backward_node(trainable_var, trainable_var).unwrap();
    assert_eq!(
        graph.get_node_jacobi(trainable_var).unwrap().unwrap(),
        &Tensor::eyes(4)
    );
    graph.clear_jacobi().unwrap();
    assert!(graph.get_node_jacobi(trainable_var).unwrap().is_none());
    // 2.2. NOTE: 这里不用对具有正常父子节点的Variable节点进行反向传播测试，因其没有反向传播运算。

    // 3. 特殊情况
    // 3.1 可训练但未初始化的节点对其自身的雅可比（应失败）
    let trainable_but_non_inited_var = graph.new_variable_node(&[2, 2], false, true, None).unwrap();
    assert_eq!(
        graph.backward_node(trainable_but_non_inited_var, trainable_but_non_inited_var),
        Err(GraphError::ComputationError(
            "反向传播：节点[id=2, name=variable_2, type=Variable]没有值".to_string()
        ))
    );

    // 3.2 不可训练但初始化的节点对其自身的雅可比（应失败）
    let non_trainable_var = graph.new_variable_node(&[2, 2], true, false, None).unwrap();
    assert_eq!(
        graph.backward_node(non_trainable_var, non_trainable_var),
        Err(GraphError::InvalidOperation(
            "不能对不可训练的节点[id=3, name=variable_3, type=Variable]进行反向传播".to_string()
        ))
    );

    // 3.3 可训练节点对不可训练节点的雅可比（应失败）
    assert_eq!(
        graph.backward_node(non_trainable_var, trainable_var),
        Err(GraphError::InvalidOperation(
            "不能对不可训练的节点[id=3, name=variable_3, type=Variable]进行反向传播".to_string()
        ))
    );

    // 3.4 可训练节点对另一可训练节点的雅可比，但彼此没有关系（应失败）
    let other_trainable_var = graph.new_variable_node(&[2, 2], true, true, None).unwrap();
    assert_eq!(
        graph.backward_node(trainable_var, other_trainable_var),
        Err(GraphError::InvalidOperation(
            "无法对没有子节点的节点[id=1, name=variable_1, type=Variable]进行反向传播".to_string()
        ))
    );
}
