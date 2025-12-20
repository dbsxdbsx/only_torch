use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_node_input_creation() {
    let mut graph = Graph::new();

    // 1. 测试基本创建
    let input = graph.new_input_node(&[2, 3], Some("input1")).unwrap();

    // 1.1 验证基本属性
    assert_eq!(graph.get_node_name(input).unwrap(), "input1");
    assert_eq!(graph.get_node_parents(input).unwrap().len(), 0);
    assert_eq!(graph.get_node_children(input).unwrap().len(), 0);
    assert!(!graph.is_node_inited(input).unwrap()); // Input节点创建时未初始化
}

#[test]
fn test_node_input_creation_with_invalid_shape() {
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

        let result = graph.new_input_node(&shape, None);
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
fn test_node_input_name_generation() {
    let mut graph = Graph::new();

    // 1. 测试节点显式命名
    let input1 = graph
        .new_input_node(&[2, 2], Some("explicit_input"))
        .unwrap();
    assert_eq!(graph.get_node_name(input1).unwrap(), "explicit_input");

    // 2. 测试节点自动命名
    let input2 = graph.new_input_node(&[2, 2], None).unwrap();
    assert_eq!(graph.get_node_name(input2).unwrap(), "input_1");

    // 3. 测试节点名称重复
    let result = graph.new_input_node(&[2, 2], Some("explicit_input"));
    assert_eq!(
        result,
        Err(GraphError::DuplicateNodeName(
            "节点explicit_input在图default_graph中重复".to_string()
        ))
    );
}

#[test]
fn test_node_input_manually_set_value() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 2], Some("test_input")).unwrap();

    // 1. 测试有效赋值
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    {
        let cloned_tensor = test_value.clone();
        graph.set_node_value(input, Some(&cloned_tensor)).unwrap();
    } // cloned_tensor在这里被释放

    // 1.1 验证节点状态
    assert!(graph.is_node_inited(input).unwrap());
    assert_eq!(graph.get_node_value(input).unwrap().unwrap(), &test_value);

    // 2. 测试错误形状的赋值
    let invalid_cases = [
        Tensor::new(&[1.0], &[1, 1]),
        Tensor::new(&[1.0, 2.0], &[2, 1]),
        Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]),
    ];
    for value in invalid_cases {
        assert_eq!(
            graph.set_node_value(input, Some(&value)),
            Err(GraphError::ShapeMismatch {
                expected: vec![2, 2],
                got: value.shape().to_vec(),
                message: format!(
                    "新张量的形状 {:?} 与节点 '{}' 现有张量的形状 {:?} 不匹配。",
                    value.shape(),
                    graph.get_node_name(input).unwrap(),
                    &[2, 2]
                ),
            })
        );
    }

    // 3. 测试设置空值（清除值）
    graph.set_node_value(input, None).unwrap();
    assert!(!graph.is_node_inited(input).unwrap());
    assert!(graph.get_node_value(input).unwrap().is_none());
}

#[test]
fn test_node_input_expected_shape() {
    let mut graph = Graph::new();

    // 1. 测试基本的Input节点预期形状
    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();
    assert_eq!(graph.get_node_value_shape(input).unwrap(), None); // 实际值形状为None
    assert_eq!(graph.get_node_value_expected_shape(input).unwrap(), &[2, 3]); // 预期形状已确定

    // 2. 设置值后检查
    let value = Tensor::zeros(&[2, 3]);
    graph.set_node_value(input, Some(&value)).unwrap();
    assert_eq!(graph.get_node_value_shape(input).unwrap().unwrap(), &[2, 3]); // 设置值后实际形状
    assert_eq!(graph.get_node_value_expected_shape(input).unwrap(), &[2, 3]); // 预期形状保持不变

    // 3. 清除值后检查
    graph.set_node_value(input, None).unwrap();
    assert_eq!(graph.get_node_value_shape(input).unwrap(), None); // 清除后实际值形状为None
    assert_eq!(graph.get_node_value_expected_shape(input).unwrap(), &[2, 3]); // 预期形状仍然保持
}

#[test]
fn test_node_input_forward_propagation() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();

    // 1. 测试前向传播（应该失败，因为Input节点不支持前向传播）
    assert_eq!(
        graph.forward_node(input),
        Err(GraphError::InvalidOperation(format!(
            "节点[id=1, name=input, type=Input]是输入或参数节点，其值应通过set_value设置，而不是通过父节点前向传播计算"
        )))
    );

    // 2. 设置值后仍然不能前向传播
    let value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();
    assert_eq!(
        graph.forward_node(input),
        Err(GraphError::InvalidOperation(format!(
            "节点[id=1, name=input, type=Input]是输入或参数节点，其值应通过set_value设置，而不是通过父节点前向传播计算"
        )))
    );
}

#[test]
fn test_node_input_backward_propagation() {
    let mut graph = Graph::new();

    // 1. 创建输入节点
    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // 2. 初始时雅可比矩阵应为空，但对于输入节点来说，不应该有雅可比矩阵
    assert_eq!(
        graph.get_node_jacobi(input),
        Err(GraphError::InvalidOperation(format!(
            "输入节点[id=1, name=input, type=Input]不应该有雅可比矩阵",
        )))
    );

    // 3. 对自身的反向传播（本应生成单位矩阵，但输入节点不应该有雅可比矩阵，所以应该失败）
    graph.set_node_value(input, Some(&value)).unwrap();
    assert_eq!(
        graph.backward_nodes(&[input], input),
        Err(GraphError::InvalidOperation(format!(
            "输入节点[id=1, name=input, type=Input]不应该有雅可比矩阵",
        )))
    );

    // 4. 清除雅可比矩阵并验证（输入节点不应该有雅可比矩阵）
    graph.clear_jacobi().unwrap();
    assert_eq!(
        graph.get_node_jacobi(input),
        Err(GraphError::InvalidOperation(format!(
            "输入节点[id=1, name=input, type=Input]不应该有雅可比矩阵",
        )))
    );

    // 5. 对没有值的Input节点的反向传播（应失败）
    graph.set_node_value(input, None).unwrap();
    assert_eq!(
        graph.backward_nodes(&[input], input),
        Err(GraphError::InvalidOperation(format!(
            "输入节点[id=1, name=input, type=Input]不应该有雅可比矩阵"
        )))
    );

    // 6. 对其他未关联的Input节点的反向传播（应失败）
    let other_input = graph.new_input_node(&[2, 2], Some("other_input")).unwrap();
    graph.set_node_value(other_input, Some(&value)).unwrap();
    assert_eq!(
        graph.backward_nodes(&[input], other_input),
        Err(GraphError::InvalidOperation(format!(
            "输入节点[id=1, name=input, type=Input]不应该有雅可比矩阵"
        )))
    );
}
