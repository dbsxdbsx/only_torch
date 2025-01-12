use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_node_parameter_creation() {
    let mut graph = Graph::new();

    // 1. 测试基本创建
    let param = graph.new_parameter_node(&[2, 3], Some("param1")).unwrap();

    // 1.1 验证基本属性
    assert_eq!(graph.get_node_name(param).unwrap(), "param1");
    assert_eq!(graph.get_node_parents(param).unwrap().len(), 0);
    assert_eq!(graph.get_node_children(param).unwrap().len(), 0);
    assert!(graph.is_node_inited(param).unwrap()); // Parameter节点创建时已初始化

    // 1.2 验证初始化值
    let value = graph.get_node_value(param).unwrap().unwrap();
    let mean = value.mean();
    let std_dev = value.std_dev();
    assert!(mean.abs() < 0.1); // 均值应接近0
    assert!((std_dev - 0.001).abs() < 0.001); // 标准差应接近0.001
}

#[test]
fn test_node_parameter_creation_with_invalid_shape() {
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

        let result = graph.new_parameter_node(&shape, None);
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
fn test_node_parameter_name_generation() {
    let mut graph = Graph::new();

    // 1. 测试节点显式命名
    let param1 = graph
        .new_parameter_node(&[2, 2], Some("explicit_param"))
        .unwrap();
    assert_eq!(graph.get_node_name(param1).unwrap(), "explicit_param");

    // 2. 测试节点自动命名
    let param2 = graph.new_parameter_node(&[2, 2], None).unwrap();
    assert_eq!(graph.get_node_name(param2).unwrap(), "parameter_1");

    // 3. 测试节点名称重复
    let result = graph.new_parameter_node(&[2, 2], Some("explicit_param"));
    assert_eq!(
        result,
        Err(GraphError::DuplicateNodeName(
            "节点explicit_param在图default_graph中重复".to_string()
        ))
    );
}

#[test]
fn test_node_parameter_manually_set_value() {
    let mut graph = Graph::new();
    let param = graph
        .new_parameter_node(&[2, 2], Some("test_param"))
        .unwrap();

    // 1. 测试有效赋值
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    {
        let cloned_tensor = test_value.clone();
        graph.set_node_value(param, Some(&cloned_tensor)).unwrap();
    } // cloned_tensor在这里被释放

    // 1.1 验证节点状态
    assert!(graph.is_node_inited(param).unwrap());
    assert_eq!(graph.get_node_value(param).unwrap().unwrap(), &test_value);

    // 2. 测试错误形状的赋值
    let invalid_cases = [
        Tensor::new(&[1.0], &[1, 1]),
        Tensor::new(&[1.0, 2.0], &[2, 1]),
        Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]),
    ];
    for value in invalid_cases {
        assert_eq!(
            graph.set_node_value(param, Some(&value)),
            Err(GraphError::ShapeMismatch {
                expected: vec![2, 2],
                got: value.shape().to_vec(),
                message: format!(
                    "新张量的形状 {:?} 与节点 '{}' 现有张量的形状 {:?} 不匹配。",
                    value.shape(),
                    graph.get_node_name(param).unwrap(),
                    &[2, 2]
                ),
            })
        );
    }

    // 3. 测试设置空值（清除值）
    graph.set_node_value(param, None).unwrap();
    assert!(!graph.is_node_inited(param).unwrap());
    assert!(graph.get_node_value(param).unwrap().is_none());
}

#[test]
fn test_node_parameter_expected_shape() {
    let mut graph = Graph::new();

    // 1. 测试基本的Parameter节点预期形状
    let param = graph.new_parameter_node(&[2, 3], Some("param")).unwrap();
    assert_eq!(graph.get_node_value_shape(param).unwrap().unwrap(), &[2, 3]); // 实际值形状（已初始化）
    assert_eq!(graph.get_node_value_expected_shape(param).unwrap(), &[2, 3]); // 预期形状已确定

    // 2. 设置新值后检查
    let value = Tensor::zeros(&[2, 3]);
    graph.set_node_value(param, Some(&value)).unwrap();
    assert_eq!(graph.get_node_value_shape(param).unwrap().unwrap(), &[2, 3]); // 设置值后实际形状
    assert_eq!(graph.get_node_value_expected_shape(param).unwrap(), &[2, 3]); // 预期形状保持不变

    // 3. 清除值后检查
    graph.set_node_value(param, None).unwrap();
    assert_eq!(graph.get_node_value_shape(param).unwrap(), None); // 清除后实际值形状为None
    assert_eq!(graph.get_node_value_expected_shape(param).unwrap(), &[2, 3]); // 预期形状仍然保持
}

#[test]
fn test_node_parameter_forward_propagation() {
    let mut graph = Graph::new();
    let param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();

    // 1. 测试前向传播（应该失败，因为Parameter节点不支持前向传播）
    assert_eq!(
        graph.forward_node(param),
        Err(GraphError::InvalidOperation(
            format!("节点[id=1, name=param, type=Parameter]是输入或参数节点，其值应通过set_value设置，而不是通过父节点前向传播计算")
        ))
    );

    // 2. 设置新值后仍然不能前向传播
    let value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    graph.set_node_value(param, Some(&value)).unwrap();
    assert_eq!(
        graph.forward_node(param),
        Err(GraphError::InvalidOperation(
            format!("节点[id=1, name=param, type=Parameter]是输入或参数节点，其值应通过set_value设置，而不是通过父节点前向传播计算")
        ))
    );
}

#[test]
fn test_node_parameter_backward_propagation() {
    let mut graph = Graph::new();
    let param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();

    // 1. 初始时雅可比矩阵应为空
    assert!(graph.get_node_jacobi(param).unwrap().is_none());

    // 2. 对自身的反向传播（应生成单位矩阵）
    graph.backward_node(param, param).unwrap();
    assert_eq!(
        graph.get_node_jacobi(param).unwrap().unwrap(),
        &Tensor::eyes(4)
    );

    // 3. 清除雅可比矩阵并验证
    graph.clear_jacobi().unwrap();
    assert!(graph.get_node_jacobi(param).unwrap().is_none());

    // 4. 清除值(未初始化)后对自身的反向传播（应失败）
    graph.set_node_value(param, None).unwrap();
    assert_eq!(
        graph.backward_node(param, param),
        Err(GraphError::ComputationError(
            "反向传播：节点[id=1, name=param, type=Parameter]没有值".to_string()
        ))
    );

    // 5. 对其他未关联的Parameter节点的反向传播（应失败）
    let other_param = graph
        .new_parameter_node(&[2, 2], Some("other_param"))
        .unwrap();
    assert_eq!(
        graph.backward_node(param, other_param),
        Err(GraphError::InvalidOperation(
            "无法对没有子节点的节点[id=1, name=param, type=Parameter]进行反向传播".to_string()
        ))
    );
}
