use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_node_multiply_creation() {
    let mut graph = Graph::new();

    // 1. 测试正常创建：两个形状相同的矩阵
    {
        let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
        let right = graph.new_input_node(&[2, 3], Some("right")).unwrap();
        let result = graph.new_multiply_node(left, right, Some("mul")).unwrap();

        // 验证基本属性
        assert_eq!(graph.get_node_name(result).unwrap(), "mul");
        assert_eq!(graph.get_node_parents(result).unwrap().len(), 2);
        assert_eq!(graph.get_node_children(result).unwrap().len(), 0);

        // 验证输出形状与输入形状相同
        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[2, 3]
        );
    }

    // 2. 测试标量 * 标量(1x1)
    {
        let scalar1 = graph.new_parameter_node(&[1, 1], Some("scalar_a")).unwrap();
        let scalar2 = graph.new_parameter_node(&[1, 1], Some("scalar_b")).unwrap();
        let result = graph
            .new_multiply_node(scalar1, scalar2, Some("scalar_mul"))
            .unwrap();

        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[1, 1]
        );
    }
}

#[test]
fn test_node_multiply_creation_with_invalid_shape() {
    let mut graph = Graph::new();

    // 测试形状不匹配（应该失败）
    let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
    let right = graph.new_input_node(&[3, 4], Some("right")).unwrap();

    let result = graph.new_multiply_node(left, right, None);
    assert_eq!(
        result,
        Err(GraphError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![3, 4],
            message: "Multiply节点的两个父节点形状必须相同".to_string(),
        })
    );
}

#[test]
fn test_node_multiply_name_generation() {
    let mut graph = Graph::new();

    let left = graph.new_parameter_node(&[2, 3], Some("l")).unwrap();
    let right = graph.new_input_node(&[2, 3], Some("r")).unwrap();

    // 1. 测试显式命名
    let result1 = graph
        .new_multiply_node(left, right, Some("my_mul"))
        .unwrap();
    assert_eq!(graph.get_node_name(result1).unwrap(), "my_mul");

    // 2. 测试自动命名
    let result2 = graph.new_multiply_node(left, right, None).unwrap();
    assert_eq!(graph.get_node_name(result2).unwrap(), "multiply_1");

    // 3. 测试名称重复
    let result = graph.new_multiply_node(left, right, Some("my_mul"));
    assert_eq!(
        result,
        Err(GraphError::DuplicateNodeName(
            "节点my_mul在图default_graph中重复".to_string()
        ))
    );
}

#[test]
fn test_node_multiply_manually_set_value() {
    let mut graph = Graph::new();
    let left = graph.new_parameter_node(&[2, 3], Some("l")).unwrap();
    let right = graph.new_input_node(&[2, 3], Some("r")).unwrap();
    let result = graph.new_multiply_node(left, right, Some("mul")).unwrap();

    // 测试直接设置Multiply节点的值（应该失败）
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(
        graph.set_node_value(result, Some(&test_value)),
        Err(GraphError::InvalidOperation(
            "节点[id=3, name=mul, type=Multiply]的值只能通过前向传播计算得到，不能直接设置".into()
        ))
    );
}

#[test]
fn test_node_multiply_forward_propagation() {
    let mut graph = Graph::new();

    // 创建节点
    let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
    let right = graph.new_input_node(&[2, 3], Some("right")).unwrap();
    let result = graph
        .new_multiply_node(left, right, Some("result"))
        .unwrap();

    // 设置输入值
    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let right_value = Tensor::new(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 3]);
    graph.set_node_value(left, Some(&left_value)).unwrap();
    graph.set_node_value(right, Some(&right_value)).unwrap();

    // 前向传播
    graph.forward_node(result).unwrap();

    // 验证输出：逐元素乘法
    let output = graph.get_node_value(result).unwrap().unwrap();
    let expected = Tensor::new(&[2.0, 6.0, 12.0, 20.0, 30.0, 42.0], &[2, 3]);
    assert_eq!(output, &expected);
}

#[test]
fn test_node_multiply_backward_propagation() {
    let mut graph = Graph::new();

    // 创建节点
    let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
    let right = graph.new_parameter_node(&[2, 3], Some("right")).unwrap();
    let result = graph
        .new_multiply_node(left, right, Some("result"))
        .unwrap();

    // 设置输入值
    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let right_value = Tensor::new(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 3]);
    graph.set_node_value(left, Some(&left_value)).unwrap();
    graph.set_node_value(right, Some(&right_value)).unwrap();

    // 前向传播
    graph.forward_node(result).unwrap();

    // 测试对left的反向传播：∂C/∂A = diag(B.flatten())（retain_graph 以便继续 backward）
    graph.backward_nodes_ex(&[left], result, true).unwrap();
    let left_jacobi = graph.get_node_jacobi(left).unwrap().unwrap();
    // 对角矩阵，对角元素是right的值
    let expected_left_jacobi = right_value.flatten().diag();
    assert_eq!(left_jacobi, &expected_left_jacobi);

    // 测试对right的反向传播：∂C/∂B = diag(A.flatten())
    graph.backward_nodes(&[right], result).unwrap();
    let right_jacobi = graph.get_node_jacobi(right).unwrap().unwrap();
    // 对角矩阵，对角元素是left的值
    let expected_right_jacobi = left_value.flatten().diag();
    assert_eq!(right_jacobi, &expected_right_jacobi);
}

#[test]
fn test_node_multiply_gradient_accumulation() {
    let mut graph = Graph::new();

    // 创建节点
    let left = graph.new_parameter_node(&[2, 2], Some("left")).unwrap();
    let right = graph.new_parameter_node(&[2, 2], Some("right")).unwrap();
    let result = graph
        .new_multiply_node(left, right, Some("result"))
        .unwrap();

    // 设置输入值
    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    graph.set_node_value(left, Some(&left_value)).unwrap();
    graph.set_node_value(right, Some(&right_value)).unwrap();

    // 前向传播
    graph.forward_node(result).unwrap();

    // 第1次反向传播（retain_graph=true 以便多次 backward）
    graph.backward_nodes_ex(&[left], result, true).unwrap();
    let jacobi_first = graph.get_node_jacobi(left).unwrap().unwrap().clone();

    // 第2次反向传播（梯度应该累积）
    graph.backward_nodes_ex(&[left], result, true).unwrap();
    let jacobi_second = graph.get_node_jacobi(left).unwrap().unwrap();

    // 验证梯度累积
    assert_eq!(jacobi_second, &(&jacobi_first * 2.0));

    // 清除梯度后再次反向传播（最后一次可以不保留图）
    graph.clear_jacobi().unwrap();
    graph.backward_nodes(&[left], result).unwrap();
    let jacobi_after_clear = graph.get_node_jacobi(left).unwrap().unwrap();

    // 验证清除后梯度回到初始值
    assert_eq!(jacobi_after_clear, &jacobi_first);
}
