use crate::assert_err;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_node_scalar_multiply_creation() {
    let mut graph = Graph::new();

    // 1. 测试正常创建：标量(1x1) * 矩阵(2x3)
    {
        let scalar = graph.new_parameter_node(&[1, 1], Some("scalar")).unwrap();
        let matrix = graph.new_input_node(&[2, 3], Some("matrix")).unwrap();
        let result = graph
            .new_scalar_multiply_node(scalar, matrix, Some("scalar_mul"))
            .unwrap();

        // 验证基本属性
        assert_eq!(graph.get_node_name(result).unwrap(), "scalar_mul");
        assert_eq!(graph.get_node_parents(result).unwrap().len(), 2);
        assert_eq!(graph.get_node_children(result).unwrap().len(), 0);

        // 验证输出形状与矩阵形状相同
        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[2, 3]
        );
    }

    // 2. 测试标量 * 向量(1xN)
    {
        let scalar = graph.new_parameter_node(&[1, 1], Some("scalar2")).unwrap();
        let vector = graph.new_input_node(&[1, 5], Some("vector")).unwrap();
        let result = graph
            .new_scalar_multiply_node(scalar, vector, Some("scalar_mul_vec"))
            .unwrap();

        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[1, 5]
        );
    }

    // 3. 测试标量 * 标量(1x1)
    {
        let scalar1 = graph
            .new_parameter_node(&[1, 1], Some("scalar_a"))
            .unwrap();
        let scalar2 = graph
            .new_parameter_node(&[1, 1], Some("scalar_b"))
            .unwrap();
        let result = graph
            .new_scalar_multiply_node(scalar1, scalar2, Some("scalar_mul_scalar"))
            .unwrap();

        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[1, 1]
        );
    }
}

#[test]
fn test_node_scalar_multiply_creation_with_invalid_shape() {
    let mut graph = Graph::new();

    // 1. 测试第1个参数不是标量（应该失败）
    let non_scalar = graph
        .new_parameter_node(&[2, 3], Some("non_scalar"))
        .unwrap();
    let matrix = graph.new_input_node(&[3, 4], Some("matrix")).unwrap();

    let result = graph.new_scalar_multiply_node(non_scalar, matrix, None);
    assert_err!(result, GraphError::ShapeMismatch([1, 1], [2, 3], "ScalarMultiply的第1个父节点必须是标量(形状为[1,1])"));

    // 2. 测试第1个参数是向量(1xN)而非标量（应该失败）
    let vector = graph.new_parameter_node(&[1, 3], Some("vector")).unwrap();
    let result = graph.new_scalar_multiply_node(vector, matrix, None);
    assert_err!(result, GraphError::ShapeMismatch([1, 1], [1, 3], "ScalarMultiply的第1个父节点必须是标量(形状为[1,1])"));
}

#[test]
fn test_node_scalar_multiply_name_generation() {
    let mut graph = Graph::new();

    let scalar = graph.new_parameter_node(&[1, 1], Some("s")).unwrap();
    let matrix = graph.new_input_node(&[2, 3], Some("m")).unwrap();

    // 1. 测试显式命名
    let result1 = graph
        .new_scalar_multiply_node(scalar, matrix, Some("my_scalar_mul"))
        .unwrap();
    assert_eq!(graph.get_node_name(result1).unwrap(), "my_scalar_mul");

    // 2. 测试自动命名
    let result2 = graph
        .new_scalar_multiply_node(scalar, matrix, None)
        .unwrap();
    assert_eq!(graph.get_node_name(result2).unwrap(), "scalar_multiply_1");

    // 3. 测试名称重复
    let result = graph.new_scalar_multiply_node(scalar, matrix, Some("my_scalar_mul"));
    assert_err!(result, GraphError::DuplicateNodeName("节点my_scalar_mul在图default_graph中重复"));
}

#[test]
fn test_node_scalar_multiply_manually_set_value() {
    let mut graph = Graph::new();
    let scalar = graph.new_parameter_node(&[1, 1], Some("s")).unwrap();
    let matrix = graph.new_input_node(&[2, 3], Some("m")).unwrap();
    let result = graph
        .new_scalar_multiply_node(scalar, matrix, Some("sm"))
        .unwrap();

    // 测试直接设置ScalarMultiply节点的值（应该失败）
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_err!(
        graph.set_node_value(result, Some(&test_value)),
        GraphError::InvalidOperation("节点[id=3, name=sm, type=ScalarMultiply]的值只能通过前向传播计算得到，不能直接设置")
    );
}

#[test]
fn test_node_scalar_multiply_forward_propagation() {
    // 测试数据（与Python测试tests/python/calc_jacobi_by_pytorch/node_scalar_multiply.py保持一致）
    let scalar_data = &[2.0];
    let matrix_data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expected_output = &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

    let mut graph = Graph::new();

    // 创建节点
    let scalar = graph.new_parameter_node(&[1, 1], Some("scalar")).unwrap();
    let matrix = graph.new_input_node(&[2, 3], Some("matrix")).unwrap();
    let result = graph
        .new_scalar_multiply_node(scalar, matrix, Some("result"))
        .unwrap();

    // 设置输入值
    let scalar_value = Tensor::new(scalar_data, &[1, 1]);
    let matrix_value = Tensor::new(matrix_data, &[2, 3]);
    graph.set_node_value(scalar, Some(&scalar_value)).unwrap();
    graph.set_node_value(matrix, Some(&matrix_value)).unwrap();

    // 前向传播
    graph.forward_node(result).unwrap();

    // 验证输出
    let output = graph.get_node_value(result).unwrap().unwrap();
    let expected = Tensor::new(expected_output, &[2, 3]);
    assert_eq!(output, &expected);
}

#[test]
fn test_node_scalar_multiply_backward_propagation() {
    // 测试数据（与Python测试tests/python/calc_jacobi_by_pytorch/node_scalar_multiply.py保持一致）
    let scalar_data = &[2.0];
    let matrix_data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // 对标量的雅可比矩阵：M.flatten().T → shape: [6, 1]
    let expected_jacobi_scalar = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // 对矩阵的雅可比矩阵：s * I_6 → shape: [6, 6]
    #[rustfmt::skip]
    let expected_jacobi_matrix = &[
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 2.0,
    ];

    let mut graph = Graph::new();

    // 创建节点
    let scalar = graph.new_parameter_node(&[1, 1], Some("scalar")).unwrap();
    let matrix = graph.new_parameter_node(&[2, 3], Some("matrix")).unwrap();
    let result = graph
        .new_scalar_multiply_node(scalar, matrix, Some("result"))
        .unwrap();

    // 设置输入值
    let scalar_value = Tensor::new(scalar_data, &[1, 1]);
    let matrix_value = Tensor::new(matrix_data, &[2, 3]);
    graph.set_node_value(scalar, Some(&scalar_value)).unwrap();
    graph.set_node_value(matrix, Some(&matrix_value)).unwrap();

    // 前向传播
    graph.forward_node(result).unwrap();

    // 测试对标量的反向传播（retain_graph=true 以便继续 backward）
    graph.backward_nodes_ex(&[scalar], result, true).unwrap();
    let scalar_jacobi = graph.get_node_jacobi(scalar).unwrap().unwrap();
    let expected_scalar_jacobi = Tensor::new(expected_jacobi_scalar, &[6, 1]);
    assert_eq!(scalar_jacobi, &expected_scalar_jacobi);

    // 测试对矩阵的反向传播
    graph.backward_nodes(&[matrix], result).unwrap();
    let matrix_jacobi = graph.get_node_jacobi(matrix).unwrap().unwrap();
    let expected_matrix_jacobi = Tensor::new(expected_jacobi_matrix, &[6, 6]);
    assert_eq!(matrix_jacobi, &expected_matrix_jacobi);
}

#[test]
fn test_node_scalar_multiply_gradient_accumulation() {
    let mut graph = Graph::new();

    // 创建节点
    let scalar = graph.new_parameter_node(&[1, 1], Some("scalar")).unwrap();
    let matrix = graph.new_parameter_node(&[2, 3], Some("matrix")).unwrap();
    let result = graph
        .new_scalar_multiply_node(scalar, matrix, Some("result"))
        .unwrap();

    // 设置输入值
    let scalar_value = Tensor::new(&[2.0], &[1, 1]);
    let matrix_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(scalar, Some(&scalar_value)).unwrap();
    graph.set_node_value(matrix, Some(&matrix_value)).unwrap();

    // 前向传播
    graph.forward_node(result).unwrap();

    // 第1次反向传播（retain_graph=true 以便多次 backward）
    graph.backward_nodes_ex(&[scalar], result, true).unwrap();
    let jacobi_first = graph.get_node_jacobi(scalar).unwrap().unwrap().clone();

    // 第2次反向传播（梯度应该累积）
    graph.backward_nodes_ex(&[scalar], result, true).unwrap();
    let jacobi_second = graph.get_node_jacobi(scalar).unwrap().unwrap();

    // 验证梯度累积
    assert_eq!(jacobi_second, &(&jacobi_first * 2.0));

    // 清除梯度后再次反向传播（最后一次可以不保留图）
    graph.clear_jacobi().unwrap();
    graph.backward_nodes(&[scalar], result).unwrap();
    let jacobi_after_clear = graph.get_node_jacobi(scalar).unwrap().unwrap();

    // 验证清除后梯度回到初始值
    assert_eq!(jacobi_after_clear, &jacobi_first);
}

/// 测试ScalarMultiply在更复杂计算图中的使用（模拟batch训练中的bias广播）
#[test]
fn test_node_scalar_multiply_in_batch_training_scenario() {
    let mut graph = Graph::new();

    // 模拟batch训练场景：
    // X: [batch_size, features] = [3, 2]
    // W: [features, 1] = [2, 1]
    // b: [1, 1] 标量偏置
    // ones: [batch_size, 1] = [3, 1] 全1向量
    // bias = ScalarMultiply(b, ones) → [3, 1]
    // output = MatMul(X, W) + bias → [3, 1]

    let batch_size = 3;
    let features = 2;

    // 创建节点
    let x = graph
        .new_input_node(&[batch_size, features], Some("X"))
        .unwrap();
    let w = graph.new_parameter_node(&[features, 1], Some("W")).unwrap();
    let b = graph.new_parameter_node(&[1, 1], Some("b")).unwrap();
    let ones = graph
        .new_input_node(&[batch_size, 1], Some("ones"))
        .unwrap();

    // 构建计算图
    let xw = graph.new_mat_mul_node(x, w, Some("xw")).unwrap();
    let bias = graph
        .new_scalar_multiply_node(b, ones, Some("bias"))
        .unwrap();
    let output = graph.new_add_node(&[xw, bias], Some("output")).unwrap();

    // 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[batch_size, features]);
    let w_value = Tensor::new(&[0.5, 0.5], &[features, 1]);
    let b_value = Tensor::new(&[1.0], &[1, 1]);
    let ones_value = Tensor::new(&[1.0, 1.0, 1.0], &[batch_size, 1]);

    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(w, Some(&w_value)).unwrap();
    graph.set_node_value(b, Some(&b_value)).unwrap();
    graph.set_node_value(ones, Some(&ones_value)).unwrap();

    // 前向传播
    graph.forward_node(output).unwrap();

    // 验证输出
    // xw = [[1*0.5+2*0.5], [3*0.5+4*0.5], [5*0.5+6*0.5]] = [[1.5], [3.5], [5.5]]
    // bias = 1 * [1,1,1]^T = [[1], [1], [1]]
    // output = xw + bias = [[2.5], [4.5], [6.5]]
    let output_value = graph.get_node_value(output).unwrap().unwrap();
    let expected_output = Tensor::new(&[2.5, 4.5, 6.5], &[batch_size, 1]);
    assert_eq!(output_value, &expected_output);

    // 反向传播：计算b的梯度
    graph.backward_nodes(&[b], output).unwrap();
    let b_jacobi = graph.get_node_jacobi(b).unwrap().unwrap();

    // b的雅可比应该是 [3, 1]（因为bias对b的雅可比是ones，add对bias的雅可比是单位矩阵）
    // 最终 b 的雅可比 = I_3 @ ones.T = [[1], [1], [1]]
    let expected_b_jacobi = Tensor::new(&[1.0, 1.0, 1.0], &[3, 1]);
    assert_eq!(b_jacobi, &expected_b_jacobi);
}

