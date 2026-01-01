/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : Flatten 节点单元测试
 */

use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 keep_first_dim=true（保留首维度）- 2D 张量保持不变
#[test]
fn test_flatten_keep_first_dim_2d() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    // 2D 输入 [3, 4]
    let input = graph.new_input_node(&[3, 4], Some("input"))?;
    let flat = graph.new_flatten_node(input, true, Some("flat"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 4]);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward_node(flat)?;

    let output = graph.get_node_value(flat)?.unwrap();
    // 2D 张量保持不变
    assert_eq!(output.shape(), &[3, 4]);
    assert_eq!(output, &input_data);

    Ok(())
}

/// 测试 keep_first_dim=false（完全展平为行向量）
#[test]
fn test_flatten_to_row_vector() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[2, 3], Some("input"))?;
    let flat = graph.new_flatten_node(input, false, Some("flat"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward_node(flat)?;

    let output = graph.get_node_value(flat)?.unwrap();
    // 完全展平为 [1, 6]
    assert_eq!(output.shape(), &[1, 6]);

    // 验证元素顺序
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    assert_eq!(output, &expected);

    Ok(())
}

/// 测试方形矩阵的展平
#[test]
fn test_flatten_square_matrix() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[4, 4], Some("input"))?;
    let flat = graph.new_flatten_node(input, false, Some("flat"))?;

    let input_data = Tensor::normal_seeded(0.0, 1.0, &[4, 4], 42);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward_node(flat)?;

    let output = graph.get_node_value(flat)?.unwrap();
    assert_eq!(output.shape(), &[1, 16]);

    // 验证元素一致
    for i in 0..16 {
        let in_row = i / 4;
        let in_col = i % 4;
        assert_abs_diff_eq!(input_data[[in_row, in_col]], output[[0, i]], epsilon = 1e-6);
    }

    Ok(())
}

// ==================== Jacobi 测试（单样本模式）====================

/// 测试 Flatten 的 Jacobi 是单位矩阵
#[test]
fn test_flatten_jacobi_is_identity() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    // 使用 Parameter 节点
    let parent = graph.new_parameter_node(&[2, 3], Some("parent"))?;
    let flat = graph.new_flatten_node(parent, false, Some("flat"))?;

    let parent_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(parent, Some(&parent_data))?;

    graph.forward_node(flat)?;
    graph.backward_nodes(&[parent], flat)?;

    let jacobi = graph.get_node_jacobi(parent)?.unwrap();
    assert_eq!(jacobi.shape(), &[6, 6]);

    // 验证是单位矩阵
    let expected = Tensor::eyes(6);
    assert_eq!(jacobi, &expected);

    Ok(())
}

/// 测试 Flatten 在链式网络中的 Jacobi
#[test]
fn test_flatten_jacobi_in_chain() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    // Parameter -> Flatten -> Sigmoid
    let parent = graph.new_parameter_node(&[2, 3], Some("parent"))?;
    let flat = graph.new_flatten_node(parent, false, Some("flat"))?;
    let output = graph.new_sigmoid_node(flat, Some("output"))?;

    let parent_data = Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    graph.set_node_value(parent, Some(&parent_data))?;

    graph.forward_node(output)?;
    // 使用 retain_graph=true 以便后续访问节点值
    graph.backward_nodes_ex(&[parent], output, true)?;

    let jacobi = graph.get_node_jacobi(parent)?.unwrap();
    assert_eq!(jacobi.shape(), &[6, 6]);

    // 计算预期的 Sigmoid 导数
    let sigmoid_out = graph.get_node_value(output)?.unwrap();
    let one_minus_sigmoid = Tensor::ones(sigmoid_out.shape()) - sigmoid_out;
    let sigmoid_deriv = sigmoid_out * &one_minus_sigmoid;

    // 验证对角线元素
    for i in 0..6 {
        let expected_val = sigmoid_deriv[[0, i]];
        assert_abs_diff_eq!(jacobi[[i, i]], expected_val, epsilon = 1e-6);
    }

    Ok(())
}

// ==================== Batch 模式测试 ====================

/// 测试 Batch 模式的前向传播
#[test]
fn test_flatten_batch_forward() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    // 输入 [batch=4, features=6]
    let input = graph.new_input_node(&[4, 6], Some("input"))?;
    // keep_first_dim=true: 对于 2D，形状不变
    let flat = graph.new_flatten_node(input, true, Some("flat"))?;

    let input_data = Tensor::normal_seeded(0.0, 1.0, &[4, 6], 42);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward_batch(flat)?;

    let output = graph.get_node_value(flat)?.unwrap();
    assert_eq!(output.shape(), &[4, 6]);
    assert_eq!(output, &input_data);

    Ok(())
}

/// 测试 Batch 模式的梯度传播
#[test]
fn test_flatten_batch_gradient() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    let batch_size = 3;
    let input_features = 6;
    let output_features = 2;

    // 网络: Input -> Flatten -> MatMul -> Loss
    let x = graph.new_input_node(&[batch_size, input_features], Some("x"))?;
    let flat = graph.new_flatten_node(x, true, Some("flat"))?; // 2D 不变
    let w = graph.new_parameter_node(&[input_features, output_features], Some("w"))?;
    let y = graph.new_mat_mul_node(flat, w, Some("y"))?;

    let labels = graph.new_input_node(&[batch_size, output_features], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(y, labels, Some("loss"))?;

    // 设置输入
    let x_data = Tensor::normal_seeded(0.0, 1.0, &[batch_size, input_features], 100);
    let mut labels_data = Tensor::zeros(&[batch_size, output_features]);
    for i in 0..batch_size {
        labels_data[[i, i % output_features]] = 1.0;
    }

    graph.set_node_value(x, Some(&x_data))?;
    graph.set_node_value(labels, Some(&labels_data))?;

    graph.forward_batch(loss)?;
    graph.backward_batch(loss, None)?;

    // 验证梯度存在且形状正确
    let grad_w = graph.get_node_grad_batch(w)?;
    assert!(grad_w.is_some());
    assert_eq!(grad_w.unwrap().shape(), &[input_features, output_features]);

    Ok(())
}

// ==================== 与其他节点组合测试 ====================

/// 测试 Flatten + MatMul（典型 CNN 到 FC 场景）
#[test]
fn test_flatten_with_matmul() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // 模拟 CNN 输出 [batch=2, features=8]
    let batch_size = 2;
    let cnn_features = 8;
    let hidden_size = 4;

    let x = graph.new_input_node(&[batch_size, cnn_features], Some("cnn_out"))?;
    let flat = graph.new_flatten_node(x, true, Some("flat"))?;
    let w = graph.new_parameter_node(&[cnn_features, hidden_size], Some("w"))?;
    let h = graph.new_mat_mul_node(flat, w, Some("hidden"))?;

    let x_data = Tensor::normal_seeded(0.0, 1.0, &[batch_size, cnn_features], 100);
    graph.set_node_value(x, Some(&x_data))?;

    graph.forward_node(h)?;

    let output = graph.get_node_value(h)?.unwrap();
    assert_eq!(output.shape(), &[batch_size, hidden_size]);

    Ok(())
}

/// 测试 Flatten -> Reshape 链
#[test]
fn test_flatten_reshape_chain() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[3, 4], Some("input"))?;
    // 先展平为行向量 [1, 12]
    let flat = graph.new_flatten_node(input, false, Some("flat"))?;
    // 再 reshape 为 [4, 3]
    let reshaped = graph.new_reshape_node(flat, &[4, 3], Some("reshaped"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 4]);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward_node(reshaped)?;

    let output = graph.get_node_value(reshaped)?.unwrap();
    assert_eq!(output.shape(), &[4, 3]);

    // 验证元素顺序不变
    for i in 0..12 {
        let in_row = i / 4;
        let in_col = i % 4;
        let out_row = i / 3;
        let out_col = i % 3;
        assert_abs_diff_eq!(input_data[[in_row, in_col]], output[[out_row, out_col]], epsilon = 1e-6);
    }

    Ok(())
}

/// 测试单样本反向传播
#[test]
fn test_flatten_single_sample_backward() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // Parameter -> Flatten -> MatMul -> Loss
    let x = graph.new_parameter_node(&[2, 3], Some("x"))?;
    let flat = graph.new_flatten_node(x, false, Some("flat"))?;  // [1, 6]
    let w = graph.new_parameter_node_seeded(&[6, 1], Some("w"), 100)?;
    let y = graph.new_mat_mul_node(flat, w, Some("y"))?;
    let loss = graph.new_perception_loss_node(y, Some("loss"))?;

    graph.forward_node(loss)?;
    graph.backward_nodes(&[w], loss)?;

    // 验证 w 的 Jacobi 存在
    let jacobi_w = graph.get_node_jacobi(w)?.unwrap();
    assert_eq!(jacobi_w.shape(), &[1, 6]); // loss [1,1], w [6,1]

    Ok(())
}

