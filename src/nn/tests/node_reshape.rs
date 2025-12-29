/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : Reshape 节点单元测试
 */

use crate::assert_err;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试基本的 reshape 功能：2x3 -> 3x2
#[test]
fn test_reshape_basic_2x3_to_3x2() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    // 创建输入节点 [2, 3]
    let input = graph.new_input_node(&[2, 3], Some("input"))?;

    // Reshape 到 [3, 2]
    let reshaped = graph.new_reshape_node(input, &[3, 2], Some("reshaped"))?;

    // 设置输入值: [[1,2,3], [4,5,6]]
    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;

    // 前向传播
    graph.forward_node(reshaped)?;

    // 验证输出形状
    let output = graph.get_node_value(reshaped)?.unwrap();
    assert_eq!(output.shape(), &[3, 2]);

    // 验证输出值（按行优先顺序重排）
    // [1,2,3; 4,5,6] reshape 为 [1,2; 3,4; 5,6]
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(output, &expected);

    Ok(())
}

/// 测试 reshape 保持元素顺序：行优先
#[test]
fn test_reshape_preserves_row_major_order() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    // 创建 1x6 输入
    let input = graph.new_input_node(&[1, 6], Some("input"))?;
    let reshaped = graph.new_reshape_node(input, &[2, 3], Some("reshaped"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.forward_node(reshaped)?;

    let output = graph.get_node_value(reshaped)?.unwrap();
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(output, &expected);

    Ok(())
}

/// 测试 reshape 到列向量
#[test]
fn test_reshape_to_column_vector() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[2, 3], Some("input"))?;
    let reshaped = graph.new_reshape_node(input, &[6, 1], Some("column"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.forward_node(reshaped)?;

    let output = graph.get_node_value(reshaped)?.unwrap();
    assert_eq!(output.shape(), &[6, 1]);

    // 验证展平后的元素顺序
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6, 1]);
    assert_eq!(output, &expected);

    Ok(())
}

/// 测试 reshape 到行向量
#[test]
fn test_reshape_to_row_vector() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[3, 2], Some("input"))?;
    let reshaped = graph.new_reshape_node(input, &[1, 6], Some("row"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.forward_node(reshaped)?;

    let output = graph.get_node_value(reshaped)?.unwrap();
    assert_eq!(output.shape(), &[1, 6]);

    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    assert_eq!(output, &expected);

    Ok(())
}

// ==================== Jacobi 测试（单样本模式）====================

/// 测试 Reshape 的 Jacobi 矩阵是单位矩阵
#[test]
fn test_reshape_jacobi_is_identity() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    // 使用 Parameter 节点，因为 Input 节点不能有 Jacobi
    let parent = graph.new_parameter_node(&[2, 3], Some("parent"))?;
    let reshaped = graph.new_reshape_node(parent, &[3, 2], Some("reshaped"))?;

    let parent_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(parent, Some(&parent_data))?;

    // 前向传播
    graph.forward_node(reshaped)?;

    // 反向传播
    graph.backward_nodes(&[parent], reshaped)?;

    // 获取 Jacobi
    let jacobi = graph.get_node_jacobi(parent)?.unwrap();

    // 验证是单位矩阵 6x6
    assert_eq!(jacobi.shape(), &[6, 6]);
    let expected = Tensor::eyes(6);
    assert_eq!(jacobi, &expected);

    Ok(())
}

/// 测试复杂图中的 Jacobi：Parameter -> Reshape -> Sigmoid
#[test]
fn test_reshape_jacobi_in_chain() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    // 使用 Parameter 节点，因为 Input 节点不能有 Jacobi
    let parent = graph.new_parameter_node(&[2, 3], Some("parent"))?;
    let reshaped = graph.new_reshape_node(parent, &[3, 2], Some("reshaped"))?;
    let output = graph.new_sigmoid_node(reshaped, Some("output"))?;

    let parent_data = Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    graph.set_node_value(parent, Some(&parent_data))?;

    graph.forward_node(output)?;
    // 使用 retain_graph=true 以便后续访问节点值
    graph.backward_nodes_ex(&[parent], output, true)?;

    // Jacobi 应该是 Sigmoid 导数的对角矩阵（因为 Reshape 的 Jacobi 是单位矩阵）
    let jacobi = graph.get_node_jacobi(parent)?.unwrap();
    assert_eq!(jacobi.shape(), &[6, 6]);

    // 计算预期的 Sigmoid 导数
    let sigmoid_out = graph.get_node_value(output)?.unwrap();
    let one_minus_sigmoid = Tensor::ones(sigmoid_out.shape()) - sigmoid_out;
    let sigmoid_deriv = sigmoid_out * &one_minus_sigmoid;

    // 验证对角线元素
    for i in 0..6 {
        let row = i / 2;
        let col = i % 2;
        let expected_val = sigmoid_deriv[[row, col]];
        assert_abs_diff_eq!(jacobi[[i, i]], expected_val, epsilon = 1e-6);
    }

    Ok(())
}

// ==================== Batch 模式测试 ====================

/// 测试 Batch 模式的前向传播
#[test]
fn test_reshape_batch_forward() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let batch_size = 4;

    // 输入形状: [batch, 6]
    let input = graph.new_input_node(&[batch_size, 6], Some("input"))?;
    // Reshape 到: [2, 12]（演示非 batch 维度的 reshape）
    let reshaped = graph.new_reshape_node(input, &[2, 12], Some("reshaped"))?;

    // 创建 batch 数据
    let input_data = Tensor::normal(0.0, 1.0, &[batch_size, 6]);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward_batch(reshaped)?;

    let output = graph.get_node_value(reshaped)?.unwrap();
    assert_eq!(output.shape(), &[2, 12]);

    // 验证元素总数和顺序不变
    for i in 0..24 {
        let in_row = i / 6;
        let in_col = i % 6;
        let out_row = i / 12;
        let out_col = i % 12;
        assert_abs_diff_eq!(
            input_data[[in_row, in_col]],
            output[[out_row, out_col]],
            epsilon = 1e-6
        );
    }

    Ok(())
}

/// 测试 Batch 模式的梯度传播
#[test]
fn test_reshape_batch_gradient() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 3;
    let input_features = 6;
    let output_features = 2;

    // 网络: Input [3,6] -> Reshape [6,3] -> MatMul [6,3]@[3,2]=[6,2] -> Loss
    let x = graph.new_input_node(&[batch_size, input_features], Some("x"))?;
    let reshaped = graph.new_reshape_node(x, &[input_features, batch_size], Some("reshaped"))?;

    let w = graph.new_parameter_node(&[batch_size, output_features], Some("w"))?;
    let y = graph.new_mat_mul_node(reshaped, w, Some("y"))?;

    // 标签
    let labels = graph.new_input_node(&[input_features, output_features], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(y, labels, Some("loss"))?;

    // 设置输入
    let x_data = Tensor::normal_seeded(0.0, 1.0, &[batch_size, input_features], 100);
    let mut labels_data = Tensor::zeros(&[input_features, output_features]);
    // 设置 one-hot 标签
    for i in 0..input_features {
        labels_data[[i, i % output_features]] = 1.0;
    }

    graph.set_node_value(x, Some(&x_data))?;
    graph.set_node_value(labels, Some(&labels_data))?;

    // Batch forward & backward
    graph.forward_batch(loss)?;
    graph.backward_batch(loss)?;

    // 验证梯度存在且形状正确
    let grad_w = graph.get_node_grad_batch(w)?;
    assert!(grad_w.is_some());
    assert_eq!(grad_w.unwrap().shape(), &[batch_size, output_features]);

    Ok(())
}

// ==================== 错误处理测试 ====================

/// 测试形状不匹配错误
#[test]
fn test_reshape_shape_mismatch_error() {
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();

    // 尝试 reshape 到元素数量不匹配的形状
    let result = graph.new_reshape_node(input, &[2, 2], Some("bad_reshape"));

    assert_err!(result, GraphError::ShapeMismatch { expected, got, .. }
        if expected == &[2, 3] && got == &[2, 2]);
}

/// 测试空形状错误
#[test]
fn test_reshape_empty_shape_error() {
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();

    // 尝试 reshape 到空形状
    let result = graph.new_reshape_node(input, &[], Some("empty_reshape"));

    assert_err!(result, GraphError::InvalidOperation(msg) if msg.contains("空"));
}

// ==================== 与其他节点组合测试 ====================

/// 测试 Reshape 与 Add 组合
#[test]
fn test_reshape_with_add() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    // 两个输入，形状不同但 reshape 后相同
    let a = graph.new_input_node(&[2, 3], Some("a"))?;
    let b = graph.new_input_node(&[3, 2], Some("b"))?;

    // 将 a reshape 为 [3, 2]
    let a_reshaped = graph.new_reshape_node(a, &[3, 2], Some("a_reshaped"))?;

    // 相加
    let sum = graph.new_add_node(&[a_reshaped, b], Some("sum"))?;

    // 设置值
    // a: [[1,2,3], [4,5,6]] reshape 后: [[1,2], [3,4], [5,6]]
    let a_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // b: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    let b_data = Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]);
    graph.set_node_value(a, Some(&a_data))?;
    graph.set_node_value(b, Some(&b_data))?;

    graph.forward_node(sum)?;

    let output = graph.get_node_value(sum)?.unwrap();
    // a reshape 后: [[1,2], [3,4], [5,6]]
    // 加上 b: [[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]
    let expected = Tensor::new(&[1.1, 2.2, 3.3, 4.4, 5.5, 6.6], &[3, 2]);
    assert_abs_diff_eq!(output, &expected, epsilon = 1e-6);

    Ok(())
}

/// 测试多次 Reshape（连续变换）
#[test]
fn test_reshape_chain() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    // 使用 Parameter 节点进行 Jacobi 测试
    let parent = graph.new_parameter_node(&[2, 6], Some("parent"))?;
    let reshape1 = graph.new_reshape_node(parent, &[3, 4], Some("reshape1"))?;
    let reshape2 = graph.new_reshape_node(reshape1, &[4, 3], Some("reshape2"))?;
    let reshape3 = graph.new_reshape_node(reshape2, &[6, 2], Some("reshape3"))?;

    let parent_data = Tensor::normal_seeded(0.0, 1.0, &[2, 6], 42);
    graph.set_node_value(parent, Some(&parent_data))?;

    graph.forward_node(reshape3)?;

    // 验证最终形状
    let output = graph.get_node_value(reshape3)?.unwrap();
    assert_eq!(output.shape(), &[6, 2]);

    // 验证数据保持不变（只是形状改变）
    for i in 0..12 {
        let in_row = i / 6;
        let in_col = i % 6;
        let out_row = i / 2;
        let out_col = i % 2;
        assert_abs_diff_eq!(
            parent_data[[in_row, in_col]],
            output[[out_row, out_col]],
            epsilon = 1e-6
        );
    }

    // 反向传播验证
    graph.backward_nodes(&[parent], reshape3)?;
    let jacobi = graph.get_node_jacobi(parent)?.unwrap();

    // 连续 Reshape 的 Jacobi 仍然是单位矩阵
    assert_eq!(jacobi.shape(), &[12, 12]);
    let expected = Tensor::eyes(12);
    assert_eq!(jacobi, &expected);

    Ok(())
}

/// 测试 Reshape 在实际 MLP 场景中的使用（模拟 Flatten）
#[test]
fn test_reshape_as_flatten_in_mlp() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // 模拟 CNN 输出: [batch=2, features=8]
    let batch_size = 2;
    let cnn_features = 8;
    let hidden_size = 4;
    let num_classes = 3;

    // 输入（假设是 CNN 输出已经展平）
    let x = graph.new_input_node(&[batch_size, cnn_features], Some("cnn_out"))?;

    // 全连接层
    let w1 = graph.new_parameter_node(&[cnn_features, hidden_size], Some("w1"))?;
    let h = graph.new_mat_mul_node(x, w1, Some("hidden"))?;
    let h_act = graph.new_sigmoid_node(h, Some("h_act"))?;

    let w2 = graph.new_parameter_node(&[hidden_size, num_classes], Some("w2"))?;
    let logits = graph.new_mat_mul_node(h_act, w2, Some("logits"))?;

    // 标签
    let labels = graph.new_input_node(&[batch_size, num_classes], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(logits, labels, Some("loss"))?;

    // 设置数据
    let x_data = Tensor::normal_seeded(0.0, 1.0, &[batch_size, cnn_features], 100);
    let mut labels_data = Tensor::zeros(&[batch_size, num_classes]);
    labels_data[[0, 0]] = 1.0;
    labels_data[[1, 2]] = 1.0;

    graph.set_node_value(x, Some(&x_data))?;
    graph.set_node_value(labels, Some(&labels_data))?;

    // 训练一步
    graph.forward_batch(loss)?;
    let loss_val = graph.get_node_value(loss)?.unwrap()[[0, 0]];
    assert!(loss_val > 0.0, "Loss 应该为正数");

    graph.backward_batch(loss)?;

    // 验证梯度存在
    assert!(graph.get_node_grad_batch(w1)?.is_some());
    assert!(graph.get_node_grad_batch(w2)?.is_some());

    Ok(())
}

/// 测试单样本模式下的反向传播正确性
#[test]
fn test_reshape_single_sample_backward() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // 简单网络: Input [1,6] -> Reshape [2,3] -> MatMul [2,3]@[3,1]=[2,1] -> Loss
    let x = graph.new_input_node(&[1, 6], Some("x"))?;
    let reshaped = graph.new_reshape_node(x, &[2, 3], Some("reshaped"))?;
    let w = graph.new_parameter_node_seeded(&[3, 1], Some("w"), 100)?;
    let y = graph.new_mat_mul_node(reshaped, w, Some("y"))?;
    let loss = graph.new_perception_loss_node(y, Some("loss"))?;

    // 设置输入
    let x_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    graph.set_node_value(x, Some(&x_data))?;

    // 前向传播
    graph.forward_node(loss)?;

    // 反向传播
    graph.backward_nodes(&[w], loss)?;

    // 验证 w 的 Jacobi 存在且形状正确
    let jacobi_w = graph.get_node_jacobi(w)?.unwrap();
    assert_eq!(jacobi_w.shape(), &[2, 3]); // loss shape [2,1], w shape [3,1]

    Ok(())
}
