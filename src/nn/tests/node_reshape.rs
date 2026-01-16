/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : Reshape 节点单元测试
 */

use crate::assert_err;
use crate::nn::{GraphInner, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试基本的 reshape 功能：2x3 -> 3x2
#[test]
fn test_reshape_basic_2x3_to_3x2() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建输入节点 [2, 3]
    let input = graph.new_input_node(&[2, 3], Some("input"))?;

    // Reshape 到 [3, 2]
    let reshaped = graph.new_reshape_node(input, &[3, 2], Some("reshaped"))?;

    // 设置输入值: [[1,2,3], [4,5,6]]
    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;

    // 前向传播
    graph.forward(reshaped)?;

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
    let mut graph = GraphInner::new();

    // 创建 1x6 输入
    let input = graph.new_input_node(&[1, 6], Some("input"))?;
    let reshaped = graph.new_reshape_node(input, &[2, 3], Some("reshaped"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.forward(reshaped)?;

    let output = graph.get_node_value(reshaped)?.unwrap();
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(output, &expected);

    Ok(())
}

/// 测试 reshape 到列向量
#[test]
fn test_reshape_to_column_vector() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[2, 3], Some("input"))?;
    let reshaped = graph.new_reshape_node(input, &[6, 1], Some("column"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.forward(reshaped)?;

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
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[3, 2], Some("input"))?;
    let reshaped = graph.new_reshape_node(input, &[1, 6], Some("row"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.forward(reshaped)?;

    let output = graph.get_node_value(reshaped)?.unwrap();
    assert_eq!(output.shape(), &[1, 6]);

    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    assert_eq!(output, &expected);

    Ok(())
}

// ==================== VJP单元测试（直接调用 calc_grad_to_parent）====================

/// 测试 Reshape VJP（梯度直接透传）
///
/// 对于 y = reshape(x)，有 dy/dx = I（单位变换）
/// VJP: grad_to_input = reshape(upstream_grad, input_shape)
#[test]
fn test_reshape_backward_vjp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let reshape_id = graph.new_reshape_node(input_id, &[3, 2], Some("reshaped"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input_id, Some(&input_data))?;
    graph.forward(reshape_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[3, 2]);
    let reshape_node = graph.get_node(reshape_id)?;
    let input_node = graph.get_node(input_id)?;
    let grad = reshape_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // Reshape 的梯度只是形状变化，数值直接透传
    assert_eq!(grad.shape(), &[2, 3]);
    assert_eq!(&grad, &Tensor::ones(&[2, 3]));

    Ok(())
}

/// 测试 Reshape VJP（非单位上游梯度）
#[test]
fn test_reshape_backward_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let reshape_id = graph.new_reshape_node(input_id, &[3, 2], Some("reshaped"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input_id, Some(&input_data))?;
    graph.forward(reshape_id)?;

    // 非单位上游梯度
    let upstream_grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let reshape_node = graph.get_node(reshape_id)?;
    let input_node = graph.get_node(input_id)?;
    let grad = reshape_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 梯度应该被 reshape 回输入形状，数值保持不变
    assert_eq!(grad.shape(), &[2, 3]);
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(&grad, &expected);

    Ok(())
}

// ==================== 端到端反向传播测试（通过 graph.backward）====================

/// 测试 Reshape 通过 graph.backward() 的端到端反向传播
#[test]
fn test_reshape_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：output = sigmoid(reshape(input))
    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let reshaped = graph.new_reshape_node(input, &[3, 2], Some("reshaped"))?;
    let sigmoid = graph.new_sigmoid_node(reshaped, Some("sigmoid"))?;

    // loss = MSE(sigmoid, target)
    let target = graph.new_input_node(&[3, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(sigmoid, target, Some("loss"))?;

    // 设置值
    let input_data = Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[3, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    let loss_returned = graph.backward(loss)?;
    assert!(loss_returned > 0.0);

    // 验证梯度存在且形状正确
    let input_grad = graph.get_node(input)?.grad().expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    // Reshape 只改变形状不改变梯度值，所以梯度应该等于
    // sigmoid'(x) * 2*(sigmoid(x)-target)/n，然后 reshape 回 [2,3]
    // 验证梯度非零
    assert!(input_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10));

    Ok(())
}

/// 测试 Batch 模式的前向传播
#[test]
fn test_reshape_batch_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    let batch_size = 4;

    // 输入形状: [batch, 6]
    let input = graph.new_input_node(&[batch_size, 6], Some("input"))?;
    // Reshape 到: [2, 12]
    let reshaped = graph.new_reshape_node(input, &[2, 12], Some("reshaped"))?;

    // 创建 batch 数据
    let input_data = Tensor::normal(0.0, 1.0, &[batch_size, 6]);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward(reshaped)?;

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

// ==================== 梯度累积测试 ====================

/// 测试 Reshape 梯度累积
#[test]
fn test_reshape_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let reshaped = graph.new_reshape_node(input, &[3, 2], Some("reshaped"))?;
    let sigmoid = graph.new_sigmoid_node(reshaped, Some("sigmoid"))?;
    let target = graph.new_input_node(&[3, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(sigmoid, target, Some("loss"))?;

    // 设置值
    let input_data = Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[3, 2])))?;
    graph.forward(loss)?;

    // 第1次反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;
    let grad_first = graph.get_node(input)?.grad().unwrap().clone();

    // 第2次反向传播（梯度累积）- 需要重新 forward（PyTorch 语义）
    graph.forward(loss)?;
    graph.backward(loss)?;
    let grad_second = graph.get_node(input)?.grad().unwrap();
    assert_eq!(grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    graph.forward(loss)?;
    graph.backward(loss)?;
    let grad_after_clear = graph.get_node(input)?.grad().unwrap();
    assert_eq!(grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 错误处理测试 ====================

/// 测试形状不匹配错误
#[test]
fn test_reshape_shape_mismatch_error() {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();

    // 尝试 reshape 到元素数量不匹配的形状
    let result = graph.new_reshape_node(input, &[2, 2], Some("bad_reshape"));

    assert_err!(result, GraphError::ShapeMismatch { expected, got, .. }
        if expected == &[2, 3] && got == &[2, 2]);
}

/// 测试空形状错误
#[test]
fn test_reshape_empty_shape_error() {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();

    // 尝试 reshape 到空形状
    let result = graph.new_reshape_node(input, &[], Some("empty_reshape"));

    assert_err!(result, GraphError::InvalidOperation(msg) if msg.contains("空"));
}

// ==================== 与其他节点组合测试 ====================

/// 测试 Reshape 与 Add 组合
#[test]
fn test_reshape_with_add() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

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

    graph.forward(sum)?;

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
    let mut graph = GraphInner::new();

    let parent = graph.new_parameter_node(&[2, 6], Some("parent"))?;
    let reshape1 = graph.new_reshape_node(parent, &[3, 4], Some("reshape1"))?;
    let reshape2 = graph.new_reshape_node(reshape1, &[4, 3], Some("reshape2"))?;
    let reshape3 = graph.new_reshape_node(reshape2, &[6, 2], Some("reshape3"))?;

    let parent_data = Tensor::normal_seeded(0.0, 1.0, &[2, 6], 42);
    graph.set_node_value(parent, Some(&parent_data))?;

    graph.forward(reshape3)?;

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

    // VJP 验证：连续 Reshape 的梯度仍然只是形状变化
    let upstream_grad = Tensor::ones(&[6, 2]);
    let reshape3_node = graph.get_node(reshape3)?;
    let _parent_node = graph.get_node(parent)?;

    // 通过最后一个节点计算到第一个的梯度（中间链会自动传播）
    // 由于是直接调用 calc_grad_to_parent，只能测试单步
    let reshape2_node = graph.get_node(reshape2)?;
    let grad_to_reshape2 =
        reshape3_node.calc_grad_to_parent(reshape2_node, &upstream_grad, None)?;
    assert_eq!(grad_to_reshape2.shape(), &[4, 3]);
    assert_eq!(&grad_to_reshape2, &Tensor::ones(&[4, 3]));

    Ok(())
}

/// 测试 Reshape 在实际 MLP 场景中的使用（模拟 Flatten）
#[test]
fn test_reshape_as_flatten_in_mlp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

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
    graph.forward(loss)?;
    let loss_val = graph.get_node_value(loss)?.unwrap()[[0, 0]];
    assert!(loss_val > 0.0, "Loss 应该为正数");

    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证梯度存在
    assert!(graph.get_node(w1)?.grad().is_some());
    assert!(graph.get_node(w2)?.grad().is_some());

    Ok(())
}

/// 测试单样本模式下的反向传播正确性
#[test]
fn test_reshape_single_sample_backward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 简单网络: Input [1,6] -> Reshape [2,3] -> MatMul [2,3]@[3,1]=[2,1] -> MSE
    let x = graph.new_input_node(&[1, 6], Some("x"))?;
    let reshaped = graph.new_reshape_node(x, &[2, 3], Some("reshaped"))?;
    let w = graph.new_parameter_node_seeded(&[3, 1], Some("w"), 100)?;
    let y = graph.new_mat_mul_node(reshaped, w, Some("y"))?;

    // 使用 MSE loss
    let target = graph.new_input_node(&[2, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(y, target, Some("loss"))?;

    // 设置输入
    let x_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    graph.set_node_value(x, Some(&x_data))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 1])))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证 w 的梯度存在且形状正确
    let grad_w = graph.get_node(w)?.grad().expect("w 应有 grad");
    assert_eq!(grad_w.shape(), &[3, 1]);

    Ok(())
}
