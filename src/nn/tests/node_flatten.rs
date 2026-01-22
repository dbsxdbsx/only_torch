/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : Flatten 节点单元测试
 */

use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 keep_first_dim=true（保留首维度）- 2D 张量保持不变
#[test]
fn test_flatten_keep_first_dim_2d() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 2D 输入 [3, 4]
    let input = graph.new_input_node(&[3, 4], Some("input"))?;
    let flat = graph.new_flatten_node(input, true, Some("flat"))?;

    let input_data = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward(flat)?;

    let output = graph.get_node_value(flat)?.unwrap();
    // 2D 张量保持不变
    assert_eq!(output.shape(), &[3, 4]);
    assert_eq!(output, &input_data);

    Ok(())
}

/// 测试 keep_first_dim=false（完全展平为行向量）
#[test]
fn test_flatten_to_row_vector() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[2, 3], Some("input"))?;
    let flat = graph.new_flatten_node(input, false, Some("flat"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward(flat)?;

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
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[4, 4], Some("input"))?;
    let flat = graph.new_flatten_node(input, false, Some("flat"))?;

    let input_data = Tensor::normal_seeded(0.0, 1.0, &[4, 4], 42);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward(flat)?;

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

// ==================== VJP单元测试（直接调用 calc_grad_to_parent）====================

/// 测试 Flatten VJP（梯度直接透传并 reshape）
///
/// 对于 y = flatten(x)，梯度只是形状变化
/// VJP: grad_to_input = reshape(upstream_grad, input_shape)
#[test]
fn test_flatten_backward_vjp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let flat_id = graph.new_flatten_node(input_id, false, Some("flat"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input_id, Some(&input_data))?;
    graph.forward(flat_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[1, 6]);
    let flat_node = graph.get_node(flat_id)?;
    let input_node = graph.get_node(input_id)?;
    let grad = flat_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // Flatten 的梯度只是形状变化，数值直接透传
    assert_eq!(grad.shape(), &[2, 3]);
    assert_eq!(&grad, &Tensor::ones(&[2, 3]));

    Ok(())
}

/// 测试 Flatten VJP（非单位上游梯度）
#[test]
fn test_flatten_backward_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let flat_id = graph.new_flatten_node(input_id, false, Some("flat"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input_id, Some(&input_data))?;
    graph.forward(flat_id)?;

    // 非单位上游梯度
    let upstream_grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    let flat_node = graph.get_node(flat_id)?;
    let input_node = graph.get_node(input_id)?;
    let grad = flat_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 梯度应该被 reshape 回输入形状，数值保持不变
    assert_eq!(grad.shape(), &[2, 3]);
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(&grad, &expected);

    Ok(())
}

// ==================== 端到端反向传播测试（通过 graph.backward）====================

/// 测试 Flatten 通过 graph.backward() 的端到端反向传播
#[test]
fn test_flatten_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：output = sigmoid(flatten(input))
    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let flat = graph.new_flatten_node(input, false, Some("flat"))?;
    let sigmoid = graph.new_sigmoid_node(flat, Some("sigmoid"))?;

    // loss = MSE(sigmoid, target)
    let target = graph.new_input_node(&[1, 6], Some("target"))?;
    let loss = graph.new_mse_loss_node(sigmoid, target, Some("loss"))?;

    // 设置值
    let input_data = Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[1, 6])))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    let loss_returned = graph.backward(loss)?;
    assert!(loss_returned > 0.0);

    // 验证梯度存在且形状正确
    let input_grad = graph.get_node(input)?.grad().expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    // 验证梯度非零
    assert!(input_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10));

    Ok(())
}

/// 测试 Batch 模式的前向传播
#[test]
fn test_flatten_batch_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入 [batch=4, features=6]
    let input = graph.new_input_node(&[4, 6], Some("input"))?;
    // keep_first_dim=true: 对于 2D，形状不变
    let flat = graph.new_flatten_node(input, true, Some("flat"))?;

    let input_data = Tensor::normal_seeded(0.0, 1.0, &[4, 6], 42);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward(flat)?;

    let output = graph.get_node_value(flat)?.unwrap();
    assert_eq!(output.shape(), &[4, 6]);
    assert_eq!(output, &input_data);

    Ok(())
}

// ==================== 梯度累积测试 ====================

/// 测试 Flatten 梯度累积
#[test]
fn test_flatten_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let flat = graph.new_flatten_node(input, false, Some("flat"))?;
    let sigmoid = graph.new_sigmoid_node(flat, Some("sigmoid"))?;
    let target = graph.new_input_node(&[1, 6], Some("target"))?;
    let loss = graph.new_mse_loss_node(sigmoid, target, Some("loss"))?;

    // 设置值
    let input_data = Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[1, 6])))?;
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

// ==================== 与其他节点组合测试 ====================

/// 测试 Flatten + MatMul（典型 CNN 到 FC 场景）
#[test]
fn test_flatten_with_matmul() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

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

    graph.forward(h)?;

    let output = graph.get_node_value(h)?.unwrap();
    assert_eq!(output.shape(), &[batch_size, hidden_size]);

    Ok(())
}

/// 测试 Flatten -> Reshape 链
#[test]
fn test_flatten_reshape_chain() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[3, 4], Some("input"))?;
    // 先展平为行向量 [1, 12]
    let flat = graph.new_flatten_node(input, false, Some("flat"))?;
    // 再 reshape 为 [4, 3]
    let reshaped = graph.new_reshape_node(flat, &[4, 3], Some("reshaped"))?;

    let input_data = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward(reshaped)?;

    let output = graph.get_node_value(reshaped)?.unwrap();
    assert_eq!(output.shape(), &[4, 3]);

    // 验证元素顺序不变
    for i in 0..12 {
        let in_row = i / 4;
        let in_col = i % 4;
        let out_row = i / 3;
        let out_col = i % 3;
        assert_abs_diff_eq!(
            input_data[[in_row, in_col]],
            output[[out_row, out_col]],
            epsilon = 1e-6
        );
    }

    Ok(())
}

/// 测试单样本反向传播
#[test]
fn test_flatten_single_sample_backward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // Parameter -> Flatten -> MatMul -> MSE
    let x = graph.new_parameter_node(&[2, 3], Some("x"))?;
    let flat = graph.new_flatten_node(x, false, Some("flat"))?; // [1, 6]
    let w = graph.new_parameter_node_seeded(&[6, 1], Some("w"), 100)?;
    let y = graph.new_mat_mul_node(flat, w, Some("y"))?;

    // 使用 MSE loss
    let target = graph.new_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(y, target, Some("loss"))?;

    graph.set_node_value(target, Some(&Tensor::zeros(&[1, 1])))?;
    graph.forward(loss)?;
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证 w 的梯度存在且形状正确
    let grad_w = graph.get_node(w)?.grad().expect("w 应有 grad");
    assert_eq!(grad_w.shape(), &[6, 1]);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 Flatten 节点的动态形状传播
///
/// Flatten 在 keep_first_dim=true 时支持动态 batch：
/// - 输入: [batch, c, h, w] 或 [batch, features]
/// - 输出: [batch, c*h*w] 或 [batch, features]
#[test]
fn test_flatten_dynamic_shape_propagation() {
    use crate::nn::var_ops::VarShapeOps;
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建 4D 输入：[batch, channels, height, width]
    // Input 节点默认支持动态 batch
    let x = graph.input(&Tensor::zeros(&[2, 3, 4, 4])).unwrap();

    // Flatten (keep_first_dim=true by default): [batch, 3, 4, 4] -> [batch, 48]
    let flat = x.flatten().unwrap();

    // 验证动态形状传播
    let dyn_shape = flat.dynamic_expected_shape();
    assert!(
        dyn_shape.is_dynamic(0),
        "batch 维度应该是动态的（因为输入是动态的且 keep_first_dim=true）"
    );
    assert!(!dyn_shape.is_dynamic(1), "第二维应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(48), "第二维应该是 3*4*4=48");
}

/// 测试 Flatten 在不同 batch_size 下的前向计算
#[test]
fn test_flatten_dynamic_batch_forward() {
    use crate::nn::var_ops::VarShapeOps;
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建 4D 输入：[batch, channels, height, width]
    let x = graph.input(&Tensor::zeros(&[2, 3, 4, 4])).unwrap();

    // Flatten (keep_first_dim=true by default)
    let flat = x.flatten().unwrap();

    // 第一次 forward：batch=2
    flat.forward().unwrap();
    let value1 = flat.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 48], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[5, 3, 4, 4])).unwrap();

    // 第二次 forward：batch=5
    flat.forward().unwrap();
    let value2 = flat.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[5, 48], "第二次 forward: batch=5");
}

/// 测试 Flatten 在不同 batch_size 下的反向传播
#[test]
fn test_flatten_dynamic_batch_backward() {
    use crate::nn::var_ops::{VarLossOps, VarShapeOps};
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建 4D 输入
    let x = graph
        .input(&Tensor::normal_seeded(0.0, 1.0, &[2, 3, 4, 4], 42))
        .unwrap();

    // Flatten (keep_first_dim=true by default) -> MSE
    let flat = x.flatten().unwrap();
    let target = graph.input(&Tensor::zeros(&[2, 48])).unwrap();
    let loss = flat.mse_loss(&target).unwrap();

    // 第一次训练：batch=2
    loss.forward().unwrap();
    let loss_val1 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);
    graph.zero_grad();
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::normal_seeded(0.0, 1.0, &[5, 3, 4, 4], 100))
        .unwrap();
    target.set_value(&Tensor::zeros(&[5, 48])).unwrap();

    // 第二次训练：batch=5
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
    graph.zero_grad();
    loss.backward().unwrap();
}
