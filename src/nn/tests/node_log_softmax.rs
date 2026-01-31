/*
 * @Author       : 老董
 * @Description  : LogSoftmax 节点单元测试
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）
 * 2. 前向传播测试
 * 3. VJP 单元测试（直接调用 calc_grad_to_parent）
 * 4. 端到端反向传播测试（通过 graph.backward）
 * 5. 与 Softmax 的一致性测试
 */

use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 LogSoftmax 节点创建
#[test]
fn test_log_softmax_creation() {
    let mut graph = GraphInner::new();

    // 1. Input 节点作为父节点
    {
        let input = graph.new_basic_input_node(&[2, 3], Some("input1")).unwrap();
        let log_softmax = graph
            .new_log_softmax_node(input, Some("log_softmax_with_input"))
            .unwrap();

        assert_eq!(
            graph.get_node_name(log_softmax).unwrap(),
            "log_softmax_with_input"
        );
        assert_eq!(graph.get_node_parents(log_softmax).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(log_softmax).unwrap().len(), 0);
        assert_eq!(
            graph.get_node_value_expected_shape(log_softmax).unwrap(),
            &[2, 3]
        );
    }

    // 2. Parameter 节点作为父节点
    {
        let param = graph.new_parameter_node(&[4, 5], Some("param1")).unwrap();
        let log_softmax = graph
            .new_log_softmax_node(param, Some("log_softmax_with_param"))
            .unwrap();

        assert_eq!(
            graph.get_node_name(log_softmax).unwrap(),
            "log_softmax_with_param"
        );
        assert_eq!(graph.get_node_parents(log_softmax).unwrap().len(), 1);
        assert_eq!(
            graph.get_node_value_expected_shape(log_softmax).unwrap(),
            &[4, 5]
        );
    }
}

/// 测试 LogSoftmax 需要 2D 输入
#[test]
fn test_log_softmax_requires_2d() {
    let mut graph = GraphInner::new();

    // 3D 输入应失败（LogSoftmax 只接受 2D）
    let input_3d = graph
        .new_basic_input_node(&[2, 3, 4], Some("input_3d"))
        .unwrap();
    let result = graph.new_log_softmax_node(input_3d, Some("log_softmax"));
    assert_err!(
        result,
        GraphError::InvalidOperation(
            "LogSoftmax 节点需要 2D 输入 [batch, num_classes]，但得到 [2, 3, 4]"
        )
    );
}

/// 测试 LogSoftmax 节点命名
#[test]
fn test_log_softmax_name_generation() {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();

    // 1. 显式命名
    let ls1 = graph
        .new_log_softmax_node(input, Some("my_log_softmax"))
        .unwrap();
    assert_eq!(graph.get_node_name(ls1).unwrap(), "my_log_softmax");

    // 2. 自动命名
    let ls2 = graph.new_log_softmax_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(ls2).unwrap(), "log_softmax_1");

    // 3. 名称重复
    let result = graph.new_log_softmax_node(input, Some("my_log_softmax"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_log_softmax在图default_graph中重复")
    );
}

/// 测试 LogSoftmax 节点不能直接设置值
#[test]
fn test_log_softmax_cannot_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();
    let log_softmax = graph
        .new_log_softmax_node(input, Some("log_softmax"))
        .unwrap();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_err!(
        graph.set_node_value(log_softmax, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=log_softmax, type=LogSoftmax]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试 LogSoftmax 前向传播
#[test]
fn test_log_softmax_forward() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input")).unwrap();
    let log_softmax = graph
        .new_log_softmax_node(input, Some("log_softmax"))
        .unwrap();

    // 测试数据
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(log_softmax).unwrap();

    let output = graph.get_node_value(log_softmax).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 3]);

    // log_softmax 输出应该都是负数
    for i in 0..2 {
        for j in 0..3 {
            assert!(output[[i, j]] < 0.0, "log_softmax 输出应为负数");
        }
    }

    // exp(log_softmax) 每行和应为 1
    for i in 0..2 {
        let sum = output[[i, 0]].exp() + output[[i, 1]].exp() + output[[i, 2]].exp();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }
}

/// 测试 LogSoftmax 与 Softmax 的一致性
#[test]
fn test_log_softmax_vs_softmax() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input")).unwrap();
    let log_softmax = graph
        .new_log_softmax_node(input, Some("log_softmax"))
        .unwrap();
    let softmax = graph.new_softmax_node(input, Some("softmax")).unwrap();

    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(log_softmax).unwrap();
    graph.forward(softmax).unwrap();

    let log_probs = graph.get_node_value(log_softmax).unwrap().unwrap();
    let probs = graph.get_node_value(softmax).unwrap().unwrap();

    // exp(log_softmax) 应该等于 softmax
    for i in 0..2 {
        for j in 0..3 {
            assert_abs_diff_eq!(log_probs[[i, j]].exp(), probs[[i, j]], epsilon = 1e-6);
        }
    }
}

/// 测试 LogSoftmax 数值稳定性
#[test]
fn test_log_softmax_numerical_stability() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let log_softmax = graph
        .new_log_softmax_node(input, Some("log_softmax"))
        .unwrap();

    // 大数值不应该溢出
    let input_value = Tensor::new(&[1000.0, 1001.0, 1002.0], &[1, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(log_softmax).unwrap();

    let output = graph.get_node_value(log_softmax).unwrap().unwrap();

    // 输出应该是有限值
    for j in 0..3 {
        assert!(output[[0, j]].is_finite(), "输出应为有限值");
    }

    // exp 后和应为 1
    let sum = output[[0, 0]].exp() + output[[0, 1]].exp() + output[[0, 2]].exp();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 LogSoftmax 对父节点的梯度计算
///
/// 对于 y = log_softmax(x)，有：
/// ∂y_i/∂x_j = δ_ij - softmax(x)_j
///
/// VJP: dL/dx_i = upstream_grad_i - softmax_i * sum(upstream_grad)
#[test]
fn test_log_softmax_backward_vjp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let log_softmax_id = graph.new_log_softmax_node(input_id, Some("log_softmax"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(log_softmax_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 3]);
    let log_softmax_node = graph.get_node(log_softmax_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = log_softmax_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 验证梯度形状
    assert_eq!(grad.shape(), &[2, 3]);

    // 当 upstream_grad 全为 1 时，sum(upstream_grad) = num_classes = 3
    // dL/dx_i = 1 - softmax_i * 3
    // 由于 sum(softmax) = 1，sum(grad) = num_classes - 3 * sum(softmax) = 3 - 3 = 0
    // 所以每行梯度和应该接近 0
    let row0_sum = grad[[0, 0]] + grad[[0, 1]] + grad[[0, 2]];
    let row1_sum = grad[[1, 0]] + grad[[1, 1]] + grad[[1, 2]];
    assert_abs_diff_eq!(row0_sum, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(row1_sum, 0.0, epsilon = 1e-5);

    Ok(())
}

/// 测试 LogSoftmax 梯度计算（非单位 upstream_grad）
#[test]
fn test_log_softmax_backward_with_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 3], Some("input"))?;
    let log_softmax_id = graph.new_log_softmax_node(input_id, Some("log_softmax"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(log_softmax_id)?;

    // 非单位 upstream_grad
    let upstream_grad = Tensor::new(&[0.0, 0.0, 1.0], &[1, 3]);
    let log_softmax_node = graph.get_node(log_softmax_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = log_softmax_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 验证梯度形状
    assert_eq!(grad.shape(), &[1, 3]);

    // 梯度应该非零
    assert!(grad.data_as_slice().iter().any(|&g| g.abs() > 1e-6));

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 LogSoftmax 通过 graph.backward() 的端到端反向传播
#[test]
fn test_log_softmax_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = log_softmax(input)
    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let result = graph.new_log_softmax_node(input, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_basic_input_node(&[2, 3], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
    let target_value = Tensor::new(&[-2.0, -1.0, 0.0, -1.1, -1.1, -1.1], &[2, 3]);
    graph.set_node_value(input, Some(&input_value))?;
    graph.set_node_value(target, Some(&target_value))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    let loss_returned = graph.backward(loss)?;
    assert!(loss_returned >= 0.0);

    // 验证梯度存在且形状正确
    let input_grad = graph.get_node(input)?.grad().expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    // 梯度应该非零
    assert!(input_grad.data_as_slice().iter().any(|&g| g.abs() > 1e-6));

    Ok(())
}

/// 测试 LogSoftmax 在链式网络中的端到端反向传播
#[test]
fn test_log_softmax_backward_e2e_chain() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 构建网络: output = log_softmax(w @ x + b)
    let x = graph.new_basic_input_node(&[2, 1], Some("x"))?;
    let w = graph.new_parameter_node(&[3, 2], Some("w"))?;
    let b = graph.new_parameter_node(&[3, 1], Some("b"))?;
    let wx = graph.new_mat_mul_node(w, x, Some("wx"))?;
    let z = graph.new_add_node(&[wx, b], Some("z"))?;

    // 需要 reshape 成 [1, 3] 用于 log_softmax
    let z_reshaped = graph.new_reshape_node(z, &[1, 3], Some("z_reshaped"))?;
    let output = graph.new_log_softmax_node(z_reshaped, Some("output"))?;

    // loss = MSE(output, target)
    let target = graph.new_basic_input_node(&[1, 3], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 设置输入
    graph.set_node_value(x, Some(&Tensor::new(&[1.0, 0.5], &[2, 1])))?;
    graph.set_node_value(w, Some(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2])))?;
    graph.set_node_value(b, Some(&Tensor::new(&[0.0, 0.0, 0.0], &[3, 1])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[-1.0, -1.0, -1.0], &[1, 3])))?;

    // 前向传播
    graph.forward(loss)?;

    // 验证 log_softmax 输出为负数
    let output_val = graph.get_node_value(output)?.unwrap();
    for j in 0..3 {
        assert!(output_val[[0, j]] < 0.0, "log_softmax 输出应为负数");
    }

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证 w 和 b 的梯度存在且形状正确
    let w_grad = graph.get_node(w)?.grad().expect("w 应有 grad");
    let b_grad = graph.get_node(b)?.grad().expect("b 应有 grad");
    assert_eq!(w_grad.shape(), &[3, 2]);
    assert_eq!(b_grad.shape(), &[3, 1]);

    Ok(())
}

// ==================== 梯度累积测试 ====================

/// 测试 LogSoftmax 梯度累积
#[test]
fn test_log_softmax_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let result = graph.new_log_softmax_node(input, Some("result"))?;
    let target = graph.new_basic_input_node(&[2, 3], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3])))?;
    graph.set_node_value(
        target,
        Some(&Tensor::new(&[-2.0, -1.0, 0.0, -1.1, -1.1, -1.1], &[2, 3])),
    )?;
    graph.forward(loss)?;

    // 第 1 次反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;
    let grad_first = graph.get_node(input)?.grad().unwrap().clone();

    // 第 2 次反向传播（梯度累积）
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

// ==================== 动态形状测试 ====================

/// 测试 LogSoftmax 节点的动态形状传播
#[test]
fn test_log_softmax_dynamic_shape_propagation() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarActivationOps;

    let graph = Graph::new();

    // 创建 2D 输入
    let x = graph.input(&Tensor::zeros(&[4, 10])).unwrap();

    // LogSoftmax
    let log_probs = x.log_softmax();

    // 验证动态形状传播
    let dyn_shape = log_probs.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "num_classes 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(10), "num_classes 应该是 10");
}

/// 测试 LogSoftmax 节点在不同 batch_size 下的前向计算
#[test]
fn test_log_softmax_dynamic_batch_forward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarActivationOps;

    let graph = Graph::new();

    // 创建 2D 输入
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))
        .unwrap();

    // LogSoftmax
    let log_probs = x.log_softmax();

    // 第一次 forward：batch=2
    log_probs.forward().unwrap();
    let value1 = log_probs.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 3], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        &[4, 3],
    ))
    .unwrap();

    // 第二次 forward：batch=4
    log_probs.forward().unwrap();
    let value2 = log_probs.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[4, 3], "第二次 forward: batch=4");
}

/// 测试 LogSoftmax 节点在不同 batch_size 下的反向传播
#[test]
fn test_log_softmax_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::{VarActivationOps, VarLossOps};

    let graph = Graph::new();

    // 创建参数和目标
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[-2.0, -1.0, 0.0, -1.1, -1.1, -1.1], &[2, 3]))
        .unwrap();

    // LogSoftmax + MSE
    let log_probs = x.log_softmax();
    let loss = log_probs.mse_loss(&target).unwrap();

    // 第一次训练：batch=2
    loss.forward().unwrap();
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::normal_seeded(0.0, 1.0, &[5, 3], 42))
        .unwrap();
    target.set_value(&Tensor::zeros(&[5, 3])).unwrap();

    // 第二次训练：batch=5
    loss.forward().unwrap();
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
}
