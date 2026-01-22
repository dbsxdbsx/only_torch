/*
 * @Author       : 老董
 * @Description  : Softmax 节点单元测试
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）
 * 2. 前向传播测试
 * 3. VJP 单元测试（直接调用 calc_grad_to_parent）
 * 4. 端到端反向传播测试（通过 graph.backward）
 * 5. 梯度累积测试
 * 6. 动态形状测试
 */

use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 Softmax 节点创建
#[test]
fn test_softmax_creation() {
    let mut graph = GraphInner::new();

    // 1. Input 节点作为父节点
    {
        let input = graph.new_input_node(&[2, 3], Some("input1")).unwrap();
        let softmax = graph
            .new_softmax_node(input, Some("softmax_with_input"))
            .unwrap();

        assert_eq!(graph.get_node_name(softmax).unwrap(), "softmax_with_input");
        assert_eq!(graph.get_node_parents(softmax).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(softmax).unwrap().len(), 0);
        assert_eq!(
            graph.get_node_value_expected_shape(softmax).unwrap(),
            &[2, 3]
        );
    }

    // 2. Parameter 节点作为父节点
    {
        let param = graph.new_parameter_node(&[4, 5], Some("param1")).unwrap();
        let softmax = graph
            .new_softmax_node(param, Some("softmax_with_param"))
            .unwrap();

        assert_eq!(graph.get_node_name(softmax).unwrap(), "softmax_with_param");
        assert_eq!(graph.get_node_parents(softmax).unwrap().len(), 1);
        assert_eq!(
            graph.get_node_value_expected_shape(softmax).unwrap(),
            &[4, 5]
        );
    }
}

/// 测试 Softmax 节点要求 2D 输入
#[test]
fn test_softmax_requires_2d_input() {
    let mut graph = GraphInner::new();

    // 3D 输入应该失败（Softmax 只支持 2D）
    let input_3d = graph.new_input_node(&[2, 3, 4], Some("input_3d")).unwrap();
    let result = graph.new_softmax_node(input_3d, Some("softmax_3d"));
    assert!(result.is_err());

    // 4D 输入应该失败
    let input_4d = graph.new_input_node(&[2, 3, 4, 5], Some("input_4d")).unwrap();
    let result = graph.new_softmax_node(input_4d, Some("softmax_4d"));
    assert!(result.is_err());
}

/// 测试 Softmax 节点命名
#[test]
fn test_softmax_name_generation() {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();

    // 1. 显式命名
    let softmax1 = graph.new_softmax_node(input, Some("my_softmax")).unwrap();
    assert_eq!(graph.get_node_name(softmax1).unwrap(), "my_softmax");

    // 2. 自动命名
    let softmax2 = graph.new_softmax_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(softmax2).unwrap(), "softmax_1");

    // 3. 名称重复
    let result = graph.new_softmax_node(input, Some("my_softmax"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_softmax在图default_graph中重复")
    );
}

/// 测试 Softmax 节点不能直接设置值
#[test]
fn test_softmax_cannot_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();
    let softmax = graph.new_softmax_node(input, Some("softmax")).unwrap();

    let test_value = Tensor::new(&[0.1, 0.2, 0.7, 0.3, 0.3, 0.4], &[2, 3]);
    assert_err!(
        graph.set_node_value(softmax, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=softmax, type=Softmax]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试 Softmax 前向传播
#[test]
fn test_softmax_forward() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input")).unwrap();
    let softmax = graph.new_softmax_node(input, Some("softmax")).unwrap();

    // 测试数据
    // 第一行 [1, 2, 3]：softmax ≈ [0.09, 0.24, 0.67]
    // 第二行 [1, 1, 1]：softmax = [1/3, 1/3, 1/3]
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(softmax).unwrap();

    let output = graph.get_node_value(softmax).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 3]);

    // 验证每行归一化为 1
    let row0_sum: f32 = output.data_as_slice()[0..3].iter().sum();
    assert_abs_diff_eq!(row0_sum, 1.0, epsilon = 1e-5);

    let row1_sum: f32 = output.data_as_slice()[3..6].iter().sum();
    assert_abs_diff_eq!(row1_sum, 1.0, epsilon = 1e-5);

    // 验证第二行均匀分布
    assert_abs_diff_eq!(output[[1, 0]], 1.0 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[1, 1]], 1.0 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[1, 2]], 1.0 / 3.0, epsilon = 1e-5);
}

/// 测试 Softmax 前向传播（数值稳定性）
#[test]
fn test_softmax_forward_numerical_stability() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let softmax = graph.new_softmax_node(input, Some("softmax")).unwrap();

    // 使用大数值测试数值稳定性
    // 如果不使用 log-sum-exp 技巧，exp(100) 会溢出
    let input_value = Tensor::new(&[100.0, 100.0, 100.0], &[1, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(softmax).unwrap();

    let output = graph.get_node_value(softmax).unwrap().unwrap();

    // 应该是均匀分布 [1/3, 1/3, 1/3]
    assert_abs_diff_eq!(output[[0, 0]], 1.0 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 1]], 1.0 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 2]], 1.0 / 3.0, epsilon = 1e-5);
}

/// 测试 Softmax 前向传播（极端差异）
#[test]
fn test_softmax_forward_extreme_difference() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let softmax = graph.new_softmax_node(input, Some("softmax")).unwrap();

    // 一个很大，其他很小：应该接近 [0, 1, 0]
    let input_value = Tensor::new(&[-100.0, 100.0, -100.0], &[1, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(softmax).unwrap();

    let output = graph.get_node_value(softmax).unwrap().unwrap();

    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-30);
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-30);
    assert_abs_diff_eq!(output[[0, 2]], 0.0, epsilon = 1e-30);
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 Softmax 对父节点的梯度计算
///
/// 对于 y = softmax(x)，有：
/// - ∂y_i/∂x_j = y_i * (δ_ij - y_j)
/// - VJP: grad_to_parent_j = Σ_i upstream_grad_i * y_i * (δ_ij - y_j)
///                         = upstream_grad_j * y_j - y_j * Σ_i (upstream_grad_i * y_i)
#[test]
fn test_softmax_backward_vjp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 3], Some("input"))?;
    let softmax_id = graph.new_softmax_node(input_id, Some("softmax"))?;

    // 设置值：均匀输入 -> 均匀 softmax
    let input_value = Tensor::new(&[0.0, 0.0, 0.0], &[1, 3]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(softmax_id)?;

    // 验证 softmax 输出
    let softmax_output = graph.get_node_value(softmax_id)?.unwrap();
    assert_abs_diff_eq!(softmax_output[[0, 0]], 1.0 / 3.0, epsilon = 1e-5);

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[1, 3]);
    let softmax_node = graph.get_node(softmax_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = softmax_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 对于均匀 softmax 和全 1 upstream_grad：
    // grad_j = upstream_grad_j * y_j - y_j * Σ(upstream_grad_i * y_i)
    //        = 1 * (1/3) - (1/3) * (1 * 1/3 * 3) = 1/3 - 1/3 = 0
    assert_eq!(grad.shape(), &[1, 3]);
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 2]], 0.0, epsilon = 1e-5);

    Ok(())
}

/// 测试 Softmax 梯度计算（非均匀情况）
#[test]
fn test_softmax_backward_vjp_non_uniform() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 3], Some("input"))?;
    let softmax_id = graph.new_softmax_node(input_id, Some("softmax"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(softmax_id)?;

    // upstream_grad = [1, 0, 0]（只对第一个输出有梯度）
    let upstream_grad = Tensor::new(&[1.0, 0.0, 0.0], &[1, 3]);
    let softmax_node = graph.get_node(softmax_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = softmax_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // softmax([1,2,3]) ≈ [0.09, 0.24, 0.67]
    // grad_j = upstream_grad_j * y_j - y_j * Σ(upstream_grad_i * y_i)
    //        = upstream_grad_j * y_j - y_j * (1 * y_0)
    //        = upstream_grad_j * y_j - y_j * y_0
    // grad_0 = 1 * y_0 - y_0 * y_0 = y_0 * (1 - y_0) > 0
    // grad_1 = 0 * y_1 - y_1 * y_0 = -y_1 * y_0 < 0
    // grad_2 = 0 * y_2 - y_2 * y_0 = -y_2 * y_0 < 0
    assert_eq!(grad.shape(), &[1, 3]);
    assert!(grad[[0, 0]] > 0.0, "第一个梯度应该为正");
    assert!(grad[[0, 1]] < 0.0, "第二个梯度应该为负");
    assert!(grad[[0, 2]] < 0.0, "第三个梯度应该为负");

    // 梯度和为 0（softmax 的特性）
    let grad_sum: f32 = grad.data_as_slice().iter().sum();
    assert_abs_diff_eq!(grad_sum, 0.0, epsilon = 1e-5);

    Ok(())
}

/// 测试 Softmax 梯度计算（batch 模式）
#[test]
fn test_softmax_backward_vjp_batch() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let softmax_id = graph.new_softmax_node(input_id, Some("softmax"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(softmax_id)?;

    // 非单位 upstream_grad
    let upstream_grad = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
    let softmax_node = graph.get_node(softmax_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = softmax_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 验证形状
    assert_eq!(grad.shape(), &[2, 3]);

    // 每行梯度和应该为 0
    let row0_sum: f32 = grad.data_as_slice()[0..3].iter().sum();
    let row1_sum: f32 = grad.data_as_slice()[3..6].iter().sum();
    assert_abs_diff_eq!(row0_sum, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(row1_sum, 0.0, epsilon = 1e-5);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 Softmax 通过 graph.backward() 的端到端反向传播
#[test]
fn test_softmax_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = softmax(input)
    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let result = graph.new_softmax_node(input, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_input_node(&[2, 3], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
    // target 是 one-hot：第一行选择第三个，第二行选择第一个
    let target_value = Tensor::new(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_value))?;
    graph.set_node_value(target, Some(&target_value))?;

    // 前向传播
    graph.forward(loss)?;

    // 验证 loss 为正
    let loss_value = graph.get_node_value(loss)?.unwrap();
    assert!(loss_value[[0, 0]] > 0.0);

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证梯度存在且形状正确
    let input_grad = graph.get_node(input)?.grad().expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    // 每行梯度和应该为 0（softmax 的特性）
    let row0_sum: f32 = input_grad.data_as_slice()[0..3].iter().sum();
    let row1_sum: f32 = input_grad.data_as_slice()[3..6].iter().sum();
    assert_abs_diff_eq!(row0_sum, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(row1_sum, 0.0, epsilon = 1e-5);

    Ok(())
}

/// 测试 Softmax 在链式网络中的端到端反向传播
///
/// 网络结构: x -> MatMul(w) -> Add(b) -> Softmax -> output
#[test]
fn test_softmax_backward_e2e_chain() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 构建网络: output = softmax(w @ x + b)
    let x = graph.new_input_node(&[2, 3], Some("x"))?;
    let w = graph.new_parameter_node(&[3, 4], Some("w"))?;
    let b = graph.new_parameter_node(&[2, 4], Some("b"))?;
    let wx = graph.new_mat_mul_node(x, w, Some("xw"))?;
    let z = graph.new_add_node(&[wx, b], Some("z"))?;
    let output = graph.new_softmax_node(z, Some("output"))?;

    // loss = MSE(output, target)
    let target = graph.new_input_node(&[2, 4], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 设置输入
    graph.set_node_value(x, Some(&Tensor::normal_seeded(0.0, 1.0, &[2, 3], 100)))?;
    graph.set_node_value(w, Some(&Tensor::normal_seeded(0.0, 0.5, &[3, 4], 101)))?;
    graph.set_node_value(b, Some(&Tensor::zeros(&[2, 4])))?;
    // one-hot target
    let mut target_data = Tensor::zeros(&[2, 4]);
    target_data[[0, 0]] = 1.0;
    target_data[[1, 2]] = 1.0;
    graph.set_node_value(target, Some(&target_data))?;

    // 前向传播
    graph.forward(loss)?;

    // 验证输出是概率分布
    let output_val = graph.get_node_value(output)?.unwrap();
    let row0_sum: f32 = (0..4).map(|j| output_val[[0, j]]).sum();
    let row1_sum: f32 = (0..4).map(|j| output_val[[1, j]]).sum();
    assert_abs_diff_eq!(row0_sum, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(row1_sum, 1.0, epsilon = 1e-5);

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证 w 和 b 的梯度存在且形状正确
    let w_grad = graph.get_node(w)?.grad().expect("w 应有 grad");
    let b_grad = graph.get_node(b)?.grad().expect("b 应有 grad");
    assert_eq!(w_grad.shape(), &[3, 4]);
    assert_eq!(b_grad.shape(), &[2, 4]);

    Ok(())
}

// ==================== 梯度累积测试 ====================

/// 测试 Softmax 梯度累积
#[test]
fn test_softmax_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let result = graph.new_softmax_node(input, Some("result"))?;
    let target = graph.new_input_node(&[2, 3], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3])))?;
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

/// 测试 Softmax 节点的动态形状传播
#[test]
fn test_softmax_dynamic_shape_propagation() {
    use crate::nn::var_ops::VarActivationOps;
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建 2D 输入（Softmax 要求 2D）：[batch, num_classes]
    // Input 节点默认支持动态 batch
    let x = graph.input(&Tensor::zeros(&[4, 10])).unwrap();

    // Softmax
    let probs = x.softmax();

    // 验证动态形状传播
    let dyn_shape = probs.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "num_classes 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(10), "num_classes 应该是 10");
}

/// 测试 Softmax 节点在不同 batch_size 下的前向计算
#[test]
fn test_softmax_dynamic_batch_forward() {
    use crate::nn::var_ops::VarActivationOps;
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建 2D 输入
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))
        .unwrap();

    // Softmax
    let probs = x.softmax();

    // 第一次 forward：batch=2
    probs.forward().unwrap();
    let value1 = probs.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 3], "第一次 forward: batch=2");
    // 验证每行归一化
    let row0_sum: f32 = value1.data_as_slice()[0..3].iter().sum();
    assert_abs_diff_eq!(row0_sum, 1.0, epsilon = 1e-5);

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        &[4, 3],
    ))
    .unwrap();

    // 第二次 forward：batch=4
    probs.forward().unwrap();
    let value2 = probs.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[4, 3], "第二次 forward: batch=4");
}

/// 测试 Softmax 节点在不同 batch_size 下的反向传播
#[test]
fn test_softmax_dynamic_batch_backward() {
    use crate::nn::var_ops::{VarActivationOps, VarLossOps};
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建参数和目标
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3]))
        .unwrap();

    // Softmax + MSE
    let probs = x.softmax();
    let loss = probs.mse_loss(&target).unwrap();

    // 第一次训练：batch=2
    loss.forward().unwrap();
    let loss_val1 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);
    graph.zero_grad();
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::normal_seeded(0.0, 1.0, &[5, 3], 42))
        .unwrap();
    target.set_value(&Tensor::zeros(&[5, 3])).unwrap();

    // 第二次训练：batch=5
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
    graph.zero_grad();
    loss.backward().unwrap();
}
