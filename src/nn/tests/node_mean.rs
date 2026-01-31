/*
 * @Author       : 老董
 * @Description  : Mean 节点单元测试
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）
 * 2. 前向传播测试（全局均值、按轴均值）
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

/// 测试 Mean 节点创建（全局模式）
#[test]
fn test_mean_creation_global() {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();
    let mean = graph
        .new_mean_node(input, None, Some("mean_global"))
        .unwrap();

    assert_eq!(graph.get_node_name(mean).unwrap(), "mean_global");
    assert_eq!(graph.get_node_parents(mean).unwrap().len(), 1);
    // 全局均值输出 [1, 1]
    assert_eq!(graph.get_node_value_expected_shape(mean).unwrap(), &[1, 1]);
}

/// 测试 Mean 节点创建（按轴模式）
#[test]
fn test_mean_creation_axis() {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();

    // axis=0: [2, 3] -> [1, 3]
    let mean0 = graph
        .new_mean_node(input, Some(0), Some("mean_axis0"))
        .unwrap();
    assert_eq!(
        graph.get_node_value_expected_shape(mean0).unwrap(),
        &[1, 3]
    );

    // axis=1: [2, 3] -> [2, 1]
    let mean1 = graph
        .new_mean_node(input, Some(1), Some("mean_axis1"))
        .unwrap();
    assert_eq!(
        graph.get_node_value_expected_shape(mean1).unwrap(),
        &[2, 1]
    );
}

/// 测试 Mean 节点 axis 超出范围
#[test]
fn test_mean_invalid_axis() {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();
    let result = graph.new_mean_node(input, Some(2), Some("mean_invalid"));
    assert_err!(
        result,
        GraphError::InvalidOperation("Mean: axis 2 超出输入维度范围 2")
    );
}

/// 测试 Mean 节点命名
#[test]
fn test_mean_name_generation() {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();

    // 1. 显式命名
    let mean1 = graph.new_mean_node(input, None, Some("my_mean")).unwrap();
    assert_eq!(graph.get_node_name(mean1).unwrap(), "my_mean");

    // 2. 自动命名
    let mean2 = graph.new_mean_node(input, None, None).unwrap();
    assert_eq!(graph.get_node_name(mean2).unwrap(), "mean_1");

    // 3. 名称重复
    let result = graph.new_mean_node(input, None, Some("my_mean"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_mean在图default_graph中重复")
    );
}

/// 测试 Mean 节点不能直接设置值
#[test]
fn test_mean_cannot_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();
    let mean = graph.new_mean_node(input, None, Some("mean")).unwrap();

    let test_value = Tensor::new(&[3.5], &[1, 1]);
    assert_err!(
        graph.set_node_value(mean, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=mean, type=Mean]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试 Mean 全局均值
#[test]
fn test_mean_forward_global() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input")).unwrap();
    let mean = graph.new_mean_node(input, None, Some("mean")).unwrap();

    // 设置值 [[1, 2, 3], [4, 5, 6]]
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(mean).unwrap();

    let output = graph.get_node_value(mean).unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 1]);
    // mean = (1+2+3+4+5+6) / 6 = 3.5
    assert_abs_diff_eq!(output[[0, 0]], 3.5, epsilon = 1e-6);
}

/// 测试 Mean 按轴均值（axis=0）
#[test]
fn test_mean_forward_axis0() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input")).unwrap();
    let mean = graph.new_mean_node(input, Some(0), Some("mean")).unwrap();

    // [[1, 2, 3], [4, 5, 6]] -> [[2.5, 3.5, 4.5]]
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(mean).unwrap();

    let output = graph.get_node_value(mean).unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 3]);
    assert_abs_diff_eq!(output[[0, 0]], 2.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 3.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 4.5, epsilon = 1e-6);
}

/// 测试 Mean 按轴均值（axis=1）
#[test]
fn test_mean_forward_axis1() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input")).unwrap();
    let mean = graph.new_mean_node(input, Some(1), Some("mean")).unwrap();

    // [[1, 2, 3], [4, 5, 6]] -> [[2], [5]]
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(mean).unwrap();

    let output = graph.get_node_value(mean).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 1]);
    assert_abs_diff_eq!(output[[0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 5.0, epsilon = 1e-6);
}

/// 测试 Mean 3D 张量按轴均值
#[test]
fn test_mean_forward_3d() {
    let mut graph = GraphInner::new();

    let input = graph
        .new_basic_input_node(&[2, 3, 4], Some("input"))
        .unwrap();
    let mean = graph.new_mean_node(input, Some(1), Some("mean")).unwrap();

    // [2, 3, 4] -> [2, 1, 4]
    assert_eq!(
        graph.get_node_value_expected_shape(mean).unwrap(),
        &[2, 1, 4]
    );

    // 设置值
    let input_value = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ],
        &[2, 3, 4],
    );
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(mean).unwrap();

    let output = graph.get_node_value(mean).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 1, 4]);
    // 第一个 batch: mean = sum / 3 = [15,18,21,24] / 3 = [5,6,7,8]
    assert_abs_diff_eq!(output[[0, 0, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 2]], 7.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 3]], 8.0, epsilon = 1e-6);
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 Mean 全局均值的 VJP
#[test]
fn test_mean_backward_vjp_global() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let mean_id = graph.new_mean_node(input_id, None, Some("mean"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(mean_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::new(&[6.0], &[1, 1]);
    let mean_node = graph.get_node(mean_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = mean_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 验证梯度形状
    assert_eq!(grad.shape(), &[2, 3]);

    // 全局均值梯度：upstream_grad / n 广播到输入形状
    // n = 6，所以所有元素都应该等于 6.0 / 6 = 1.0
    for val in grad.data_as_slice() {
        assert_abs_diff_eq!(*val, 1.0, epsilon = 1e-6);
    }

    Ok(())
}

/// 测试 Mean 按轴均值的 VJP
#[test]
fn test_mean_backward_vjp_axis() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let mean_id = graph.new_mean_node(input_id, Some(1), Some("mean"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(mean_id)?;

    // 直接测试 VJP
    // upstream_grad shape: [2, 1]
    let upstream_grad = Tensor::new(&[3.0, 6.0], &[2, 1]);
    let mean_node = graph.get_node(mean_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = mean_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 验证梯度形状
    assert_eq!(grad.shape(), &[2, 3]);

    // 按轴均值梯度：upstream_grad / n 沿 axis 广播回输入形状
    // n = 3（axis=1 的大小）
    // 第一行全为 3.0 / 3 = 1.0，第二行全为 6.0 / 3 = 2.0
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 2]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 2]], 2.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 Mean 通过 graph.backward() 的端到端反向传播（全局模式）
#[test]
fn test_mean_backward_e2e_global() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：loss = MSE(mean(input), target)
    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let mean = graph.new_mean_node(input, None, Some("mean"))?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(mean, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[3.0], &[1, 1])))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证梯度存在且形状正确
    let input_grad = graph.get_node(input)?.grad().expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    // 梯度应该非零且相等（因为全局均值对每个元素贡献相同）
    let first_grad = input_grad[[0, 0]];
    for val in input_grad.data_as_slice() {
        assert_abs_diff_eq!(*val, first_grad, epsilon = 1e-6);
    }

    Ok(())
}

/// 测试 Mean 通过 graph.backward() 的端到端反向传播（按轴模式）
#[test]
fn test_mean_backward_e2e_axis() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：loss = MSE(mean(input, axis=1), target)
    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let mean = graph.new_mean_node(input, Some(1), Some("mean"))?;
    let target = graph.new_basic_input_node(&[2, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(mean, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[2.0, 5.0], &[2, 1])))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证梯度存在且形状正确
    let input_grad = graph.get_node(input)?.grad().expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    // 同一行的梯度应该相等
    assert_abs_diff_eq!(input_grad[[0, 0]], input_grad[[0, 1]], epsilon = 1e-6);
    assert_abs_diff_eq!(input_grad[[0, 1]], input_grad[[0, 2]], epsilon = 1e-6);
    assert_abs_diff_eq!(input_grad[[1, 0]], input_grad[[1, 1]], epsilon = 1e-6);
    assert_abs_diff_eq!(input_grad[[1, 1]], input_grad[[1, 2]], epsilon = 1e-6);

    Ok(())
}

/// 测试 Mean 在链式网络中的端到端反向传播
#[test]
fn test_mean_backward_e2e_chain() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 构建网络: output = mean(w @ x + b, axis=1)
    let x = graph.new_basic_input_node(&[2, 1], Some("x"))?;
    let w = graph.new_parameter_node(&[3, 2], Some("w"))?;
    let b = graph.new_parameter_node(&[3, 1], Some("b"))?;
    let wx = graph.new_mat_mul_node(w, x, Some("wx"))?;
    let z = graph.new_add_node(&[wx, b], Some("z"))?;

    // 需要 reshape 成 [1, 3] 以便沿 axis=1 求均值
    let z_reshaped = graph.new_reshape_node(z, &[1, 3], Some("z_reshaped"))?;
    let output = graph.new_mean_node(z_reshaped, Some(1), Some("output"))?;

    // loss = MSE(output, target)
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 设置输入
    graph.set_node_value(x, Some(&Tensor::new(&[1.0, 0.5], &[2, 1])))?;
    graph.set_node_value(w, Some(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2])))?;
    graph.set_node_value(b, Some(&Tensor::new(&[0.0, 0.0, 0.0], &[3, 1])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // 前向传播
    graph.forward(loss)?;

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

/// 测试 Mean 梯度累积
#[test]
fn test_mean_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let mean = graph.new_mean_node(input, None, Some("mean"))?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(mean, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[3.0], &[1, 1])))?;
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

/// 测试 Mean 节点的动态形状传播
#[test]
fn test_mean_dynamic_shape_propagation() {
    use crate::nn::var_ops::VarReduceOps;
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建 2D 输入
    let x = graph.input(&Tensor::zeros(&[4, 10])).unwrap();

    // 全局 Mean
    let mean_global = x.mean();
    let dyn_shape_global = mean_global.dynamic_expected_shape();
    // 全局均值输出 [1, 1]，固定形状
    assert!(!dyn_shape_global.is_dynamic(0));
    assert!(!dyn_shape_global.is_dynamic(1));

    // 按轴 Mean (axis=1)
    let mean_axis = x.mean_axis(1);
    let dyn_shape_axis = mean_axis.dynamic_expected_shape();
    // axis=1 均值后 [4, 1]，batch 维度仍是动态的
    assert!(dyn_shape_axis.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape_axis.is_dynamic(1), "均值后的维度应该是固定的");
}

/// 测试 Mean 节点在不同 batch_size 下的前向计算
#[test]
fn test_mean_dynamic_batch_forward() {
    use crate::nn::var_ops::VarReduceOps;
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建 2D 输入
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    // 按轴 Mean (axis=1)
    let mean = x.mean_axis(1);

    // 第一次 forward：batch=2
    mean.forward().unwrap();
    let value1 = mean.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 1], "第一次 forward: batch=2");
    assert_abs_diff_eq!(value1[[0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(value1[[1, 0]], 5.0, epsilon = 1e-6);

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(
        &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        &[4, 3],
    ))
    .unwrap();

    // 第二次 forward：batch=4
    mean.forward().unwrap();
    let value2 = mean.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[4, 1], "第二次 forward: batch=4");
    assert_abs_diff_eq!(value2[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(value2[[1, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(value2[[2, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(value2[[3, 0]], 4.0, epsilon = 1e-6);
}

/// 测试 Mean 节点在不同 batch_size 下的反向传播
#[test]
fn test_mean_dynamic_batch_backward() {
    use crate::nn::var_ops::{VarLossOps, VarReduceOps};
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建参数和目标
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[2.0, 5.0], &[2, 1])).unwrap();

    // Mean + MSE
    let mean = x.mean_axis(1);
    let loss = mean.mse_loss(&target).unwrap();

    // 第一次训练：batch=2
    loss.forward().unwrap();
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::normal_seeded(0.0, 1.0, &[5, 3], 42))
        .unwrap();
    target.set_value(&Tensor::zeros(&[5, 1])).unwrap();

    // 第二次训练：batch=5
    loss.forward().unwrap();
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
}

// ==================== 数值正确性验证 ====================

/// 验证 Mean 与 Sum/n 等价
#[test]
fn test_mean_equals_sum_div_n() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let mean = graph.new_mean_node(input, None, Some("mean"))?;
    let sum = graph.new_sum_node(input, None, Some("sum"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_value))?;

    graph.forward(mean)?;
    graph.forward(sum)?;

    let mean_val = graph.get_node_value(mean)?.unwrap()[[0, 0]];
    let sum_val = graph.get_node_value(sum)?.unwrap()[[0, 0]];

    // mean = sum / n
    assert_abs_diff_eq!(mean_val, sum_val / 6.0, epsilon = 1e-6);

    Ok(())
}
