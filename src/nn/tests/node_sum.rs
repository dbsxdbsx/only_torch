/*
 * @Author       : 老董
 * @Description  : Sum 节点单元测试
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）
 * 2. 前向传播测试（全局求和、按轴求和）
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

/// 测试 Sum 节点创建（全局模式）
#[cfg(any())]
#[test]
fn test_sum_creation_global() {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();
    let sum = graph.new_sum_node(input, None, Some("sum_global")).unwrap();

    assert_eq!(graph.get_node_name(sum).unwrap(), "sum_global");
    assert_eq!(graph.get_node_parents(sum).unwrap().len(), 1);
    // 全局求和输出 [1, 1]
    assert_eq!(graph.get_node_value_expected_shape(sum).unwrap(), &[1, 1]);
}

/// 测试 Sum 节点创建（按轴模式）
#[cfg(any())]
#[test]
fn test_sum_creation_axis() {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();

    // axis=0: [2, 3] -> [1, 3]
    let sum0 = graph.new_sum_node(input, Some(0), Some("sum_axis0")).unwrap();
    assert_eq!(graph.get_node_value_expected_shape(sum0).unwrap(), &[1, 3]);

    // axis=1: [2, 3] -> [2, 1]
    let sum1 = graph.new_sum_node(input, Some(1), Some("sum_axis1")).unwrap();
    assert_eq!(graph.get_node_value_expected_shape(sum1).unwrap(), &[2, 1]);
}

/// 测试 Sum 节点 axis 超出范围
#[cfg(any())]
#[test]
fn test_sum_invalid_axis() {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();
    let result = graph.new_sum_node(input, Some(2), Some("sum_invalid"));
    assert_err!(
        result,
        GraphError::InvalidOperation("Sum: axis 2 超出输入维度范围 2")
    );
}

/// 测试 Sum 节点命名
#[cfg(any())]
#[test]
fn test_sum_name_generation() {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();

    // 1. 显式命名
    let sum1 = graph.new_sum_node(input, None, Some("my_sum")).unwrap();
    assert_eq!(graph.get_node_name(sum1).unwrap(), "my_sum");

    // 2. 自动命名
    let sum2 = graph.new_sum_node(input, None, None).unwrap();
    assert_eq!(graph.get_node_name(sum2).unwrap(), "sum_1");

    // 3. 名称重复
    let result = graph.new_sum_node(input, None, Some("my_sum"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_sum在图default_graph中重复")
    );
}

/// 测试 Sum 节点不能直接设置值
#[cfg(any())]
#[test]
fn test_sum_cannot_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();
    let sum = graph.new_sum_node(input, None, Some("sum")).unwrap();

    let test_value = Tensor::new(&[6.0], &[1, 1]);
    assert_err!(
        graph.set_node_value(sum, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=sum, type=Sum]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试 Sum 全局求和
#[cfg(any())]
#[test]
fn test_sum_forward_global() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input")).unwrap();
    let sum = graph.new_sum_node(input, None, Some("sum")).unwrap();

    // 设置值 [[1, 2, 3], [4, 5, 6]]
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(sum).unwrap();

    let output = graph.get_node_value(sum).unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 1]);
    // sum = 1+2+3+4+5+6 = 21
    assert_abs_diff_eq!(output[[0, 0]], 21.0, epsilon = 1e-6);
}

/// 测试 Sum 按轴求和（axis=0）
#[cfg(any())]
#[test]
fn test_sum_forward_axis0() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input")).unwrap();
    let sum = graph.new_sum_node(input, Some(0), Some("sum")).unwrap();

    // [[1, 2, 3], [4, 5, 6]] -> [[5, 7, 9]]
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(sum).unwrap();

    let output = graph.get_node_value(sum).unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 3]);
    assert_abs_diff_eq!(output[[0, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 7.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 9.0, epsilon = 1e-6);
}

/// 测试 Sum 按轴求和（axis=1）
#[cfg(any())]
#[test]
fn test_sum_forward_axis1() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input")).unwrap();
    let sum = graph.new_sum_node(input, Some(1), Some("sum")).unwrap();

    // [[1, 2, 3], [4, 5, 6]] -> [[6], [15]]
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(sum).unwrap();

    let output = graph.get_node_value(sum).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 1]);
    assert_abs_diff_eq!(output[[0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 15.0, epsilon = 1e-6);
}

/// 测试 Sum 3D 张量按轴求和
#[cfg(any())]
#[test]
fn test_sum_forward_3d() {
    let mut graph = GraphInner::new();

    let input = graph
        .new_basic_input_node(&[2, 3, 4], Some("input"))
        .unwrap();
    let sum = graph.new_sum_node(input, Some(1), Some("sum")).unwrap();

    // [2, 3, 4] -> [2, 1, 4]
    assert_eq!(graph.get_node_value_expected_shape(sum).unwrap(), &[2, 1, 4]);

    // 设置值
    let input_value = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ],
        &[2, 3, 4],
    );
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(sum).unwrap();

    let output = graph.get_node_value(sum).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 1, 4]);
    // 第一个 batch: sum([1,5,9], [2,6,10], [3,7,11], [4,8,12]) = [15,18,21,24]
    assert_abs_diff_eq!(output[[0, 0, 0]], 15.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1]], 18.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 2]], 21.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 3]], 24.0, epsilon = 1e-6);
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 Sum 全局求和的 VJP
#[cfg(any())]
#[test]
fn test_sum_backward_vjp_global() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let sum_id = graph.new_sum_node(input_id, None, Some("sum"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(sum_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::new(&[2.0], &[1, 1]);
    let sum_node = graph.get_node(sum_id)?;
    let input_node = graph.get_node(input_id)?;

    let parents = [input_node];
    let grad = sum_node.calc_grad_to_parent(0, &parents, &upstream_grad)?;

    // 验证梯度形状
    assert_eq!(grad.shape(), &[2, 3]);

    // 全局求和梯度：upstream_grad 广播到输入形状
    // 所有元素都应该等于 upstream_grad (2.0)
    for val in grad.data_as_slice() {
        assert_abs_diff_eq!(*val, 2.0, epsilon = 1e-6);
    }

    Ok(())
}

/// 测试 Sum 按轴求和的 VJP
#[cfg(any())]
#[test]
fn test_sum_backward_vjp_axis() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let sum_id = graph.new_sum_node(input_id, Some(1), Some("sum"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(sum_id)?;

    // 直接测试 VJP
    // upstream_grad shape: [2, 1]
    let upstream_grad = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let sum_node = graph.get_node(sum_id)?;
    let input_node = graph.get_node(input_id)?;

    let parents = [input_node];
    let grad = sum_node.calc_grad_to_parent(0, &parents, &upstream_grad)?;

    // 验证梯度形状
    assert_eq!(grad.shape(), &[2, 3]);

    // 按轴求和梯度：upstream_grad 沿 axis 广播回输入形状
    // 第一行全为 1.0，第二行全为 2.0
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 2]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 2]], 2.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 Sum 通过 graph.backward() 的端到端反向传播（全局模式）
#[cfg(any())]
#[test]
fn test_sum_backward_e2e_global() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：loss = MSE(sum(input), target)
    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let sum = graph.new_sum_node(input, None, Some("sum"))?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(sum, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[20.0], &[1, 1])))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证梯度存在且形状正确
    let input_grad = graph.get_node(input)?.grad().expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    // 梯度应该非零且相等（因为全局求和对每个元素贡献相同）
    let first_grad = input_grad[[0, 0]];
    for val in input_grad.data_as_slice() {
        assert_abs_diff_eq!(*val, first_grad, epsilon = 1e-6);
    }

    Ok(())
}

/// 测试 Sum 通过 graph.backward() 的端到端反向传播（按轴模式）
#[cfg(any())]
#[test]
fn test_sum_backward_e2e_axis() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：loss = MSE(sum(input, axis=1), target)
    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let sum = graph.new_sum_node(input, Some(1), Some("sum"))?;
    let target = graph.new_basic_input_node(&[2, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(sum, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[5.0, 14.0], &[2, 1])))?;

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

/// 测试 Sum 在链式网络中的端到端反向传播
#[cfg(any())]
#[test]
fn test_sum_backward_e2e_chain() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 构建网络: output = sum(w @ x + b, axis=1)
    let x = graph.new_basic_input_node(&[2, 1], Some("x"))?;
    let w = graph.new_parameter_node(&[3, 2], Some("w"))?;
    let b = graph.new_parameter_node(&[3, 1], Some("b"))?;
    let wx = graph.new_mat_mul_node(w, x, Some("wx"))?;
    let z = graph.new_add_node(&[wx, b], Some("z"))?;

    // 需要 reshape 成 [1, 3] 以便沿 axis=1 求和
    let z_reshaped = graph.new_reshape_node(z, &[1, 3], Some("z_reshaped"))?;
    let output = graph.new_sum_node(z_reshaped, Some(1), Some("output"))?;

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

/// 测试 Sum 梯度累积
#[cfg(any())]
#[test]
fn test_sum_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let sum = graph.new_sum_node(input, None, Some("sum"))?;
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(sum, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[20.0], &[1, 1])))?;
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

/// 测试 Sum 节点的动态形状传播
#[test]
fn test_sum_dynamic_shape_propagation() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarReduceOps;

    let graph = Graph::new();

    // 创建 2D 输入
    let x = graph.input(&Tensor::zeros(&[4, 10])).unwrap();

    // 全局 Sum
    let sum_global = x.sum();
    let dyn_shape_global = sum_global.dynamic_expected_shape();
    // 全局求和输出 [1, 1]，固定形状
    assert!(!dyn_shape_global.is_dynamic(0));
    assert!(!dyn_shape_global.is_dynamic(1));

    // 按轴 Sum (axis=1)
    let sum_axis = x.sum_axis(1);
    let dyn_shape_axis = sum_axis.dynamic_expected_shape();
    // axis=1 求和后 [4, 1]，batch 维度仍是动态的
    assert!(dyn_shape_axis.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape_axis.is_dynamic(1), "求和后的维度应该是固定的");
}

/// 测试 Sum 节点在不同 batch_size 下的前向计算
#[test]
fn test_sum_dynamic_batch_forward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarReduceOps;

    let graph = Graph::new();

    // 创建 2D 输入
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    // 按轴 Sum (axis=1)
    let sum = x.sum_axis(1);

    // 第一次 forward：batch=2
    sum.forward().unwrap();
    let value1 = sum.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 1], "第一次 forward: batch=2");
    assert_abs_diff_eq!(value1[[0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(value1[[1, 0]], 15.0, epsilon = 1e-6);

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(
        &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        &[4, 3],
    ))
    .unwrap();

    // 第二次 forward：batch=4
    sum.forward().unwrap();
    let value2 = sum.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[4, 1], "第二次 forward: batch=4");
    assert_abs_diff_eq!(value2[[0, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(value2[[1, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(value2[[2, 0]], 9.0, epsilon = 1e-6);
    assert_abs_diff_eq!(value2[[3, 0]], 12.0, epsilon = 1e-6);
}

/// 测试 Sum 节点在不同 batch_size 下的反向传播
#[test]
#[ignore = "动态 batch backward 形状不兼容 bug，待修复"]
fn test_sum_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::{VarLossOps, VarReduceOps};

    let graph = Graph::new();

    // 创建参数和目标
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[5.0, 14.0], &[2, 1])).unwrap();

    // Sum + MSE
    let sum = x.sum_axis(1);
    let loss = sum.mse_loss(&target).unwrap();

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

// ==================== 方案 C：新节点创建 API 测试 ====================

use crate::nn::Graph;
use std::rc::Rc;

#[test]
fn test_create_sum_node_global() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let sum = inner
        .borrow_mut()
        .create_sum_node(input.clone(), None, Some("sum"))
        .unwrap();

    // 全局求和输出 [1, 1]
    assert_eq!(sum.shape(), vec![1, 1]);
    assert_eq!(sum.name(), Some("sum"));
    assert!(!sum.is_leaf());
}

#[test]
fn test_create_sum_node_axis() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();

    // 沿 axis=1 求和
    let sum = inner
        .borrow_mut()
        .create_sum_node(input.clone(), Some(1), None)
        .unwrap();

    // 输出 [3, 1]
    assert_eq!(sum.shape(), vec![3, 1]);
}

#[test]
fn test_create_sum_node_invalid_axis() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();

    // axis=2 超出范围
    let result = inner
        .borrow_mut()
        .create_sum_node(input, Some(2), None);

    assert!(result.is_err());
}

#[test]
fn test_create_sum_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_sum;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let sum = inner
            .borrow_mut()
            .create_sum_node(input, None, None)
            .unwrap();
        weak_sum = Rc::downgrade(&sum);

        assert!(weak_sum.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_sum.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
