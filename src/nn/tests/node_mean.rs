/*
 * @Author       : 老董
 * @Description  : Mean 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ global mean=3.5; axis0→[[2.5,3.5,4.5]]; axis1→[[2],[5]]; cannot_set_value
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ global VJP; axis=1 VJP
 * 3. 端到端反向传播测试（高层 API）→ global + axis
 * 4. 梯度累积测试（高层 API）
 * 5. 动态形状测试（KEEP AS-IS）
 * 6. Create API 测试（KEEP AS-IS）
 *
 * Mean 特性：与 Sum 类似但除以 count
 * - Global: mean([[1,2,3],[4,5,6]])=3.5, grad = upstream/n broadcast (n=total elements)
 * - Axis=0: [2,3]→[1,3], mean=[[2.5,3.5,4.5]], grad = upstream/n_axis broadcast
 * - Axis=1: [2,3]→[2,1], mean=[[2],[5]], grad = upstream/3 broadcast along axis
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarReduceOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 API）====================

/// 测试 Mean 全局均值：mean([[1,2,3],[4,5,6]])=3.5
#[test]
fn test_mean_forward_global() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    let mean = x.mean();
    mean.forward().unwrap();

    let output = mean.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 1]);
    assert_abs_diff_eq!(output[[0, 0]], 3.5, epsilon = 1e-6);
}

/// 测试 Mean 按轴均值（axis=0）：[2,3]→[1,3], mean=[[2.5,3.5,4.5]]
#[test]
fn test_mean_forward_axis0() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    let mean = x.mean_axis(0);
    mean.forward().unwrap();

    let output = mean.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 3]);
    assert_abs_diff_eq!(output[[0, 0]], 2.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 3.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 4.5, epsilon = 1e-6);
}

/// 测试 Mean 按轴均值（axis=1）：[2,3]→[2,1], mean=[[2],[5]]
#[test]
fn test_mean_forward_axis1() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    let mean = x.mean_axis(1);
    mean.forward().unwrap();

    let output = mean.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 1]);
    assert_abs_diff_eq!(output[[0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 5.0, epsilon = 1e-6);
}

/// 测试 Mean 节点不能直接设置值
#[test]
fn test_mean_cannot_set_value() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let mean = x.mean();

    let test_value = Tensor::new(&[3.5], &[1, 1]);
    let err = mean.set_value(&test_value);
    assert!(err.is_err(), "Mean 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================

/// 测试 Mean 全局均值的 VJP
///
/// upstream=6.0, n=6 → grad 全为 1.0
#[test]
fn test_mean_vjp_global() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("input"))
        .unwrap();
    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))?;

    let m = inner
        .borrow_mut()
        .create_mean_node(x.clone(), None, Some("m"))
        .unwrap();
    m.forward_recursive(1, false)?;

    let upstream_grad = Tensor::new(&[6.0], &[1, 1]);
    let grad = m.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 3]);
    for val in grad.data_as_slice() {
        assert_abs_diff_eq!(*val, 1.0, epsilon = 1e-6);
    }

    Ok(())
}

/// 测试 Mean 按轴均值的 VJP
///
/// upstream=[3, 6], n=3 → row0 全 1.0, row1 全 2.0
#[test]
fn test_mean_vjp_axis1() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("input"))
        .unwrap();
    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))?;

    let m = inner
        .borrow_mut()
        .create_mean_node(x.clone(), Some(1), Some("m"))
        .unwrap();
    m.forward_recursive(1, false)?;

    let upstream_grad = Tensor::new(&[3.0, 6.0], &[2, 1]);
    let grad = m.calc_grad_to_parent_index(0, &upstream_grad)?.resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 3]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 2]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 2]], 2.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 API）====================

/// 测试 Mean 端到端反向传播（全局模式）：验证梯度均匀
#[test]
fn test_mean_backward_e2e_global() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "input")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))?;

    let mean = x.mean();
    let target = graph.input(&Tensor::new(&[3.0], &[1, 1]))?;
    let loss = mean.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    let input_grad = x.grad()?.expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    let first_grad = input_grad[[0, 0]];
    for val in input_grad.data_as_slice() {
        assert_abs_diff_eq!(*val, first_grad, epsilon = 1e-6);
    }

    Ok(())
}

/// 测试 Mean 端到端反向传播（按轴模式）：验证同一行梯度相等
#[test]
fn test_mean_backward_e2e_axis() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "input")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))?;

    let mean = x.mean_axis(1);
    let target = graph.input(&Tensor::new(&[2.0, 5.0], &[2, 1]))?;
    let loss = mean.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    let input_grad = x.grad()?.expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    assert_abs_diff_eq!(input_grad[[0, 0]], input_grad[[0, 1]], epsilon = 1e-6);
    assert_abs_diff_eq!(input_grad[[0, 1]], input_grad[[0, 2]], epsilon = 1e-6);
    assert_abs_diff_eq!(input_grad[[1, 0]], input_grad[[1, 1]], epsilon = 1e-6);
    assert_abs_diff_eq!(input_grad[[1, 1]], input_grad[[1, 2]], epsilon = 1e-6);

    Ok(())
}

// ==================== 梯度累积测试（高层 API）====================

/// 测试 Mean 梯度累积
#[test]
fn test_mean_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "input")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))?;

    let mean = x.mean();
    let target = graph.input(&Tensor::new(&[3.0], &[1, 1]))?;
    let loss = mean.mse_loss(&target)?;

    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = x.grad()?.unwrap().clone();

    loss.forward().unwrap();
    loss.backward()?;
    let grad_second = x.grad()?.unwrap();
    assert_eq!(&grad_second, &(&grad_first * 2.0));

    graph.zero_grad()?;
    loss.forward().unwrap();
    loss.backward()?;
    let grad_after_clear = x.grad()?.unwrap();
    assert_eq!(&grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 Mean 节点的动态形状传播
#[test]
fn test_mean_dynamic_shape_propagation() {
    use crate::nn::Graph;
    use crate::nn::var::ops::VarReduceOps;

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
    use crate::nn::Graph;
    use crate::nn::var::ops::VarReduceOps;

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
    use crate::nn::Graph;
    use crate::nn::var::ops::{VarLossOps, VarReduceOps};

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

// ==================== 节点创建 API 测试 ====================

#[test]
fn test_create_mean_node_global() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let mean = inner
        .borrow_mut()
        .create_mean_node(input.clone(), None, Some("mean"))
        .unwrap();

    // 全局均值输出 [1, 1]
    assert_eq!(mean.shape(), vec![1, 1]);
    assert_eq!(mean.name(), Some("mean"));
}

#[test]
fn test_create_mean_node_axis() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();

    // 沿 axis=0 求均值
    let mean = inner
        .borrow_mut()
        .create_mean_node(input.clone(), Some(0), None)
        .unwrap();

    // 输出 [1, 4]
    assert_eq!(mean.shape(), vec![1, 4]);
}

#[test]
fn test_create_mean_node_invalid_axis() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();

    // axis=5 超出范围
    let result = inner.borrow_mut().create_mean_node(input, Some(5), None);

    assert!(result.is_err());
}

#[test]
fn test_create_mean_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_mean;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let mean = inner
            .borrow_mut()
            .create_mean_node(input, None, None)
            .unwrap();
        weak_mean = Rc::downgrade(&mean);

        assert!(weak_mean.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_mean.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
