/*
 * @Author       : 老董
 * @Description  : Sum 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 Graph + Var API）→ global / axis0 / axis1 / cannot_set_value
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ global / axis=1
 * 3. 端到端反向传播测试（高层 Graph + Var API）
 * 4. 梯度累积测试（高层 Graph + Var API）
 * 5. 动态形状测试（已有）
 * 6. 新节点创建 API 测试（已有）
 *
 * Key: Sum 支持全局 (axis=None→[1,1]) 和按轴归约。
 * - Global: [2,3]→[1,1], grad = upstream broadcast 到输入形状（全相同值）
 * - Axis=0: [2,3]→[1,3], grad = upstream 沿 axis 0 broadcast
 * - Axis=1: [2,3]→[2,1], grad = upstream 沿 axis 1 broadcast
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarMatrixOps, VarReduceOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 Sum 全局求和
///
/// sum([[1,2,3],[4,5,6]]) = 21
#[test]
fn test_sum_forward_global() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let sum = x.sum();

    sum.forward().unwrap();

    let output = sum.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 1]);
    assert_abs_diff_eq!(output[[0, 0]], 21.0, epsilon = 1e-6);
}

/// 测试 Sum 按轴求和（axis=0）
///
/// [[1,2,3],[4,5,6]] → [[5,7,9]]
#[test]
fn test_sum_forward_axis0() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let sum = x.sum_axis(0);

    sum.forward().unwrap();

    let output = sum.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 3]);
    assert_abs_diff_eq!(output[[0, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 7.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 9.0, epsilon = 1e-6);
}

/// 测试 Sum 按轴求和（axis=1）
///
/// [[1,2,3],[4,5,6]] → [[6],[15]]
#[test]
fn test_sum_forward_axis1() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let sum = x.sum_axis(1);

    sum.forward().unwrap();

    let output = sum.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 1]);
    assert_abs_diff_eq!(output[[0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 15.0, epsilon = 1e-6);
}

/// 测试 Sum 节点不能直接设置值
#[test]
fn test_sum_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let sum = x.sum();

    let test_value = Tensor::new(&[6.0], &[1, 1]);
    let err = sum.set_value(&test_value);
    assert!(err.is_err(), "Sum 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证梯度计算公式。
// Sum VJP: grad = broadcast(upstream_grad, input_shape)

/// 测试 Sum 全局求和的 VJP
///
/// upstream [1,1] = 2.0 → grad [2,3] 全为 2.0
#[test]
fn test_sum_vjp_global() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();
    let sum = inner
        .borrow_mut()
        .create_sum_node(x.clone(), None, Some("s"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))
        .unwrap();
    sum.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0], &[1, 1]);
    let grad = sum.calc_grad_to_parent_index(0, &upstream_grad)?;

    assert_eq!(grad.shape(), &[2, 3]);
    for val in grad.data_as_slice() {
        assert_abs_diff_eq!(*val, 2.0, epsilon = 1e-6);
    }

    Ok(())
}

/// 测试 Sum 按轴求和的 VJP（axis=1）
///
/// upstream [2,1] = [1,2] → grad [2,3] 第一行全 1，第二行全 2
#[test]
fn test_sum_vjp_axis1() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();
    let sum = inner
        .borrow_mut()
        .create_sum_node(x.clone(), Some(1), Some("s"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))
        .unwrap();
    sum.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let grad = sum.calc_grad_to_parent_index(0, &upstream_grad)?;

    assert_eq!(grad.shape(), &[2, 3]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 2]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 2]], 2.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 Sum 通过 loss.backward() 的端到端反向传播（全局模式）
///
/// loss = MSE(sum(input), target)，验证 input 梯度均匀
#[test]
fn test_sum_backward_e2e_global() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "input")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))?;

    let sum = x.sum();
    let target = graph.input(&Tensor::new(&[20.0], &[1, 1]))?;
    let loss = sum.mse_loss(&target)?;

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

/// 测试 Sum 通过 loss.backward() 的端到端反向传播（按轴模式）
///
/// loss = MSE(sum(input, axis=1), target)，验证同一行梯度相等
#[test]
fn test_sum_backward_e2e_axis() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "input")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))?;

    let sum = x.sum_axis(1);
    let target = graph.input(&Tensor::new(&[5.0, 14.0], &[2, 1]))?;
    let loss = sum.mse_loss(&target)?;

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

/// 测试 Sum 在链式网络中的端到端反向传播
#[test]
fn test_sum_backward_e2e_chain() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::new(&[1.0, 0.5], &[2, 1]))?;
    let w = graph.parameter(&[3, 2], Init::Zeros, "w")?;
    let b = graph.parameter(&[3, 1], Init::Zeros, "b")?;

    w.set_value(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]))?;
    b.set_value(&Tensor::new(&[0.0, 0.0, 0.0], &[3, 1]))?;

    let wx = w.matmul(&x)?;
    let z = &wx + &b;
    let z_reshaped = z.reshape(&[1, 3])?;
    let output = z_reshaped.sum_axis(1);

    let target = graph.input(&Tensor::new(&[1.0], &[1, 1]))?;
    let loss = output.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    let w_grad = w.grad()?.expect("w 应有 grad");
    let b_grad = b.grad()?.expect("b 应有 grad");
    assert_eq!(w_grad.shape(), &[3, 2]);
    assert_eq!(b_grad.shape(), &[3, 1]);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试 Sum 梯度累积
#[test]
fn test_sum_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "input")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))?;

    let sum = x.sum();
    let target = graph.input(&Tensor::new(&[20.0], &[1, 1]))?;
    let loss = sum.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = x.grad()?.unwrap().clone();

    loss.backward()?;
    let grad_second = x.grad()?.unwrap();
    assert_eq!(grad_second, &(&grad_first * 2.0));

    graph.zero_grad()?;
    loss.backward()?;
    let grad_after_clear = x.grad()?.unwrap();
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

// ==================== 节点创建 API 测试 ====================

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
    let result = inner.borrow_mut().create_sum_node(input, Some(2), None);

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
