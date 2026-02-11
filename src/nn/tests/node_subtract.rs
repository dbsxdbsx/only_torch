/*
 * @Author       : 老董
 * @Description  : Subtract 节点单元测试（逐元素减法）
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）→ 底层 create_* API（文件末尾）
 * 2. 前向传播测试 → 高层 Graph + Var API
 * 3. VJP 单元测试（calc_grad_to_parent）→ 底层 NodeInner + calc_grad_to_parent_index
 * 4. 端到端反向传播测试 → 高层 Graph + Var API
 * 5. 梯度累积测试 → 高层 Graph + Var API
 * 6. 广播测试 → 混合（高层前向/e2e + 底层 VJP）
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 Subtract 前向传播（两个父节点）
#[test]
fn test_subtract_forward() {
    let graph = Graph::new();

    let left = graph
        .input(&Tensor::new(&[6.0, 8.0, 12.0, 20.0, 30.0, 42.0], &[2, 3]))
        .unwrap();
    let right = graph
        .input(&Tensor::new(&[2.0, 4.0, 3.0, 5.0, 6.0, 7.0], &[2, 3]))
        .unwrap();
    let result = &left - &right;

    result.forward().unwrap();

    // left=[6,8,12,20,30,42], right=[2,4,3,5,6,7]
    // result = [4,4,9,15,24,35]
    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(&[4.0, 4.0, 9.0, 15.0, 24.0, 35.0], &[2, 3]);
    assert_eq!(output, expected);
}

/// 测试 Subtract 节点不能直接设置值（高层 Var API）
#[test]
fn test_subtract_cannot_set_value() {
    let graph = Graph::new();

    let left = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let right = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();
    let sub = &left - &right;

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let result = sub.set_value(&test_value);
    assert!(result.is_err(), "Subtract 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证每个父节点的梯度计算公式。

/// 测试 Subtract 对 left（被减数）的 VJP
///
/// result = left - right, ∂result/∂left = I → VJP: grad = upstream_grad
///
/// 使用 NodeInner::calc_grad_to_parent_index 直接测试梯度公式
#[test]
fn test_subtract_vjp_to_left() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let left = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("left"))
        .unwrap();
    let right = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("right"))
        .unwrap();
    let sub = inner
        .borrow_mut()
        .create_subtract_node(vec![left.clone(), right.clone()], Some("sub"))
        .unwrap();

    left.set_value(Some(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2])))
        .unwrap();
    right
        .set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    sub.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = sub.calc_grad_to_parent_index(0, &upstream_grad)?;

    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &upstream_grad);

    Ok(())
}

/// 测试 Subtract 对 right（减数）的 VJP
///
/// result = left - right, ∂result/∂right = -I → VJP: grad = -upstream_grad
#[test]
fn test_subtract_vjp_to_right() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let left = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("left"))
        .unwrap();
    let right = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("right"))
        .unwrap();
    let sub = inner
        .borrow_mut()
        .create_subtract_node(vec![left.clone(), right.clone()], Some("sub"))
        .unwrap();

    left.set_value(Some(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2])))
        .unwrap();
    right
        .set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    sub.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = sub.calc_grad_to_parent_index(1, &upstream_grad)?;

    // grad_to_right = -upstream = [[-1,-1],[-1,-1]]
    assert_eq!(grad.shape(), &[2, 2]);
    let expected = Tensor::new(&[-1.0, -1.0, -1.0, -1.0], &[2, 2]);
    assert_eq!(&grad, &expected);

    Ok(())
}

/// 测试 Subtract VJP（非全 1 上游梯度）
#[test]
fn test_subtract_vjp_with_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let left = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("left"))
        .unwrap();
    let right = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("right"))
        .unwrap();
    let sub = inner
        .borrow_mut()
        .create_subtract_node(vec![left.clone(), right.clone()], Some("sub"))
        .unwrap();

    left.set_value(Some(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2])))
        .unwrap();
    right
        .set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    sub.forward_recursive(1, false).unwrap();

    // upstream_grad = [[2,3],[4,5]]
    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad_to_left = sub.calc_grad_to_parent_index(0, &upstream_grad)?;
    let grad_to_right = sub.calc_grad_to_parent_index(1, &upstream_grad)?;

    // grad_to_left = upstream = [[2,3],[4,5]]
    assert_eq!(&grad_to_left, &upstream_grad);
    // grad_to_right = -upstream = [[-2,-3],[-4,-5]]
    let expected_right = Tensor::new(&[-2.0, -3.0, -4.0, -5.0], &[2, 2]);
    assert_eq!(&grad_to_right, &expected_right);

    Ok(())
}

/// 测试 Subtract VJP（负数值，验证梯度公式在负数下仍正确）
#[test]
fn test_subtract_vjp_with_negative_values() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let left = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("left"))
        .unwrap();
    let right = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("right"))
        .unwrap();
    let sub = inner
        .borrow_mut()
        .create_subtract_node(vec![left.clone(), right.clone()], Some("sub"))
        .unwrap();

    left.set_value(Some(&Tensor::new(&[-1.0, -2.0, -3.0, -4.0], &[2, 2])))
        .unwrap();
    right
        .set_value(Some(&Tensor::new(&[5.0, -6.0, 7.0, -8.0], &[2, 2])))
        .unwrap();
    sub.forward_recursive(1, false).unwrap();

    // 验证前向传播: [-1-5, -2-(-6), -3-7, -4-(-8)] = [-6, 4, -10, 4]
    let output = sub.value().unwrap();
    assert_eq!(output, Tensor::new(&[-6.0, 4.0, -10.0, 4.0], &[2, 2]));

    let upstream_grad = Tensor::new(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]);
    let grad_to_left = sub.calc_grad_to_parent_index(0, &upstream_grad)?;
    let grad_to_right = sub.calc_grad_to_parent_index(1, &upstream_grad)?;
    // left 梯度 = upstream_grad
    assert_eq!(&grad_to_left, &upstream_grad);
    // right 梯度 = -upstream_grad = [1, -2, 3, -4]
    let expected_right = Tensor::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2]);
    assert_eq!(&grad_to_right, &expected_right);

    Ok(())
}

// ==================== 广播 VJP 测试（底层 API）====================

/// 测试 Subtract 广播 VJP：[2,3] - [1,3]
///
/// 对 [1,3] bias 的梯度需要先取负再沿 axis=0 求和
#[test]
fn test_subtract_broadcast_vjp() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let matrix = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("matrix"))
        .unwrap();
    let bias = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("bias"))
        .unwrap();
    let sub = inner
        .borrow_mut()
        .create_subtract_node(vec![matrix.clone(), bias.clone()], Some("sub"))
        .unwrap();

    matrix
        .set_value(Some(&Tensor::new(&[10., 20., 30., 40., 50., 60.], &[2, 3])))
        .unwrap();
    bias.set_value(Some(&Tensor::new(&[1., 2., 3.], &[1, 3])))
        .unwrap();
    sub.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 3]);

    // 对 matrix [2,3] 的梯度：直接传递
    let grad_to_matrix = sub.calc_grad_to_parent_index(0, &upstream_grad)?;
    assert_eq!(grad_to_matrix.shape(), &[2, 3]);
    assert_eq!(&grad_to_matrix, &upstream_grad);

    // 对 bias [1,3] 的梯度：-upstream，然后沿 axis=0 求和
    // -[[1,1,1],[1,1,1]] = [[-1,-1,-1],[-1,-1,-1]]
    // sum(axis=0) = [[-2,-2,-2]]
    let grad_to_bias = sub.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(grad_to_bias.shape(), &[1, 3]);
    assert_eq!(&grad_to_bias, &Tensor::new(&[-2., -2., -2.], &[1, 3]));

    Ok(())
}

/// 测试 Subtract 广播 VJP（非全 1 上游梯度）
#[test]
fn test_subtract_broadcast_vjp_non_unit() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let matrix = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("matrix"))
        .unwrap();
    let bias = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("bias"))
        .unwrap();
    let sub = inner
        .borrow_mut()
        .create_subtract_node(vec![matrix.clone(), bias.clone()], Some("sub"))
        .unwrap();

    matrix
        .set_value(Some(&Tensor::new(&[10., 20., 30., 40., 50., 60.], &[2, 3])))
        .unwrap();
    bias.set_value(Some(&Tensor::new(&[1., 2., 3.], &[1, 3])))
        .unwrap();
    sub.forward_recursive(1, false).unwrap();

    // upstream = [[1,2,3],[4,5,6]]
    // -upstream = [[-1,-2,-3],[-4,-5,-6]], sum(axis=0) = [[-5,-7,-9]]
    let upstream_grad = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);

    let grad_to_bias = sub.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(grad_to_bias.shape(), &[1, 3]);
    assert_eq!(&grad_to_bias, &Tensor::new(&[-5., -7., -9.], &[1, 3]));

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 Subtract 端到端反向传播：result = left - right → loss = MSE(result, target)
#[test]
fn test_subtract_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let left = graph.parameter(&[2, 2], Init::Zeros, "left")?;
    let right = graph.parameter(&[2, 2], Init::Zeros, "right")?;
    left.set_value(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2]))?;
    right.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;

    let result = &left - &right;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // result = [[3,4],[5,6]], loss = mean([9,16,25,36]) = 86/4 = 21.5
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 21.5, epsilon = 1e-6);

    let left_grad = left.grad()?.expect("left 应有 grad");
    let right_grad = right.grad()?.expect("right 应有 grad");

    // ∂loss/∂result = 2*(result - target)/n = result/2 = [[1.5,2],[2.5,3]]
    // ∂loss/∂left = ∂loss/∂result = [[1.5,2],[2.5,3]]
    // ∂loss/∂right = -∂loss/∂result = [[-1.5,-2],[-2.5,-3]]
    let expected_left_grad = Tensor::new(&[1.5, 2.0, 2.5, 3.0], &[2, 2]);
    let expected_right_grad = Tensor::new(&[-1.5, -2.0, -2.5, -3.0], &[2, 2]);
    assert_abs_diff_eq!(&left_grad, &expected_left_grad, epsilon = 1e-6);
    assert_abs_diff_eq!(&right_grad, &expected_right_grad, epsilon = 1e-6);

    Ok(())
}

/// 测试 Subtract 广播端到端反向传播
#[test]
fn test_subtract_broadcast_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let features = graph.parameter(&[2, 3], Init::Zeros, "features")?;
    let bias = graph.parameter(&[1, 3], Init::Zeros, "bias")?;
    features.set_value(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]))?;
    bias.set_value(&Tensor::ones(&[1, 3]))?;

    let result = &features - &bias;
    let target = graph.input(&Tensor::zeros(&[2, 3]))?;
    let loss = result.mse_loss(&target)?;

    // result = [[0,1,2],[3,4,5]], target = 0
    graph.zero_grad()?;
    loss.backward()?;

    let features_grad = features.grad()?.expect("features 应有 grad");
    let bias_grad = bias.grad()?.expect("bias 应有 grad");
    assert_eq!(features_grad.shape(), &[2, 3]);
    assert_eq!(bias_grad.shape(), &[1, 3]);

    // ∂loss/∂result = 2*(result - target)/n = result/3
    //               = [[0, 1/3, 2/3], [1, 4/3, 5/3]]
    //
    // ∂loss/∂features = ∂loss/∂result = [[0, 1/3, 2/3], [1, 4/3, 5/3]]
    // ∂loss/∂bias = -sum(∂loss/∂result, axis=0)
    //             = -[[(0+1), (1/3+4/3), (2/3+5/3)]]
    //             = -[[1, 5/3, 7/3]]
    //             = [[-1, -5/3, -7/3]]
    let expected_bias_grad = Tensor::new(&[-1., -5. / 3., -7. / 3.], &[1, 3]);
    assert_abs_diff_eq!(&bias_grad, &expected_bias_grad, epsilon = 1e-4);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试梯度累积：多次 backward 不调用 zero_grad，梯度应累加
#[test]
fn test_subtract_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let left = graph.parameter(&[2, 2], Init::Zeros, "left")?;
    let right = graph.parameter(&[2, 2], Init::Zeros, "right")?;
    left.set_value(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2]))?;
    right.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;

    let result = &left - &right;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // 第 1 次 backward（内部 ensure-forward，自动执行前向传播）
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = left.grad()?.unwrap().clone();

    // 第 2 次 backward（不 zero_grad → 梯度累积）
    // 注意：backward 内部有 ensure-forward，无需手动调 forward
    loss.backward()?;
    let grad_second = left.grad()?.unwrap();
    assert_eq!(&grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    loss.backward()?;
    let grad_after_clear = left.grad()?.unwrap();
    assert_eq!(&grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 广播前向传播测试（高层 API）====================

/// 测试 Subtract 广播前向传播：[2,3] - [1,3] → [2,3]
#[test]
fn test_subtract_broadcast_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let matrix = graph.input(&Tensor::new(&[10., 20., 30., 40., 50., 60.], &[2, 3]))?;
    let bias = graph.input(&Tensor::new(&[1., 2., 3.], &[1, 3]))?;
    let result = &matrix - &bias;

    result.forward()?;

    // matrix = [[10,20,30],[40,50,60]], bias = [[1,2,3]]
    // result = [[9,18,27],[39,48,57]]
    let output = result.value()?.unwrap();
    let expected = Tensor::new(&[9., 18., 27., 39., 48., 57.], &[2, 3]);
    assert_eq!(output, expected);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 Subtract 节点的动态形状传播
#[test]
fn test_subtract_dynamic_shape_propagation() {
    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建一个固定形状的参数
    let bias = graph
        .parameter(&[1, 16], crate::nn::Init::Zeros, "bias")
        .unwrap();

    // Subtract: h0 - bias
    let result = &h0 - &bias;

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "feature 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "feature 维度应该是 16");
}

/// 测试 Subtract 节点在不同 batch_size 下的前向计算
#[test]
fn test_subtract_dynamic_batch_forward() {
    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]
    let bias = graph
        .parameter(&[1, 16], crate::nn::Init::Ones, "bias")
        .unwrap();

    // Subtract: h0 - bias
    let result = &h0 - &bias;

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 16], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 16], "第二次 forward: batch=8");
}

/// 测试 Subtract 节点在不同 batch_size 下的反向传播
#[test]
fn test_subtract_dynamic_batch_backward() {
    use crate::nn::var_ops::VarLossOps;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]
    let bias = graph
        .parameter(&[1, 4], crate::nn::Init::Ones, "bias")
        .unwrap();

    // Subtract: h0 - bias
    let result = &h0 - &bias;

    // 创建目标和损失
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    // 验证 bias 有梯度
    let bias_grad1 = bias.grad().unwrap().unwrap();
    assert_eq!(bias_grad1.shape(), &[1, 4], "bias 梯度形状应为 [1, 4]");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[6, 8])).unwrap();
    target.set_value(&Tensor::zeros(&[6, 4])).unwrap();

    // 第二次 forward + backward：batch=6
    loss.forward().unwrap();
    assert_eq!(
        result.value().unwrap().unwrap().shape(),
        &[6, 4],
        "第二次 forward: batch=6"
    );
    loss.backward().unwrap();

    // 验证 bias 仍有正确形状的梯度
    let bias_grad2 = bias.grad().unwrap().unwrap();
    assert_eq!(bias_grad2.shape(), &[1, 4], "bias 梯度形状仍应为 [1, 4]");
}

// ==================== 方案 C：新节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_subtract_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 创建两个输入节点作为父节点
    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], Some("a"))
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], Some("b"))
        .unwrap();

    // 创建 Subtract 节点
    let sub = inner
        .borrow_mut()
        .create_subtract_node(vec![a.clone(), b.clone()], Some("diff"))
        .unwrap();

    // 验证节点属性
    assert_eq!(sub.shape(), vec![4, 8]);
    assert_eq!(sub.name(), Some("diff"));
    assert!(!sub.is_leaf());
    assert_eq!(sub.parents().len(), 2);
}

#[test]
fn test_create_subtract_auto_name() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();

    let sub = inner
        .borrow_mut()
        .create_subtract_node(vec![a, b], None)
        .unwrap();

    let name = sub.name().unwrap();
    assert!(name.contains("subtract"), "名称应包含 'subtract': {}", name);
}

#[test]
fn test_create_subtract_broadcast_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 不同形状的父节点（支持广播）
    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 8], None) // 可以广播到 [4, 8]
        .unwrap();

    let sub = inner
        .borrow_mut()
        .create_subtract_node(vec![a, b], None)
        .unwrap();

    // 验证广播后的形状
    assert_eq!(sub.shape(), vec![4, 8]);
}

#[test]
fn test_create_subtract_incompatible_shapes() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 7], None) // 无法广播
        .unwrap();

    // 应该失败
    let result = inner.borrow_mut().create_subtract_node(vec![a, b], None);
    assert!(result.is_err());
}

#[test]
fn test_create_subtract_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_sub;
    let weak_a;
    {
        let a = inner
            .borrow_mut()
            .create_basic_input_node(&[4, 8], None)
            .unwrap();
        let b = inner
            .borrow_mut()
            .create_basic_input_node(&[4, 8], None)
            .unwrap();
        weak_a = Rc::downgrade(&a);

        let sub = inner
            .borrow_mut()
            .create_subtract_node(vec![a, b], None)
            .unwrap();
        weak_sub = Rc::downgrade(&sub);

        assert!(weak_sub.upgrade().is_some());
        assert!(weak_a.upgrade().is_some());
    }
    // sub 离开作用域，sub 和其父节点都被释放
    assert!(weak_sub.upgrade().is_none());
    assert!(weak_a.upgrade().is_none());
}

#[test]
fn test_create_subtract_requires_two_parents() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();

    // 只有一个父节点应该失败
    let result = inner.borrow_mut().create_subtract_node(vec![a], None);
    assert!(result.is_err());

    // 三个父节点也应该失败
    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();
    let c = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();
    let result = inner.borrow_mut().create_subtract_node(vec![a, b, c], None);
    assert!(result.is_err());
}
