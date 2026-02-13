/*
 * @Author       : 老董
 * @Description  : Add 节点单元测试
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

/// 测试 Add 前向传播（两个父节点）
#[test]
fn test_add_forward() {
    let graph = Graph::new();

    let p1 = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let p2 = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();
    let add = &p1 + &p2;

    add.forward().unwrap();

    // p1=[[1,2],[3,4]], p2=[[5,6],[7,8]] → add=[[6,8],[10,12]]
    let output = add.value().unwrap().unwrap();
    let expected = Tensor::new(&[6.0, 8.0, 10.0, 12.0], &[2, 2]);
    assert_eq!(output, expected);
}

/// 测试 Add 前向传播（三个父节点，底层 API）
#[test]
fn test_add_forward_three_parents() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p2"))
        .unwrap();
    let p3 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p3"))
        .unwrap();
    let add = inner
        .borrow_mut()
        .create_add_node(vec![p1.clone(), p2.clone(), p3.clone()], Some("add"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))
        .unwrap();
    p3.set_value(Some(&Tensor::new(&[10.0, 10.0, 10.0, 10.0], &[2, 2])))
        .unwrap();

    add.forward_recursive(1, false).unwrap();

    let output = add.value().unwrap();
    let expected = Tensor::new(&[16.0, 18.0, 20.0, 22.0], &[2, 2]);
    assert_eq!(output, expected);
}

/// 测试 Add 节点不能直接设置值（高层 Var API）
#[test]
fn test_add_cannot_set_value() {
    let graph = Graph::new();

    let p1 = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let p2 = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();
    let add = &p1 + &p2;

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let result = add.set_value(&test_value);
    assert!(result.is_err(), "Add 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证每个父节点的梯度计算公式，
// 独立验证每个父节点的梯度计算公式。

/// 测试 Add 对第一个父节点的 VJP
///
/// result = p1 + p2, ∂result/∂p1 = I → VJP: grad = upstream_grad
///
/// 使用 NodeInner::calc_grad_to_parent_index 直接测试梯度公式
#[test]
fn test_add_vjp_to_first_parent() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p2"))
        .unwrap();
    let add = inner
        .borrow_mut()
        .create_add_node(vec![p1.clone(), p2.clone()], Some("add"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))
        .unwrap();
    add.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = add.calc_grad_to_parent_index(0, &upstream_grad)?;

    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &upstream_grad);

    Ok(())
}

/// 测试 Add 对第二个父节点的 VJP
#[test]
fn test_add_vjp_to_second_parent() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p2"))
        .unwrap();
    let add = inner
        .borrow_mut()
        .create_add_node(vec![p1.clone(), p2.clone()], Some("add"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))
        .unwrap();
    add.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = add.calc_grad_to_parent_index(1, &upstream_grad)?;

    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &upstream_grad);

    Ok(())
}

/// 测试 Add VJP（非单位 upstream_grad）
#[test]
fn test_add_vjp_with_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p2"))
        .unwrap();
    let add = inner
        .borrow_mut()
        .create_add_node(vec![p1.clone(), p2.clone()], Some("add"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))
        .unwrap();
    add.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let grad_to_p1 = add.calc_grad_to_parent_index(0, &upstream_grad)?;
    let grad_to_p2 = add.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(&grad_to_p1, &upstream_grad);
    assert_eq!(&grad_to_p2, &upstream_grad);

    Ok(())
}

/// 测试 Add VJP（负数值，验证梯度与值无关）
#[test]
fn test_add_vjp_with_negative_values() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p2"))
        .unwrap();
    let add = inner
        .borrow_mut()
        .create_add_node(vec![p1.clone(), p2.clone()], Some("add"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[-1.0, -2.0, -3.0, -4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, -6.0, 7.0, -8.0], &[2, 2])))
        .unwrap();
    add.forward_recursive(1, false).unwrap();

    // 验证前向传播
    let output = add.value().unwrap();
    assert_eq!(output, Tensor::new(&[4.0, -8.0, 4.0, -12.0], &[2, 2]));

    let upstream_grad = Tensor::new(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]);
    let grad_to_p1 = add.calc_grad_to_parent_index(0, &upstream_grad)?;
    let grad_to_p2 = add.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(&grad_to_p1, &upstream_grad);
    assert_eq!(&grad_to_p2, &upstream_grad);

    Ok(())
}

/// 测试 Add VJP（三个父节点）
#[test]
fn test_add_vjp_three_parents() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p2"))
        .unwrap();
    let p3 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("p3"))
        .unwrap();
    let add = inner
        .borrow_mut()
        .create_add_node(vec![p1.clone(), p2.clone(), p3.clone()], Some("add"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))
        .unwrap();
    p3.set_value(Some(&Tensor::new(&[10.0, 10.0, 10.0, 10.0], &[2, 2])))
        .unwrap();
    add.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    let grad_0 = add.calc_grad_to_parent_index(0, &upstream_grad)?;
    let grad_1 = add.calc_grad_to_parent_index(1, &upstream_grad)?;
    let grad_2 = add.calc_grad_to_parent_index(2, &upstream_grad)?;

    assert_eq!(&grad_0, &upstream_grad);
    assert_eq!(&grad_1, &upstream_grad);
    assert_eq!(&grad_2, &upstream_grad);

    Ok(())
}

// ==================== 广播 VJP 测试（底层 API）====================

/// 测试 Add 广播 VJP：[3,4] + [1,4]
///
/// 对 [1,4] bias 的梯度需要沿 axis=0 求和
#[test]
fn test_add_broadcast_vjp() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let matrix = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("matrix"))
        .unwrap();
    let bias = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("bias"))
        .unwrap();
    let add = inner
        .borrow_mut()
        .create_add_node(vec![matrix.clone(), bias.clone()], Some("add"))
        .unwrap();

    matrix
        .set_value(Some(&Tensor::new(
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            &[3, 4],
        )))
        .unwrap();
    bias.set_value(Some(&Tensor::new(&[10., 20., 30., 40.], &[1, 4])))
        .unwrap();
    add.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[3, 4]);

    // 对 matrix [3,4] 的梯度：直接传递
    let grad_to_matrix = add.calc_grad_to_parent_index(0, &upstream_grad)?;
    assert_eq!(grad_to_matrix.shape(), &[3, 4]);
    assert_eq!(&grad_to_matrix, &upstream_grad);

    // 对 bias [1,4] 的梯度：沿 axis=0 求和 → [[3,3,3,3]]
    let grad_to_bias = add.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(grad_to_bias.shape(), &[1, 4]);
    assert_eq!(&grad_to_bias, &Tensor::new(&[3., 3., 3., 3.], &[1, 4]));

    Ok(())
}

/// 测试 Add 广播 VJP（非全 1 上游梯度）
#[test]
fn test_add_broadcast_vjp_non_unit() -> Result<(), GraphError> {
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
    let add = inner
        .borrow_mut()
        .create_add_node(vec![matrix.clone(), bias.clone()], Some("add"))
        .unwrap();

    matrix
        .set_value(Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])))
        .unwrap();
    bias.set_value(Some(&Tensor::new(&[10., 20., 30.], &[1, 3])))
        .unwrap();
    add.forward_recursive(1, false).unwrap();

    // upstream = [[1,2,3],[4,5,6]], sum(axis=0) = [[5,7,9]]
    // upstream = [[1,2,3],[4,5,6]], sum(axis=0) = [[5,7,9]]
    let upstream_grad = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);

    let grad_to_bias = add.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(grad_to_bias.shape(), &[1, 3]);
    assert_eq!(&grad_to_bias, &Tensor::new(&[5., 7., 9.], &[1, 3]));

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 Add 端到端反向传播：result = p1 + p2 → loss = MSE(result, target)
#[test]
fn test_add_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[2, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[2, 2], Init::Zeros, "p2")?;
    p1.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    p2.set_value(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))?;

    let result = &p1 + &p2;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // result = [[6,8],[10,12]], loss = mean([36,64,100,144]) = 86
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 86.0, epsilon = 1e-6);

    let p1_grad = p1.grad()?.expect("p1 应有 grad");
    let p2_grad = p2.grad()?.expect("p2 应有 grad");

    // ∂loss/∂p1 = ∂loss/∂p2 = result/2 = [[3,4],[5,6]]
    let expected_grad = Tensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);
    assert_eq!(&p1_grad, &expected_grad);
    assert_eq!(&p2_grad, &expected_grad);
    assert_eq!(&p1_grad, &p2_grad);

    Ok(())
}

/// 测试 Add 端到端反向传播（三个父节点，底层 create_add_node）
#[test]
fn test_add_backward_e2e_three_parents() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[2, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[2, 2], Init::Zeros, "p2")?;
    let p3 = graph.parameter(&[2, 2], Init::Zeros, "p3")?;
    p1.set_value(&Tensor::new(&[1.0; 4], &[2, 2]))?;
    p2.set_value(&Tensor::new(&[2.0; 4], &[2, 2]))?;
    p3.set_value(&Tensor::new(&[3.0; 4], &[2, 2]))?;

    // 三个节点相加需要底层 API（高层 + 运算符只支持两个操作数）
    let inner = graph.inner_rc();
    let add_node = inner.borrow_mut().create_add_node(
        vec![p1.node().clone(), p2.node().clone(), p3.node().clone()],
        Some("result"),
    )?;
    let result = crate::nn::Var::new_with_rc_graph(add_node, &inner);

    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // result = [[6,6],[6,6]], loss = 36
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 36.0, epsilon = 1e-6);

    // ∂loss/∂p_i = 2*6/4 = 3，三个参数梯度相同
    let expected_grad = Tensor::new(&[3.0; 4], &[2, 2]);
    assert_eq!(&p1.grad()?.unwrap(), &expected_grad);
    assert_eq!(&p2.grad()?.unwrap(), &expected_grad);
    assert_eq!(&p3.grad()?.unwrap(), &expected_grad);

    Ok(())
}

/// 测试 Add 广播端到端反向传播
#[test]
fn test_add_broadcast_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let matrix = graph.parameter(&[2, 3], Init::Zeros, "matrix")?;
    let bias = graph.parameter(&[1, 3], Init::Zeros, "bias")?;
    matrix.set_value(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]))?;
    bias.set_value(&Tensor::new(&[10., 20., 30.], &[1, 3]))?;

    let result = &matrix + &bias;
    let target = graph.input(&Tensor::zeros(&[2, 3]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    let matrix_grad = matrix.grad()?.expect("matrix 应有 grad");
    let bias_grad = bias.grad()?.expect("bias 应有 grad");
    assert_eq!(matrix_grad.shape(), &[2, 3]);
    assert_eq!(bias_grad.shape(), &[1, 3]);

    // bias 梯度 = sum(∂loss/∂result, axis=0) = [[25/3, 47/3, 69/3]]
    let expected_bias_grad = Tensor::new(&[25. / 3., 47. / 3., 69. / 3.], &[1, 3]);
    assert_abs_diff_eq!(&bias_grad, &expected_bias_grad, epsilon = 1e-4);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试梯度累积：多次 backward 不调用 zero_grad，梯度应累加
#[test]
fn test_add_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[2, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[2, 2], Init::Zeros, "p2")?;
    p1.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    p2.set_value(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))?;

    let result = &p1 + &p2;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // 第 1 次 backward（内部 ensure-forward，自动执行前向传播）
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = p1.grad()?.unwrap().clone();

    // 第 2 次 backward（不 zero_grad → 梯度累积）
    // 注意：backward 内部有 ensure-forward，无需手动调 forward
    loss.backward()?;
    let grad_second = p1.grad()?.unwrap();
    assert_eq!(&grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    loss.backward()?;
    let grad_after_clear = p1.grad()?.unwrap();
    assert_eq!(&grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 广播前向传播测试（高层 API）====================

/// 测试 Add 广播前向传播：[3,4] + [1,4] → [3,4]
#[test]
fn test_add_broadcast_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.input(&Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[3, 4],
    ))?;
    let p2 = graph.input(&Tensor::new(&[10., 20., 30., 40.], &[1, 4]))?;
    let add = &p1 + &p2;

    add.forward()?;

    let output = add.value()?.unwrap();
    let expected = Tensor::new(
        &[11., 22., 33., 44., 15., 26., 37., 48., 19., 30., 41., 52.],
        &[3, 4],
    );
    assert_eq!(output, expected);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 Add 节点的动态形状传播
#[test]
fn test_add_dynamic_shape_propagation() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建一个固定形状的参数
    let bias = graph
        .parameter(&[1, 16], crate::nn::Init::Zeros, "bias")
        .unwrap();

    // Add: h0 + bias
    let result = &h0 + &bias;

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "feature 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "feature 维度应该是 16");
}

/// 测试 Add 节点在不同 batch_size 下的前向计算
#[test]
fn test_add_dynamic_batch_forward() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]
    let bias = graph
        .parameter(&[1, 16], crate::nn::Init::Ones, "bias")
        .unwrap();

    // Add: h0 + bias
    let result = &h0 + &bias;

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

/// 测试 Add 节点在不同 batch_size 下的反向传播
#[test]
fn test_add_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var::ops::VarLossOps;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]
    let bias = graph
        .parameter(&[1, 4], crate::nn::Init::Ones, "bias")
        .unwrap();

    // Add: h0 + bias
    let result = &h0 + &bias;

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

// ==================== 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_add_node() {
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

    // 创建 Add 节点
    let add = inner
        .borrow_mut()
        .create_add_node(vec![a.clone(), b.clone()], Some("sum"))
        .unwrap();

    // 验证节点属性
    assert_eq!(add.shape(), vec![4, 8]);
    assert_eq!(add.name(), Some("sum"));
    assert!(!add.is_leaf()); // Add 不是叶子节点
    assert_eq!(add.parents().len(), 2);
}

#[test]
fn test_create_add_auto_name() {
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

    let add = inner
        .borrow_mut()
        .create_add_node(vec![a, b], None)
        .unwrap();

    let name = add.name().unwrap();
    assert!(name.contains("add"), "名称应包含 'add': {}", name);
}

#[test]
fn test_create_add_broadcast_shape() {
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

    let add = inner
        .borrow_mut()
        .create_add_node(vec![a, b], None)
        .unwrap();

    // 验证广播后的形状
    assert_eq!(add.shape(), vec![4, 8]);
}

#[test]
fn test_create_add_incompatible_shapes() {
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
    let result = inner.borrow_mut().create_add_node(vec![a, b], None);
    assert!(result.is_err());
}

#[test]
fn test_create_add_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_add;
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

        let add = inner
            .borrow_mut()
            .create_add_node(vec![a, b], None)
            .unwrap();
        weak_add = Rc::downgrade(&add);

        assert!(weak_add.upgrade().is_some());
        assert!(weak_a.upgrade().is_some()); // a 被 add 持有
    }
    // add 离开作用域，add 和其父节点都被释放
    assert!(weak_add.upgrade().is_none());
    assert!(weak_a.upgrade().is_none());
}

#[test]
fn test_create_add_multiple_parents() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 三个输入
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

    let add = inner
        .borrow_mut()
        .create_add_node(vec![a, b, c], None)
        .unwrap();

    assert_eq!(add.parents().len(), 3);
    assert_eq!(add.shape(), vec![4, 8]);
}

#[test]
fn test_create_add_insufficient_parents() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();

    // 只有一个父节点应该失败
    let result = inner.borrow_mut().create_add_node(vec![a], None);
    assert!(result.is_err());
}
