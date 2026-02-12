/*
 * @Author       : 老董
 * @Description  : Multiply 节点单元测试（逐元素乘法 Hadamard 积）
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

/// 测试 Multiply 前向传播（两个父节点）
#[test]
fn test_multiply_forward() {
    let graph = Graph::new();

    let p1 = graph.input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])).unwrap();
    let p2 = graph.input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])).unwrap();
    let mul = &p1 * &p2;

    mul.forward().unwrap();

    // p1=[[1,2],[3,4]], p2=[[5,6],[7,8]] → mul=[[5,12],[21,32]]
    let output = mul.value().unwrap().unwrap();
    let expected = Tensor::new(&[5.0, 12.0, 21.0, 32.0], &[2, 2]);
    assert_eq!(output, expected);
}

/// 测试 Multiply 节点不能直接设置值（高层 Var API）
#[test]
fn test_multiply_cannot_set_value() {
    let graph = Graph::new();

    let p1 = graph.input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])).unwrap();
    let p2 = graph.input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])).unwrap();
    let mul = &p1 * &p2;

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let result = mul.set_value(&test_value);
    assert!(result.is_err(), "Multiply 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证每个父节点的梯度计算公式。

/// 测试 Multiply 对第一个父节点的 VJP
///
/// result = p1 ⊙ p2, ∂result/∂p1 = diag(p2) → VJP: grad = upstream_grad ⊙ p2
///
/// 使用 NodeInner::calc_grad_to_parent_index 直接测试梯度公式
#[test]
fn test_multiply_vjp_to_left() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p1")).unwrap();
    let p2 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p2")).unwrap();
    let mul = inner
        .borrow_mut()
        .create_multiply_node(vec![p1.clone(), p2.clone()], Some("mul"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))).unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))).unwrap();
    mul.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = mul.calc_grad_to_parent_index(0, &upstream_grad)?;

    // grad_to_left = upstream ⊙ right = ones ⊙ [5,6,7,8] = [5,6,7,8]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]));

    Ok(())
}

/// 测试 Multiply 对第二个父节点的 VJP
///
/// result = p1 ⊙ p2, ∂result/∂p2 = diag(p1) → VJP: grad = upstream_grad ⊙ p1
#[test]
fn test_multiply_vjp_to_right() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p1")).unwrap();
    let p2 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p2")).unwrap();
    let mul = inner
        .borrow_mut()
        .create_multiply_node(vec![p1.clone(), p2.clone()], Some("mul"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))).unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))).unwrap();
    mul.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = mul.calc_grad_to_parent_index(1, &upstream_grad)?;

    // grad_to_right = upstream ⊙ left = ones ⊙ [1,2,3,4] = [1,2,3,4]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

    Ok(())
}

/// 测试 Multiply VJP（非单位 upstream_grad）
#[test]
fn test_multiply_vjp_with_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p1")).unwrap();
    let p2 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p2")).unwrap();
    let mul = inner
        .borrow_mut()
        .create_multiply_node(vec![p1.clone(), p2.clone()], Some("mul"))
        .unwrap();

    // left=[[1,2],[3,4]], right=[[5,6],[7,8]]
    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))).unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))).unwrap();
    mul.forward_recursive(1, false).unwrap();

    // upstream = [[2,2],[2,2]]
    let upstream_grad = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);

    // grad_to_left = upstream ⊙ right = [2,2,2,2] ⊙ [5,6,7,8] = [10,12,14,16]
    let grad_to_p1 = mul.calc_grad_to_parent_index(0, &upstream_grad)?;
    assert_eq!(&grad_to_p1, &Tensor::new(&[10.0, 12.0, 14.0, 16.0], &[2, 2]));

    // grad_to_right = upstream ⊙ left = [2,2,2,2] ⊙ [1,2,3,4] = [2,4,6,8]
    let grad_to_p2 = mul.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(&grad_to_p2, &Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]));

    Ok(())
}

/// 测试 Multiply VJP（负数值，验证 VJP 在负数值场景下的正确性）
#[test]
fn test_multiply_vjp_with_negative_values() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p1")).unwrap();
    let p2 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p2")).unwrap();
    let mul = inner
        .borrow_mut()
        .create_multiply_node(vec![p1.clone(), p2.clone()], Some("mul"))
        .unwrap();

    // left=[[-1,2],[-3,4]], right=[[5,-6],[7,-8]]
    let left_value = Tensor::new(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, -6.0, 7.0, -8.0], &[2, 2]);
    p1.set_value(Some(&left_value)).unwrap();
    p2.set_value(Some(&right_value)).unwrap();
    mul.forward_recursive(1, false).unwrap();

    // 验证前向传播：[[-1*5, 2*-6], [-3*7, 4*-8]] = [[-5,-12],[-21,-32]]
    let output = mul.value().unwrap();
    assert_eq!(output, Tensor::new(&[-5.0, -12.0, -21.0, -32.0], &[2, 2]));

    let upstream_grad = Tensor::ones(&[2, 2]);

    // grad_to_left = upstream ⊙ right = [[5,-6],[7,-8]]
    let grad_to_p1 = mul.calc_grad_to_parent_index(0, &upstream_grad)?;
    assert_eq!(&grad_to_p1, &right_value);

    // grad_to_right = upstream ⊙ left = [[-1,2],[-3,4]]
    let grad_to_p2 = mul.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(&grad_to_p2, &left_value);

    Ok(())
}

/// 测试 Multiply VJP（含零值）
///
/// 零值是重要边界情况：0*x=0，但梯度仍应正确传播
#[test]
fn test_multiply_backward_with_zero_value() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p1")).unwrap();
    let p2 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p2")).unwrap();
    let mul = inner
        .borrow_mut()
        .create_multiply_node(vec![p1.clone(), p2.clone()], Some("mul"))
        .unwrap();

    // left=[[0,2],[3,0]], right=[[5,0],[0,8]]
    let left_value = Tensor::new(&[0.0, 2.0, 3.0, 0.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, 0.0, 0.0, 8.0], &[2, 2]);
    p1.set_value(Some(&left_value)).unwrap();
    p2.set_value(Some(&right_value)).unwrap();
    mul.forward_recursive(1, false).unwrap();

    // 验证前向传播：[[0*5, 2*0], [3*0, 0*8]] = [[0,0],[0,0]]
    let output = mul.value().unwrap();
    assert_eq!(output, Tensor::zeros(&[2, 2]));

    let upstream_grad = Tensor::ones(&[2, 2]);

    // 即使输出全为 0，梯度仍应正确计算
    // grad_to_left = upstream ⊙ right = [[5,0],[0,8]]
    let grad_to_p1 = mul.calc_grad_to_parent_index(0, &upstream_grad)?;
    assert_eq!(&grad_to_p1, &right_value);

    // grad_to_right = upstream ⊙ left = [[0,2],[3,0]]
    let grad_to_p2 = mul.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(&grad_to_p2, &left_value);

    Ok(())
}

/// 测试 Multiply 梯度计算父节点索引越界时报错
#[test]
fn test_multiply_backward_invalid_parent_index() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p1")).unwrap();
    let p2 = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("p2")).unwrap();
    let mul = inner
        .borrow_mut()
        .create_multiply_node(vec![p1.clone(), p2.clone()], Some("mul"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))).unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))).unwrap();
    mul.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let result = mul.calc_grad_to_parent_index(2, &upstream_grad); // 索引 2 越界
    assert!(result.is_err(), "索引越界应返回错误");

    Ok(())
}

// ==================== 广播 VJP 测试（底层 API）====================

/// 测试 Multiply 广播 VJP：[2,3] ⊙ [1,3]
///
/// 对 [1,3] scale 的梯度需要沿 axis=0 求和
#[test]
fn test_multiply_broadcast_vjp() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let matrix = inner.borrow_mut().create_basic_input_node(&[2, 3], Some("matrix")).unwrap();
    let scale = inner.borrow_mut().create_basic_input_node(&[1, 3], Some("scale")).unwrap();
    let mul = inner
        .borrow_mut()
        .create_multiply_node(vec![matrix.clone(), scale.clone()], Some("mul"))
        .unwrap();

    // matrix = [[1,2,3],[4,5,6]], scale = [[2,3,4]]
    matrix
        .set_value(Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])))
        .unwrap();
    scale.set_value(Some(&Tensor::new(&[2., 3., 4.], &[1, 3]))).unwrap();
    mul.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 3]);

    // 对 matrix [2,3] 的梯度：upstream ⊙ scale（广播后）= [[2,3,4],[2,3,4]]
    let grad_to_matrix = mul.calc_grad_to_parent_index(0, &upstream_grad)?;
    assert_eq!(grad_to_matrix.shape(), &[2, 3]);
    assert_eq!(&grad_to_matrix, &Tensor::new(&[2., 3., 4., 2., 3., 4.], &[2, 3]));

    // 对 scale [1,3] 的梯度：sum(upstream ⊙ matrix, axis=0) = sum([[1,2,3],[4,5,6]], axis=0) = [[5,7,9]]
    let grad_to_scale = mul.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(grad_to_scale.shape(), &[1, 3]);
    assert_eq!(&grad_to_scale, &Tensor::new(&[5., 7., 9.], &[1, 3]));

    Ok(())
}

/// 测试 Multiply 广播 VJP（非全 1 上游梯度）
///
/// 实际训练中，upstream_grad 几乎不会是全 1，而是由链式法则层层计算得到的各种值。
/// 此测试验证 sum_to_shape 在这种真实场景下的正确性。
#[test]
fn test_multiply_broadcast_vjp_non_unit() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let matrix = inner.borrow_mut().create_basic_input_node(&[2, 3], Some("matrix")).unwrap();
    let scale = inner.borrow_mut().create_basic_input_node(&[1, 3], Some("scale")).unwrap();
    let mul = inner
        .borrow_mut()
        .create_multiply_node(vec![matrix.clone(), scale.clone()], Some("mul"))
        .unwrap();

    // matrix = [[1,2,3],[4,5,6]], scale = [[2,3,4]]
    matrix
        .set_value(Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])))
        .unwrap();
    scale.set_value(Some(&Tensor::new(&[2., 3., 4.], &[1, 3]))).unwrap();
    mul.forward_recursive(1, false).unwrap();

    // upstream = [[1,2,3],[4,5,6]]（非全 1）
    let upstream_grad = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);

    // 对 scale [1,3] 的梯度：sum(upstream ⊙ matrix, axis=0)
    // [[1*1,2*2,3*3],[4*4,5*5,6*6]] = [[1,4,9],[16,25,36]]
    // sum(axis=0) = [[17,29,45]]
    let grad_to_scale = mul.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(grad_to_scale.shape(), &[1, 3]);
    assert_eq!(&grad_to_scale, &Tensor::new(&[17., 29., 45.], &[1, 3]));

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 Multiply 端到端反向传播：result = p1 ⊙ p2 → loss = MSE(result, target)
#[test]
fn test_multiply_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[2, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[2, 2], Init::Zeros, "p2")?;
    p1.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    p2.set_value(&Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2]))?;

    let result = &p1 * &p2;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // result = [[2,4],[6,8]], loss = mean([4,16,36,64]) = 120/4 = 30
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 30.0, epsilon = 1e-6);

    let p1_grad = p1.grad()?.expect("p1 应有 grad");
    let p2_grad = p2.grad()?.expect("p2 应有 grad");

    // ∂loss/∂result = 2*(result - target)/n = result/2 = [[1,2],[3,4]]
    // ∂loss/∂p1 = ∂loss/∂result ⊙ p2 = [[1,2],[3,4]] ⊙ [[2,2],[2,2]] = [[2,4],[6,8]]
    let expected_p1_grad = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    assert_eq!(&p1_grad, &expected_p1_grad);

    // ∂loss/∂p2 = ∂loss/∂result ⊙ p1 = [[1,2],[3,4]] ⊙ [[1,2],[3,4]] = [[1,4],[9,16]]
    let expected_p2_grad = Tensor::new(&[1.0, 4.0, 9.0, 16.0], &[2, 2]);
    assert_eq!(&p2_grad, &expected_p2_grad);

    Ok(())
}

/// 测试 Multiply 广播端到端反向传播
#[test]
fn test_multiply_broadcast_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let features = graph.parameter(&[2, 3], Init::Zeros, "features")?;
    let scale = graph.parameter(&[1, 3], Init::Zeros, "scale")?;
    features.set_value(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]))?;
    scale.set_value(&Tensor::ones(&[1, 3]))?;

    let result = &features * &scale;
    let target = graph.input(&Tensor::zeros(&[2, 3]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    let features_grad = features.grad()?.expect("features 应有 grad");
    let scale_grad = scale.grad()?.expect("scale 应有 grad");
    assert_eq!(features_grad.shape(), &[2, 3]);
    assert_eq!(scale_grad.shape(), &[1, 3]);

    // result = [[1,2,3],[4,5,6]]
    // ∂loss/∂result = 2*(result - target)/n = result/3
    //               = [[1/3, 2/3, 1], [4/3, 5/3, 2]]
    //
    // ∂loss/∂scale = sum(∂loss/∂result ⊙ features, axis=0)
    //             = sum([[1*1/3, 2*2/3, 3*1], [4*4/3, 5*5/3, 6*2]], axis=0)
    //             = sum([[1/3, 4/3, 3], [16/3, 25/3, 12]], axis=0)
    //             = [[(1+16)/3, (4+25)/3, (3+12)]]
    //             = [[17/3, 29/3, 15]]
    let expected_scale_grad = Tensor::new(&[17. / 3., 29. / 3., 15.], &[1, 3]);
    assert_abs_diff_eq!(&scale_grad, &expected_scale_grad, epsilon = 1e-4);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试梯度累积：多次 backward 不调用 zero_grad，梯度应累加
#[test]
fn test_multiply_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[2, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[2, 2], Init::Zeros, "p2")?;
    p1.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    p2.set_value(&Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2]))?;

    let result = &p1 * &p2;
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

/// 测试 Multiply 广播前向传播：[2,3] ⊙ [1,3] → [2,3]
#[test]
fn test_multiply_broadcast_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let matrix = graph.input(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]))?;
    let scale = graph.input(&Tensor::new(&[2., 3., 4.], &[1, 3]))?;
    let result = &matrix * &scale;

    result.forward()?;

    // result = [[1*2,2*3,3*4],[4*2,5*3,6*4]] = [[2,6,12],[8,15,24]]
    let output = result.value()?.unwrap();
    let expected = Tensor::new(&[2., 6., 12., 8., 15., 24.], &[2, 3]);
    assert_eq!(output, expected);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 Multiply 节点的动态形状传播
#[test]
fn test_multiply_dynamic_shape_propagation() {
    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建一个固定形状的参数
    let scale = graph
        .parameter(&[1, 16], Init::Ones, "scale")
        .unwrap();

    // Multiply: h0 * scale
    let result = &h0 * &scale;

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "feature 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "feature 维度应该是 16");
}

/// 测试 Multiply 节点在不同 batch_size 下的前向计算
#[test]
fn test_multiply_dynamic_batch_forward() {
    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::ones(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]
    let scale = graph
        .parameter(&[1, 16], Init::Ones, "scale")
        .unwrap();

    // Multiply: h0 * scale（结果全零，因为 h0 是零）
    let result = &h0 * &scale;

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 16], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::ones(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 16], "第二次 forward: batch=8");
}

/// 测试 Multiply 节点在不同 batch_size 下的反向传播
#[test]
fn test_multiply_dynamic_batch_backward() {
    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::ones(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]
    let scale = graph
        .parameter(&[1, 4], Init::Ones, "scale")
        .unwrap();

    // Multiply: h0 * scale
    let result = &h0 * &scale;

    // 创建目标和损失
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    // 验证 scale 有梯度
    let scale_grad1 = scale.grad().unwrap().unwrap();
    assert_eq!(scale_grad1.shape(), &[1, 4], "scale 梯度形状应为 [1, 4]");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::ones(&[6, 8])).unwrap();
    target.set_value(&Tensor::zeros(&[6, 4])).unwrap();

    // 第二次 forward + backward：batch=6
    loss.forward().unwrap();
    assert_eq!(
        result.value().unwrap().unwrap().shape(),
        &[6, 4],
        "第二次 forward: batch=6"
    );
    loss.backward().unwrap();

    // 验证 scale 仍有正确形状的梯度
    let scale_grad2 = scale.grad().unwrap().unwrap();
    assert_eq!(scale_grad2.shape(), &[1, 4], "scale 梯度形状仍应为 [1, 4]");
}

// ==================== 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_multiply_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], Some("a"))
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], Some("b"))
        .unwrap();

    let mul = inner
        .borrow_mut()
        .create_multiply_node(vec![a.clone(), b.clone()], Some("prod"))
        .unwrap();

    assert_eq!(mul.shape(), vec![4, 8]);
    assert_eq!(mul.name(), Some("prod"));
    assert!(!mul.is_leaf());
    assert_eq!(mul.parents().len(), 2);
}

#[test]
fn test_create_multiply_auto_name() {
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

    let mul = inner
        .borrow_mut()
        .create_multiply_node(vec![a, b], None)
        .unwrap();

    let name = mul.name().unwrap();
    assert!(name.contains("multiply"), "名称应包含 'multiply': {}", name);
}

#[test]
fn test_create_multiply_broadcast_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 8], None)
        .unwrap();

    let mul = inner
        .borrow_mut()
        .create_multiply_node(vec![a, b], None)
        .unwrap();

    assert_eq!(mul.shape(), vec![4, 8]);
}

#[test]
fn test_create_multiply_incompatible_shapes() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 7], None)
        .unwrap();

    let result = inner.borrow_mut().create_multiply_node(vec![a, b], None);
    assert!(result.is_err());
}

#[test]
fn test_create_multiply_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_mul;
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

        let mul = inner
            .borrow_mut()
            .create_multiply_node(vec![a, b], None)
            .unwrap();
        weak_mul = Rc::downgrade(&mul);

        assert!(weak_mul.upgrade().is_some());
        assert!(weak_a.upgrade().is_some());
    }
    assert!(weak_mul.upgrade().is_none());
    assert!(weak_a.upgrade().is_none());
}

#[test]
fn test_create_multiply_requires_two_parents() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();

    let result = inner.borrow_mut().create_multiply_node(vec![a], None);
    assert!(result.is_err());
}
