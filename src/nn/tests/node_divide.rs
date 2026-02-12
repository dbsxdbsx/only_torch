/*
 * @Author       : 老董
 * @Description  : Divide 节点单元测试（逐元素除法）
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）→ 底层 create_* API（文件末尾）
 * 2. 前向传播测试 → 高层 Graph + Var API
 * 3. VJP 单元测试（calc_grad_to_parent_index）→ 底层 NodeInner
 * 4. 端到端反向传播测试 → 高层 Graph + Var API
 * 5. 梯度累积测试 → 高层 Graph + Var API
 * 6. 广播测试 → 混合（高层前向/e2e + 底层 VJP）
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 Divide 前向传播
#[test]
fn test_divide_forward() {
    let graph = Graph::new();

    let left = graph
        .input(&Tensor::new(&[6.0, 8.0, 12.0, 20.0, 30.0, 42.0], &[2, 3]))
        .unwrap();
    let right = graph
        .input(&Tensor::new(&[2.0, 4.0, 3.0, 5.0, 6.0, 7.0], &[2, 3]))
        .unwrap();
    let result = &left / &right;

    result.forward().unwrap();

    // left=[6,8,12,20,30,42], right=[2,4,3,5,6,7]
    // result = [3, 2, 4, 4, 5, 6]
    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(&[3.0, 2.0, 4.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(output, expected);
}

/// 测试除数为零时的行为
///
/// 当前实现：Tensor 层检测到除零会 panic
/// 这是一种安全策略，防止产生 Inf/NaN 导致后续计算异常
#[test]
#[should_panic(expected = "作为除数的张量中存在为零元素")]
fn test_divide_by_zero_panics() {
    let graph = Graph::new();

    let left = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let right = graph
        .input(&Tensor::new(&[1.0, 0.0, 1.0, 1.0], &[2, 2]))
        .unwrap();
    let result = &left / &right;

    // 前向传播应该 panic
    result.forward().unwrap();
}

/// 测试 Divide 节点不能直接设置值（高层 Var API）
#[test]
fn test_divide_cannot_set_value() {
    let graph = Graph::new();

    let left = graph
        .input(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2]))
        .unwrap();
    let right = graph
        .input(&Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]))
        .unwrap();
    let div = &left / &right;

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let result = div.set_value(&test_value);
    assert!(result.is_err(), "Divide 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证每个父节点的梯度计算公式。

/// 测试 Divide 对 left（被除数）的 VJP
///
/// result = left / right, ∂result/∂left = 1/right → VJP: grad = upstream_grad / right
///
/// 使用 NodeInner::calc_grad_to_parent_index 直接测试梯度公式
#[test]
fn test_divide_vjp_to_left() -> Result<(), GraphError> {
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
    let result = inner
        .borrow_mut()
        .create_divide_node(vec![left.clone(), right.clone()], Some("result"))
        .unwrap();

    left.set_value(Some(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2])))
        .unwrap();
    right
        .set_value(Some(&Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2])))
        .unwrap();
    result.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = result.calc_grad_to_parent_index(0, &upstream_grad)?;

    // grad_to_left = upstream / right = ones / [2,3,4,5] = [0.5, 0.333, 0.25, 0.2]
    assert_eq!(grad.shape(), &[2, 2]);
    let expected = Tensor::new(&[0.5, 1.0 / 3.0, 0.25, 0.2], &[2, 2]);
    assert_abs_diff_eq!(grad, expected, epsilon = 1e-6);

    Ok(())
}

/// 测试 Divide 对 right（除数）的 VJP
///
/// result = left / right, ∂result/∂right = -left/right²
/// → VJP: grad = -upstream_grad * left / right²
#[test]
fn test_divide_vjp_to_right() -> Result<(), GraphError> {
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
    let result = inner
        .borrow_mut()
        .create_divide_node(vec![left.clone(), right.clone()], Some("result"))
        .unwrap();

    left.set_value(Some(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2])))
        .unwrap();
    right
        .set_value(Some(&Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2])))
        .unwrap();
    result.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = result.calc_grad_to_parent_index(1, &upstream_grad)?;

    // grad_to_right = -upstream * left / right²
    //               = -1 * [4,6,8,10] / [4,9,16,25]
    //               = [-1, -0.667, -0.5, -0.4]
    assert_eq!(grad.shape(), &[2, 2]);
    let expected = Tensor::new(&[-1.0, -6.0 / 9.0, -0.5, -0.4], &[2, 2]);
    assert_abs_diff_eq!(grad, expected, epsilon = 1e-6);

    Ok(())
}

/// 测试 Divide VJP（非单位 upstream_grad）
#[test]
fn test_divide_vjp_with_non_unit_upstream() -> Result<(), GraphError> {
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
    let result = inner
        .borrow_mut()
        .create_divide_node(vec![left.clone(), right.clone()], Some("result"))
        .unwrap();

    left.set_value(Some(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2])))
        .unwrap();
    right
        .set_value(Some(&Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2])))
        .unwrap();
    result.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // grad_to_left = upstream / right = [1,2,3,4] / [2,3,4,5] = [0.5, 0.667, 0.75, 0.8]
    let grad_to_left = result.calc_grad_to_parent_index(0, &upstream_grad)?;
    let expected_left = Tensor::new(&[0.5, 2.0 / 3.0, 0.75, 0.8], &[2, 2]);
    assert_abs_diff_eq!(grad_to_left, expected_left, epsilon = 1e-6);

    // grad_to_right = -upstream * left / right²
    //               = -[1,2,3,4] * [4,6,8,10] / [4,9,16,25]
    //               = -[4,12,24,40] / [4,9,16,25]
    //               = [-1, -4/3, -3/2, -8/5]
    let grad_to_right = result.calc_grad_to_parent_index(1, &upstream_grad)?;
    let expected_right = Tensor::new(&[-1.0, -4.0 / 3.0, -1.5, -1.6], &[2, 2]);
    assert_abs_diff_eq!(grad_to_right, expected_right, epsilon = 1e-6);

    Ok(())
}

/// 测试 Divide VJP（负数值）
#[test]
fn test_divide_vjp_with_negative_values() -> Result<(), GraphError> {
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
    let result = inner
        .borrow_mut()
        .create_divide_node(vec![left.clone(), right.clone()], Some("result"))
        .unwrap();

    left.set_value(Some(&Tensor::new(&[-4.0, 6.0, -8.0, 10.0], &[2, 2])))
        .unwrap();
    right
        .set_value(Some(&Tensor::new(&[2.0, -3.0, 4.0, -5.0], &[2, 2])))
        .unwrap();
    result.forward_recursive(1, false).unwrap();

    // 验证前向传播: [-4/2, 6/-3, -8/4, 10/-5] = [-2, -2, -2, -2]
    let output = result.value().unwrap();
    assert_eq!(output, Tensor::new(&[-2.0, -2.0, -2.0, -2.0], &[2, 2]));

    let upstream_grad = Tensor::new(&[1.0, -1.0, 2.0, -2.0], &[2, 2]);

    // grad_to_left = upstream / right = [1,-1,2,-2] / [2,-3,4,-5]
    //             = [0.5, 1/3, 0.5, 0.4]
    let grad_to_left = result.calc_grad_to_parent_index(0, &upstream_grad)?;
    let expected_left = Tensor::new(&[0.5, 1.0 / 3.0, 0.5, 0.4], &[2, 2]);
    assert_abs_diff_eq!(grad_to_left, expected_left, epsilon = 1e-6);

    // grad_to_right = -upstream * left / right²
    //               = -[1,-1,2,-2] * [-4,6,-8,10] / [4,9,16,25]
    //               = -[-4,-6,-16,-20] / [4,9,16,25]
    //               = [4,6,16,20] / [4,9,16,25]
    //               = [1, 2/3, 1, 4/5]
    let grad_to_right = result.calc_grad_to_parent_index(1, &upstream_grad)?;
    let expected_right = Tensor::new(&[1.0, 2.0 / 3.0, 1.0, 4.0 / 5.0], &[2, 2]);
    assert_abs_diff_eq!(grad_to_right, expected_right, epsilon = 1e-6);

    Ok(())
}

/// 测试 Divide VJP 越界索引
#[test]
fn test_divide_backward_invalid_parent_index() -> Result<(), GraphError> {
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
    let result = inner
        .borrow_mut()
        .create_divide_node(vec![left.clone(), right.clone()], Some("result"))
        .unwrap();

    left.set_value(Some(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2])))
        .unwrap();
    right
        .set_value(Some(&Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2])))
        .unwrap();
    result.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad_result = result.calc_grad_to_parent_index(2, &upstream_grad); // 索引 2 越界
    assert!(grad_result.is_err());

    Ok(())
}

// ==================== 广播 VJP 测试（底层 API）====================

/// 测试 Divide 广播 VJP：[2, 3] / [1, 3]
///
/// 对 [1,3] scale 的梯度需要沿 axis=0 求和
#[test]
fn test_divide_broadcast_vjp() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let matrix = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("matrix"))
        .unwrap();
    let scale = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("scale"))
        .unwrap();
    let result = inner
        .borrow_mut()
        .create_divide_node(vec![matrix.clone(), scale.clone()], Some("result"))
        .unwrap();

    // matrix = [[2,6,12], [4,9,16]]
    // scale = [[2, 3, 4]]
    matrix
        .set_value(Some(&Tensor::new(
            &[2., 6., 12., 4., 9., 16.],
            &[2, 3],
        )))
        .unwrap();
    scale
        .set_value(Some(&Tensor::new(&[2., 3., 4.], &[1, 3])))
        .unwrap();
    result.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 3]);

    // 对 matrix [2,3] 的梯度：upstream / scale（广播后）
    // upstream / [[2,3,4],[2,3,4]] = [[0.5,0.333,0.25],[0.5,0.333,0.25]]
    let grad_to_matrix = result.calc_grad_to_parent_index(0, &upstream_grad)?;
    assert_eq!(grad_to_matrix.shape(), &[2, 3]);
    let expected_matrix_grad =
        Tensor::new(&[0.5, 1.0 / 3.0, 0.25, 0.5, 1.0 / 3.0, 0.25], &[2, 3]);
    assert_abs_diff_eq!(grad_to_matrix, expected_matrix_grad, epsilon = 1e-6);

    // 对 scale [1,3] 的梯度：-upstream * matrix / scale²，然后沿 axis=0 求和
    // -[[1,1,1],[1,1,1]] * [[2,6,12],[4,9,16]] / [[4,9,16],[4,9,16]]
    // = [[-0.5,-0.667,-0.75],[-1,-1,-1]]
    // sum(axis=0) = [[-1.5,-1.667,-1.75]]
    let grad_to_scale = result.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(grad_to_scale.shape(), &[1, 3]);
    let expected_scale_grad = Tensor::new(&[-1.5, -5.0 / 3.0, -1.75], &[1, 3]);
    assert_abs_diff_eq!(grad_to_scale, expected_scale_grad, epsilon = 1e-6);

    Ok(())
}

/// 测试 Divide 广播 VJP（非全 1 上游梯度）
///
/// 实际训练中，upstream_grad 几乎不会是全 1，而是由链式法则层层计算得到的各种值。
/// 此测试验证 sum_to_shape 在这种真实场景下的正确性。
#[test]
fn test_divide_broadcast_vjp_non_unit() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let matrix = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("matrix"))
        .unwrap();
    let scale = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("scale"))
        .unwrap();
    let result = inner
        .borrow_mut()
        .create_divide_node(vec![matrix.clone(), scale.clone()], Some("result"))
        .unwrap();

    // matrix = [[1,2,3], [4,5,6]]
    // scale = [[1, 2, 3]]
    matrix
        .set_value(Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])))
        .unwrap();
    scale
        .set_value(Some(&Tensor::new(&[1., 2., 3.], &[1, 3])))
        .unwrap();
    result.forward_recursive(1, false).unwrap();

    // upstream_grad = [[1,2,3], [4,5,6]]（非全 1）
    let upstream_grad = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);

    // 对 scale [1,3] 的梯度：-upstream * matrix / scale²，然后沿 axis=0 求和
    // -[[1,2,3],[4,5,6]] * [[1,2,3],[4,5,6]] / [[1,4,9],[1,4,9]]
    // = -[[1,4,9],[16,25,36]] / [[1,4,9],[1,4,9]]
    // = [[-1,-1,-1],[-16,-6.25,-4]]
    // sum = [[-17,-7.25,-5]]
    let grad_to_scale = result.calc_grad_to_parent_index(1, &upstream_grad)?;
    assert_eq!(grad_to_scale.shape(), &[1, 3]);
    let expected = Tensor::new(&[-17., -7.25, -5.], &[1, 3]);
    assert_abs_diff_eq!(grad_to_scale, expected, epsilon = 1e-6);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 Divide 端到端反向传播：result = left / right → loss = MSE(result, target)
#[test]
fn test_divide_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let left = graph.parameter(&[2, 2], Init::Zeros, "left")?;
    let right = graph.parameter(&[2, 2], Init::Zeros, "right")?;
    left.set_value(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2]))?;
    right.set_value(&Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2]))?;

    let result = &left / &right;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // result = [[2,3],[4,5]]
    // loss = mean([4,9,16,25]) = 54/4 = 13.5
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 13.5, epsilon = 1e-6);

    let left_grad = left.grad()?.expect("left 应有 grad");
    let right_grad = right.grad()?.expect("right 应有 grad");
    assert_eq!(left_grad.shape(), &[2, 2]);
    assert_eq!(right_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = result/2 = [[1,1.5],[2,2.5]]
    // ∂loss/∂left = ∂loss/∂result / right = [[1,1.5],[2,2.5]] / [[2,2],[2,2]]
    //             = [[0.5, 0.75],[1, 1.25]]
    let expected_left_grad = Tensor::new(&[0.5, 0.75, 1.0, 1.25], &[2, 2]);
    assert_abs_diff_eq!(left_grad, &expected_left_grad, epsilon = 1e-6);

    Ok(())
}

/// 测试 Divide 广播端到端反向传播
///
/// 验证广播在完整训练场景中的正确性
#[test]
fn test_divide_broadcast_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 模拟特征归一化：result = features / scale
    // features [2,3], scale [1,3]
    let features = graph.parameter(&[2, 3], Init::Zeros, "features")?;
    let scale = graph.parameter(&[1, 3], Init::Zeros, "scale")?;
    features.set_value(&Tensor::new(&[2., 4., 6., 4., 8., 12.], &[2, 3]))?;
    scale.set_value(&Tensor::new(&[2., 2., 2.], &[1, 3]))?;

    let result = &features / &scale;
    let target = graph.input(&Tensor::zeros(&[2, 3]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    let features_grad = features.grad()?.expect("features 应有 grad");
    let scale_grad = scale.grad()?.expect("scale 应有 grad");
    assert_eq!(features_grad.shape(), &[2, 3], "features 梯度形状应为 [2,3]");
    assert_eq!(scale_grad.shape(), &[1, 3], "scale 梯度形状应为 [1,3]");

    // result = [[1,2,3], [2,4,6]]
    // ∂loss/∂result = 2*(result - target)/n = result/3
    //               = [[1/3, 2/3, 1], [2/3, 4/3, 2]]
    //
    // ∂loss/∂features = ∂loss/∂result / scale
    //                 = [[1/3, 2/3, 1], [2/3, 4/3, 2]] / [[2,2,2],[2,2,2]]
    //                 = [[1/6, 1/3, 0.5], [1/3, 2/3, 1]]
    let expected_features_grad = Tensor::new(
        &[1.0 / 6.0, 1.0 / 3.0, 0.5, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        &[2, 3],
    );
    assert_abs_diff_eq!(features_grad, &expected_features_grad, epsilon = 1e-4);

    Ok(())
}

// ==================== 广播前向传播测试（高层 API）====================

/// 测试 Divide 广播前向传播：[2, 3] / [1, 3] → [2, 3]
#[test]
fn test_divide_broadcast_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let matrix = graph.input(&Tensor::new(&[2., 6., 12., 4., 15., 24.], &[2, 3]))?;
    let scale = graph.input(&Tensor::new(&[2., 3., 4.], &[1, 3]))?;
    let result = &matrix / &scale;

    result.forward()?;

    // matrix = [[2,6,12], [4,15,24]]
    // scale = [[2, 3, 4]]
    // result = [[1,2,3], [2,5,6]]
    let output = result.value()?.unwrap();
    let expected = Tensor::new(&[1., 2., 3., 2., 5., 6.], &[2, 3]);
    assert_eq!(output, expected);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试梯度累积：多次 backward 不调用 zero_grad，梯度应累加
#[test]
fn test_divide_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let left = graph.parameter(&[2, 2], Init::Zeros, "left")?;
    let right = graph.parameter(&[2, 2], Init::Zeros, "right")?;
    left.set_value(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2]))?;
    right.set_value(&Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2]))?;

    let result = &left / &right;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // 第 1 次 backward（内部 ensure-forward，自动执行前向传播）
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = left.grad()?.unwrap().clone();

    // 第 2 次 backward（不 zero_grad → 梯度累积）
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

// ==================== 动态形状测试 ====================

/// 测试 Divide 节点的动态形状传播
#[test]
fn test_divide_dynamic_shape_propagation() {
    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::ones(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建一个固定形状的参数（非零，避免除零）
    let scale = graph
        .parameter(&[1, 16], Init::Ones, "scale")
        .unwrap();

    // Divide: h0 / scale
    let result = &h0 / &scale;

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "feature 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "feature 维度应该是 16");
}

/// 测试 Divide 节点在不同 batch_size 下的前向计算
#[test]
fn test_divide_dynamic_batch_forward() {
    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::ones(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]
    let scale = graph
        .parameter(&[1, 16], Init::Ones, "scale")
        .unwrap();

    // Divide: h0 / scale（结果全零，因为 h0 是零）
    let result = &h0 / &scale;

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

/// 测试 Divide 节点在不同 batch_size 下的反向传播
#[test]
fn test_divide_dynamic_batch_backward() {
    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::ones(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]
    let scale = graph
        .parameter(&[1, 4], Init::Ones, "scale")
        .unwrap();

    // Divide: h0 / scale
    let result = &h0 / &scale;

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
fn test_create_divide_node() {
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

    let div = inner
        .borrow_mut()
        .create_divide_node(vec![a.clone(), b.clone()], Some("quot"))
        .unwrap();

    assert_eq!(div.shape(), vec![4, 8]);
    assert_eq!(div.name(), Some("quot"));
    assert!(!div.is_leaf());
    assert_eq!(div.parents().len(), 2);
}

#[test]
fn test_create_divide_auto_name() {
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

    let div = inner
        .borrow_mut()
        .create_divide_node(vec![a, b], None)
        .unwrap();

    let name = div.name().unwrap();
    assert!(name.contains("divide"), "名称应包含 'divide': {}", name);
}

#[test]
fn test_create_divide_broadcast_shape() {
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

    let div = inner
        .borrow_mut()
        .create_divide_node(vec![a, b], None)
        .unwrap();

    assert_eq!(div.shape(), vec![4, 8]);
}

#[test]
fn test_create_divide_incompatible_shapes() {
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

    let result = inner.borrow_mut().create_divide_node(vec![a, b], None);
    assert!(result.is_err());
}

#[test]
fn test_create_divide_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_div;
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

        let div = inner
            .borrow_mut()
            .create_divide_node(vec![a, b], None)
            .unwrap();
        weak_div = Rc::downgrade(&div);

        assert!(weak_div.upgrade().is_some());
        assert!(weak_a.upgrade().is_some());
    }
    assert!(weak_div.upgrade().is_none());
    assert!(weak_a.upgrade().is_none());
}

#[test]
fn test_create_divide_requires_two_parents() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();

    let result = inner.borrow_mut().create_divide_node(vec![a], None);
    assert!(result.is_err());
}
