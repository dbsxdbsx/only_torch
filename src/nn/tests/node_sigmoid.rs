/*
 * @Author       : 老董
 * @Description  : Sigmoid 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 Graph + Var API）→ basic forward + edge cases + cannot_set_value
 * 2. VJP 单元测试（calc_grad_to_parent_index）→ 底层 NodeInner
 * 3. 端到端反向传播测试（高层 Graph + Var API）
 * 4. 梯度累积测试（高层 Graph + Var API）
 * 5. 动态形状测试（已有）
 * 6. 新节点创建 API 测试（已有）
 *
 * 梯度公式：
 *   sigmoid(x) = 1/(1+e^(-x))
 *   dy/dx = y*(1-y)，其中 y = sigmoid(x)
 *   VJP: grad_to_parent = upstream_grad ⊙ y ⊙ (1-y)
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 Sigmoid 前向传播
///
/// sigmoid([0.5, -1.0, 0.0, 2.0]) = [0.62245935, 0.26894143, 0.5, 0.88079703]
#[test]
fn test_sigmoid_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))
        .unwrap();
    let result = x.sigmoid();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.62245935, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 0.26894143, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 0.88079703, epsilon = 1e-6);
}

/// 测试 Sigmoid 前向传播（边界值）
///
/// sigmoid(0)=0.5, sigmoid(10)≈1, sigmoid(-10)≈0
#[test]
fn test_sigmoid_forward_edge_cases() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[0.0, 10.0, -10.0, 0.001], &[1, 4]))
        .unwrap();
    let result = x.sigmoid();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 0.5, epsilon = 1e-6); // sigmoid(0) = 0.5
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-4); // sigmoid(10) ≈ 1
    assert_abs_diff_eq!(output[[0, 2]], 0.0, epsilon = 1e-4); // sigmoid(-10) ≈ 0
    assert_abs_diff_eq!(output[[0, 3]], 0.50025, epsilon = 1e-4); // sigmoid(0.001) ≈ 0.50025
}

/// 测试 Sigmoid 节点不能直接设置值
#[test]
fn test_sigmoid_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.sigmoid();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Sigmoid 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证梯度计算公式。
// sigmoid'(x) = y*(1-y)，VJP: grad = upstream ⊙ y ⊙ (1-y)

/// 测试 Sigmoid VJP（全 1 上游梯度）
///
/// sigmoid([0.5, -1.0, 0.0, 2.0]) = [0.6225, 0.2689, 0.5, 0.8808]
/// y*(1-y) = [0.23500371, 0.19661194, 0.25, 0.10499363]
#[test]
fn test_sigmoid_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let sigmoid = inner
        .borrow_mut()
        .create_sigmoid_node(x.clone(), Some("sigmoid"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    sigmoid.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[2, 2]);
    let grad = sigmoid.calc_grad_to_parent_index(0, &upstream_grad)?;

    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 0.23500371, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.19661194, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 0.10499363, epsilon = 1e-6);

    Ok(())
}

/// 测试 Sigmoid VJP（非单位上游梯度）
///
/// grad = upstream ⊙ y ⊙ (1-y)
#[test]
fn test_sigmoid_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let sigmoid = inner
        .borrow_mut()
        .create_sigmoid_node(x.clone(), Some("sigmoid"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))
        .unwrap();
    sigmoid.forward_recursive(1, false).unwrap();

    // upstream_grad = [[2,3],[4,5]]
    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let grad = sigmoid.calc_grad_to_parent_index(0, &upstream_grad)?;

    // y*(1-y) = [0.23500371, 0.19661194, 0.25, 0.10499363]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0 * 0.23500371, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 3.0 * 0.19661194, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0]], 4.0 * 0.25, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1]], 5.0 * 0.10499363, epsilon = 1e-5);

    Ok(())
}

/// 测试 Sigmoid 梯度计算（接近饱和区）
///
/// 当输入绝对值很大时，sigmoid 接近 0 或 1，梯度接近 0（梯度消失）
#[test]
fn test_sigmoid_vjp_saturation() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("x"))
        .unwrap();
    let sigmoid = inner
        .borrow_mut()
        .create_sigmoid_node(x.clone(), Some("sigmoid"))
        .unwrap();

    // 饱和区输入：大正数和大负数
    x.set_value(Some(&Tensor::new(&[5.0, -5.0], &[1, 2])))
        .unwrap();
    sigmoid.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[1, 2]);
    let grad = sigmoid.calc_grad_to_parent_index(0, &upstream_grad)?;

    // sigmoid(5) ≈ 0.9933，sigmoid(-5) ≈ 0.0067
    // y*(1-y) ≈ 0.0066（两者都接近 0）
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-2);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-2);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 Sigmoid 端到端反向传播：result = sigmoid(x) → loss = MSE(result, target)
#[test]
fn test_sigmoid_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.sigmoid();
    let target = graph.input(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // result = sigmoid(input) = [[0.6225, 0.2689], [0.5, 0.8808]]
    // diff = result - target = [[0.1225, -0.2311], [0.0, 0.3808]]
    // loss = mean(diff²)
    let diff: [f32; 4] = [
        0.62245935 - 0.5,
        0.26894143 - 0.5,
        0.5 - 0.5,
        0.88079703 - 0.5,
    ];
    let expected_loss =
        (diff[0].powi(2) + diff[1].powi(2) + diff[2].powi(2) + diff[3].powi(2)) / 4.0;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, expected_loss, epsilon = 1e-5);

    let input_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = (result - target)/2
    // ∂loss/∂input = ∂loss/∂result * sigmoid'(x) = ∂loss/∂result * y * (1-y)
    // input[1,0] = 0，sigmoid(0) = 0.5，target = 0.5，所以 diff = 0，grad 应为 0
    assert!(input_grad[[0, 0]].abs() > 1e-6);
    assert!(input_grad[[0, 1]].abs() > 1e-6);
    assert_abs_diff_eq!(input_grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert!(input_grad[[1, 1]].abs() > 1e-6);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试 Sigmoid 梯度累积
///
/// 验证语义：参数的 grad 在多次 backward 之间累积，直到调用 zero_grad()。
#[test]
fn test_sigmoid_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]))?;

    let result = x.sigmoid();
    let target = graph.input(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // 第 1 次反向传播
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = x.grad()?.unwrap().clone();

    // 第 2 次反向传播（梯度累积）
    loss.backward()?;
    let grad_second = x.grad()?.unwrap();
    assert_eq!(&grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    loss.backward()?;
    let grad_after_clear = x.grad()?.unwrap();
    assert_eq!(&grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 Sigmoid 节点的动态形状传播
#[test]
fn test_sigmoid_dynamic_shape_propagation() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建 Sigmoid: h0 -> sigmoid(h0) -> [?, 16]
    use crate::nn::var_ops::VarActivationOps;
    let result = h0.sigmoid();

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 Sigmoid 节点在不同 batch_size 下的前向计算
#[test]
fn test_sigmoid_dynamic_batch_forward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarActivationOps;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // Sigmoid: h0 -> sigmoid(h0)
    let result = h0.sigmoid();

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

/// 测试 Sigmoid 节点在不同 batch_size 下的反向传播
#[test]
fn test_sigmoid_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::{VarActivationOps, VarLossOps};

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]

    // Sigmoid: h0 -> sigmoid(h0) -> [?, 4]
    let result = h0.sigmoid();

    // 创建目标和损失
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

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
}

// ==================== 方案 C：新节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_sigmoid_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();

    let sigmoid = inner
        .borrow_mut()
        .create_sigmoid_node(input.clone(), Some("sigmoid"))
        .unwrap();

    assert_eq!(sigmoid.shape(), vec![2, 3]);
    assert_eq!(sigmoid.name(), Some("sigmoid"));
    assert!(!sigmoid.is_leaf());
    assert_eq!(sigmoid.parents().len(), 1);
}

#[test]
fn test_create_sigmoid_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 测试各种形状都正确保留（节点必须是 2-4 维）
    let input_2d = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 10], None)
        .unwrap();
    let sigmoid_2d = inner
        .borrow_mut()
        .create_sigmoid_node(input_2d, None)
        .unwrap();
    assert_eq!(sigmoid_2d.shape(), vec![3, 10]);

    let input_3d = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4], None)
        .unwrap();
    let sigmoid_3d = inner
        .borrow_mut()
        .create_sigmoid_node(input_3d, None)
        .unwrap();
    assert_eq!(sigmoid_3d.shape(), vec![2, 3, 4]);

    let input_4d = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4, 5], None)
        .unwrap();
    let sigmoid_4d = inner
        .borrow_mut()
        .create_sigmoid_node(input_4d, None)
        .unwrap();
    assert_eq!(sigmoid_4d.shape(), vec![2, 3, 4, 5]);
}

#[test]
fn test_create_sigmoid_node_chain() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 测试连续两个 sigmoid
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], None)
        .unwrap();
    let sig1 = inner
        .borrow_mut()
        .create_sigmoid_node(input, None)
        .unwrap();
    let sig2 = inner
        .borrow_mut()
        .create_sigmoid_node(sig1.clone(), None)
        .unwrap();

    assert_eq!(sig2.shape(), vec![2, 2]);
    assert_eq!(sig2.parents().len(), 1);
}

#[test]
fn test_create_sigmoid_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_sigmoid;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let sigmoid = inner
            .borrow_mut()
            .create_sigmoid_node(input, None)
            .unwrap();
        weak_sigmoid = Rc::downgrade(&sigmoid);

        assert!(weak_sigmoid.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_sigmoid.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
