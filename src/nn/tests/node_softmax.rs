/*
 * @Author       : 老董
 * @Description  : Softmax 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 Graph + Var API）→ basic [2,3], 数值稳定性, 极端差异, cannot_set_value
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ 均匀输入+全1 upstream→grad 全0；非均匀 [1,2,3]+[1,0,0]→grad sum=0；batch [2,3]+非单位 upstream→行 grad sum=0
 * 3. 端到端反向传播测试（高层）→ one-hot target，验证 row grad sums=0
 * 4. 梯度累积测试（高层）
 * 5. 动态形状测试（KEEP AS-IS）
 * 6. 新节点创建 API 测试（KEEP AS-IS）
 *
 * 关键点：Softmax 要求 2D 输入。softmax(x) 逐行归一化，每行和为 1。
 * VJP: grad_j = upstream_j * y_j - y_j * Σ(upstream_i * y_i)。行梯度 sum = 0。
 * 高层 API: VarActivationOps 的 .softmax()
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 Softmax 前向传播
///
/// 第一行 [1, 2, 3]：softmax ≈ [0.09, 0.24, 0.67]，每行 sum=1
/// 第二行 [1, 1, 1]：softmax = [1/3, 1/3, 1/3]
#[test]
fn test_softmax_forward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))
        .unwrap();
    let result = x.softmax();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
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
///
/// softmax([100,100,100]) = [1/3, 1/3, 1/3]（大数值→均匀）
#[test]
fn test_softmax_forward_numerical_stability() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[100.0, 100.0, 100.0], &[1, 3]))
        .unwrap();
    let result = x.softmax();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 1.0 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 1]], 1.0 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 2]], 1.0 / 3.0, epsilon = 1e-5);
}

/// 测试 Softmax 前向传播（极端差异）
///
/// softmax([-100, 100, -100]) ≈ [0, 1, 0]（一个主导→接近 one-hot）
#[test]
fn test_softmax_forward_extreme_difference() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[-100.0, 100.0, -100.0], &[1, 3]))
        .unwrap();
    let result = x.softmax();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-30);
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-30);
    assert_abs_diff_eq!(output[[0, 2]], 0.0, epsilon = 1e-30);
}

/// 测试 Softmax 节点不能直接设置值
#[test]
fn test_softmax_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))
        .unwrap();
    let result = x.softmax();

    let test_value = Tensor::new(&[0.1, 0.2, 0.7, 0.3, 0.3, 0.4], &[2, 3]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Softmax 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证梯度计算公式。
// VJP: grad_j = upstream_j * y_j - y_j * Σ(upstream_i * y_i)。行梯度 sum = 0。

/// 测试 Softmax VJP：均匀输入 + 全 1 upstream → grad 全 0
///
/// 均匀 softmax y=[1/3,1/3,1/3]，upstream=[1,1,1]
/// grad_j = 1*(1/3) - (1/3)*(1*1/3*3) = 1/3 - 1/3 = 0
#[test]
fn test_softmax_vjp_uniform_input_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("x"))
        .unwrap();
    let sm = inner
        .borrow_mut()
        .create_softmax_node(x.clone(), Some("sm"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3])))
        .unwrap();
    sm.forward_recursive(1, false).unwrap();

    let softmax_output = sm.value().expect("softmax 应有值");
    assert_abs_diff_eq!(softmax_output[[0, 0]], 1.0 / 3.0, epsilon = 1e-5);

    let upstream = Tensor::ones(&[1, 3]);
    let grad = sm.calc_grad_to_parent_index(0, &upstream)?;

    assert_eq!(grad.shape(), &[1, 3]);
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 2]], 0.0, epsilon = 1e-5);

    Ok(())
}

/// 测试 Softmax VJP：非均匀输入 [1,2,3] + 选择性 upstream [1,0,0] → grad sum=0
///
/// softmax([1,2,3]) ≈ [0.09, 0.24, 0.67]
/// grad_0 = y_0*(1-y_0) > 0，grad_1 = -y_1*y_0 < 0，grad_2 = -y_2*y_0 < 0
/// 行梯度和 = 0
#[test]
fn test_softmax_vjp_non_uniform_selective_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("x"))
        .unwrap();
    let sm = inner
        .borrow_mut()
        .create_softmax_node(x.clone(), Some("sm"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    sm.forward_recursive(1, false).unwrap();

    let upstream = Tensor::new(&[1.0, 0.0, 0.0], &[1, 3]);
    let grad = sm.calc_grad_to_parent_index(0, &upstream)?;

    assert_eq!(grad.shape(), &[1, 3]);
    assert!(grad[[0, 0]] > 0.0, "第一个梯度应该为正");
    assert!(grad[[0, 1]] < 0.0, "第二个梯度应该为负");
    assert!(grad[[0, 2]] < 0.0, "第三个梯度应该为负");

    let grad_sum: f32 = grad.data_as_slice().iter().sum();
    assert_abs_diff_eq!(grad_sum, 0.0, epsilon = 1e-5);

    Ok(())
}

/// 测试 Softmax VJP：batch [2,3] + 非单位 upstream → 每行 grad sum=0
#[test]
fn test_softmax_vjp_batch_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let sm = inner
        .borrow_mut()
        .create_softmax_node(x.clone(), Some("sm"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3])))
        .unwrap();
    sm.forward_recursive(1, false).unwrap();

    let upstream = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
    let grad = sm.calc_grad_to_parent_index(0, &upstream)?;

    assert_eq!(grad.shape(), &[2, 3]);

    let row0_sum: f32 = grad.data_as_slice()[0..3].iter().sum();
    let row1_sum: f32 = grad.data_as_slice()[3..6].iter().sum();
    assert_abs_diff_eq!(row0_sum, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(row1_sum, 0.0, epsilon = 1e-5);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 Softmax 端到端反向传播
///
/// result = softmax(x)，loss = MSE(result, target)，target 为 one-hot
/// 验证 input 梯度存在且每行 grad sum=0
#[test]
fn test_softmax_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))?;

    let result = x.softmax();
    let target = graph.input(&Tensor::new(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3]))?;
    let loss = result.mse_loss(&target)?;

    loss.forward().unwrap();
    let loss_val = loss.value().unwrap().unwrap();
    assert!(loss_val[[0, 0]] > 0.0);

    graph.zero_grad()?;
    loss.backward()?;

    let input_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    let row0_sum: f32 = input_grad.data_as_slice()[0..3].iter().sum();
    let row1_sum: f32 = input_grad.data_as_slice()[3..6].iter().sum();
    assert_abs_diff_eq!(row0_sum, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(row1_sum, 0.0, epsilon = 1e-5);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试 Softmax 梯度累积
#[test]
fn test_softmax_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3]))?;

    let result = x.softmax();
    let target = graph.input(&Tensor::new(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3]))?;
    let loss = result.mse_loss(&target)?;

    loss.forward().unwrap();

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

// ==================== 动态形状测试（KEEP AS-IS）====================

/// 测试 Softmax 节点的动态形状传播
#[test]
fn test_softmax_dynamic_shape_propagation() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarActivationOps;

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
    use crate::nn::Graph;
    use crate::nn::var_ops::VarActivationOps;

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
    use crate::nn::Graph;
    use crate::nn::var_ops::{VarActivationOps, VarLossOps};

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
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::normal_seeded(0.0, 1.0, &[5, 3], 42))
        .unwrap();
    target.set_value(&Tensor::zeros(&[5, 3])).unwrap();

    // 第二次训练：batch=5
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
}

// ==================== 节点创建 API 测试 ====================

#[test]
fn test_create_softmax_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();

    let softmax = inner
        .borrow_mut()
        .create_softmax_node(input.clone(), Some("softmax"))
        .unwrap();

    assert_eq!(softmax.shape(), vec![2, 3]);
    assert_eq!(softmax.name(), Some("softmax"));
    assert!(!softmax.is_leaf());
    assert_eq!(softmax.parents().len(), 1);
}

#[test]
fn test_create_softmax_node_requires_2d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 3D 输入应该失败
    let input_3d = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4], None)
        .unwrap();

    let result = inner.borrow_mut().create_softmax_node(input_3d, None);

    assert!(result.is_err());
    match result.unwrap_err() {
        GraphError::InvalidOperation(msg) => {
            assert!(msg.contains("2D"));
        }
        _ => panic!("应该返回 InvalidOperation 错误"),
    }
}

#[test]
fn test_create_softmax_node_preserves_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 测试 2D 形状保留
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let softmax = inner.borrow_mut().create_softmax_node(input, None).unwrap();
    assert_eq!(softmax.shape(), vec![5, 10]);
}

#[test]
fn test_create_softmax_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_softmax;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let softmax = inner.borrow_mut().create_softmax_node(input, None).unwrap();
        weak_softmax = Rc::downgrade(&softmax);

        assert!(weak_softmax.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_softmax.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
