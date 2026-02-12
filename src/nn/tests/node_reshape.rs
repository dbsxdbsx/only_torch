/*
 * @Author       : 老董
 * @Description  : Reshape 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ basic 2x3→3x2; 行优先; 列/行向量; batch; 组合; 链式; 错误处理; cannot_set_value
 * 2. VJP 单元测试（底层）→ unit upstream; 非 unit upstream
 * 3. E2E 反向传播（高层）→ sigmoid chain; single sample
 * 4. 梯度累积
 * 5. 动态形状（KEEP AS-IS）
 * 6. Create API（KEEP AS-IS）
 *
 * Reshape 只改变形状不改变数据。VJP: grad_to_input = reshape(upstream, input_shape)
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试基本的 reshape 功能：2x3 -> 3x2
#[test]
fn test_reshape_basic_2x3_to_3x2() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let reshaped = x.reshape(&[3, 2]).unwrap();

    reshaped.forward().unwrap();

    let output = reshaped.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 2]);
    // [1,2,3;4,5,6] reshape → [1,2;3,4;5,6]（数据不变，按行优先重排形状）
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(output, expected);
}

/// 测试 reshape 保持元素顺序：行优先
#[test]
fn test_reshape_preserves_row_major_order() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]))
        .unwrap();
    let reshaped = x.reshape(&[2, 3]).unwrap();

    reshaped.forward().unwrap();

    let output = reshaped.value().unwrap().unwrap();
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(output, expected);
}

/// 测试 reshape 到列向量
#[test]
fn test_reshape_to_column_vector() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let reshaped = x.reshape(&[6, 1]).unwrap();

    reshaped.forward().unwrap();

    let output = reshaped.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[6, 1]);
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6, 1]);
    assert_eq!(output, expected);
}

/// 测试 reshape 到行向量
#[test]
fn test_reshape_to_row_vector() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]))
        .unwrap();
    let reshaped = x.reshape(&[1, 6]).unwrap();

    reshaped.forward().unwrap();

    let output = reshaped.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 6]);
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    assert_eq!(output, expected);
}

/// 测试批量输入的前向传播
#[test]
fn test_reshape_batch_forward() {
    let graph = Graph::new();

    // [4, 6] -> [2, 12]
    let input_data = Tensor::normal_seeded(0.0, 1.0, &[4, 6], 42);
    let x = graph.input(&input_data).unwrap();
    let reshaped = x.reshape(&[2, 12]).unwrap();

    reshaped.forward().unwrap();

    let output = reshaped.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 12]);

    // 验证元素总数和顺序不变
    for i in 0..24 {
        let in_row = i / 6;
        let in_col = i % 6;
        let out_row = i / 12;
        let out_col = i % 12;
        assert_abs_diff_eq!(
            input_data[[in_row, in_col]],
            output[[out_row, out_col]],
            epsilon = 1e-6
        );
    }
}

/// 测试 Reshape 与 Add 组合
#[test]
fn test_reshape_with_add() {
    let graph = Graph::new();

    // a: [2,3], b: [3,2]
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]))
        .unwrap();

    // a reshape 为 [3,2] 后与 b 相加
    let a_reshaped = a.reshape(&[3, 2]).unwrap();
    let sum = &a_reshaped + &b;

    sum.forward().unwrap();

    let output = sum.value().unwrap().unwrap();
    // a reshape: [[1,2],[3,4],[5,6]] + b: [[0.1,0.2],[0.3,0.4],[0.5,0.6]]
    let expected = Tensor::new(&[1.1, 2.2, 3.3, 4.4, 5.5, 6.6], &[3, 2]);
    assert_abs_diff_eq!(output, expected, epsilon = 1e-6);
}

/// 测试多次 Reshape（连续变换）
#[test]
fn test_reshape_chain() {
    let graph = Graph::new();

    let data = Tensor::normal_seeded(0.0, 1.0, &[2, 6], 42);
    let x = graph.input(&data).unwrap();

    // [2,6] -> [3,4] -> [4,3] -> [6,2]
    let r1 = x.reshape(&[3, 4]).unwrap();
    let r2 = r1.reshape(&[4, 3]).unwrap();
    let r3 = r2.reshape(&[6, 2]).unwrap();

    r3.forward().unwrap();

    let output = r3.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[6, 2]);

    // 数据不变，仅形状改变
    for i in 0..12 {
        let in_row = i / 6;
        let in_col = i % 6;
        let out_row = i / 2;
        let out_col = i % 2;
        assert_abs_diff_eq!(
            data[[in_row, in_col]],
            output[[out_row, out_col]],
            epsilon = 1e-6
        );
    }
}

/// 测试形状不匹配错误
#[test]
fn test_reshape_shape_mismatch_error() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    // [2,3] (6 元素) -> [2,2] (4 元素) 应失败
    let result = x.reshape(&[2, 2]);
    assert!(result.is_err(), "元素数量不匹配时应报错");
}

/// 测试空形状错误
#[test]
fn test_reshape_empty_shape_error() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    let result = x.reshape(&[]);
    assert!(result.is_err(), "空形状应报错");
}

/// 测试 Reshape 节点不能直接设置值
#[test]
fn test_reshape_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let reshaped = x.reshape(&[3, 2]).unwrap();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let result = reshaped.set_value(&test_value);
    assert!(result.is_err(), "Reshape 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证梯度计算公式。

/// 测试 Reshape VJP（单位上游梯度）
///
/// y = reshape(x), VJP: grad_to_input = reshape(upstream, input_shape)
#[test]
fn test_reshape_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("input"))
        .unwrap();
    let reshaped = inner
        .borrow_mut()
        .create_reshape_node(input.clone(), &[3, 2], Some("reshaped"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        )))
        .unwrap();
    reshaped.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[3, 2]);
    let grad = reshaped.calc_grad_to_parent_index(0, &upstream)?;

    // 梯度只改变形状，数值直接透传
    assert_eq!(grad.shape(), &[2, 3]);
    assert_eq!(&grad, &Tensor::ones(&[2, 3]));

    Ok(())
}

/// 测试 Reshape VJP（非单位上游梯度）
#[test]
fn test_reshape_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("input"))
        .unwrap();
    let reshaped = inner
        .borrow_mut()
        .create_reshape_node(input.clone(), &[3, 2], Some("reshaped"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        )))
        .unwrap();
    reshaped.forward_recursive(1, false).unwrap();

    let upstream = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let grad = reshaped.calc_grad_to_parent_index(0, &upstream)?;

    // 梯度被 reshape 回输入形状，数值保持不变
    assert_eq!(grad.shape(), &[2, 3]);
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(&grad, &expected);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 Reshape + Sigmoid 端到端反向传播
///
/// loss = MSE(sigmoid(reshape(x)), target)
#[test]
fn test_reshape_backward_e2e_sigmoid_chain() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]))?;

    let reshaped = x.reshape(&[3, 2])?;
    let activated = reshaped.sigmoid();
    let target = graph.input(&Tensor::zeros(&[3, 2]))?;
    let loss = activated.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val > 0.0, "Loss 应为正");

    // 验证梯度存在且形状正确
    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 3]);

    // 梯度非零
    assert!(x_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10));

    Ok(())
}

/// 测试单样本 Reshape 反向传播
///
/// Input [1,6] -> Reshape [2,3] -> MatMul [2,3]@[3,1]=[2,1] -> MSE
#[test]
fn test_reshape_single_sample_backward() -> Result<(), GraphError> {
    use crate::nn::VarMatrixOps;

    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]))?;
    let reshaped = x.reshape(&[2, 3])?;

    let w = graph.parameter(&[3, 1], Init::Zeros, "w")?;
    w.set_value(&Tensor::new(&[0.1, 0.2, 0.3], &[3, 1]))?;

    let y = reshaped.matmul(&w)?;
    let target = graph.input(&Tensor::zeros(&[2, 1]))?;
    let loss = y.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    // 验证 w 的梯度存在且形状正确
    let grad_w = w.grad()?.expect("w 应有 grad");
    assert_eq!(grad_w.shape(), &[3, 1]);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试 Reshape 梯度累积：多次 backward 不调用 zero_grad，梯度应累加
#[test]
fn test_reshape_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]))?;

    let reshaped = x.reshape(&[3, 2])?;
    let activated = reshaped.sigmoid();
    let target = graph.input(&Tensor::zeros(&[3, 2]))?;
    let loss = activated.mse_loss(&target)?;

    // 第 1 次 backward
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = x.grad()?.unwrap().clone();

    // 第 2 次 backward（不 zero_grad → 梯度累积）
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

/// 测试 Reshape 节点的动态形状传播
///
/// 注意：Reshape 的动态 batch 支持有限制：
/// - 只有当目标形状保持第一维（batch）时才有意义
/// - 典型场景：[batch, 3, 4] -> [batch, 12] 或反过来
#[test]
fn test_reshape_dynamic_shape_propagation() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建 3D 输入：[batch, height, width]
    // Input 节点默认支持动态 batch
    let x = graph.input(&Tensor::zeros(&[2, 3, 4])).unwrap();

    // Reshape: [batch, 3, 4] -> [batch, 12]（保持 batch 维度）
    // 注意：这里 target_shape 是 [2, 12]，对应 batch=2 时的形状
    use crate::nn::var_ops::VarShapeOps;
    let reshaped = x.reshape(&[2, 12]).unwrap();

    // 验证动态形状传播
    let dyn_shape = reshaped.dynamic_expected_shape();
    assert!(
        dyn_shape.is_dynamic(0),
        "batch 维度应该是动态的（因为输入是动态的）"
    );
    assert!(!dyn_shape.is_dynamic(1), "第二维应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(12), "第二维应该是 12");
}

/// 测试 Reshape 在不同 batch_size 下的前向计算
#[test]
fn test_reshape_dynamic_batch_forward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarShapeOps;

    let graph = Graph::new();

    // 创建 3D 输入：[batch, height, width]
    let x = graph.input(&Tensor::zeros(&[2, 3, 4])).unwrap();

    // Reshape: [2, 3, 4] -> [2, 12]（保留 batch 维度）
    let reshaped = x.reshape(&[2, 12]).unwrap();

    // 第一次 forward：batch=2
    reshaped.forward().unwrap();
    let value1 = reshaped.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 12], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[4, 3, 4])).unwrap();

    // 第二次 forward：batch=4（目标形状按比例调整为 [4, 12]）
    reshaped.forward().unwrap();
    let value2 = reshaped.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[4, 12], "第二次 forward: batch=4");
}

/// 测试 Reshape 在不同 batch_size 下的反向传播
#[test]
fn test_reshape_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::{VarLossOps, VarShapeOps};

    let graph = Graph::new();

    // 创建 3D 输入
    let x = graph
        .input(&Tensor::normal_seeded(0.0, 1.0, &[2, 3, 4], 42))
        .unwrap();

    // Reshape: [2, 3, 4] -> [2, 12] + MSE
    let reshaped = x.reshape(&[2, 12]).unwrap();
    let target = graph.input(&Tensor::zeros(&[2, 12])).unwrap();
    let loss = reshaped.mse_loss(&target).unwrap();

    // 第一次训练：batch=2
    loss.forward().unwrap();
    let loss_val1 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::normal_seeded(0.0, 1.0, &[4, 3, 4], 100))
        .unwrap();
    target.set_value(&Tensor::zeros(&[4, 12])).unwrap();

    // 第二次训练：batch=4
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
}

// ==================== 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_reshape_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // [2, 6] -> [3, 4]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 6], Some("input"))
        .unwrap();

    let reshaped = inner
        .borrow_mut()
        .create_reshape_node(input.clone(), &[3, 4], Some("reshaped"))
        .unwrap();

    assert_eq!(reshaped.shape(), vec![3, 4]);
    assert_eq!(reshaped.name(), Some("reshaped"));
    assert!(!reshaped.is_leaf());
    assert_eq!(reshaped.parents().len(), 1);
}

#[test]
fn test_create_reshape_node_3d_to_2d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // [2, 3, 4] -> [2, 12]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4], None)
        .unwrap();

    let reshaped = inner
        .borrow_mut()
        .create_reshape_node(input, &[2, 12], None)
        .unwrap();

    assert_eq!(reshaped.shape(), vec![2, 12]);
}

#[test]
fn test_create_reshape_node_size_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // [2, 3] (6 元素) -> [2, 2] (4 元素) 应失败
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_reshape_node(input, &[2, 2], None);
    assert!(result.is_err());
}

#[test]
fn test_create_reshape_node_empty_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_reshape_node(input, &[], None);
    assert!(result.is_err());
}

#[test]
fn test_create_reshape_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_reshaped;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 6], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let reshaped = inner
            .borrow_mut()
            .create_reshape_node(input, &[3, 4], None)
            .unwrap();
        weak_reshaped = Rc::downgrade(&reshaped);

        assert!(weak_reshaped.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_reshaped.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
