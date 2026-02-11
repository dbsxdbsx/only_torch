/*
 * @Author       : 老董
 * @Description  : Stack 节点单元测试（含 concat 和 stack 两种模式）
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ concat axis=0/1; stack axis=0/1/末尾; 三父节点; 错误; cannot_set_value
 * 2. VJP 单元测试（底层）→ stack mode 切片; concat mode 分段; axis=1
 * 3. E2E 反向传播（高层）→ concat same/diff shape; stack mode; 三父节点; axis=1
 * 4. Create API（KEEP AS-IS）
 *
 * Stack 支持 concat（沿现有轴拼接）和 stack（插入新维度）两种模式。
 * VJP: concat→按 offset 分段; stack→按 axis 切片
 */

use crate::assert_err;
use crate::nn::{Graph, GraphError, Init, Var, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试（高层 API）====================

/// concat axis=0: [2,2]+[1,2] → [3,2]
#[test]
fn test_stack_forward_concat_axis0() {
    let graph = Graph::new();

    let p1 = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let p2 = graph.input(&Tensor::new(&[5.0, 6.0], &[1, 2])).unwrap();
    let result = Var::stack(&[&p1, &p2], 0, false).unwrap();

    result.forward().unwrap();

    // [[1,2],[3,4]] ++ [[5,6]] → [[1,2],[3,4],[5,6]]
    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(output, expected);
}

/// concat axis=1: [2,2]+[2,3] → [2,5]
#[test]
fn test_stack_forward_concat_axis1() {
    let graph = Graph::new();

    let p1 = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let p2 = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]))
        .unwrap();
    let result = Var::stack(&[&p1, &p2], 1, false).unwrap();

    result.forward().unwrap();

    // [[1,2],[3,4]] ++ [[5,6,7],[8,9,10]] axis=1 → [[1,2,5,6,7],[3,4,8,9,10]]
    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(
        &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0],
        &[2, 5],
    );
    assert_eq!(output, expected);
}

/// stack axis=0: [2,2]+[2,2] → [2,2,2]
#[test]
fn test_stack_forward_stack_axis0() {
    let graph = Graph::new();

    let p1 = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let p2 = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();
    let result = Var::stack(&[&p1, &p2], 0, true).unwrap();

    result.forward().unwrap();

    // [[[1,2],[3,4]], [[5,6],[7,8]]]
    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2, 2]);
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    assert_eq!(output, expected);
}

/// stack axis=1: [2,3]+[2,3] → [2,2,3]
#[test]
fn test_stack_forward_stack_axis1() {
    let graph = Graph::new();

    let p1 = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let p2 = graph
        .input(&Tensor::new(
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[2, 3],
        ))
        .unwrap();
    let result = Var::stack(&[&p1, &p2], 1, true).unwrap();

    result.forward().unwrap();

    // [[[1,2,3],[7,8,9]], [[4,5,6],[10,11,12]]]
    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2, 3]);
    let expected = Tensor::new(
        &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
        &[2, 2, 3],
    );
    assert_eq!(output, expected);
}

/// stack axis=末尾: [2,2]+[2,2] → [2,2,2]
#[test]
fn test_stack_forward_stack_axis_last() {
    let graph = Graph::new();

    let p1 = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let p2 = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();
    let result = Var::stack(&[&p1, &p2], 2, true).unwrap();

    result.forward().unwrap();

    // axis=2（末尾）→ [[[1,5],[2,6]], [[3,7],[4,8]]]
    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2, 2]);
    let expected = Tensor::new(&[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], &[2, 2, 2]);
    assert_eq!(output, expected);
}

/// 三个父节点 stack axis=0: 3×[1,2] → [3,1,2]
#[test]
fn test_stack_forward_three_parents() {
    let graph = Graph::new();

    let p1 = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let p2 = graph.input(&Tensor::new(&[3.0, 4.0], &[1, 2])).unwrap();
    let p3 = graph.input(&Tensor::new(&[5.0, 6.0], &[1, 2])).unwrap();
    let result = Var::stack(&[&p1, &p2, &p3], 0, true).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 1, 2]);
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 1, 2]);
    assert_eq!(output, expected);
}

/// 错误：shape mismatch（concat 模式，底层 API 精确验证错误消息）
#[test]
fn test_stack_error_shape_mismatch_concat() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input1"))
        .unwrap();
    let input2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("input2"))
        .unwrap();

    // concat axis=0: 维度 1 不一致 (3 != 4)
    let result = inner
        .borrow_mut()
        .create_stack_node(vec![input1, input2], 0, false, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch(
            [2, 3],
            [2, 4],
            "Stack (new_dim=false): 父节点 1 在维度 1 大小不一致"
        )
    );
}

/// 错误：shape mismatch（stack 模式要求形状完全相同）
#[test]
fn test_stack_error_shape_mismatch_stack() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input1"))
        .unwrap();
    let input2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("input2"))
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_stack_node(vec![input1, input2], 0, true, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 3], [2, 4], "Stack (new_dim=true): 父节点 1 形状不一致")
    );
}

/// 错误：axis 越界（底层 API）
#[test]
fn test_stack_error_invalid_axis() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input1"))
        .unwrap();
    let input2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input2"))
        .unwrap();

    // concat 模式：axis 最大为 ndim-1 = 1
    let result = inner.borrow_mut().create_stack_node(
        vec![input1.clone(), input2.clone()],
        2,
        false,
        None,
    );
    assert_err!(
        result,
        GraphError::InvalidOperation("Stack: axis 2 超出有效范围 [0, 1]")
    );

    // stack 模式：axis 最大为 ndim = 2
    let result =
        inner
            .borrow_mut()
            .create_stack_node(vec![input1, input2], 3, true, None);
    assert_err!(
        result,
        GraphError::InvalidOperation("Stack: axis 3 超出有效范围 [0, 2]")
    );
}

/// Stack 节点不能直接设置值
#[test]
fn test_stack_cannot_set_value() {
    let graph = Graph::new();

    let p1 = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let p2 = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();
    let stacked = Var::stack(&[&p1, &p2], 0, true).unwrap();

    let test_value = Tensor::new(&[1.0; 8], &[2, 2, 2]);
    let result = stacked.set_value(&test_value);
    assert!(result.is_err(), "Stack 节点不应支持直接设值");
}

// ==================== 2. VJP 单元测试（底层 calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，直接验证每个父节点的梯度分配。
// stack mode: 沿 axis 切片; concat mode: 按 offset 分段

/// stack mode axis=0 → 切片到第一个父节点
///
/// upstream [2,2,2] → p1 = upstream[0,:,:] = [2,2]
#[test]
fn test_stack_vjp_stack_to_first_parent() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_parameter_node(&[2, 2], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_parameter_node(&[2, 2], Some("p2"))
        .unwrap();
    let stack = inner
        .borrow_mut()
        .create_stack_node(vec![p1.clone(), p2.clone()], 0, true, Some("stack"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))
        .unwrap();
    stack.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[2, 2, 2]);
    let grad_p1 = stack.calc_grad_to_parent_index(0, &upstream)?;

    // p1 → upstream[0, :, :] = [[1,1],[1,1]]
    assert_eq!(grad_p1.shape(), &[2, 2]);
    assert_eq!(&grad_p1, &Tensor::ones(&[2, 2]));

    Ok(())
}

/// stack mode axis=0 → 切片到第二个父节点
///
/// upstream [2,2,2] → p2 = upstream[1,:,:] = [[5,6],[7,8]]
#[test]
fn test_stack_vjp_stack_to_second_parent() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_parameter_node(&[2, 2], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_parameter_node(&[2, 2], Some("p2"))
        .unwrap();
    let stack = inner
        .borrow_mut()
        .create_stack_node(vec![p1.clone(), p2.clone()], 0, true, Some("stack"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))
        .unwrap();
    stack.forward_recursive(1, false).unwrap();

    // upstream = [[[1,2],[3,4]], [[5,6],[7,8]]]
    let upstream = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let grad_p2 = stack.calc_grad_to_parent_index(1, &upstream)?;

    // p2 → upstream[1, :, :] = [[5,6],[7,8]]
    assert_eq!(grad_p2.shape(), &[2, 2]);
    assert_eq!(
        &grad_p2,
        &Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])
    );

    Ok(())
}

/// concat mode axis=0: upstream [3,2] → 按 offset 分段到 [2,2] 和 [1,2]
#[test]
fn test_stack_vjp_concat_axis0() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_parameter_node(&[2, 2], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_parameter_node(&[1, 2], Some("p2"))
        .unwrap();
    let stack = inner
        .borrow_mut()
        .create_stack_node(vec![p1.clone(), p2.clone()], 0, false, Some("stack"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0], &[1, 2])))
        .unwrap();
    stack.forward_recursive(1, false).unwrap();

    // upstream [3, 2]
    let upstream = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    // p1 → upstream[0:2, :] = [[1,2],[3,4]]
    let grad_p1 = stack.calc_grad_to_parent_index(0, &upstream)?;
    assert_eq!(grad_p1.shape(), &[2, 2]);
    assert_eq!(
        &grad_p1,
        &Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
    );

    // p2 → upstream[2:3, :] = [[5,6]]
    let grad_p2 = stack.calc_grad_to_parent_index(1, &upstream)?;
    assert_eq!(grad_p2.shape(), &[1, 2]);
    assert_eq!(&grad_p2, &Tensor::new(&[5.0, 6.0], &[1, 2]));

    Ok(())
}

/// concat mode axis=1: upstream [2,5] → 按 offset 分段到 [2,2] 和 [2,3]
#[test]
fn test_stack_vjp_concat_axis1() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_parameter_node(&[2, 2], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("p2"))
        .unwrap();
    let stack = inner
        .borrow_mut()
        .create_stack_node(vec![p1.clone(), p2.clone()], 1, false, Some("stack"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(
        &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[2, 3],
    )))
    .unwrap();
    stack.forward_recursive(1, false).unwrap();

    // 输出 [2, 5], upstream 递增值
    let upstream = Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[2, 5],
    );

    // p1 → upstream[:, 0:2] = [[1,2],[6,7]]
    let grad_p1 = stack.calc_grad_to_parent_index(0, &upstream)?;
    assert_eq!(grad_p1.shape(), &[2, 2]);
    assert_eq!(
        &grad_p1,
        &Tensor::new(&[1.0, 2.0, 6.0, 7.0], &[2, 2])
    );

    // p2 → upstream[:, 2:5] = [[3,4,5],[8,9,10]]
    let grad_p2 = stack.calc_grad_to_parent_index(1, &upstream)?;
    assert_eq!(grad_p2.shape(), &[2, 3]);
    assert_eq!(
        &grad_p2,
        &Tensor::new(&[3.0, 4.0, 5.0, 8.0, 9.0, 10.0], &[2, 3])
    );

    Ok(())
}

/// stack mode axis=1: upstream [2,2,3] → 沿 axis=1 切片
#[test]
fn test_stack_vjp_stack_axis1() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("p2"))
        .unwrap();
    let stack = inner
        .borrow_mut()
        .create_stack_node(vec![p1.clone(), p2.clone()], 1, true, Some("stack"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    )))
    .unwrap();
    p2.set_value(Some(&Tensor::new(
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[2, 3],
    )))
    .unwrap();
    stack.forward_recursive(1, false).unwrap();

    // upstream [2, 2, 3] 递增值
    let upstream = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
    );

    // p1 → upstream[:, 0, :] = [[1,2,3],[7,8,9]]
    let grad_p1 = stack.calc_grad_to_parent_index(0, &upstream)?;
    assert_eq!(grad_p1.shape(), &[2, 3]);
    assert_eq!(
        &grad_p1,
        &Tensor::new(&[1.0, 2.0, 3.0, 7.0, 8.0, 9.0], &[2, 3])
    );

    // p2 → upstream[:, 1, :] = [[4,5,6],[10,11,12]]
    let grad_p2 = stack.calc_grad_to_parent_index(1, &upstream)?;
    assert_eq!(grad_p2.shape(), &[2, 3]);
    assert_eq!(
        &grad_p2,
        &Tensor::new(&[4.0, 5.0, 6.0, 10.0, 11.0, 12.0], &[2, 3])
    );

    Ok(())
}

// ==================== 3. E2E 反向传播测试（高层 API）====================

/// concat same shape: result = concat([p1, p2], axis=0), loss = MSE(result, zeros)
#[test]
fn test_stack_e2e_concat_same_shape() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[1, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[1, 2], Init::Zeros, "p2")?;
    p1.set_value(&Tensor::new(&[1.0, 2.0], &[1, 2]))?;
    p2.set_value(&Tensor::new(&[3.0, 4.0], &[1, 2]))?;

    let result = Var::stack(&[&p1, &p2], 0, false)?;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // result = [[1,2],[3,4]], loss = mean([1,4,9,16]) = 30/4 = 7.5
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 7.5, epsilon = 1e-6);

    // ∂loss/∂result = 2*result/n = result/2
    // ∂loss/∂p1 = [[0.5, 1.0]], ∂loss/∂p2 = [[1.5, 2.0]]
    let p1_grad = p1.grad()?.expect("p1 应有 grad");
    let p2_grad = p2.grad()?.expect("p2 应有 grad");

    assert_abs_diff_eq!(&p1_grad, &Tensor::new(&[0.5, 1.0], &[1, 2]), epsilon = 1e-6);
    assert_abs_diff_eq!(&p2_grad, &Tensor::new(&[1.5, 2.0], &[1, 2]), epsilon = 1e-6);

    Ok(())
}

/// stack mode: result = stack([p1, p2], axis=0, new_dim=true), loss = MSE(result, zeros)
#[test]
fn test_stack_e2e_stack_mode() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[2, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[2, 2], Init::Zeros, "p2")?;
    p1.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    p2.set_value(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))?;

    let result = Var::stack(&[&p1, &p2], 0, true)?;
    let target = graph.input(&Tensor::zeros(&[2, 2, 2]))?;
    let loss = result.mse_loss(&target)?;

    // result = [[[1,2],[3,4]], [[5,6],[7,8]]]
    // loss = mean([1,4,9,16,25,36,49,64]) = 204/8 = 25.5
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 25.5, epsilon = 1e-6);

    // ∂loss/∂result = result/4
    // ∂loss/∂p1 = [[0.25,0.5],[0.75,1.0]], ∂loss/∂p2 = [[1.25,1.5],[1.75,2.0]]
    let p1_grad = p1.grad()?.expect("p1 应有 grad");
    let p2_grad = p2.grad()?.expect("p2 应有 grad");

    assert_abs_diff_eq!(
        &p1_grad,
        &Tensor::new(&[0.25, 0.5, 0.75, 1.0], &[2, 2]),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        &p2_grad,
        &Tensor::new(&[1.25, 1.5, 1.75, 2.0], &[2, 2]),
        epsilon = 1e-6
    );

    Ok(())
}

/// concat diff shape: result = concat([p1(2,2), p2(1,2)], axis=0), loss = MSE(result, zeros)
#[test]
fn test_stack_e2e_concat_diff_shape() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[2, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[1, 2], Init::Zeros, "p2")?;
    p1.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    p2.set_value(&Tensor::new(&[5.0, 6.0], &[1, 2]))?;

    let result = Var::stack(&[&p1, &p2], 0, false)?;
    let target = graph.input(&Tensor::zeros(&[3, 2]))?;
    let loss = result.mse_loss(&target)?;

    // result = [[1,2],[3,4],[5,6]]
    // loss = mean([1,4,9,16,25,36]) = 91/6 ≈ 15.167
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 91.0 / 6.0, epsilon = 1e-4);

    // ∂loss/∂result = result/3
    let p1_grad = p1.grad()?.expect("p1 应有 grad");
    let p2_grad = p2.grad()?.expect("p2 应有 grad");
    assert_eq!(p1_grad.shape(), &[2, 2]);
    assert_eq!(p2_grad.shape(), &[1, 2]);

    assert_abs_diff_eq!(
        &p1_grad,
        &Tensor::new(&[1.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0], &[2, 2]),
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        &p2_grad,
        &Tensor::new(&[5.0 / 3.0, 2.0], &[1, 2]),
        epsilon = 1e-4
    );

    Ok(())
}

/// 三个父节点 concat: result = concat([p1, p2, p3], axis=0), loss = MSE(result, zeros)
#[test]
fn test_stack_e2e_three_parents() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[1, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[1, 2], Init::Zeros, "p2")?;
    let p3 = graph.parameter(&[1, 2], Init::Zeros, "p3")?;
    p1.set_value(&Tensor::new(&[1.0; 2], &[1, 2]))?;
    p2.set_value(&Tensor::new(&[2.0; 2], &[1, 2]))?;
    p3.set_value(&Tensor::new(&[3.0; 2], &[1, 2]))?;

    let result = Var::stack(&[&p1, &p2, &p3], 0, false)?;
    let target = graph.input(&Tensor::zeros(&[3, 2]))?;
    let loss = result.mse_loss(&target)?;

    // result = [[1,1],[2,2],[3,3]]
    // loss = mean([1,1,4,4,9,9]) = 28/6
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 28.0 / 6.0, epsilon = 1e-4);

    // ∂loss/∂result = result/3
    let p1_grad = p1.grad()?.expect("p1 应有 grad");
    let p2_grad = p2.grad()?.expect("p2 应有 grad");
    let p3_grad = p3.grad()?.expect("p3 应有 grad");

    assert_abs_diff_eq!(
        &p1_grad,
        &Tensor::new(&[1.0 / 3.0; 2], &[1, 2]),
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        &p2_grad,
        &Tensor::new(&[2.0 / 3.0; 2], &[1, 2]),
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(&p3_grad, &Tensor::new(&[1.0; 2], &[1, 2]), epsilon = 1e-4);

    Ok(())
}

/// concat axis=1 same shape: result = concat([p1, p2], axis=1), loss = MSE(result, zeros)
#[test]
fn test_stack_e2e_concat_axis1() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[2, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[2, 2], Init::Zeros, "p2")?;
    p1.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    p2.set_value(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))?;

    let result = Var::stack(&[&p1, &p2], 1, false)?;
    let target = graph.input(&Tensor::zeros(&[2, 4]))?;
    let loss = result.mse_loss(&target)?;

    // result = [[1,2,5,6],[3,4,7,8]]
    // loss = mean([1,4,25,36,9,16,49,64]) = 204/8 = 25.5
    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert_abs_diff_eq!(loss_val, 25.5, epsilon = 1e-4);

    // ∂loss/∂result = result/4
    // ∂loss/∂p1 = [[0.25,0.5],[0.75,1.0]], ∂loss/∂p2 = [[1.25,1.5],[1.75,2.0]]
    let p1_grad = p1.grad()?.expect("p1 应有 grad");
    let p2_grad = p2.grad()?.expect("p2 应有 grad");

    assert_abs_diff_eq!(
        &p1_grad,
        &Tensor::new(&[0.25, 0.5, 0.75, 1.0], &[2, 2]),
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        &p2_grad,
        &Tensor::new(&[1.25, 1.5, 1.75, 2.0], &[2, 2]),
        epsilon = 1e-4
    );

    Ok(())
}

// ==================== 4. Create API（KEEP AS-IS）====================

use std::rc::Rc;

#[test]
fn test_create_stack_node_concat_axis0() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // concat 模式：[2, 3] + [1, 3] -> [3, 3]
    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("p2"))
        .unwrap();

    let stack = inner
        .borrow_mut()
        .create_stack_node(vec![p1.clone(), p2.clone()], 0, false, Some("stack"))
        .unwrap();

    assert_eq!(stack.shape(), vec![3, 3]);
    assert_eq!(stack.name(), Some("stack"));
    assert!(!stack.is_leaf());
    assert_eq!(stack.parents().len(), 2);
}

#[test]
fn test_create_stack_node_stack_axis0() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // stack 模式：在 axis=0 插入新维度
    // [2, 3] + [2, 3] -> [2, 2, 3]
    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();

    let stack = inner
        .borrow_mut()
        .create_stack_node(vec![p1, p2], 0, true, None)
        .unwrap();

    assert_eq!(stack.shape(), vec![2, 2, 3]);
}

#[test]
fn test_create_stack_node_three_parents() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], None)
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], None)
        .unwrap();
    let p3 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], None)
        .unwrap();

    let stack = inner
        .borrow_mut()
        .create_stack_node(vec![p1, p2, p3], 0, true, None)
        .unwrap();

    // [2, 2] * 3 -> [3, 2, 2]
    assert_eq!(stack.shape(), vec![3, 2, 2]);
}

#[test]
fn test_create_stack_node_shape_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // stack 模式要求形状完全相同
    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None) // 形状不同
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_stack_node(vec![p1, p2], 0, true, None);
    assert!(result.is_err());
}

#[test]
fn test_create_stack_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_stack;
    let weak_p1;
    let weak_p2;
    {
        let p1 = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        let p2 = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_p1 = Rc::downgrade(&p1);
        weak_p2 = Rc::downgrade(&p2);

        let stack = inner
            .borrow_mut()
            .create_stack_node(vec![p1, p2], 0, true, None)
            .unwrap();
        weak_stack = Rc::downgrade(&stack);

        assert!(weak_stack.upgrade().is_some());
        assert!(weak_p1.upgrade().is_some());
        assert!(weak_p2.upgrade().is_some());
    }
    assert!(weak_stack.upgrade().is_none());
    assert!(weak_p1.upgrade().is_none());
    assert!(weak_p2.upgrade().is_none());
}
