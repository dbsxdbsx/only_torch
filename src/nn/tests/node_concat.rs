/*
 * @Author       : 老董
 * @Description  : Concat 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ concat axis=0/1; 错误
 * 2. VJP 单元测试（底层）→ concat mode 按 offset 分段
 * 3. E2E 反向传播（高层）→ concat same/diff shape; 三父节点; axis=1
 * 4. Create API
 *
 * Concat 节点沿现有维度拼接张量。
 * VJP: 按各父节点在 axis 维度的大小，按偏移分段提取梯度。
 */

use crate::assert_err;
use crate::nn::{Graph, GraphError, Init, Var, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试（高层 API）====================

/// concat axis=0: [2,2]+[1,2] → [3,2]
#[test]
fn test_concat_forward_axis0() {
    let graph = Graph::new();

    let p1 = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let p2 = graph.input(&Tensor::new(&[5.0, 6.0], &[1, 2])).unwrap();
    let result = Var::concat(&[&p1, &p2], 0).unwrap();

    result.forward().unwrap();

    // [[1,2],[3,4]] ++ [[5,6]] → [[1,2],[3,4],[5,6]]
    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(output, expected);
}

/// concat axis=1: [2,2]+[2,3] → [2,5]
#[test]
fn test_concat_forward_axis1() {
    let graph = Graph::new();

    let p1 = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let p2 = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]))
        .unwrap();
    let result = Var::concat(&[&p1, &p2], 1).unwrap();

    result.forward().unwrap();

    // [[1,2],[3,4]] ++ [[5,6,7],[8,9,10]] axis=1 → [[1,2,5,6,7],[3,4,8,9,10]]
    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(
        &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0],
        &[2, 5],
    );
    assert_eq!(output, expected);
}

/// 错误：shape mismatch（concat 模式）
#[test]
fn test_concat_error_shape_mismatch() {
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
        .create_concat_node(vec![input1, input2], 0, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 3], [2, 4], "Concat: 父节点 1 在维度 1 大小不一致")
    );
}

/// 错误：axis 越界（concat 模式 axis 最大为 ndim-1）
#[test]
fn test_concat_error_invalid_axis() {
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
    let result = inner
        .borrow_mut()
        .create_concat_node(vec![input1, input2], 2, None);
    assert_err!(
        result,
        GraphError::InvalidOperation("Concat: axis 2 超出有效范围 [0, 1]")
    );
}

/// Concat 节点不能直接设置值
#[test]
fn test_concat_cannot_set_value() {
    let graph = Graph::new();

    let p1 = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let p2 = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();
    let concat = Var::concat(&[&p1, &p2], 0).unwrap();

    let test_value = Tensor::new(&[1.0; 8], &[4, 2]);
    let result = concat.set_value(&test_value);
    assert!(result.is_err(), "Concat 节点不应支持直接设值");
}

// ==================== 2. VJP 单元测试（底层 calc_grad_to_parent_index）====================

/// concat axis=0: upstream [3,2] → 按 offset 分段到 [2,2] 和 [1,2]
#[test]
fn test_concat_vjp_axis0() -> Result<(), GraphError> {
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
    let concat = inner
        .borrow_mut()
        .create_concat_node(vec![p1.clone(), p2.clone()], 0, Some("concat"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(&[5.0, 6.0], &[1, 2])))
        .unwrap();
    concat.forward_recursive(1, false).unwrap();

    // upstream [3, 2]
    let upstream = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    // p1 → upstream[0:2, :] = [[1,2],[3,4]]
    let grad_p1 = concat
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    assert_eq!(grad_p1.shape(), &[2, 2]);
    assert_eq!(&grad_p1, &Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));

    // p2 → upstream[2:3, :] = [[5,6]]
    let grad_p2 = concat
        .calc_grad_to_parent_index(1, &upstream)?
        .resolve(&upstream);
    assert_eq!(grad_p2.shape(), &[1, 2]);
    assert_eq!(&grad_p2, &Tensor::new(&[5.0, 6.0], &[1, 2]));

    Ok(())
}

/// concat axis=1: upstream [2,5] → 按 offset 分段到 [2,2] 和 [2,3]
#[test]
fn test_concat_vjp_axis1() -> Result<(), GraphError> {
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
    let concat = inner
        .borrow_mut()
        .create_concat_node(vec![p1.clone(), p2.clone()], 1, Some("concat"))
        .unwrap();

    p1.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    p2.set_value(Some(&Tensor::new(
        &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[2, 3],
    )))
    .unwrap();
    concat.forward_recursive(1, false).unwrap();

    // 输出 [2, 5], upstream 递增值
    let upstream = Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[2, 5],
    );

    // p1 → upstream[:, 0:2] = [[1,2],[6,7]]
    let grad_p1 = concat
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    assert_eq!(grad_p1.shape(), &[2, 2]);
    assert_eq!(&grad_p1, &Tensor::new(&[1.0, 2.0, 6.0, 7.0], &[2, 2]));

    // p2 → upstream[:, 2:5] = [[3,4,5],[8,9,10]]
    let grad_p2 = concat
        .calc_grad_to_parent_index(1, &upstream)?
        .resolve(&upstream);
    assert_eq!(grad_p2.shape(), &[2, 3]);
    assert_eq!(
        &grad_p2,
        &Tensor::new(&[3.0, 4.0, 5.0, 8.0, 9.0, 10.0], &[2, 3])
    );

    Ok(())
}

// ==================== 3. E2E 反向传播测试（高层 API）====================

/// concat same shape: result = concat([p1, p2], axis=0), loss = MSE(result, zeros)
#[test]
fn test_concat_e2e_same_shape() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[1, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[1, 2], Init::Zeros, "p2")?;
    p1.set_value(&Tensor::new(&[1.0, 2.0], &[1, 2]))?;
    p2.set_value(&Tensor::new(&[3.0, 4.0], &[1, 2]))?;

    let result = Var::concat(&[&p1, &p2], 0)?;
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

/// concat diff shape: result = concat([p1(2,2), p2(1,2)], axis=0), loss = MSE(result, zeros)
#[test]
fn test_concat_e2e_diff_shape() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[2, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[1, 2], Init::Zeros, "p2")?;
    p1.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    p2.set_value(&Tensor::new(&[5.0, 6.0], &[1, 2]))?;

    let result = Var::concat(&[&p1, &p2], 0)?;
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
fn test_concat_e2e_three_parents() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[1, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[1, 2], Init::Zeros, "p2")?;
    let p3 = graph.parameter(&[1, 2], Init::Zeros, "p3")?;
    p1.set_value(&Tensor::new(&[1.0; 2], &[1, 2]))?;
    p2.set_value(&Tensor::new(&[2.0; 2], &[1, 2]))?;
    p3.set_value(&Tensor::new(&[3.0; 2], &[1, 2]))?;

    let result = Var::concat(&[&p1, &p2, &p3], 0)?;
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
fn test_concat_e2e_axis1() -> Result<(), GraphError> {
    let graph = Graph::new();

    let p1 = graph.parameter(&[2, 2], Init::Zeros, "p1")?;
    let p2 = graph.parameter(&[2, 2], Init::Zeros, "p2")?;
    p1.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    p2.set_value(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))?;

    let result = Var::concat(&[&p1, &p2], 1)?;
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

// ==================== 4. Create API ====================

#[test]
fn test_create_concat_node_axis0() {
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

    let concat = inner
        .borrow_mut()
        .create_concat_node(vec![p1.clone(), p2.clone()], 0, Some("concat"))
        .unwrap();

    assert_eq!(concat.shape(), vec![3, 3]);
    assert_eq!(concat.name(), Some("concat"));
    assert!(!concat.is_leaf());
    assert_eq!(concat.parents().len(), 2);
}
