/*
 * @Author       : 老董
 * @Description  : MatMul 节点单元测试（矩阵乘法）
 *
 * 测试策略：
 * 1. 前向传播测试（高层 Graph + Var API）→ basic [2,3]@[3,4]; vector [1,3]@[3,2]; cannot_set_value
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ to left; to right; non-unit upstream; negative; zero; invalid parent index
 * 3. 端到端反向传播测试（高层 Graph + Var API）
 * 4. 梯度累积测试（高层 Graph + Var API）
 * 5. 动态形状测试（KEEP AS-IS）
 * 6. Create API 测试（KEEP AS-IS）
 *
 * Key: MatMul 为二元运算 C = A @ B。形状：[m,k] @ [k,n] = [m,n]。
 * - VJP 向左(A): grad_A = upstream @ B^T
 * - VJP 向右(B): grad_B = A^T @ upstream
 * - 高层 API: .matmul(&other) 来自 VarMatrixOps
 */

use crate::assert_err;
use crate::nn::Mode;
use crate::nn::{Graph, GraphError, Init, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 Graph + Var API）====================

/// 测试 MatMul 基础前向：[2,3]@[3,4]=[[74,80,86,92],[173,188,203,218]]
#[test]
fn test_mat_mul_forward_basic() {
    let graph = Graph::new();

    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(
            &[
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            &[3, 4],
        ))
        .unwrap();

    let result = a.matmul(&b).unwrap();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(
        &[74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0],
        &[2, 4],
    );
    assert_eq!(output, expected);
}

/// 测试 MatMul 前向（向量乘矩阵）：[1,3]@[3,2]=[22,28]
///
/// vec=[1,2,3], mat=[[1,2],[3,4],[5,6]] → result=[22,28]
#[test]
fn test_mat_mul_forward_vector() {
    let graph = Graph::new();

    let vec = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let mat = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]))
        .unwrap();

    let result = vec.matmul(&mat).unwrap();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(&[22.0, 28.0], &[1, 2]);
    assert_eq!(output, expected);
}

/// 测试 MatMul 节点不能直接设置值
#[test]
fn test_mat_mul_cannot_set_value() {
    let graph = Graph::new();

    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let b = graph.input(&Tensor::new(&[1.0; 12], &[3, 4])).unwrap();

    let result = a.matmul(&b).unwrap();

    let test_value = Tensor::new(&[1.0; 8], &[2, 4]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "MatMul 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================
//
// MatMul 有 2 个父节点。grad_to_left = upstream @ B^T, grad_to_right = A^T @ upstream

/// 测试 MatMul 对左父节点的 VJP：grad = upstream @ B^T
///
/// left=[2,3], right=[3,4], upstream=[2,4]
#[test]
fn test_mat_mul_vjp_to_left() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let left = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("left"))
        .unwrap();
    let right = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("right"))
        .unwrap();
    let mm = inner
        .borrow_mut()
        .create_mat_mul_node(vec![left.clone(), right.clone()], Some("mm"))
        .unwrap();

    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let right_value = Tensor::new(
        &[
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    );
    left.set_value(Some(&left_value))?;
    right.set_value(Some(&right_value))?;
    mm.forward_recursive(1, Mode::Train)?;

    let upstream = Tensor::ones(&[2, 4]);
    let grad_to_left = mm
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    let expected = upstream.mat_mul(&right_value.transpose());
    assert_eq!(grad_to_left.shape(), &[2, 3]);
    assert_eq!(&grad_to_left, &expected);

    Ok(())
}

/// 测试 MatMul 对右父节点的 VJP：grad = A^T @ upstream
#[test]
fn test_mat_mul_vjp_to_right() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let left = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("left"))
        .unwrap();
    let right = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("right"))
        .unwrap();
    let mm = inner
        .borrow_mut()
        .create_mat_mul_node(vec![left.clone(), right.clone()], Some("mm"))
        .unwrap();

    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let right_value = Tensor::new(
        &[
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    );
    left.set_value(Some(&left_value))?;
    right.set_value(Some(&right_value))?;
    mm.forward_recursive(1, Mode::Train)?;

    let upstream = Tensor::ones(&[2, 4]);
    let grad_to_right = mm
        .calc_grad_to_parent_index(1, &upstream)?
        .resolve(&upstream);

    let expected = left_value.transpose().mat_mul(&upstream);
    assert_eq!(grad_to_right.shape(), &[3, 4]);
    assert_eq!(&grad_to_right, &expected);

    Ok(())
}

/// 测试 MatMul VJP（非单位 upstream）：[2,2]@[2,2]，upstream=[[1,2],[3,4]]
#[test]
fn test_mat_mul_vjp_non_unit_upstream() -> Result<(), GraphError> {
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
    let mm = inner
        .borrow_mut()
        .create_mat_mul_node(vec![left.clone(), right.clone()], Some("mm"))
        .unwrap();

    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    left.set_value(Some(&left_value))?;
    right.set_value(Some(&right_value))?;
    mm.forward_recursive(1, Mode::Train)?;

    let upstream = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let grad_to_left = mm
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    let expected_left = upstream.mat_mul(&right_value.transpose());
    assert_eq!(&grad_to_left, &expected_left);

    let grad_to_right = mm
        .calc_grad_to_parent_index(1, &upstream)?
        .resolve(&upstream);
    let expected_right = left_value.transpose().mat_mul(&upstream);
    assert_eq!(&grad_to_right, &expected_right);

    Ok(())
}

/// 测试 MatMul VJP（负数值）
#[test]
fn test_mat_mul_vjp_negative_values() -> Result<(), GraphError> {
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
    let mm = inner
        .borrow_mut()
        .create_mat_mul_node(vec![left.clone(), right.clone()], Some("mm"))
        .unwrap();

    let left_value = Tensor::new(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, -6.0, 7.0, -8.0], &[2, 2]);
    left.set_value(Some(&left_value))?;
    right.set_value(Some(&right_value))?;
    mm.forward_recursive(1, Mode::Train)?;

    let output = mm.value().unwrap();
    let expected_output = Tensor::new(&[9.0, -10.0, 13.0, -14.0], &[2, 2]);
    assert_eq!(output, expected_output);

    let upstream = Tensor::ones(&[2, 2]);

    let grad_to_left = mm
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    let expected_left = upstream.mat_mul(&right_value.transpose());
    assert_eq!(&grad_to_left, &expected_left);

    let grad_to_right = mm
        .calc_grad_to_parent_index(1, &upstream)?
        .resolve(&upstream);
    let expected_right = left_value.transpose().mat_mul(&upstream);
    assert_eq!(&grad_to_right, &expected_right);

    Ok(())
}

/// 测试 MatMul VJP（含零值）
#[test]
fn test_mat_mul_vjp_zero_values() -> Result<(), GraphError> {
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
    let mm = inner
        .borrow_mut()
        .create_mat_mul_node(vec![left.clone(), right.clone()], Some("mm"))
        .unwrap();

    let left_value = Tensor::new(&[0.0, 1.0, 2.0, 0.0], &[2, 2]);
    let right_value = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    left.set_value(Some(&left_value))?;
    right.set_value(Some(&right_value))?;
    mm.forward_recursive(1, Mode::Train)?;

    let output = mm.value().unwrap();
    let expected_output = Tensor::new(&[0.0, 1.0, 2.0, 0.0], &[2, 2]);
    assert_eq!(output, expected_output);

    let upstream = Tensor::ones(&[2, 2]);

    let grad_to_left = mm
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    let expected_left = upstream.mat_mul(&right_value.transpose());
    assert_eq!(&grad_to_left, &expected_left);

    let grad_to_right = mm
        .calc_grad_to_parent_index(1, &upstream)?
        .resolve(&upstream);
    let expected_right = left_value.transpose().mat_mul(&upstream);
    assert_eq!(&grad_to_right, &expected_right);

    Ok(())
}

/// 测试 MatMul 无效父节点索引 → 报错
#[test]
fn test_mat_mul_vjp_invalid_parent_index() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let left = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("left"))
        .unwrap();
    let right = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("right"))
        .unwrap();
    let mm = inner
        .borrow_mut()
        .create_mat_mul_node(vec![left.clone(), right.clone()], Some("mm"))
        .unwrap();

    left.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))?;
    right.set_value(Some(&Tensor::new(&[1.0; 12], &[3, 4])))?;
    mm.forward_recursive(1, Mode::Train)?;

    let upstream = Tensor::ones(&[2, 4]);
    let result = mm.calc_grad_to_parent_index(2, &upstream);
    assert_err!(
        result,
        GraphError::ComputationError(msg) if msg.contains("MatMul 节点只有 2 个父节点") && msg.contains("索引 2")
    );

    Ok(())
}

// ==================== 端到端反向传播测试（高层 Graph + Var API）====================

/// 测试 MatMul 端到端反向传播
///
/// left@right(单位矩阵)，loss=MSE，验证 exact grad 值
/// left=[[1,2],[3,4]], right=I=[[1,0],[0,1]], target=zeros
/// loss=mean([1,4,9,16])=7.5
/// grad_left=[[0.5,1],[1.5,2]], grad_right=[[5,7],[7,10]]
#[test]
fn test_mat_mul_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let left = graph.parameter(&[2, 2], Init::Zeros, "left")?;
    let right = graph.parameter(&[2, 2], Init::Zeros, "right")?;

    left.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    right.set_value(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]))?;

    let result = left.matmul(&right)?;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    loss.forward().unwrap();
    let loss_value = loss.value().unwrap().unwrap();
    assert_abs_diff_eq!(loss_value.get_data_number().unwrap(), 7.5, epsilon = 1e-6);

    graph.zero_grad()?;
    loss.backward()?;

    let left_grad = left.grad()?.expect("left 应有 grad");
    let right_grad = right.grad()?.expect("right 应有 grad");
    assert_eq!(left_grad.shape(), &[2, 2]);
    assert_eq!(right_grad.shape(), &[2, 2]);

    let expected_left_grad = Tensor::new(&[0.5, 1.0, 1.5, 2.0], &[2, 2]);
    assert_eq!(left_grad, &expected_left_grad);

    let expected_right_grad = Tensor::new(&[5.0, 7.0, 7.0, 10.0], &[2, 2]);
    assert_eq!(right_grad, &expected_right_grad);

    Ok(())
}

// ==================== 梯度累积测试（高层 Graph + Var API）====================

/// 测试 MatMul 梯度累积
#[test]
fn test_mat_mul_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let left = graph.parameter(&[2, 2], Init::Zeros, "left")?;
    let right = graph.parameter(&[2, 2], Init::Zeros, "right")?;

    left.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    right.set_value(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]))?;

    let result = left.matmul(&right)?;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    loss.forward().unwrap();
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = left.grad()?.unwrap().clone();

    loss.forward().unwrap();
    loss.backward()?;
    let grad_second = left.grad()?.unwrap();
    assert_eq!(grad_second, &(&grad_first * 2.0));

    graph.zero_grad()?;
    loss.forward().unwrap();
    loss.backward()?;
    let grad_after_clear = left.grad()?.unwrap();
    assert_eq!(grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 MatMul 节点的动态形状传播
#[test]
fn test_mat_mul_dynamic_shape_propagation() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建一个固定形状的权重
    let w = graph
        .parameter(&[16, 32], crate::nn::Init::Kaiming, "w")
        .unwrap();

    // MatMul: h0 @ w -> [?, 32]
    use crate::nn::var::ops::VarMatrixOps;
    let result = h0.matmul(&w).unwrap();

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "output 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(32), "output 维度应该是 32");
}

/// 测试 MatMul 节点在不同 batch_size 下的前向计算
#[test]
fn test_mat_mul_dynamic_batch_forward() {
    use crate::nn::Graph;
    use crate::nn::var::ops::VarMatrixOps;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]
    let w = graph
        .parameter(&[16, 32], crate::nn::Init::Kaiming, "w")
        .unwrap();

    // MatMul: h0 @ w
    let result = h0.matmul(&w).unwrap();

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 32], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 32], "第二次 forward: batch=8");
}

/// 测试 MatMul 节点在不同 batch_size 下的反向传播
#[test]
fn test_mat_mul_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var::ops::{VarLossOps, VarMatrixOps};

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]
    let w = graph
        .parameter(&[4, 8], crate::nn::Init::Kaiming, "w")
        .unwrap();

    // MatMul: h0 @ w -> [?, 8]
    let result = h0.matmul(&w).unwrap();

    // 创建目标和损失
    let target = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 8]);
    loss.backward().unwrap();

    // 验证 w 有梯度
    let w_grad1 = w.grad().unwrap().unwrap();
    assert_eq!(w_grad1.shape(), &[4, 8], "w 梯度形状应为 [4, 8]");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[6, 8])).unwrap();
    target.set_value(&Tensor::zeros(&[6, 8])).unwrap();

    // 第二次 forward + backward：batch=6
    loss.forward().unwrap();
    assert_eq!(
        result.value().unwrap().unwrap().shape(),
        &[6, 8],
        "第二次 forward: batch=6"
    );
    loss.backward().unwrap();

    // 验证 w 仍有正确形状的梯度
    let w_grad2 = w.grad().unwrap().unwrap();
    assert_eq!(w_grad2.shape(), &[4, 8], "w 梯度形状仍应为 [4, 8]");
}

// ==================== 节点创建 API 测试 ====================

#[test]
fn test_create_mat_mul_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // [4, 8] @ [8, 6] = [4, 6]
    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], Some("a"))
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_parameter_node(&[8, 6], Some("b"))
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_mat_mul_node(vec![a.clone(), b.clone()], Some("result"))
        .unwrap();

    assert_eq!(result.shape(), vec![4, 6]);
    assert_eq!(result.name(), Some("result"));
    assert!(!result.is_leaf());
    assert_eq!(result.parents().len(), 2);
}

#[test]
fn test_create_mat_mul_auto_name() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_parameter_node(&[8, 6], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_mat_mul_node(vec![a, b], None)
        .unwrap();

    let name = result.name().unwrap();
    assert!(name.contains("matmul"), "名称应包含 'matmul': {}", name);
}

#[test]
fn test_create_mat_mul_incompatible_shapes() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // [4, 8] @ [7, 6] 应该失败（8 != 7）
    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_parameter_node(&[7, 6], None)
        .unwrap();

    let result = inner.borrow_mut().create_mat_mul_node(vec![a, b], None);
    assert!(result.is_err());
}

#[test]
fn test_create_mat_mul_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_result;
    let weak_a;
    {
        let a = inner
            .borrow_mut()
            .create_basic_input_node(&[4, 8], None)
            .unwrap();
        let b = inner
            .borrow_mut()
            .create_parameter_node(&[8, 6], None)
            .unwrap();
        weak_a = Rc::downgrade(&a);

        let result = inner
            .borrow_mut()
            .create_mat_mul_node(vec![a, b], None)
            .unwrap();
        weak_result = Rc::downgrade(&result);

        assert!(weak_result.upgrade().is_some());
        assert!(weak_a.upgrade().is_some());
    }
    assert!(weak_result.upgrade().is_none());
    assert!(weak_a.upgrade().is_none());
}

#[test]
fn test_create_mat_mul_requires_two_parents() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], None)
        .unwrap();

    let result = inner.borrow_mut().create_mat_mul_node(vec![a], None);
    assert!(result.is_err());
}

// ==================== 3D 批量 MatMul（bmm）====================
//
// [B, m, k] @ [B, k, n] = [B, m, n]，batch 维严格相等（不做跨 batch 广播）。
// VJP：grad_A = upstream bmm B^T；grad_B = A^T bmm upstream（逐 batch，无求和归约）。

/// 3D 批量前向：每个 batch 与对应 2D matmul 逐 bit 一致
#[test]
fn test_mat_mul_batched_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    // batch=2：第 0 个 batch 与 2D basic 测试同数据，第 1 个 batch 数据平移
    let a_data: Vec<f32> = (1..=12).map(|i| i as f32).collect(); // [2,2,3]
    let b_data: Vec<f32> = (7..=30).map(|i| i as f32).collect(); // [2,3,4]
    let a = graph.input(&Tensor::new(&a_data, &[2, 2, 3]))?;
    let b = graph.input(&Tensor::new(&b_data, &[2, 3, 4]))?;

    let result = a.matmul(&b)?;
    result.forward()?;

    let output = result.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 2, 4]);

    // 期望值 = 逐 batch 2D matmul
    let out_flat = output.to_vec();
    for i in 0..2 {
        let a_i = Tensor::new(&a_data[i * 6..(i + 1) * 6], &[2, 3]);
        let b_i = Tensor::new(&b_data[i * 12..(i + 1) * 12], &[3, 4]);
        let expected = a_i.mat_mul(&b_i).to_vec();
        let actual = &out_flat[i * 8..(i + 1) * 8];
        for j in 0..8 {
            assert_eq!(
                actual[j].to_bits(),
                expected[j].to_bits(),
                "batch {i} 元素 {j} 与 2D matmul 不一致"
            );
        }
    }

    Ok(())
}

/// 3D 批量 VJP：对左右父节点的梯度与 NT/TN 原语一致
#[test]
fn test_mat_mul_batched_vjp() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let left = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2, 3], Some("left"))
        .unwrap();
    let right = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4], Some("right"))
        .unwrap();
    let mm = inner
        .borrow_mut()
        .create_mat_mul_node(vec![left.clone(), right.clone()], Some("mm"))
        .unwrap();

    let left_value = Tensor::normal_seeded(0.0, 1.0, &[2, 2, 3], 71);
    let right_value = Tensor::normal_seeded(0.0, 1.0, &[2, 3, 4], 72);
    left.set_value(Some(&left_value))?;
    right.set_value(Some(&right_value))?;
    mm.forward_recursive(1, Mode::Train)?;

    let upstream = Tensor::normal_seeded(0.0, 1.0, &[2, 2, 4], 73);

    let grad_to_left = mm
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    let expected_left = upstream.batched_mat_mul_nt(&right_value);
    assert_eq!(grad_to_left.shape(), &[2, 2, 3]);
    assert_eq!(&grad_to_left, &expected_left);

    let grad_to_right = mm
        .calc_grad_to_parent_index(1, &upstream)?
        .resolve(&upstream);
    let expected_right = left_value.batched_mat_mul_tn(&upstream);
    assert_eq!(grad_to_right.shape(), &[2, 3, 4]);
    assert_eq!(&grad_to_right, &expected_right);

    Ok(())
}

/// 3D 批量端到端反向传播（手算字面值对照）
///
/// A=[[[1,2],[3,4]], [[5,6],[7,8]]] (2,2,2)，B = 两个单位阵 (2,2,2)，target=0
/// C = A；loss = mean(A^2) = (1+4+9+16+25+36+49+64)/8 = 25.5
/// dL/dC = 2*A/8 = A/4
/// grad_A = dL/dC bmm I^T = A/4
/// grad_B = A^T bmm dL/dC = A^T @ A / 4（逐 batch）
///   batch0: A0^T@A0 = [[10,14],[14,20]] → /4 = [[2.5,3.5],[3.5,5]]
///   batch1: A1^T@A1 = [[74,86],[86,100]] → /4 = [[18.5,21.5],[21.5,25]]
#[test]
fn test_mat_mul_batched_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let left = graph.parameter(&[2, 2, 2], Init::Zeros, "left")?;
    let right = graph.parameter(&[2, 2, 2], Init::Zeros, "right")?;

    left.set_value(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2],
    ))?;
    right.set_value(&Tensor::new(
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        &[2, 2, 2],
    ))?;

    let result = left.matmul(&right)?;
    let target = graph.input(&Tensor::zeros(&[2, 2, 2]))?;
    let loss = result.mse_loss(&target)?;

    loss.forward().unwrap();
    let loss_value = loss.value().unwrap().unwrap();
    assert_abs_diff_eq!(loss_value.get_data_number().unwrap(), 25.5, epsilon = 1e-6);

    graph.zero_grad()?;
    loss.backward()?;

    let left_grad = left.grad()?.expect("left 应有 grad");
    let right_grad = right.grad()?.expect("right 应有 grad");
    assert_eq!(left_grad.shape(), &[2, 2, 2]);
    assert_eq!(right_grad.shape(), &[2, 2, 2]);

    let expected_left_grad = Tensor::new(&[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], &[2, 2, 2]);
    assert_eq!(left_grad, &expected_left_grad);

    let expected_right_grad =
        Tensor::new(&[2.5, 3.5, 3.5, 5.0, 18.5, 21.5, 21.5, 25.0], &[2, 2, 2]);
    assert_eq!(right_grad, &expected_right_grad);

    Ok(())
}

/// 3D 批量 MatMul 建图校验：batch 不等 / 内维不匹配 / 混秩均应报错
#[test]
fn test_create_mat_mul_batched_shape_validation() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let make = |shape: &[usize]| {
        inner
            .borrow_mut()
            .create_basic_input_node(shape, None)
            .unwrap()
    };

    // batch 维不等：[2,3,4] @ [3,4,5]
    let (a, b) = (make(&[2, 3, 4]), make(&[3, 4, 5]));
    let r = inner.borrow_mut().create_mat_mul_node(vec![a, b], None);
    assert_err!(r, GraphError::ShapeMismatch { message, .. } if message.contains("batch 维必须严格相等"));

    // 内维不匹配：[2,3,4] @ [2,5,6]
    let (a, b) = (make(&[2, 3, 4]), make(&[2, 5, 6]));
    let r = inner.borrow_mut().create_mat_mul_node(vec![a, b], None);
    assert_err!(r, GraphError::ShapeMismatch { message, .. } if message.contains("内维不匹配"));

    // 混秩：[2,3,4] @ [4,5]
    let (a, b) = (make(&[2, 3, 4]), make(&[4, 5]));
    let r = inner.borrow_mut().create_mat_mul_node(vec![a, b], None);
    assert_err!(r, GraphError::InvalidOperation(msg) if msg.contains("2D@2D 或 3D@3D"));

    // 合法 3D：输出形状正确
    let (a, b) = (make(&[2, 3, 4]), make(&[2, 4, 5]));
    let ok = inner
        .borrow_mut()
        .create_mat_mul_node(vec![a, b], None)
        .unwrap();
    assert_eq!(ok.shape(), vec![2, 3, 5]);
}
