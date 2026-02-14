/*
 * @Author       : 老董
 * @Description  : Permute 节点单元测试
 *
 * 测试策略（6 类标准测试）：
 * 1. 前向传播测试（高层 Graph + Var API）→ 2D transpose、3D permute、值正确性
 * 2. VJP 单元测试（calc_grad_to_parent_index）→ 逆排列梯度
 * 3. 端到端反向传播测试（高层 Graph + Var API）
 * 4. 梯度累积测试（高层 Graph + Var API）
 * 5. 动态形状测试
 * 6. 节点创建 API 测试
 *
 * 梯度公式：
 *   forward: output = input.permute(dims)
 *   backward: grad = upstream_grad.permute(inverse_dims)
 *   其中 inverse_dims[dims[i]] = i
 *
 * Python 对照脚本: tests/python/calc_jacobi_by_pytorch/node_permute.py
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试 ====================

/// 2D 转置：[2, 3] → permute([1, 0]) → [3, 2]
///
/// PyTorch 参考：
///   torch.tensor([[1,2,3],[4,5,6]]).float().permute(1,0)
///   → [[1,4],[2,5],[3,6]]
#[test]
fn test_permute_forward_2d_transpose() {
    let graph = Graph::new();

    let input_data = Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    );
    let x = graph.input(&input_data).unwrap();
    let result = x.permute(&[1, 0]).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 2]);
    // [[1,4],[2,5],[3,6]]
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 1]], 6.0, epsilon = 1e-6);
}

/// 3D permute：[2, 3, 4] → permute([0, 2, 1]) → [2, 4, 3]
///
/// 交换后两维，保持 batch 维不变
#[test]
fn test_permute_forward_3d() {
    let graph = Graph::new();

    // 创建 [2, 3, 4] 张量，值为 1..24
    let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
    let input_data = Tensor::new(&data, &[2, 3, 4]);
    let x = graph.input(&input_data).unwrap();
    let result = x.permute(&[0, 2, 1]).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 4, 3]);

    // 验证部分元素：output[b, j, i] == input[b, i, j]
    // input[0, 0, 0] = 1 → output[0, 0, 0] = 1
    assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-6);
    // input[0, 0, 1] = 2 → output[0, 1, 0] = 2
    assert_abs_diff_eq!(output[[0, 1, 0]], 2.0, epsilon = 1e-6);
    // input[0, 1, 0] = 5 → output[0, 0, 1] = 5
    assert_abs_diff_eq!(output[[0, 0, 1]], 5.0, epsilon = 1e-6);
    // input[0, 2, 3] = 12 → output[0, 3, 2] = 12
    assert_abs_diff_eq!(output[[0, 3, 2]], 12.0, epsilon = 1e-6);
    // input[1, 0, 0] = 13 → output[1, 0, 0] = 13
    assert_abs_diff_eq!(output[[1, 0, 0]], 13.0, epsilon = 1e-6);
    // input[1, 2, 3] = 24 → output[1, 3, 2] = 24
    assert_abs_diff_eq!(output[[1, 3, 2]], 24.0, epsilon = 1e-6);
}

/// transpose 便捷接口：[2, 3, 4] → transpose(1, 2) → [2, 4, 3]
///
/// 与 permute([0, 2, 1]) 等价
#[test]
fn test_permute_forward_transpose_convenience() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
    let input_data = Tensor::new(&data, &[2, 3, 4]);
    let x = graph.input(&input_data).unwrap();

    let result_permute = x.permute(&[0, 2, 1]).unwrap();
    let result_transpose = x.transpose(1, 2).unwrap();

    result_permute.forward().unwrap();
    result_transpose.forward().unwrap();

    let out_p = result_permute.value().unwrap().unwrap();
    let out_t = result_transpose.value().unwrap().unwrap();

    assert_eq!(out_p.shape(), out_t.shape());
    // 值应完全一致
    for i in 0..2 {
        for j in 0..4 {
            for k in 0..3 {
                assert_abs_diff_eq!(out_p[[i, j, k]], out_t[[i, j, k]], epsilon = 1e-6);
            }
        }
    }
}

/// 恒等排列：permute([0, 1]) 不改变任何东西
#[test]
fn test_permute_forward_identity() {
    let graph = Graph::new();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let x = graph.input(&input_data).unwrap();
    let result = x.permute(&[0, 1]).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_abs_diff_eq!(
                output[[i, j]],
                input_data[[i, j]],
                epsilon = 1e-6
            );
        }
    }
}

/// Permute 节点不能直接设置值
#[test]
fn test_permute_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let result = x.permute(&[1, 0]).unwrap();

    let test_value = Tensor::ones(&[3, 2]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Permute 节点不应支持直接设值");
}

// ==================== 2. VJP 单元测试 ====================

/// VJP 逆排列：2D 转置 dims=[1, 0]，inverse=[1, 0]
///
/// input [2, 3] → permute([1, 0]) → output [3, 2]
/// upstream [3, 2] → permute([1, 0]) → grad [2, 3]
#[test]
fn test_permute_vjp_2d() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("input"))
        .unwrap();
    let permuted = inner
        .borrow_mut()
        .create_permute_node(input.clone(), &[1, 0], Some("permute"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        )))
        .unwrap();
    permuted.forward_recursive(1, false).unwrap();

    // upstream shape = [3, 2]（permuted 的输出形状）
    let upstream = Tensor::new(&[10.0, 40.0, 20.0, 50.0, 30.0, 60.0], &[3, 2]);
    let grad = permuted.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);

    assert_eq!(grad.shape(), &[2, 3]);
    // grad = upstream.permute([1, 0])
    // upstream [[10,40],[20,50],[30,60]] → grad [[10,20,30],[40,50,60]]
    assert_abs_diff_eq!(grad[[0, 0]], 10.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 20.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 2]], 30.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 40.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 50.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 2]], 60.0, epsilon = 1e-6);

    Ok(())
}

/// VJP 逆排列：3D dims=[0, 2, 1]，inverse=[0, 2, 1]（自反排列）
///
/// input [2, 3, 4] → permute([0, 2, 1]) → output [2, 4, 3]
/// upstream [2, 4, 3] → permute([0, 2, 1]) → grad [2, 3, 4]
#[test]
fn test_permute_vjp_3d() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3, 4], Some("input"))
        .unwrap();
    let permuted = inner
        .borrow_mut()
        .create_permute_node(input.clone(), &[0, 2, 1], Some("permute"))
        .unwrap();

    let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
    input
        .set_value(Some(&Tensor::new(&data, &[2, 3, 4])))
        .unwrap();
    permuted.forward_recursive(1, false).unwrap();

    // upstream 全 1.0，形状 [2, 4, 3]
    let upstream = Tensor::ones(&[2, 4, 3]);
    let grad = permuted.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);

    // grad = upstream.permute([0, 2, 1]) → 全 1 排列后仍全 1
    assert_eq!(grad.shape(), &[2, 3, 4]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_abs_diff_eq!(grad[[i, j, k]], 1.0, epsilon = 1e-6);
            }
        }
    }

    Ok(())
}

/// VJP 非自反排列：3D dims=[2, 0, 1]，inverse=[1, 2, 0]
///
/// input [2, 3, 4] → permute([2, 0, 1]) → output [4, 2, 3]
/// upstream [4, 2, 3] → permute([1, 2, 0]) → grad [2, 3, 4]
#[test]
fn test_permute_vjp_3d_non_self_inverse() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3, 4], Some("input"))
        .unwrap();
    let permuted = inner
        .borrow_mut()
        .create_permute_node(input.clone(), &[2, 0, 1], Some("permute"))
        .unwrap();

    let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
    input
        .set_value(Some(&Tensor::new(&data, &[2, 3, 4])))
        .unwrap();
    permuted.forward_recursive(1, false).unwrap();

    // 构造非 unit upstream [4, 2, 3]
    let upstream_data: Vec<f32> = (1..=24).map(|i| i as f32 * 0.1).collect();
    let upstream = Tensor::new(&upstream_data, &[4, 2, 3]);
    let grad = permuted.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);

    // grad = upstream.permute([1, 2, 0])
    // 验证：grad[i, j, k] == upstream[k, i, j]
    assert_eq!(grad.shape(), &[2, 3, 4]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_abs_diff_eq!(
                    grad[[i, j, k]],
                    upstream[[k, i, j]],
                    epsilon = 1e-6
                );
            }
        }
    }

    Ok(())
}

// ==================== 3. 端到端反向传播测试 ====================

/// permute(x) → MSE loss → backward
///
/// x [2, 3] → permute([1, 0]) → [3, 2] → MSE(target [3, 2])
/// 梯度应正确流回 x，形状为 [2, 3]
#[test]
fn test_permute_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    ))?;

    let permuted = x.permute(&[1, 0])?;
    let target = graph.input(&Tensor::zeros(&[3, 2]))?;
    let loss = permuted.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    // loss 应为有限正数
    assert!(loss_val > 0.0);
    assert!(loss_val.is_finite());

    let input_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    // 所有元素都有非零梯度（MSE 对每个元素都有贡献）
    let mut has_nonzero = false;
    for i in 0..2 {
        for j in 0..3 {
            if input_grad[[i, j]].abs() > 1e-10 {
                has_nonzero = true;
            }
        }
    }
    assert!(has_nonzero, "permute 反向传播应有非零梯度");

    Ok(())
}

/// 3D 端到端：x [2, 3, 4] → permute([0, 2, 1]) → [2, 4, 3] → MSE
#[test]
fn test_permute_backward_e2e_3d() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3, 4], Init::Zeros, "x")?;
    let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
    x.set_value(&Tensor::new(&data, &[2, 3, 4]))?;

    let permuted = x.permute(&[0, 2, 1])?;
    let target = graph.input(&Tensor::zeros(&[2, 4, 3]))?;
    let loss = permuted.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    assert!(loss_val > 0.0);
    assert!(loss_val.is_finite());

    let input_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3, 4]);

    Ok(())
}

/// transpose 便捷接口端到端
#[test]
fn test_transpose_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3, 4], Init::Zeros, "x")?;
    let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
    x.set_value(&Tensor::new(&data, &[2, 3, 4]))?;

    let transposed = x.transpose(1, 2)?;
    let target = graph.input(&Tensor::zeros(&[2, 4, 3]))?;
    let loss = transposed.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    assert!(loss_val > 0.0);
    assert!(loss_val.is_finite());

    let input_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3, 4]);

    Ok(())
}

// ==================== 4. 梯度累积测试 ====================

/// 测试 Permute 梯度累积 + zero_grad
#[test]
fn test_permute_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 3], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    ))?;

    let permuted = x.permute(&[1, 0])?;
    let target = graph.input(&Tensor::ones(&[3, 2]))?;
    let loss = permuted.mse_loss(&target)?;

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

// ==================== 5. 动态形状测试 ====================

/// 测试 Permute 节点的动态形状传播
///
/// Input [4, 8] → permute([1, 0]) → [8, ?]
/// batch 维度 (dim 0) 经过转置后变成 dim 1，应为动态
#[test]
fn test_permute_dynamic_shape_propagation() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let result = x.permute(&[1, 0]).unwrap();

    let dyn_shape = result.dynamic_expected_shape();
    // 原始 dim 0 是动态（batch），permute([1,0]) 后 dim 1 变动态
    assert!(!dyn_shape.is_dynamic(0), "原 dim 1（固定 8）移到 dim 0，应固定");
    assert!(dyn_shape.is_dynamic(1), "原 dim 0（动态 batch）移到 dim 1，应动态");
    assert_eq!(dyn_shape.dim(0), Some(8));
    assert_eq!(dyn_shape.dims().len(), 2);
}

/// 3D 动态形状：Input [4, 3, 8] → permute([0, 2, 1]) → [?, 8, 3]
#[test]
fn test_permute_dynamic_shape_3d() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[4, 3, 8])).unwrap();
    let result = x.permute(&[0, 2, 1]).unwrap();

    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应保持动态");
    assert!(!dyn_shape.is_dynamic(1), "dim 1（原 dim 2=8）应固定");
    assert!(!dyn_shape.is_dynamic(2), "dim 2（原 dim 1=3）应固定");
    assert_eq!(dyn_shape.dim(1), Some(8));
    assert_eq!(dyn_shape.dim(2), Some(3));
}

/// 测试 Permute 节点在不同 batch_size 下的前向计算
#[test]
fn test_permute_dynamic_batch_forward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 3])).unwrap();
    let result = x.permute(&[1, 0]).unwrap();

    // 第一次 forward：batch=2
    result.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[3, 2]);

    // 更新输入为不同 batch_size
    x.set_value(&Tensor::zeros(&[5, 3])).unwrap();

    // 第二次 forward：batch=5
    result.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[3, 5]);
}

/// 测试 Permute 节点在不同 batch_size 下的反向传播
#[test]
fn test_permute_dynamic_batch_backward() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::normal_seeded(0.0, 1.0, &[2, 3], 42))
        .unwrap();
    let result = x.permute(&[1, 0]).unwrap();
    let target = graph.input(&Tensor::zeros(&[3, 2])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次：batch=2
    loss.forward().unwrap();
    let loss_val1 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);
    loss.backward().unwrap();

    // 更新为不同 batch_size
    x.set_value(&Tensor::normal_seeded(0.0, 1.0, &[5, 3], 100))
        .unwrap();
    target.set_value(&Tensor::zeros(&[3, 5])).unwrap();

    // 第二次：batch=5
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
    loss.backward().unwrap();
}

// ==================== 6. 节点创建 API 测试 ====================

use std::rc::Rc;

/// 基本创建：验证输出形状
///
/// [2, 3] permute([1, 0]) → [3, 2]
#[test]
fn test_create_permute_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();
    let permuted = inner
        .borrow_mut()
        .create_permute_node(input.clone(), &[1, 0], Some("permuted"))
        .unwrap();

    assert_eq!(permuted.shape(), vec![3, 2]);
    assert_eq!(permuted.name(), Some("permuted"));
    assert!(!permuted.is_leaf());
    assert_eq!(permuted.parents().len(), 1);
}

/// 3D 创建：验证输出形状
///
/// [2, 3, 4] permute([0, 2, 1]) → [2, 4, 3]
#[test]
fn test_create_permute_node_3d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4], Some("input"))
        .unwrap();
    let permuted = inner
        .borrow_mut()
        .create_permute_node(input.clone(), &[0, 2, 1], Some("permuted"))
        .unwrap();

    assert_eq!(permuted.shape(), vec![2, 4, 3]);
}

/// 无效 dims 长度（应报错）
///
/// [2, 3] permute([0, 1, 2]) → dims 长度 3 != ndim 2
#[test]
fn test_create_permute_node_invalid_dims_length() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_permute_node(input, &[0, 1, 2], None);
    assert!(result.is_err());
}

/// 无效维度值（应报错）
///
/// [2, 3] permute([0, 5]) → 维度 5 超出范围
#[test]
fn test_create_permute_node_invalid_dim_value() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_permute_node(input, &[0, 5], None);
    assert!(result.is_err());
}

/// 重复维度（应报错）
///
/// [2, 3] permute([0, 0]) → 维度 0 重复
#[test]
fn test_create_permute_node_duplicate_dims() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_permute_node(input, &[0, 0], None);
    assert!(result.is_err());
}

/// drop 释放
#[test]
fn test_create_permute_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_permuted;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let permuted = inner
            .borrow_mut()
            .create_permute_node(input, &[1, 0], None)
            .unwrap();
        weak_permuted = Rc::downgrade(&permuted);

        assert!(weak_permuted.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_permuted.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}

/// transpose 错误处理：dim 超出范围
#[test]
fn test_transpose_invalid_dim() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    let err = x.transpose(0, 5);
    assert!(err.is_err(), "超出范围的 dim 应报错");
}
