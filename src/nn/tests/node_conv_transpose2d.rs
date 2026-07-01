/*
 * @Author       : 老董
 * @Date         : 2026-04-19
 * @Description  : ConvTranspose2d 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播（底层 inner_rc API）→ simple; stride; 多通道; batch; 错误处理
 * 2. VJP 单元测试（calc_grad_to_parent_index）→ 对 input; 对 kernel
 * 3. 端到端反向传播测试（底层构建完整图）
 *
 * Key: ConvTranspose2d 是二元运算 (input, kernel)。
 * - 输入 [batch, C_in, H, W]，卷积核 [C_in, C_out, kH, kW]
 * - 输出 [batch, C_out, H_out, W_out]
 * - H_out = (H - 1) * stride - 2 * padding + kernel + output_padding
 */

use crate::assert_err;
use crate::nn::Mode;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 1. 前向传播测试 ====================

/// 全 1 输入 × 全 1 核 → 已知 scatter-add 结果
#[test]
fn test_conv_transpose2d_forward_simple() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let deconv = inner.borrow_mut().create_conv_transpose2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        (0, 0),
        Some("deconv"),
    )?;

    input.set_value(Some(&Tensor::ones(&[1, 1, 2, 2])))?;
    kernel.set_value(Some(&Tensor::ones(&[1, 1, 2, 2])))?;

    deconv.forward_recursive(1, Mode::Train)?;

    let output = deconv.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 3, 3]);

    // PyTorch: [[1,2,1],[2,4,2],[1,2,1]]
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 0, 2]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 4.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 1, 2]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 2, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 2, 1]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 2, 2]], 1.0, epsilon = 1e-5);

    Ok(())
}

/// 测试非均匀输入的前向传播
#[test]
fn test_conv_transpose2d_forward_nonuniform() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let deconv = inner.borrow_mut().create_conv_transpose2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        (0, 0),
        Some("deconv"),
    )?;

    // 输入 [[1,2],[3,4]]，核 [[1,0],[0,1]]
    input.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])))?;
    kernel.set_value(Some(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])))?;

    deconv.forward_recursive(1, Mode::Train)?;

    let output = deconv.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 3, 3]);

    // 手动计算 scatter-add:
    // (0,0)=1, (0,1)=2, (0,2)=0
    // (1,0)=3, (1,1)=5, (1,2)=2
    // (2,0)=0, (2,1)=3, (2,2)=4
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 0, 2]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 5.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 1, 2]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 2, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 2, 1]], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 2, 2]], 4.0, epsilon = 1e-5);

    Ok(())
}

/// 测试 batch 维度
#[test]
fn test_conv_transpose2d_forward_batch() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 2, 2], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let deconv = inner.borrow_mut().create_conv_transpose2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        (0, 0),
        Some("deconv"),
    )?;

    // batch 0: 全 1，batch 1: 全 2
    let mut input_data = vec![1.0f32; 4];
    input_data.extend(vec![2.0f32; 4]);
    input.set_value(Some(&Tensor::new(&input_data, &[2, 1, 2, 2])))?;
    kernel.set_value(Some(&Tensor::ones(&[1, 1, 2, 2])))?;

    deconv.forward_recursive(1, Mode::Train)?;

    let output = deconv.value().unwrap();
    assert_eq!(output.shape(), &[2, 1, 3, 3]);

    // batch 0 中心 = 4.0，batch 1 中心 = 8.0
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 4.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[1, 0, 1, 1]], 8.0, epsilon = 1e-5);

    Ok(())
}

/// 测试多通道
#[test]
fn test_conv_transpose2d_multi_channel() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // in_ch=2, out_ch=1, kernel [2,1,2,2]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2, 2, 2], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 2, 2], Some("kernel"))?;

    let deconv = inner.borrow_mut().create_conv_transpose2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        (0, 0),
        Some("deconv"),
    )?;

    input.set_value(Some(&Tensor::ones(&[1, 2, 2, 2])))?;
    kernel.set_value(Some(&Tensor::ones(&[2, 1, 2, 2])))?;

    deconv.forward_recursive(1, Mode::Train)?;

    let output = deconv.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 3, 3]);

    // 两个输入通道各贡献 [[1,2,1],[2,4,2],[1,2,1]]，叠加后中心 = 8.0
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 8.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 2.0, epsilon = 1e-5);

    Ok(())
}

/// 测试 stride=2 上采样
#[test]
fn test_conv_transpose2d_stride2() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let deconv = inner.borrow_mut().create_conv_transpose2d_node(
        vec![input.clone(), kernel.clone()],
        (2, 2),
        (0, 0),
        (0, 0),
        Some("deconv"),
    )?;

    input.set_value(Some(&Tensor::ones(&[1, 1, 2, 2])))?;
    kernel.set_value(Some(&Tensor::ones(&[1, 1, 2, 2])))?;

    deconv.forward_recursive(1, Mode::Train)?;

    // H_out = (2-1)*2 + 2 = 4
    let output = deconv.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 4, 4]);

    Ok(())
}

// ==================== 2. 错误处理测试 ====================

/// 测试通道数不匹配
#[test]
fn test_conv_transpose2d_channel_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入 C_in=3，核 C_in=1 → 不匹配
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3, 4, 4], Some("input"))
        .unwrap();
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[1, 2, 3, 3], Some("kernel"))
        .unwrap();

    let result = inner.borrow_mut().create_conv_transpose2d_node(
        vec![input, kernel],
        (1, 1),
        (0, 0),
        (0, 0),
        Some("deconv"),
    );
    assert_err!(result, GraphError::ShapeMismatch { message, .. } if message.contains("通道数"));
}

/// 测试无效输入维度
#[test]
fn test_conv_transpose2d_invalid_input_dims() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入 2D（缺少 batch 和通道维度）
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 4], Some("input"))
        .unwrap();
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[1, 1, 3, 3], Some("kernel"))
        .unwrap();

    let result = inner.borrow_mut().create_conv_transpose2d_node(
        vec![input, kernel],
        (1, 1),
        (0, 0),
        (0, 0),
        Some("deconv"),
    );
    assert_err!(result, GraphError::ShapeMismatch { message, .. } if message.contains("4D"));
}

// ==================== 3. 端到端反向传播测试 ====================

/// 端到端反向传播：ConvTranspose2d → MSE Loss
#[test]
fn test_conv_transpose2d_e2e_backward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 构建: input → ConvTranspose2d → (sum) → MSE
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[1, 1, 2, 2], Some("kernel"))?;

    let deconv = inner.borrow_mut().create_conv_transpose2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        (0, 0),
        Some("deconv"),
    )?;

    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 3, 3], Some("target"))?;

    let mse = inner.borrow_mut().create_mse_node(
        Rc::clone(&deconv),
        Rc::clone(&target),
        crate::nn::nodes::raw_node::Reduction::Mean,
        Some("mse"),
    )?;

    input.set_value(Some(&Tensor::ones(&[1, 1, 2, 2])))?;
    kernel.set_value(Some(&Tensor::ones(&[1, 1, 2, 2])))?;
    target.set_value(Some(&Tensor::zeros(&[1, 1, 3, 3])))?;

    inner.borrow_mut().forward_via_node_inner(&mse)?;
    inner.borrow_mut().backward_via_node_inner(&mse)?;

    let k_grad = kernel.grad().expect("kernel 应有 grad");
    assert_eq!(k_grad.shape(), &[1, 1, 2, 2]);
    let grad_sum: f32 = k_grad.data_as_slice().iter().map(|v| v.abs()).sum();
    assert!(grad_sum > 0.0, "kernel 梯度不应全零");

    Ok(())
}

// ==================== 非连续内存（contiguity）回归测试 ====================

/// **回归测试**：ConvTranspose2d 前向/反向拿到**非连续 kernel**（上游是 `permute`）时不得 panic，
/// 且结果正确。
///
/// 历史脆弱点：`kernel.flatten_view()`（`into_shape`）对非连续视图会 panic。input 经
/// `input[[..]]`、upstream 经 `d_y[[..]]` 索引读取，对非连续本就安全，故只需覆盖 kernel。
///
/// 参考路径把非连续 kernel **物化为连续值**喂给 deconv；测试路径 `k → permute → deconv`。
/// 两者 forward 输出与 input 梯度应逐元素一致，且 kernel 的梯度存在（dK 路径不 panic）。
#[test]
fn test_conv_transpose2d_noncontiguous_kernel() {
    let input_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let kernel_base = Tensor::new(&[0.5, -0.5, 1.0, 2.0], &[1, 1, 2, 2]);
    // permute [0,1,3,2]：kH<->kW 转置，形状仍 [1,1,2,2] 但非连续
    let kernel_perm_contig = kernel_base.permute(&[0, 1, 3, 2]).into_contiguous();

    // 参考：直接用物化后的连续 kernel
    let (out_ref, gi_ref) = {
        let graph = Graph::new();
        let inner = graph.inner_rc();
        let input = inner
            .borrow_mut()
            .create_parameter_node(&[1, 1, 2, 2], Some("input"))
            .unwrap();
        let kernel = inner
            .borrow_mut()
            .create_parameter_node(&[1, 1, 2, 2], Some("kernel"))
            .unwrap();
        let deconv = inner
            .borrow_mut()
            .create_conv_transpose2d_node(
                vec![input.clone(), kernel.clone()],
                (1, 1),
                (0, 0),
                (0, 0),
                Some("deconv"),
            )
            .unwrap();
        let loss = inner
            .borrow_mut()
            .create_sum_node(Rc::clone(&deconv), None, None)
            .unwrap();
        input.set_value(Some(&input_val)).unwrap();
        kernel.set_value(Some(&kernel_perm_contig)).unwrap();
        inner.borrow_mut().forward_via_node_inner(&loss).unwrap();
        inner.borrow_mut().backward_via_node_inner(&loss).unwrap();
        (
            deconv.value().unwrap().clone(),
            input.grad().expect("input 应有 grad"),
        )
    };

    // 测试：k → permute → deconv，deconv 拿到非连续 kernel
    let (out_t, gi_t) = {
        let graph = Graph::new();
        let inner = graph.inner_rc();
        let input = inner
            .borrow_mut()
            .create_parameter_node(&[1, 1, 2, 2], Some("input"))
            .unwrap();
        let kernel = inner
            .borrow_mut()
            .create_parameter_node(&[1, 1, 2, 2], Some("kernel"))
            .unwrap();
        let kp = inner
            .borrow_mut()
            .create_permute_node(Rc::clone(&kernel), &[0, 1, 3, 2], None)
            .unwrap();
        let deconv = inner
            .borrow_mut()
            .create_conv_transpose2d_node(
                vec![input.clone(), kp],
                (1, 1),
                (0, 0),
                (0, 0),
                Some("deconv"),
            )
            .unwrap();
        let loss = inner
            .borrow_mut()
            .create_sum_node(Rc::clone(&deconv), None, None)
            .unwrap();
        input.set_value(Some(&input_val)).unwrap();
        kernel.set_value(Some(&kernel_base)).unwrap();
        inner.borrow_mut().forward_via_node_inner(&loss).unwrap();
        inner.borrow_mut().backward_via_node_inner(&loss).unwrap();
        assert!(
            kernel.grad().is_some(),
            "kernel 梯度应存在（dK 路径未 panic）"
        );
        (
            deconv.value().unwrap().clone(),
            input.grad().expect("input 应有 grad"),
        )
    };

    assert_eq!(out_t.shape(), out_ref.shape(), "forward 输出形状应一致");
    for (x, y) in out_t.to_vec().iter().zip(out_ref.to_vec().iter()) {
        assert_abs_diff_eq!(*x, *y, epsilon = 1e-5);
    }
    for (x, y) in gi_t.to_vec().iter().zip(gi_ref.to_vec().iter()) {
        assert_abs_diff_eq!(*x, *y, epsilon = 1e-5);
    }
}
