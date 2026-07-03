/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : AvgPool2d 节点单元测试
 *
 * 测试策略（底层 inner_rc API，AvgPool2d 无高层 Var API）：
 * 1. 前向传播测试 → simple; batch; 多通道; stride; 重叠窗口; 错误处理
 * 2. VJP 单元测试（calc_grad_to_parent_index）→ 全 1 input 梯度验证
 * 3. E2E 反向传播测试（完整图到 loss）→ loss 梯度验证; conv+pool 串联
 * 4. 动态形状 + 动态 batch（前向 + 反向）
 * 5. Create API 测试（保持原样）
 */

use crate::nn::Mode;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 1. 前向传播测试（底层 API）====================

/// 测试 AvgPool2d 前向传播（简单 4x4 → 2x2）
///
/// kernel=2x2, stride=2（默认），输入 1..16 递增
/// 窗口 [0:2,0:2]: avg(1,2,5,6)=3.5, [0:2,2:4]: avg(3,4,7,8)=5.5
/// 窗口 [2:4,0:2]: avg(9,10,13,14)=11.5, [2:4,2:4]: avg(11,12,15,16)=13.5
#[test]
fn test_avg_pool2d_forward_simple() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))?;
    let pool =
        inner
            .borrow_mut()
            .create_avg_pool2d_node(input.clone(), (2, 2), None, Some("pool"))?;

    #[rustfmt::skip]
    let tensor = Tensor::new(&[
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], &[1, 1, 4, 4]);

    input.set_value(Some(&tensor))?;
    pool.forward_recursive(1, Mode::Train)?;

    let output = pool.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 3.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 5.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 11.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 13.5, epsilon = 1e-6);

    Ok(())
}

/// 测试 AvgPool2d 前向传播（batch=2）
///
/// 第一个 batch 全 4 → 平均值 4，第二个 batch 全 8 → 平均值 8
#[test]
fn test_avg_pool2d_forward_batch() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 4, 4], Some("input"))?;
    let pool =
        inner
            .borrow_mut()
            .create_avg_pool2d_node(input.clone(), (2, 2), None, Some("pool"))?;

    let tensor = Tensor::new(
        &[vec![4.0f32; 16], vec![8.0f32; 16]].concat(),
        &[2, 1, 4, 4],
    );

    input.set_value(Some(&tensor))?;
    pool.forward_recursive(1, Mode::Train)?;

    let output = pool.value().unwrap();
    assert_eq!(output.shape(), &[2, 1, 2, 2]);

    // 第一个 batch 平均值全为 4
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 4.0, epsilon = 1e-6);
    // 第二个 batch 平均值全为 8
    assert_abs_diff_eq!(output[[1, 0, 0, 0]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0, 1, 1]], 8.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 AvgPool2d 多通道前向传播
///
/// [1,2,4,4]，通道 0 全 1 → avg=1，通道 1 全 2 → avg=2
#[test]
fn test_avg_pool2d_forward_multi_channel() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2, 4, 4], Some("input"))?;
    let pool =
        inner
            .borrow_mut()
            .create_avg_pool2d_node(input.clone(), (2, 2), None, Some("pool"))?;

    let tensor = Tensor::new(
        &[vec![1.0f32; 16], vec![2.0f32; 16]].concat(),
        &[1, 2, 4, 4],
    );

    input.set_value(Some(&tensor))?;
    pool.forward_recursive(1, Mode::Train)?;

    let output = pool.value().unwrap();
    assert_eq!(output.shape(), &[1, 2, 2, 2]);
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 2.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 AvgPool2d 自定义 stride
///
/// kernel=3x3, stride=2x2，输入 [1,1,6,6] → 输出 [1,1,2,2]
#[test]
fn test_avg_pool2d_forward_with_stride() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 6, 6], Some("input"))?;
    let pool = inner.borrow_mut().create_avg_pool2d_node(
        input.clone(),
        (3, 3),
        Some((2, 2)),
        Some("pool"),
    )?;

    assert_eq!(pool.shape(), vec![1, 1, 2, 2]);

    // 全 1 输入 → 平均值也全为 1
    let tensor = Tensor::ones(&[1, 1, 6, 6]);
    input.set_value(Some(&tensor))?;
    pool.forward_recursive(1, Mode::Train)?;

    let output = pool.value().unwrap();
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 1.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 AvgPool2d 重叠窗口（stride < kernel_size）
///
/// kernel=2x2, stride=1x1，输入 [1,1,4,4] → 输出 [1,1,3,3]
/// [0:2,0:2]: avg(1,2,5,6)=3.5, [0:2,1:3]: avg(2,3,6,7)=4.5, ...
#[test]
fn test_avg_pool2d_forward_overlapping_windows() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))?;
    let pool = inner.borrow_mut().create_avg_pool2d_node(
        input.clone(),
        (2, 2),
        Some((1, 1)),
        Some("pool"),
    )?;

    assert_eq!(pool.shape(), vec![1, 1, 3, 3]);

    #[rustfmt::skip]
    let tensor = Tensor::new(&[
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], &[1, 1, 4, 4]);

    input.set_value(Some(&tensor))?;
    pool.forward_recursive(1, Mode::Train)?;

    let output = pool.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 3, 3]);
    // 第一行
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 3.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 4.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 0, 2]], 5.5, epsilon = 1e-6);
    // 第二行
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 7.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 8.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 2]], 9.5, epsilon = 1e-6);
    // 第三行
    assert_abs_diff_eq!(output[[0, 0, 2, 0]], 11.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 2, 1]], 12.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 2, 2]], 13.5, epsilon = 1e-6);

    Ok(())
}

/// 测试无效输入维度（2D 输入 → 应报错）
#[test]
fn test_avg_pool2d_forward_invalid_dims() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 4], Some("input"))
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_avg_pool2d_node(input, (2, 2), None, Some("pool"));
    assert!(result.is_err());
}

/// 测试池化窗口过大（kernel 5x5 > 输入 4x4）
#[test]
fn test_avg_pool2d_forward_kernel_too_large() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_avg_pool2d_node(input, (5, 5), None, None);
    assert!(result.is_err());
}

// ==================== 2. VJP 单元测试（calc_grad_to_parent_index）====================
//
// AvgPool2d VJP：每个输出位置对应 kernel_size 个输入位置，
// 每个输入位置贡献 1/pool_size。对于全 1 upstream，
// 非重叠（stride=kernel）时每个位置梯度 = 1/pool_size = 0.25（2x2 窗口）。

/// 测试 VJP 全 1 upstream（非重叠窗口）
///
/// kernel=2x2, stride=2，全 1 input → 全 1 output
/// upstream 全 1 → 每个输入位置梯度 = 1/4 = 0.25
#[test]
fn test_avg_pool2d_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))?;
    let pool =
        inner
            .borrow_mut()
            .create_avg_pool2d_node(input.clone(), (2, 2), None, Some("pool"))?;

    input.set_value(Some(&Tensor::ones(&[1, 1, 4, 4])))?;
    pool.forward_recursive(1, Mode::Train)?;

    let upstream = Tensor::ones(&[1, 1, 2, 2]);
    let grad = pool
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_eq!(grad.shape(), &[1, 1, 4, 4]);
    // 非重叠窗口：每个输入位置仅被一个窗口覆盖，梯度 = 1 / 4 = 0.25
    for h in 0..4 {
        for w in 0..4 {
            assert_abs_diff_eq!(grad[[0, 0, h, w]], 0.25, epsilon = 1e-6);
        }
    }

    Ok(())
}

/// 测试 VJP 非单位 upstream
///
/// upstream = [[2,3],[4,5]]，梯度 = upstream * (1/pool_size)
#[test]
fn test_avg_pool2d_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))?;
    let pool =
        inner
            .borrow_mut()
            .create_avg_pool2d_node(input.clone(), (2, 2), None, Some("pool"))?;

    input.set_value(Some(&Tensor::ones(&[1, 1, 4, 4])))?;
    pool.forward_recursive(1, Mode::Train)?;

    // upstream_grad: 每个窗口不同值
    let upstream = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[1, 1, 2, 2]);
    let grad = pool
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_eq!(grad.shape(), &[1, 1, 4, 4]);
    // 窗口 [0:2,0:2] 对应 upstream=2，grad = 2/4 = 0.5
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 0.5, epsilon = 1e-6);
    // 窗口 [0:2,2:4] 对应 upstream=3，grad = 3/4 = 0.75
    assert_abs_diff_eq!(grad[[0, 0, 0, 2]], 0.75, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 0, 3]], 0.75, epsilon = 1e-6);
    // 窗口 [2:4,0:2] 对应 upstream=4，grad = 4/4 = 1.0
    assert_abs_diff_eq!(grad[[0, 0, 2, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 3, 1]], 1.0, epsilon = 1e-6);
    // 窗口 [2:4,2:4] 对应 upstream=5，grad = 5/4 = 1.25
    assert_abs_diff_eq!(grad[[0, 0, 2, 2]], 1.25, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 3, 3]], 1.25, epsilon = 1e-6);

    Ok(())
}

/// 测试 VJP batch 维度
///
/// [2,1,4,4]，全 1 input，全 1 upstream → 每个位置梯度 0.25
#[test]
fn test_avg_pool2d_vjp_batch() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 4, 4], Some("input"))?;
    let pool =
        inner
            .borrow_mut()
            .create_avg_pool2d_node(input.clone(), (2, 2), None, Some("pool"))?;

    input.set_value(Some(&Tensor::ones(&[2, 1, 4, 4])))?;
    pool.forward_recursive(1, Mode::Train)?;

    let upstream = Tensor::ones(&[2, 1, 2, 2]);
    let grad = pool
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_eq!(grad.shape(), &[2, 1, 4, 4]);
    // 两个 batch 梯度都是 0.25
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 3, 3]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0, 0, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0, 3, 3]], 0.25, epsilon = 1e-6);

    Ok(())
}

/// 测试 VJP 重叠窗口（stride=1, kernel=2）
///
/// 重叠窗口下，中心输入位置被多个窗口覆盖，梯度叠加
#[test]
fn test_avg_pool2d_vjp_overlapping() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 3, 3], Some("input"))?;
    // kernel=2x2, stride=1x1 → output [1,1,2,2]
    let pool = inner.borrow_mut().create_avg_pool2d_node(
        input.clone(),
        (2, 2),
        Some((1, 1)),
        Some("pool"),
    )?;

    assert_eq!(pool.shape(), vec![1, 1, 2, 2]);

    input.set_value(Some(&Tensor::ones(&[1, 1, 3, 3])))?;
    pool.forward_recursive(1, Mode::Train)?;

    let upstream = Tensor::ones(&[1, 1, 2, 2]);
    let grad = pool
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_eq!(grad.shape(), &[1, 1, 3, 3]);
    // 角落位置 (0,0): 仅被 1 个窗口覆盖 → 0.25
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 0.25, epsilon = 1e-6);
    // 边缘位置 (0,1): 被 2 个窗口覆盖 → 0.25 * 2 = 0.5
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 0.5, epsilon = 1e-6);
    // 中心位置 (1,1): 被 4 个窗口覆盖 → 0.25 * 4 = 1.0
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 1.0, epsilon = 1e-6);
    // 角落位置 (2,2): 仅被 1 个窗口覆盖 → 0.25
    assert_abs_diff_eq!(grad[[0, 0, 2, 2]], 0.25, epsilon = 1e-6);

    Ok(())
}

// ==================== 3. E2E 反向传播测试（底层完整图到 loss）====================

/// 测试 E2E：input(parameter) → pool → flatten → MSE loss
///
/// 全 1 input → pool 输出全 1 → loss = mean((1-0)^2) = 1
/// d_loss/d_pool = 2*(pool-target)/n = 2*1/4 = 0.5 → 每个位置
/// d_pool/d_input = 1/pool_size = 0.25
/// d_loss/d_input = 0.5 * 0.25 = 0.125
#[test]
fn test_avg_pool2d_e2e_loss_grad() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // parameter 节点可累积梯度
    let input = inner
        .borrow_mut()
        .create_parameter_node(&[1, 1, 4, 4], Some("input"))?;
    let pool =
        inner
            .borrow_mut()
            .create_avg_pool2d_node(input.clone(), (2, 2), None, Some("pool"))?;
    let flat = inner
        .borrow_mut()
        .create_reshape_node(pool.clone(), &[1, 4], Some("flat"))?;
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("target"))?;
    let loss =
        inner
            .borrow_mut()
            .create_mse_mean_node(flat.clone(), target.clone(), Some("loss"))?;

    // 设值：全 1 input，全 0 target
    input.set_value(Some(&Tensor::ones(&[1, 1, 4, 4])))?;
    target.set_value(Some(&Tensor::zeros(&[1, 4])))?;

    // 前向
    loss.forward_recursive(1, Mode::Train)?;

    let loss_val = loss.value().unwrap();
    // loss = mean([1,1,1,1]^2) = 1.0
    assert_abs_diff_eq!(loss_val[[0, 0]], 1.0, epsilon = 1e-5);

    // 反向：通过 backward_via_node_inner 自动处理种子梯度和中间节点清理
    inner.borrow_mut().backward_via_node_inner(&loss)?;

    let grad = input.grad().expect("input 应有 grad");
    assert_eq!(grad.shape(), &[1, 1, 4, 4]);
    // 每个输入位置梯度 = 0.5 * 0.25 = 0.125
    for h in 0..4 {
        for w in 0..4 {
            assert_abs_diff_eq!(grad[[0, 0, h, w]], 0.125, epsilon = 1e-5);
        }
    }

    Ok(())
}

/// 测试 E2E：Conv2d + AvgPool2d 串联
///
/// input [2,1,8,8] → conv [2,4,8,8] → pool [2,4,4,4]
/// 验证形状正确、前向传播不报错
#[test]
fn test_avg_pool2d_e2e_conv_pool_chain() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入和卷积核
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 8, 8], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[4, 1, 3, 3], Some("kernel"))?;

    // Conv2d: stride=1, padding=1 → 保持空间尺寸 [2,4,8,8]
    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (1, 1),
        (1, 1),
        Some("conv"),
    )?;
    assert_eq!(conv.shape(), vec![2, 4, 8, 8]);

    // AvgPool2d: kernel=2x2 → [2,4,4,4]
    let pool =
        inner
            .borrow_mut()
            .create_avg_pool2d_node(conv.clone(), (2, 2), None, Some("pool"))?;
    assert_eq!(pool.shape(), vec![2, 4, 4, 4]);

    // 全 1 输入 + 全 1 卷积核
    input.set_value(Some(&Tensor::ones(&[2, 1, 8, 8])))?;
    kernel.set_value(Some(&Tensor::ones(&[4, 1, 3, 3])))?;

    pool.forward_recursive(1, Mode::Train)?;

    let output = pool.value().unwrap();
    assert_eq!(output.shape(), &[2, 4, 4, 4]);
    // 全 1 输入通过全 1 kernel 卷积（padding=1），中心区域值 = 9（3x3 全 1 求和）
    // 池化后中心区域值仍为 9（均匀值取平均不变）
    // 验证输出非零
    assert!(output[[0, 0, 1, 1]] > 0.0);

    Ok(())
}

// ==================== 4. 动态形状 + 动态 batch（前向 + 反向）====================

/// 测试动态形状传播
///
/// batch 维度应标记为动态，channels/H/W 维度固定
#[test]
fn test_avg_pool2d_dynamic_shape_propagation() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 8, 8], Some("input"))?;
    let pool = inner.borrow_mut().create_avg_pool2d_node(
        input.clone(),
        (2, 2),
        Some((2, 2)),
        Some("pool"),
    )?;

    let dyn_shape = pool.dynamic_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "channels 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(3));

    Ok(())
}

/// 测试动态 batch 前向传播
///
/// 先 batch=2 再 batch=5，输出形状应自动适配
#[test]
fn test_avg_pool2d_dynamic_batch_forward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 4, 4], Some("input"))?;
    let pool = inner.borrow_mut().create_avg_pool2d_node(
        input.clone(),
        (2, 2),
        Some((2, 2)),
        Some("pool"),
    )?;

    // 第一次 forward：batch=2
    input.set_value(Some(&Tensor::ones(&[2, 1, 4, 4])))?;
    pool.forward_recursive(1, Mode::Train)?;
    let value1 = pool.value().unwrap();
    assert_eq!(value1.shape(), &[2, 1, 2, 2], "第一次 forward: batch=2");

    // 第二次 forward：batch=5
    input.set_value(Some(&Tensor::ones(&[5, 1, 4, 4])))?;
    pool.forward_recursive(2, Mode::Train)?;
    let value2 = pool.value().unwrap();
    assert_eq!(value2.shape(), &[5, 1, 2, 2], "第二次 forward: batch=5");

    Ok(())
}

/// 测试动态 batch 反向传播
///
/// 构建 input → pool → flatten → MSE loss 完整图，
/// 先 batch=2 再 batch=4，验证两次都能成功反向传播
#[test]
fn test_avg_pool2d_dynamic_batch_backward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 1, 4, 4], Some("input"))?;
    let pool = inner.borrow_mut().create_avg_pool2d_node(
        input.clone(),
        (2, 2),
        Some((2, 2)),
        Some("pool"),
    )?;
    let flat = inner
        .borrow_mut()
        .create_flatten_node(pool.clone(), true, Some("flat"))?;
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("target"))?;
    let loss =
        inner
            .borrow_mut()
            .create_mse_mean_node(flat.clone(), target.clone(), Some("loss"))?;

    // 第一次训练：batch=2
    input.set_value(Some(&Tensor::normal_seeded(0.0, 1.0, &[2, 1, 4, 4], 42)))?;
    target.set_value(Some(&Tensor::zeros(&[2, 4])))?;

    loss.forward_recursive(1, Mode::Train)?;
    let loss_val1 = loss.value().unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);

    inner.borrow_mut().backward_via_node_inner(&loss)?;

    let grad1 = input.grad().expect("第一次反向应有 grad");
    assert_eq!(grad1.shape(), &[2, 1, 4, 4]);

    // 第二次训练：batch=4（动态 batch 变化）
    // 清除参数梯度（batch 尺寸变化，旧梯度形状不兼容）
    input.clear_grad()?;

    input.set_value(Some(&Tensor::normal_seeded(0.0, 1.0, &[4, 1, 4, 4], 100)))?;
    target.set_value(Some(&Tensor::zeros(&[4, 4])))?;

    loss.forward_recursive(2, Mode::Train)?;
    let loss_val2 = loss.value().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);

    inner.borrow_mut().backward_via_node_inner(&loss)?;

    let grad2 = input.grad().expect("第二次反向应有 grad");
    assert_eq!(grad2.shape(), &[4, 1, 4, 4]);

    Ok(())
}

// ==================== 5. Create API 测试（保持原样）====================

#[test]
fn test_create_avg_pool2d_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [2, 3, 8, 8]，kernel=2x2，stride=2
    // 输出: [2, 3, 4, 4]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 8, 8], Some("input"))
        .unwrap();

    let pool = inner
        .borrow_mut()
        .create_avg_pool2d_node(input.clone(), (2, 2), Some((2, 2)), Some("pool"))
        .unwrap();

    assert_eq!(pool.shape(), vec![2, 3, 4, 4]);
    assert_eq!(pool.name(), Some("pool"));
    assert!(!pool.is_leaf());
    assert_eq!(pool.parents().len(), 1);
}

#[test]
fn test_create_avg_pool2d_default_stride() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // stride=None -> 默认等于 kernel_size
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 6, 6], None)
        .unwrap();

    let pool = inner
        .borrow_mut()
        .create_avg_pool2d_node(input, (3, 3), None, None)
        .unwrap();

    // (6 - 3) / 3 + 1 = 2
    assert_eq!(pool.shape(), vec![1, 1, 2, 2]);
}

#[test]
fn test_create_avg_pool2d_overlapping() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 重叠池化：kernel=3x3，stride=1
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2, 5, 5], None)
        .unwrap();

    let pool = inner
        .borrow_mut()
        .create_avg_pool2d_node(input, (3, 3), Some((1, 1)), None)
        .unwrap();

    // (5 - 3) / 1 + 1 = 3
    assert_eq!(pool.shape(), vec![1, 2, 3, 3]);
}

#[test]
fn test_create_avg_pool2d_kernel_too_large() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], None)
        .unwrap();

    // kernel 5x5 > 输入 4x4 -> 应失败
    let result = inner
        .borrow_mut()
        .create_avg_pool2d_node(input, (5, 5), None, None);
    assert!(result.is_err());
}

#[test]
fn test_create_avg_pool2d_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_pool;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[1, 1, 8, 8], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let pool = inner
            .borrow_mut()
            .create_avg_pool2d_node(input, (2, 2), None, None)
            .unwrap();
        weak_pool = Rc::downgrade(&pool);

        assert!(weak_pool.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_pool.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}

// ==================== 7. 平铺直写重构 vs 旧链逐 bit 金测试 ====================
//
// 2026-07-03 优化（优化候选 #6）：前向/反向从「IxDyn 逐元素索引 + Vec<Vec> flatten」
// 改为「contiguous 守卫 + 平铺 slice + par_chunks_mut 直写」。
// 本组测试以旧实现（IxDyn 索引、串行）为参考实现，锁死逐 bit 一致性。

/// 旧实现参考：前向（IxDyn 索引逐窗口累加，窗口内 (kh, kw) 行序）
fn avg_pool2d_reference_forward(
    input: &Tensor,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> Tensor {
    let shape = input.shape();
    let (batch_size, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
    let (k_h, k_w) = kernel;
    let (s_h, s_w) = stride;
    let out_h = (in_h - k_h) / s_h + 1;
    let out_w = (in_w - k_w) / s_w + 1;
    let pool_size = (k_h * k_w) as f32;

    let mut all_data = Vec::with_capacity(batch_size * channels * out_h * out_w);
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            sum += input[[b, c, oh * s_h + kh, ow * s_w + kw]];
                        }
                    }
                    all_data.push(sum / pool_size);
                }
            }
        }
    }
    Tensor::new(&all_data, &[batch_size, channels, out_h, out_w])
}

/// 旧实现参考：反向（`upstream * grad_val` 在窗口循环内逐次相乘）
fn avg_pool2d_reference_backward(
    upstream: &Tensor,
    input_shape: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
) -> Tensor {
    let grad_shape = upstream.shape();
    let (batch_size, channels, out_h, out_w) =
        (grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]);
    let (k_h, k_w) = kernel;
    let (s_h, s_w) = stride;
    let grad_val = 1.0 / (k_h * k_w) as f32;
    let (in_h, in_w) = (input_shape[2], input_shape[3]);
    let single_sample_size = channels * in_h * in_w;

    let mut all_data = vec![0.0f32; batch_size * single_sample_size];
    for b in 0..batch_size {
        let sample_grad = &mut all_data[b * single_sample_size..(b + 1) * single_sample_size];
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let up = upstream[[b, c, oh, ow]];
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let ih = oh * s_h + kh;
                            let iw = ow * s_w + kw;
                            sample_grad[c * in_h * in_w + ih * in_w + iw] += up * grad_val;
                        }
                    }
                }
            }
        }
    }
    Tensor::new(&all_data, input_shape)
}

fn assert_bitwise_eq(actual: &Tensor, expected: &Tensor, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{ctx}: 形状不一致");
    let a = actual.data_as_slice();
    let e = expected.data_as_slice();
    for i in 0..a.len() {
        assert_eq!(
            a[i].to_bits(),
            e[i].to_bits(),
            "{ctx}: i={i} actual={} expected={}",
            a[i],
            e[i]
        );
    }
}

/// 平铺直写前向/反向 vs 旧实现逐 bit（含默认 stride、非方核、重叠窗口、多 batch 多通道）
#[test]
fn test_avg_pool2d_bitwise_matches_reference() -> Result<(), GraphError> {
    // (输入形状, kernel, stride)
    let cases: &[(&[usize], (usize, usize), Option<(usize, usize)>)] = &[
        (&[2, 3, 8, 8], (2, 2), None),          // 默认 stride = kernel
        (&[3, 2, 9, 11], (2, 3), Some((2, 2))), // 非方核 + 非整除尺寸
        (&[1, 4, 7, 7], (3, 3), Some((1, 1))),  // 重叠窗口
    ];

    for (case_idx, &(shape, kernel, stride)) in cases.iter().enumerate() {
        let graph = Graph::new();
        let inner = graph.inner_rc();

        let input = inner
            .borrow_mut()
            .create_basic_input_node(shape, Some("input"))?;
        let pool = inner.borrow_mut().create_avg_pool2d_node(
            input.clone(),
            kernel,
            stride,
            Some("pool"),
        )?;

        let input_val = Tensor::normal_seeded(0.0, 1.0, shape, 42 + case_idx as u64);
        input.set_value(Some(&input_val))?;
        pool.forward_recursive(1, Mode::Train)?;

        let effective_stride = stride.unwrap_or(kernel);
        let expected_fwd = avg_pool2d_reference_forward(&input_val, kernel, effective_stride);
        let actual_fwd = pool.value().unwrap();
        assert_bitwise_eq(
            &actual_fwd,
            &expected_fwd,
            &format!("avg_pool2d 前向 case {case_idx}"),
        );

        let upstream = Tensor::normal_seeded(0.0, 1.0, expected_fwd.shape(), 100 + case_idx as u64);
        let grad = pool
            .calc_grad_to_parent_index(0, &upstream)?
            .resolve(&upstream);
        let expected_bwd =
            avg_pool2d_reference_backward(&upstream, shape, kernel, effective_stride);
        assert_bitwise_eq(
            &grad,
            &expected_bwd,
            &format!("avg_pool2d 反向 case {case_idx}"),
        );
    }

    Ok(())
}
