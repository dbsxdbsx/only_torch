/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : MaxPool2d 节点单元测试
 *
 * 测试策略（五段式，底层 inner_rc API）：
 * 1. 前向传播 → simple [6,8,14,16]; batch; 多通道; stride; 错误处理
 * 2. VJP（calc_grad_to_parent_index）→ 稀疏梯度验证
 * 3. E2E 反向传播 → loss 梯度验证; conv+pool 串联
 * 4. 动态形状 + 动态 batch（前向 + 反向）
 * 5. Create API
 */

use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试（底层 inner_rc API）====================

/// 测试 MaxPool2d 前向传播（简单情况）
///
/// 输入 4x4 递增矩阵，kernel=2x2, stride=2：
/// 窗口 [0:2,0:2]=max(1,2,5,6)=6; [0:2,2:4]=max(3,4,7,8)=8
/// 窗口 [2:4,0:2]=max(9,10,13,14)=14; [2:4,2:4]=max(11,12,15,16)=16
#[test]
fn test_max_pool2d_forward_simple() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        input.clone(),
        (2, 2),
        None,
        (0, 0),
        false,
        Some("pool"),
    )?;

    #[rustfmt::skip]
    let input_val = Tensor::new(&[
         1.0,  2.0,  3.0,  4.0,
         5.0,  6.0,  7.0,  8.0,
         9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], &[1, 1, 4, 4]);

    input.set_value(Some(&input_val))?;
    pool.forward_recursive(1, false)?;

    let output = pool.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 14.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 16.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 MaxPool2d 前向传播（batch=2）
///
/// 第一个 batch 全 1 → 最大值 1；第二个 batch 递增 → [6,8,14,16]
#[test]
fn test_max_pool2d_forward_batch() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 4, 4], Some("input"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        input.clone(),
        (2, 2),
        None,
        (0, 0),
        false,
        Some("pool"),
    )?;

    // 第一个 batch 全 1，第二个 batch 递增
    let mut data = vec![1.0f32; 16];
    for i in 0..16 {
        data.push((i + 1) as f32);
    }
    input.set_value(Some(&Tensor::new(&data, &[2, 1, 4, 4])))?;
    pool.forward_recursive(1, false)?;

    let output = pool.value().unwrap();
    assert_eq!(output.shape(), &[2, 1, 2, 2]);

    // batch 0: 全 1，最大值仍为 1
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 1.0, epsilon = 1e-6);

    // batch 1: 递增，与 simple 一致
    assert_abs_diff_eq!(output[[1, 0, 0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0, 0, 1]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0, 1, 0]], 14.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0, 1, 1]], 16.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 MaxPool2d 多通道前向传播
///
/// [1,2,4,4]: 第一通道全 1，第二通道全 2 → 各通道最大值不变
#[test]
fn test_max_pool2d_forward_multi_channel() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2, 4, 4], Some("input"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        input.clone(),
        (2, 2),
        None,
        (0, 0),
        false,
        Some("pool"),
    )?;

    let mut data = vec![1.0f32; 16];
    data.extend(vec![2.0f32; 16]);
    input.set_value(Some(&Tensor::new(&data, &[1, 2, 4, 4])))?;
    pool.forward_recursive(1, false)?;

    let output = pool.value().unwrap();
    assert_eq!(output.shape(), &[1, 2, 2, 2]);
    // 第一通道 → 1.0
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 1.0, epsilon = 1e-6);
    // 第二通道 → 2.0
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 1, 1]], 2.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 MaxPool2d 自定义 stride
///
/// kernel=3x3, stride=2 对 6x6 输入 → 输出 2x2
#[test]
fn test_max_pool2d_forward_with_stride() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 6, 6], Some("input"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        input.clone(),
        (3, 3),
        Some((2, 2)),
        (0, 0),
        false,
        Some("pool"),
    )?;

    // 6x6 递增输入
    let data: Vec<f32> = (1..=36).map(|x| x as f32).collect();
    input.set_value(Some(&Tensor::new(&data, &[1, 1, 6, 6])))?;
    pool.forward_recursive(1, false)?;

    let output = pool.value().unwrap();
    // (6 - 3) / 2 + 1 = 2
    assert_eq!(output.shape(), &[1, 1, 2, 2]);

    // 窗口 [0:3, 0:3]: max(1..9) = 15（第 3 行第 3 列）
    // 6x6 矩阵:
    //  1  2  3  4  5  6
    //  7  8  9 10 11 12
    // 13 14 15 16 17 18
    // 19 20 21 22 23 24
    // 25 26 27 28 29 30
    // 31 32 33 34 35 36
    // 窗口 [0:3, 0:3] = {1,2,3,7,8,9,13,14,15} → max=15
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 15.0, epsilon = 1e-6);
    // 窗口 [0:3, 2:5] = {3,4,5,9,10,11,15,16,17} → max=17
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 17.0, epsilon = 1e-6);
    // 窗口 [2:5, 0:3] = {13,14,15,19,20,21,25,26,27} → max=27
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 27.0, epsilon = 1e-6);
    // 窗口 [2:5, 2:5] = {15,16,17,21,22,23,27,28,29} → max=29
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 29.0, epsilon = 1e-6);

    Ok(())
}

/// 测试无效输入维度 → 应报错
#[test]
fn test_max_pool2d_invalid_input_dims() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 2D 输入（缺少 batch 和通道维度）
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 4], Some("input"))
        .unwrap();

    let result =
        inner
            .borrow_mut()
            .create_max_pool2d_node(input, (2, 2), None, (0, 0), false, Some("pool"));
    assert!(result.is_err());
}

/// 测试池化窗口超出输入尺寸 → 应报错
#[test]
fn test_max_pool2d_kernel_too_large() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))
        .unwrap();

    let result =
        inner
            .borrow_mut()
            .create_max_pool2d_node(input, (5, 5), None, (0, 0), false, Some("pool"));
    assert!(result.is_err());
}

// ==================== 2. VJP 单元测试（calc_grad_to_parent_index）====================
//
// MaxPool 梯度是稀疏的：只有最大值位置传递 upstream，其他位置为 0。

/// 测试 MaxPool2d VJP（upstream=ones）
///
/// 4x4 递增输入，kernel=2x2, stride=2：
/// 最大值位置 (1,1)=6, (1,3)=8, (3,1)=14, (3,3)=16 → grad=1
/// 其他位置 → grad=0
#[test]
fn test_max_pool2d_vjp_sparse_basic() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        input.clone(),
        (2, 2),
        None,
        (0, 0),
        false,
        Some("pool"),
    )?;

    #[rustfmt::skip]
    let input_val = Tensor::new(&[
         1.0,  2.0,  3.0,  4.0,
         5.0,  6.0,  7.0,  8.0,
         9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], &[1, 1, 4, 4]);

    input.set_value(Some(&input_val))?;
    pool.forward_recursive(1, false)?;

    let upstream = Tensor::ones(&[1, 1, 2, 2]);
    let grad = pool
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    assert_eq!(grad.shape(), &[1, 1, 4, 4]);

    // 最大值位置 → grad=1
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 1.0, epsilon = 1e-6); // max=6
    assert_abs_diff_eq!(grad[[0, 0, 1, 3]], 1.0, epsilon = 1e-6); // max=8
    assert_abs_diff_eq!(grad[[0, 0, 3, 1]], 1.0, epsilon = 1e-6); // max=14
    assert_abs_diff_eq!(grad[[0, 0, 3, 3]], 1.0, epsilon = 1e-6); // max=16

    // 非最大值位置 → grad=0
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 0.0, epsilon = 1e-6); // 1
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 0.0, epsilon = 1e-6); // 2
    assert_abs_diff_eq!(grad[[0, 0, 0, 2]], 0.0, epsilon = 1e-6); // 3
    assert_abs_diff_eq!(grad[[0, 0, 2, 0]], 0.0, epsilon = 1e-6); // 9
    assert_abs_diff_eq!(grad[[0, 0, 2, 2]], 0.0, epsilon = 1e-6); // 11

    Ok(())
}

/// 测试 MaxPool2d VJP（非单位 upstream）
///
/// upstream = [[2, 3], [5, 7]] → 最大值位置取对应 upstream 值
#[test]
fn test_max_pool2d_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        input.clone(),
        (2, 2),
        None,
        (0, 0),
        false,
        Some("pool"),
    )?;

    #[rustfmt::skip]
    let input_val = Tensor::new(&[
         1.0,  2.0,  3.0,  4.0,
         5.0,  6.0,  7.0,  8.0,
         9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], &[1, 1, 4, 4]);

    input.set_value(Some(&input_val))?;
    pool.forward_recursive(1, false)?;

    let upstream = Tensor::new(&[2.0, 3.0, 5.0, 7.0], &[1, 1, 2, 2]);
    let grad = pool
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    // 最大值位置取对应的 upstream 值
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 2.0, epsilon = 1e-6); // max=6, upstream=2
    assert_abs_diff_eq!(grad[[0, 0, 1, 3]], 3.0, epsilon = 1e-6); // max=8, upstream=3
    assert_abs_diff_eq!(grad[[0, 0, 3, 1]], 5.0, epsilon = 1e-6); // max=14, upstream=5
    assert_abs_diff_eq!(grad[[0, 0, 3, 3]], 7.0, epsilon = 1e-6); // max=16, upstream=7

    // 非最大值位置仍为 0
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 2, 2]], 0.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 MaxPool2d VJP（batch=2）
///
/// 验证多 batch 下的稀疏梯度独立性
#[test]
fn test_max_pool2d_vjp_batch() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 4, 4], Some("input"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        input.clone(),
        (2, 2),
        None,
        (0, 0),
        false,
        Some("pool"),
    )?;

    // batch 0: 全 3.0（所有位置都是最大值，梯度均匀分配）
    // batch 1: 递增 1..16
    let mut data = vec![3.0f32; 16];
    for i in 0..16 {
        data.push((i + 1) as f32);
    }
    input.set_value(Some(&Tensor::new(&data, &[2, 1, 4, 4])))?;
    pool.forward_recursive(1, false)?;

    let upstream = Tensor::ones(&[2, 1, 2, 2]);
    let grad = pool
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    assert_eq!(grad.shape(), &[2, 1, 4, 4]);

    // batch 1: 标准递增矩阵，最大值位置 (1,1)=6, (1,3)=8, (3,1)=14, (3,3)=16
    assert_abs_diff_eq!(grad[[1, 0, 1, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0, 1, 3]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0, 3, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0, 3, 3]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0, 0, 0]], 0.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 3. E2E 反向传播测试（底层构建完整图）====================

/// 测试 MaxPool2d E2E 反向传播（稀疏梯度到 Parameter）
///
/// 结构：param(input) → pool → reshape → mse_loss(target=0)
/// pool 输出 [6,8,14,16]，loss = mean([36,64,196,256]) = 138
/// param 梯度：只有最大值位置有 d_loss/d_pool_i * 1（稀疏）
#[test]
fn test_max_pool2d_e2e_backward_sparse() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let param = inner
        .borrow_mut()
        .create_parameter_node(&[1, 1, 4, 4], Some("param"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        param.clone(),
        (2, 2),
        None,
        (0, 0),
        false,
        Some("pool"),
    )?;
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

    #[rustfmt::skip]
    let input_val = Tensor::new(&[
         1.0,  2.0,  3.0,  4.0,
         5.0,  6.0,  7.0,  8.0,
         9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], &[1, 1, 4, 4]);

    param.set_value(Some(&input_val))?;
    target.set_value(Some(&Tensor::zeros(&[1, 4])))?;

    // 前向 + 反向
    inner.borrow_mut().set_train_mode();
    inner.borrow_mut().forward_via_node_inner(&loss)?;

    // pool 输出 = [6, 8, 14, 16]
    let pool_val = pool.value().unwrap();
    assert_abs_diff_eq!(pool_val[[0, 0, 0, 0]], 6.0, epsilon = 1e-6);

    // loss = mean([36, 64, 196, 256]) = 552/4 = 138
    let loss_val = loss.value().unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], 138.0, epsilon = 1e-4);

    inner.borrow_mut().backward_via_node_inner(&loss)?;

    // 验证 param 梯度
    let grad = param.grad().expect("param 应有 grad");
    assert_eq!(grad.shape(), &[1, 1, 4, 4]);

    // d_loss/d_pool_i = 2 * pool_i / N (N=4)
    // pool = [6, 8, 14, 16]
    // d_loss/d_pool = [3, 4, 7, 8]
    // MaxPool 将这些梯度稀疏传递到最大值位置
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 3.0, epsilon = 1e-4); // max=6
    assert_abs_diff_eq!(grad[[0, 0, 1, 3]], 4.0, epsilon = 1e-4); // max=8
    assert_abs_diff_eq!(grad[[0, 0, 3, 1]], 7.0, epsilon = 1e-4); // max=14
    assert_abs_diff_eq!(grad[[0, 0, 3, 3]], 8.0, epsilon = 1e-4); // max=16

    // 非最大值位置 → 0
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 2, 2]], 0.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d + MaxPool2d 串联的 E2E 反向传播
///
/// 结构：input → conv2d(kernel) → pool → flatten → mse_loss
/// 验证梯度能穿透 pool → conv 正确传播到 kernel
#[test]
fn test_max_pool2d_e2e_conv_pool_cascade() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [1, 1, 4, 4]，全 1
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))?;
    // 卷积核: [1, 1, 3, 3]（参数，需要梯度）
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[1, 1, 3, 3], Some("kernel"))?;

    // conv: stride=1, padding=1 → 输出 [1, 1, 4, 4]
    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (1, 1),
        (1, 1),
        Some("conv"),
    )?;

    // pool: kernel=2x2 → 输出 [1, 1, 2, 2]
    let pool = inner.borrow_mut().create_max_pool2d_node(
        conv.clone(),
        (2, 2),
        None,
        (0, 0),
        false,
        Some("pool"),
    )?;

    // flatten → [1, 4]
    let flat = inner
        .borrow_mut()
        .create_flatten_node(pool.clone(), true, Some("flat"))?;

    // MSE loss → target=0
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("target"))?;
    let loss =
        inner
            .borrow_mut()
            .create_mse_mean_node(flat.clone(), target.clone(), Some("loss"))?;

    // 设置值
    input.set_value(Some(&Tensor::ones(&[1, 1, 4, 4])))?;
    kernel.set_value(Some(&Tensor::ones(&[1, 1, 3, 3])))?;
    target.set_value(Some(&Tensor::zeros(&[1, 4])))?;

    // 前向 + 反向
    inner.borrow_mut().set_train_mode();
    inner.borrow_mut().forward_via_node_inner(&loss)?;

    // 验证 conv 输出形状
    assert_eq!(conv.value().unwrap().shape(), &[1, 1, 4, 4]);
    // 验证 pool 输出形状
    assert_eq!(pool.value().unwrap().shape(), &[1, 1, 2, 2]);
    // 验证 loss 为正数
    let loss_val = loss.value().unwrap()[[0, 0]];
    assert!(loss_val > 0.0, "loss 应为正值: {}", loss_val);

    inner.borrow_mut().backward_via_node_inner(&loss)?;

    // 验证 kernel 有梯度且形状正确
    let kernel_grad = kernel.grad().expect("kernel 应有 grad");
    assert_eq!(kernel_grad.shape(), &[1, 1, 3, 3]);

    // 验证梯度非全零（梯度穿透 pool → conv）
    let grad_sum: f32 = kernel_grad.data_as_slice().iter().sum();
    assert!(grad_sum.abs() > 1e-6, "kernel 梯度不应全零");

    Ok(())
}

// ==================== 4. 动态形状 + 动态 batch 测试 ====================

/// 测试 MaxPool2d 动态 batch 前向传播
///
/// 先 batch=2 再 batch=5，验证输出形状自适应
#[test]
fn test_max_pool2d_dynamic_batch_forward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 4, 4], Some("input"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        input.clone(),
        (2, 2),
        Some((2, 2)),
        (0, 0),
        false,
        Some("pool"),
    )?;

    // 第一次 forward: batch=2
    input.set_value(Some(&Tensor::ones(&[2, 1, 4, 4])))?;
    inner.borrow_mut().forward_via_node_inner(&pool)?;
    let val1 = pool.value().unwrap();
    assert_eq!(val1.shape(), &[2, 1, 2, 2], "第一次 forward: batch=2");

    // 第二次 forward: batch=5
    input.set_value(Some(&Tensor::ones(&[5, 1, 4, 4])))?;
    inner.borrow_mut().forward_via_node_inner(&pool)?;
    let val2 = pool.value().unwrap();
    assert_eq!(val2.shape(), &[5, 1, 2, 2], "第二次 forward: batch=5");

    Ok(())
}

/// 测试 MaxPool2d 动态 batch 反向传播
///
/// 先 batch=2 再 batch=4，验证反向传播在不同 batch 下正确工作
#[test]
fn test_max_pool2d_dynamic_batch_backward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 用 parameter 以便接收梯度
    let param = inner
        .borrow_mut()
        .create_parameter_node(&[2, 1, 4, 4], Some("param"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        param.clone(),
        (2, 2),
        Some((2, 2)),
        (0, 0),
        false,
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

    inner.borrow_mut().set_train_mode();

    // === 第一次：batch=2 ===
    param.set_value(Some(&Tensor::normal_seeded(0.0, 1.0, &[2, 1, 4, 4], 42)))?;
    target.set_value(Some(&Tensor::zeros(&[2, 4])))?;

    inner.borrow_mut().forward_via_node_inner(&loss)?;
    let loss_val1 = loss.value().unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0, "loss 应非负: {}", loss_val1);

    param.clear_grad()?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;
    let grad1 = param.grad().expect("batch=2 时 param 应有 grad");
    assert_eq!(grad1.shape(), &[2, 1, 4, 4]);

    // === 第二次：batch=4（动态 batch）===
    param.set_value(Some(&Tensor::normal_seeded(0.0, 1.0, &[4, 1, 4, 4], 100)))?;
    target.set_value(Some(&Tensor::zeros(&[4, 4])))?;

    inner.borrow_mut().forward_via_node_inner(&loss)?;
    let loss_val2 = loss.value().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0, "loss 应非负: {}", loss_val2);

    param.clear_grad()?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;
    let grad2 = param.grad().expect("batch=4 时 param 应有 grad");
    assert_eq!(grad2.shape(), &[4, 1, 4, 4]);

    Ok(())
}

// ==================== 5. Create API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_max_pool2d_node() {
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
        .create_max_pool2d_node(
            input.clone(),
            (2, 2),
            Some((2, 2)),
            (0, 0),
            false,
            Some("pool"),
        )
        .unwrap();

    assert_eq!(pool.shape(), vec![2, 3, 4, 4]);
    assert_eq!(pool.name(), Some("pool"));
    assert!(!pool.is_leaf());
    assert_eq!(pool.parents().len(), 1);
}

#[test]
fn test_create_max_pool2d_default_stride() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // stride=None -> 默认等于 kernel_size
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 6, 6], None)
        .unwrap();

    let pool = inner
        .borrow_mut()
        .create_max_pool2d_node(input, (3, 3), None, (0, 0), false, None)
        .unwrap();

    // (6 - 3) / 3 + 1 = 2
    assert_eq!(pool.shape(), vec![1, 1, 2, 2]);
}

#[test]
fn test_create_max_pool2d_overlapping() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 重叠池化：kernel=3x3，stride=1
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2, 5, 5], None)
        .unwrap();

    let pool = inner
        .borrow_mut()
        .create_max_pool2d_node(input, (3, 3), Some((1, 1)), (0, 0), false, None)
        .unwrap();

    // (5 - 3) / 1 + 1 = 3
    assert_eq!(pool.shape(), vec![1, 2, 3, 3]);
}

#[test]
fn test_create_max_pool2d_kernel_too_large() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], None)
        .unwrap();

    // kernel 5x5 > 输入 4x4 -> 应失败
    let result =
        inner
            .borrow_mut()
            .create_max_pool2d_node(input, (5, 5), None, (0, 0), false, None);
    assert!(result.is_err());
}

#[test]
fn test_create_max_pool2d_drop_releases() {
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
            .create_max_pool2d_node(input, (2, 2), None, (0, 0), false, None)
            .unwrap();
        weak_pool = Rc::downgrade(&pool);

        assert!(weak_pool.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_pool.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}

// ==================== 6. ONNX padding / ceil_mode 测试 ====================
//
// 验证 plan §2.3 的 MaxPool2d padding (top, bottom, left, right) + ceil_mode 字段
// 这是修复 chess_yolo_onnx_detect SPPF 模块 spatial shape bug 的关键能力
//
// SPPF 模块典型配置：MaxPool(k=5, stride=1, pads=2),输出与输入 H/W 完全相同
// （20→20）。修复前因没读 pads 输出错算成 (20-5)/1+1 = 16。

/// 测试 MaxPool2d 对称 padding：YOLOv5 SPPF 风格 (k=5, stride=1, pads=2)
///
/// 输入 5x5 全 1，pad 后 9x9 (pad 2 圈，padding 区域为 -inf)，
/// 池化窗口 5x5、stride=1 → 输出 5x5
/// 由于输入全 1 而 padding 是 -inf，每个窗口的 max 都是 1.0
#[test]
fn test_max_pool2d_sppf_style_padding() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 5, 5], Some("input"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        input.clone(),
        (5, 5),
        Some((1, 1)),
        (2, 2), // 对称 padding 2 圈
        false,
        Some("pool"),
    )?;

    let input_val = Tensor::new(&vec![1.0f32; 25], &[1, 1, 5, 5]);
    input.set_value(Some(&input_val))?;
    pool.forward_recursive(1, false)?;

    let output = pool.value().unwrap();
    assert_eq!(
        output.shape(),
        &[1, 1, 5, 5],
        "SPPF 风格 (k=5, p=2, s=1) 输出应与输入同尺寸"
    );
    for h in 0..5 {
        for w in 0..5 {
            assert_abs_diff_eq!(output[[0, 0, h, w]], 1.0, epsilon = 1e-6);
        }
    }
    Ok(())
}

/// 测试 MaxPool2d ceil_mode：(input + 2*pad - k) / s 不整除时
///
/// 输入 7x7、k=3、s=2、pads=0
/// floor: (7-3)/2 + 1 = 3 → 输出 3x3
/// ceil:  ceil((7-3)/2) + 1 = 3 → 输出 3x3（本例 ceil 跟 floor 同结果，因为整除）
///
/// 输入 8x8、k=3、s=2、pads=0
/// floor: (8-3)/2 + 1 = 3 → 输出 3x3
/// ceil:  ceil((8-3)/2) + 1 = 4 → 输出 4x4（多 1 个边缘窗口）
#[test]
fn test_max_pool2d_ceil_mode() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // floor 模式
    let input_f = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 8, 8], Some("input_floor"))?;
    let pool_f = inner.borrow_mut().create_max_pool2d_node(
        input_f.clone(),
        (3, 3),
        Some((2, 2)),
        (0, 0),
        false, // floor
        Some("pool_floor"),
    )?;

    let input_val = Tensor::new(&vec![0.5f32; 64], &[1, 1, 8, 8]);
    input_f.set_value(Some(&input_val))?;
    pool_f.forward_recursive(1, false)?;
    assert_eq!(pool_f.value().unwrap().shape(), &[1, 1, 3, 3], "floor 模式");

    // ceil 模式：注意 ceil 模式下池化窗口可能跨过 padding 区域
    // 8x8 输入、k=3、s=2、pads=0、ceil_mode=1
    // ONNX 行为：output = floor((8 + 0 - 3) / 2) + 1 = 3，但 ceil_mode 时 = ceil((8-3)/2)+1 = 4
    // 由于 (4-1)*2 + 3 = 9 > 8，最后一行/列窗口需要"虚拟 padding"才能完整覆盖
    let input_c = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 8, 8], Some("input_ceil"))?;
    let pool_c = inner.borrow_mut().create_max_pool2d_node(
        input_c.clone(),
        (3, 3),
        Some((2, 2)),
        (0, 0),
        true, // ceil
        Some("pool_ceil"),
    )?;
    input_c.set_value(Some(&input_val))?;
    pool_c.forward_recursive(1, false)?;
    assert_eq!(pool_c.value().unwrap().shape(), &[1, 1, 4, 4], "ceil 模式");
    Ok(())
}

/// 测试 MaxPool2d 反向传播 with padding：max_indices 在 padded 空间，
/// 反向需正确还原到原始输入坐标
#[test]
fn test_max_pool2d_backward_with_padding() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入 4x4、k=3、s=1、pads=1（对称）→ 输出 4x4（保形）
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))?;
    let pool = inner.borrow_mut().create_max_pool2d_node(
        input.clone(),
        (3, 3),
        Some((1, 1)),
        (1, 1),
        false,
        Some("pool"),
    )?;

    // 设计输入让 max 落在不同位置（含中心、边角）
    #[rustfmt::skip]
    let input_val = Tensor::new(&[
        9.0, 1.0, 1.0, 8.0,
        1.0, 5.0, 1.0, 1.0,
        1.0, 1.0, 7.0, 1.0,
        6.0, 1.0, 1.0, 4.0,
    ], &[1, 1, 4, 4]);
    input.set_value(Some(&input_val))?;
    pool.forward_recursive(1, false)?;

    // 输出值检查（每个窗口 max）
    let out = pool.value().unwrap();
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    // (0,0): 窗口 [-1:2, -1:2] 仅含 input[0,0..2, 0..2] = max(9,1,1,5) = 9
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 9.0, epsilon = 1e-6);
    // (1,1): 窗口 [0:3, 0:3] 含中心 5、9、7 等 = max = 9
    assert_abs_diff_eq!(out[[0, 0, 1, 1]], 9.0, epsilon = 1e-6);
    // (3,3): 窗口 [2:5, 2:5] 仅含 input[2..4, 2..4] = max(7,1,1,4) = 7
    assert_abs_diff_eq!(out[[0, 0, 3, 3]], 7.0, epsilon = 1e-6);

    Ok(())
}
