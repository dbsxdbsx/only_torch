/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : AvgPool2d 节点单元测试
 *
 * 测试策略：
 * 1. 基础功能测试（形状、前向传播）
 * 2. Jacobi 模式反向传播（单样本）
 * 3. Batch 模式反向传播
 * 4. 各种参数组合（kernel_size, stride）
 */

use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 AvgPool2d 节点创建（单样本）
#[test]
fn test_avg_pool2d_creation_single() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [C=1, H=4, W=4]
    let input = graph.new_input_node(&[1, 4, 4], Some("input"))?;

    // 创建 AvgPool2d: kernel_size=2x2, stride=2x2（默认）
    let pool = graph.new_avg_pool2d_node(input, (2, 2), None, Some("pool"))?;

    // 验证输出形状: [C=1, H'=2, W'=2]
    let output_shape = graph.get_node(pool)?.value_expected_shape();
    assert_eq!(output_shape, &[1, 2, 2]);

    Ok(())
}

/// 测试 AvgPool2d 节点创建（Batch）
#[test]
fn test_avg_pool2d_creation_batch() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [batch=4, C=16, H=28, W=28]
    let input = graph.new_input_node(&[4, 16, 28, 28], Some("input"))?;

    // 创建 AvgPool2d: kernel_size=2x2, stride=2x2
    let pool = graph.new_avg_pool2d_node(input, (2, 2), None, Some("pool"))?;

    // 验证输出形状: [batch=4, C=16, H'=14, W'=14]
    let output_shape = graph.get_node(pool)?.value_expected_shape();
    assert_eq!(output_shape, &[4, 16, 14, 14]);

    Ok(())
}

/// 测试 AvgPool2d 带自定义 stride
#[test]
fn test_avg_pool2d_with_stride() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [batch=2, C=1, H=6, W=6]
    let input = graph.new_input_node(&[2, 1, 6, 6], Some("input"))?;

    // 创建 AvgPool2d: kernel_size=3x3, stride=2x2
    let pool = graph.new_avg_pool2d_node(input, (3, 3), Some((2, 2)), Some("pool"))?;

    // 验证输出形状: [batch=2, C=1, H'=2, W'=2]
    let output_shape = graph.get_node(pool)?.value_expected_shape();
    assert_eq!(output_shape, &[2, 1, 2, 2]);

    Ok(())
}

// ==================== 前向传播测试 ====================

/// 测试 AvgPool2d 前向传播（简单情况）
#[test]
fn test_avg_pool2d_forward_simple() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [C=1, H=4, W=4]
    let input = graph.new_input_node(&[1, 4, 4], Some("input"))?;
    let pool = graph.new_avg_pool2d_node(input, (2, 2), None, Some("pool"))?;

    // 设置输入值
    #[rustfmt::skip]
    let input_val = Tensor::new(&[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], &[1, 4, 4]);

    graph.set_node_value(input, Some(&input_val))?;
    graph.forward(pool)?;

    // 验证输出
    // 窗口 [0:2, 0:2]: avg(1,2,5,6) = 14/4 = 3.5
    // 窗口 [0:2, 2:4]: avg(3,4,7,8) = 22/4 = 5.5
    // 窗口 [2:4, 0:2]: avg(9,10,13,14) = 46/4 = 11.5
    // 窗口 [2:4, 2:4]: avg(11,12,15,16) = 54/4 = 13.5
    let output = graph.get_node_value(pool)?.unwrap();
    assert_eq!(output.shape(), &[1, 2, 2]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 3.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1]], 5.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 0]], 11.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 1]], 13.5, epsilon = 1e-6);

    Ok(())
}

/// 测试 AvgPool2d 前向传播（Batch 模式）
#[test]
fn test_avg_pool2d_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [batch=2, C=1, H=4, W=4]
    let input = graph.new_input_node(&[2, 1, 4, 4], Some("input"))?;
    let pool = graph.new_avg_pool2d_node(input, (2, 2), None, Some("pool"))?;

    // 设置输入：第一个 batch 全 4，第二个 batch 全 8
    let input_val = Tensor::new(
        &[vec![4.0f32; 16], vec![8.0f32; 16]].concat(),
        &[2, 1, 4, 4],
    );

    graph.set_node_value(input, Some(&input_val))?;
    graph.forward(pool)?;

    let output = graph.get_node_value(pool)?.unwrap();
    assert_eq!(output.shape(), &[2, 1, 2, 2]);

    // 第一个 batch: 平均值都是 4
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    // 第二个 batch: 平均值都是 8
    assert_abs_diff_eq!(output[[1, 0, 0, 0]], 8.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 AvgPool2d 多通道
#[test]
fn test_avg_pool2d_multi_channel() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [C=2, H=4, W=4]
    let input = graph.new_input_node(&[2, 4, 4], Some("input"))?;
    let pool = graph.new_avg_pool2d_node(input, (2, 2), None, Some("pool"))?;

    // 设置输入：第一通道全 1，第二通道全 2
    let input_val = Tensor::new(&[vec![1.0f32; 16], vec![2.0f32; 16]].concat(), &[2, 4, 4]);

    graph.set_node_value(input, Some(&input_val))?;
    graph.forward(pool)?;

    let output = graph.get_node_value(pool)?.unwrap();
    assert_eq!(output.shape(), &[2, 2, 2]);

    // 第一通道平均值都是 1
    assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-6);
    // 第二通道平均值都是 2
    assert_abs_diff_eq!(output[[1, 0, 0]], 2.0, epsilon = 1e-6);

    Ok(())
}

// ==================== Jacobi 模式测试（单样本）====================

/// 测试 AvgPool2d Jacobi 矩阵
#[test]
/// 测试 AvgPool2d 梯度（VJP 模式）
///
/// 构建完整计算图：input -> pool -> flatten -> mse_loss
fn test_avg_pool2d_jacobi() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [C=1, H=4, W=4]，使用 Parameter 以便计算梯度
    // pool 输出: [C=1, H=2, W=2]（kernel=2x2, stride=2）
    let input = graph.new_parameter_node(&[1, 4, 4], Some("input"))?;
    let pool = graph.new_avg_pool2d_node(input, (2, 2), None, Some("pool"))?;

    // 将 pool 输出 reshape 为 [1, 4] 并添加 MSE loss
    let pool_flat = graph.new_reshape_node(pool, &[1, 4], Some("pool_flat"))?;
    let target = graph.new_input_node(&[1, 4], Some("target"))?;
    let loss = graph.new_mse_loss_node(pool_flat, target, Some("loss"))?;

    // 设置输入值
    let input_val = Tensor::ones(&[1, 4, 4]);
    let target_val = Tensor::zeros(&[1, 4]);

    graph.set_node_value(input, Some(&input_val))?;
    graph.set_node_value(target, Some(&target_val))?;

    // 前向传播
    graph.forward(loss)?;

    // pool 输出全为 1（因为输入全为 1，取平均还是 1）
    // loss = mean((1-0)^2 * 4) = 1

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // VJP 模式下验证 grad 形状应与输入值一致：[1, 4, 4]
    let grad = graph.get_node(input)?.grad().expect("应有 grad");
    assert_eq!(grad.shape(), &[1, 4, 4]);

    // d_loss/d_pool = 2 * (pool - target) / 4 = 0.5 * [1,1,1,1] = [0.5, 0.5, 0.5, 0.5]
    // d_pool/d_input 对每个 2x2 区域：每个输入位置贡献 0.25 到对应输出
    // d_loss/d_input = 0.5 * 0.25 = 0.125
    assert_abs_diff_eq!(grad[[0, 0, 0]], 0.125, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1]], 0.125, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1, 0]], 0.125, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1, 1]], 0.125, epsilon = 1e-6);

    Ok(())
}

// ==================== Batch 模式测试 ====================

/// 测试 AvgPool2d Batch 梯度
#[test]
fn test_avg_pool2d_batch_grad() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [batch=1, C=1, H=4, W=4]
    let input_id = graph.new_input_node(&[1, 1, 4, 4], Some("input"))?;
    let pool_id = graph.new_avg_pool2d_node(input_id, (2, 2), None, Some("pool"))?;

    let input_val = Tensor::ones(&[1, 1, 4, 4]);

    graph.set_node_value(input_id, Some(&input_val))?;
    graph.forward(pool_id)?;

    // upstream_grad 全 1
    let upstream_grad = Tensor::ones(&[1, 1, 2, 2]);

    let pool_node = graph.get_node(pool_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = pool_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    assert_eq!(grad.shape(), &[1, 1, 4, 4]);

    // AvgPool 梯度：每个位置的梯度 = upstream_grad / pool_size = 1 / 4 = 0.25
    // 由于窗口不重叠（stride=kernel_size），每个输入位置只被一个窗口覆盖
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 2, 2]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 3, 3]], 0.25, epsilon = 1e-6);

    Ok(())
}

/// 测试 AvgPool2d 与 Conv2d 串联
#[test]
fn test_avg_pool2d_after_conv2d() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 输入: [batch=2, C_in=1, H=8, W=8]
    let input = graph.new_input_node(&[2, 1, 8, 8], Some("input"))?;
    // 卷积核: [C_out=4, C_in=1, kH=3, kW=3]
    let kernel = graph.new_parameter_node(&[4, 1, 3, 3], Some("kernel"))?;

    // Conv2d: stride=1, padding=1（保持尺寸）
    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (1, 1), Some("conv"))?;
    // 输出: [2, 4, 8, 8]

    // AvgPool2d: 2x2
    let pool = graph.new_avg_pool2d_node(conv, (2, 2), None, Some("pool"))?;
    // 输出: [2, 4, 4, 4]

    // 验证形状
    assert_eq!(graph.get_node(conv)?.value_expected_shape(), &[2, 4, 8, 8]);
    assert_eq!(graph.get_node(pool)?.value_expected_shape(), &[2, 4, 4, 4]);

    // 设置输入并前向传播
    let input_val = Tensor::ones(&[2, 1, 8, 8]);
    let kernel_val = Tensor::ones(&[4, 1, 3, 3]);

    graph.set_node_value(input, Some(&input_val))?;
    graph.set_node_value(kernel, Some(&kernel_val))?;

    graph.forward(pool)?;

    // 验证池化输出有值
    let pool_output = graph.get_node_value(pool)?.unwrap();
    assert_eq!(pool_output.shape(), &[2, 4, 4, 4]);

    Ok(())
}

/// 测试 AvgPool2d 重叠窗口（stride < kernel_size）
#[test]
fn test_avg_pool2d_overlapping_windows() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [C=1, H=4, W=4]
    let input = graph.new_input_node(&[1, 4, 4], Some("input"))?;
    // kernel_size=2x2, stride=1x1（重叠窗口）
    let pool = graph.new_avg_pool2d_node(input, (2, 2), Some((1, 1)), Some("pool"))?;

    // 输出形状: [C=1, H'=3, W'=3]
    let output_shape = graph.get_node(pool)?.value_expected_shape();
    assert_eq!(output_shape, &[1, 3, 3]);

    // 设置输入并前向传播
    #[rustfmt::skip]
    let input_val = Tensor::new(&[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], &[1, 4, 4]);

    graph.set_node_value(input, Some(&input_val))?;
    graph.forward(pool)?;

    let output = graph.get_node_value(pool)?.unwrap();
    // 窗口 [0:2, 0:2]: avg(1,2,5,6) = 3.5
    assert_abs_diff_eq!(output[[0, 0, 0]], 3.5, epsilon = 1e-6);
    // 窗口 [0:2, 1:3]: avg(2,3,6,7) = 4.5
    assert_abs_diff_eq!(output[[0, 0, 1]], 4.5, epsilon = 1e-6);

    Ok(())
}

// ==================== 错误处理测试 ====================

/// 测试无效的输入维度
#[test]
fn test_avg_pool2d_invalid_input_dims() {
    let mut graph = GraphInner::new();

    // 输入: [H=4, W=4]（2D，缺少通道维度）
    let input = graph.new_input_node(&[4, 4], Some("input")).unwrap();

    let result = graph.new_avg_pool2d_node(input, (2, 2), None, Some("pool"));
    assert!(result.is_err());
}

/// 测试池化窗口过大
#[test]
fn test_avg_pool2d_kernel_too_large() {
    let mut graph = GraphInner::new();

    // 输入: [C=1, H=4, W=4]
    let input = graph.new_input_node(&[1, 4, 4], Some("input")).unwrap();

    // kernel_size=5x5 超出输入尺寸
    let result = graph.new_avg_pool2d_node(input, (5, 5), None, Some("pool"));
    assert!(result.is_err());
}

// ==================== 动态形状测试 ====================

/// 测试 AvgPool2d 节点的动态形状传播
#[test]
fn test_avg_pool2d_dynamic_shape_propagation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建 4D 输入：[batch, channels, height, width]
    let input = graph.new_input_node(&[2, 3, 8, 8], Some("input"))?;

    // AvgPool2d: kernel=2x2, stride=2x2
    // [batch, 3, 8, 8] -> [batch, 3, 4, 4]
    let pool = graph.new_avg_pool2d_node(input, (2, 2), Some((2, 2)), Some("pool"))?;

    // 验证动态形状传播
    let pool_node = graph.get_node(pool)?;
    let dyn_shape = pool_node.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "channels 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(3), "channels 应该是 3");

    Ok(())
}

/// 测试 AvgPool2d 在不同 batch_size 下的前向计算
#[test]
fn test_avg_pool2d_dynamic_batch_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建 4D 输入
    let input = graph.new_input_node(&[2, 1, 4, 4], Some("input"))?;
    let pool = graph.new_avg_pool2d_node(input, (2, 2), Some((2, 2)), Some("pool"))?;

    // 设置初始值
    graph.set_node_value(input, Some(&Tensor::ones(&[2, 1, 4, 4])))?;

    // 第一次 forward：batch=2
    graph.forward(pool)?;
    let value1 = graph.get_node_value(pool)?.unwrap();
    assert_eq!(value1.shape(), &[2, 1, 2, 2], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    graph.set_node_value(input, Some(&Tensor::ones(&[5, 1, 4, 4])))?;

    // 第二次 forward：batch=5
    graph.forward(pool)?;
    let value2 = graph.get_node_value(pool)?.unwrap();
    assert_eq!(value2.shape(), &[5, 1, 2, 2], "第二次 forward: batch=5");

    Ok(())
}

/// 测试 AvgPool2d 在不同 batch_size 下的反向传播
#[test]
fn test_avg_pool2d_dynamic_batch_backward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建 4D 输入
    let input = graph.new_input_node(&[2, 1, 4, 4], Some("input"))?;
    let pool = graph.new_avg_pool2d_node(input, (2, 2), Some((2, 2)), Some("pool"))?;
    // 输出形状: [batch, 1, 2, 2]
    let flat = graph.new_flatten_node(pool, true, Some("flat"))?;
    // 输出形状: [batch, 4]
    let target = graph.new_input_node(&[2, 4], Some("target"))?;
    let loss = graph.new_mse_loss_node(flat, target, Some("loss"))?;

    // 设置初始值
    graph.set_node_value(input, Some(&Tensor::normal_seeded(0.0, 1.0, &[2, 1, 4, 4], 42)))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 4])))?;

    // 第一次训练：batch=2
    graph.forward(loss)?;
    let loss_val1 = graph.get_node_value(loss)?.unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 更新输入为不同的 batch_size
    graph.set_node_value(input, Some(&Tensor::normal_seeded(0.0, 1.0, &[4, 1, 4, 4], 100)))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[4, 4])))?;

    // 第二次训练：batch=4
    graph.forward(loss)?;
    let loss_val2 = graph.get_node_value(loss)?.unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
    graph.zero_grad()?;
    graph.backward(loss)?;

    Ok(())
}
