/*
 * @Author       : 老董
 * @Date         : 2026-04-24
 * @Description  : Upsample2d 节点单元测试
 *
 * 测试策略（五段式，与 max_pool2d / avg_pool2d 测试风格对齐）：
 * 1. 前向传播 → 简单 / batch / 多通道 / 不对称 scale / scale=3 / 错误处理
 * 2. VJP（calc_grad_to_parent_index）→ ones upstream / 非单位 upstream / 不对称 scale / batch
 * 3. E2E 反向传播 → loss 梯度验证；MaxPool + Upsample 串联（先降再升）
 * 4. 动态形状 + 动态 batch（前向 + 反向）
 * 5. Create API
 *
 * 反向传播数学（核心易错点）：
 *   dL/dx[i, j] = sum over (s × s) block of dL/dy
 * 即按 (scale_h × scale_w) 块"求和"，注意是 sum 不是 mean。
 * 单测预期值由 Python 参考脚本 tests/python_reference/upsample2d_reference.py 验证一致。
 */

use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试 ====================

/// 测试 Upsample2d 前向传播（简单 scale=2x2）
///
/// 输入 [[1, 2], [3, 4]] (1x1x2x2), scale=(2, 2)
/// 输出 (1x1x4x4):
///   [[1, 1, 2, 2],
///    [1, 1, 2, 2],
///    [3, 3, 4, 4],
///    [3, 3, 4, 4]]
#[test]
fn test_upsample2d_forward_simple() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(input.clone(), 2, 2, Some("up"))?;

    let input_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    input.set_value(Some(&input_val))?;
    up.forward_recursive(1, false)?;

    let output = up.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 4, 4]);

    // 第 0 行
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 0, 2]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 0, 3]], 2.0, epsilon = 1e-6);
    // 第 1 行（与第 0 行一致，nearest 复制）
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 3]], 2.0, epsilon = 1e-6);
    // 第 2/3 行（来自输入的第 1 行）
    assert_abs_diff_eq!(output[[0, 0, 2, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 3, 3]], 4.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Upsample2d 前向传播（batch=2）
///
/// batch 0 全 5.0 → 全部输出 5.0
/// batch 1 = [[1, 2], [3, 4]] → 与 simple 一致
#[test]
fn test_upsample2d_forward_batch() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 2, 2], Some("input"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(input.clone(), 2, 2, Some("up"))?;

    let mut data = vec![5.0f32; 4]; // batch 0
    data.extend_from_slice(&[1.0, 2.0, 3.0, 4.0]); // batch 1
    input.set_value(Some(&Tensor::new(&data, &[2, 1, 2, 2])))?;
    up.forward_recursive(1, false)?;

    let output = up.value().unwrap();
    assert_eq!(output.shape(), &[2, 1, 4, 4]);

    // batch 0 全 5.0
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 3, 3]], 5.0, epsilon = 1e-6);
    // batch 1 与 simple 一致
    assert_abs_diff_eq!(output[[1, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0, 0, 3]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0, 3, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0, 3, 3]], 4.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Upsample2d 多通道前向传播
///
/// channel 0 = [[1, 2], [3, 4]]，channel 1 全 7.0
#[test]
fn test_upsample2d_forward_multi_channel() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2, 2, 2], Some("input"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(input.clone(), 2, 2, Some("up"))?;

    let mut data = vec![1.0, 2.0, 3.0, 4.0]; // channel 0
    data.extend(vec![7.0f32; 4]); // channel 1
    input.set_value(Some(&Tensor::new(&data, &[1, 2, 2, 2])))?;
    up.forward_recursive(1, false)?;

    let output = up.value().unwrap();
    assert_eq!(output.shape(), &[1, 2, 4, 4]);

    // channel 0
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 3, 3]], 4.0, epsilon = 1e-6);
    // channel 1 全 7.0
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 7.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 2, 1]], 7.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 3, 3]], 7.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Upsample2d 不对称 scale=(2, 3)
///
/// 输入 [[1, 2], [3, 4]] (1x1x2x2), scale=(2, 3)
/// 输出 (1x1x4x6):
///   [[1, 1, 1, 2, 2, 2],
///    [1, 1, 1, 2, 2, 2],
///    [3, 3, 3, 4, 4, 4],
///    [3, 3, 3, 4, 4, 4]]
#[test]
fn test_upsample2d_forward_asymmetric_scale() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(input.clone(), 2, 3, Some("up"))?;

    let input_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    input.set_value(Some(&input_val))?;
    up.forward_recursive(1, false)?;

    let output = up.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 4, 6]);

    // 第 0 行（来自 input[0]）：[1,1,1, 2,2,2]
    for j in 0..3 {
        assert_abs_diff_eq!(output[[0, 0, 0, j]], 1.0, epsilon = 1e-6);
    }
    for j in 3..6 {
        assert_abs_diff_eq!(output[[0, 0, 0, j]], 2.0, epsilon = 1e-6);
    }
    // 第 2 行（来自 input[1]）：[3,3,3, 4,4,4]
    for j in 0..3 {
        assert_abs_diff_eq!(output[[0, 0, 2, j]], 3.0, epsilon = 1e-6);
    }
    for j in 3..6 {
        assert_abs_diff_eq!(output[[0, 0, 2, j]], 4.0, epsilon = 1e-6);
    }

    Ok(())
}

/// 测试 Upsample2d scale=(3, 3)
///
/// 输入 [[1, 2], [3, 4]] (1x1x2x2), scale=(3, 3)
/// 输出 (1x1x6x6)，每个输入像素被复制到 3x3 块
#[test]
fn test_upsample2d_forward_scale3() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(input.clone(), 3, 3, Some("up"))?;

    input.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])))?;
    up.forward_recursive(1, false)?;

    let output = up.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 6, 6]);

    // 检查 4 个 3x3 块各自的值
    for i in 0..3 {
        for j in 0..3 {
            assert_abs_diff_eq!(output[[0, 0, i, j]], 1.0, epsilon = 1e-6);
            assert_abs_diff_eq!(output[[0, 0, i, j + 3]], 2.0, epsilon = 1e-6);
            assert_abs_diff_eq!(output[[0, 0, i + 3, j]], 3.0, epsilon = 1e-6);
            assert_abs_diff_eq!(output[[0, 0, i + 3, j + 3]], 4.0, epsilon = 1e-6);
        }
    }

    Ok(())
}

/// 测试无效输入维度 → 应报错
#[test]
fn test_upsample2d_invalid_input_dims() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 2D 输入（缺少 batch 和通道维度）
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 4], Some("input"))
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_upsample2d_node(input, 2, 2, Some("up"));
    assert!(result.is_err());
}

/// 测试 scale=0 → 应报错
#[test]
fn test_upsample2d_invalid_scale_zero() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_upsample2d_node(input, 0, 2, Some("up"));
    assert!(result.is_err());
}

// ==================== 2. VJP 单元测试（calc_grad_to_parent_index）====================
//
// 反向数学：dL/dx[i,j] = sum over (s_h × s_w) block of dL/dy
// （sum 不是 mean，跟 avg_pool 反向除以 N 不同）

/// 测试 Upsample2d VJP（scale=2x2，upstream=ones）
///
/// 每个输入位置对应输出 2x2 块；upstream 全 1 → grad 全 4
#[test]
fn test_upsample2d_vjp_basic() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(input.clone(), 2, 2, Some("up"))?;

    input.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])))?;
    up.forward_recursive(1, false)?;

    let upstream = Tensor::ones(&[1, 1, 4, 4]);
    let grad = up
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    assert_eq!(grad.shape(), &[1, 1, 2, 2]);

    // 每个位置 = sum of 2x2 ones = 4
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 4.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Upsample2d VJP（非单位 upstream）
///
/// upstream = 1..16 (4x4)
/// grad[i,j] = sum of 2x2 block in upstream
/// upstream 矩阵：
///   [ 1  2  3  4]
///   [ 5  6  7  8]
///   [ 9 10 11 12]
///   [13 14 15 16]
/// grad[0,0] = 1+2+5+6 = 14
/// grad[0,1] = 3+4+7+8 = 22
/// grad[1,0] = 9+10+13+14 = 46
/// grad[1,1] = 11+12+15+16 = 54
#[test]
fn test_upsample2d_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(input.clone(), 2, 2, Some("up"))?;

    input.set_value(Some(&Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 2, 2])))?;
    up.forward_recursive(1, false)?;

    let upstream_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let upstream = Tensor::new(&upstream_data, &[1, 1, 4, 4]);
    let grad = up
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 14.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 22.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 0]], 46.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 54.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Upsample2d VJP（不对称 scale=2x3）
///
/// 输入 [1,1,2,2],upstream=ones [1,1,4,6]
/// 每个输入位置对应 2x3=6 个输出 → grad 全 6
#[test]
fn test_upsample2d_vjp_asymmetric_scale() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(input.clone(), 2, 3, Some("up"))?;

    input.set_value(Some(&Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 2, 2])))?;
    up.forward_recursive(1, false)?;

    let upstream = Tensor::ones(&[1, 1, 4, 6]);
    let grad = up
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    assert_eq!(grad.shape(), &[1, 1, 2, 2]);

    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(grad[[0, 0, i, j]], 6.0, epsilon = 1e-6);
        }
    }

    Ok(())
}

/// 测试 Upsample2d VJP（batch=2）
///
/// 验证多 batch 下梯度独立计算
#[test]
fn test_upsample2d_vjp_batch() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 2, 2], Some("input"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(input.clone(), 2, 2, Some("up"))?;

    input.set_value(Some(&Tensor::zeros(&[2, 1, 2, 2])))?;
    up.forward_recursive(1, false)?;

    // batch 0: upstream 全 1 → grad 全 4
    // batch 1: upstream = 2 → grad 全 8
    let mut up_data = vec![1.0f32; 16];
    up_data.extend(vec![2.0f32; 16]);
    let upstream = Tensor::new(&up_data, &[2, 1, 4, 4]);
    let grad = up
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_eq!(grad.shape(), &[2, 1, 2, 2]);
    // batch 0
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 4.0, epsilon = 1e-6);
    // batch 1
    assert_abs_diff_eq!(grad[[1, 0, 0, 0]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0, 1, 1]], 8.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 3. E2E 反向传播测试 ====================

/// 测试 Upsample2d E2E 反向传播
///
/// 结构：param[1,1,2,2] → upsample(2x2) → flatten → mse_mean(target=0)
///
/// 设 x = [[1, 2], [3, 4]]
/// y = upsample(x) = [1,1,2,2, 1,1,2,2, 3,3,4,4, 3,3,4,4] (16 个元素)
/// loss = mean(y^2) = (4*1 + 4*4 + 4*9 + 4*16) / 16 = (4+16+36+64)/16 = 7.5
///
/// dL/dy = 2*y/N = y/8
/// dL/dx[i,j] = sum over 2x2 block of dL/dy
///   = sum(y[2i:2i+2, 2j:2j+2]) / 8
///   = (4 * x[i,j]) / 8 = x[i,j] / 2
/// 即 dL/dx = [[0.5, 1.0], [1.5, 2.0]]
#[test]
fn test_upsample2d_e2e_backward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let param = inner
        .borrow_mut()
        .create_parameter_node(&[1, 1, 2, 2], Some("param"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(param.clone(), 2, 2, Some("up"))?;
    let flat = inner
        .borrow_mut()
        .create_flatten_node(up.clone(), true, Some("flat"))?;
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 16], Some("target"))?;
    let loss =
        inner
            .borrow_mut()
            .create_mse_mean_node(flat.clone(), target.clone(), Some("loss"))?;

    param.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])))?;
    target.set_value(Some(&Tensor::zeros(&[1, 16])))?;

    inner.borrow_mut().set_train_mode();
    inner.borrow_mut().forward_via_node_inner(&loss)?;

    // upsample 输出元素总和验证
    let up_val = up.value().unwrap();
    assert_eq!(up_val.shape(), &[1, 1, 4, 4]);
    assert_abs_diff_eq!(up_val[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(up_val[[0, 0, 3, 3]], 4.0, epsilon = 1e-6);

    // loss = mean(y^2) = 7.5
    let loss_val = loss.value().unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], 7.5, epsilon = 1e-4);

    inner.borrow_mut().backward_via_node_inner(&loss)?;

    // 验证 param 梯度 = [[0.5, 1.0], [1.5, 2.0]]
    let grad = param.grad().expect("param 应有 grad");
    assert_eq!(grad.shape(), &[1, 1, 2, 2]);
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 0.5, epsilon = 1e-4);
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 1.0, epsilon = 1e-4);
    assert_abs_diff_eq!(grad[[0, 0, 1, 0]], 1.5, epsilon = 1e-4);
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 2.0, epsilon = 1e-4);

    Ok(())
}

/// 测试 MaxPool2d + Upsample2d 串联（先降再升 = 分辨率回升模拟 U-Net）
///
/// 结构：param[1,1,4,4] → maxpool(2x2) → upsample(2x2) → flatten → mse_loss
///
/// 验证梯度能穿透 upsample → maxpool 正确传播到 param
#[test]
fn test_upsample2d_e2e_pool_upsample_cascade() -> Result<(), GraphError> {
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
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(pool.clone(), 2, 2, Some("up"))?;
    let flat = inner
        .borrow_mut()
        .create_flatten_node(up.clone(), true, Some("flat"))?;
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 16], Some("target"))?;
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
    target.set_value(Some(&Tensor::zeros(&[1, 16])))?;

    inner.borrow_mut().set_train_mode();
    inner.borrow_mut().forward_via_node_inner(&loss)?;

    // pool 输出 [1,1,2,2] = [6, 8, 14, 16]，upsample 后 [1,1,4,4]
    let up_val = up.value().unwrap();
    assert_eq!(up_val.shape(), &[1, 1, 4, 4]);
    assert_abs_diff_eq!(up_val[[0, 0, 0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(up_val[[0, 0, 0, 3]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(up_val[[0, 0, 3, 0]], 14.0, epsilon = 1e-6);
    assert_abs_diff_eq!(up_val[[0, 0, 3, 3]], 16.0, epsilon = 1e-6);

    inner.borrow_mut().backward_via_node_inner(&loss)?;

    // 验证 param 梯度形状正确，且 pool 选中的最大值位置（(1,1), (1,3), (3,1), (3,3)）非零
    let grad = param.grad().expect("param 应有 grad");
    assert_eq!(grad.shape(), &[1, 1, 4, 4]);
    assert!(grad[[0, 0, 1, 1]].abs() > 1e-6, "max=6 位置应有梯度");
    assert!(grad[[0, 0, 1, 3]].abs() > 1e-6, "max=8 位置应有梯度");
    assert!(grad[[0, 0, 3, 1]].abs() > 1e-6, "max=14 位置应有梯度");
    assert!(grad[[0, 0, 3, 3]].abs() > 1e-6, "max=16 位置应有梯度");
    // 非最大值位置仍为 0（被 pool 屏蔽）
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 2, 2]], 0.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 4. 动态形状 + 动态 batch 测试 ====================

/// 测试 Upsample2d 动态 batch 前向传播
///
/// 先 batch=2，再 batch=5，验证输出形状自适应
#[test]
fn test_upsample2d_dynamic_batch_forward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 2, 2], Some("input"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(input.clone(), 2, 2, Some("up"))?;

    input.set_value(Some(&Tensor::ones(&[2, 1, 2, 2])))?;
    inner.borrow_mut().forward_via_node_inner(&up)?;
    let val1 = up.value().unwrap();
    assert_eq!(val1.shape(), &[2, 1, 4, 4], "第一次 forward: batch=2");

    input.set_value(Some(&Tensor::ones(&[5, 1, 2, 2])))?;
    inner.borrow_mut().forward_via_node_inner(&up)?;
    let val2 = up.value().unwrap();
    assert_eq!(val2.shape(), &[5, 1, 4, 4], "第二次 forward: batch=5");

    Ok(())
}

/// 测试 Upsample2d 动态 batch 反向传播
#[test]
fn test_upsample2d_dynamic_batch_backward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let param = inner
        .borrow_mut()
        .create_parameter_node(&[2, 1, 2, 2], Some("param"))?;
    let up = inner
        .borrow_mut()
        .create_upsample2d_node(param.clone(), 2, 2, Some("up"))?;
    let flat = inner
        .borrow_mut()
        .create_flatten_node(up.clone(), true, Some("flat"))?;
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 16], Some("target"))?;
    let loss =
        inner
            .borrow_mut()
            .create_mse_mean_node(flat.clone(), target.clone(), Some("loss"))?;

    inner.borrow_mut().set_train_mode();

    // 第一次：batch=2
    param.set_value(Some(&Tensor::normal_seeded(0.0, 1.0, &[2, 1, 2, 2], 42)))?;
    target.set_value(Some(&Tensor::zeros(&[2, 16])))?;
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    param.clear_grad()?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;
    let grad1 = param.grad().expect("batch=2 时 param 应有 grad");
    assert_eq!(grad1.shape(), &[2, 1, 2, 2]);

    // 第二次：batch=4
    param.set_value(Some(&Tensor::normal_seeded(0.0, 1.0, &[4, 1, 2, 2], 100)))?;
    target.set_value(Some(&Tensor::zeros(&[4, 16])))?;
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    param.clear_grad()?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;
    let grad2 = param.grad().expect("batch=4 时 param 应有 grad");
    assert_eq!(grad2.shape(), &[4, 1, 2, 2]);

    Ok(())
}

// ==================== 5. Create API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_upsample2d_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 8, 8], Some("input"))
        .unwrap();

    let up = inner
        .borrow_mut()
        .create_upsample2d_node(input.clone(), 2, 2, Some("up"))
        .unwrap();

    assert_eq!(up.shape(), vec![2, 3, 16, 16]);
    assert_eq!(up.name(), Some("up"));
    assert!(!up.is_leaf());
    assert_eq!(up.parents().len(), 1);
}

#[test]
fn test_create_upsample2d_invalid_scale() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], None)
        .unwrap();

    // scale_w=0 → 应失败
    let result = inner.borrow_mut().create_upsample2d_node(input, 2, 0, None);
    assert!(result.is_err());
}

#[test]
fn test_create_upsample2d_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_up;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[1, 1, 8, 8], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let up = inner
            .borrow_mut()
            .create_upsample2d_node(input, 2, 2, None)
            .unwrap();
        weak_up = Rc::downgrade(&up);

        assert!(weak_up.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_up.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
