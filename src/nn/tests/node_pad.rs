/*
 * @Author       : 老董
 * @Description  : Pad（填充）节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ 2D/4D/非对称/非零填充值
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ 裁切回原形状
 * 3. 端到端反向传播测试（高层 API）
 * 4. Create API 测试
 *
 * 梯度公式：
 *   填充区域梯度为 0，原始区域梯度直接传递。
 *   反向传播 = 从 upstream_grad 中 slice 出原始区域。
 *
 * Python 对照值 (numpy):
 *   [[1,2,3],[4,5,6]] pad [(1,1),(2,2)] →
 *     [[0,0,0,0,0,0,0],[0,0,1,2,3,0,0],[0,0,4,5,6,0,0],[0,0,0,0,0,0,0]]
 *   shape: [4, 7]
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 前向传播测试（高层 API）====================

/// 测试 Pad 前向传播（2D 对称填充）
///
/// [[1,2,3],[4,5,6]] pad [(1,1),(2,2)] →
/// shape [4, 7]，中间区域为原值，周围为 0
#[test]
fn test_pad_forward_2d() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]))
        .unwrap();
    let result = x.pad(&[(1, 1), (2, 2)], 0.0).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[4, 7]);
    // 第 0 行全 0
    for j in 0..7 {
        assert_abs_diff_eq!(output[[0, j]], 0.0, epsilon = 1e-7);
    }
    // 第 1 行：[0, 0, 1, 2, 3, 0, 0]
    assert_abs_diff_eq!(output[[1, 0]], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[1, 1]], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[1, 2]], 1.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[1, 3]], 2.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[1, 4]], 3.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[1, 5]], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[1, 6]], 0.0, epsilon = 1e-7);
    // 第 2 行：[0, 0, 4, 5, 6, 0, 0]
    assert_abs_diff_eq!(output[[2, 2]], 4.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[2, 3]], 5.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[2, 4]], 6.0, epsilon = 1e-7);
}

/// 测试 Pad 前向传播（非对称填充）
///
/// [[1,2],[3,4]] pad [(0,2),(1,0)] → shape [4, 3]
#[test]
fn test_pad_forward_asymmetric() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1., 2., 3., 4.], &[2, 2]))
        .unwrap();
    let result = x.pad(&[(0, 2), (1, 0)], 0.0).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[4, 3]);
    // [[0,1,2],[0,3,4],[0,0,0],[0,0,0]]
    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[0, 2]], 2.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[1, 0]], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[1, 1]], 3.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[1, 2]], 4.0, epsilon = 1e-7);
    // 后两行全 0
    for i in 2..4 {
        for j in 0..3 {
            assert_abs_diff_eq!(output[[i, j]], 0.0, epsilon = 1e-7);
        }
    }
}

/// 测试 Pad 前向传播（非零填充值）
///
/// [[1,2],[3,4]] pad [(1,1),(1,1)] value=-1
#[test]
fn test_pad_forward_nonzero_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1., 2., 3., 4.], &[2, 2]))
        .unwrap();
    let result = x.pad(&[(1, 1), (1, 1)], -1.0).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[4, 4]);
    // 边缘应为 -1
    assert_abs_diff_eq!(output[[0, 0]], -1.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[0, 1]], -1.0, epsilon = 1e-7);
    // 中心应为原值
    assert_abs_diff_eq!(output[[1, 1]], 1.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[1, 2]], 2.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[2, 1]], 3.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[2, 2]], 4.0, epsilon = 1e-7);
}

/// 测试 Pad 前向传播（4D CNN 风格填充）
///
/// [1,1,2,2] pad [(0,0),(0,0),(1,1),(1,1)] → [1,1,4,4]
#[test]
fn test_pad_forward_4d_cnn() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1., 2., 3., 4.], &[1, 1, 2, 2]))
        .unwrap();
    let result = x.pad(&[(0, 0), (0, 0), (1, 1), (1, 1)], 0.0).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 1, 4, 4]);
    // 中心 2x2 应有原值
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 1.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[0, 0, 1, 2]], 2.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[0, 0, 2, 1]], 3.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[0, 0, 2, 2]], 4.0, epsilon = 1e-7);
    // 边缘应为 0
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(output[[0, 0, 3, 3]], 0.0, epsilon = 1e-7);
}

/// 测试 Pad 节点不能直接设置值
#[test]
fn test_pad_cannot_set_value() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1., 2., 3., 4.], &[2, 2]))
        .unwrap();
    let result = x.pad(&[(1, 1), (1, 1)], 0.0).unwrap();

    let test_value = Tensor::ones(&[4, 4]);
    let err = result.set_value(&test_value);
    assert!(err.is_err(), "Pad 节点不应支持直接设值");
}

// ==================== VJP 单元测试（底层 calc_grad_to_parent_index）====================

/// 测试 Pad VJP：梯度裁切回原形状
///
/// input [2,3] → pad [(1,1),(2,2)] → [4,7]
/// upstream [4,7] 全 1 → grad 应为 [2,3] 全 1（只取中间区域）
#[test]
fn test_pad_vjp_unit_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let pad = inner
        .borrow_mut()
        .create_pad_node(x.clone(), vec![(1, 1), (2, 2)], 0.0, Some("pad"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])))
        .unwrap();
    pad.forward_recursive(1, false).unwrap();

    let upstream_grad = Tensor::ones(&[4, 7]);
    let grad = pad
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 3]);
    // 所有元素应为 1（从全 1 的 upstream 中提取）
    for &v in grad.data_as_slice() {
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-7);
    }

    Ok(())
}

/// 测试 Pad VJP：非均匀上游梯度
///
/// 验证裁切位置正确
#[test]
fn test_pad_vjp_non_uniform_upstream() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let pad = inner
        .borrow_mut()
        .create_pad_node(x.clone(), vec![(1, 1), (1, 1)], 0.0, Some("pad"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1., 2., 3., 4.], &[2, 2])))
        .unwrap();
    pad.forward_recursive(1, false).unwrap();

    // upstream [4, 4] 用不同值填充，验证裁切位置
    #[rustfmt::skip]
    let upstream_grad = Tensor::new(&[
        10., 20., 30., 40.,
        50., 60., 70., 80.,
        90., 100., 110., 120.,
        130., 140., 150., 160.,
    ], &[4, 4]);

    let grad = pad
        .calc_grad_to_parent_index(0, &upstream_grad)?
        .resolve(&upstream_grad);

    assert_eq!(grad.shape(), &[2, 2]);
    // 应提取 upstream[1:3, 1:3] = [[60, 70], [100, 110]]
    assert_abs_diff_eq!(grad[[0, 0]], 60.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 70.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0]], 100.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1]], 110.0, epsilon = 1e-5);

    Ok(())
}

// ==================== 端到端反向传播测试（高层 API）====================

/// 测试 Pad 端到端反向传播
#[test]
fn test_pad_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1., 2., 3., 4.], &[2, 2]))?;

    let padded = x.pad(&[(1, 1), (1, 1)], 0.0)?;
    let target = graph.input(&Tensor::zeros(&[4, 4]))?;
    let loss = padded.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);

    Ok(())
}

// ==================== 节点创建 API 测试 ====================

#[test]
fn test_create_pad_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();

    let pad = inner
        .borrow_mut()
        .create_pad_node(input.clone(), vec![(1, 1), (2, 2)], 0.0, Some("pad"))
        .unwrap();

    assert_eq!(pad.shape(), vec![4, 7]);
    assert_eq!(pad.name(), Some("pad"));
}

#[test]
fn test_create_pad_node_4d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3, 8, 8], None)
        .unwrap();

    let pad = inner
        .borrow_mut()
        .create_pad_node(
            input.clone(),
            vec![(0, 0), (0, 0), (1, 1), (1, 1)],
            0.0,
            None,
        )
        .unwrap();

    assert_eq!(pad.shape(), vec![1, 3, 10, 10]);
}

#[test]
fn test_create_pad_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_pad;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let pad = inner
            .borrow_mut()
            .create_pad_node(input, vec![(1, 1), (1, 1)], 0.0, None)
            .unwrap();
        weak_pad = Rc::downgrade(&pad);

        assert!(weak_pad.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_pad.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}

/// 测试 paddings 长度不匹配应报错
#[test]
fn test_pad_invalid_paddings_length() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1., 2., 3., 4.], &[2, 2]))
        .unwrap();
    // paddings 只给了 1 维，但输入是 2D
    let result = x.pad(&[(1, 1)], 0.0);
    assert!(result.is_err());
}
