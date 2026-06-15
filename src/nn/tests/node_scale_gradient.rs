/*
 * scale_gradient 测试（梯度缩放：前向恒等，反向 ×scale）
 *
 * 通过 `Var::scale_gradient(scale)` 创建，语义与 PyTorch/JAX 的 scale_gradient、
 * MuZero 论文伪码的 scale_gradient(tensor, scale) 一致：
 *   forward:  y = x
 *   backward: dx = scale · dy
 *
 * detach() 是其 scale=0 特例（完全阻断）；scale=1 是恒等透传。
 *
 * # 与 detach 的关系
 *
 * | 操作 | forward | backward | 用途 |
 * |------|---------|----------|------|
 * | detach()            | y = x | dx = 0       | 完全梯度截断（见 node_detach.rs） |
 * | scale_gradient(s)   | y = x | dx = s·dy    | 部分梯度缩放（MuZero K 步 unroll 半衰） |
 *
 * # 测试覆盖
 * 1. 前向恒等透传（任意 scale）
 * 2. 反向梯度按 scale 缩放（0.5 / 0.25 / 1.0）
 * 3. scale=0 等价 detach（阻断 + is_detached）
 * 4. 链式缩放的乘性衰减（0.5×0.5 == 0.25，对应 K 步 unroll）
 */

use crate::nn::{Graph, Init, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ============================================================================
// 1. 前向恒等透传
// ============================================================================

/// 测试: scale_gradient 前向恒等透传（scale=0.5）
#[test]
fn test_scale_gradient_forward_identity_half() {
    let graph = Graph::new();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let x = graph.input(&input_data).unwrap();
    let y = x.scale_gradient(0.5);

    y.forward().unwrap();

    let out = y.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.data_as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// 测试: scale_gradient 前向恒等透传（任意 scale=0.3，验证分解无前向误差）
#[test]
fn test_scale_gradient_forward_identity_arbitrary() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[2.0, -4.0, 0.0, 7.5], &[2, 2]))
        .unwrap();
    let y = x.scale_gradient(0.3);

    y.forward().unwrap();

    let out = y.value().unwrap().unwrap();
    // x = 0.3·x + 0.7·x，前向应恰为 x（容许 1 ulp 级误差）
    for (a, b) in out.data_as_slice().iter().zip([2.0, -4.0, 0.0, 7.5]) {
        assert_abs_diff_eq!(*a, b, epsilon = 1e-6);
    }
}

// ============================================================================
// 2. 反向梯度按 scale 缩放
// ============================================================================

/// 测试: scale=0.5 时回传梯度减半
///
/// x(参数)=1.0 → y=scale_gradient(x,0.5) → loss=mse(y,0)=y²
/// 无缩放时 dL/dx = 2y = 2.0；缩放 0.5 后应为 1.0。
#[test]
fn test_scale_gradient_halves_gradient() {
    let graph = Graph::new();

    let x = graph.parameter(&[1, 1], Init::Ones, "x").unwrap();
    let y = x.scale_gradient(0.5);
    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    let g = x.grad().unwrap().unwrap();
    assert_abs_diff_eq!(g.data_as_slice()[0], 1.0, epsilon = 1e-6);
}

/// 测试: scale=0.25 时回传梯度为 1/4
#[test]
fn test_scale_gradient_quarter_gradient() {
    let graph = Graph::new();

    let x = graph.parameter(&[1, 1], Init::Ones, "x").unwrap();
    let y = x.scale_gradient(0.25);
    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 全梯度 2.0 × 0.25 = 0.5
    let g = x.grad().unwrap().unwrap();
    assert_abs_diff_eq!(g.data_as_slice()[0], 0.5, epsilon = 1e-6);
}

/// 测试: scale=1.0 是恒等透传（梯度不变）
#[test]
fn test_scale_gradient_one_is_identity() {
    let graph = Graph::new();

    let x = graph.parameter(&[1, 1], Init::Ones, "x").unwrap();
    let y = x.scale_gradient(1.0);
    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 无缩放：dL/dx = 2y = 2.0
    let g = x.grad().unwrap().unwrap();
    assert_abs_diff_eq!(g.data_as_slice()[0], 2.0, epsilon = 1e-6);
}

// ============================================================================
// 3. scale=0 等价 detach
// ============================================================================

/// 测试: scale=0.0 完全阻断梯度（等价 detach）
#[test]
fn test_scale_gradient_zero_blocks_like_detach() {
    let graph = Graph::new();

    let x = graph.parameter(&[1, 1], Init::Ones, "x").unwrap();
    let y = x.scale_gradient(0.0);

    // scale=0 应返回 detach 节点
    assert!(y.is_detached(), "scale_gradient(0.0) 应等价 detach");

    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = y.mse_loss(&target).unwrap();
    loss.backward().unwrap();

    // 梯度被完全阻断
    assert!(
        x.grad().unwrap().is_none(),
        "scale_gradient(0.0) 应阻断梯度回流"
    );
}

// ============================================================================
// 4. 链式缩放的乘性衰减（K 步 unroll 场景）
// ============================================================================

/// 测试: 链式 scale_gradient(0.5)×2 == 单次 scale_gradient(0.25)
///
/// 对应 MuZero K 步 unroll：每步半衰，第 k 步梯度按 0.5^k 衰减。
#[test]
fn test_scale_gradient_chained_decay() {
    let graph = Graph::new();

    let x = graph.parameter(&[1, 1], Init::Ones, "x").unwrap();
    let y = x.scale_gradient(0.5).scale_gradient(0.5); // 0.5 × 0.5 = 0.25
    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 两步半衰 = 0.25 倍：2.0 × 0.25 = 0.5
    let g = x.grad().unwrap().unwrap();
    assert_abs_diff_eq!(g.data_as_slice()[0], 0.5, epsilon = 1e-6);
}
