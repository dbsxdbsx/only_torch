/*
 * @Author       : 老董
 * @Date         : 2026-05-01
 * @Description  : Atan2 节点单元测试
 *
 * 测试覆盖：
 * - 前向传播（四象限典型角度、边界点 (0,0)/(0,±1)/(±1,0)、广播）
 * - 反向传播（解析公式 x/(x²+y²) 与 -y/(x²+y²)）
 * - (y, x) = (0, 0) 时 fallback 为 0（项目意图，区别于 PyTorch NaN）
 * - 接近 (0, 0) 但非零的小邻域 fallback 不过早触发
 * - broadcast 梯度收缩
 * - 节点创建 API + 形状不兼容报错
 */

use crate::nn::graph::Graph;
use crate::nn::{Init, Var, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::f32::consts::PI;
use std::rc::Rc;

// ==================== 前向传播测试 ====================

/// 四象限典型角度 + 坐标轴边界
#[test]
fn test_atan2_forward_quadrants() {
    let graph = Graph::new();

    let y = graph
        .input(&Tensor::new(
            &[1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0],
            &[8, 1],
        ))
        .unwrap();
    let x = graph
        .input(&Tensor::new(
            &[1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 0.0, 0.0],
            &[8, 1],
        ))
        .unwrap();

    let out_node = graph
        .inner_mut()
        .create_atan2_node(Rc::clone(y.node()), Rc::clone(x.node()), Some("atan2"))
        .unwrap();
    let out_var = Var::new_with_rc_graph(out_node, &graph.inner_rc());

    out_var.forward().unwrap();

    let result = out_var.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[8, 1]);
    let expected = [
        PI / 4.0,        // (1, 1) 第一象限
        -PI / 4.0,       // (-1, 1) 第四象限
        -3.0 * PI / 4.0, // (-1, -1) 第三象限
        3.0 * PI / 4.0,  // (1, -1) 第二象限
        0.0,             // (0, 1) 正 x 轴
        PI,              // (0, -1) 负 x 轴
        PI / 2.0,        // (1, 0) 正 y 轴
        -PI / 2.0,       // (-1, 0) 负 y 轴
    ];
    for (i, exp) in expected.iter().enumerate() {
        assert_abs_diff_eq!(result[[i, 0]], *exp, epsilon = 1e-6);
    }
}

/// (0, 0) forward 按 IEEE 754 / Rust 约定返回 0
#[test]
fn test_atan2_forward_zero_origin() {
    let graph = Graph::new();

    let y = graph.input(&Tensor::new(&[0.0], &[1, 1])).unwrap();
    let x = graph.input(&Tensor::new(&[0.0], &[1, 1])).unwrap();

    let out_node = graph
        .inner_mut()
        .create_atan2_node(Rc::clone(y.node()), Rc::clone(x.node()), None)
        .unwrap();
    let out_var = Var::new_with_rc_graph(out_node, &graph.inner_rc());
    out_var.forward().unwrap();

    let result = out_var.value().unwrap().unwrap();
    assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-6);
}

/// y[N,1] vs x[1,N] 广播到 [N, N]
#[test]
fn test_atan2_forward_broadcast() {
    let graph = Graph::new();

    let y = graph
        .input(&Tensor::new(&[1.0, 1.0, 0.0], &[3, 1]))
        .unwrap();
    let x = graph
        .input(&Tensor::new(&[1.0, 0.0, -1.0], &[1, 3]))
        .unwrap();

    let out_node = graph
        .inner_mut()
        .create_atan2_node(Rc::clone(y.node()), Rc::clone(x.node()), None)
        .unwrap();
    let out_var = Var::new_with_rc_graph(out_node, &graph.inner_rc());
    out_var.forward().unwrap();

    let result = out_var.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[3, 3]);

    // y=1 行：atan2(1, 1)=π/4，atan2(1, 0)=π/2，atan2(1, -1)=3π/4
    assert_abs_diff_eq!(result[[0, 0]], PI / 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result[[0, 1]], PI / 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result[[0, 2]], 3.0 * PI / 4.0, epsilon = 1e-6);

    // y=1 行（重复）
    assert_abs_diff_eq!(result[[1, 0]], PI / 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result[[1, 1]], PI / 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result[[1, 2]], 3.0 * PI / 4.0, epsilon = 1e-6);

    // y=0 行：atan2(0, 1)=0、atan2(0, 0)=0、atan2(0, -1)=π
    assert_abs_diff_eq!(result[[2, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result[[2, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result[[2, 2]], PI, epsilon = 1e-6);
}

// ==================== 反向传播测试 ====================

/// 解析梯度公式：∂out/∂y = x/(x²+y²)，∂out/∂x = -y/(x²+y²)
///
/// 用 mse_loss(target=0) 触发：upstream = 2*out/N。
#[test]
fn test_atan2_backward_basic() {
    let graph = Graph::new();

    // 选 (y=2, x=1)：denom = 5，∂/∂y = 1/5 = 0.2，∂/∂x = -2/5 = -0.4
    let y = graph.parameter(&[1, 1], Init::Zeros, "y").unwrap();
    let x = graph.parameter(&[1, 1], Init::Zeros, "x").unwrap();
    y.set_value(&Tensor::new(&[2.0], &[1, 1])).unwrap();
    x.set_value(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let out_node = graph
        .inner_mut()
        .create_atan2_node(Rc::clone(y.node()), Rc::clone(x.node()), None)
        .unwrap();
    let out_var = Var::new_with_rc_graph(out_node, &graph.inner_rc());

    let target = Tensor::zeros(&[1, 1]);
    let loss = out_var.mse_loss(&target).unwrap();
    loss.forward().unwrap();
    loss.backward().unwrap();

    // out = atan2(2, 1) ≈ 1.1071，N = 1，d(mse)/d(out) = 2*out
    let out_val = out_var.value().unwrap().unwrap();
    let upstream = 2.0 * out_val[[0, 0]];

    let y_grad = y.grad().unwrap().unwrap();
    let x_grad = x.grad().unwrap().unwrap();

    assert_abs_diff_eq!(y_grad[[0, 0]], upstream * 0.2, epsilon = 1e-5);
    assert_abs_diff_eq!(x_grad[[0, 0]], upstream * (-0.4), epsilon = 1e-5);
}

/// (y, x) = (0, 0) 处 backward fallback 为 0（项目意图，区别于 PyTorch NaN）
///
/// 这是显式断言：如果未来有人不小心改成 PyTorch 风格 NaN，本测试会立刻挂掉。
#[test]
fn test_atan2_backward_zero_fallback() {
    let graph = Graph::new();

    let y = graph.parameter(&[2, 1], Init::Zeros, "y").unwrap();
    let x = graph.parameter(&[2, 1], Init::Zeros, "x").unwrap();
    y.set_value(&Tensor::new(&[0.0, 1.0], &[2, 1])).unwrap();
    x.set_value(&Tensor::new(&[0.0, 1.0], &[2, 1])).unwrap();

    let out_node = graph
        .inner_mut()
        .create_atan2_node(Rc::clone(y.node()), Rc::clone(x.node()), None)
        .unwrap();
    let out_var = Var::new_with_rc_graph(out_node, &graph.inner_rc());

    let target = Tensor::zeros(&[2, 1]);
    let loss = out_var.mse_loss(&target).unwrap();
    loss.forward().unwrap();
    loss.backward().unwrap();

    let y_grad = y.grad().unwrap().unwrap();
    let x_grad = x.grad().unwrap().unwrap();

    // 第 0 行：(0, 0)，项目 fallback：grad_y = grad_x = 0（PyTorch 在此点是 NaN）
    assert!(
        y_grad[[0, 0]].is_finite(),
        "fallback 期望有限值，得到 NaN/Inf"
    );
    assert!(
        x_grad[[0, 0]].is_finite(),
        "fallback 期望有限值，得到 NaN/Inf"
    );
    assert_abs_diff_eq!(y_grad[[0, 0]], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(x_grad[[0, 0]], 0.0, epsilon = 1e-7);

    // 第 1 行：(1, 1) 非退化点，正常解析梯度
    // out = atan2(1, 1) = π/4，N = 2，upstream = 2 * (π/4) / 2 = π/4
    // denom = 2，∂/∂y = 1/2，∂/∂x = -1/2
    let upstream = 2.0 * (PI / 4.0) / 2.0;
    assert_abs_diff_eq!(y_grad[[1, 0]], upstream * 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(x_grad[[1, 0]], upstream * (-0.5), epsilon = 1e-5);
}

/// 接近 (0, 0) 的小邻域：fallback 不过早触发，仍走解析公式
///
/// 验证 fallback 阈值是 `denom == 0.0` 严格相等，而不是 `denom < eps`
/// 这种近似阈值——后者会误把工作正常的小输入也按 0 处理。
#[test]
fn test_atan2_backward_near_zero_no_premature_fallback() {
    let graph = Graph::new();

    let y = graph.parameter(&[1, 1], Init::Zeros, "y").unwrap();
    let x = graph.parameter(&[1, 1], Init::Zeros, "x").unwrap();
    let yv = 1e-3;
    let xv = 1e-3;
    y.set_value(&Tensor::new(&[yv], &[1, 1])).unwrap();
    x.set_value(&Tensor::new(&[xv], &[1, 1])).unwrap();

    let out_node = graph
        .inner_mut()
        .create_atan2_node(Rc::clone(y.node()), Rc::clone(x.node()), None)
        .unwrap();
    let out_var = Var::new_with_rc_graph(out_node, &graph.inner_rc());

    let target = Tensor::zeros(&[1, 1]);
    let loss = out_var.mse_loss(&target).unwrap();
    loss.forward().unwrap();
    loss.backward().unwrap();

    let denom: f32 = xv.mul_add(xv, yv * yv);
    let out_val = out_var.value().unwrap().unwrap()[[0, 0]];
    let upstream = 2.0 * out_val;
    let expected_dy = upstream * (xv / denom);
    let expected_dx = upstream * (-yv / denom);

    let y_grad = y.grad().unwrap().unwrap();
    let x_grad = x.grad().unwrap().unwrap();
    assert_abs_diff_eq!(y_grad[[0, 0]], expected_dy, epsilon = 1e-3);
    assert_abs_diff_eq!(x_grad[[0, 0]], expected_dx, epsilon = 1e-3);
}

/// 广播场景：y[2,1] vs x[1,2] -> [2,2]，反向梯度沿广播轴 sum_to_shape
#[test]
fn test_atan2_backward_broadcast() {
    let graph = Graph::new();

    let y = graph.parameter(&[2, 1], Init::Zeros, "y").unwrap();
    let x = graph.parameter(&[1, 2], Init::Zeros, "x").unwrap();
    y.set_value(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    x.set_value(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();

    let out_node = graph
        .inner_mut()
        .create_atan2_node(Rc::clone(y.node()), Rc::clone(x.node()), None)
        .unwrap();
    let out_var = Var::new_with_rc_graph(out_node, &graph.inner_rc());

    // mse_loss target=0：upstream[i] = 2 * out[i] / N，N = 4
    let target = Tensor::zeros(&[2, 2]);
    let loss = out_var.mse_loss(&target).unwrap();
    loss.forward().unwrap();
    loss.backward().unwrap();

    // 网格：(y, x)
    // (1, 1) -> denom=2，dy=1/2，dx=-1/2
    // (1, 2) -> denom=5，dy=2/5，dx=-1/5
    // (2, 1) -> denom=5，dy=1/5，dx=-2/5
    // (2, 2) -> denom=8，dy=2/8=1/4，dx=-2/8=-1/4
    let out_val = out_var.value().unwrap().unwrap();
    let n = 4.0_f32;

    // y_grad[0, 0] 沿 axis=1 求和：(1,1) 与 (1,2) 两列的 upstream*∂out/∂y
    let up_00 = 2.0 * out_val[[0, 0]] / n;
    let up_01 = 2.0 * out_val[[0, 1]] / n;
    let up_10 = 2.0 * out_val[[1, 0]] / n;
    let up_11 = 2.0 * out_val[[1, 1]] / n;

    let y_grad_0 = up_00.mul_add(0.5, up_01 * (2.0 / 5.0));
    let y_grad_1 = up_10.mul_add(1.0 / 5.0, up_11 * (1.0 / 4.0));

    let x_grad_0 = up_00.mul_add(-0.5, up_10 * (-2.0 / 5.0));
    let x_grad_1 = up_01.mul_add(-1.0 / 5.0, up_11 * (-1.0 / 4.0));

    let y_grad = y.grad().unwrap().unwrap();
    let x_grad = x.grad().unwrap().unwrap();

    assert_abs_diff_eq!(y_grad[[0, 0]], y_grad_0, epsilon = 1e-5);
    assert_abs_diff_eq!(y_grad[[1, 0]], y_grad_1, epsilon = 1e-5);
    assert_abs_diff_eq!(x_grad[[0, 0]], x_grad_0, epsilon = 1e-5);
    assert_abs_diff_eq!(x_grad[[0, 1]], x_grad_1, epsilon = 1e-5);
}

// ==================== Var::atan2 链式调用 ====================

/// 验证 `Var::atan2` 与 builder 路径行为一致（self 是 y，参数是 x）
#[test]
fn test_var_atan2_method_alignment() {
    let graph = Graph::new();

    let y_var = graph.input(&Tensor::new(&[1.0, 0.5], &[2, 1])).unwrap();
    let x_var = graph.input(&Tensor::new(&[1.0, -1.0], &[2, 1])).unwrap();

    let out_var = y_var.atan2(&x_var).unwrap();
    out_var.forward().unwrap();

    let result = out_var.value().unwrap().unwrap();
    assert_abs_diff_eq!(result[[0, 0]], (1.0_f32).atan2(1.0), epsilon = 1e-6);
    assert_abs_diff_eq!(result[[1, 0]], (0.5_f32).atan2(-1.0), epsilon = 1e-6);
}

// ==================== 节点创建 API 测试 ====================

#[test]
fn test_create_atan2_node_same_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let y = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("y"))
        .unwrap();
    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("x"))
        .unwrap();

    let node = inner
        .borrow_mut()
        .create_atan2_node(y.clone(), x.clone(), Some("ang"))
        .unwrap();

    assert_eq!(node.shape(), vec![3, 4]);
    assert_eq!(node.name(), Some("ang"));
    assert_eq!(node.parents().len(), 2);
}

#[test]
fn test_create_atan2_node_broadcast() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let y = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 1], None)
        .unwrap();
    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], None)
        .unwrap();

    let node = inner.borrow_mut().create_atan2_node(y, x, None).unwrap();

    assert_eq!(node.shape(), vec![3, 4]);
}

#[test]
fn test_create_atan2_node_shape_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let y = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();
    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 6], None)
        .unwrap();

    let result = inner.borrow_mut().create_atan2_node(y, x, None);
    assert!(result.is_err());
}
