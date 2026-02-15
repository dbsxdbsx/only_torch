/*
 * @Author       : 老董
 * @Description  : HardTanh 激活节点单元测试
 *
 * 前向：y = min(max(min_val, x), max_val)
 * 反向：dy/dx = 1 if min_val < x < max_val, else 0
 *
 * Python 对照:
 *   hard_tanh([-2, -0.5, 0.5, 2], -1, 1) = [-1, -0.5, 0.5, 1]
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_hard_tanh_forward_default() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[-2.0, -0.5, 0.5, 2.0], &[2, 2]))
        .unwrap();
    let result = x.hard_tanh(-1.0, 1.0);
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], -1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], -0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 1.0, epsilon = 1e-6);
}

#[test]
fn test_hard_tanh_forward_custom_range() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[-5.0, -2.0, 0.0, 3.0, 5.0, 10.0], &[2, 3]))
        .unwrap();
    let result = x.hard_tanh(-3.0, 5.0);
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], -3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], -2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 2]], 5.0, epsilon = 1e-6);
}

/// VJP: min_val < x < max_val 区域梯度通过
#[test]
fn test_hard_tanh_vjp() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner.borrow_mut().create_basic_input_node(&[2, 2], Some("x"))?;
    let ht = inner.borrow_mut().create_hard_tanh_node(x.clone(), -1.0, 1.0, Some("ht"))?;

    x.set_value(Some(&Tensor::new(&[-2.0, -0.5, 0.5, 2.0], &[2, 2])))?;
    ht.forward_recursive(1, false)?;

    let upstream = Tensor::ones(&[2, 2]);
    let grad = ht.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);

    // x=-2 → clipped → 0, x=-0.5 → 1, x=0.5 → 1, x=2 → clipped → 0
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 0.0, epsilon = 1e-6);
    Ok(())
}

#[test]
fn test_hard_tanh_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();
    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[-2.0, -0.5, 0.5, 2.0], &[2, 2]))?;

    let result = x.hard_tanh(-1.0, 1.0);
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;
    assert!(x.grad()?.is_some());
    Ok(())
}
