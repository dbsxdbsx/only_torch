/*
 * @Author       : 老董
 * @Description  : Reciprocal（倒数）节点单元测试
 *
 * 梯度公式：y = 1/x, dy/dx = -1/x²
 *
 * Python 对照:
 *   1/[1,2,4,5] = [1, 0.5, 0.25, 0.2]
 *   grad(1/x) = -1/x² → -1/[1,4,16,25] = [-1, -0.25, -0.0625, -0.04]
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_reciprocal_forward() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 4.0, 5.0], &[2, 2]))
        .unwrap();
    let result = x.reciprocal();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 0.2, epsilon = 1e-6);
}

/// VJP: grad = upstream * (-1/x²)
#[test]
fn test_reciprocal_vjp() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))?;
    let rec = inner
        .borrow_mut()
        .create_reciprocal_node(x.clone(), Some("rec"))?;

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 4.0, 5.0], &[2, 2])))?;
    rec.forward_recursive(1, false)?;

    let upstream = Tensor::ones(&[2, 2]);
    let grad = rec
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    // -1/x² → [-1, -0.25, -0.0625, -0.04]
    assert_abs_diff_eq!(grad[[0, 0]], -1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], -0.25, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0]], -0.0625, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1]], -0.04, epsilon = 1e-5);
    Ok(())
}

#[test]
fn test_reciprocal_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();
    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 4.0, 5.0], &[2, 2]))?;

    let result = x.reciprocal();
    let target = graph.input(&Tensor::new(&[1.0, 0.5, 0.25, 0.2], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;
    assert!(x.grad()?.is_some());
    Ok(())
}
