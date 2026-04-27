/*
 * @Author       : 老董
 * @Description  : Log10（以 10 为底的对数）节点单元测试
 *
 * 梯度公式：y = log10(x), dy/dx = 1/(x * ln(10))
 *
 * Python 对照:
 *   np.log10([1, 10, 100, 1000]) = [0, 1, 2, 3]
 *   grad = 1/(x * ln(10)) → [0.4343, 0.04343, 0.004343, 0.000434]
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_log10_forward() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 10.0, 100.0, 1000.0], &[2, 2]))
        .unwrap();
    let result = x.log10();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[1, 1]], 3.0, epsilon = 1e-5);
}

/// VJP: grad = upstream / (x * ln(10))
#[test]
fn test_log10_vjp() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))?;
    let log = inner
        .borrow_mut()
        .create_log10_node(x.clone(), Some("log10"))?;

    x.set_value(Some(&Tensor::new(&[1.0, 10.0, 100.0, 1000.0], &[2, 2])))?;
    log.forward_recursive(1, false)?;

    let upstream = Tensor::ones(&[2, 2]);
    let grad = log
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    let ln10 = 10.0_f32.ln();
    assert_abs_diff_eq!(grad[[0, 0]], 1.0 / (1.0 * ln10), epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 1.0 / (10.0 * ln10), epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0]], 1.0 / (100.0 * ln10), epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1]], 1.0 / (1000.0 * ln10), epsilon = 1e-5);
    Ok(())
}

#[test]
fn test_log10_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();
    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 10.0, 100.0, 1000.0], &[2, 2]))?;

    let result = x.log10();
    let target = graph.input(&Tensor::new(&[0.0, 1.0, 2.0, 3.0], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;
    assert!(x.grad()?.is_some());
    Ok(())
}
