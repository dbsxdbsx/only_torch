/*
 * @Author       : 老董
 * @Description  : Square（平方）节点单元测试
 *
 * 梯度公式：y = x², dy/dx = 2x
 *
 * Python 对照 (numpy):
 *   [1,2,3,4]² = [1, 4, 9, 16]
 *   grad(x²) = 2*[1,2,3,4] = [2, 4, 6, 8]
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_square_forward() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let result = x.square();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 9.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 16.0, epsilon = 1e-6);
}

#[test]
fn test_square_forward_negative() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[-3.0, -2.0, 0.0, 5.0], &[2, 2]))
        .unwrap();
    let result = x.square();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 9.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 25.0, epsilon = 1e-6);
}

/// VJP: grad = upstream * 2x
#[test]
fn test_square_vjp() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))?;
    let sq = inner
        .borrow_mut()
        .create_square_node(x.clone(), Some("sq"))?;

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    sq.forward_recursive(1, false)?;

    let upstream = Tensor::ones(&[2, 2]);
    let grad = sq
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_abs_diff_eq!(grad[[0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 8.0, epsilon = 1e-6);
    Ok(())
}

#[test]
fn test_square_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();
    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;

    let result = x.square();
    let target = graph.input(&Tensor::new(&[1.0, 3.0, 8.0, 15.0], &[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;
    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 2]);
    Ok(())
}

#[test]
fn test_square_cannot_set_value() {
    let graph = Graph::new();
    let x = graph.input(&Tensor::ones(&[2, 2])).unwrap();
    let result = x.square();
    assert!(result.set_value(&Tensor::ones(&[2, 2])).is_err());
}

#[test]
fn test_create_square_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();
    let sq = inner
        .borrow_mut()
        .create_square_node(input, Some("sq"))
        .unwrap();
    assert_eq!(sq.shape(), vec![3, 4]);
}
