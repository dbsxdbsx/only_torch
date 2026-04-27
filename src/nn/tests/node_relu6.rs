/*
 * @Author       : 老董
 * @Description  : ReLU6 激活节点单元测试
 *
 * 前向：y = min(max(0, x), 6)
 * 反向：dy/dx = 1 if 0 < x < 6, else 0
 *
 * Python 对照:
 *   relu6([-2, -1, 0, 3, 6, 7]) = [0, 0, 0, 3, 6, 6]
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_relu6_forward() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[-2.0, -1.0, 0.0, 3.0, 6.0, 7.0], &[2, 3]))
        .unwrap();
    let result = x.relu6();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 3]);
    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 2]], 6.0, epsilon = 1e-6);
}

/// VJP: 0 < x < 6 区域梯度通过，其余为 0
#[test]
fn test_relu6_vjp() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))?;
    let r6 = inner
        .borrow_mut()
        .create_relu6_node(x.clone(), Some("relu6"))?;

    x.set_value(Some(&Tensor::new(
        &[-2.0, -1.0, 0.0, 3.0, 6.0, 7.0],
        &[2, 3],
    )))?;
    r6.forward_recursive(1, false)?;

    let upstream = Tensor::ones(&[2, 3]);
    let grad = r6
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    // x=-2 → 0, x=-1 → 0, x=0 → 0, x=3 → 1, x=6 → 0, x=7 → 0
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 2]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 2]], 0.0, epsilon = 1e-6);
    Ok(())
}

#[test]
fn test_relu6_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();
    let x = graph.parameter(&[2, 3], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[-2.0, -1.0, 0.0, 3.0, 6.0, 7.0], &[2, 3]))?;

    let result = x.relu6();
    let target = graph.input(&Tensor::zeros(&[2, 3]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;
    assert!(x.grad()?.is_some());
    Ok(())
}

#[test]
fn test_relu6_cannot_set_value() {
    let graph = Graph::new();
    let x = graph.input(&Tensor::ones(&[2, 2])).unwrap();
    let result = x.relu6();
    assert!(result.set_value(&Tensor::ones(&[2, 2])).is_err());
}
