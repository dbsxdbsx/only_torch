/*
 * @Author       : 老董
 * @Description  : Repeat 节点单元测试
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// Repeat 前向传播
#[test]
fn test_repeat_forward() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let rep = inner
        .borrow_mut()
        .create_repeat_node(x.clone(), vec![2, 3], Some("rep"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])))
        .unwrap();

    rep.forward_recursive(1, true).unwrap();
    let val = rep.value().unwrap();

    assert_eq!(val.shape(), &[4, 9]);
    // 第一行 [1,2,3] 重复 3 次: [1,2,3,1,2,3,1,2,3]
    assert_abs_diff_eq!(val[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(val[[0, 3]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(val[[0, 6]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(val[[0, 1]], 2.0, epsilon = 1e-6);
}

/// Repeat Var API
#[test]
fn test_repeat_var_api() -> Result<(), GraphError> {
    let graph = Graph::new();
    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;
    let y = x.repeat(&[1, 3])?;
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 6]);
    // [1,2] 重复 3 次 → [1,2,1,2,1,2]
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 4]], 1.0, epsilon = 1e-6);

    Ok(())
}

/// Repeat 反向传播
#[test]
fn test_repeat_backward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;

    let y = x.repeat(&[2, 3])?;
    let target = graph.input(&Tensor::zeros(&[4, 6]))?;
    let loss = y.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    let grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(grad.shape(), &[2, 2]);
    // 每个元素被重复了 2*3=6 次，梯度应被放大
    let flat = grad.flatten_view();
    assert!(flat.iter().all(|&v| v.abs() > 0.0), "梯度不应全为 0");

    Ok(())
}
