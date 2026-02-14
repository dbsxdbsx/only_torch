/*
 * @Author       : 老董
 * @Description  : RMSNormOp 节点单元测试
 *
 * RMSNorm: x_hat = x / sqrt(mean(x^2) + eps)
 * 与 LayerNorm 的区别：不减均值
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// RMSNorm 前向传播（2D [2, 4]）
///
/// Python 对照：
/// ```python
/// x = torch.tensor([[1., 2., 3., 4.], [0.1, 0.2, 0.3, 0.4]])
/// rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)
/// x_hat = x / rms
/// ```
/// rms[0] = sqrt((1+4+9+16)/4 + 1e-5) = sqrt(7.50001) ≈ 2.7386
/// x_hat[0] = [0.3651, 0.7303, 1.0954, 1.4606]
#[test]
fn test_rms_norm_op_forward_2d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("x"))
        .unwrap();
    let rn = inner
        .borrow_mut()
        .create_rms_norm_op_node(x.clone(), 1, 1e-5, Some("rn"))
        .unwrap();

    x.set_value(Some(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4],
        &[2, 4],
    )))
    .unwrap();

    rn.forward_recursive(1, true).unwrap();
    let val = rn.value().unwrap();

    assert_eq!(val.shape(), &[2, 4]);
    // rms[0] = sqrt(7.5 + 1e-5) ≈ 2.73861
    // x_hat[0] = [1/2.73861, 2/2.73861, 3/2.73861, 4/2.73861]
    assert_abs_diff_eq!(val[[0, 0]], 0.3651, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[0, 1]], 0.7303, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[0, 2]], 1.0954, epsilon = 1e-3);
    assert_abs_diff_eq!(val[[0, 3]], 1.4606, epsilon = 1e-3);
    // 第二行同比例缩放
    assert_abs_diff_eq!(val[[1, 0]], 0.3651, epsilon = 1e-3);
}

/// RMSNorm 前向传播（3D [2, 2, 3]）
#[test]
fn test_rms_norm_op_forward_3d() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2, 3], Some("x"))
        .unwrap();
    let rn = inner
        .borrow_mut()
        .create_rms_norm_op_node(x.clone(), 1, 1e-5, Some("rn"))
        .unwrap();

    let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    x.set_value(Some(&Tensor::new(&data, &[2, 2, 3]))).unwrap();

    rn.forward_recursive(1, true).unwrap();
    let val = rn.value().unwrap();

    assert_eq!(val.shape(), &[2, 2, 3]);
    // [1,2,3] → rms = sqrt((1+4+9)/3 + eps) = sqrt(4.667) ≈ 2.1602
    assert_abs_diff_eq!(val[[0, 0, 0]], 1.0 / 2.1602, epsilon = 1e-3);
}

/// RMSNorm VJP（sum loss → 梯度非零但有结构约束）
#[test]
fn test_rms_norm_op_vjp() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("x"))
        .unwrap();
    let rn = inner
        .borrow_mut()
        .create_rms_norm_op_node(x.clone(), 1, 1e-5, Some("rn"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4])))
        .unwrap();

    rn.forward_recursive(1, true).unwrap();

    let upstream = Tensor::ones(&[1, 4]);
    let grad = rn
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_eq!(grad.shape(), &[1, 4]);
    // RMSNorm 梯度存在但不一定为零
    // 与 LayerNorm 不同，RMSNorm 的 sum(upstream) 梯度不一定为 0
    let flat = grad.flatten_view();
    let grad_abs: f32 = flat.iter().map(|v| v.abs()).sum();
    // 至少应该有非零值
    assert!(grad_abs > 0.0, "RMSNorm 梯度不应全为 0");

    Ok(())
}

/// 端到端反向传播
#[test]
fn test_rms_norm_op_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 4], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0],
        &[2, 4],
    ))?;

    let rn_out = {
        use std::rc::Rc;
        let rn_node = graph
            .inner_mut()
            .create_rms_norm_op_node(Rc::clone(x.node()), 1, 1e-5, Some("rn"))?;
        crate::nn::Var::new_with_rc_graph(rn_node, &graph.inner_rc())
    };

    let target = graph.input(&Tensor::zeros(&[2, 4]))?;
    let loss = rn_out.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 4]);

    Ok(())
}
