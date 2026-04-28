/*
 * @Author       : 老董
 * @Date         : 2026-04-28
 * @Description  : DeformableConv2d layer 单元测试
 */

use crate::nn::{DeformableConv2d, Graph, GraphError, Module, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_deformable_conv2d_layer_zero_offset_forward() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
    ))?;
    let layer = DeformableConv2d::new(
        &graph,
        1,
        1,
        (2, 2),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        false,
        "deform",
    )?;
    layer
        .kernel()
        .set_value(&Tensor::new(&[1.0, -1.0, 0.5, 2.0], &[1, 1, 2, 2]))?;

    let y = layer.forward(&x);
    y.forward()?;
    assert_eq!(y.value()?.unwrap().shape(), &[1, 1, 2, 2]);
    for (&actual, &expected) in y
        .value()?
        .unwrap()
        .data_as_slice()
        .iter()
        .zip([11.0, 13.5, 18.5, 21.0].iter())
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
    }
    Ok(())
}

#[test]
fn test_deformable_conv2d_layer_backward_updates_main_and_offset_params() -> Result<(), GraphError>
{
    let graph = Graph::new_with_seed(7);
    let x = graph.input(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
    ))?;
    let target = graph.input(&Tensor::zeros(&[1, 1, 2, 2]))?;
    let layer = DeformableConv2d::new(
        &graph,
        1,
        1,
        (2, 2),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        true,
        "deform_train",
    )?;
    layer
        .kernel()
        .set_value(&Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[1, 1, 2, 2]))?;

    let y = layer.forward(&x);
    let loss = y.mse_loss(&target)?;
    loss.backward()?;

    assert_eq!(layer.parameters().len(), 4);
    assert!(layer.kernel().grad()?.is_some());
    assert!(layer.bias().unwrap().grad()?.is_some());
    assert!(layer.offset_kernel().grad()?.is_some());
    assert!(layer.offset_bias().grad()?.is_some());
    let offset_grad_sum: f32 = layer
        .offset_kernel()
        .grad()?
        .unwrap()
        .data_as_slice()
        .iter()
        .map(|v| v.abs())
        .sum();
    assert!(
        offset_grad_sum > 0.0,
        "offset predictor 应通过 offset 梯度获得训练信号"
    );
    Ok(())
}
