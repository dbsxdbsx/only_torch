/*
 * @Author       : 老董
 * @Date         : 2026-04-28
 * @Description  : DeformableConv2d 节点单元测试
 *
 * 参考值来源：PyTorch 2.6 + torchvision.ops.deform_conv2d。
 */

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::{Graph, GraphError, Var};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

fn assert_slice_close(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(a, e, epsilon = eps);
    }
}

fn reference_input() -> Tensor {
    Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
    )
}

fn reference_kernel() -> Tensor {
    Tensor::new(&[1.0, -1.0, 0.5, 2.0], &[1, 1, 2, 2])
}

fn reference_offset() -> Tensor {
    #[rustfmt::skip]
    let data = [
        0.25, 0.0, 0.0, 0.0,
        0.50, 0.0, 0.0, 0.0,
        0.0, -0.25, 0.0, 0.0,
        0.0, 0.25, 0.0, 0.0,
        0.0, 0.0, 0.50, 0.0,
        0.0, 0.0, -0.50, 0.0,
        0.0, 0.0, 0.0, -0.50,
        0.0, 0.0, 0.0, -0.25,
    ];
    Tensor::new(&data, &[1, 8, 2, 2])
}

fn build_reference_node(
    offset_shape: &[usize],
) -> Result<
    (
        Graph,
        Rc<crate::nn::NodeInner>,
        Rc<crate::nn::NodeInner>,
        Rc<crate::nn::NodeInner>,
        Rc<crate::nn::NodeInner>,
    ),
    GraphError,
> {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 3, 3], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;
    let offset = inner
        .borrow_mut()
        .create_basic_input_node(offset_shape, Some("offset"))?;
    let deform = inner.borrow_mut().create_deformable_conv2d_node(
        vec![input.clone(), kernel.clone(), offset.clone()],
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        Some("deform"),
    )?;
    Ok((graph, input, kernel, offset, deform))
}

#[test]
fn test_deformable_conv2d_zero_offset_matches_pytorch_reference() -> Result<(), GraphError> {
    let (_graph, input, kernel, offset, deform) = build_reference_node(&[1, 8, 2, 2])?;
    input.set_value(Some(&reference_input()))?;
    kernel.set_value(Some(&reference_kernel()))?;
    offset.set_value(Some(&Tensor::zeros(&[1, 8, 2, 2])))?;

    deform.forward_recursive(1, false)?;
    let output = deform.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
    assert_slice_close(output.data_as_slice(), &[11.0, 13.5, 18.5, 21.0], 1e-5);

    let upstream = Tensor::ones(&[1, 1, 2, 2]);
    let input_grad = deform
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    assert_slice_close(
        input_grad.data_as_slice(),
        &[1.0, 0.0, -1.0, 1.5, 2.5, 1.0, 0.5, 2.5, 2.0],
        1e-5,
    );

    let kernel_grad = deform
        .calc_grad_to_parent_index(1, &upstream)?
        .resolve(&upstream);
    assert_slice_close(kernel_grad.data_as_slice(), &[12.0, 16.0, 24.0, 28.0], 1e-5);

    Ok(())
}

#[test]
fn test_deformable_conv2d_nonzero_offset_matches_pytorch_reference() -> Result<(), GraphError> {
    let (_graph, input, kernel, offset, deform) = build_reference_node(&[1, 8, 2, 2])?;
    input.set_value(Some(&reference_input()))?;
    kernel.set_value(Some(&reference_kernel()))?;
    offset.set_value(Some(&reference_offset()))?;

    deform.forward_recursive(1, false)?;
    let output = deform.value().unwrap();
    assert_slice_close(
        output.data_as_slice(),
        &[12.25, 14.8125, 15.875, 17.5],
        1e-5,
    );

    let upstream = Tensor::ones(&[1, 1, 2, 2]);
    let input_grad = deform
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);
    assert_slice_close(
        input_grad.data_as_slice(),
        &[0.375, 0.375, -0.5625, 1.625, 2.875, 1.75, 0.125, 2.75, 0.75],
        1e-5,
    );

    let kernel_grad = deform
        .calc_grad_to_parent_index(1, &upstream)?
        .resolve(&upstream);
    assert_slice_close(
        kernel_grad.data_as_slice(),
        &[13.25, 14.6875, 18.75, 26.25],
        1e-5,
    );

    let offset_grad = deform
        .calc_grad_to_parent_index(2, &upstream)?
        .resolve(&upstream);
    #[rustfmt::skip]
    let expected_offset_grad = [
        3.0, 3.0, 3.0, 3.0,
        1.0, 1.0, 1.0, 1.0,
        -3.0, -2.25, -3.0, -3.0,
        -1.0, 2.25, -1.0, 6.0,
        1.5, 1.5, -1.75, -4.0,
        0.5, 0.5, 1.75, 0.5,
        6.0, 6.0, -16.0, 6.0,
        2.0, -12.0, 2.0, 2.0,
    ];
    assert_slice_close(offset_grad.data_as_slice(), &expected_offset_grad, 1e-5);

    Ok(())
}

#[test]
fn test_deformable_conv2d_rejects_bad_offset_shape() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 3, 3], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;
    let bad_offset = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 6, 2, 2], Some("bad_offset"))?;

    let result = inner.borrow_mut().create_deformable_conv2d_node(
        vec![input, kernel, bad_offset],
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        Some("deform"),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("offset"));
    Ok(())
}

#[test]
fn test_deformable_conv2d_descriptor_roundtrip() -> Result<(), GraphError> {
    let (graph, input, kernel, offset, deform) = build_reference_node(&[1, 8, 2, 2])?;
    input.set_value(Some(&reference_input()))?;
    kernel.set_value(Some(&reference_kernel()))?;
    offset.set_value(Some(&reference_offset()))?;
    deform.forward_recursive(1, false)?;

    let deform_var = Var::new_with_rc_graph(deform.clone(), &graph.inner_rc());
    let desc = deform_var.to_graph_descriptor();
    assert!(desc.nodes.iter().any(|node| {
        matches!(
            node.node_type,
            NodeTypeDescriptor::DeformableConv2d {
                stride: (1, 1),
                padding: (0, 0),
                dilation: (1, 1),
                deformable_groups: 1,
            }
        )
    }));

    let rebuilt = Graph::from_descriptor(&desc)?;
    assert_eq!(rebuilt.outputs.len(), 1);
    Ok(())
}
