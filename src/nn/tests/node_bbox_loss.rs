/*
 * @Description  : BBox IoU-family 损失节点单元测试
 *
 * 覆盖通用检测框回归损失：IoU / GIoU / DIoU / CIoU、坐标格式转换、
 * reduction、shape 校验与 VJP。
 */

use crate::nn::nodes::raw_node::Reduction;
use crate::nn::{BBoxLossKind, Graph, GraphError, Mode, NodeTypeDescriptor, VarLossOps};
use crate::tensor::Tensor;
use crate::vision::detection::BoxFormat;
use approx::assert_abs_diff_eq;

#[test]
fn test_bbox_iou_family_forward_known_values_xyxy() {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 2.0, 2.0], &[1, 4]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 0.0, 3.0, 2.0], &[1, 4]))
        .unwrap();

    let iou = input
        .bbox_loss(&target, BBoxLossKind::IoU, BoxFormat::XyXy)
        .unwrap();
    let giou = input.giou_loss(&target, BoxFormat::XyXy).unwrap();
    let diou = input.diou_loss(&target, BoxFormat::XyXy).unwrap();
    let ciou = input.ciou_loss(&target, BoxFormat::XyXy).unwrap();

    iou.forward().unwrap();
    giou.forward().unwrap();
    diou.forward().unwrap();
    ciou.forward().unwrap();

    assert_abs_diff_eq!(iou.item().unwrap(), 2.0 / 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(giou.item().unwrap(), 2.0 / 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(diou.item().unwrap(), 29.0 / 39.0, epsilon = 1e-6);
    assert_abs_diff_eq!(ciou.item().unwrap(), 29.0 / 39.0, epsilon = 1e-6);
}

#[test]
fn test_bbox_loss_cxcywh_perfect_match_is_zero() {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(&[0.5, 0.5, 1.0, 1.0], &[1, 4]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.5, 0.5, 1.0, 1.0], &[1, 4]))
        .unwrap();

    let loss = input
        .bbox_loss(&target, BBoxLossKind::CIoU, BoxFormat::CxCyWh)
        .unwrap();
    loss.forward().unwrap();

    assert_abs_diff_eq!(loss.item().unwrap(), 0.0, epsilon = 1e-6);
}

#[test]
fn test_bbox_giou_penalizes_non_overlapping_boxes() {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0, 1.0], &[1, 4]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[2.0, 0.0, 3.0, 1.0], &[1, 4]))
        .unwrap();

    let loss = input.giou_loss(&target, BoxFormat::XyXy).unwrap();
    loss.forward().unwrap();

    assert_abs_diff_eq!(loss.item().unwrap(), 4.0 / 3.0, epsilon = 1e-6);
}

#[test]
fn test_bbox_loss_sum_reduction() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("input"))?;
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("target"))?;
    let loss = inner.borrow_mut().create_bbox_loss_node(
        input.clone(),
        target.clone(),
        BBoxLossKind::IoU,
        BoxFormat::XyXy,
        Reduction::Sum,
        Some("bbox_loss"),
    )?;

    input.set_value(Some(&Tensor::new(
        &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0],
        &[2, 4],
    )))?;
    target.set_value(Some(&Tensor::new(
        &[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 3.0, 2.0],
        &[2, 4],
    )))?;
    loss.forward_recursive(1, Mode::Train)?;

    let loss_val = loss.value().unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], 2.0 / 3.0, epsilon = 1e-6);
    Ok(())
}

#[test]
fn test_bbox_loss_rejects_non_bbox_shape() {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0, 1.0], &[1, 4]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0, 1.0, 0.9], &[1, 5]))
        .unwrap();

    let err = input
        .bbox_loss(&target, BBoxLossKind::IoU, BoxFormat::XyXy)
        .unwrap_err();
    assert!(
        format!("{err}").contains("[N, 4]"),
        "错误信息应说明 bbox shape，实际: {err}"
    );
}

#[test]
fn test_bbox_loss_vjp_xyxy_matches_known_iou_gradient() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("input"))?;
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("target"))?;
    let loss = inner.borrow_mut().create_bbox_loss_node(
        input.clone(),
        target.clone(),
        BBoxLossKind::IoU,
        BoxFormat::XyXy,
        Reduction::Mean,
        Some("bbox_loss"),
    )?;

    input.set_value(Some(&Tensor::new(&[0.0, 0.0, 2.0, 2.0], &[1, 4])))?;
    target.set_value(Some(&Tensor::new(&[0.5, 0.0, 2.5, 2.0], &[1, 4])))?;
    loss.forward_recursive(1, Mode::Train)?;

    let upstream = Tensor::ones(&[1, 1]);
    let grad = loss
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_eq!(grad.shape(), &[1, 4]);
    assert_abs_diff_eq!(grad[[0, 0]], -0.24, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[0, 2]], -0.4, epsilon = 1e-3);
    Ok(())
}

#[test]
fn test_bbox_loss_vjp_scales_with_upstream_gradient() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("input"))?;
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("target"))?;
    let loss = inner.borrow_mut().create_bbox_loss_node(
        input.clone(),
        target.clone(),
        BBoxLossKind::IoU,
        BoxFormat::XyXy,
        Reduction::Mean,
        Some("bbox_loss"),
    )?;

    input.set_value(Some(&Tensor::new(&[0.0, 0.0, 2.0, 2.0], &[1, 4])))?;
    target.set_value(Some(&Tensor::new(&[0.5, 0.0, 2.5, 2.0], &[1, 4])))?;
    loss.forward_recursive(1, Mode::Train)?;

    let upstream = Tensor::new(&[2.0], &[1, 1]);
    let grad = loss
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    assert_abs_diff_eq!(grad[[0, 0]], -0.48, epsilon = 2e-3);
    assert_abs_diff_eq!(grad[[0, 2]], -0.8, epsilon = 2e-3);
    Ok(())
}

#[test]
fn test_bbox_loss_target_has_no_gradient() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("input"))?;
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("target"))?;
    let loss = inner.borrow_mut().create_bbox_loss_node(
        input.clone(),
        target.clone(),
        BBoxLossKind::GIoU,
        BoxFormat::XyXy,
        Reduction::Mean,
        Some("bbox_loss"),
    )?;

    input.set_value(Some(&Tensor::new(&[0.0, 0.0, 1.0, 1.0], &[1, 4])))?;
    target.set_value(Some(&Tensor::new(&[0.0, 0.0, 1.0, 1.0], &[1, 4])))?;
    loss.forward_recursive(1, Mode::Train)?;

    let err = loss
        .calc_grad_to_parent_index(1, &Tensor::ones(&[1, 1]))
        .unwrap_err();
    assert!(
        format!("{err}").contains("target"),
        "错误信息应说明 target 不计算梯度，实际: {err}"
    );
    Ok(())
}

#[test]
fn test_bbox_loss_descriptor_roundtrip() -> Result<(), GraphError> {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 2.0, 2.0], &[1, 4]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 0.0, 3.0, 2.0], &[1, 4]))
        .unwrap();
    let loss = input.ciou_loss(&target, BoxFormat::XyXy).unwrap();

    let desc = loss.to_graph_descriptor();
    assert!(desc.nodes.iter().any(|node| {
        matches!(
            node.node_type,
            NodeTypeDescriptor::BBoxLoss {
                kind: BBoxLossKind::CIoU,
                format: BoxFormat::XyXy,
                reduction: Reduction::Mean,
            }
        )
    }));

    let rebuilt = Graph::from_descriptor(&desc)?;
    assert_eq!(rebuilt.outputs.len(), 1);
    Ok(())
}

