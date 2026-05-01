//! 拼接式 IoU / GIoU / DIoU / CIoU 损失实现
//!
//! 与 [`crate::vision::detection::BBox`] 中的 CPU 标量版本算法保持等价，但全部
//! 用计算图上的可微算子拼接，由 autograd 自动反向，避免 finite-diff 的精度损耗
//! 与 O(N²) 开销。
//!
//! 设计要点（详见 `.cursor/plans/bbox_loss_autograd_migration_*.plan.md` Phase 2）：
//!
//! 1. **可视化折叠**：每个 helper 入口建立 [`NodeGroupContext`]，把 30+ 内部
//!    节点在 `.dot` 输出里折叠成单个 cluster；4 套 IoU 用各自独立的 group_type
//!    （`IoULoss / GIoULoss / DIoULoss / CIoULoss`），用户视角与 fused 节点等价。
//! 2. **shape 严格相等**：直接拒绝 `[N,4]` vs `[1,4]` 的隐式 broadcast，避免
//!    Maximum/Minimum/Subtract 节点的 numpy-style 隐式扩展把训练误差吃掉。
//! 3. **target 显式 detach**：拼接式 helper 必须显式 `target.detach()`，否则
//!    若 target 是 Parameter 会反向收梯度（fused 时代由节点 hard-reject 实现，
//!    现在由 helper 入口统一兜底）。
//! 4. **所有宽高 max(0)**：用 [`VarActivationOps::relu`] 取代 `Maximum(_, 0)`——
//!    ReLU 等价 `max(0, x)`、零分配、天然属于 cluster；与 [`BBox::width / height`]
//!    的 `max(0)` 语义一致。
//! 5. **CIoU 零面积短路**：[`BBox::ciou`] 在任一 w/h ≤ 0 时直接返回 DIoU；
//!    宽高已由 ReLU 截为非负，因此用 [`VarActivationOps::sign`] 构造 `w > 0`
//!    的运行时 valid mask，`ciou = diou - valid * α * v` 拼接式实现等价短路。

use crate::nn::{GraphError, NodeGroupContext, Var, VarActivationOps, VarReduceOps, VarShapeOps};
use crate::vision::detection::{BBoxLossKind, BoxFormat};

/// IoU / GIoU / DIoU / CIoU 公共数值稳定项
const IOU_NUMERICAL_EPS: f32 = 1e-7;

const FOUR_OVER_PI_SQUARED: f32 = 4.0 / (std::f32::consts::PI * std::f32::consts::PI);

/// 顶层入口：按 [`BBoxLossKind`] 分发到 4 套拼接式 helper。
///
/// 公共契约：
/// - 调用方传入的 `target` 可以是任意 Var，本函数内部会先 `detach()`，确保 target
///   永不收梯度（与 fused 节点 hard-reject 行为等价）。
/// - 返回 shape `[1, 1]` 的标量 Var（`Mean` reduction），与旧 fused 节点 API 完全一致。
pub(crate) fn compute_bbox_loss(
    input: &Var,
    target: &Var,
    kind: BBoxLossKind,
    format: BoxFormat,
) -> Result<Var, GraphError> {
    if !input.same_graph(target) {
        return Err(GraphError::InvalidOperation(
            "bbox_loss: input 和 target 必须来自同一 Graph".to_string(),
        ));
    }
    validate_bbox_shape(input, target)?;

    let group_type = match kind {
        BBoxLossKind::IoU => "IoULoss",
        BBoxLossKind::GIoU => "GIoULoss",
        BBoxLossKind::DIoU => "DIoULoss",
        BBoxLossKind::CIoU => "CIoULoss",
    };
    let instance_id = input.graph().borrow_mut().next_node_group_instance_id();
    let _guard = NodeGroupContext::new(input, group_type, instance_id);

    let target = target.detach();
    let pred = box_to_xyxy(input, format)?;
    let target_box = box_to_xyxy(&target, format)?;

    match kind {
        BBoxLossKind::IoU => iou_loss(&pred, &target_box),
        BBoxLossKind::GIoU => giou_loss(&pred, &target_box),
        BBoxLossKind::DIoU => diou_loss(&pred, &target_box),
        BBoxLossKind::CIoU => ciou_loss(&pred, &target_box),
    }
}

/// 拼接式可微视角下的 bbox：四个角坐标 + 已经 `max(0)` 处理过的 w/h。
///
/// `w / h` 字段保证非负，等价 [`BBox::width / height`] 的 `(self.x2 - self.x1).max(0.0)`。
struct Box4 {
    x1: Var,
    y1: Var,
    x2: Var,
    y2: Var,
    w: Var,
    h: Var,
}

/// 校验 `input` 与 `target` 必须是 `[N, 4]` 且 N 严格相等。
///
/// 拼接式拓扑下 [`Var`] 的 `Maximum / Minimum / Subtract / Divide` 等节点都默认走
/// numpy 风格 broadcast，会让 `[N, 4]` 与 `[1, 4]` 隐式扩展过去——拼接式入口
/// 必须在 helper 入口处显式拒绝，等价 fused 时代节点 ctor 的 hard check。
fn validate_bbox_shape(input: &Var, target: &Var) -> Result<(), GraphError> {
    let in_shape = input.node().shape();
    let tg_shape = target.node().shape();
    if in_shape.len() != 2 || in_shape[1] != 4 {
        return Err(GraphError::ShapeMismatch {
            expected: vec![0, 4],
            got: in_shape.clone(),
            message: "bbox_loss: input 形状必须为 [N, 4]".to_string(),
        });
    }
    if in_shape != tg_shape {
        return Err(GraphError::ShapeMismatch {
            expected: in_shape,
            got: tg_shape,
            message: "bbox_loss: input 与 target 形状必须严格相等（不允许隐式 broadcast）"
                .to_string(),
        });
    }
    Ok(())
}

/// 把 `[N, 4]` Var 解构为四个 `[N, 1]` 坐标 + 已 `max(0)` 的 w / h。
///
/// `CxCyWh` 入口与 [`crate::vision::detection::BBox::from_cxcywh`] 一致：先把 w / h
/// 截到非负，再据此推导 xyxy；保证下游不会对负宽高做"反向重叠"几何。
fn box_to_xyxy(input: &Var, format: BoxFormat) -> Result<Box4, GraphError> {
    let c0 = input.narrow(1, 0, 1)?;
    let c1 = input.narrow(1, 1, 1)?;
    let c2 = input.narrow(1, 2, 1)?;
    let c3 = input.narrow(1, 3, 1)?;

    match format {
        BoxFormat::XyXy => {
            let w = (&c2 - &c0).relu();
            let h = (&c3 - &c1).relu();
            Ok(Box4 {
                x1: c0,
                y1: c1,
                x2: c2,
                y2: c3,
                w,
                h,
            })
        }
        BoxFormat::CxCyWh => {
            let cx = c0;
            let cy = c1;
            let raw_w = c2;
            let raw_h = c3;
            let w = raw_w.relu();
            let h = raw_h.relu();
            let half_w = &w * 0.5_f32;
            let half_h = &h * 0.5_f32;
            let x1 = &cx - &half_w;
            let y1 = &cy - &half_h;
            let x2 = &cx + &half_w;
            let y2 = &cy + &half_h;
            Ok(Box4 {
                x1,
                y1,
                x2,
                y2,
                w,
                h,
            })
        }
    }
}

/// IoU / GIoU / DIoU / CIoU 共用的中间量。
struct IouComponents {
    iou: Var,
    union: Var,
    /// enclosing box 的宽（已 `max(0)`）
    enc_w: Var,
    /// enclosing box 的高（已 `max(0)`）
    enc_h: Var,
}

/// 计算 IoU 与 enclosing box 的几何中间量。
fn iou_components(pred: &Box4, target: &Box4) -> Result<IouComponents, GraphError> {
    let inter_x1 = pred.x1.maximum(&target.x1)?;
    let inter_y1 = pred.y1.maximum(&target.y1)?;
    let inter_x2 = pred.x2.minimum(&target.x2)?;
    let inter_y2 = pred.y2.minimum(&target.y2)?;
    let inter_w = (&inter_x2 - &inter_x1).relu();
    let inter_h = (&inter_y2 - &inter_y1).relu();
    let inter = &inter_w * &inter_h;

    let area_pred = &pred.w * &pred.h;
    let area_target = &target.w * &target.h;
    let union_no_eps = &area_pred + &area_target - &inter;
    let union = union_no_eps + IOU_NUMERICAL_EPS;
    let iou = &inter / &union;

    let enc_x1 = pred.x1.minimum(&target.x1)?;
    let enc_y1 = pred.y1.minimum(&target.y1)?;
    let enc_x2 = pred.x2.maximum(&target.x2)?;
    let enc_y2 = pred.y2.maximum(&target.y2)?;
    let enc_w = (&enc_x2 - &enc_x1).relu();
    let enc_h = (&enc_y2 - &enc_y1).relu();

    Ok(IouComponents {
        iou,
        union,
        enc_w,
        enc_h,
    })
}

/// `1 - IoU` mean
fn iou_loss(pred: &Box4, target: &Box4) -> Result<Var, GraphError> {
    let comp = iou_components(pred, target)?;
    let one_minus = 1.0_f32 - &comp.iou;
    Ok(one_minus.mean())
}

/// `1 - GIoU` mean
///
/// `GIoU = IoU - (A_enc - union) / A_enc`
fn giou_loss(pred: &Box4, target: &Box4) -> Result<Var, GraphError> {
    let comp = iou_components(pred, target)?;
    let enc_area = &(&comp.enc_w * &comp.enc_h) + IOU_NUMERICAL_EPS;
    let extra = &(&enc_area - &comp.union) / &enc_area;
    let giou = &comp.iou - &extra;
    let one_minus = 1.0_f32 - &giou;
    Ok(one_minus.mean())
}

/// 计算 DIoU 主体（不做最终 `1 - x` 与 mean），供 CIoU 复用。
fn diou_core(pred: &Box4, target: &Box4) -> Result<(Var, IouComponents), GraphError> {
    let comp = iou_components(pred, target)?;

    let cx_pred = (&pred.x1 + &pred.x2) * 0.5_f32;
    let cy_pred = (&pred.y1 + &pred.y2) * 0.5_f32;
    let cx_target = (&target.x1 + &target.x2) * 0.5_f32;
    let cy_target = (&target.y1 + &target.y2) * 0.5_f32;
    let dx = &cx_pred - &cx_target;
    let dy = &cy_pred - &cy_target;
    let center_dist_sq = &dx.square() + &dy.square();

    let diag_sq_no_eps = &comp.enc_w.square() + &comp.enc_h.square();
    let diag_sq = diag_sq_no_eps + IOU_NUMERICAL_EPS;
    let center_term = &center_dist_sq / &diag_sq;

    let diou = &comp.iou - &center_term;
    Ok((diou, comp))
}

/// `1 - DIoU` mean
fn diou_loss(pred: &Box4, target: &Box4) -> Result<Var, GraphError> {
    let (diou, _comp) = diou_core(pred, target)?;
    let one_minus = 1.0_f32 - &diou;
    Ok(one_minus.mean())
}

/// `1 - CIoU` mean，与 [`BBox::ciou`] 等价（含零面积短路退化为 DIoU）
///
/// 短路实现：宽高已由 ReLU 保证非负，用 [`VarActivationOps::sign`] 得到
/// `w > 0` 的硬 mask，反向梯度恒为 0。这里不用 [`VarFilterOps::where_cond`]
/// 是因为当前 `WhereCond` 接收构建期静态 Tensor mask，无法表达依赖 pred/target
/// 运行时数值的退化条件。
fn ciou_loss(pred: &Box4, target: &Box4) -> Result<Var, GraphError> {
    let (diou, comp) = diou_core(pred, target)?;

    let atan_target = target.w.atan2(&target.h)?;
    let atan_pred = pred.w.atan2(&pred.h)?;
    let angle_diff = &atan_target - &atan_pred;
    let v = angle_diff.square() * FOUR_OVER_PI_SQUARED;

    let alpha_den = &(1.0_f32 - &comp.iou) + &v + IOU_NUMERICAL_EPS;
    let alpha = &v / &alpha_den;
    let alpha_v = &alpha * &v;

    // 退化短路：w_pred / h_pred / w_target / h_target 任一 <= 0 时
    // valid_mask = 0，让 ciou_full 的角度惩罚归零，等效"退化为 DIoU"。
    let valid_pw = pred.w.sign();
    let valid_ph = pred.h.sign();
    let valid_tw = target.w.sign();
    let valid_th = target.h.sign();
    let valid = &(&(&valid_pw * &valid_ph) * &valid_tw) * &valid_th;

    let penalty = &valid * &alpha_v;
    let ciou = &diou - &penalty;
    let one_minus = 1.0_f32 - &ciou;
    Ok(one_minus.mean())
}
