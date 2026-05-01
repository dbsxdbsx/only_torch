//! 拼接式 IoU / GIoU / DIoU / CIoU bbox_loss 单元测试
//!
//! 取代旧版 fused 节点测试 `src/nn/tests/node_bbox_loss.rs`。覆盖：
//!
//! - 4 套 IoU 系列损失的 forward known-value（手算可核对的 IoU=2/3、零面积短路等）。
//! - 4 套 IoU 系列损失的 backward 对照 PyTorch oracle（由 `tests/bbox_loss_reference.py`
//!   生成，统一收紧到 `epsilon = 1e-5`）。
//! - shape 严格相等（拒绝 `[1,4]` vs `[N,4]` 的隐式 broadcast）。
//! - target 即便是 [`Init`] 创建的 Parameter，也不会反向收梯度（拼接式 helper 内部
//!   显式 `target.detach()`）。
//! - CIoU 在零面积退化样本上仍能稳定 forward，且与 DIoU 等价。
//! - 4 套 IoU 系列均把内部 30+ 节点折叠成单个 `NodeGroupTag` cluster，可视化时
//!   呈现为单一的 IoULoss / GIoULoss / DIoULoss / CIoULoss 节点。
//!
//! oracle 设计原则：所有 backward 样本都精心避开 ReLU / Maximum / Minimum / Atan2
//! 的非可微拐点（perfect match、零面积、刚好相切、`(0, 0)`），保证 only_torch 与
//! PyTorch 在 `epsilon ≤ 1e-5` 内逐元素一致；零面积退化 forward 短路语义在
//! [`test_ciou_degenerate_box_falls_back_to_diou`] 里单独硬编码测试。

use approx::assert_abs_diff_eq;

use crate::nn::{Graph, GraphError, Init, VarLossOps};
use crate::tensor::Tensor;
use crate::vision::detection::{BBoxLossKind, BoxFormat};

// ============================================================================
// PyTorch oracle 常量（由 tests/bbox_loss_reference.py 生成）
//
// 命名约定：`<样本名>_<IoU 类型>_LOSS / _GRAD`
//   - `_LOSS`：mean reduction 下的标量 loss（f32）
//   - `_GRAD`：input 张量的梯度（reduction = mean，flatten 顺序与 [N, 4] 一致）
//
// 任何 oracle 数值的更新都应通过重跑 `python tests/bbox_loss_reference.py`，
// 不要手工编辑这些常量。
// ============================================================================

// allow excessive_precision：oracle 由 PyTorch float64 生成，写成 10 位有效
// 数字方便与 Python 端对照；Rust 编译器会自动截断到 f32，clippy 抱怨过精也无碍
#[allow(clippy::excessive_precision)]
mod oracle {
    pub const PARTIAL_OVERLAP_IOU_ONE_THIRD_IOU_LOSS: f32 = 6.6666667222e-01;
    pub const PARTIAL_OVERLAP_IOU_ONE_THIRD_IOU_GRAD: [f32; 4] = [
        -1.1111110741e-01_f32,
        1.3888888512e-09_f32,
        -3.3333332778e-01_f32,
        -1.3888888512e-09_f32,
    ];
    pub const PARTIAL_OVERLAP_IOU_ONE_THIRD_GIOU_LOSS: f32 = 6.6666667222e-01;
    pub const PARTIAL_OVERLAP_IOU_ONE_THIRD_GIOU_GRAD: [f32; 4] = [
        -1.1111110741e-01_f32,
        1.3888888650e-09_f32,
        -3.3333332778e-01_f32,
        -1.3888888650e-09_f32,
    ];
    pub const PARTIAL_OVERLAP_IOU_ONE_THIRD_DIOU_LOSS: f32 = 7.4358974855e-01;
    pub const PARTIAL_OVERLAP_IOU_ONE_THIRD_DIOU_GRAD: [f32; 4] = [
        -1.5253122571e-01_f32,
        1.1834320733e-02_f32,
        -4.1025640411e-01_f32,
        -1.1834320733e-02_f32,
    ];
    pub const PARTIAL_OVERLAP_IOU_ONE_THIRD_CIOU_LOSS: f32 = 7.4358974855e-01;
    pub const PARTIAL_OVERLAP_IOU_ONE_THIRD_CIOU_GRAD: [f32; 4] = [
        -1.5253122571e-01_f32,
        1.1834320733e-02_f32,
        -4.1025640411e-01_f32,
        -1.1834320733e-02_f32,
    ];

    pub const DIAGONAL_OVERLAP_IOU_LOSS: f32 = 8.5714285765e-01;
    pub const DIAGONAL_OVERLAP_IOU_GRAD: [f32; 4] = [
        -2.0408163120e-02_f32,
        -2.0408163120e-02_f32,
        -6.1224489614e-02_f32,
        -6.1224489614e-02_f32,
    ];
    pub const DIAGONAL_OVERLAP_GIOU_LOSS: f32 = 1.0793650793e+00;
    pub const DIAGONAL_OVERLAP_GIOU_GRAD: [f32; 4] = [
        -3.8926681689e-02_f32,
        -3.8926681689e-02_f32,
        -1.1678004501e-01_f32,
        -1.1678004501e-01_f32,
    ];
    pub const DIAGONAL_OVERLAP_DIOU_LOSS: f32 = 9.6825396861e-01;
    pub const DIAGONAL_OVERLAP_DIOU_GRAD: [f32; 4] = [
        -2.9667422392e-02_f32,
        -2.9667422392e-02_f32,
        -8.9002267353e-02_f32,
        -8.9002267353e-02_f32,
    ];
    pub const DIAGONAL_OVERLAP_CIOU_LOSS: f32 = 9.6825396861e-01;
    pub const DIAGONAL_OVERLAP_CIOU_GRAD: [f32; 4] = [
        -2.9667422392e-02_f32,
        -2.9667422392e-02_f32,
        -8.9002267353e-02_f32,
        -8.9002267353e-02_f32,
    ];

    pub const PRED_CONTAINS_TARGET_IOU_LOSS: f32 = 7.5000000156e-01;
    pub const PRED_CONTAINS_TARGET_IOU_GRAD: [f32; 4] = [
        -6.2499999219e-02_f32,
        -6.2499999219e-02_f32,
        6.2499999219e-02_f32,
        6.2499999219e-02_f32,
    ];
    pub const PRED_CONTAINS_TARGET_GIOU_LOSS: f32 = 7.5000000156e-01;
    pub const PRED_CONTAINS_TARGET_GIOU_GRAD: [f32; 4] = [
        -6.2499999219e-02_f32,
        -6.2499999219e-02_f32,
        6.2499999219e-02_f32,
        6.2499999219e-02_f32,
    ];
    pub const PRED_CONTAINS_TARGET_DIOU_LOSS: f32 = 7.5000000156e-01;
    pub const PRED_CONTAINS_TARGET_DIOU_GRAD: [f32; 4] = [
        -6.2499999219e-02_f32,
        -6.2499999219e-02_f32,
        6.2499999219e-02_f32,
        6.2499999219e-02_f32,
    ];
    pub const PRED_CONTAINS_TARGET_CIOU_LOSS: f32 = 7.5000000156e-01;
    pub const PRED_CONTAINS_TARGET_CIOU_GRAD: [f32; 4] = [
        -6.2499999219e-02_f32,
        -6.2499999219e-02_f32,
        6.2499999219e-02_f32,
        6.2499999219e-02_f32,
    ];

    pub const ASPECT_RATIO_MISMATCH_IOU_LOSS: f32 = 8.0000000200e-01;
    pub const ASPECT_RATIO_MISMATCH_IOU_GRAD: [f32; 4] = [
        -7.9999998400e-02_f32,
        -3.9999999200e-02_f32,
        -1.5999999880e-01_f32,
        3.9999999200e-02_f32,
    ];
    pub const ASPECT_RATIO_MISMATCH_GIOU_LOSS: f32 = 9.6666666728e-01;
    pub const ASPECT_RATIO_MISMATCH_GIOU_GRAD: [f32; 4] = [
        -2.4444443770e-02_f32,
        -8.1666665867e-02_f32,
        -3.2666666408e-01_f32,
        8.1666665867e-02_f32,
    ];
    pub const ASPECT_RATIO_MISMATCH_DIOU_LOSS: f32 = 8.4000000184e-01;
    pub const ASPECT_RATIO_MISMATCH_DIOU_GRAD: [f32; 4] = [
        -1.1039999832e-01_f32,
        -2.7199999302e-02_f32,
        -1.9999999864e-01_f32,
        2.7199999302e-02_f32,
    ];
    pub const ASPECT_RATIO_MISMATCH_CIOU_LOSS: f32 = 8.4209078032e-01;
    pub const ASPECT_RATIO_MISMATCH_CIOU_GRAD: [f32; 4] = [
        -1.0513235772e-01_f32,
        -2.9635160572e-02_f32,
        -2.0467166214e-01_f32,
        2.9635160572e-02_f32,
    ];

    pub const CXCYWH_X_OFFSET_IOU_LOSS: f32 = 6.6666667222e-01;
    pub const CXCYWH_X_OFFSET_IOU_GRAD: [f32; 4] = [
        -4.4444443519e-01_f32,
        0.0000000000e+00_f32,
        -1.1111111019e-01_f32,
        -1.3888888512e-09_f32,
    ];
    pub const CXCYWH_X_OFFSET_GIOU_LOSS: f32 = 6.6666667222e-01;
    pub const CXCYWH_X_OFFSET_GIOU_GRAD: [f32; 4] = [
        -4.4444443519e-01_f32,
        0.0000000000e+00_f32,
        -1.1111111019e-01_f32,
        -1.3888888650e-09_f32,
    ];
    pub const CXCYWH_X_OFFSET_DIOU_LOSS: f32 = 7.4358974855e-01;
    pub const CXCYWH_X_OFFSET_DIOU_GRAD: [f32; 4] = [
        -5.6278762981e-01_f32,
        0.0000000000e+00_f32,
        -1.2886258920e-01_f32,
        -1.1834320733e-02_f32,
    ];
    pub const CXCYWH_X_OFFSET_CIOU_LOSS: f32 = 7.4358974855e-01;
    pub const CXCYWH_X_OFFSET_CIOU_GRAD: [f32; 4] = [
        -5.6278762981e-01_f32,
        0.0000000000e+00_f32,
        -1.2886258920e-01_f32,
        -1.1834320733e-02_f32,
    ];

    pub const CXCYWH_ASPECT_CHANGE_IOU_LOSS: f32 = 6.6666667222e-01;
    pub const CXCYWH_ASPECT_CHANGE_IOU_GRAD: [f32; 4] = [
        0.0000000000e+00_f32,
        0.0000000000e+00_f32,
        -1.1111111019e-01_f32,
        1.1111110741e-01_f32,
    ];
    pub const CXCYWH_ASPECT_CHANGE_GIOU_LOSS: f32 = 9.1666666910e-01;
    pub const CXCYWH_ASPECT_CHANGE_GIOU_GRAD: [f32; 4] = [
        0.0000000000e+00_f32,
        0.0000000000e+00_f32,
        -2.3611110862e-01_f32,
        2.3611110741e-01_f32,
    ];
    pub const CXCYWH_ASPECT_CHANGE_DIOU_LOSS: f32 = 6.6666667222e-01;
    pub const CXCYWH_ASPECT_CHANGE_DIOU_GRAD: [f32; 4] = [
        0.0000000000e+00_f32,
        0.0000000000e+00_f32,
        -1.1111111019e-01_f32,
        1.1111110741e-01_f32,
    ];
    pub const CXCYWH_ASPECT_CHANGE_CIOU_LOSS: f32 = 6.8451335317e-01;
    pub const CXCYWH_ASPECT_CHANGE_CIOU_GRAD: [f32; 4] = [
        0.0000000000e+00_f32,
        0.0000000000e+00_f32,
        -1.3911928478e-01_f32,
        1.3911928206e-01_f32,
    ];

    pub const BATCH_TWO_BOXES_IOU_LOSS: f32 = 7.6190476570e-01;
    pub const BATCH_TWO_BOXES_IOU_GRAD: [f32; 8] = [
        -5.5555553704e-02_f32,
        6.9444442558e-10_f32,
        -1.6666666389e-01_f32,
        -6.9444442558e-10_f32,
        -2.0408162682e-02_f32,
        -2.0408162682e-02_f32,
        -6.1224489067e-02_f32,
        -6.1224489067e-02_f32,
    ];
    pub const BATCH_TWO_BOXES_GIOU_LOSS: f32 = 8.7301587558e-01;
    pub const BATCH_TWO_BOXES_GIOU_GRAD: [f32; 8] = [
        -5.5555553704e-02_f32,
        6.9444443251e-10_f32,
        -1.6666666389e-01_f32,
        -6.9444443251e-10_f32,
        -3.8926681406e-02_f32,
        -3.8926681406e-02_f32,
        -1.1678004401e-01_f32,
        -1.1678004401e-01_f32,
    ];
    pub const BATCH_TWO_BOXES_DIOU_LOSS: f32 = 8.5592185912e-01;
    pub const BATCH_TWO_BOXES_DIOU_GRAD: [f32; 8] = [
        -7.6265612853e-02_f32,
        5.9171603667e-03_f32,
        -2.0512820205e-01_f32,
        -5.9171603667e-03_f32,
        -2.9667421993e-02_f32,
        -2.9667421993e-02_f32,
        -8.9002266691e-02_f32,
        -8.9002266691e-02_f32,
    ];
    pub const BATCH_TWO_BOXES_CIOU_LOSS: f32 = 8.5592185912e-01;
    pub const BATCH_TWO_BOXES_CIOU_GRAD: [f32; 8] = [
        -7.6265612853e-02_f32,
        5.9171603667e-03_f32,
        -2.0512820205e-01_f32,
        -5.9171603667e-03_f32,
        -2.9667421993e-02_f32,
        -2.9667421993e-02_f32,
        -8.9002266691e-02_f32,
        -8.9002266691e-02_f32,
    ];

    pub const NON_OVERLAP_IOU_LOSS: f32 = 1.0000000000e+00;
    pub const NON_OVERLAP_IOU_GRAD: [f32; 4] = [
        0.0000000000e+00_f32,
        0.0000000000e+00_f32,
        0.0000000000e+00_f32,
        0.0000000000e+00_f32,
    ];
    pub const NON_OVERLAP_GIOU_LOSS: f32 = 1.8749999945e+00;
    pub const NON_OVERLAP_GIOU_GRAD: [f32; 4] = [
        3.1249998438e-02_f32,
        3.1249998438e-02_f32,
        -6.2499999609e-02_f32,
        -6.2499999609e-02_f32,
    ];
    pub const NON_OVERLAP_DIOU_LOSS: f32 = 1.5624999982e+00;
    pub const NON_OVERLAP_DIOU_GRAD: [f32; 4] = [
        4.6874999414e-02_f32,
        4.6874999414e-02_f32,
        -9.3749999707e-02_f32,
        -9.3749999707e-02_f32,
    ];
    pub const NON_OVERLAP_CIOU_LOSS: f32 = 1.5624999982e+00;
    pub const NON_OVERLAP_CIOU_GRAD: [f32; 4] = [
        4.6874999414e-02_f32,
        4.6874999414e-02_f32,
        -9.3749999707e-02_f32,
        -9.3749999707e-02_f32,
    ];
}

// ============================================================================
// 测试 helper
// ============================================================================

/// 拼接式 bbox_loss 默认对照 epsilon。
///
/// only_torch 全程 f32，PyTorch oracle 用 f64 算后转 f32，主要误差来源：
/// 中间结果累计 + `IOU_NUMERICAL_EPS = 1e-7` 与 PyTorch 默认 `eps = 1e-7` 同。
/// 1e-5 既能可靠区分实现 bug 又给 f32 量化保留缓冲。
const EPSILON: f32 = 1e-5;

fn run_bbox_loss(
    pred: &[f32],
    target: &[f32],
    n: usize,
    kind: BBoxLossKind,
    fmt: BoxFormat,
) -> Result<(f32, Vec<f32>), GraphError> {
    let graph = Graph::new();
    let input = graph.parameter(&[n, 4], Init::Zeros, "input")?;
    input.set_value(&Tensor::new(pred, &[n, 4]))?;
    let target_var = graph.parameter(&[n, 4], Init::Zeros, "target")?;
    target_var.set_value(&Tensor::new(target, &[n, 4]))?;

    let loss = input.bbox_loss(&target_var, kind, fmt)?;
    graph.zero_grad()?;
    let loss_value = loss.backward()?;
    let grad = input.grad()?.expect("input 应有 grad");
    let grad_flat: Vec<f32> = grad.data_as_slice().to_vec();
    Ok((loss_value, grad_flat))
}

fn assert_grad_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: grad 长度不一致 actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_abs_diff_eq!(*a, *e, epsilon = EPSILON);
        let _ = i; // 让 panic message 出现在 assert_abs_diff_eq 自带的提示里就够了
    }
}

// ============================================================================
// Forward known-value：手算可核对的几何值
// ============================================================================

/// `partial_overlap`：pred=[0,0,2,2]、target=[1,0,3,2]
/// inter=2, union=4+4-2=6, IoU=2/6=1/3, loss=1-1/3=2/3
#[test]
fn test_iou_loss_partial_overlap_forward() {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 2.0, 2.0], &[1, 4]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 0.0, 3.0, 2.0], &[1, 4]))
        .unwrap();
    let loss = input
        .bbox_loss(&target, BBoxLossKind::IoU, BoxFormat::XyXy)
        .unwrap();
    loss.forward().unwrap();
    assert_abs_diff_eq!(loss.item().unwrap(), 2.0 / 3.0, epsilon = EPSILON);
}

/// `non_overlap`：pred=[0,0,1,1]、target=[2,0,3,1]
/// inter=0, union=2, IoU=0, GIoU=IoU - (A_enc-union)/A_enc = 0 - (3-2)/3 = -1/3
/// loss=1-GIoU=4/3
#[test]
fn test_giou_loss_non_overlap_forward() {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0, 1.0], &[1, 4]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[2.0, 0.0, 3.0, 1.0], &[1, 4]))
        .unwrap();
    let loss = input.giou_loss(&target, BoxFormat::XyXy).unwrap();
    loss.forward().unwrap();
    assert_abs_diff_eq!(loss.item().unwrap(), 4.0 / 3.0, epsilon = EPSILON);
}

/// CIoU/DIoU 在 `pred == target` 上 loss 严格 = 0
#[test]
fn test_ciou_loss_perfect_match_is_zero() {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(&[0.5, 0.5, 1.0, 1.0], &[1, 4]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.5, 0.5, 1.0, 1.0], &[1, 4]))
        .unwrap();
    let loss = input.ciou_loss(&target, BoxFormat::CxCyWh).unwrap();
    loss.forward().unwrap();
    assert_abs_diff_eq!(loss.item().unwrap(), 0.0, epsilon = EPSILON);
}

// ============================================================================
// Backward 对照 PyTorch oracle（reduction=mean，input 梯度逐元素 1e-5 一致）
// ============================================================================

#[test]
fn test_partial_overlap_backward_matches_pytorch_oracle() -> Result<(), GraphError> {
    let pred = [0.0, 0.0, 2.0, 2.0];
    let target = [1.0, 0.0, 3.0, 2.0];
    for (kind, expected_loss, expected_grad) in [
        (
            BBoxLossKind::IoU,
            oracle::PARTIAL_OVERLAP_IOU_ONE_THIRD_IOU_LOSS,
            &oracle::PARTIAL_OVERLAP_IOU_ONE_THIRD_IOU_GRAD[..],
        ),
        (
            BBoxLossKind::GIoU,
            oracle::PARTIAL_OVERLAP_IOU_ONE_THIRD_GIOU_LOSS,
            &oracle::PARTIAL_OVERLAP_IOU_ONE_THIRD_GIOU_GRAD[..],
        ),
        (
            BBoxLossKind::DIoU,
            oracle::PARTIAL_OVERLAP_IOU_ONE_THIRD_DIOU_LOSS,
            &oracle::PARTIAL_OVERLAP_IOU_ONE_THIRD_DIOU_GRAD[..],
        ),
        (
            BBoxLossKind::CIoU,
            oracle::PARTIAL_OVERLAP_IOU_ONE_THIRD_CIOU_LOSS,
            &oracle::PARTIAL_OVERLAP_IOU_ONE_THIRD_CIOU_GRAD[..],
        ),
    ] {
        let (loss, grad) = run_bbox_loss(&pred, &target, 1, kind, BoxFormat::XyXy)?;
        assert_abs_diff_eq!(loss, expected_loss, epsilon = EPSILON);
        assert_grad_close(&grad, expected_grad, &format!("partial_overlap/{kind:?}"));
    }
    Ok(())
}

#[test]
fn test_diagonal_overlap_backward_matches_pytorch_oracle() -> Result<(), GraphError> {
    let pred = [0.0, 0.0, 4.0, 4.0];
    let target = [2.0, 2.0, 6.0, 6.0];
    for (kind, expected_loss, expected_grad) in [
        (
            BBoxLossKind::IoU,
            oracle::DIAGONAL_OVERLAP_IOU_LOSS,
            &oracle::DIAGONAL_OVERLAP_IOU_GRAD[..],
        ),
        (
            BBoxLossKind::GIoU,
            oracle::DIAGONAL_OVERLAP_GIOU_LOSS,
            &oracle::DIAGONAL_OVERLAP_GIOU_GRAD[..],
        ),
        (
            BBoxLossKind::DIoU,
            oracle::DIAGONAL_OVERLAP_DIOU_LOSS,
            &oracle::DIAGONAL_OVERLAP_DIOU_GRAD[..],
        ),
        (
            BBoxLossKind::CIoU,
            oracle::DIAGONAL_OVERLAP_CIOU_LOSS,
            &oracle::DIAGONAL_OVERLAP_CIOU_GRAD[..],
        ),
    ] {
        let (loss, grad) = run_bbox_loss(&pred, &target, 1, kind, BoxFormat::XyXy)?;
        assert_abs_diff_eq!(loss, expected_loss, epsilon = EPSILON);
        assert_grad_close(&grad, expected_grad, &format!("diagonal_overlap/{kind:?}"));
    }
    Ok(())
}

#[test]
fn test_pred_contains_target_backward_matches_pytorch_oracle() -> Result<(), GraphError> {
    let pred = [0.0, 0.0, 4.0, 4.0];
    let target = [1.0, 1.0, 3.0, 3.0];
    for (kind, expected_loss, expected_grad) in [
        (
            BBoxLossKind::IoU,
            oracle::PRED_CONTAINS_TARGET_IOU_LOSS,
            &oracle::PRED_CONTAINS_TARGET_IOU_GRAD[..],
        ),
        (
            BBoxLossKind::GIoU,
            oracle::PRED_CONTAINS_TARGET_GIOU_LOSS,
            &oracle::PRED_CONTAINS_TARGET_GIOU_GRAD[..],
        ),
        (
            BBoxLossKind::DIoU,
            oracle::PRED_CONTAINS_TARGET_DIOU_LOSS,
            &oracle::PRED_CONTAINS_TARGET_DIOU_GRAD[..],
        ),
        (
            BBoxLossKind::CIoU,
            oracle::PRED_CONTAINS_TARGET_CIOU_LOSS,
            &oracle::PRED_CONTAINS_TARGET_CIOU_GRAD[..],
        ),
    ] {
        let (loss, grad) = run_bbox_loss(&pred, &target, 1, kind, BoxFormat::XyXy)?;
        assert_abs_diff_eq!(loss, expected_loss, epsilon = EPSILON);
        assert_grad_close(
            &grad,
            expected_grad,
            &format!("pred_contains_target/{kind:?}"),
        );
    }
    Ok(())
}

#[test]
fn test_aspect_ratio_mismatch_backward_matches_pytorch_oracle() -> Result<(), GraphError> {
    let pred = [0.0, 0.0, 2.0, 4.0];
    let target = [1.0, 1.0, 3.0, 3.0];
    for (kind, expected_loss, expected_grad) in [
        (
            BBoxLossKind::IoU,
            oracle::ASPECT_RATIO_MISMATCH_IOU_LOSS,
            &oracle::ASPECT_RATIO_MISMATCH_IOU_GRAD[..],
        ),
        (
            BBoxLossKind::GIoU,
            oracle::ASPECT_RATIO_MISMATCH_GIOU_LOSS,
            &oracle::ASPECT_RATIO_MISMATCH_GIOU_GRAD[..],
        ),
        (
            BBoxLossKind::DIoU,
            oracle::ASPECT_RATIO_MISMATCH_DIOU_LOSS,
            &oracle::ASPECT_RATIO_MISMATCH_DIOU_GRAD[..],
        ),
        (
            BBoxLossKind::CIoU,
            oracle::ASPECT_RATIO_MISMATCH_CIOU_LOSS,
            &oracle::ASPECT_RATIO_MISMATCH_CIOU_GRAD[..],
        ),
    ] {
        let (loss, grad) = run_bbox_loss(&pred, &target, 1, kind, BoxFormat::XyXy)?;
        assert_abs_diff_eq!(loss, expected_loss, epsilon = EPSILON);
        assert_grad_close(
            &grad,
            expected_grad,
            &format!("aspect_ratio_mismatch/{kind:?}"),
        );
    }
    Ok(())
}

#[test]
fn test_cxcywh_x_offset_backward_matches_pytorch_oracle() -> Result<(), GraphError> {
    let pred = [1.0, 1.0, 2.0, 2.0];
    let target = [2.0, 1.0, 2.0, 2.0];
    for (kind, expected_loss, expected_grad) in [
        (
            BBoxLossKind::IoU,
            oracle::CXCYWH_X_OFFSET_IOU_LOSS,
            &oracle::CXCYWH_X_OFFSET_IOU_GRAD[..],
        ),
        (
            BBoxLossKind::GIoU,
            oracle::CXCYWH_X_OFFSET_GIOU_LOSS,
            &oracle::CXCYWH_X_OFFSET_GIOU_GRAD[..],
        ),
        (
            BBoxLossKind::DIoU,
            oracle::CXCYWH_X_OFFSET_DIOU_LOSS,
            &oracle::CXCYWH_X_OFFSET_DIOU_GRAD[..],
        ),
        (
            BBoxLossKind::CIoU,
            oracle::CXCYWH_X_OFFSET_CIOU_LOSS,
            &oracle::CXCYWH_X_OFFSET_CIOU_GRAD[..],
        ),
    ] {
        let (loss, grad) = run_bbox_loss(&pred, &target, 1, kind, BoxFormat::CxCyWh)?;
        assert_abs_diff_eq!(loss, expected_loss, epsilon = EPSILON);
        assert_grad_close(&grad, expected_grad, &format!("cxcywh_x_offset/{kind:?}"));
    }
    Ok(())
}

#[test]
fn test_cxcywh_aspect_change_backward_matches_pytorch_oracle() -> Result<(), GraphError> {
    let pred = [1.0, 1.0, 2.0, 2.0];
    let target = [1.0, 1.0, 4.0, 1.0];
    for (kind, expected_loss, expected_grad) in [
        (
            BBoxLossKind::IoU,
            oracle::CXCYWH_ASPECT_CHANGE_IOU_LOSS,
            &oracle::CXCYWH_ASPECT_CHANGE_IOU_GRAD[..],
        ),
        (
            BBoxLossKind::GIoU,
            oracle::CXCYWH_ASPECT_CHANGE_GIOU_LOSS,
            &oracle::CXCYWH_ASPECT_CHANGE_GIOU_GRAD[..],
        ),
        (
            BBoxLossKind::DIoU,
            oracle::CXCYWH_ASPECT_CHANGE_DIOU_LOSS,
            &oracle::CXCYWH_ASPECT_CHANGE_DIOU_GRAD[..],
        ),
        (
            BBoxLossKind::CIoU,
            oracle::CXCYWH_ASPECT_CHANGE_CIOU_LOSS,
            &oracle::CXCYWH_ASPECT_CHANGE_CIOU_GRAD[..],
        ),
    ] {
        let (loss, grad) = run_bbox_loss(&pred, &target, 1, kind, BoxFormat::CxCyWh)?;
        assert_abs_diff_eq!(loss, expected_loss, epsilon = EPSILON);
        assert_grad_close(
            &grad,
            expected_grad,
            &format!("cxcywh_aspect_change/{kind:?}"),
        );
    }
    Ok(())
}

#[test]
fn test_batch_two_boxes_backward_matches_pytorch_oracle() -> Result<(), GraphError> {
    let pred = [0.0, 0.0, 2.0, 2.0, -2.0, -2.0, 0.0, 0.0];
    let target = [1.0, 0.0, 3.0, 2.0, -1.0, -1.0, 1.0, 1.0];
    for (kind, expected_loss, expected_grad) in [
        (
            BBoxLossKind::IoU,
            oracle::BATCH_TWO_BOXES_IOU_LOSS,
            &oracle::BATCH_TWO_BOXES_IOU_GRAD[..],
        ),
        (
            BBoxLossKind::GIoU,
            oracle::BATCH_TWO_BOXES_GIOU_LOSS,
            &oracle::BATCH_TWO_BOXES_GIOU_GRAD[..],
        ),
        (
            BBoxLossKind::DIoU,
            oracle::BATCH_TWO_BOXES_DIOU_LOSS,
            &oracle::BATCH_TWO_BOXES_DIOU_GRAD[..],
        ),
        (
            BBoxLossKind::CIoU,
            oracle::BATCH_TWO_BOXES_CIOU_LOSS,
            &oracle::BATCH_TWO_BOXES_CIOU_GRAD[..],
        ),
    ] {
        let (loss, grad) = run_bbox_loss(&pred, &target, 2, kind, BoxFormat::XyXy)?;
        assert_abs_diff_eq!(loss, expected_loss, epsilon = EPSILON);
        assert_grad_close(&grad, expected_grad, &format!("batch_two_boxes/{kind:?}"));
    }
    Ok(())
}

#[test]
fn test_non_overlap_backward_matches_pytorch_oracle() -> Result<(), GraphError> {
    let pred = [0.0, 0.0, 1.0, 1.0];
    let target = [3.0, 3.0, 4.0, 4.0];
    for (kind, expected_loss, expected_grad) in [
        (
            BBoxLossKind::IoU,
            oracle::NON_OVERLAP_IOU_LOSS,
            &oracle::NON_OVERLAP_IOU_GRAD[..],
        ),
        (
            BBoxLossKind::GIoU,
            oracle::NON_OVERLAP_GIOU_LOSS,
            &oracle::NON_OVERLAP_GIOU_GRAD[..],
        ),
        (
            BBoxLossKind::DIoU,
            oracle::NON_OVERLAP_DIOU_LOSS,
            &oracle::NON_OVERLAP_DIOU_GRAD[..],
        ),
        (
            BBoxLossKind::CIoU,
            oracle::NON_OVERLAP_CIOU_LOSS,
            &oracle::NON_OVERLAP_CIOU_GRAD[..],
        ),
    ] {
        let (loss, grad) = run_bbox_loss(&pred, &target, 1, kind, BoxFormat::XyXy)?;
        assert_abs_diff_eq!(loss, expected_loss, epsilon = EPSILON);
        assert_grad_close(&grad, expected_grad, &format!("non_overlap/{kind:?}"));
    }
    Ok(())
}

// ============================================================================
// Shape 校验：拼接式入口必须显式拒绝 [N, 4] vs [1, 4] 的 broadcast
// ============================================================================

#[test]
fn test_bbox_loss_rejects_non_bbox_last_dim() {
    let graph = Graph::new();
    // input 自身就不是 [N, 4]：shape [1, 5] 直接撞 input 形状校验分支
    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0, 1.0, 0.9], &[1, 5]))
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
fn test_bbox_loss_rejects_non_2d_input() {
    let graph = Graph::new();
    // 3D 输入：通过 graph.input 的 2-4 维基础校验，再撞 helper 的 [N, 4] check
    let input = graph.input(&Tensor::new(&[0.0; 24], &[2, 3, 4])).unwrap();
    let target = graph.input(&Tensor::new(&[0.0; 24], &[2, 3, 4])).unwrap();

    let err = input
        .bbox_loss(&target, BBoxLossKind::IoU, BoxFormat::XyXy)
        .unwrap_err();
    assert!(
        format!("{err}").contains("[N, 4]"),
        "错误信息应说明 bbox shape，实际: {err}"
    );
}

#[test]
fn test_bbox_loss_rejects_implicit_broadcast() {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(
            &[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0],
            &[3, 4],
        ))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.5, 0.5, 1.5, 1.5], &[1, 4]))
        .unwrap();

    let err = input
        .bbox_loss(&target, BBoxLossKind::GIoU, BoxFormat::XyXy)
        .unwrap_err();
    assert!(
        format!("{err}").contains("严格相等"),
        "错误信息应明确拒绝隐式 broadcast，实际: {err}"
    );
}

// ============================================================================
// target detach 语义：拼接式 helper 内部对 target 显式 detach，
// 即便 target 是 Parameter，也不会反向收梯度
// ============================================================================

#[test]
fn test_target_parameter_receives_no_gradient() -> Result<(), GraphError> {
    let graph = Graph::new();
    let input = graph.parameter(&[1, 4], Init::Zeros, "input")?;
    input.set_value(&Tensor::new(&[0.0, 0.0, 2.0, 2.0], &[1, 4]))?;
    let target = graph.parameter(&[1, 4], Init::Zeros, "target")?;
    target.set_value(&Tensor::new(&[1.0, 0.0, 3.0, 2.0], &[1, 4]))?;

    let loss = input.bbox_loss(&target, BBoxLossKind::IoU, BoxFormat::XyXy)?;
    graph.zero_grad()?;
    loss.backward()?;

    let input_grad = input.grad()?.expect("input 应有 grad");
    assert!(
        !input_grad.data_as_slice().iter().all(|x| *x == 0.0),
        "input 梯度不应全 0（部分重叠 case）"
    );

    assert!(
        target.grad()?.is_none(),
        "target 是 Parameter 但被 helper 内部 detach，应当没有梯度"
    );
    Ok(())
}

// ============================================================================
// CIoU 退化短路：宽 / 高 ≤ 0 时 valid mask 归零，CIoU 等价 DIoU
// ============================================================================

#[test]
fn test_ciou_degenerate_box_falls_back_to_diou() -> Result<(), GraphError> {
    // pred 宽=2-2=0，满足 BBox::ciou 的 w/h <= 0 退化条件；target 正常
    let pred = [2.0, 0.0, 2.0, 4.0];
    let target = [1.0, 0.0, 3.0, 4.0];

    let (ciou_loss, _) = run_bbox_loss(&pred, &target, 1, BBoxLossKind::CIoU, BoxFormat::XyXy)?;
    let (diou_loss, _) = run_bbox_loss(&pred, &target, 1, BBoxLossKind::DIoU, BoxFormat::XyXy)?;

    assert_abs_diff_eq!(ciou_loss, diou_loss, epsilon = EPSILON);
    assert!(
        ciou_loss.is_finite(),
        "退化框 CIoU 必须保持 finite，实际: {ciou_loss}"
    );
    Ok(())
}

#[test]
fn test_ciou_tiny_positive_box_keeps_aspect_penalty() -> Result<(), GraphError> {
    // 标量 BBox::ciou 的退化条件是 w/h <= 0。极小但正的宽高仍应走 CIoU
    // aspect-ratio penalty，而不是被数值阈值误判成 DIoU。
    let pred = [0.0, 0.0, 1e-8, 1.0];
    let target = [0.0, 0.0, 1.0, 1.0];

    let (ciou_loss, _) = run_bbox_loss(&pred, &target, 1, BBoxLossKind::CIoU, BoxFormat::XyXy)?;
    let (diou_loss, _) = run_bbox_loss(&pred, &target, 1, BBoxLossKind::DIoU, BoxFormat::XyXy)?;

    assert!(
        ciou_loss > diou_loss + 1e-3,
        "正宽高不应触发退化短路：ciou_loss={ciou_loss}, diou_loss={diou_loss}"
    );
    Ok(())
}

// ============================================================================
// 可视化分组：4 套 IoU 都把内部 30+ 节点折叠成单个 NodeGroupTag cluster
// ============================================================================

#[test]
fn test_iou_loss_node_group_tag_collapses_into_cluster() {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 2.0, 2.0], &[1, 4]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 0.0, 3.0, 2.0], &[1, 4]))
        .unwrap();

    for (kind, expected_group) in [
        (BBoxLossKind::IoU, "IoULoss"),
        (BBoxLossKind::GIoU, "GIoULoss"),
        (BBoxLossKind::DIoU, "DIoULoss"),
        (BBoxLossKind::CIoU, "CIoULoss"),
    ] {
        let loss = input.bbox_loss(&target, kind, BoxFormat::XyXy).unwrap();
        let tag = loss
            .node_group_tag()
            .unwrap_or_else(|| panic!("{kind:?} 入口应被 NodeGroupContext 自动打 tag"));
        assert_eq!(
            tag.group_type, expected_group,
            "{kind:?} 应折叠到 {expected_group} cluster，实际 {}",
            tag.group_type
        );

        let mut checked_internal_nodes = 0;
        for node in loss.node().backward_topo_order() {
            // 原始 input/target 和标量常量是叶子节点，不应被 NodeGroupContext 污染；
            // helper 内部真正新建的计算节点都应该落入同一个可视化 cluster。
            if node.parents().is_empty() {
                continue;
            }

            let node_tag = node
                .node_group_tag()
                .unwrap_or_else(|| panic!("{kind:?} 内部节点 {} 缺少 NodeGroupTag", node.id()));
            assert_eq!(
                node_tag.group_type,
                expected_group,
                "{kind:?} 内部节点 {} 应属于 {expected_group} cluster，实际 {}",
                node.id(),
                node_tag.group_type
            );
            assert_eq!(
                node_tag.instance_id,
                tag.instance_id,
                "{kind:?} 内部节点 {} 应与输出节点属于同一 cluster 实例",
                node.id()
            );
            checked_internal_nodes += 1;
        }
        assert!(
            checked_internal_nodes > 0,
            "{kind:?} 应至少包含一个被分组的内部计算节点"
        );
    }
    // 用户传入的 input/target 不应被打 tag（NodeGroupContext 不标记 Input 节点）
    assert!(input.node_group_tag().is_none());
    assert!(target.node_group_tag().is_none());
}
