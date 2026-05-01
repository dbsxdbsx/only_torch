"""bbox loss PyTorch 参考实现 -- 生成 Rust 测试 oracle

为 only_torch 的拼接式 IoU/GIoU/DIoU/CIoU 损失生成对照基线：

- 用 PyTorch autograd 计算 4 套 IoU loss 在 8 组典型 (input, target) 样本上的
  forward 标量与 backward grad（仅 input，target 显式 detach）。
- 输出可直接粘贴到 ``src/vision/tests/bbox_loss_composed.rs`` 的 Rust 常量代码。
- **重要**：所有 backward 测试样本都精心避开了 ReLU / Maximum / Minimum / Atan2
  的非可微拐点（perfect match、零面积、刚好相切、(0, 0) 等），保证 only_torch
  与 PyTorch 在 epsilon ≤ 1e-5 内逐元素一致；零面积退化的 forward 短路语义
  在 Rust 端单独硬编码测试，不在本脚本生成。

执行方式（仓库根）::

    python tests/bbox_loss_reference.py > tests/bbox_loss_reference_output.txt

依赖：python>=3.10、torch（CPU 即可）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _box_areas(boxes: torch.Tensor) -> torch.Tensor:
    """boxes: [N, 4] xyxy → 面积 [N]，宽高用 ReLU 截非负（与 only_torch 一致）。"""
    w = torch.relu(boxes[:, 2] - boxes[:, 0])
    h = torch.relu(boxes[:, 3] - boxes[:, 1])
    return w * h


def _intersection(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.maximum(pred[:, 0], target[:, 0])
    inter_y1 = torch.maximum(pred[:, 1], target[:, 1])
    inter_x2 = torch.minimum(pred[:, 2], target[:, 2])
    inter_y2 = torch.minimum(pred[:, 3], target[:, 3])
    w = torch.relu(inter_x2 - inter_x1)
    h = torch.relu(inter_y2 - inter_y1)
    return w * h


def _enclosing(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (enc_w, enc_h)，宽高 ReLU 截非负。"""
    enc_x1 = torch.minimum(pred[:, 0], target[:, 0])
    enc_y1 = torch.minimum(pred[:, 1], target[:, 1])
    enc_x2 = torch.maximum(pred[:, 2], target[:, 2])
    enc_y2 = torch.maximum(pred[:, 3], target[:, 3])
    enc_w = torch.relu(enc_x2 - enc_x1)
    enc_h = torch.relu(enc_y2 - enc_y1)
    return enc_w, enc_h


def _iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    inter = _intersection(pred, target)
    union = _box_areas(pred) + _box_areas(target) - inter + eps
    return inter / union


def iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    iou = _iou(pred, target)
    return (1.0 - iou).mean()


def giou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    inter = _intersection(pred, target)
    a_pred = _box_areas(pred)
    a_target = _box_areas(target)
    union = a_pred + a_target - inter + eps
    iou = inter / union
    enc_w, enc_h = _enclosing(pred, target)
    enc_area = enc_w * enc_h + eps
    extra = (enc_area - union) / enc_area
    giou = iou - extra
    return (1.0 - giou).mean()


def diou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    iou = _iou(pred, target, eps)
    enc_w, enc_h = _enclosing(pred, target)

    cx_p = (pred[:, 0] + pred[:, 2]) * 0.5
    cy_p = (pred[:, 1] + pred[:, 3]) * 0.5
    cx_t = (target[:, 0] + target[:, 2]) * 0.5
    cy_t = (target[:, 1] + target[:, 3]) * 0.5
    center_dist_sq = (cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2
    diag_sq = enc_w ** 2 + enc_h ** 2 + eps
    diou = iou - center_dist_sq / diag_sq
    return (1.0 - diou).mean()


def ciou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """CIoU loss（不含零面积短路；只在非退化样本上调用）。

    only_torch 的 ``ciou`` helper 用 Step 节点构造 valid mask 实现退化短路；
    本 oracle 不复刻该短路：只要保证传入 pred / target 的所有 w / h 都
    严格 > 0，short-circuit 永不触发，ciou_full 与 only_torch 输出等价。
    """
    iou = _iou(pred, target, eps)
    enc_w, enc_h = _enclosing(pred, target)

    cx_p = (pred[:, 0] + pred[:, 2]) * 0.5
    cy_p = (pred[:, 1] + pred[:, 3]) * 0.5
    cx_t = (target[:, 0] + target[:, 2]) * 0.5
    cy_t = (target[:, 1] + target[:, 3]) * 0.5
    center_dist_sq = (cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2
    diag_sq = enc_w ** 2 + enc_h ** 2 + eps
    diou = iou - center_dist_sq / diag_sq

    p_w = torch.relu(pred[:, 2] - pred[:, 0])
    p_h = torch.relu(pred[:, 3] - pred[:, 1])
    t_w = torch.relu(target[:, 2] - target[:, 0])
    t_h = torch.relu(target[:, 3] - target[:, 1])

    angle_diff = torch.atan2(t_w, t_h) - torch.atan2(p_w, p_h)
    v = (4.0 / (math.pi ** 2)) * angle_diff ** 2
    alpha = v / (1.0 - iou + v + eps)
    ciou = diou - alpha * v
    return (1.0 - ciou).mean()


LOSS_FUNCTIONS = {
    "IoU": iou_loss,
    "GIoU": giou_loss,
    "DIoU": diou_loss,
    "CIoU": ciou_loss,
}


@dataclass
class Sample:
    name: str
    pred: list[list[float]]
    target: list[list[float]]
    fmt: str  # "XyXy" 或 "CxCyWh"
    note: str


def _cxcywh_to_xyxy_torch(cxcywh: torch.Tensor) -> torch.Tensor:
    """对 [N, 4] cxcywh tensor 用可微 PyTorch 算子转 xyxy。

    与 only_torch ``vision::detection::iou_loss::box_to_xyxy`` 完全一致：
    先用 ReLU 把 w / h 截到非负，再 ±half 推出 (x1, y1, x2, y2)。

    叶子参数保留为 cxcywh，确保 ``pred.grad`` 是关于 cxcywh 维度而非 xyxy。
    """
    cx = cxcywh[:, 0]
    cy = cxcywh[:, 1]
    raw_w = cxcywh[:, 2]
    raw_h = cxcywh[:, 3]
    w = torch.relu(raw_w)
    h = torch.relu(raw_h)
    half_w = w * 0.5
    half_h = h * 0.5
    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h
    return torch.stack([x1, y1, x2, y2], dim=1)


# ============================================================================
# 8 组 backward 测试样本 + forward-only 退化样本
#
# 选样原则：
#   1. 所有非退化样本的 pred/target 在所有几何位置都明显 ≠ 0：
#      - inter_x2 - inter_x1 > 0（避开 ReLU 边界）
#      - p_w / p_h / t_w / t_h > 0.05（避开 atan2(0, 0) 的项目 fallback）
#      - 1 - iou + v 远 > 0（避开 alpha 分母 zero）
#   2. 几何分布尽量覆盖：部分重叠、对角偏移、含住、不同 aspect ratio、
#      CxCyWh 格式、N=2 batch、负坐标（验证不依赖坐标系平移）
# ============================================================================


BACKWARD_SAMPLES: list[Sample] = [
    Sample(
        name="partial_overlap_iou_one_third",
        pred=[[0.0, 0.0, 2.0, 2.0]],
        target=[[1.0, 0.0, 3.0, 2.0]],
        fmt="XyXy",
        note="部分横向重叠，IoU = 2/6 = 1/3",
    ),
    Sample(
        name="diagonal_overlap",
        pred=[[0.0, 0.0, 4.0, 4.0]],
        target=[[2.0, 2.0, 6.0, 6.0]],
        fmt="XyXy",
        note="对角偏移重叠",
    ),
    Sample(
        name="pred_contains_target",
        pred=[[0.0, 0.0, 4.0, 4.0]],
        target=[[1.0, 1.0, 3.0, 3.0]],
        fmt="XyXy",
        note="pred 完整包住 target，验证 GIoU enclosing 极端 case",
    ),
    Sample(
        name="aspect_ratio_mismatch",
        pred=[[0.0, 0.0, 2.0, 4.0]],
        target=[[1.0, 1.0, 3.0, 3.0]],
        fmt="XyXy",
        note="不同 aspect ratio，CIoU angle 项显著",
    ),
    Sample(
        name="cxcywh_x_offset",
        pred=[[1.0, 1.0, 2.0, 2.0]],
        target=[[2.0, 1.0, 2.0, 2.0]],
        fmt="CxCyWh",
        note="CxCyWh 格式，仅 x 轴中心偏移",
    ),
    Sample(
        name="cxcywh_aspect_change",
        pred=[[1.0, 1.0, 2.0, 2.0]],
        target=[[1.0, 1.0, 4.0, 1.0]],
        fmt="CxCyWh",
        note="CxCyWh 同中心、不同 aspect ratio，CIoU angle 项显著",
    ),
    Sample(
        name="batch_two_boxes",
        pred=[[0.0, 0.0, 2.0, 2.0], [-2.0, -2.0, 0.0, 0.0]],
        target=[[1.0, 0.0, 3.0, 2.0], [-1.0, -1.0, 1.0, 1.0]],
        fmt="XyXy",
        note="N=2 batch，包含负坐标，验证 reduction = mean 与坐标平移不变性",
    ),
    Sample(
        name="non_overlap_giou_negative",
        pred=[[0.0, 0.0, 1.0, 1.0]],
        target=[[3.0, 3.0, 4.0, 4.0]],
        fmt="XyXy",
        note="完全不重叠（IoU = 0），GIoU/DIoU 显著惩罚 enclosing",
    ),
]


def _generate_backward_oracle(samples: list[Sample]) -> str:
    """对每个样本和每种 IoU loss，计算 PyTorch forward 标量 + input 梯度。"""
    lines: list[str] = []
    lines.append("// ============================================================")
    lines.append("// 由 tests/bbox_loss_reference.py 生成，请勿手改")
    lines.append("// ============================================================\n")

    for sample in samples:
        for kind, loss_fn in LOSS_FUNCTIONS.items():
            # 叶子参数保留 sample.fmt，only_torch 端 input 也是按 sample.fmt 喂进 bbox_loss，
            # 这样 pred.grad 与 only_torch input.grad 维度一致（XyXy 直接是 [x1,y1,x2,y2]，
            # CxCyWh 直接是 [cx,cy,w,h]）。
            pred = torch.tensor(sample.pred, dtype=torch.float64, requires_grad=True)
            target_raw = torch.tensor(sample.target, dtype=torch.float64, requires_grad=False).detach()

            if sample.fmt == "XyXy":
                pred_xyxy = pred
                target_xyxy = target_raw
            else:
                pred_xyxy = _cxcywh_to_xyxy_torch(pred)
                target_xyxy = _cxcywh_to_xyxy_torch(target_raw)

            loss = loss_fn(pred_xyxy, target_xyxy)
            loss.backward()
            grad = pred.grad

            const_name = f"{sample.name.upper()}_{kind.upper()}"
            n = len(sample.pred)
            lines.append(f"// {sample.note} [{kind}, {sample.fmt}, N={n}]")
            lines.append(f"pub const {const_name}_LOSS: f32 = {loss.item():.10e};")
            grad_flat = ", ".join(f"{x:.10e}_f32" for x in grad.flatten().tolist())
            lines.append(f"pub const {const_name}_GRAD: [f32; {n * 4}] = [{grad_flat}];")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    print(_generate_backward_oracle(BACKWARD_SAMPLES))


if __name__ == "__main__":
    main()
