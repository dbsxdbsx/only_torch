---
status: suspended
created: 2026-05-01
updated: 2026-05-01
---

# BBoxLoss 反向传播应替换为解析梯度

## 背景

`src/nn/nodes/raw_node/loss/bbox.rs` 实现了 IoU / GIoU / DIoU / CIoU 四种 bbox
loss 节点（`BBoxLoss`），并通过 `Var::bbox_loss / giou_loss / diou_loss /
ciou_loss` 暴露给用户。

当前节点实现的反向传播 `BBoxLoss::finite_diff_grad` 用的是**有限差分近似**：

```rust
// src/nn/nodes/raw_node/loss/bbox.rs
const FINITE_DIFF_EPSILON: f32 = 1e-3;

fn finite_diff_grad(...) -> Tensor {
    for idx in 0..input.size() {
        // 对每个 input 元素都做一次 plus / minus 前向，求中心差
        let plus_loss = Self::loss_sum(&plus, target, kind, format);
        let minus_loss = Self::loss_sum(&minus, target, kind, format);
        grad_data[idx] = (plus_loss - minus_loss) / (2.0 * FINITE_DIFF_EPSILON)
            * reduction_scale * upstream_scale;
    }
}
```

节点 doc-comment 自己也明确写了：

> 反向传播当前使用有限差分近似梯度，适合小规模验证和 adapter 原型；若进入真实
> detection fine-tune，需要优先替换为解析梯度以降低训练成本并提升数值稳定性。

## 现象 / 影响

- **性能**：每个 input 元素都要做 2 次 forward (plus + minus)。`[N, 4]` 输入
  下，反向一次就需要 `8N` 次 IoU/GIoU/DIoU/CIoU 计算，相比解析梯度的常数级开销
  慢一个数量级以上，detection fine-tune 训练循环里会成为瓶颈。
- **数值精度**：`epsilon = 1e-3` 在 IoU/GIoU 接近 0 或 1 的边界附近会出现明显
  截断误差；`tests/node_bbox_loss.rs` 里的 VJP 断言也只能容忍到 `2e-3`。
- **API 风险**：`vision::detection::DetectionLossComponents` 已经把 bbox_loss
  作为 detection 多任务 loss 组合的一员暴露出去。第一个原生 detection example
  接入时，如果还是有限差分梯度，就会卡在训练性能上。

## 已尝试

尚未做系统替换；当前只是把"有限差分"作为节点上线时的占位实现，等真实 detection
训练需求出现时再做。

## 当前卡点

- IoU 解析梯度公式相对直接（按 `intersection / union` 求 partial），但
  GIoU / DIoU / CIoU 涉及 enclosing box 的对角线、中心距、宽高比 atan2 项，
  公式较多，需要逐项推导并对齐 PyTorch / mmdetection / Ultralytics 的参考
  实现。
- 现有 `loss_sum` 内部用的是 `BBox` 结构体的封装方法（`iou / giou / diou /
  ciou`），如果保留这层封装，解析梯度也需要把中间量（intersection / union /
  enclosing area / center distance / atan diff 等）暴露出来；如果不保留，会
  让 `BBox` 几何工具与 `BBoxLoss` 节点的实现重复一遍坐标和面积计算。需要先
  决定是"BBox 增加梯度友好辅助方法"还是"BBoxLoss 自己直接算坐标"。

## 暂缓原因

- 当前 `vision/detection/` 还没有原生 detector example，`DetectionLossComponents`
  也尚无下游消费者（属于 `.doc/design/spatial_vision_tasks_roadmap.md` 中
  "契约比实现先行" 的状态）。
- 在没有真实训练循环消费它之前，先用有限差分换取实现简单 + 单元测试（已覆盖
  IoU 已知数值梯度），把精力集中在 `Backbone` / `DetectionHeadDecode` /
  `Assigner` 这些更前置的契约和原生 detector 上。

## 下次恢复条件

满足以下任一条件即可恢复处理：

- 着手实现第一个 only_torch 原生 detector（YOLO-lite 等）的 fine-tune 闭环。
- detection adapter（如现有 `chinese_chess_yolov5_onnx_finetune`）从纯推理
  扩展到需要在本框架内做反向传播的训练 / fine-tune。
- 性能 profiling 显示 `BBoxLoss::finite_diff_grad` 出现在热点。

## 下一步建议

- 先固定四类公式：把 IoU / GIoU / DIoU / CIoU 对 `(x1, y1, x2, y2)`（XyXy 输入）
  和 `(cx, cy, w, h)`（CxCyWh 输入）的解析偏导逐项写到设计文档（可挂在
  `.doc/design/spatial_vision_tasks_roadmap.md` 下或单独建 `bbox_loss_grad.md`），
  对齐 mmdetection / Ultralytics 的参考实现。
- 实现时优先考虑在 `BBox` 上扩 `intersection_components()` 之类的中间量获取
  方法，避免 `BBoxLoss` 节点重抄面积 / 中心距计算。
- 单元测试保留现有 `node_bbox_loss.rs` 的 known-value 断言（`-0.24 / -0.4`），
  并新增针对 GIoU / DIoU / CIoU 的解析梯度对照值（用 PyTorch 或 SymPy 算出
  reference）。
- 切换实现时，把 `FINITE_DIFF_EPSILON` 常量删除以避免误用。
