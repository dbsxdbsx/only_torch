---
status: resolved
created: 2026-05-01
updated: 2026-05-01
resolved: 2026-05-01
---

# BBoxLoss 反向传播应替换为解析梯度

## 解决方案（2026-05-01 闭环）

按 [`bbox_loss_autograd_migration`](../../.cursor/plans/bbox_loss_autograd_migration_301fff0e.plan.md) 规划落地，**完全废弃** fused `BBoxLoss` 节点 + 有限差分梯度，改为基础算子拼接式实现：

- 新增 `Atan2` 可微节点（CIoU 角度差所需），`(0, 0)` 处梯度 fallback 为 0（与 PyTorch 的 `NaN` 不一致，避免污染训练）。
- 新增 `src/vision/detection/iou_loss.rs`：用 `Maximum / Minimum / ReLU / Square / Atan2 / Sign / Mean` 等基础算子拼接 IoU / GIoU / DIoU / CIoU 四套损失，由 autograd 自动反向；CIoU 零面积退化复用 ReLU 的非负保证，用 `Sign` 直接拿到 `w > 0` 的运行时硬 mask 实现等价短路（不引入 epsilon 阈值，行为对齐 `BBox::ciou` 的 `<= 0` 严格判断）。
- 入口 `Var::bbox_loss / giou_loss / diou_loss / ciou_loss` 直接 delegate 到拼接式 helper；用 `NodeGroupContext` 把 30+ 内部节点折叠成单个 cluster，可视化层面与原 fused 节点等价。
- `BBoxLossKind` 从 `nn::nodes::raw_node::loss` 移到 `vision::detection`，与其语义所属模块对齐。
- 删除 `src/nn/nodes/raw_node/loss/bbox.rs`、`create_bbox_loss_node`、`NodeTypeDescriptor::BBoxLoss` variant、descriptor rebuild 分支、ONNX TrainingOnly 分类、evolution 注册、var/descriptor 分支共 11 处引用。
- 测试体系：`tests/bbox_loss_reference.py` 用 PyTorch autograd 在 8 组典型样本上算 4 套 IoU 的 forward + backward oracle；新建 `src/vision/tests/bbox_loss_composed.rs`（18 个 test）按 `epsilon = 1e-5` 逐元素对照（取代旧 `2e-3` 的有限差分容忍）。

## 原始内容（保留作为历史参考）

`src/nn/nodes/raw_node/loss/bbox.rs` 实现了 IoU / GIoU / DIoU / CIoU 四种 bbox
loss 节点（`BBoxLoss`），并通过 `Var::bbox_loss / giou_loss / diou_loss /
ciou_loss` 暴露给用户。

旧节点的反向传播 `BBoxLoss::finite_diff_grad` 用的是**有限差分近似**：

```rust
// src/nn/nodes/raw_node/loss/bbox.rs（已删除）
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

### 问题

- **性能**：每个 input 元素都要做 2 次 forward (plus + minus)。`[N, 4]` 输入下，反向一次就需要 `8N` 次 IoU/GIoU/DIoU/CIoU 计算，相比解析梯度的常数级开销慢一个数量级以上。
- **数值精度**：$epsilon = 1e-3$ 在 IoU/GIoU 接近 0 或 1 的边界附近会出现明显截断误差；旧 `tests/node_bbox_loss.rs` 的 VJP 断言只能容忍到 `2e-3`。
- **API 风险**：`vision::detection::DetectionLossComponents` 已经把 bbox_loss 作为 detection 多任务 loss 组合的一员暴露出去，第一个原生 detection example 接入时会卡在训练性能上。

### 当时暂缓的理由

- 当时 `vision/detection/` 还没有原生 detector example，`DetectionLossComponents` 也尚无下游消费者。
- 缺少 `Atan2` 这种 CIoU 必需的基础算子，IoU/GIoU/DIoU/CIoU 的解析梯度推导成本较高。

实际闭环时通过"拼接式 + autograd 自动反向"绕开了手推解析梯度的工程成本，直接拿到 PyTorch 同级的精度（1e-5 以内）和 O(N) 反向 + 后续 fused 优化潜力。
