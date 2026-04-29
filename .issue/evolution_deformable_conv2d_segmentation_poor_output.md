---
status: suspended
created: 2026-04-29
updated: 2026-04-29
---

# Evolution DeformableConv2d 分割输出质量很差

## 背景

`examples/evolution/deformable_conv2d_segmentation` 用于验证 DeformableConv2d 进入演化主流程，并在 16x16 合成语义分割任务上输出前景概率图。

用户口头提到的 `deformable_detr_2d_segmentation` 目前对应仓库里的 `evolution_deformable_conv2d_segmentation` 示例路径。

## 现象 / 影响

当前演化结果虽然命令能跑通，但预测图质量很差，和目标 mask 差异明显。直观证据见：

- `examples/evolution/deformable_conv2d_segmentation/test_out.png`
- `examples/evolution/deformable_conv2d_segmentation/test_target.png`
- `examples/evolution/deformable_conv2d_segmentation/test_in.png`

这说明该示例目前只能作为流程 smoke test，不能作为分割质量达标的示例或 benchmark。

## 已尝试

- 已运行 `just example-evolution-deformable-conv2d-segmentation`，命令成功完成。
- 输出指标中 `Binary IoU` 仅达到示例当前较低目标阈值附近，但可视化质量明显不足。

## 当前卡点

还没有系统区分是以下哪类问题导致质量差：

- 初始候选族或演化搜索预算过弱。
- DeformableConv2d offset-only 配置表达能力不足。
- 训练样本过少、目标阈值过低，导致示例过早通过。
- 指标与可视化观感不一致，当前评价方式没有充分惩罚糟糕 mask。

## 暂缓原因

当前主线是清理 evolution 诊断、修复 memory unit 可视化回归并完成提交；DeformableConv2d 分割质量需要单独排查，不适合混在本次修复里继续扩大改动范围。

## 下次恢复条件

当继续完善空间视觉任务或准备把 DeformableConv2d 演化示例作为正式 benchmark 时，恢复该问题。

## 下一步建议

- 固定 seed 后对比 traditional `deformable_conv2d_segmentation` 与 evolution 版本的输入、目标、指标和输出图。
- 提高该示例的训练样本数、评估样本数和 target Binary IoU，观察是否仍然退化。
- 检查候选族是否只验证“包含 DeformableConv2d”，而没有保证输出 head、offset 分支或后处理足够合理。
- 若指标误导明显，补充 Dice / Pixel Accuracy / per-sample IoU 的失败样本报告。

## 相关文件 / 命令 / 对话

- `examples/evolution/deformable_conv2d_segmentation/main.rs`
- `examples/evolution/deformable_conv2d_segmentation/test_out.png`
- `just example-evolution-deformable-conv2d-segmentation`
