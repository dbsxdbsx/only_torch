//! YOLO 系列的推理时 adapter。
//!
//! 不同 YOLO 版本 head 输出格式差异很大，**每个版本一个独立 module**，互不耦合：
//!
//! | 版本 | head 输出 | 关键差异 |
//! |---|---|---|
//! | [`v5`] | `[1, num_anchors, 5+nc]`（已合并三尺度，已 sigmoid） | 含 `obj_conf`，简单 cxcywh 重排 |
//!
//! 计划但**尚未实现**的版本（按需添加）：
//!
//! - YOLOv3/v4：`[N, 3, 5+nc, H, W]` 多尺度 anchor head
//! - YOLOv8/v11：`[N, 4+nc, 8400]`，无 obj_conf，DFL 解码
//! - YOLOX：anchor-free + decoupled head
//!
//! 训练时 assigner（grid / SimOTA / TaskAlignedAssigner 等）按需驱动添加，
//! 落到 [`crate::vision::detection::contract::Assigner`] trait。

pub mod v5;
