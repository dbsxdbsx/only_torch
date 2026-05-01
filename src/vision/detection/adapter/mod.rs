//! 第三方 / 外部预训练检测器的输入输出适配层。
//!
//! 这里收纳"把外部检测器的特殊格式转换成 only_torch 通用类型"的代码。
//! 各 module 与具体模型族一一对应，互相独立、互不耦合：
//!
//! | 子模块 | 适配对象 | 范围 |
//! |---|---|---|
//! | [`yolo`] | YOLO 系列（v3 / v5 / v8 ...） | head 输出解码（推理时） |
//!
//! 设计原则：
//!
//! - **只收"推理时输出解码"这层稳定 API**：因为 ONNX export 之后格式锁死，
//!   实现一次后基本不动。训练时的 anchor matching / SimOTA / Hungarian
//!   matching 等差异最大、版本演化最频繁，留在 example 或下游 crate。
//! - **每个版本独立一个文件**：YOLOv5 / YOLOv8 head shape 完全不同，强行
//!   抽象会两边都不舒服。需要时再升级到 trait（参考
//!   [`crate::vision::detection::contract`]）。
//! - **不依赖具体 ONNX runtime**：输入是 `Tensor`（来自 `Graph::predict`），
//!   输出是 `Vec<Detection>`，跟图加载方式解耦。

pub mod yolo;
