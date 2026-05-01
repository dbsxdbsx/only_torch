//! 传统 CV 算法（OpenCV 风格）。
//!
//! 这里收纳 only_torch 提供的、不依赖深度神经网络的"传统 CV 算法"——
//! 例如 Hough 变换。它们的设计目标只是**让 toy 示例和教学场景跑得起来**，
//! 不追求与 OpenCV 的性能对齐；真实生产请直接使用 OpenCV 等成熟实现。
//!
//! PyTorch / torchvision / JAX 等基础 ML 框架均不维护这一类算法，本目录只是
//! only_torch 的便利集合，与"现代检测 / 分割积木"（在 `vision::detection`
//! 等子模块）严格分开。

pub mod hough_circles;

pub use hough_circles::{HoughCircle, detect_circles_hough};
