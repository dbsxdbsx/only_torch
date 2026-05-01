//! 计算机视觉模块。
//!
//! 按职能划分子模块；新代码请使用模块函数而非类型方法（不再有 `Vision`
//! 命名空间），并优先接收 / 返回 `image::DynamicImage` 等强类型。
//!
//! # 子模块
//!
//! | 子模块 | 用途 |
//! |---|---|
//! | [`color`] | 色彩空间转换（如 RGB → Luma） |
//! | [`cv`] | 传统 CV 算法（Hough 等；与 PyTorch / torchvision 不收录的范畴对齐） |
//! | [`detection`] | 目标检测专属积木：`BBox` / NMS / mAP-friendly 类型 / Backbone 契约 / loss 组合 |
//! | [`draw`] | 在图像上绘制 bbox / 矩形 / 圆等可视化 |
//! | [`filter`] | 图像滤波（中值等） |
//! | [`geom`] | 通用图像几何（resize / center_crop 等） |
//! | [`io`] | 图像加载与保存 |
//! | [`preprocess`] | 高层组合（letterbox / image_to_nchw_normalized 等） |

pub mod color;
pub mod cv;
pub mod detection;
pub mod draw;
pub mod filter;
pub mod geom;
pub mod io;
pub mod preprocess;

#[cfg(test)]
mod tests;
