//! # 评估指标模块
//!
//! 提供机器学习模型评估的常用指标函数。
//!
//! ## 模块结构
//!
//! - [`classification`] - 分类指标（Accuracy, Precision, Recall 等）
//! - [`regression`] - 回归指标（R², MSE, MAE 等）
//! - [`traits`] - 输入类型转换 Trait
//!
//! ## 多态输入
//!
//! 所有指标函数都支持多种输入类型，无需手动转换：
//!
//! - **分类指标**：支持 `&[usize]`、`Vec<i32>`、`Tensor`（自动 argmax）等
//! - **回归指标**：支持 `&[f32]`、`Vec<f64>`、`Tensor` 等
//!
//! ## 使用示例
//!
//! ```rust
//! use only_torch::metrics::{accuracy, r2_score};
//! use only_torch::tensor::Tensor;
//!
//! // 分类任务 - 直接传 slice
//! let acc = accuracy(&[0, 1, 1, 0, 1], &[0, 1, 0, 0, 1]);
//! println!("Accuracy = {:.1}%", acc * 100.0);  // 80.0%
//!
//! // 分类任务 - 传 Tensor（自动 argmax）
//! let logits = Tensor::new(&[3, 2], &[0.1, 0.9, 0.8, 0.2, 0.3, 0.7]);
//! let labels = Tensor::new(&[3, 2], &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
//! let acc = accuracy(&logits, &labels);
//!
//! // 回归任务
//! let r2 = r2_score(&[2.5, 0.0, 2.0, 8.0], &[3.0, -0.5, 2.0, 7.0]);
//! println!("R² = {:.4}", r2);  // R² ≈ 0.9486
//! ```

pub mod classification;
pub mod regression;
pub mod traits;

#[cfg(test)]
mod tests;

// 导出 Trait（用户可能需要为自定义类型实现）
pub use traits::{IntoClassLabels, IntoFloatValues};

// 导出常用函数到顶层，方便用户使用
pub use classification::{accuracy, confusion_matrix, f1_score, precision, recall};
pub use regression::r2_score;
