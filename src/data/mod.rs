//! 数据加载模块
//!
//! 提供数据集加载、变换和批处理功能。
//!
//! # 主要组件
//!
//! - [`MnistDataset`]: MNIST 手写数字数据集
//! - [`transforms`]: 数据变换函数（归一化、one-hot 等）
//! - [`DataError`]: 数据加载错误类型
//!
//! # 使用示例
//!
//! ```ignore
//! use only_torch::data::{MnistDataset, transforms};
//!
//! // 加载 MNIST 训练集
//! let train_data = MnistDataset::train()?;
//!
//! // 获取 flatten 版本（用于 MLP）
//! let train_flat = train_data.flatten();
//!
//! // 获取单个样本
//! let (image, label) = train_flat.get(0)?;
//! ```

pub mod datasets;
pub mod error;
pub mod transforms;

#[cfg(test)]
mod tests;

// Re-exports
pub use datasets::{default_data_dir, MnistDataset};
pub use error::DataError;

