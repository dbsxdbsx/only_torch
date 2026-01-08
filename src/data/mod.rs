//! 数据加载模块
//!
//! 提供数据集加载、变换和批处理功能。
//!
//! # 主要组件
//!
//! - [`MnistDataset`]: MNIST 手写数字数据集（分类任务）
//! - [`CaliforniaHousingDataset`]: California Housing 房价数据集（回归任务）
//! - [`transforms`]: 数据变换函数（归一化、one-hot 等）
//! - [`DataError`]: 数据加载错误类型
//!
//! # 使用示例
//!
//! ```ignore
//! use only_torch::data::{MnistDataset, CaliforniaHousingDataset, transforms};
//!
//! // 加载 MNIST 训练集（分类）
//! let train_data = MnistDataset::train()?;
//! let train_flat = train_data.flatten();
//!
//! // 加载 California Housing（回归）
//! let housing = CaliforniaHousingDataset::load_default()?
//!     .standardize();
//! let (train, test) = housing.train_test_split(0.2, Some(42))?;
//! ```

pub mod datasets;
pub mod error;
pub mod transforms;

#[cfg(test)]
mod tests;

// Re-exports
pub use datasets::{CaliforniaHousingDataset, MnistDataset, default_data_dir};
pub use error::DataError;
