//! 数据加载模块
//!
//! 提供数据集加载、变换和批处理功能。
//!
//! # 主要组件
//!
//! - [`DataLoader`]: `PyTorch` 风格的数据批量加载器
//! - [`TensorDataset`]: 持有特征和标签的数据集
//! - [`MnistDataset`]: MNIST 手写数字数据集（分类任务）
//! - [`CaliforniaHousingDataset`]: California Housing 房价数据集（回归任务）
//! - [`transforms`]: 数据变换函数（归一化、one-hot 等）
//! - [`DataError`]: 数据加载错误类型
//!
//! # 使用示例
//!
//! ```ignore
//! use only_torch::data::{DataLoader, TensorDataset};
//!
//! // 创建数据集和加载器
//! let dataset = TensorDataset::new(train_x, train_y);
//! let loader = DataLoader::new(dataset, 32)
//!     .shuffle(true)
//!     .seed(42);
//!
//! // PyTorch 风格训练循环
//! for (x_batch, y_batch) in loader.iter() {
//!     model.forward(&x_batch)?;
//!     loss.backward()?;
//!     optimizer.step()?;
//! }
//! ```

mod dataloader;
pub mod datasets;
pub mod download;
pub mod error;
pub mod transforms;

#[cfg(test)]
mod tests;

// Re-exports
pub use dataloader::{BucketedDataLoader, DataLoader, TensorDataset, VarLenDataset, VarLenSample};
pub use datasets::{CaliforniaHousingDataset, MnistDataset, default_data_dir};
pub use error::DataError;
