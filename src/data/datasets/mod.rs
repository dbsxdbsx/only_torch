//! 内置数据集
//!
//! 提供常用的预定义数据集，如 MNIST。

mod mnist;

pub use mnist::{default_data_dir, MnistDataset};

