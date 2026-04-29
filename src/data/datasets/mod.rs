//! 内置数据集
//!
//! 提供常用的预定义数据集：
//! - MNIST：手写数字分类（分类任务经典）
//! - California Housing：房价回归（回归任务经典）

mod california_housing;
mod mnist;
mod yolo;

pub use california_housing::CaliforniaHousingDataset;
pub use mnist::{MnistDataset, default_data_dir};
pub use yolo::{parse_yolo_txt_file, parse_yolo_txt_labels};
