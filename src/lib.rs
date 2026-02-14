//! # Only Torch
//!
//! `only_torch`项目旨在用纯rust将[pytorch](https://pytorch.org)这类基于梯度的机器学习算法
//! 和[NEAT](https://ieeexplore.ieee.org/document/6790655)这类网络突变（类似遗传算法）整合在一起，
//! 打造一个相对来说轻便的跨平台（windows，linux，android...）快速推理AI框架。
//!

// BLAS 后端：强制链接 -src crate（空 crate，仅提供本地库）
#[cfg(feature = "blas-mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "blas-openblas")]
extern crate openblas_src;

pub mod data;
pub mod errors;
pub mod logic;
pub mod metrics;
pub mod nn;
pub mod rl;
pub mod tensor;
pub mod utils;
pub mod vision;

pub use data::{DataError, MnistDataset};
pub use tensor::Tensor;
