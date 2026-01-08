//! 数据加载错误类型定义

use std::path::PathBuf;
use thiserror::Error;

/// 数据加载相关错误
#[derive(Debug, Error)]
pub enum DataError {
    /// 文件未找到
    #[error("文件未找到: {0}")]
    FileNotFound(PathBuf),

    /// IO 错误
    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    /// 格式错误（如 magic number 不匹配）
    #[error("格式错误: {0}")]
    FormatError(String),

    /// 索引越界
    #[error("索引越界: {index} >= {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    /// 形状不匹配
    #[error("形状不匹配: 期望 {expected:?}, 实际 {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// 下载错误
    #[error("下载错误: {0}")]
    DownloadError(String),

    /// 校验和不匹配
    #[error("校验和不匹配: 期望 {expected}, 实际 {got}")]
    ChecksumMismatch { expected: String, got: String },

    /// 解压错误
    #[error("解压错误: {0}")]
    DecompressionError(String),
}
