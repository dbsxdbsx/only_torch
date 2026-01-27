/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : Graph 模块的错误类型和可视化相关类型
 */

use crate::nn::NodeId;

/// Graph 操作错误类型
#[derive(Debug, PartialEq, Eq)]
pub enum GraphError {
    GraphNotFound(String),
    NodeNotFound(NodeId),
    InvalidOperation(String),
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
        message: String,
    },
    DimensionMismatch {
        expected: usize,
        got: usize,
        message: String,
    },
    ComputationError(String),
    DuplicateName(String),
    DuplicateNodeName(String),
}

// ========== 可视化相关类型 ==========

/// 图像输出格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFormat {
    /// PNG 格式（默认）
    #[default]
    Png,
    /// SVG 矢量格式
    Svg,
    /// PDF 格式
    Pdf,
}

impl ImageFormat {
    /// 获取文件扩展名（不含点号）
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Svg => "svg",
            Self::Pdf => "pdf",
        }
    }

    /// 从扩展名解析格式（用于错误提示）
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "png" => Some(Self::Png),
            "svg" => Some(Self::Svg),
            "pdf" => Some(Self::Pdf),
            _ => None,
        }
    }
}

/// 可视化输出结果
#[derive(Debug)]
pub struct VisualizationOutput {
    /// DOT 文件路径（始终生成）
    pub dot_path: std::path::PathBuf,
    /// 图像文件路径（仅当 Graphviz 可用时生成）
    pub image_path: Option<std::path::PathBuf>,
    /// Graphviz 是否可用
    pub graphviz_available: bool,
    /// 如果 Graphviz 不可用，提供安装提示
    pub graphviz_hint: Option<String>,
}
