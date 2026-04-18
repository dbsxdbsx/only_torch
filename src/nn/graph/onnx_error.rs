/*
 * @Author       : 老董
 * @Date         : 2026-04-18
 * @Description  : ONNX 导入/导出错误类型
 *
 * 所有 ONNX 相关操作的错误统一通过 OnnxError 枚举报告。
 * 设计原则：不支持的算子/格式必须明确报错，不允许静默忽略。
 */

use std::fmt;

/// ONNX 操作错误类型
#[derive(Debug)]
pub enum OnnxError {
    /// 文件读写错误
    Io(std::io::Error),

    /// protobuf 解析失败（文件损坏或格式不合法）
    ParseError(String),

    /// 不支持的 ONNX opset 版本（要求 opset 13–21）
    UnsupportedOpsetVersion {
        version: i64,
        min_supported: i64,
        max_supported: i64,
    },

    /// 不支持的 ONNX 算子（未在映射表中）
    UnsupportedOperator {
        op_type: String,
        node_name: String,
    },

    /// 不支持的数据类型（仅支持 float32）
    UnsupportedDataType {
        data_type: i32,
        context: String,
    },

    /// 不支持的属性值（如 Gemm 的 alpha ≠ 1）
    UnsupportedAttribute {
        op_type: String,
        attribute: String,
        reason: String,
    },

    /// 不支持的卷积/池化配置（如 group>1、dilation>1、3D 等）
    UnsupportedConvConfig {
        op_type: String,
        reason: String,
    },

    /// 图结构错误（缺少输入、环路等）
    InvalidGraph(String),

    /// 权重数据缺失或形状不匹配
    WeightError {
        tensor_name: String,
        reason: String,
    },

    /// 导出时遇到训练专用节点（loss/target 在输出路径上）
    TrainingNodeInExportPath {
        node_type: String,
        node_name: String,
    },

    /// GraphDescriptor 构建/转换错误
    DescriptorError(String),
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxError::Io(e) => write!(f, "ONNX I/O 错误: {e}"),
            OnnxError::ParseError(msg) => write!(f, "ONNX 解析错误: {msg}"),
            OnnxError::UnsupportedOpsetVersion {
                version,
                min_supported,
                max_supported,
            } => write!(
                f,
                "不支持的 ONNX opset 版本: {version}（支持范围: {min_supported}–{max_supported}）"
            ),
            OnnxError::UnsupportedOperator {
                op_type,
                node_name,
            } => write!(
                f,
                "不支持的 ONNX 算子: op_type=\"{op_type}\"（节点: \"{node_name}\"）"
            ),
            OnnxError::UnsupportedDataType { data_type, context } => write!(
                f,
                "不支持的数据类型: {data_type}（仅支持 float32）（{context}）"
            ),
            OnnxError::UnsupportedAttribute {
                op_type,
                attribute,
                reason,
            } => write!(
                f,
                "不支持的属性: {op_type}.{attribute} — {reason}"
            ),
            OnnxError::UnsupportedConvConfig { op_type, reason } => {
                write!(f, "不支持的卷积/池化配置: {op_type} — {reason}")
            }
            OnnxError::InvalidGraph(msg) => write!(f, "无效的 ONNX 图结构: {msg}"),
            OnnxError::WeightError {
                tensor_name,
                reason,
            } => write!(f, "权重错误: \"{tensor_name}\" — {reason}"),
            OnnxError::TrainingNodeInExportPath {
                node_type,
                node_name,
            } => write!(
                f,
                "导出路径中包含训练专用节点: {node_type}（\"{node_name}\"）— 请仅导出推理子图"
            ),
            OnnxError::DescriptorError(msg) => write!(f, "GraphDescriptor 错误: {msg}"),
        }
    }
}

impl std::error::Error for OnnxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OnnxError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for OnnxError {
    fn from(e: std::io::Error) -> Self {
        OnnxError::Io(e)
    }
}
