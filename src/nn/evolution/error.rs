/*
 * @Author       : 老董
 * @Date         : 2026-03-09
 * @Description  : 演化模块错误类型
 *
 * EvolutionError 是演化 API 的顶层错误枚举：
 * - InvalidData: 数据验证失败（空数据、数量不匹配等）
 * - InvalidConfig: 配置参数无效
 * - Graph: 底层计算图错误（透传 GraphError）
 */

use std::fmt;

use crate::nn::GraphError;

/// 演化模块错误
#[derive(Debug)]
pub enum EvolutionError {
    /// 数据验证错误（空数据、数量不匹配等）
    InvalidData(String),
    /// 配置参数无效
    InvalidConfig(String),
    /// 底层计算图错误
    Graph(GraphError),
}

impl fmt::Display for EvolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvolutionError::InvalidData(msg) => write!(f, "{msg}"),
            EvolutionError::InvalidConfig(msg) => write!(f, "{msg}"),
            EvolutionError::Graph(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for EvolutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EvolutionError::Graph(e) => Some(e),
            _ => None,
        }
    }
}

impl From<GraphError> for EvolutionError {
    fn from(e: GraphError) -> Self {
        EvolutionError::Graph(e)
    }
}
