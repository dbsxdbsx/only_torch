/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : Graph 模块：计算图的核心实现
 *
 * 公开 API：
 * - `Graph`: 用户级句柄（PyTorch 风格）
 * - `GraphInner`: 底层实现（高级用户/NEAT 使用）
 * - `GraphError`: 错误类型
 */

mod error;
mod handle;
mod inner;
mod types;

pub use error::{GraphError, ImageFormat, VisualizationOutput};
pub use handle::Graph;
pub use inner::GraphInner;
// 这些类型用于可视化分组，作为公共 API 导出供外部使用
#[allow(unused_imports)]
pub use types::{GroupKind, LayerGroup, RecurrentLayerMeta, RecurrentUnrollInfo};
