/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : InstanceNorm 层 - 实例归一化
 *
 * 每个通道独立归一化。等价于 GroupNorm(num_groups=num_channels)。
 * 常用于风格迁移 (Style Transfer) 和 GAN。
 *
 * 输入 [N, C, H, W], 输出同形。
 */

use crate::nn::{Graph, GraphError, Module, Var};

use super::GroupNorm;

/// 实例归一化层
///
/// # 使用示例
/// ```ignore
/// let inst_norm = InstanceNorm::new(&graph, 32, 1e-5, "in")?;
/// let h = inst_norm.forward(&x);  // x: [N, 32, H, W]
/// ```
pub struct InstanceNorm {
    inner: GroupNorm,
}

impl InstanceNorm {
    /// 创建 InstanceNorm 层
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `num_channels`: 通道数
    /// - `eps`: 数值稳定性常数
    /// - `name`: 层名称
    pub fn new(
        graph: &Graph,
        num_channels: usize,
        eps: f32,
        name: &str,
    ) -> Result<Self, GraphError> {
        // InstanceNorm = GroupNorm(num_groups=num_channels)
        let inner = GroupNorm::new(graph, num_channels, num_channels, eps, name)?;
        Ok(Self { inner })
    }

    /// 前向传播
    pub fn forward(&self, x: impl crate::nn::IntoVar) -> Var {
        self.inner.forward(x)
    }
}

impl Module for InstanceNorm {
    fn parameters(&self) -> Vec<Var> {
        self.inner.parameters()
    }
}
