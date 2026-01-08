/*
 * @Author       : 老董
 * @Date         : 2025-07-24 16:00:00
 * @LastEditors  : 老董
 * @LastEditTime : 2025-07-24 16:00:00
 * @Description  : 梯度下降优化器实现
 */

use super::base::{Optimizer, OptimizerState};
use crate::nn::{Graph, GraphError, NodeId};

/// SGD (随机梯度下降) 优化器
pub struct SGD {
    state: OptimizerState,
}

impl SGD {
    /// 创建新的SGD优化器（自动优化图中所有可训练节点）
    pub fn new(graph: &Graph, learning_rate: f32) -> Result<Self, GraphError> {
        let state = OptimizerState::new(graph, learning_rate)?;
        Ok(Self { state })
    }

    /// 使用指定参数创建SGD优化器
    ///
    /// 用于需要分别优化不同参数组的场景，如：
    /// - GAN 训练（G 和 D 用不同优化器）
    /// - 迁移学习（冻结部分层）
    /// - 分层学习率
    ///
    /// # 示例
    /// ```ignore
    /// // GAN 训练：分别为 G 和 D 创建优化器
    /// let optimizer_g = SGD::with_params(&g_params, 0.01);
    /// let optimizer_d = SGD::with_params(&d_params, 0.001);
    /// ```
    pub fn with_params(params: &[NodeId], learning_rate: f32) -> Self {
        let state = OptimizerState::with_params(params.to_vec(), learning_rate);
        Self { state }
    }
}

impl Optimizer for SGD {
    /// 参数更新（使用已计算的梯度）
    ///
    /// 直接使用节点的 `.grad` 进行 SGD 更新：θ = θ - α * ∇θ
    fn step(&mut self, graph: &mut Graph) -> Result<(), GraphError> {
        for &node_id in self.state.trainable_nodes() {
            // 获取节点的梯度（由 backward() 计算）
            if let Some(grad) = graph.get_node_grad(node_id)? {
                // 获取当前参数值
                let current_value = graph.get_node_value(node_id)?.unwrap();

                // 梯度下降更新：θ = θ - α * ∇θ
                let new_value = current_value - self.state.learning_rate() * &grad;

                // 设置新的参数值
                graph.set_node_value(node_id, Some(&new_value))?;
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        // SGD 无需重置状态
    }

    fn learning_rate(&self) -> f32 {
        self.state.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.state.set_learning_rate(lr);
    }
}
