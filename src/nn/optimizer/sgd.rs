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
    /// 创建新的SGD优化器
    pub fn new(graph: &Graph, learning_rate: f32) -> Result<Self, GraphError> {
        let state = OptimizerState::new(graph, learning_rate)?;
        Ok(Self { state })
    }
}

impl Optimizer for SGD {
    // ========== 单样本模式 ==========

    /// 执行一步训练：前向传播 + 反向传播 + 梯度累积
    fn one_step(&mut self, graph: &mut Graph, target_node: NodeId) -> Result<(), GraphError> {
        self.state.forward_backward_accumulate(graph, target_node)
    }

    /// 更新参数（使用梯度下降算法）
    fn update(&mut self, graph: &mut Graph) -> Result<(), GraphError> {
        // 对每个可训练参数执行梯度下降更新
        for &node_id in self.state.trainable_nodes() {
            if let Some(avg_gradient) = self
                .state
                .gradient_accumulator()
                .get_average_gradient(node_id)
            {
                // 获取当前参数值
                let current_value = graph.get_node_value(node_id)?.unwrap();

                // 梯度下降更新：θ = θ - α * ∇θ
                let new_value = current_value - self.state.learning_rate() * &avg_gradient;

                // 设置新的参数值
                graph.set_node_value(node_id, Some(&new_value))?;
            }
        }

        // 清除累积的梯度
        self.state.reset();
        Ok(())
    }

    // ========== Batch 模式 ==========

    /// Batch 模式的一步训练
    fn one_step_batch(&mut self, graph: &mut Graph, target_node: NodeId) -> Result<(), GraphError> {
        // 清除计算图中所有节点的梯度
        graph.clear_grad()?;

        // Batch 前向传播
        graph.forward_batch(target_node)?;

        // Batch 反向传播（梯度已经对 batch 求平均）
        graph.backward_batch(target_node)?;

        Ok(())
    }

    /// Batch 模式的参数更新
    fn update_batch(&mut self, graph: &mut Graph) -> Result<(), GraphError> {
        // 对每个可训练参数执行梯度下降更新
        for &node_id in self.state.trainable_nodes() {
            if let Some(gradient) = graph.get_node_grad_batch(node_id)? {
                // 获取当前参数值
                let current_value = graph.get_node_value(node_id)?.unwrap();

                // 梯度下降更新：θ = θ - α * ∇θ
                let new_value = current_value - self.state.learning_rate() * gradient;

                // 设置新的参数值
                graph.set_node_value(node_id, Some(&new_value))?;
            }
        }

        Ok(())
    }

    // ========== 通用方法 ==========

    /// 重置累积状态
    fn reset(&mut self) {
        self.state.reset();
    }

    /// 获取学习率
    fn learning_rate(&self) -> f32 {
        self.state.learning_rate()
    }

    /// 设置学习率
    fn set_learning_rate(&mut self, lr: f32) {
        self.state.set_learning_rate(lr);
    }
}
