/*
 * @Author       : 老董
 * @Date         : 2025-07-24 16:30:00
 * @LastEditors  : 老董
 * @LastEditTime : 2025-07-24 16:30:00
 * @Description  : Adam优化器实现
 */

use super::base::{Optimizer, OptimizerState};
use crate::nn::{Graph, GraphError, NodeId};
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Adam优化器
pub struct Adam {
    state: OptimizerState,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    /// 一阶矩估计
    m: HashMap<NodeId, Tensor>,
    /// 二阶矩估计
    v: HashMap<NodeId, Tensor>,
    /// 时间步
    t: usize,
}

impl Adam {
    /// 创建新的Adam优化器（自动优化图中所有可训练节点）
    pub fn new(
        graph: &Graph,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<Self, GraphError> {
        let state = OptimizerState::new(graph, learning_rate)?;
        Ok(Self {
            state,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        })
    }

    /// 使用默认参数创建Adam优化器
    pub fn new_default(graph: &Graph, learning_rate: f32) -> Result<Self, GraphError> {
        Self::new(graph, learning_rate, 0.9, 0.999, 1e-8)
    }

    /// 使用指定参数创建Adam优化器
    ///
    /// 用于需要分别优化不同参数组的场景，如：
    /// - GAN 训练（G 和 D 用不同优化器）
    /// - 迁移学习（冻结部分层）
    /// - 分层学习率
    ///
    /// # 示例
    /// ```ignore
    /// // GAN 训练：分别为 G 和 D 创建优化器
    /// let optimizer_g = Adam::with_params(&g_params, 0.0002, 0.5, 0.999, 1e-8);
    /// let optimizer_d = Adam::with_params(&d_params, 0.0002, 0.5, 0.999, 1e-8);
    /// ```
    pub fn with_params(
        params: &[NodeId],
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Self {
        let state = OptimizerState::with_params(params.to_vec(), learning_rate);
        Self {
            state,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    /// 参数更新（使用已计算的梯度）
    ///
    /// 直接使用节点的 `.grad` 进行 Adam 更新
    fn step(&mut self, graph: &mut Graph) -> Result<(), GraphError> {
        self.t += 1;

        // 先收集所有需要更新的参数及其梯度（避免借用冲突）
        let trainable_nodes: Vec<_> = self.state.trainable_nodes().to_vec();
        let gradients: Vec<_> = trainable_nodes
            .iter()
            .filter_map(|&node_id| {
                graph
                    .get_node_grad(node_id)
                    .ok()
                    .flatten()
                    .map(|g| (node_id, g))
            })
            .collect();

        // 对每个可训练参数执行 Adam 更新
        for (node_id, gradient) in gradients {
            self.adam_update_with_gradient(graph, node_id, &gradient)?;
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }

    fn learning_rate(&self) -> f32 {
        self.state.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.state.set_learning_rate(lr);
    }
}

impl Adam {
    /// Adam 参数更新的核心逻辑（单样本和 Batch 共用）
    fn adam_update_with_gradient(
        &mut self,
        graph: &mut Graph,
        node_id: NodeId,
        gradient: &Tensor,
    ) -> Result<(), GraphError> {
        // 获取当前参数值
        let current_value = graph.get_node_value(node_id)?.unwrap();

        // 预计算缩放后的梯度项
        let scaled_gradient = gradient * (1.0 - self.beta1);
        let gradient_squared = gradient * gradient;
        let scaled_gradient_squared = &gradient_squared * (1.0 - self.beta2);

        // 原地更新一阶矩估计: m = β1 * m + (1 - β1) * g
        let m = self
            .m
            .entry(node_id)
            .or_insert_with(|| Tensor::zeros(gradient.shape()));
        *m *= self.beta1;
        *m += &scaled_gradient;

        // 原地更新二阶矩估计: v = β2 * v + (1 - β2) * g²
        let v = self
            .v
            .entry(node_id)
            .or_insert_with(|| Tensor::zeros(gradient.shape()));
        *v *= self.beta2;
        *v += &scaled_gradient_squared;

        // 偏差修正
        let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));

        // 参数更新: θ = θ - α * m_hat / (√v_hat + ε)
        let v_hat_sqrt = v_hat.sqrt();
        let denominator = &v_hat_sqrt + self.epsilon;
        let update = &m_hat / &denominator;
        let new_value = current_value - self.state.learning_rate() * &update;

        // 设置新的参数值
        graph.set_node_value(node_id, Some(&new_value))?;

        Ok(())
    }
}
