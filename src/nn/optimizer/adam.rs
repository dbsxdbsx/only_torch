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
    // ========== 单样本模式 ==========

    /// 执行一步训练：前向传播 + 反向传播 + 梯度累积
    fn one_step(&mut self, graph: &mut Graph, target_node: NodeId) -> Result<(), GraphError> {
        self.state.forward_backward_accumulate(graph, target_node)
    }

    /// 更新参数（使用Adam算法）
    fn update(&mut self, graph: &mut Graph) -> Result<(), GraphError> {
        self.t += 1;

        // 收集可训练节点和它们的梯度（避免借用冲突）
        let trainable_nodes: Vec<_> = self.state.trainable_nodes().to_vec();
        let gradients: Vec<_> = trainable_nodes
            .iter()
            .filter_map(|&node_id| {
                self.state
                    .gradient_accumulator()
                    .get_average_gradient(node_id)
                    .map(|g| (node_id, g))
            })
            .collect();

        // 对每个可训练参数执行Adam更新
        for (node_id, gradient) in gradients {
            self.adam_update_with_gradient(graph, node_id, &gradient)?;
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

        // Batch 反向传播（只计算 trainable_nodes 的梯度，避免浪费）
        graph.backward_batch(target_node, Some(self.state.trainable_nodes()))?;

        Ok(())
    }

    /// Batch 模式的参数更新（使用Adam算法）
    ///
    /// # Panics
    /// 如果 optimizer 的 trainable_nodes 中有参数没有梯度，会 panic。
    /// 这通常意味着 `backward_batch` 的 `target_params` 与 optimizer 的参数范围不一致。
    fn update_batch(&mut self, graph: &mut Graph) -> Result<(), GraphError> {
        self.t += 1;

        // 收集所有 trainable_nodes 的梯度（若缺失会 panic）
        let gradients = self.state.collect_batch_gradients(graph)?;

        // 对每个可训练参数执行Adam更新
        for (node_id, gradient) in gradients {
            self.adam_update_with_gradient(graph, node_id, &gradient)?;
        }

        Ok(())
    }

    // ========== 通用方法 ==========

    /// 重置累积状态
    fn reset(&mut self) {
        self.state.reset();
        self.m.clear();
        self.v.clear();
        self.t = 0;
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
