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
    /// 创建新的Adam优化器
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
}

impl Optimizer for Adam {
    /// 执行一步训练：前向传播 + 反向传播 + 梯度累积
    fn one_step(&mut self, graph: &mut Graph, target_node: NodeId) -> Result<(), GraphError> {
        self.state.forward_backward_accumulate(graph, target_node)
    }

    /// 更新参数（使用Adam算法）
    fn update(&mut self, graph: &mut Graph) -> Result<(), GraphError> {
        self.t += 1;

        // 对每个可训练参数执行Adam更新
        for &node_id in self.state.trainable_nodes() {
            if let Some(gradient) = self
                .state
                .gradient_accumulator()
                .get_average_gradient(node_id)
            {
                // 获取当前参数值
                let current_value = graph.get_node_value(node_id)?.unwrap();

                // 获取或初始化一阶矩和二阶矩
                let m_prev = self
                    .m
                    .get(&node_id)
                    .cloned()
                    .unwrap_or_else(|| Tensor::zeros(gradient.shape()));
                let v_prev = self
                    .v
                    .get(&node_id)
                    .cloned()
                    .unwrap_or_else(|| Tensor::zeros(gradient.shape()));

                // 更新一阶矩估计: m_t = β1 * m_{t-1} + (1 - β1) * g_t
                let m_t = &m_prev * self.beta1 + &gradient * (1.0 - self.beta1);

                // 更新二阶矩估计: v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                let gradient_squared = &gradient * &gradient;
                let v_t = &v_prev * self.beta2 + &gradient_squared * (1.0 - self.beta2);

                // 偏差修正
                let m_hat = &m_t / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = &v_t / (1.0 - self.beta2.powi(self.t as i32));

                // 参数更新: θ = θ - α * m_hat / (√v_hat + ε)
                let v_hat_sqrt = v_hat.sqrt();
                let denominator = &v_hat_sqrt + self.epsilon;
                let update = &m_hat / &denominator;
                let new_value = current_value - self.state.learning_rate() * &update;

                // 设置新的参数值
                graph.set_node_value(node_id, Some(&new_value))?;

                // 保存状态
                self.m.insert(node_id, m_t);
                self.v.insert(node_id, v_t);
            }
        }

        // 清除累积的梯度
        self.state.reset();
        Ok(())
    }

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
