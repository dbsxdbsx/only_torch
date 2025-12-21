/*
 * @Author       : 老董
 * @Date         : 2025-07-24 16:00:00
 * @LastEditors  : 老董
 * @LastEditTime : 2025-07-24 16:00:00
 * @Description  : 优化器基础trait和辅助结构
 */

use crate::nn::{Graph, GraphError, NodeId};
use crate::tensor::Tensor;
use std::collections::HashMap;

/// 优化器核心trait
pub trait Optimizer {
    // ========== 单样本模式（Jacobi-based）==========

    /// 执行一步训练：前向传播 + 反向传播 + 梯度累积
    fn one_step(&mut self, graph: &mut Graph, target_node: NodeId) -> Result<(), GraphError>;

    /// 更新参数（执行具体的优化算法）
    fn update(&mut self, graph: &mut Graph) -> Result<(), GraphError>;

    // ========== Batch 模式（Gradient-based）==========

    /// Batch 模式的一步训练：batch 前向传播 + batch 反向传播
    ///
    /// 与单样本模式不同，batch 模式下：
    /// 1. 输入节点的 value 应包含 batch 维度
    /// 2. 梯度在 backward_batch 中已经对 batch 求平均
    /// 3. 不需要额外的梯度累积
    fn one_step_batch(&mut self, graph: &mut Graph, target_node: NodeId) -> Result<(), GraphError>;

    /// Batch 模式的参数更新
    ///
    /// 使用 batch backward 计算的梯度更新参数
    fn update_batch(&mut self, graph: &mut Graph) -> Result<(), GraphError>;

    // ========== 通用方法 ==========

    /// 重置累积状态
    fn reset(&mut self);

    /// 获取学习率
    fn learning_rate(&self) -> f32;

    /// 设置学习率
    fn set_learning_rate(&mut self, lr: f32);
}

/// 梯度累积器（内部实现，不对外暴露）
pub(crate) struct GradientAccumulator {
    /// 累积的梯度：NodeId -> 累积梯度
    accumulated_gradients: HashMap<NodeId, Tensor>,
    /// 累积的样本数量
    sample_count: usize,
}

impl GradientAccumulator {
    /// 创建新的梯度累积器
    pub(crate) fn new() -> Self {
        Self {
            accumulated_gradients: HashMap::new(),
            sample_count: 0,
        }
    }

    /// 累积单个样本的梯度
    pub(crate) fn accumulate(
        &mut self,
        node_id: NodeId,
        gradient: &Tensor,
    ) -> Result<(), GraphError> {
        if let Some(existing_gradient) = self.accumulated_gradients.get_mut(&node_id) {
            *existing_gradient = existing_gradient.clone() + gradient;
        } else {
            self.accumulated_gradients.insert(node_id, gradient.clone());
        }
        Ok(())
    }

    /// 增加样本计数
    pub(crate) const fn increment_sample_count(&mut self) {
        self.sample_count += 1;
    }

    /// 获取平均梯度
    pub(crate) fn get_average_gradient(&self, node_id: NodeId) -> Option<Tensor> {
        if self.sample_count == 0 {
            return None;
        }

        self.accumulated_gradients
            .get(&node_id)
            .map(|gradient| gradient.clone() / (self.sample_count as f32))
    }

    /// 清除累积状态
    pub(crate) fn clear(&mut self) {
        self.accumulated_gradients.clear();
        self.sample_count = 0;
    }

    /// 获取累积的样本数量
    pub(crate) const fn sample_count(&self) -> usize {
        self.sample_count
    }
}

/// 优化器状态管理（内部实现，不对外暴露）
pub(crate) struct OptimizerState {
    /// 可训练参数的节点ID列表
    trainable_nodes: Vec<NodeId>,
    /// 梯度累积器
    gradient_accumulator: GradientAccumulator,
    /// 学习率
    learning_rate: f32,
}

impl OptimizerState {
    /// 创建新的优化器状态
    pub(crate) fn new(graph: &Graph, learning_rate: f32) -> Result<Self, GraphError> {
        // 获取所有可训练的参数节点
        let trainable_nodes = graph.get_trainable_nodes();

        Ok(Self {
            trainable_nodes,
            gradient_accumulator: GradientAccumulator::new(),
            learning_rate,
        })
    }

    /// 获取可训练节点列表
    pub(crate) fn trainable_nodes(&self) -> &[NodeId] {
        &self.trainable_nodes
    }

    /// 获取梯度累积器的可变引用
    pub(crate) const fn gradient_accumulator_mut(&mut self) -> &mut GradientAccumulator {
        &mut self.gradient_accumulator
    }

    /// 获取梯度累积器的不可变引用
    pub(crate) const fn gradient_accumulator(&self) -> &GradientAccumulator {
        &self.gradient_accumulator
    }

    /// 获取学习率
    pub(crate) const fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// 设置学习率
    pub(crate) const fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    /// 执行前向反向传播并累积梯度
    pub(crate) fn forward_backward_accumulate(
        &mut self,
        graph: &mut Graph,
        target_node: NodeId,
    ) -> Result<(), GraphError> {
        // 清除计算图中所有节点的雅可比矩阵
        graph.clear_jacobi()?;

        // 前向传播计算目标节点
        graph.forward_node(target_node)?;

        // 反向传播计算雅可比矩阵
        graph.backward_nodes(&self.trainable_nodes, target_node)?;

        // 累积梯度
        for &node_id in &self.trainable_nodes {
            if let Some(gradient) = graph.get_node_grad(node_id)? {
                self.gradient_accumulator.accumulate(node_id, &gradient)?;
            }
        }

        // 增加样本计数
        self.gradient_accumulator.increment_sample_count();

        Ok(())
    }

    /// 重置累积状态
    pub(crate) fn reset(&mut self) {
        self.gradient_accumulator.clear();
    }
}
