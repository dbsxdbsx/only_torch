/*
 * @Author       : 老董
 * @Date         : 2025-07-24 16:00:00
 * @LastEditors  : 老董
 * @LastEditTime : 2025-07-24 16:00:00
 * @Description  : 优化器基础trait和辅助结构
 */

use crate::nn::{Graph, GraphError, NodeId};

/// 优化器核心 trait
pub trait Optimizer {
    /// 参数更新（使用已计算的梯度）
    ///
    /// `PyTorch` 风格训练循环：
    /// ```ignore
    /// optimizer.zero_grad();      // 或 graph.zero_grad()
    /// graph.forward(loss)?;
    /// graph.backward(loss)?;
    /// optimizer.step(&mut graph)?; // ← 只更新参数，不做 forward/backward
    /// ```
    ///
    /// 此方法直接使用节点的 `.grad` 进行参数更新，
    /// 不区分单样本和批量（API 已统一）。
    fn step(&mut self, graph: &mut Graph) -> Result<(), GraphError>;

    /// 重置累积状态
    fn reset(&mut self);

    /// 获取学习率
    fn learning_rate(&self) -> f32;

    /// 设置学习率
    fn set_learning_rate(&mut self, lr: f32);
}

/// 优化器状态管理（内部实现，不对外暴露）
pub(crate) struct OptimizerState {
    /// 可训练参数的节点 ID 列表
    trainable_nodes: Vec<NodeId>,
    /// 学习率
    learning_rate: f32,
}

impl OptimizerState {
    /// 创建新的优化器状态（自动获取图中所有可训练节点）
    pub(crate) fn new(graph: &Graph, learning_rate: f32) -> Result<Self, GraphError> {
        let trainable_nodes = graph.get_trainable_nodes();
        Ok(Self {
            trainable_nodes,
            learning_rate,
        })
    }

    /// 使用指定参数创建优化器状态
    ///
    /// 用于需要分别优化不同参数组的场景，如：
    /// - GAN 训练（G 和 D 用不同优化器）
    /// - 迁移学习（冻结部分层）
    /// - 分层学习率
    pub(crate) const fn with_params(params: Vec<NodeId>, learning_rate: f32) -> Self {
        Self {
            trainable_nodes: params,
            learning_rate,
        }
    }

    /// 获取可训练节点列表
    pub(crate) fn trainable_nodes(&self) -> &[NodeId] {
        &self.trainable_nodes
    }

    /// 获取学习率
    pub(crate) const fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// 设置学习率
    pub(crate) const fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}
