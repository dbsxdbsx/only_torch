/*
 * @Author       : 老董
 * @Date         : 2025-01-21
 * @Description  : 损失函数封装（PyTorch 风格 API）
 *
 * 提供类似 PyTorch 的损失函数使用体验，支持智能缓存。
 *
 * # 智能缓存
 * 自动为不同的 output 节点创建独立的 loss 计算图，
 * 支持变长序列等不同形状输入的场景。
 *
 * ```ignore
 * let criterion = CrossEntropyLoss::new();
 * for (x_batch, y_batch) in bucketed_loader.iter() {
 *     // x_batch 可能每个 batch 形状不同
 *     let output = model.forward(&x_batch)?;  // 不同形状 -> 不同 output 节点
 *     let loss = criterion.forward(&output, &y_batch)?;  // 自动缓存！
 *     loss.backward()?;
 * }
 * ```
 */

use super::{GraphError, NodeId, Var, VarLossOps};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::collections::HashMap;

// ==================== 内部状态结构 ====================

/// 单个 output 节点对应的损失状态
struct LossState {
    /// 内部创建的 target 输入节点
    target_node: Var,
    /// 内部创建的 loss 节点
    loss_node: Var,
}

// ==================== CrossEntropyLoss ====================

/// 交叉熵损失函数（PyTorch 风格）
///
/// 内置 Softmax，适用于多分类任务。
/// 支持智能缓存：自动为不同的 output 节点创建独立的 loss 子图。
///
/// # 使用示例
/// ```ignore
/// let criterion = CrossEntropyLoss::new();
///
/// for (x_batch, y_batch) in train_loader.iter() {
///     let output = model.forward(&x_batch)?;
///     let loss = criterion.forward(&output, &y_batch)?;
///     loss.backward()?;
///     optimizer.step()?;
/// }
/// ```
///
/// # 变长序列支持
/// ```ignore
/// // 不同长度的序列会产生不同的 output 节点
/// // CrossEntropyLoss 自动为每个节点创建独立的 loss 子图
/// for (seq_len, samples) in bucketed_data {
///     let output = model.forward(&x_batch)?;  // 每个 seq_len 产生不同节点
///     let loss = criterion.forward(&output, &y_batch)?;  // 自动缓存！
/// }
/// ```
pub struct CrossEntropyLoss {
    /// 按 output 节点 ID 缓存的 loss 状态
    cache: RefCell<HashMap<NodeId, LossState>>,
}

impl CrossEntropyLoss {
    /// 创建交叉熵损失函数
    pub fn new() -> Self {
        Self {
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// 计算损失（PyTorch 风格）
    ///
    /// # 参数
    /// - `output`: 模型输出（logits）
    /// - `target`: 目标标签（one-hot 编码）
    ///
    /// # 返回
    /// 损失值节点（可直接调用 `.backward()`）
    ///
    /// # 智能缓存
    /// - 相同 output 节点复用已创建的 loss 子图
    /// - 不同 output 节点自动创建新的 loss 子图
    pub fn forward(&self, output: &Var, target: &Tensor) -> Result<Var, GraphError> {
        let output_id = output.node_id();
        let mut cache = self.cache.borrow_mut();

        if let Some(s) = cache.get(&output_id) {
            // 缓存命中：复用已有的 loss 子图
            s.target_node.set_value(target)?;
            return Ok(s.loss_node.clone());
        }

        // 缓存未命中：创建新的 loss 子图
        let graph = output.get_graph();
        let target_node = graph.zeros(target.shape())?;
        target_node.set_value(target)?;
        let loss_node = output.cross_entropy(&target_node)?;

        cache.insert(
            output_id,
            LossState {
                target_node,
                loss_node: loss_node.clone(),
            },
        );

        Ok(loss_node)
    }

    /// 获取缓存数量
    pub fn cache_size(&self) -> usize {
        self.cache.borrow().len()
    }

    /// 清空缓存
    pub fn clear_cache(&self) {
        self.cache.borrow_mut().clear();
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== MseLoss ====================

/// 均方误差损失函数（PyTorch 风格）
///
/// 适用于回归任务。
/// 支持智能缓存：自动为不同的 output 节点创建独立的 loss 子图。
///
/// # 使用示例
/// ```ignore
/// let criterion = MseLoss::new();
///
/// for (x_batch, y_batch) in train_loader.iter() {
///     let output = model.forward(&x_batch)?;
///     let loss = criterion.forward(&output, &y_batch)?;
///     loss.backward()?;
///     optimizer.step()?;
/// }
/// ```
pub struct MseLoss {
    /// 按 output 节点 ID 缓存的 loss 状态
    cache: RefCell<HashMap<NodeId, LossState>>,
}

impl MseLoss {
    /// 创建均方误差损失函数
    pub fn new() -> Self {
        Self {
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// 计算损失（PyTorch 风格）
    ///
    /// # 参数
    /// - `output`: 模型输出
    /// - `target`: 目标值
    ///
    /// # 返回
    /// 损失值节点（可直接调用 `.backward()`）
    pub fn forward(&self, output: &Var, target: &Tensor) -> Result<Var, GraphError> {
        let output_id = output.node_id();
        let mut cache = self.cache.borrow_mut();

        if let Some(s) = cache.get(&output_id) {
            // 缓存命中：复用已有的 loss 子图
            s.target_node.set_value(target)?;
            return Ok(s.loss_node.clone());
        }

        // 缓存未命中：创建新的 loss 子图
        let graph = output.get_graph();
        let target_node = graph.zeros(target.shape())?;
        target_node.set_value(target)?;
        let loss_node = output.mse_loss(&target_node)?;

        cache.insert(
            output_id,
            LossState {
                target_node,
                loss_node: loss_node.clone(),
            },
        );

        Ok(loss_node)
    }

    /// 获取缓存数量
    pub fn cache_size(&self) -> usize {
        self.cache.borrow().len()
    }

    /// 清空缓存
    pub fn clear_cache(&self) {
        self.cache.borrow_mut().clear();
    }
}

impl Default for MseLoss {
    fn default() -> Self {
        Self::new()
    }
}
