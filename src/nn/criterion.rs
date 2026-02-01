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
        let target_node = graph.target(target.shape())?;
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
        let target_node = graph.target(target.shape())?;
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

// ==================== MaeLoss ====================

/// 平均绝对误差损失函数（PyTorch 风格）
///
/// 适用于回归任务，相比 MSE 对异常值更鲁棒。
/// 支持智能缓存：自动为不同的 output 节点创建独立的 loss 子图。
///
/// # 使用示例
/// ```ignore
/// let criterion = MaeLoss::new();
///
/// for (x_batch, y_batch) in train_loader.iter() {
///     let output = model.forward(&x_batch)?;
///     let loss = criterion.forward(&output, &y_batch)?;
///     loss.backward()?;
///     optimizer.step()?;
/// }
/// ```
pub struct MaeLoss {
    /// 按 output 节点 ID 缓存的 loss 状态
    cache: RefCell<HashMap<NodeId, LossState>>,
}

impl MaeLoss {
    /// 创建平均绝对误差损失函数
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
        let target_node = graph.target(target.shape())?;
        target_node.set_value(target)?;
        let loss_node = output.mae_loss(&target_node)?;

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

impl Default for MaeLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// BCE（Binary Cross Entropy，二元交叉熵）损失函数（PyTorch 风格封装）
///
/// 采用 `BCEWithLogitsLoss` 形式，内置 Sigmoid 激活，数值稳定。
/// 适用于二分类和多标签分类任务。
///
/// ## 使用场景
/// - **二分类**：输出单个 logit，预测 0/1
/// - **多标签分类**：输出多个 logits，每个独立预测 0/1（一个样本可同时属于多个类别）
///
/// 相比 `cross_entropy`（Softmax CE），BCE 的核心区别是：
/// - Softmax CE：所有类别概率和 = 1（互斥类别）
/// - BCE：每个输出独立（非互斥类别）
///
/// ## 智能缓存
/// 与 `CrossEntropyLoss` 相同，按 `output` 节点 ID 缓存 loss 子图，
/// 避免重复创建节点。
///
/// # 示例
/// ```ignore
/// let criterion = BceLoss::new();
///
/// for batch in dataloader {
///     let logits = model.forward(&x_batch)?;
///     let loss = criterion.forward(&logits, &y_batch)?;  // y_batch 是 0/1 标签
///     loss.backward()?;
///     optimizer.step()?;
/// }
/// ```
pub struct BceLoss {
    /// 按 output 节点 ID 缓存的 loss 状态
    cache: RefCell<HashMap<NodeId, LossState>>,
}

impl BceLoss {
    /// 创建二元交叉熵损失函数
    pub fn new() -> Self {
        Self {
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// 计算损失（PyTorch 风格）
    ///
    /// # 参数
    /// - `logits`: 模型输出（未激活的原始值）
    /// - `target`: 二值标签（0 或 1）
    ///
    /// # 返回
    /// 损失值节点（可直接调用 `.backward()`）
    pub fn forward(&self, logits: &Var, target: &Tensor) -> Result<Var, GraphError> {
        let output_id = logits.node_id();
        let mut cache = self.cache.borrow_mut();

        if let Some(s) = cache.get(&output_id) {
            // 缓存命中：复用已有的 loss 子图
            s.target_node.set_value(target)?;
            return Ok(s.loss_node.clone());
        }

        // 缓存未命中：创建新的 loss 子图
        let graph = logits.get_graph();
        let target_node = graph.target(target.shape())?;
        target_node.set_value(target)?;
        let loss_node = logits.bce_loss(&target_node)?;

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

impl Default for BceLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Huber Loss（Smooth L1 Loss）损失函数（PyTorch 风格封装）
///
/// 结合 MSE（小误差）和 MAE（大误差）的优点，对异常值更鲁棒。
/// 是强化学习（DQN 等）的标准损失函数。
///
/// ## 行为
/// - |error| ≤ δ 时行为像 MSE（对小误差敏感）
/// - |error| > δ 时行为像 MAE（梯度被"裁剪"到 ±δ）
///
/// ## 典型应用
/// - **强化学习**：DQN 的 Q 值训练（δ=1.0 是标准配置）
/// - **带离群值的回归**：数据中存在异常值时
///
/// ## 智能缓存
/// 与其他 Criterion 相同，按 `output` 节点 ID 缓存 loss 子图。
///
/// # 示例
/// ```ignore
/// // 默认配置（δ=1.0，强化学习标准）
/// let criterion = HuberLoss::new();
///
/// // 自定义 δ
/// let criterion = HuberLoss::with_delta(0.5);
///
/// for batch in dataloader {
///     let q_values = model.forward(&states)?;
///     let loss = criterion.forward(&q_values, &target_q)?;
///     loss.backward()?;
///     optimizer.step()?;
/// }
/// ```
pub struct HuberLoss {
    /// δ 参数：小误差/大误差的分界阈值
    delta: f32,
    /// 按 output 节点 ID 缓存的 loss 状态
    cache: RefCell<HashMap<NodeId, LossState>>,
}

impl HuberLoss {
    /// 创建 Huber 损失函数（默认 δ=1.0，强化学习标准配置）
    pub fn new() -> Self {
        Self {
            delta: crate::nn::nodes::raw_node::DEFAULT_HUBER_DELTA,
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// 创建 Huber 损失函数（指定 δ 参数）
    pub fn with_delta(delta: f32) -> Self {
        Self {
            delta,
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// 获取 δ 参数
    pub const fn delta(&self) -> f32 {
        self.delta
    }

    /// 计算损失（PyTorch 风格）
    ///
    /// # 参数
    /// - `input`: 模型输出（预测值）
    /// - `target`: 目标值
    ///
    /// # 返回
    /// 损失值节点（可直接调用 `.backward()`）
    pub fn forward(&self, input: &Var, target: &Tensor) -> Result<Var, GraphError> {
        let output_id = input.node_id();
        let mut cache = self.cache.borrow_mut();

        if let Some(s) = cache.get(&output_id) {
            // 缓存命中：复用已有的 loss 子图
            s.target_node.set_value(target)?;
            return Ok(s.loss_node.clone());
        }

        // 缓存未命中：创建新的 loss 子图
        let graph = input.get_graph();
        let target_node = graph.target(target.shape())?;
        target_node.set_value(target)?;

        // 使用自定义 δ 创建 Huber Loss 节点
        let loss_inner = graph.inner_mut().create_huber_node(
            std::rc::Rc::clone(input.node()),
            std::rc::Rc::clone(target_node.node()),
            crate::nn::Reduction::Mean,
            self.delta,
            None,
        )?;
        let loss_node = Var::new_with_rc_graph(loss_inner, &graph.inner_rc());

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

impl Default for HuberLoss {
    fn default() -> Self {
        Self::new()
    }
}
