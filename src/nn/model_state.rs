/*
 * @Author       : 老董
 * @Date         : 2025-01-21
 * @Description  : 模型状态管理器（支持 PyTorch 风格的 forward API）
 *
 * ModelState 使得用户定义的模型可以直接接收 Tensor 作为输入，
 * 内部自动处理计算图节点的创建和复用。
 *
 * # 智能缓存
 * 支持不同形状的输入（如变长序列），自动为每种形状维护独立的计算图子图。
 *
 * # 使用示例
 * ```ignore
 * pub struct MyModel {
 *     fc1: Linear,
 *     fc2: Linear,
 *     state: ModelState,
 * }
 *
 * impl MyModel {
 *     pub fn new(graph: &Graph) -> Result<Self, GraphError> {
 *         Ok(Self {
 *             fc1: Linear::new(graph, 2, 4, true, "fc1")?,
 *             fc2: Linear::new(graph, 4, 2, true, "fc2")?,
 *             state: ModelState::new(graph),
 *         })
 *     }
 *
 *     pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
 *         self.state.forward(x, |input| {
 *             Ok(self.fc2.forward(&self.fc1.forward(input).tanh()))
 *         })
 *     }
 * }
 * ```
 */

use super::{Graph, GraphError, Var};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::collections::HashMap;

/// 模型状态缓存（单个形状）
struct StateCache {
    /// 输入节点
    input: Var,
    /// 输出节点（预构建的计算图终点）
    output: Var,
}

/// 模型状态管理器
///
/// 提供"延迟绑定 + 智能缓存"机制，使模型可以直接接收 Tensor 输入。
/// 支持变长序列等不同形状的输入，自动为每种形状创建独立的计算图子图。
///
/// # 工作原理
/// - **首次调用**某形状：创建输入节点，构建计算图，缓存结果
/// - **后续调用**相同形状：复用已创建的节点，只更新输入值
/// - **不同形状**：自动创建新的子图并缓存
///
/// # 使用示例
/// ```ignore
/// pub struct VarLenRNN {
///     rnn: Rnn,
///     fc: Linear,
///     state: ModelState,
/// }
///
/// impl VarLenRNN {
///     pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
///         // x 可以是 [batch, seq_len=5, 1] 或 [batch, seq_len=10, 1]
///         // ModelState 自动为每种形状创建独立缓存
///         self.state.forward(x, |input| {
///             let h = self.rnn.forward(input)?;
///             Ok(self.fc.forward(&h))
///         })
///     }
/// }
/// ```
pub struct ModelState {
    graph: Graph,
    /// 按输入形状缓存的子图：shape -> (input, output)
    cache: RefCell<HashMap<Vec<usize>, StateCache>>,
}

impl ModelState {
    /// 创建新的模型状态管理器
    ///
    /// # 参数
    /// - `graph`: 计算图引用（与模型的层共享同一个图）
    pub fn new(graph: &Graph) -> Self {
        Self {
            graph: graph.clone(),
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// PyTorch 风格的 forward（延迟绑定 + 智能缓存）
    ///
    /// # 参数
    /// - `x`: 输入数据（Tensor）
    /// - `compute`: 计算逻辑闭包，接收 `&Var` 输入，返回 `Result<Var, GraphError>`
    ///
    /// # 返回
    /// 模型输出节点（Var）
    ///
    /// # 智能缓存
    /// - 相同形状的输入复用已创建的计算图
    /// - 不同形状的输入自动创建新的子图
    ///
    /// # 示例
    /// ```ignore
    /// // 简单层（如 Linear）：用 Ok(...) 包装
    /// self.state.forward(x, |input| {
    ///     Ok(self.fc2.forward(&self.fc1.forward(input).tanh()))
    /// })
    ///
    /// // 复杂层（如 RNN）：直接用 ? 传播错误
    /// self.state.forward(x, |input| {
    ///     let h = self.rnn.forward(input)?;
    ///     Ok(self.fc.forward(&h))
    /// })
    /// ```
    pub fn forward<F>(&self, x: &Tensor, compute: F) -> Result<Var, GraphError>
    where
        F: FnOnce(&Var) -> Result<Var, GraphError>,
    {
        let shape = x.shape().to_vec();
        let mut cache = self.cache.borrow_mut();

        if let Some(c) = cache.get(&shape) {
            // 缓存命中：复用已有子图
            c.input.set_value(x)?;
            c.output.forward()?;
            return Ok(c.output.clone());
        }

        // 缓存未命中：创建新的子图
        let input = self.graph.zeros(x.shape())?;
        input.set_value(x)?;

        // 调用用户提供的计算逻辑
        let output = compute(&input)?;

        // 触发前向传播，确保输出值是最新的
        output.forward()?;

        // 缓存
        cache.insert(shape, StateCache { input, output: output.clone() });

        Ok(output)
    }

    /// 获取当前缓存的形状列表
    pub fn cached_shapes(&self) -> Vec<Vec<usize>> {
        self.cache.borrow().keys().cloned().collect()
    }

    /// 获取缓存数量
    pub fn cache_size(&self) -> usize {
        self.cache.borrow().len()
    }

    /// 检查是否已初始化（至少有一个缓存）
    pub fn is_initialized(&self) -> bool {
        !self.cache.borrow().is_empty()
    }

    /// 清空缓存
    ///
    /// 用于需要重置模型状态的场景（通常不需要调用）
    pub fn clear_cache(&self) {
        self.cache.borrow_mut().clear();
    }
}
