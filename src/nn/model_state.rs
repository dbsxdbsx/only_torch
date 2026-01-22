/*
 * @Author       : 老董
 * @Date         : 2025-01-21
 * @Description  : 模型状态管理器（支持 PyTorch 风格的 forward API）
 *
 * ModelState 使得用户定义的模型可以接收 Tensor 或 Var 作为输入，
 * 内部自动处理计算图节点的创建和复用。
 *
 * # 核心机制：GradientRouter
 *
 * ModelState 使用 GradientRouter 节点作为模型的入口点：
 * - 无论输入是 Tensor 还是 Var，都复用同一个 GradientRouter
 * - GradientRouter 可以动态设置 detached 状态和梯度路由目标
 * - 这实现了"Archive 的效率 + PyTorch 的优雅"
 *
 * # 智能行为
 *
 * | 输入类型 | 行为 |
 * |---------|------|
 * | `&Tensor` | 复制值到 GradientRouter，无梯度路由 |
 * | `&Var`（detached）| 复制值到 GradientRouter，无梯度路由 |
 * | `&Var`（非 detached）| 复制值到 GradientRouter，设置梯度路由到源 Var |
 *
 * # 结果
 * - 图结构只构建一次（按形状缓存）
 * - 无论多少批次，图节点数保持 O(1)
 * - 梯度正确路由到需要的地方
 *
 * # 使用示例
 * ```ignore
 * // GAN 训练
 * let fake = G.forward(&noise)?;
 * let d_fake = D.forward(&fake.detach())?;  // D 训练：复用结构，无梯度路由
 * let d_fake_for_g = D.forward(&fake)?;     // G 训练：复用结构，梯度路由到 fake
 * ```
 */

use super::{DetachedVar, Graph, GraphError, NodeId, Var};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::collections::HashMap;

// ============================================================================
// ForwardInput Trait
// ============================================================================

/// 模型前向输入类型
///
/// 实现此 trait 的类型可以作为 `ModelState::forward()` 的输入。
/// 支持 `&Tensor`、`Tensor`、`&Var`、`Var` 四种类型。
///
/// # 统一缓存
///
/// 所有输入类型都使用相同的缓存策略（按形状缓存）。
/// 区别在于梯度路由：
/// - Tensor / detached Var: 无梯度路由
/// - 非 detached Var: 梯度路由到源 Var
pub trait ForwardInput {
    /// 获取输入的形状（用于缓存键）
    fn shape(&self) -> Vec<usize>;

    /// 获取输入的值
    fn get_value(&self) -> Result<Tensor, GraphError>;

    /// 是否处于 detached 状态
    ///
    /// - Tensor: 总是 true（没有梯度流）
    /// - Var: 取决于 Var 是否被 detach
    fn is_detached(&self) -> bool;

    /// 如果是 Var，返回其 NodeId（用于梯度路由）
    fn var_node_id(&self) -> Option<NodeId>;
}

impl ForwardInput for &Tensor {
    fn shape(&self) -> Vec<usize> {
        Tensor::shape(self).to_vec()
    }

    fn get_value(&self) -> Result<Tensor, GraphError> {
        Ok((*self).clone())
    }

    fn is_detached(&self) -> bool {
        true // Tensor 没有梯度流
    }

    fn var_node_id(&self) -> Option<NodeId> {
        None
    }
}

impl ForwardInput for Tensor {
    fn shape(&self) -> Vec<usize> {
        Tensor::shape(self).to_vec()
    }

    fn get_value(&self) -> Result<Tensor, GraphError> {
        Ok(self.clone())
    }

    fn is_detached(&self) -> bool {
        true
    }

    fn var_node_id(&self) -> Option<NodeId> {
        None
    }
}

impl ForwardInput for &Var {
    fn shape(&self) -> Vec<usize> {
        self.value_expected_shape()
    }

    fn get_value(&self) -> Result<Tensor, GraphError> {
        // 如果 Var 已有值，直接使用
        if let Ok(Some(value)) = self.value() {
            return Ok(value);
        }
        // 否则触发 forward 计算
        self.forward()?;
        self.value()?.ok_or_else(|| {
            GraphError::ComputationError("Var 计算后仍没有值".to_string())
        })
    }

    fn is_detached(&self) -> bool {
        Var::is_detached(self)
    }

    fn var_node_id(&self) -> Option<NodeId> {
        Some(self.node_id())
    }
}

impl ForwardInput for Var {
    fn shape(&self) -> Vec<usize> {
        self.value_expected_shape()
    }

    fn get_value(&self) -> Result<Tensor, GraphError> {
        // 如果 Var 已有值，直接使用
        if let Ok(Some(value)) = self.value() {
            return Ok(value);
        }
        // 否则触发 forward 计算
        self.forward()?;
        self.value()?.ok_or_else(|| {
            GraphError::ComputationError("Var 计算后仍没有值".to_string())
        })
    }

    fn is_detached(&self) -> bool {
        Var::is_detached(self)
    }

    fn var_node_id(&self) -> Option<NodeId> {
        Some(self.node_id())
    }
}

// ============================================================================
// DetachedVar 的 ForwardInput 实现
// ============================================================================

impl ForwardInput for &DetachedVar {
    fn shape(&self) -> Vec<usize> {
        self.value_expected_shape()
    }

    fn get_value(&self) -> Result<Tensor, GraphError> {
        // 如果内部 Var 已有值，直接使用
        if let Ok(Some(value)) = self.value() {
            return Ok(value);
        }
        // 否则触发 forward 计算
        self.forward()?;
        self.value()?.ok_or_else(|| {
            GraphError::ComputationError("DetachedVar 计算后仍没有值".to_string())
        })
    }

    fn is_detached(&self) -> bool {
        true // DetachedVar 始终是 detached
    }

    fn var_node_id(&self) -> Option<NodeId> {
        None // detached 不需要梯度路由
    }
}

impl ForwardInput for DetachedVar {
    fn shape(&self) -> Vec<usize> {
        self.value_expected_shape()
    }

    fn get_value(&self) -> Result<Tensor, GraphError> {
        (&self).get_value()
    }

    fn is_detached(&self) -> bool {
        true
    }

    fn var_node_id(&self) -> Option<NodeId> {
        None
    }
}

/// 模型状态缓存（单个形状）
struct StateCache {
    /// GradientRouter 节点（模型入口点）
    router: Var,
    /// 输出节点（预构建的计算图终点）
    output: Var,
}

/// 模型状态管理器
///
/// 使用 GradientRouter 实现"Archive 的效率 + PyTorch 的优雅"：
/// - 图结构只构建一次（按形状缓存）
/// - 无论多少批次，图节点数保持 O(1)
/// - 梯度通过 GradientRouter 自动路由
///
/// # 工作原理
/// - **首次调用**某形状：创建 GradientRouter，构建计算图，缓存结果
/// - **后续调用**相同形状：复用 GradientRouter，更新值和梯度路由设置
/// - **不同形状**：自动创建新的子图并缓存
pub struct ModelState {
    graph: Graph,
    /// 按输入形状缓存的子图：shape -> (router, output)
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

    /// PyTorch 风格的 forward（统一缓存 + 梯度路由）
    ///
    /// # 参数
    /// - `x`: 输入数据（`&Tensor`、`Tensor`、`&Var` 或 `Var`）
    /// - `compute`: 计算逻辑闭包，接收 `&Var` 输入，返回 `Result<Var, GraphError>`
    ///
    /// # 返回
    /// 模型输出节点（Var）
    ///
    /// # 示例
    /// ```ignore
    /// // 普通训练
    /// let out = model.forward(&batch_x)?;
    ///
    /// // GAN 训练
    /// let fake = G.forward(&noise)?;
    /// let d_fake = D.forward(&fake.detach())?;  // 复用结构，无梯度路由
    /// let d_fake_for_g = D.forward(&fake)?;     // 复用结构，梯度路由到 fake
    /// ```
    pub fn forward<X, F>(&self, x: X, compute: F) -> Result<Var, GraphError>
    where
        X: ForwardInput,
        F: FnOnce(&Var) -> Result<Var, GraphError>,
    {
        let shape = x.shape();
        let value = x.get_value()?;
        let is_detached = x.is_detached();
        let gradient_target = if is_detached {
            None
        } else {
            x.var_node_id()
        };

        let mut cache = self.cache.borrow_mut();

        if let Some(c) = cache.get(&shape) {
            // 缓存命中：复用已有结构
            
            // 1. 更新 GradientRouter 的值
            c.router.set_value(&value)?;

            // 2. 设置 GradientRouter 的 detached 状态
            self.graph
                .inner_mut()
                .set_router_detached(c.router.node_id(), is_detached)?;

            // 3. 设置梯度路由目标
            self.graph
                .inner_mut()
                .set_gradient_target(c.router.node_id(), gradient_target)?;

            // 4. 重新计算
            c.output.forward()?;

            return Ok(c.output.clone());
        }

        // 缓存未命中：创建新的子图

        // 1. 创建 GradientRouter 作为模型入口
        let router_id = self
            .graph
            .inner_mut()
            .new_gradient_router_node(&shape, None)?;
        let router = Var::new(router_id, self.graph.inner_rc());

        // 2. 设置初始值
        router.set_value(&value)?;

        // 3. 设置 detached 状态和梯度路由目标
        self.graph
            .inner_mut()
            .set_router_detached(router_id, is_detached)?;
        self.graph
            .inner_mut()
            .set_gradient_target(router_id, gradient_target)?;

        // 4. 调用用户提供的计算逻辑
        let output = compute(&router)?;

        // 5. 触发前向传播
        output.forward()?;

        // 6. 缓存
        cache.insert(
            shape,
            StateCache {
                router,
                output: output.clone(),
            },
        );

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
