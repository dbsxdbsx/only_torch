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
use std::any::Any;
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
    /// - `None`: 这个输入本身没有梯度流概念（如 Tensor），问它是否 detach 没有意义
    /// - `Some(true)`: 被显式 detach（如 `DetachedVar`）
    /// - `Some(false)`: 正常传播梯度（如 Var）
    fn is_detached(&self) -> Option<bool>;

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

    fn is_detached(&self) -> Option<bool> {
        None // Tensor 本身没有梯度流概念
    }

    fn var_node_id(&self) -> Option<NodeId> {
        None
    }
}

impl ForwardInput for Tensor {
    fn shape(&self) -> Vec<usize> {
        Self::shape(self).to_vec()
    }

    fn get_value(&self) -> Result<Tensor, GraphError> {
        Ok(self.clone())
    }

    fn is_detached(&self) -> Option<bool> {
        None // Tensor 本身没有梯度流概念
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
        self.value()?
            .ok_or_else(|| GraphError::ComputationError("Var 计算后仍没有值".to_string()))
    }

    fn is_detached(&self) -> Option<bool> {
        Some(false) // Var 正常传播梯度
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
        self.value()?
            .ok_or_else(|| GraphError::ComputationError("Var 计算后仍没有值".to_string()))
    }

    fn is_detached(&self) -> Option<bool> {
        Some(false) // Var 正常传播梯度
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
        self.value()?
            .ok_or_else(|| GraphError::ComputationError("DetachedVar 计算后仍没有值".to_string()))
    }

    fn is_detached(&self) -> Option<bool> {
        Some(true) // DetachedVar 是用户显式调用 .detach() 创建的
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

    fn is_detached(&self) -> Option<bool> {
        Some(true) // DetachedVar 是用户显式调用 .detach() 创建的
    }

    fn var_node_id(&self) -> Option<NodeId> {
        None
    }
}

// ============================================================================
// ForwardOutput Trait（支持多输出）
// ============================================================================

/// 模型前向输出类型
///
/// 实现此 trait 的类型可以作为 `ModelState::forward()` 的返回值。
/// 支持 `Var`、`(Var, Var)`、`(Var, Var, Var)` 等类型。
///
/// # 多输出支持
///
/// 用户可以在 forward 闭包中返回元组，实现多任务学习等场景：
///
/// ```ignore
/// // 返回单个输出
/// state.forward(&x, |input| Ok(self.fc.forward(input)))?;
///
/// // 返回双输出
/// state.forward(&x, |input| {
///     let feat = self.shared.forward(input);
///     Ok((self.head1.forward(&feat), self.head2.forward(&feat)))
/// })?;
/// ```
pub trait ForwardOutput: Clone + 'static {
    /// 触发前向传播
    ///
    /// 对于多输出，只需调用其中任意一个 Var 的 forward()，
    /// 整个计算图就会被执行。
    fn trigger_forward(&self) -> Result<(), GraphError>;

    /// 获取所有输出节点的 ID（用于可视化）
    fn output_node_ids(&self) -> Vec<NodeId>;
}

impl ForwardOutput for Var {
    fn trigger_forward(&self) -> Result<(), GraphError> {
        self.forward()
    }

    fn output_node_ids(&self) -> Vec<NodeId> {
        vec![self.node_id()]
    }
}

impl ForwardOutput for (Var, Var) {
    fn trigger_forward(&self) -> Result<(), GraphError> {
        // 两个输出可能位于不同分支，都需要触发前向传播
        self.0.forward()?;
        self.1.forward()
    }

    fn output_node_ids(&self) -> Vec<NodeId> {
        vec![self.0.node_id(), self.1.node_id()]
    }
}

impl ForwardOutput for (Var, Var, Var) {
    fn trigger_forward(&self) -> Result<(), GraphError> {
        self.0.forward()?;
        self.1.forward()?;
        self.2.forward()
    }

    fn output_node_ids(&self) -> Vec<NodeId> {
        vec![self.0.node_id(), self.1.node_id(), self.2.node_id()]
    }
}

impl ForwardOutput for (Var, Var, Var, Var) {
    fn trigger_forward(&self) -> Result<(), GraphError> {
        self.0.forward()?;
        self.1.forward()?;
        self.2.forward()?;
        self.3.forward()
    }

    fn output_node_ids(&self) -> Vec<NodeId> {
        vec![
            self.0.node_id(),
            self.1.node_id(),
            self.2.node_id(),
            self.3.node_id(),
        ]
    }
}

/// 模型状态缓存
struct StateCache {
    /// `GradientRouter` 节点列表（每个输入一个，为多输入扩展准备）
    routers: Vec<Var>,
    /// 输出（使用 `Any` 支持任意返回类型，如 `Var`、`(Var, Var)` 等）
    output: Box<dyn Any>,
}

/// 模型状态管理器
///
/// 使用 `GradientRouter` 实现"Archive 的效率 + `PyTorch` 的优雅"：
/// - 图结构只构建一次（按特征形状缓存，忽略 batch 维度）
/// - 无论多少批次、多大 `batch_size，图节点数保持` O(1)
/// - 梯度通过 `GradientRouter` 自动路由
///
/// # 工作原理
/// - **首次调用**某特征形状：创建 GradientRouter，构建计算图，缓存结果
/// - **后续调用**相同特征形状（不同 `batch_size` 也复用）：更新值和梯度路由设置
/// - **不同特征形状**：自动创建新的子图并缓存
///
/// # Batch 维度处理（类似 Keras）
/// - 缓存键只用特征维度（忽略第一维 batch）
/// - `[256, 64]` 和 `[1, 64]` 复用同一个缓存
/// - 可视化时 batch 维度显示为 `?`
///
/// # 可视化分组
/// 可以通过 `named()` 或 `new_for::<T>()` 为模型指定名称，
/// 可视化时会自动将该模型的节点框在一起显示。
pub struct ModelState {
    graph: Graph,
    /// 按特征形状缓存的子图：`feature_shapes` -> (routers, output)
    ///
    /// 缓存键是所有输入的特征形状列表（不包含 batch 维度）。
    /// - 单输入：`[[64]]` 表示特征维度为 64
    /// - 双输入：`[[64], [32]]` 表示两个输入的特征维度分别为 64 和 32
    cache: RefCell<HashMap<Vec<Vec<usize>>, StateCache>>,
    /// 模型名称（用于可视化分组）
    name: Option<String>,
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
            name: None,
        }
    }

    /// 创建带自动类型名的模型状态管理器
    ///
    /// 使用 Rust 反射自动获取类型名称作为模型名称，
    /// 可视化时会将该模型的节点框在一起显示。
    ///
    /// # 示例
    /// ```ignore
    /// impl Generator {
    ///     pub fn new(graph: &Graph) -> Self {
    ///         Self {
    ///             // ... layers ...
    ///             state: ModelState::new_for::<Self>(graph), // 自动使用 "Generator"
    ///         }
    ///     }
    /// }
    /// ```
    pub fn new_for<T: 'static>(graph: &Graph) -> Self {
        // 从完整类型路径提取短名称
        // 例如："mnist_gan::model::Generator" -> "Generator"
        let full_name = std::any::type_name::<T>();
        let short_name = full_name.rsplit("::").next().unwrap_or(full_name);
        Self {
            graph: graph.clone(),
            cache: RefCell::new(HashMap::new()),
            name: Some(short_name.to_string()),
        }
    }

    /// 设置模型名称（用于可视化分组）
    ///
    /// 可视化时会将该模型的节点框在一起显示。
    ///
    /// # 示例
    /// ```ignore
    /// state: ModelState::new(graph).named("Discriminator"),
    /// ```
    pub fn named(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// 获取模型名称
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// `PyTorch` 风格的 forward（统一缓存 + 梯度路由 + 多输出支持）
    ///
    /// # 参数
    /// - `x`: 输入数据（`&Tensor`、`Tensor`、`&Var` 或 `Var`）
    /// - `compute`: 计算逻辑闭包，接收 `&Var` 输入，返回实现 `ForwardOutput` 的类型
    ///
    /// # 返回
    /// 模型输出（可以是 `Var`、`(Var, Var)` 等实现了 `ForwardOutput` 的类型）
    ///
    /// # Batch 维度处理
    /// 缓存键只用特征维度（忽略第一维 batch），所以：
    /// - `[256, 64]` 和 `[1, 64]` 复用同一个缓存
    /// - 可视化时 batch 维度显示为 `?`
    ///
    /// # 示例
    /// ```ignore
    /// // 单输出
    /// let out = model.forward(&batch_x, |input| {
    ///     Ok(self.fc.forward(input))
    /// })?;
    ///
    /// // 多输出（多任务学习）
    /// let (cls, reg) = model.forward(&batch_x, |input| {
    ///     let feat = self.shared.forward(input);
    ///     Ok((self.cls_head.forward(&feat), self.reg_head.forward(&feat)))
    /// })?;
    ///
    /// // GAN 训练
    /// let fake = G.forward(&noise, |z| Ok(self.gen.forward(z)))?;
    /// let d_fake = D.forward(&fake.detach(), |x| Ok(self.disc.forward(x)))?;
    /// ```
    pub fn forward<X, R, F>(&self, x: X, compute: F) -> Result<R, GraphError>
    where
        X: ForwardInput,
        R: ForwardOutput,
        F: FnOnce(&Var) -> Result<R, GraphError>,
    {
        let full_shape = x.shape();
        let value = x.get_value()?;
        let detach_status = x.is_detached();
        // 只有 Some(false)（正常 Var）才需要梯度路由
        let gradient_target = if detach_status == Some(false) {
            x.var_node_id()
        } else {
            None
        };
        // 用于 set_router_detached：None 和 Some(true) 都阻止梯度传播
        let should_detach = detach_status != Some(false);

        // 缓存键策略：统一使用特征维度（忽略第一维 batch）
        //
        // 由于 GradientRouter 和 State 节点都支持 DynamicShape（动态 batch），
        // 不同 batch_size 的输入可以复用同一个缓存结构。
        //
        // 例如：[256, 64] 和 [1, 64] 都使用 [[64]] 作为缓存键
        let feature_shape = if full_shape.len() > 1 {
            full_shape[1..].to_vec()
        } else {
            full_shape.clone()
        };
        // 单输入时，缓存键是包含单个元素的列表
        let cache_key = vec![feature_shape];

        let mut cache = self.cache.borrow_mut();

        if let Some(c) = cache.get(&cache_key) {
            // 缓存命中：复用已有结构
            let router = &c.routers[0];

            // 1. 更新 GradientRouter 的值（可能 batch_size 不同）
            router.set_value(&value)?;

            // 2. 设置 GradientRouter 的 detached 状态
            // detach_status == Some(true) 表示显式 detach，需要标记 was_ever_detached
            self.graph.inner_mut().set_router_detached(
                router.node_id(),
                should_detach,
                detach_status == Some(true),
            )?;

            // 3. 设置梯度路由目标
            self.graph
                .inner_mut()
                .set_gradient_target(router.node_id(), gradient_target)?;

            // 4. 从缓存取出输出并 downcast
            let cached_output = c
                .output
                .downcast_ref::<R>()
                .ok_or_else(|| {
                    GraphError::ComputationError(
                        "缓存输出类型不匹配（同一 ModelState 应返回相同类型）".to_string(),
                    )
                })?
                .clone();

            // 5. 重新计算（使用新的 batch_size）
            cached_output.trigger_forward()?;

            return Ok(cached_output);
        }

        // 缓存未命中：创建新的子图

        // 1. 创建 GradientRouter 作为模型入口
        // 使用完整形状创建（包含首次调用时的 batch_size），
        // 但缓存键使用特征形状，所以不同 batch_size 会复用同一个 SmartInput
        let router_id = self
            .graph
            .inner_mut()
            .new_smart_input_node(&full_shape, None)?;
        let router = Var::new(router_id, self.graph.inner_rc());

        // 2. 设置初始值（包含实际的 batch_size）
        router.set_value(&value)?;

        // 3. 设置 detached 状态和梯度路由目标
        // detach_status == Some(true) 表示显式 detach，需要标记 was_ever_detached
        self.graph.inner_mut().set_router_detached(
            router_id,
            should_detach,
            detach_status == Some(true),
        )?;
        self.graph
            .inner_mut()
            .set_gradient_target(router_id, gradient_target)?;

        // 4. 调用用户提供的计算逻辑
        let output = compute(&router)?;

        // 5. 触发前向传播
        output.trigger_forward()?;

        // 6. 注册模型分组（如果有名称）
        // 多输出时注册第一个输出节点作为代表
        if let Some(ref name) = self.name {
            let output_ids = output.output_node_ids();
            if let Some(&first_output_id) = output_ids.first() {
                self.graph
                    .inner_mut()
                    .register_model_group(name, &[router_id], first_output_id)?;
            }
        }

        // 7. 缓存（使用特征形状列表作为键）
        cache.insert(
            cache_key,
            StateCache {
                routers: vec![router],
                output: Box::new(output.clone()),
            },
        );

        Ok(output)
    }

    /// 双输入 forward（PyTorch 风格，支持多输出）
    ///
    /// # 参数
    /// - `x`: 第一个输入（`&Tensor`、`Tensor`、`&Var`、`Var` 或 `DetachedVar`）
    /// - `y`: 第二个输入
    /// - `compute`: 计算逻辑闭包，接收两个 `&Var` 输入，返回实现 `ForwardOutput` 的类型
    ///
    /// # 示例
    /// ```ignore
    /// // 多模态融合（单输出）
    /// let out = model.forward2(&image, &text, |img, txt| {
    ///     let img_feat = self.image_encoder.forward(img);
    ///     let txt_feat = self.text_encoder.forward(txt);
    ///     Ok(self.fusion.forward(&Var::concat(&[&img_feat, &txt_feat], 1)?))
    /// })?;
    ///
    /// // Siamese 网络（双输出）
    /// let (feat1, feat2) = model.forward2(&x1, &x2, |a, b| {
    ///     Ok((self.encoder.forward(a), self.encoder.forward(b)))
    /// })?;
    /// ```
    pub fn forward2<X, Y, R, F>(&self, x: X, y: Y, compute: F) -> Result<R, GraphError>
    where
        X: ForwardInput,
        Y: ForwardInput,
        R: ForwardOutput,
        F: FnOnce(&Var, &Var) -> Result<R, GraphError>,
    {
        // 收集两个输入的信息
        let x_shape = x.shape();
        let x_value = x.get_value()?;
        let x_detach_status = x.is_detached();
        let x_gradient_target = if x_detach_status == Some(false) {
            x.var_node_id()
        } else {
            None
        };
        let x_should_detach = x_detach_status != Some(false);

        let y_shape = y.shape();
        let y_value = y.get_value()?;
        let y_detach_status = y.is_detached();
        let y_gradient_target = if y_detach_status == Some(false) {
            y.var_node_id()
        } else {
            None
        };
        let y_should_detach = y_detach_status != Some(false);

        // 构建缓存键：两个输入的特征形状
        let x_feature = if x_shape.len() > 1 {
            x_shape[1..].to_vec()
        } else {
            x_shape.clone()
        };
        let y_feature = if y_shape.len() > 1 {
            y_shape[1..].to_vec()
        } else {
            y_shape.clone()
        };
        let cache_key = vec![x_feature, y_feature];

        let mut cache = self.cache.borrow_mut();

        if let Some(c) = cache.get(&cache_key) {
            // 缓存命中：复用已有结构
            let router_x = &c.routers[0];
            let router_y = &c.routers[1];

            // 更新第一个输入
            router_x.set_value(&x_value)?;
            self.graph.inner_mut().set_router_detached(
                router_x.node_id(),
                x_should_detach,
                x_detach_status == Some(true),
            )?;
            self.graph
                .inner_mut()
                .set_gradient_target(router_x.node_id(), x_gradient_target)?;

            // 更新第二个输入
            router_y.set_value(&y_value)?;
            self.graph.inner_mut().set_router_detached(
                router_y.node_id(),
                y_should_detach,
                y_detach_status == Some(true),
            )?;
            self.graph
                .inner_mut()
                .set_gradient_target(router_y.node_id(), y_gradient_target)?;

            // 从缓存取出输出并 downcast
            let cached_output = c
                .output
                .downcast_ref::<R>()
                .ok_or_else(|| {
                    GraphError::ComputationError(
                        "缓存输出类型不匹配（同一 ModelState 应返回相同类型）".to_string(),
                    )
                })?
                .clone();

            // 重新计算
            cached_output.trigger_forward()?;
            return Ok(cached_output);
        }

        // 缓存未命中：创建新的子图

        // 创建两个 GradientRouter
        let router_x_id = self
            .graph
            .inner_mut()
            .new_smart_input_node(&x_shape, None)?;
        let router_x = Var::new(router_x_id, self.graph.inner_rc());
        router_x.set_value(&x_value)?;
        self.graph.inner_mut().set_router_detached(
            router_x_id,
            x_should_detach,
            x_detach_status == Some(true),
        )?;
        self.graph
            .inner_mut()
            .set_gradient_target(router_x_id, x_gradient_target)?;

        let router_y_id = self
            .graph
            .inner_mut()
            .new_smart_input_node(&y_shape, None)?;
        let router_y = Var::new(router_y_id, self.graph.inner_rc());
        router_y.set_value(&y_value)?;
        self.graph.inner_mut().set_router_detached(
            router_y_id,
            y_should_detach,
            y_detach_status == Some(true),
        )?;
        self.graph
            .inner_mut()
            .set_gradient_target(router_y_id, y_gradient_target)?;

        // 调用用户计算逻辑
        let output = compute(&router_x, &router_y)?;
        output.trigger_forward()?;

        // 注册模型分组（多输出时使用第一个输出节点）
        if let Some(ref name) = self.name {
            let output_ids = output.output_node_ids();
            if let Some(&first_output_id) = output_ids.first() {
                self.graph.inner_mut().register_model_group(
                    name,
                    &[router_x_id, router_y_id],
                    first_output_id,
                )?;
            }
        }

        // 缓存
        cache.insert(
            cache_key,
            StateCache {
                routers: vec![router_x, router_y],
                output: Box::new(output.clone()),
            },
        );

        Ok(output)
    }

    /// 三输入 forward（PyTorch 风格，支持多输出）
    ///
    /// # 参数
    /// - `x`, `y`, `z`: 三个输入
    /// - `compute`: 计算逻辑闭包，接收三个 `&Var` 输入，返回实现 `ForwardOutput` 的类型
    ///
    /// # 示例
    /// ```ignore
    /// // 三模态融合（单输出）
    /// let out = model.forward3(&audio, &video, &text, |a, v, t| {
    ///     let combined = Var::stack(&[a, v, t], 1, StackMode::Concat)?;
    ///     Ok(self.fusion.forward(&combined))
    /// })?;
    ///
    /// // 三输入多输出
    /// let (out1, out2) = model.forward3(&x, &y, &z, |a, b, c| {
    ///     let feat = self.encoder.forward3(a, b, c);
    ///     Ok((self.head1.forward(&feat), self.head2.forward(&feat)))
    /// })?;
    /// ```
    pub fn forward3<X, Y, Z, R, F>(&self, x: X, y: Y, z: Z, compute: F) -> Result<R, GraphError>
    where
        X: ForwardInput,
        Y: ForwardInput,
        Z: ForwardInput,
        R: ForwardOutput,
        F: FnOnce(&Var, &Var, &Var) -> Result<R, GraphError>,
    {
        // 收集三个输入的信息
        let inputs_info: Vec<_> = [
            (&x as &dyn ForwardInput, x.shape(), x.get_value()?),
            (&y as &dyn ForwardInput, y.shape(), y.get_value()?),
            (&z as &dyn ForwardInput, z.shape(), z.get_value()?),
        ]
        .into_iter()
        .map(|(input, shape, value)| {
            let detach_status = input.is_detached();
            let gradient_target = if detach_status == Some(false) {
                input.var_node_id()
            } else {
                None
            };
            let should_detach = detach_status != Some(false);
            let feature = if shape.len() > 1 {
                shape[1..].to_vec()
            } else {
                shape.clone()
            };
            (
                shape,
                value,
                detach_status,
                gradient_target,
                should_detach,
                feature,
            )
        })
        .collect();

        // 构建缓存键
        let cache_key: Vec<Vec<usize>> = inputs_info.iter().map(|i| i.5.clone()).collect();

        let mut cache = self.cache.borrow_mut();

        if let Some(c) = cache.get(&cache_key) {
            // 缓存命中
            for (i, router) in c.routers.iter().enumerate() {
                let (_, ref value, detach_status, gradient_target, should_detach, _) =
                    inputs_info[i];
                router.set_value(value)?;
                self.graph.inner_mut().set_router_detached(
                    router.node_id(),
                    should_detach,
                    detach_status == Some(true),
                )?;
                self.graph
                    .inner_mut()
                    .set_gradient_target(router.node_id(), gradient_target)?;
            }

            // 从缓存取出输出并 downcast
            let cached_output = c
                .output
                .downcast_ref::<R>()
                .ok_or_else(|| {
                    GraphError::ComputationError(
                        "缓存输出类型不匹配（同一 ModelState 应返回相同类型）".to_string(),
                    )
                })?
                .clone();

            cached_output.trigger_forward()?;
            return Ok(cached_output);
        }

        // 缓存未命中：创建三个 GradientRouter
        let mut routers = Vec::with_capacity(3);
        let mut router_ids = Vec::with_capacity(3);

        for (shape, value, detach_status, gradient_target, should_detach, _) in &inputs_info {
            let router_id = self.graph.inner_mut().new_smart_input_node(shape, None)?;
            let router = Var::new(router_id, self.graph.inner_rc());
            router.set_value(value)?;
            self.graph.inner_mut().set_router_detached(
                router_id,
                *should_detach,
                *detach_status == Some(true),
            )?;
            self.graph
                .inner_mut()
                .set_gradient_target(router_id, *gradient_target)?;
            router_ids.push(router_id);
            routers.push(router);
        }

        // 调用用户计算逻辑
        let output = compute(&routers[0], &routers[1], &routers[2])?;
        output.trigger_forward()?;

        // 注册模型分组（多输出时使用第一个输出节点）
        if let Some(ref name) = self.name {
            let output_ids = output.output_node_ids();
            if let Some(&first_output_id) = output_ids.first() {
                self.graph
                    .inner_mut()
                    .register_model_group(name, &router_ids, first_output_id)?;
            }
        }

        // 缓存
        cache.insert(
            cache_key,
            StateCache {
                routers,
                output: Box::new(output.clone()),
            },
        );

        Ok(output)
    }

    /// 获取当前缓存的特征形状列表
    ///
    /// 返回的是特征形状列表（不包含 batch 维度）。
    /// - 单输入模型：`[[[64]]]` 表示一个缓存，输入特征维度为 64
    /// - 双输入模型：`[[[64], [32]]]` 表示两个输入特征维度分别为 64 和 32
    pub fn cached_shapes(&self) -> Vec<Vec<Vec<usize>>> {
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
