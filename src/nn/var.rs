/*
 * @Author       : 老董
 * @Date         : 2026-01-08
 * @Description  : Smart Var - 智能变量句柄，支持算子重载和链式调用
 *
 * 这是架构的核心组件，提供 PyTorch 级用户体验。
 */

use super::graph::GraphInner;
use super::nodes::NodeInner;
use super::{GraphError, NodeId};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

// ==================== Init 枚举 ====================

/// 参数初始化策略
#[derive(Debug, Clone)]
pub enum Init {
    /// 常数初始化
    Constant(f32),
    /// 全零
    Zeros,
    /// 全一
    Ones,
    /// 正态分布（使用 Graph 的 RNG）
    Normal { mean: f32, std: f32 },
    /// Kaiming/He 初始化（适用于 `ReLU`）
    Kaiming,
    /// Xavier/Glorot 初始化（适用于 Sigmoid/Tanh）
    Xavier,
}

impl Init {
    /// 生成初始化后的 Tensor（使用全局 RNG）
    pub fn generate(&self, shape: &[usize]) -> Tensor {
        match self {
            Self::Constant(v) => &Tensor::ones(shape) * *v,
            Self::Zeros => Tensor::zeros(shape),
            Self::Ones => Tensor::ones(shape),
            Self::Normal { mean, std } => Tensor::normal(*mean, *std, shape),
            Self::Kaiming => {
                let fan_in = shape[0];
                let std = (2.0 / fan_in as f32).sqrt();
                Tensor::normal(0.0, std, shape)
            }
            Self::Xavier => {
                let (fan_in, fan_out) = (shape[0], shape.get(1).copied().unwrap_or(1));
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                Tensor::normal(0.0, std, shape)
            }
        }
    }

    /// 生成初始化后的 Tensor（使用指定的 RNG）
    pub fn generate_with_rng(&self, shape: &[usize], rng: &mut rand::rngs::StdRng) -> Tensor {
        match self {
            Self::Constant(v) => &Tensor::ones(shape) * *v,
            Self::Zeros => Tensor::zeros(shape),
            Self::Ones => Tensor::ones(shape),
            Self::Normal { mean, std } => Tensor::normal_with_rng(*mean, *std, shape, rng),
            Self::Kaiming => {
                let fan_in = shape[0];
                let std = (2.0 / fan_in as f32).sqrt();
                Tensor::normal_with_rng(0.0, std, shape, rng)
            }
            Self::Xavier => {
                let (fan_in, fan_out) = (shape[0], shape.get(1).copied().unwrap_or(1));
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                Tensor::normal_with_rng(0.0, std, shape, rng)
            }
        }
    }
}

// ==================== Var 结构 ====================

/// 智能变量句柄 - 携带图引用，支持算子重载和链式调用
///
/// # 设计原则（方案 C 过渡版本）
/// - 持有 `Rc<NodeInner>` 直接控制节点生命周期（新）
/// - 持有 `Rc<RefCell<GraphInner>>` 引用用于全局配置（过渡期）
/// - 用户无需关心内部实现，像 `PyTorch` tensor 一样使用
/// - Clone 语义（非 Copy），但开销极低（Rc clone）
///
/// # 使用示例
/// ```ignore
/// let graph = Graph::new();
/// let x = graph.input(&images)?;      // 返回 Var
/// let h = x.relu();                   // 链式调用
/// let y = h.matmul(&w)?;              // 方法调用
/// let z = &y + &b;                    // 算子重载
/// let loss = z.cross_entropy(&target)?;
/// loss.backward()?;                   // 直接在 Var 上调用
/// ```
#[derive(Clone)]
pub struct Var {
    /// 节点 ID（过渡期保留，用于旧路径访问）
    id: NodeId,
    /// 节点内部结构（方案 C 新增，优先使用）
    /// 过渡期为 Option，最终会变为必须字段
    node: Option<Rc<NodeInner>>,
    /// 图引用（过渡期仍为 Rc，最终会变为 Weak）
    graph: Rc<RefCell<GraphInner>>,
}

impl std::fmt::Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = self.node_id();
        f.debug_struct("Var")
            .field("id", &id)
            .field("has_node", &self.node.is_some())
            .finish()
    }
}

impl Var {
    /// 创建新的 Var（内部使用，旧路径 - 过渡期兼容）
    pub(crate) fn new(id: NodeId, graph: Rc<RefCell<GraphInner>>) -> Self {
        Self {
            id,
            node: None, // 过渡期：旧代码创建的 Var 没有 node
            graph,
        }
    }

    /// 创建新的 Var（新路径 - 方案 C）
    ///
    /// 直接持有 NodeInner，id 从 node 获取
    pub(crate) fn new_with_node(node: Rc<NodeInner>, graph: Rc<RefCell<GraphInner>>) -> Self {
        let id = node.id();
        Self {
            id,
            node: Some(node),
            graph,
        }
    }

    /// 获取节点 ID
    ///
    /// 优先从 `self.node` 获取，回退到 `self.id`
    pub fn node_id(&self) -> NodeId {
        if let Some(ref node) = self.node {
            node.id()
        } else {
            self.id
        }
    }

    /// 获取 NodeInner 的引用（方案 C 新增）
    ///
    /// 过渡期可能返回 None
    pub(crate) fn node(&self) -> Option<&Rc<NodeInner>> {
        self.node.as_ref()
    }

    /// 获取内部图引用（供 trait 和内部模块使用）
    pub(crate) fn graph(&self) -> &Rc<RefCell<GraphInner>> {
        &self.graph
    }

    /// 检查两个 Var 是否来自同一个 Graph
    pub fn same_graph(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.graph, &other.graph)
    }

    /// 获取 Var 所属的 Graph handle
    ///
    /// 即使原始 Graph handle 已 drop，此方法仍返回有效的 Graph。
    /// 这是因为 Var 持有 `GraphInner` 的强引用（Rc）。
    pub fn get_graph(&self) -> super::graph::Graph {
        super::graph::Graph::from_rc(Rc::clone(&self.graph))
    }

    /// 获取节点的预期输出形状
    ///
    /// 这个形状在节点创建时就已确定。
    /// 优先从 `self.node` 获取，回退到旧路径
    pub fn value_expected_shape(&self) -> Vec<usize> {
        if let Some(ref node) = self.node {
            // 方案 C 新路径：直接从 NodeInner 获取
            node.shape()
        } else {
            // 旧路径：通过 GraphInner 获取
            self.graph
                .borrow()
                .get_node_value_expected_shape(self.id)
                .expect("获取形状失败")
                .to_vec()
        }
    }

    /// 获取节点的动态形状
    ///
    /// 返回支持动态维度的形状表示（如 `[?, 128]`）
    /// 注意：方案 C 过渡期暂不支持从 NodeInner 获取动态形状，始终使用旧路径
    pub fn dynamic_expected_shape(&self) -> crate::nn::shape::DynamicShape {
        self.graph
            .borrow()
            .get_node_dynamic_expected_shape(self.node_id())
            .expect("获取动态形状失败")
    }

    /// 断言两个 Var 来自同一个 Graph，否则 panic（供 trait 使用）
    pub(crate) fn assert_same_graph(&self, other: &Self) {
        assert!(
            self.same_graph(other),
            "不能对来自不同 Graph 的 Var 进行操作"
        );
    }

    // ==================== 梯度流控制 ====================

    /// 创建一个 detached 的副本（函数式 detach）
    ///
    /// 返回一个新的 Var，它是当前节点的 Identity 副本，但梯度流被阻断。
    /// 原节点**不受影响**。
    ///
    /// # 语义（与 `PyTorch` 一致）
    /// - 返回的 Var 与原 Var 共享前向计算的值
    /// - 但反向传播时，梯度不会通过返回的 Var 传递到原 Var
    ///
    /// # 示例
    /// ```ignore
    /// // GAN 训练：训练 D 时阻止梯度流向 G
    /// let fake_images = G.forward(&noise)?;
    /// let fake_detached = fake_images.detach();  // 新 Var，阻断梯度
    /// let d_fake = D.forward(&fake_detached)?;
    /// d_loss.backward()?;  // 梯度不会流向 G
    ///
    /// // 训练 G 时正常使用
    /// let d_fake_for_g = D.forward(&fake_images)?;  // 原 Var，梯度正常流动
    /// g_loss.backward()?;  // 梯度流向 G
    /// ```
    /// 创建一个 detached 视图（轻量级包装，不创建图节点）
    ///
    /// 返回 `DetachedVar`，它是原 Var 的轻量级包装：
    /// - **不创建任何图节点**
    /// - 实现 `ForwardInput` trait，行为像 detached Var
    /// - 用于 GAN 训练等需要梯度截断的场景
    ///
    /// # 与 `detach_node()` 的区别
    /// - `detach()` → 返回 `DetachedVar`（轻量级，推荐用于 `ModelState`）
    /// - `detach_node()` → 返回 `Var`（创建 Identity 节点，用于直接图操作）
    ///
    /// # 示例
    /// ```ignore
    /// // GAN 训练（推荐）
    /// let fake = G.forward(&noise)?;
    /// let d_fake = D.forward(&fake.detach())?;  // DetachedVar，无图节点创建
    /// ```
    pub fn detach(&self) -> DetachedVar {
        DetachedVar {
            inner: self.clone(),
        }
    }

    /// 创建一个 detached 节点（创建 Identity 节点）
    ///
    /// 在图中创建一个新的 Identity 节点，标记为 detached。
    /// 返回指向该节点的 Var。
    ///
    /// # 何时使用
    /// - 需要在图中保留明确的 detach 边界
    /// - 需要对 detached 结果进行进一步的图操作
    ///
    /// # 注意
    /// 对于 `ModelState` 场景，推荐使用 `detach()` 方法（更高效）。
    ///
    /// # 示例
    /// ```ignore
    /// // 需要在图中操作 detached 结果
    /// let x_detached = x.detach_node();
    /// let y = x_detached.sigmoid();  // 可以继续构建图
    /// ```
    pub fn detach_node(&self) -> Self {
        if let Some(ref node) = self.node {
            // 方案 C 新路径：使用 create_identity_node
            let new_node = self
                .graph
                .borrow_mut()
                .create_identity_node(node.clone(), None, true) // detached=true
                .expect("内部错误：detach_node 创建 Identity 节点失败");
            Self {
                id: new_node.id(),
                node: Some(new_node),
                graph: self.graph.clone(),
            }
        } else {
            // 旧路径回退（过渡期）
            let new_id = self
                .graph
                .borrow_mut()
                .new_identity_node(self.node_id(), None, true)
                .expect("内部错误：detach_node 创建 Identity 节点失败");
            Self {
                id: new_id,
                node: None,
                graph: self.graph.clone(),
            }
        }
    }

    /// 检查此 Var 对应的节点是否处于 detached 状态
    ///
    /// detached 节点在反向传播时不会传递梯度给其父节点。
    ///
    /// # 用途
    /// - `ModelState` 使用此方法判断 Var 输入是否可以缓存
    /// - detached Var 只需要值，不需要梯度流，因此可以像 Tensor 一样缓存
    ///
    /// 优先从 `self.node` 获取，回退到旧路径
    pub fn is_detached(&self) -> bool {
        if let Some(ref node) = self.node {
            // 方案 C 新路径：直接从 NodeInner 获取
            node.is_detached()
        } else {
            // 旧路径：通过 GraphInner 获取
            self.graph
                .borrow()
                .is_node_detached(self.id)
                .unwrap_or(false)
        }
    }

    // ==================== 执行 ====================

    /// 前向传播
    ///
    /// 前向传播
    ///
    /// 过渡期：优先使用新的 NodeInner 路径，回退到旧路径
    pub fn forward(&self) -> Result<(), GraphError> {
        if let Some(ref node) = self.node {
            // 方案 C 新路径：通过 NodeInner 递归前向传播
            self.graph.borrow_mut().forward_via_node_inner(node)
        } else {
            // 旧路径：通过 GraphInner 前向传播
            self.graph.borrow_mut().forward(self.node_id())
        }
    }

    /// 反向传播（ensure-forward 语义）
    ///
    /// # 语义：ensure-forward
    /// - 自动先执行 forward()，确保 loss 值已计算
    /// - 然后执行反向传播
    ///
    /// # 返回值
    /// 返回 loss 的标量值
    pub fn backward(&self) -> Result<f32, GraphError> {
        self.backward_ex(false)
    }

    /// 反向传播（扩展版本，支持 `retain_graph`）
    ///
    /// # 参数
    /// - `retain_graph`: 是否保留计算图
    ///   - `true`: 保留图，允许多次 backward（多任务学习场景）
    ///   - `false`: 释放中间节点的值（默认行为，节省内存）
    ///
    /// # 多任务学习示例
    /// ```ignore
    /// optimizer.zero_grad()?;
    /// loss1.backward_ex(true)?;   // retain_graph=true，保留图
    /// loss2.backward_ex(false)?;  // 梯度累积到共享参数
    /// optimizer.step()?;
    /// ```
    ///
    /// # 返回值
    /// 返回 loss 的标量值
    ///
    /// 注意：过渡期仍使用 GraphInner 的反向传播逻辑
    /// 方案 C 完成后将改为基于 parents 遍历
    pub fn backward_ex(&self, retain_graph: bool) -> Result<f32, GraphError> {
        let mut g = self.graph.borrow_mut();
        let id = self.node_id();
        // ensure-forward：先执行前向传播
        g.forward(id)?;
        // 然后执行反向传播
        g.backward_ex(id, retain_graph)
    }

    // ==================== 值访问和设置 ====================

    /// 获取节点的值（克隆的 Tensor）
    ///
    /// 优先从 `self.node` 获取，回退到旧路径
    pub fn value(&self) -> Result<Option<Tensor>, GraphError> {
        if let Some(ref node) = self.node {
            // 方案 C 新路径：直接从 NodeInner 获取
            Ok(node.value())
        } else {
            // 旧路径：通过 GraphInner 获取
            Ok(self.graph.borrow().get_node_value(self.id)?.cloned())
        }
    }

    /// 设置节点的值
    ///
    /// 优先使用 `self.node`，回退到旧路径
    pub fn set_value(&self, value: &Tensor) -> Result<(), GraphError> {
        if let Some(ref node) = self.node {
            // 方案 C 新路径：直接设置 NodeInner
            node.set_value(Some(value))
        } else {
            // 旧路径：通过 GraphInner 设置
            self.graph.borrow_mut().set_node_value(self.id, Some(value))
        }
    }

    /// 获取标量值（假设是 1x1 Tensor）
    pub fn item(&self) -> Result<f32, GraphError> {
        let val = self.value()?.ok_or(GraphError::NodeNotFound(self.node_id()))?;
        val.get_data_number()
            .ok_or_else(|| GraphError::InvalidOperation("Tensor 不是标量".to_string()))
    }

    /// 获取节点的梯度
    ///
    /// 优先从 `self.node` 获取，回退到旧路径
    pub fn grad(&self) -> Result<Option<Tensor>, GraphError> {
        if let Some(ref node) = self.node {
            // 方案 C 新路径：直接从 NodeInner 获取
            Ok(node.grad())
        } else {
            // 旧路径：通过 GraphInner 获取
            self.graph.borrow().get_node_grad(self.id)
        }
    }

    // ==================== 安全版本（返回 Result）====================

    /// 安全的加法（返回 Result）
    pub fn try_add(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行加法".to_string(),
            ));
        }
        let id = self
            .graph
            .borrow_mut()
            .new_add_node(&[self.node_id(), other.node_id()], None)?;
        Ok(Self::new(id, Rc::clone(&self.graph)))
    }

    /// 安全的减法（返回 Result）
    ///
    /// 使用 Subtract 节点实现，支持广播
    pub fn try_sub(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行减法".to_string(),
            ));
        }
        let mut g = self.graph.borrow_mut();
        let id = g.new_subtract_node(self.node_id(), other.node_id(), None)?;
        Ok(Self::new(id, Rc::clone(&self.graph)))
    }

    /// 安全的元素级乘法（返回 Result）
    pub fn try_mul(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行乘法".to_string(),
            ));
        }
        let id = self
            .graph
            .borrow_mut()
            .new_multiply_node(self.node_id(), other.node_id(), None)?;
        Ok(Self::new(id, Rc::clone(&self.graph)))
    }

    /// 安全的除法（返回 Result）
    ///
    /// 逐元素除法：`self / other`
    pub fn try_div(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行除法".to_string(),
            ));
        }
        let id = self
            .graph
            .borrow_mut()
            .new_divide_node(self.node_id(), other.node_id(), None)?;
        Ok(Self::new(id, Rc::clone(&self.graph)))
    }
}

// ==================== 算子重载 ====================

// Add for &Var
impl Add for &Var {
    type Output = Var;

    fn add(self, other: &Var) -> Var {
        self.try_add(other).expect("Var 加法失败")
    }
}

// Add for Var (consumes self)
impl Add for Var {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        &self + &other
    }
}

// Add<Var> for &Var
impl Add<Var> for &Var {
    type Output = Var;

    fn add(self, other: Var) -> Var {
        self + &other
    }
}

// Add<&Var> for Var
impl Add<&Self> for Var {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        &self + other
    }
}

// Sub for &Var (实现为 self + (-1 * other))
impl Sub for &Var {
    type Output = Var;

    fn sub(self, other: &Var) -> Var {
        self.try_sub(other).expect("Var 减法失败")
    }
}

// Sub for Var
impl Sub for Var {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        &self - &other
    }
}

// Sub<Var> for &Var
impl Sub<Var> for &Var {
    type Output = Var;

    fn sub(self, other: Var) -> Var {
        self - &other
    }
}

// Sub<&Var> for Var
impl Sub<&Self> for Var {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        &self - other
    }
}

// Mul for &Var（逐元素乘法）
impl Mul for &Var {
    type Output = Var;

    fn mul(self, other: &Var) -> Var {
        self.try_mul(other).expect("Var 乘法失败")
    }
}

// Mul for Var
impl Mul for Var {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        &self * &other
    }
}

// Mul<Var> for &Var
impl Mul<Var> for &Var {
    type Output = Var;

    fn mul(self, other: Var) -> Var {
        self * &other
    }
}

// Mul<&Var> for Var
impl Mul<&Self> for Var {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        &self * other
    }
}

// Div for &Var（逐元素除法）
impl Div for &Var {
    type Output = Var;

    fn div(self, other: &Var) -> Var {
        self.try_div(other).expect("Var 除法失败")
    }
}

// Div for Var
impl Div for Var {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        &self / &other
    }
}

// Div<Var> for &Var
impl Div<Var> for &Var {
    type Output = Var;

    fn div(self, other: Var) -> Var {
        self / &other
    }
}

// Div<&Var> for Var
impl Div<&Self> for Var {
    type Output = Self;

    fn div(self, other: &Self) -> Self {
        &self / other
    }
}

// ==================== Var 与 Tensor 混合运算 ====================
//
// 支持 Var 和 Tensor 的直接运算，内部自动将 Tensor 转换为 input 节点。
// 这让用户可以像 PyTorch 一样自然地混合使用 Var 和 Tensor。

impl Var {
    /// 将 Tensor 转换为 Var（内部辅助方法）
    ///
    /// 在 Graph 中创建一个 BasicInput 节点并设置值。
    /// 用于 Var-Tensor 混合运算和 Loss 函数的 Tensor 版本。
    pub(crate) fn tensor_to_var(&self, tensor: &Tensor) -> Self {
        let mut g = self.graph.borrow_mut();
        let id = g
            .new_basic_input_node(tensor.shape(), None)
            .expect("创建 Tensor->Var 转换节点失败");
        g.set_node_value(id, Some(tensor))
            .expect("设置 Tensor 值失败");
        Self::new(id, Rc::clone(&self.graph))
    }
}

// -------------------- Add: Var + Tensor --------------------

impl Add<&Tensor> for &Var {
    type Output = Var;

    fn add(self, other: &Tensor) -> Var {
        let other_var = self.tensor_to_var(other);
        self + &other_var
    }
}

impl Add<Tensor> for &Var {
    type Output = Var;

    fn add(self, other: Tensor) -> Var {
        self + &other
    }
}

impl Add<&Tensor> for Var {
    type Output = Self;

    fn add(self, other: &Tensor) -> Self {
        &self + other
    }
}

impl Add<Tensor> for Var {
    type Output = Self;

    fn add(self, other: Tensor) -> Self {
        &self + &other
    }
}

// -------------------- Add: Tensor + Var --------------------

impl Add<&Var> for &Tensor {
    type Output = Var;

    fn add(self, other: &Var) -> Var {
        other + self // 加法交换律
    }
}

impl Add<Var> for &Tensor {
    type Output = Var;

    fn add(self, other: Var) -> Var {
        self + &other
    }
}

impl Add<&Var> for Tensor {
    type Output = Var;

    fn add(self, other: &Var) -> Var {
        &self + other
    }
}

impl Add<Var> for Tensor {
    type Output = Var;

    fn add(self, other: Var) -> Var {
        &self + &other
    }
}

// -------------------- Sub: Var - Tensor --------------------

impl Sub<&Tensor> for &Var {
    type Output = Var;

    fn sub(self, other: &Tensor) -> Var {
        let other_var = self.tensor_to_var(other);
        self - &other_var
    }
}

impl Sub<Tensor> for &Var {
    type Output = Var;

    fn sub(self, other: Tensor) -> Var {
        self - &other
    }
}

impl Sub<&Tensor> for Var {
    type Output = Self;

    fn sub(self, other: &Tensor) -> Self {
        &self - other
    }
}

impl Sub<Tensor> for Var {
    type Output = Self;

    fn sub(self, other: Tensor) -> Self {
        &self - &other
    }
}

// -------------------- Sub: Tensor - Var --------------------

impl Sub<&Var> for &Tensor {
    type Output = Var;

    fn sub(self, other: &Var) -> Var {
        let self_var = other.tensor_to_var(self);
        &self_var - other
    }
}

impl Sub<Var> for &Tensor {
    type Output = Var;

    fn sub(self, other: Var) -> Var {
        self - &other
    }
}

impl Sub<&Var> for Tensor {
    type Output = Var;

    fn sub(self, other: &Var) -> Var {
        &self - other
    }
}

impl Sub<Var> for Tensor {
    type Output = Var;

    fn sub(self, other: Var) -> Var {
        &self - &other
    }
}

// -------------------- Mul: Var * Tensor --------------------

impl Mul<&Tensor> for &Var {
    type Output = Var;

    fn mul(self, other: &Tensor) -> Var {
        let other_var = self.tensor_to_var(other);
        self * &other_var
    }
}

impl Mul<Tensor> for &Var {
    type Output = Var;

    fn mul(self, other: Tensor) -> Var {
        self * &other
    }
}

impl Mul<&Tensor> for Var {
    type Output = Self;

    fn mul(self, other: &Tensor) -> Self {
        &self * other
    }
}

impl Mul<Tensor> for Var {
    type Output = Self;

    fn mul(self, other: Tensor) -> Self {
        &self * &other
    }
}

// -------------------- Mul: Tensor * Var --------------------

impl Mul<&Var> for &Tensor {
    type Output = Var;

    fn mul(self, other: &Var) -> Var {
        other * self // 乘法交换律
    }
}

impl Mul<Var> for &Tensor {
    type Output = Var;

    fn mul(self, other: Var) -> Var {
        self * &other
    }
}

impl Mul<&Var> for Tensor {
    type Output = Var;

    fn mul(self, other: &Var) -> Var {
        &self * other
    }
}

impl Mul<Var> for Tensor {
    type Output = Var;

    fn mul(self, other: Var) -> Var {
        &self * &other
    }
}

// -------------------- Div: Var / Tensor --------------------

impl Div<&Tensor> for &Var {
    type Output = Var;

    fn div(self, other: &Tensor) -> Var {
        let other_var = self.tensor_to_var(other);
        self / &other_var
    }
}

impl Div<Tensor> for &Var {
    type Output = Var;

    fn div(self, other: Tensor) -> Var {
        self / &other
    }
}

impl Div<&Tensor> for Var {
    type Output = Self;

    fn div(self, other: &Tensor) -> Self {
        &self / other
    }
}

impl Div<Tensor> for Var {
    type Output = Self;

    fn div(self, other: Tensor) -> Self {
        &self / &other
    }
}

// -------------------- Div: Tensor / Var --------------------

impl Div<&Var> for &Tensor {
    type Output = Var;

    fn div(self, other: &Var) -> Var {
        let self_var = other.tensor_to_var(self);
        &self_var / other
    }
}

impl Div<Var> for &Tensor {
    type Output = Var;

    fn div(self, other: Var) -> Var {
        self / &other
    }
}

impl Div<&Var> for Tensor {
    type Output = Var;

    fn div(self, other: &Var) -> Var {
        &self / other
    }
}

impl Div<Var> for Tensor {
    type Output = Var;

    fn div(self, other: Var) -> Var {
        &self / &other
    }
}

// Neg for &Var（实现为 -1 * self）
impl Neg for &Var {
    type Output = Var;

    fn neg(self) -> Var {
        let mut g = self.graph.borrow_mut();
        // 创建 -1 常量
        let neg_one_id = g
            .new_basic_input_node(&[1, 1], None)
            .expect("创建 -1 节点失败");
        g.set_node_value(neg_one_id, Some(&Tensor::new(&[-1.0], &[1, 1])))
            .expect("设置 -1 值失败");
        // -self = -1 * self（Multiply 支持广播）
        let id = g
            .new_multiply_node(neg_one_id, self.node_id(), None)
            .expect("创建取反节点失败");
        Var::new(id, Rc::clone(&self.graph))
    }
}

// Neg for Var
impl Neg for Var {
    type Output = Self;

    fn neg(self) -> Self {
        -&self
    }
}

// ==================== DetachedVar ====================

/// Detached Var（轻量级包装器）
///
/// 这是 `Var::detach()` 返回的类型，用于表示"梯度截断"的 Var 视图：
/// - **不创建图节点**：与之前的 Identity 节点方式不同，这是零成本抽象
/// - **实现 `ForwardInput`**：可直接传入 `ModelState::forward()`
/// - **行为像 detached Var**：梯度不会流回原 Var
///
/// # 用法
/// ```ignore
/// let fake = G.forward(&noise)?;
/// let d_fake = D.forward(&fake.detach())?;  // DetachedVar，无图节点创建
/// ```
///
/// # 与 Var 的区别
/// - `Var::is_detached()` 检查节点本身的 detached 标志
/// - `DetachedVar` 是一个包装器，表示"以 detached 方式使用这个 Var"
#[derive(Clone)]
pub struct DetachedVar {
    inner: Var,
}

impl DetachedVar {
    /// 获取内部 Var 的引用
    pub const fn inner(&self) -> &Var {
        &self.inner
    }

    /// 获取值
    pub fn value(&self) -> Result<Option<Tensor>, GraphError> {
        self.inner.value()
    }

    /// 获取预期形状
    pub fn value_expected_shape(&self) -> Vec<usize> {
        self.inner.value_expected_shape()
    }

    /// 触发前向传播
    pub fn forward(&self) -> Result<(), GraphError> {
        self.inner.forward()
    }

    /// 始终返回 true（这是 `DetachedVar` 的核心特性）
    pub const fn is_detached(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_zeros() {
        let tensor = Init::Zeros.generate(&[2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert!(tensor.data_as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_init_ones() {
        let tensor = Init::Ones.generate(&[2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert!(tensor.data_as_slice().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_init_kaiming() {
        let tensor = Init::Kaiming.generate(&[100, 50]);
        assert_eq!(tensor.shape(), &[100, 50]);
        // Kaiming: std = sqrt(2/fan_in) = sqrt(2/100) ≈ 0.1414
        let expected_std = (2.0 / 100.0_f32).sqrt();
        let data = tensor.data_as_slice();
        let actual_std = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
        let actual_std = actual_std.sqrt();
        assert!((actual_std - expected_std).abs() < 0.05);
    }

    #[test]
    fn test_detached_var_is_lightweight() {
        // 验证 DetachedVar 不创建图节点
        use crate::nn::Graph;
        let graph = Graph::new();
        let x = graph.input(&crate::tensor::Tensor::ones(&[1, 2])).unwrap();
        let initial_count = graph.inner().nodes_count();

        // detach() 不应该创建新节点
        let _ = x.detach();
        let after_detach_count = graph.inner().nodes_count();
        assert_eq!(initial_count, after_detach_count, "detach() 不应创建新节点");

        // detach_node() 应该创建新节点
        let _ = x.detach_node();
        let after_detach_node_count = graph.inner().nodes_count();
        assert_eq!(
            initial_count + 1,
            after_detach_node_count,
            "detach_node() 应创建一个新节点"
        );
    }

    #[test]
    fn test_init_xavier() {
        let tensor = Init::Xavier.generate(&[100, 50]);
        assert_eq!(tensor.shape(), &[100, 50]);
        // Xavier: std = sqrt(2/(fan_in + fan_out)) = sqrt(2/150) ≈ 0.1155
        let expected_std = (2.0 / 150.0_f32).sqrt();
        let data = tensor.data_as_slice();
        let actual_std = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
        let actual_std = actual_std.sqrt();
        assert!((actual_std - expected_std).abs() < 0.05);
    }
}
