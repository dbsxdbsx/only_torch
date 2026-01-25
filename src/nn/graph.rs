/*
 * @Author       : 老董
 * @Date         : 2024-01-31 17:57:13
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-15 16:41:45
 * @Description  : 神经网络模型的计算图
 */

use super::NodeId;
use super::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use super::nodes::raw_node::Reduction;
use super::nodes::{NodeHandle, NodeType};
use crate::tensor::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// 分组类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupKind {
    /// 层级分组（如 Linear、Conv2d）
    Layer,
    /// 模型级分组（如 Generator、Discriminator）
    Model,
}

/// 层分组信息（用于可视化时将属于同一层/模型的节点框在一起）
#[derive(Debug, Clone)]
pub struct LayerGroup {
    /// 层名称（如 "fc1", "conv1"）
    pub name: String,
    /// 层类型（如 "Linear", "Conv2d", "Model"）
    pub layer_type: String,
    /// 层的描述信息（如 "784→128"）
    pub description: String,
    /// 属于该层的节点 ID 列表（用于可视化显示）
    pub node_ids: Vec<NodeId>,
    /// 分组类型
    pub kind: GroupKind,
    /// 循环层的时间步数（None 表示非循环层）
    /// 对于变长序列，使用 min_steps 和 max_steps
    pub recurrent_steps: Option<usize>,
    /// 循环层的最小时间步数（变长序列支持）
    pub min_steps: Option<usize>,
    /// 循环层的最大时间步数（变长序列支持）
    pub max_steps: Option<usize>,
    /// 需要在可视化中隐藏的节点 ID（如循环层的非代表性展开节点）
    pub hidden_node_ids: Vec<NodeId>,
    /// 代表性节点（折叠显示，实际代表多个节点）
    /// 格式：(node_id, 实际重复次数)
    pub folded_nodes: Vec<(NodeId, usize)>,
    /// 输出代理：(隐藏的真实输出节点, 代表性输出节点)
    /// 用于在可视化时连接代表性节点到下游（折叠层的输出流向）
    pub output_proxy: Option<(NodeId, NodeId)>,
}

/// 循环层元信息（惰性收集可视化信息）
///
/// 分为两部分：
/// - 静态信息：层创建时注册（参数节点、每步节点数）
/// - 动态信息：forward 时更新（时间步数、起始/结束节点 ID）
///
/// 只在 `save_visualization` 时才根据此元信息推断完整的展开分组，
/// 避免 forward 时的额外开销（只记录几个数值）。
#[derive(Debug, Clone)]
pub struct RecurrentLayerMeta {
    /// 层名称（如 "rnn", "lstm"）
    pub name: String,
    /// 层类型（"RNN", "LSTM", "GRU"）
    pub layer_type: String,
    /// 层描述（如 "1→16"）
    pub description: String,
    /// 参数节点 ID 列表
    pub param_node_ids: Vec<NodeId>,
    /// 每个时间步的计算节点数量（不含参数和初始状态）
    /// RNN: 6 (select, matmul, matmul, add, add, tanh)
    pub nodes_per_step: usize,
    /// 动态信息列表：每次 forward 调用都会追加（支持变长序列）
    /// 惰性推断时使用最后一次调用作为代表，其他调用的节点被隐藏
    pub unroll_infos: Vec<RecurrentUnrollInfo>,
}

/// 循环层展开的动态信息（forward 时记录）
#[derive(Debug, Clone)]
pub struct RecurrentUnrollInfo {
    /// 时间步数
    pub steps: usize,
    /// 输入节点 ID（如 SmartInput）
    pub input_node_id: NodeId,
    /// 初始状态节点 ID 列表（如 [h0] 或 [h0, c0]）
    /// - RNN/GRU：只有一个 h0
    /// - LSTM：有两个 [h0, c0]
    pub init_state_node_ids: Vec<NodeId>,
    /// 第一个时间步的第一个计算节点 ID（如 Select）
    pub first_step_start_id: NodeId,
    /// 代表性输出节点列表（第一个时间步的各状态输出）
    /// - RNN/GRU：[h_1]
    /// - LSTM：[h_1, c_1]（h 和 c 的第一个时间步输出）
    pub repr_output_node_ids: Vec<NodeId>,
    /// 真实输出节点（最后一个时间步的隐藏状态输出，如 h_N）
    pub real_output_node_id: NodeId,
}

/// BPTT 时间步快照：存储节点在某个时间步的状态
#[derive(Clone)]
pub(crate) struct StepSnapshot {
    /// 节点的值
    pub value: Option<Tensor>,
}

/// 图的完整定义（核心实现）
///
/// 这是计算图的核心实现。用户通常通过 `Graph` 句柄使用此结构，
/// 高级用户（如 NEAT）可通过 `graph.inner()` 访问底层操作。
pub struct GraphInner {
    name: String,
    nodes: HashMap<NodeId, NodeHandle>,
    /// `正向边：parent_id` -> `child_ids（父节点指向子节点`）
    forward_edges: HashMap<NodeId, Vec<NodeId>>,
    /// `反向边：child_id` -> `parent_ids（子节点指向父节点`）
    backward_edges: HashMap<NodeId, Vec<NodeId>>,
    /// 最后一次前向传播的id
    last_forward_pass_id: u64,
    /// 最后一次反向传播的id
    last_backward_pass_id: u64,
    next_id: u64,
    is_eval_mode: bool,
    /// 图级别的随机数生成器（用于参数初始化等）
    /// None 表示使用默认的 `thread_rng（非确定性`）
    rng: Option<StdRng>,
    /// 层分组信息（用于可视化）
    layer_groups: Vec<LayerGroup>,
    /// 循环层元信息（惰性收集：只在可视化时才根据此信息推断完整分组）
    recurrent_layer_metas: Vec<RecurrentLayerMeta>,

    // ========== 循环/记忆机制相关字段（Phase 1） ==========
    /// `循环边：to_node` -> `from_node（to` 节点在 `step()` 时从 from 节点的上一步值读取）
    recurrent_edges: HashMap<NodeId, NodeId>,
    /// 双缓冲：存储循环节点的上一时间步值
    prev_values: HashMap<NodeId, Tensor>,
    /// 当前时间步（用于调试，每次 `step()` `递增，reset()` 归零）
    time_step: u64,

    // ========== BPTT 相关字段（Phase 2） ==========
    /// 时间步历史：存储每个时间步的节点快照，用于 BPTT
    /// 每个元素是一个时间步的快照：NodeId -> (value, jacobi)
    /// 只在训练模式下记录
    step_history: Vec<HashMap<NodeId, StepSnapshot>>,

    /// BPTT 调试标志（仅用于调试）
    #[cfg(test)]
    bptt_debug: bool,
}

impl Default for GraphInner {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== V2 API: Graph ====================

use std::cell::RefCell;
use std::rc::Rc;

use super::var::{Init, Var};

/// Graph - 计算图句柄（PyTorch 风格用户 API）
///
/// # 设计原则
/// - 是 `Rc<RefCell<GraphInner>>` 的薄封装
/// - Clone 语义：多个 Graph 引用同一个 `GraphInner`
/// - 创建的 Var 自动持有图引用
///
/// # 使用示例
/// ```ignore
/// let graph = Graph::new();
/// let x = graph.input(&images)?;
/// let y = x.relu().matmul(&w)?;  // Var 上的链式调用
/// let loss = y.cross_entropy(&target)?;
/// loss.backward()?;
/// ```
///
/// # 底层访问
/// ```ignore
/// // 通过 inner() 访问 GraphInner 进行底层操作
/// let mut g = graph.inner_mut();
/// g.forward_node(loss_id)?;
/// ```
#[derive(Clone)]
pub struct Graph {
    inner: Rc<RefCell<GraphInner>>,
}

impl Graph {
    // ==================== 创建 ====================

    /// 创建新图
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(GraphInner::new())),
        }
    }

    /// 创建带种子的图（用于确定性训练）
    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            inner: Rc::new(RefCell::new(GraphInner::new_with_seed(seed))),
        }
    }

    /// 从现有 `GraphInner` 创建句柄
    pub fn from_inner(inner: GraphInner) -> Self {
        Self {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    /// 从现有 Rc 创建句柄（供 `Var::get_graph` 使用）
    pub(crate) const fn from_rc(inner: Rc<RefCell<GraphInner>>) -> Self {
        Self { inner }
    }

    /// 获取内部 `GraphInner` 的不可变引用（用于底层查询）
    ///
    /// **注意**：这是一个 escape hatch，正常使用不需要调用此方法。
    pub fn inner(&self) -> std::cell::Ref<'_, GraphInner> {
        self.inner.borrow()
    }

    /// 获取内部 `GraphInner` 的可变引用（用于底层操作，如 NEAT 拓扑变异）
    ///
    /// **注意**：这是一个 escape hatch，正常使用不需要调用此方法。
    pub fn inner_mut(&self) -> std::cell::RefMut<'_, GraphInner> {
        self.inner.borrow_mut()
    }

    /// 获取内部 Rc（供 Var 使用）
    pub(crate) fn inner_rc(&self) -> Rc<RefCell<GraphInner>> {
        Rc::clone(&self.inner)
    }

    /// 将 `NodeId` 包装成 Var
    ///
    /// 用于将底层 `GraphInner` 节点操作的结果包装成 Var，以便使用新版 API。
    ///
    /// **注意**：这是一个 escape hatch，用于混合新旧 API 的场景。
    /// 正常使用建议直接使用新版 Layer API（如 Conv2d、Linear 等）。
    ///
    /// # 示例
    /// ```ignore
    /// // 使用旧版 API 创建卷积层
    /// let conv = conv2d(&mut graph.inner_mut(), input, ...)?;
    /// // 包装成 Var 以便使用新版 API 链式调用
    /// let conv_var = graph.wrap_node_id(conv.output);
    /// let h = conv_var.relu();
    /// ```
    pub fn wrap_node_id(&self, node_id: NodeId) -> Var {
        Var::new(node_id, Rc::clone(&self.inner))
    }

    // ==================== 创建变量（返回 Var，自动携带图引用）====================

    /// 创建输入节点并设置数据
    ///
    /// # 示例
    /// ```ignore
    /// let x = graph.input(&images)?;  // 返回携带图引用的 Var
    /// let h = x.relu();               // 可直接链式调用
    /// ```
    pub fn input(&self, data: &Tensor) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_basic_input_node(data.shape(), None)?;
        g.set_node_value(node_id, Some(data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建命名输入节点
    pub fn input_named(&self, data: &Tensor, name: &str) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_basic_input_node(data.shape(), Some(name))?;
        g.set_node_value(node_id, Some(data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建带形状的输入节点（无初始值，支持动态 batch）
    ///
    /// 用于 RNN/LSTM/GRU 的初始隐藏状态等场景。
    /// 节点支持动态 batch：第一维可以接受任意值。
    ///
    /// # 示例
    /// ```ignore
    /// // 创建初始隐藏状态节点，后续通过 set_value 设置实际值
    /// let h0 = graph.input_shape(&[1, hidden_size], Some("h0"))?;
    /// ```
    pub fn input_shape(&self, shape: &[usize], name: Option<&str>) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_basic_input_node(shape, name)?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建参数节点（带初始化）
    ///
    /// # 示例
    /// ```ignore
    /// let w = graph.parameter(&[784, 128], Init::Kaiming, "fc1.weight")?;
    /// ```
    pub fn parameter(&self, shape: &[usize], init: Init, name: &str) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_parameter_node(shape, Some(name))?;
        // 如果 Graph 有 RNG，使用它；否则使用全局 RNG
        let init_data = if let Some(ref mut rng) = g.rng {
            init.generate_with_rng(shape, rng)
        } else {
            init.generate(shape)
        };
        g.set_node_value(node_id, Some(&init_data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建零张量
    pub fn zeros(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_basic_input_node(shape, None)?;
        g.set_node_value(node_id, Some(&Tensor::zeros(shape)))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建 Target 输入节点（用于 Loss 的目标值）
    ///
    /// 与 `zeros` 类似，但创建的是 Target 变体的 Input 节点，
    /// 在可视化中会显示为 "Target" 而不是 "Input"。
    pub fn target(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_target_input_node(shape, None)?;
        g.set_node_value(node_id, Some(&Tensor::zeros(shape)))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建全一张量
    pub fn ones(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_basic_input_node(shape, None)?;
        g.set_node_value(node_id, Some(&Tensor::ones(shape)))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建随机张量（标准正态分布 N(0,1)）
    ///
    /// 与 `PyTorch` `torch.randn()` 语义一致。
    pub fn randn(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_basic_input_node(shape, None)?;
        let data = Tensor::normal(0.0, 1.0, shape);
        g.set_node_value(node_id, Some(&data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建动态零张量节点
    ///
    /// 在每次 forward 时根据参考节点的 `batch_size` 生成零张量。
    /// 用于 RNN/LSTM/GRU 的初始隐藏状态。
    ///
    /// # 参数
    /// - `reference`: 参考 Var（用于获取 `batch_size`）
    /// - `feature_shape`: 输出的特征维度（不包括 batch）
    /// - `name`: 可选的节点名称
    ///
    /// # 示例
    /// ```ignore
    /// // 创建一个输出 [?, hidden_size] 形状的零张量节点
    /// let h0 = graph.zeros_like(&x, &[hidden_size], Some("h0"))?;
    /// ```
    pub fn zeros_like(
        &self,
        reference: &Var,
        feature_shape: &[usize],
        name: Option<&str>,
    ) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_zeros_like_node(reference.node_id(), feature_shape, name)?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建常量张量
    ///
    /// # 示例
    /// ```ignore
    /// let c = graph.constant(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))?;
    /// ```
    pub fn constant(&self, data: &Tensor) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_basic_input_node(data.shape(), None)?;
        g.set_node_value(node_id, Some(data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建命名常量张量
    pub fn constant_named(&self, data: &Tensor, name: &str) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_basic_input_node(data.shape(), Some(name))?;
        g.set_node_value(node_id, Some(data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    // ==================== 执行（也可以在 Var 上调用）====================

    /// 前向传播
    pub fn forward(&self, output: &Var) -> Result<(), GraphError> {
        self.inner.borrow_mut().forward(output.node_id())
    }

    /// 反向传播（ensure-forward）
    pub fn backward(&self, loss: &Var) -> Result<f32, GraphError> {
        loss.backward()
    }

    // ==================== 训练控制 ====================

    /// 清零所有参数的梯度
    pub fn zero_grad(&self) -> Result<(), GraphError> {
        self.inner.borrow_mut().clear_grad()
    }

    /// 设置训练模式
    pub fn train(&self) {
        self.inner.borrow_mut().set_train_mode();
    }

    /// 设置评估模式
    pub fn eval(&self) {
        self.inner.borrow_mut().set_eval_mode();
    }

    /// 是否处于评估模式
    pub fn is_eval(&self) -> bool {
        self.inner.borrow().is_eval_mode
    }

    /// 在 `no_grad` 上下文中执行闭包
    ///
    /// 临时切换到评估模式（禁用梯度计算），执行完毕后恢复原模式。
    ///
    /// # 示例
    /// ```ignore
    /// // 验证集评估
    /// let val_loss = graph.no_grad_scope(|g| {
    ///     x.set_value(&val_images)?;
    ///     y.set_value(&val_labels)?;
    ///     loss.forward()?;
    ///     loss.item()
    /// })?;
    /// ```
    pub fn no_grad_scope<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Self) -> R,
    {
        // 保存当前模式
        let was_train = !self.is_eval();

        // 切换到评估模式（禁用梯度）
        self.eval();

        // 执行闭包
        let result = f(self);

        // 恢复之前的模式
        if was_train {
            self.train();
        }

        result
    }

    // ==================== 可视化 ====================

    /// 保存计算图可视化
    ///
    /// # 示例
    /// ```ignore
    /// graph.save_visualization("outputs/model", None)?;
    /// ```
    pub fn save_visualization<P: AsRef<std::path::Path>>(
        &self,
        base_path: P,
        format: Option<ImageFormat>,
    ) -> Result<VisualizationOutput, GraphError> {
        self.inner.borrow().save_visualization(base_path, format)
    }

    /// 保存计算图可视化（启用层分组）
    pub fn save_visualization_grouped<P: AsRef<std::path::Path>>(
        &self,
        base_path: P,
        format: Option<ImageFormat>,
    ) -> Result<VisualizationOutput, GraphError> {
        // 惰性推断：在可视化时根据循环层元信息推断完整的分组
        self.inner.borrow_mut().infer_recurrent_layer_groups();
        self.inner.borrow().save_visualization_grouped(base_path, format)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphInner {
    #[cfg(test)]
    pub(in crate::nn) fn last_forward_pass_id(&self) -> u64 {
        self.last_forward_pass_id
    }

    #[allow(dead_code)]
    pub(in crate::nn) const fn last_backward_pass_id(&self) -> u64 {
        self.last_backward_pass_id
    }

    fn check_duplicate_node_name(&self, name: &str) -> Result<(), GraphError> {
        if self.nodes.values().any(|node| node.name() == name) {
            return Err(GraphError::DuplicateNodeName(format!(
                "节点{}在图{}中重复",
                name,
                self.name()
            )));
        }
        Ok(())
    }

    fn generate_valid_new_node_name(
        &self,
        base_name: &str,
        node_type: &str,
    ) -> Result<String, GraphError> {
        // 若用户提供了名称，检查重复并直接返回错误
        if !base_name.is_empty() {
            self.check_duplicate_node_name(base_name)?;
            return Ok(base_name.to_string());
        }

        // 自动生成名称（只有在用户未提供名称时才进行）
        let mut counter = 1;
        loop {
            let name = format!("{node_type}_{counter}");
            if self.check_duplicate_node_name(&name).is_ok() {
                return Ok(name);
            }
            counter += 1;
        }
    }

    // 基本操作
    pub fn new() -> Self {
        Self::with_name("default_graph")
    }

    /// 创建一个带固定种子的计算图（确保可重复性）
    ///
    /// 使用此方法创建的图会有一个独立的随机数生成器，
    /// 所有通过 `new_parameter_node()` 创建的参数都会使用这个 RNG 初始化。
    ///
    /// # NEAT 友好性
    /// 每个 Graph 有独立的 RNG 状态，多个 Graph 可以并行进化互不干扰。
    ///
    /// # 示例
    /// ```ignore
    /// let graph1 = Graph::new_with_seed(42);
    /// let graph2 = Graph::new_with_seed(42);
    /// // graph1 和 graph2 的参数初始化结果相同
    /// ```
    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            name: "default_graph".to_string(),
            nodes: HashMap::new(),
            forward_edges: HashMap::new(),
            backward_edges: HashMap::new(),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            next_id: 0,
            is_eval_mode: false,
            rng: Some(StdRng::seed_from_u64(seed)),
            layer_groups: Vec::new(),
            recurrent_layer_metas: Vec::new(),
            recurrent_edges: HashMap::new(),
            prev_values: HashMap::new(),
            time_step: 0,
            step_history: Vec::new(),
            #[cfg(test)]
            bptt_debug: false,
        }
    }

    /// 创建一个带名称和固定种子的计算图
    pub fn with_name_and_seed(name: &str, seed: u64) -> Self {
        Self {
            name: name.to_string(),
            nodes: HashMap::new(),
            forward_edges: HashMap::new(),
            backward_edges: HashMap::new(),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            next_id: 0,
            is_eval_mode: false,
            rng: Some(StdRng::seed_from_u64(seed)),
            layer_groups: Vec::new(),
            recurrent_layer_metas: Vec::new(),
            recurrent_edges: HashMap::new(),
            prev_values: HashMap::new(),
            time_step: 0,
            step_history: Vec::new(),
            #[cfg(test)]
            bptt_debug: false,
        }
    }

    pub fn with_name(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: HashMap::new(),
            forward_edges: HashMap::new(),
            backward_edges: HashMap::new(),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            next_id: 0,
            is_eval_mode: false,
            rng: None,
            layer_groups: Vec::new(),
            recurrent_layer_metas: Vec::new(),
            recurrent_edges: HashMap::new(),
            prev_values: HashMap::new(),
            time_step: 0,
            step_history: Vec::new(),
            #[cfg(test)]
            bptt_debug: false,
        }
    }

    /// 设置/重置图的随机种子
    ///
    /// 调用此方法会重置 RNG 状态，后续的参数创建将从新种子开始。
    ///
    /// # 示例
    /// ```ignore
    /// let mut graph = Graph::new();
    /// graph.set_seed(42);
    /// // 现在 graph 使用固定种子
    /// ```
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Some(StdRng::seed_from_u64(seed));
    }

    /// 检查图是否有固定种子
    pub const fn has_seed(&self) -> bool {
        self.rng.is_some()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// 注册一个层分组（用于可视化时将同一层的节点框在一起）
    ///
    /// # 参数
    /// - `name`: 层名称（如 "fc1", "conv1"）
    /// - `layer_type`: 层类型（如 "Linear", "Conv2d"）
    /// - `description`: 层描述（如 "784→128"）
    /// - `node_ids`: 属于该层的节点 ID 列表
    pub fn register_layer_group(
        &mut self,
        name: &str,
        layer_type: &str,
        description: &str,
        node_ids: Vec<NodeId>,
    ) {
        // 检查是否已存在同名分组，避免重复注册
        if self.layer_groups.iter().any(|g| g.name == name) {
            return;
        }
        self.layer_groups.push(LayerGroup {
            name: name.to_string(),
            layer_type: layer_type.to_string(),
            description: description.to_string(),
            node_ids,
            kind: GroupKind::Layer,
            recurrent_steps: None,
            min_steps: None,
            max_steps: None,
            hidden_node_ids: vec![],
            folded_nodes: vec![],
            output_proxy: None,
        });
    }

    /// 注册循环层元信息（惰性收集，用于可视化）
    ///
    /// 在层创建时调用，只记录轻量级静态元信息。动态展开信息
    /// 通过 `update_recurrent_layer_unroll_info` 在 forward 时更新。
    ///
    /// # 参数
    /// - `name`: 层名称（如 "rnn", "lstm"）
    /// - `layer_type`: 层类型（"RNN", "LSTM", "GRU"）
    /// - `description`: 层描述（如 "1→16"）
    /// - `param_node_ids`: 参数节点 ID 列表
    /// - `nodes_per_step`: 每个时间步的计算节点数量
    pub fn register_recurrent_layer_meta(
        &mut self,
        name: &str,
        layer_type: &str,
        description: &str,
        param_node_ids: Vec<NodeId>,
        nodes_per_step: usize,
    ) {
        // 检查是否已存在同名元信息，避免重复注册
        if self
            .recurrent_layer_metas
            .iter()
            .any(|m| m.name == name)
        {
            return;
        }
        self.recurrent_layer_metas.push(RecurrentLayerMeta {
            name: name.to_string(),
            layer_type: layer_type.to_string(),
            description: description.to_string(),
            param_node_ids,
            nodes_per_step,
            unroll_infos: Vec::new(),
        });
    }

    /// 追加循环层的展开信息（forward 时调用，支持变长序列）
    ///
    /// 每次 forward 调用都会追加一条记录，用于：
    /// - 计算步数范围（min_steps, max_steps）
    /// - 选择代表性调用进行折叠显示
    /// - 隐藏其他调用创建的节点
    ///
    /// # 参数
    /// - `name`: 层名称
    /// - `unroll_info`: 展开的动态信息
    pub fn update_recurrent_layer_unroll_info(
        &mut self,
        name: &str,
        unroll_info: RecurrentUnrollInfo,
    ) {
        if let Some(meta) = self
            .recurrent_layer_metas
            .iter_mut()
            .find(|m| m.name == name)
        {
            meta.unroll_infos.push(unroll_info);
        }
    }

    /// 获取循环层元信息列表
    pub fn recurrent_layer_metas(&self) -> &[RecurrentLayerMeta] {
        &self.recurrent_layer_metas
    }

    /// 注册一个循环层分组（内部方法，由惰性推断调用）
    ///
    /// # 参数
    /// - `name`: 层名称（如 "rnn", "lstm"）
    /// - `layer_type`: 层类型（如 "RNN", "LSTM", "GRU"）
    /// - `description`: 层描述（如 "1→16"）
    /// - `repr_node_ids`: 代表性节点 ID 列表（用于可视化显示）
    /// - `folded_nodes`: 折叠节点列表，格式为 (节点ID, 重复次数)
    /// - `hidden_node_ids`: 需要隐藏的节点 ID 列表（如展开的其他时间步）
    /// - `output_proxy`: 输出代理 (隐藏的真实输出节点, 代表性输出节点)，用于可视化时连接下游
    /// - `min_steps`: 最小时间步数
    /// - `max_steps`: 最大时间步数
    fn register_recurrent_layer_group_with_range(
        &mut self,
        name: &str,
        layer_type: &str,
        description: &str,
        repr_node_ids: Vec<NodeId>,
        folded_nodes: Vec<(NodeId, usize)>,
        hidden_node_ids: Vec<NodeId>,
        output_proxy: Option<(NodeId, NodeId)>,
        min_steps: usize,
        max_steps: usize,
    ) {
        // 检查是否已存在同名分组，避免重复注册
        if self.layer_groups.iter().any(|g| g.name == name) {
            return;
        }
        // 固定长度时 recurrent_steps = Some(steps)，变长时为 None
        let recurrent_steps = if min_steps == max_steps {
            Some(max_steps)
        } else {
            None
        };
        self.layer_groups.push(LayerGroup {
            name: name.to_string(),
            layer_type: layer_type.to_string(),
            description: description.to_string(),
            node_ids: repr_node_ids,
            kind: GroupKind::Layer,
            recurrent_steps,
            min_steps: Some(min_steps),
            max_steps: Some(max_steps),
            hidden_node_ids,
            folded_nodes,
            output_proxy,
        });
    }

    /// 惰性推断循环层分组（在 save_visualization 时调用）
    ///
    /// 根据 `recurrent_layer_metas` 中的元信息和 `unroll_info` 推断完整的分组：
    /// - 代表性节点：参数 + 初始状态 + 第一个时间步的计算节点
    /// - 折叠节点：第一个时间步的计算节点（代表多个重复实例）
    /// - 隐藏节点：其他时间步的计算节点
    /// - 输出代理：(真实输出, 代表性输出)
    pub fn infer_recurrent_layer_groups(&mut self) {
        // 收集需要推断的循环层（避免借用冲突）
        let metas_to_infer: Vec<_> = self
            .recurrent_layer_metas
            .iter()
            .filter(|m| !m.unroll_infos.is_empty())
            .filter(|m| !self.layer_groups.iter().any(|g| g.name == m.name))
            .cloned()
            .collect();

        for meta in metas_to_infer {
            // 计算步数范围
            let min_steps = meta.unroll_infos.iter().map(|i| i.steps).min().unwrap();
            let max_steps = meta.unroll_infos.iter().map(|i| i.steps).max().unwrap();

            // 选择最后一次调用作为代表（它的输出节点连接到下游层）
            let repr_info = meta.unroll_infos.last().unwrap();

            // 构建代表性节点列表：参数 + 初始状态 + 第一个时间步的计算节点
            let mut repr_node_ids: Vec<NodeId> = meta.param_node_ids.clone();
            // 添加所有初始状态节点（RNN/GRU: 1 个, LSTM: 2 个）
            repr_node_ids.extend(repr_info.init_state_node_ids.iter().copied());

            // 第一个时间步的节点 ID 范围
            let first_step_start = repr_info.first_step_start_id.0;
            let first_step_end = first_step_start + meta.nodes_per_step as u64;

            // 折叠节点：
            // 1. 初始状态节点（ZerosLike）：步数为 1（每次展开只使用一次）
            // 2. 第一个时间步的计算节点：步数为 min_steps-max_steps
            let mut folded_nodes: Vec<(NodeId, usize)> = Vec::new();
            // 所有初始状态节点使用固定步数 1
            for init_id in &repr_info.init_state_node_ids {
                folded_nodes.push((*init_id, 1));
            }
            // 时间步计算节点使用 max_steps（显示最大值）
            for id in first_step_start..first_step_end {
                folded_nodes.push((NodeId(id), max_steps));
            }
            // 把折叠节点加入代表性节点列表
            repr_node_ids.extend(folded_nodes.iter().map(|(id, _)| *id));

            // 隐藏节点：
            // 1. 代表性调用的其他时间步节点
            // 2. 其他所有调用的全部节点（包括初始状态）
            let mut hidden_node_ids = Vec::new();

            // 1. 代表性调用的其他时间步（step 1 到 step N-1）
            for step in 1..repr_info.steps {
                let step_start = first_step_start + (step as u64) * (meta.nodes_per_step as u64);
                let step_end = step_start + meta.nodes_per_step as u64;
                for id in step_start..step_end {
                    hidden_node_ids.push(NodeId(id));
                }
            }

            // 2. 其他调用的全部节点（输入、初始状态、时间步）
            // 注：下游节点（FC、Loss）会在 to_dot_with_options 中使用描述符扩展
            for (idx, info) in meta.unroll_infos.iter().enumerate() {
                // 跳过代表性调用（最后一个）
                if idx == meta.unroll_infos.len() - 1 {
                    continue;
                }
                // 隐藏该调用的输入节点（SmartInput）
                hidden_node_ids.push(info.input_node_id);
                // 隐藏该调用的所有初始状态节点
                hidden_node_ids.extend(info.init_state_node_ids.iter().copied());
                // 隐藏该调用的所有时间步节点
                let call_first_step_start = info.first_step_start_id.0;
                for step in 0..info.steps {
                    let step_start =
                        call_first_step_start + (step as u64) * (meta.nodes_per_step as u64);
                    let step_end = step_start + meta.nodes_per_step as u64;
                    for id in step_start..step_end {
                        hidden_node_ids.push(NodeId(id));
                    }
                }
            }

            // 输出代理：(真实输出, 代表性输出)
            // 使用第一个输出（h）作为代理，因为它是连接到下游层的主要输出
            let output_proxy = repr_info
                .repr_output_node_ids
                .first()
                .map(|&repr_id| (repr_info.real_output_node_id, repr_id));

            // 存储步数范围信息到分组中
            self.register_recurrent_layer_group_with_range(
                &meta.name,
                &meta.layer_type,
                &meta.description,
                repr_node_ids,
                folded_nodes,
                hidden_node_ids,
                output_proxy,
                min_steps,
                max_steps,
            );
        }

    }

    /// 收集需要隐藏的"根节点"（非代表性调用的输出节点）
    /// 这些根节点的后代会在 to_dot_with_options 中使用描述符来扩展
    fn hidden_output_nodes(&self) -> Vec<NodeId> {
        self.recurrent_layer_metas
            .iter()
            .flat_map(|meta| {
                meta.unroll_infos
                    .iter()
                    .take(meta.unroll_infos.len().saturating_sub(1)) // 跳过最后一个（代表性调用）
                    .map(|info| info.real_output_node_id)
            })
            .collect()
    }

    /// 获取所有层分组信息
    pub fn layer_groups(&self) -> &[LayerGroup] {
        &self.layer_groups
    }

    pub fn nodes(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    pub fn has_node_value(&self, node_id: NodeId) -> Result<bool, GraphError> {
        self.nodes
            .get(&node_id)
            .map(super::nodes::node_handle::NodeHandle::has_value)
            .ok_or(GraphError::NodeNotFound(node_id))
    }

    // 前向传播：
    pub fn forward(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        // 1. 检查节点类型
        let node = self.get_node(node_id)?;
        match node.node_type() {
            NodeType::Input(_) | NodeType::Parameter(_) | NodeType::State(_) => {
                // 如果节点已经有值，就跳过（允许 forward 调用，只是不做任何事）
                // 这对于 RecurrentOutput 等节点很重要，它们的值通过 set_value 设置
                if node.has_value() {
                    return Ok(());
                }
                return Err(GraphError::InvalidOperation(format!(
                    "{node}是输入/参数/状态节点，其值应通过set_value设置，而不是通过父节点前向传播计算"
                )));
            }
            _ => {}
        }

        // 2. 为图本次的前向传播设置新id
        let new_graph_forward_pass_id = self.last_forward_pass_id + 1;

        // 3. 通过内部方法执行完整的前向传播
        self.forward_node_internal(node_id, new_graph_forward_pass_id)?;

        // 4. 只有成功后才更新图的前向传播ID
        self.last_forward_pass_id = new_graph_forward_pass_id;
        Ok(())
    }

    // 前向传播的内部实现
    fn forward_node_internal(
        &mut self,
        node_id: NodeId,
        new_graph_forward_pass_id: u64,
    ) -> Result<(), GraphError> {
        // 1. 必要检查
        let node = self.get_node_mut(node_id)?;

        // 1.1 检查节点类型和状态
        match node.node_type() {
            // 1.1.1 输入、参数、状态节点（这些节点的值由外部设置，不从父节点计算）
            NodeType::Input(_) | NodeType::Parameter(_) | NodeType::State(_) => {
                if node.has_value() {
                    node.set_last_forward_pass_id(new_graph_forward_pass_id);
                    return Ok(());
                }
                return Err(GraphError::InvalidOperation(format!(
                    "{}不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为{}，而图的前向传播次数为{}",
                    node,
                    node.last_forward_pass_id(),
                    new_graph_forward_pass_id
                )));
            }
            _ => {
                // 1.1.2 其他类型节点，若已在本代计算过则直接返回
                if node.last_forward_pass_id() == new_graph_forward_pass_id {
                    return Ok(());
                }
            }
        }

        // 2. 递归计算所有父节点
        let parents_ids = self.get_node_parents(node_id)?;
        for parent_id in &parents_ids {
            self.forward_node_internal(*parent_id, new_graph_forward_pass_id)?;
        }

        // 3. 创建临时的父节点句柄，不持有self的引用(避免等会计算值时借用检查问题)
        let parent_nodes = parents_ids
            .iter()
            .map(|id| self.get_node(*id).unwrap().clone())
            .collect::<Vec<NodeHandle>>();

        // 4. 计算当前节点
        let node = self.get_node_mut(node_id)?;
        node.calc_value_by_parents(&parent_nodes)?;

        // 5. 更新节点的前向传播次数为当前次数
        node.set_last_forward_pass_id(new_graph_forward_pass_id);

        // 6. 返回
        Ok(())
    }

    /// 释放中间节点的值和梯度以节省内存
    ///
    /// 保留 Input 和 Parameter 节点的数据，只清除计算节点（如 Add、MatMul 等）的值和梯度。
    /// 这些值在下次 forward 时会重新计算，梯度在下次 backward 时会重新计算。
    ///
    /// # 用途
    /// - 在 `backward_ex(loss, retain_graph=false)` 后自动调用
    /// - 可手动调用以释放内存
    ///
    /// # 设计理由
    /// 当 `retain_graph=false` 时，同时释放值和梯度以保持一致性：
    /// - 值被释放：需要重新 forward 才能再次 backward
    /// - 梯度也被释放：避免用户误以为中间节点的梯度是累积的（实际只是本次的）
    ///
    /// 这更接近 `PyTorch` 的语义：中间节点的梯度默认不保留（除非 `retain_grad()`）
    fn release_intermediate_results(&mut self) -> Result<(), GraphError> {
        for node in self.nodes.values_mut() {
            match node.node_type() {
                // 保留输入、参数、状态节点的值和梯度
                // State 的值由 step()/reset() 管理，不在 backward 后清除
                NodeType::Input(_) | NodeType::Parameter(_) | NodeType::State(_) => {}
                // 清除其他节点的值和梯度
                _ => {
                    node.clear_value()?;
                    let _ = node.clear_grad();
                }
            }
        }
        Ok(())
    }

    /// 重置中间节点的 grad（保留参数节点的 grad 以支持梯度累积）
    ///
    /// `PyTorch` 语义：
    /// - 参数节点（叶节点）：梯度跨 backward 调用累积
    /// - 中间节点：每次 backward 重新计算，不累积
    ///
    /// 这确保了多任务学习等场景下梯度计算的正确性。
    fn reset_intermediate_grad(&mut self) {
        for node in self.nodes.values_mut() {
            match node.node_type() {
                // 保留参数节点的 grad（支持梯度累积）
                NodeType::Parameter(_) => {}
                // 清除其他节点的 grad（中间节点每次 backward 重新计算）
                _ => {
                    let _ = node.clear_grad();
                    // 重置 backward pass id 以便重新计算
                    node.set_last_backward_pass_id(0);
                }
            }
        }
    }

    // ========== Batch 模式（Gradient-based）==========

    /// VJP 反向传播核心实现（内部方法）
    ///
    /// 被 `backward()` 和 `backward_ex()` 共用
    fn backward_vjp_core(&mut self, loss_id: NodeId) -> Result<(), GraphError> {
        // 0. 警告：在 no_grad（eval）模式下调用 backward 通常是误用
        if !self.is_train_mode() {
            eprintln!(
                "[only_torch 警告] 在 no_grad/eval 模式下调用 backward，这通常是误用。\
                如确需此行为，请忽略此警告。"
            );
        }

        // 1. 重置中间节点的 grad（PyTorch 语义：只有参数节点梯度累积）
        self.reset_intermediate_grad();

        // 2. 验证损失节点
        let loss_node = self.get_node(loss_id)?;
        let loss_value = loss_node.value().ok_or_else(|| {
            GraphError::ComputationError(format!("损失节点 {loss_node} 没有值，请先执行 forward"))
        })?;

        // 损失应为标量
        if loss_value.size() != 1 {
            return Err(GraphError::InvalidOperation(format!(
                "反向传播要求损失为标量 [1, 1]，但得到 {:?}",
                loss_value.shape()
            )));
        }

        // 3. 损失节点的梯度为 1.0
        // 注意：不在此处调用 clear_grad()，由调用者负责（PyTorch 惯例）
        let loss_grad = Tensor::ones(&[1, 1]);
        self.get_node_mut(loss_id)?.set_grad(Some(&loss_grad))?;

        // 4. 获取拓扑排序（从损失到输入）
        let topo_order = self.topological_sort_backward(loss_id)?;

        // 5. 按拓扑顺序反向传播梯度
        for node_id in &topo_order {
            self.propagate_grad_to_parents(*node_id, loss_id, None)?;
        }

        // 6. 处理 SmartInput 的梯度路由
        // SmartInput 节点可能有梯度路由目标，需要将梯度累加到目标节点
        // 然后从目标节点继续反向传播
        let routed_targets = self.process_gradient_routing()?;

        // 6.1 从路由目标节点继续反向传播
        for target_id in routed_targets {
            self.backward_from_node(target_id)?;
        }

        // 7. 递增反向传播 pass_id
        self.last_backward_pass_id += 1;
        let new_pass_id = self.last_backward_pass_id;

        // 8. 更新参与反向传播的节点的 pass_id（用于判断节点是否参与了最近的 backward）
        for node_id in topo_order {
            // 只更新有梯度的节点（Input 节点没有梯度，不更新）
            if let Ok(node) = self.get_node_mut(node_id)
                && node.grad().is_some()
            {
                node.set_last_backward_pass_id(new_pass_id);
            }
        }

        Ok(())
    }

    /// 处理 SmartInput 节点的梯度路由
    ///
    /// 遍历所有 SmartInput 节点，如果有梯度路由目标，
    /// 将 SmartInput 的梯度累加到目标节点。
    ///
    /// # 返回
    /// 接收到路由梯度的目标节点 ID 列表（用于继续反向传播）
    fn process_gradient_routing(&mut self) -> Result<Vec<NodeId>, GraphError> {
        use super::nodes::raw_node::{InputVariant, NodeType};

        // 收集需要路由的梯度信息：(target_id, gradient)
        let mut routing_info: Vec<(NodeId, Tensor)> = Vec::new();

        for node in self.nodes.values() {
            if let NodeType::Input(InputVariant::Smart(smart) | InputVariant::RecurrentOutput(smart)) =
                node.node_type()
            {
                // 检查是否有梯度路由目标
                if let Some(target_id) = smart.gradient_target() {
                    // 检查 SmartInput 是否有梯度（且未被 detach）
                    if !smart.is_detached()
                        && let Some(grad) = node.grad()
                    {
                        routing_info.push((target_id, grad.clone()));
                    }
                }
            }
        }

        // 将梯度累加到目标节点
        let mut routed_targets = Vec::new();
        for (target_id, grad) in routing_info {
            if let Ok(target_node) = self.get_node_mut(target_id) {
                if let Some(existing_grad) = target_node.grad() {
                    let new_grad = existing_grad + &grad;
                    target_node.set_grad(Some(&new_grad))?;
                } else {
                    target_node.set_grad(Some(&grad))?;
                }
                routed_targets.push(target_id);
            }
            // 如果目标节点不存在，静默忽略（可能已被清理）
        }

        Ok(routed_targets)
    }

    /// 从指定节点继续反向传播
    ///
    /// 用于梯度路由后，从目标节点继续向其父节点传播梯度。
    fn backward_from_node(&mut self, start_id: NodeId) -> Result<(), GraphError> {
        // 获取从 start_id 到所有输入的反向拓扑排序
        let topo_order = self.topological_sort_backward(start_id)?;

        // 按拓扑顺序反向传播梯度
        for node_id in &topo_order {
            self.propagate_grad_to_parents(*node_id, start_id, None)?;
        }

        Ok(())
    }

    /// 将梯度从当前节点传播到其父节点
    ///
    /// # 参数
    /// - `node_id`: 当前节点 ID
    /// - `_loss_id`: 损失节点 ID（保留参数）
    /// - `target_params`: 可选的目标参数集合。如果提供，只为目标参数计算并设置梯度，
    ///   跳过非目标的 Parameter 节点以节省计算。
    fn propagate_grad_to_parents(
        &mut self,
        node_id: NodeId,
        _loss_id: NodeId,
        target_params: Option<&std::collections::HashSet<NodeId>>,
    ) -> Result<(), GraphError> {
        use super::nodes::raw_node::InputVariant;

        // 检查当前节点是否被 detach
        // 被 detach 的节点不向父节点传播梯度
        {
            let node = self.get_node(node_id)?;
            if node.is_detached() {
                return Ok(());
            }
        }

        // 获取父节点列表
        let parent_ids = self.get_node_parents(node_id)?;
        if parent_ids.is_empty() {
            return Ok(()); // 输入/参数节点，无需继续传播
        }

        // 先在只读阶段把需要的父节点梯度算出来
        let parent_grads: Vec<(NodeId, Tensor)> = {
            let node = self.get_node(node_id)?;
            let upstream_grad = match node.grad() {
                Some(g) => g,
                None => return Ok(()), // 没有梯度，跳过
            };

            let mut grads = Vec::with_capacity(parent_ids.len());
            for parent_id in &parent_ids {
                let parent = self.get_node(*parent_id)?;

                // 跳过 Data/Target Input 节点（它们不需要梯度）
                // 但不跳过 SmartInput/RecurrentOutput 节点（需要接收梯度以便路由）
                if let NodeType::Input(variant) = parent.node_type() {
                    match variant {
                        InputVariant::Data(_) | InputVariant::Target(_) => continue,
                        InputVariant::Smart(_) | InputVariant::RecurrentOutput(_) => {} // 不跳过
                    }
                }

                // 如果指定了 target_params，跳过非目标的 Parameter 节点
                // 这是核心优化：避免为不需要更新的参数计算梯度
                if let Some(targets) = target_params
                    && let NodeType::Parameter(_) = parent.node_type()
                    && !targets.contains(parent_id)
                {
                    continue; // 跳过非目标参数，节省计算
                }

                // 找到辅助父节点（如果需要）
                let assistant_parent_id = parent_ids.iter().find(|&&id| id != *parent_id).copied();
                let assistant = assistant_parent_id
                    .map(|id| self.get_node(id))
                    .transpose()?;

                let parent_grad = node.calc_grad_to_parent(parent, upstream_grad, assistant)?;
                grads.push((*parent_id, parent_grad));
            }
            grads
        };

        // 回写阶段：累加到父节点的梯度
        for (parent_id, parent_grad) in parent_grads {
            let parent_node = self.get_node_mut(parent_id)?;

            // 如果父节点被 detach，跳过设置梯度
            // 根据文档 2.5 节：被 detach 的节点本身的 grad 应为 None
            if parent_node.is_detached() {
                continue;
            }

            if let Some(existing_grad) = parent_node.grad() {
                let new_grad = existing_grad + &parent_grad;
                parent_node.set_grad(Some(&new_grad))?;
            } else {
                parent_node.set_grad(Some(&parent_grad))?;
            }
        }

        Ok(())
    }

    /// 获取从 loss 到所有输入的反向拓扑排序
    /// 返回的顺序：loss 在最前，input 在最后（适合反向传播）
    fn topological_sort_backward(&self, loss_id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn dfs(
            graph: &GraphInner,
            node_id: NodeId,
            visited: &mut std::collections::HashSet<NodeId>,
            result: &mut Vec<NodeId>,
        ) -> Result<(), GraphError> {
            if visited.contains(&node_id) {
                return Ok(());
            }
            visited.insert(node_id);

            // 先添加当前节点（因为我们要从 loss 向 input 方向传播）
            result.push(node_id);

            // 再访问父节点（反向传播方向）
            let parents = graph.get_node_parents(node_id)?;
            for parent_id in parents {
                dfs(graph, parent_id, visited, result)?;
            }

            Ok(())
        }

        // 从 loss 节点开始 DFS
        dfs(self, loss_id, &mut visited, &mut result)?;

        Ok(result)
    }

    /// 清除所有节点的梯度
    ///
    /// 注意：推荐使用 `zero_grad()`，与 `PyTorch` 风格一致
    pub fn clear_grad(&mut self) -> Result<(), GraphError> {
        for node in self.nodes.values_mut() {
            let _ = node.clear_grad(); // 忽略不支持 grad 的节点
        }
        Ok(())
    }

    // ==================== 统一反向传播 API（VJP 模式）====================
    //
    // 以下是 autodiff_unification_design.md 中定义的统一 API。
    // 设计目标：单样本与批量使用相同的 API，底层统一使用 VJP 模式。

    /// 反向传播（VJP 模式，单样本和批量统一）
    ///
    /// 这是 `backward_ex(loss, false)` 的简写，覆盖 90% 的训练场景。
    ///
    /// # 语义
    /// - 计算 loss 对**所有** `requires_grad` 参数的梯度
    /// - 梯度存储在节点的 `grad` 字段
    /// - 梯度会累积（需要先调用 `zero_grad()` 清零）
    /// - 返回 loss 的标量值（方便用户打印）
    ///
    /// # 梯度隔离
    /// GAN 等场景的梯度隔离通过 `detach()` 实现，而非 `target_params`。
    /// 详见 [梯度流控制设计 - 附录 A](gradient_flow_control_design.md#附录-a设计决策为什么用-detach-而非-target_params)。
    ///
    /// # 示例
    /// ```rust,ignore
    /// optimizer.zero_grad()?;
    /// let loss = model.forward(x)?.cross_entropy(&y)?;
    /// let loss_val = graph.backward(loss)?;      // 计算所有参数的梯度
    /// optimizer.step()?;                         // 更新参数
    /// println!("Loss: {:.4}", loss_val);
    /// ```
    pub fn backward(&mut self, loss: NodeId) -> Result<f32, GraphError> {
        self.backward_ex(loss, false)
    }

    /// 反向传播（扩展版本，支持 `retain_graph`）
    ///
    /// # 参数
    /// - `loss`: 损失节点 ID
    /// - `retain_graph`: 是否保留计算图（用于多次 backward）
    ///
    /// # 使用场景
    /// | 场景 | `retain_graph` |
    /// |------|----------------|
    /// | 标准训练 | `false` |
    /// | 多任务学习（多 loss 累加） | `true` |
    ///
    /// # 设计决策：移除 `target_params`
    /// `PyTorch` 的 `backward()` 没有 `target_params` 参数。
    /// GAN 等场景的梯度隔离通过 `detach()` 实现，语义更清晰、性能更优。
    /// 详见 [梯度流控制设计 - 附录 A](gradient_flow_control_design.md#附录-a设计决策为什么用-detach-而非-target_params)。
    pub fn backward_ex(&mut self, loss: NodeId, retain_graph: bool) -> Result<f32, GraphError> {
        // 1. 获取 loss 值
        let loss_node = self.get_node(loss)?;
        let loss_value = loss_node.value().ok_or_else(|| {
            GraphError::ComputationError(format!("损失节点 {loss_node} 没有值，请先执行 forward"))
        })?;

        let loss_scalar = loss_value.get_data_number().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "无法从损失节点获取标量值，形状: {:?}",
                loss_value.shape()
            ))
        })?;

        // 2. 自动检测是否需要 BPTT（PyTorch 风格：用户无需关心）
        //    条件：存在时间步历史 且 存在循环连接
        let needs_bptt = !self.step_history.is_empty() && !self.recurrent_edges.is_empty();

        if needs_bptt {
            // 使用 BPTT：自动收集所有可训练参数并反向传播
            let param_ids = self.get_trainable_nodes();
            self.backward_through_time(&param_ids, loss)?;
        } else {
            // 普通反向传播（前馈网络）
            self.backward_vjp_core(loss)?;
        }

        // 3. 如果 retain_graph=false，释放中间结果（PyTorch 默认行为）
        if !retain_graph {
            self.release_intermediate_results()?;
        }

        // 4. 返回 loss 值
        Ok(loss_scalar)
    }

    /// 清零所有参数的梯度（PyTorch 风格）
    ///
    /// 与 `PyTorch` 的 `optimizer.zero_grad()` 风格一致。
    ///
    /// # 使用时机
    /// 在每次 `backward()` 之前调用，防止梯度累积。
    ///
    /// # 示例
    /// ```rust,ignore
    /// for batch in dataloader {
    ///     graph.zero_grad()?;        // 清零梯度
    ///     let loss = forward(batch)?;
    ///     graph.backward(loss)?;     // 计算梯度
    ///     optimizer.step()?;         // 更新参数
    /// }
    /// ```
    pub fn zero_grad(&mut self) -> Result<(), GraphError> {
        self.clear_grad()
    }

    /// 获取节点梯度的引用（避免克隆）
    ///
    /// 与 `get_node_grad()` 功能相同，但返回引用而非克隆。
    /// 适用于内部计算或只读访问场景。
    pub fn get_node_grad_ref(&self, node_id: NodeId) -> Result<Option<&Tensor>, GraphError> {
        let node = self.get_node(node_id)?;

        // 输入节点不应该有梯度
        if let NodeType::Input(_) = node.node_type() {
            return Err(GraphError::InvalidOperation(format!(
                "输入节点 {node} 不应该有梯度"
            )));
        }

        Ok(node.grad())
    }

    /// 当图拓扑发生变化时调用（添加/删除节点或连接）
    /// 这会清除所有反向传播相关的状态（梯度），但保留前向传播的值
    ///
    /// # NEAT 友好性
    /// 这个方法是为神经进化算法（如 NEAT）设计的，在变异操作后调用
    /// 确保后续的前向/反向传播能正确工作
    ///
    /// # 示例
    /// ```ignore
    /// // 1. 初始图已经训练过
    /// graph.forward(loss)?;
    /// graph.backward(loss)?;
    ///
    /// // 2. 动态添加新节点（NEAT 变异）
    /// let new_node = graph.new_parameter_node(&[1, 1], Some("new"))?;
    /// let new_add = graph.new_add_node(&[old_node, new_node], None)?;
    ///
    /// // 3. 通知图拓扑已变化
    /// graph.on_topology_changed();
    ///
    /// // 4. 继续训练
    /// graph.forward(new_loss)?;
    /// graph.backward(new_loss)?;
    /// ```
    pub fn on_topology_changed(&mut self) {
        // 清除所有节点的 grad（反向传播相关状态）
        // 保留 value（前向传播结果）以便复用
        for node in self.nodes.values_mut() {
            let _ = node.clear_grad();
            // 重置节点的反向传播 pass_id，确保下次 backward 会重新计算
            node.set_last_backward_pass_id(0);
        }
        // 注意：不重置 graph 的 last_backward_pass_id，
        // 因为新的 backward 调用会自增 pass_id，从而与所有节点的 0 不匹配，触发重新计算
    }

    // ========== 参数保存/加载 ==========

    /// 参数文件魔数: "OTPR" (Only Torch `PaRams`)
    const PARAMS_MAGIC: &'static [u8; 4] = b"OTPR";
    /// 参数文件版本
    const PARAMS_VERSION: u32 = 1;

    /// 保存所有可训练参数到二进制文件
    ///
    /// 文件格式：
    /// - Header: magic(4) + version(4) + `param_count(4)`
    /// - 每个参数: `name_len(4)` + name + `shape_dims(4)` + shape + data(f32数组)
    ///
    /// # 示例
    /// ```ignore
    /// graph.save_params("model.bin")?;
    /// ```
    pub fn save_params<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphError> {
        let file = File::create(path.as_ref())
            .map_err(|e| GraphError::ComputationError(format!("无法创建参数文件: {e}")))?;
        let mut writer = BufWriter::new(file);

        // 获取所有参数节点
        let param_nodes: Vec<_> = self
            .nodes
            .iter()
            .filter_map(|(&id, node)| match node.node_type() {
                NodeType::Parameter(_) => Some((id, node)),
                _ => None,
            })
            .collect();

        // 写入 Header
        writer
            .write_all(Self::PARAMS_MAGIC)
            .map_err(|e| GraphError::ComputationError(format!("写入魔数失败: {e}")))?;
        writer
            .write_all(&Self::PARAMS_VERSION.to_le_bytes())
            .map_err(|e| GraphError::ComputationError(format!("写入版本失败: {e}")))?;
        writer
            .write_all(&(param_nodes.len() as u32).to_le_bytes())
            .map_err(|e| GraphError::ComputationError(format!("写入参数数量失败: {e}")))?;

        // 写入每个参数
        for (_id, node) in &param_nodes {
            let name = node.name();
            let value = node
                .value()
                .ok_or_else(|| GraphError::ComputationError(format!("参数 {name} 没有值")))?;
            let shape = value.shape();
            let data = value.data_as_slice();

            // 写入名称
            let name_bytes = name.as_bytes();
            writer
                .write_all(&(name_bytes.len() as u32).to_le_bytes())
                .map_err(|e| GraphError::ComputationError(format!("写入名称长度失败: {e}")))?;
            writer
                .write_all(name_bytes)
                .map_err(|e| GraphError::ComputationError(format!("写入名称失败: {e}")))?;

            // 写入形状
            writer
                .write_all(&(shape.len() as u32).to_le_bytes())
                .map_err(|e| GraphError::ComputationError(format!("写入形状维度失败: {e}")))?;
            for &dim in shape {
                writer
                    .write_all(&(dim as u32).to_le_bytes())
                    .map_err(|e| GraphError::ComputationError(format!("写入形状失败: {e}")))?;
            }

            // 写入数据（f32 数组）
            for &val in data {
                writer
                    .write_all(&val.to_le_bytes())
                    .map_err(|e| GraphError::ComputationError(format!("写入数据失败: {e}")))?;
            }
        }

        writer
            .flush()
            .map_err(|e| GraphError::ComputationError(format!("刷新缓冲区失败: {e}")))?;

        Ok(())
    }

    /// 从二进制文件加载参数
    ///
    /// 注意：需要先用代码构建相同结构的图，参数按名称匹配
    ///
    /// # 示例
    /// ```ignore
    /// // 先构建图结构
    /// let mut graph = Graph::new();
    /// let w1 = graph.new_parameter_node(&[784, 128], Some("w1"))?;
    /// // ...
    ///
    /// // 然后加载参数
    /// graph.load_params("model.bin")?;
    /// ```
    pub fn load_params<P: AsRef<Path>>(&mut self, path: P) -> Result<(), GraphError> {
        let file = File::open(path.as_ref())
            .map_err(|e| GraphError::ComputationError(format!("无法打开参数文件: {e}")))?;
        let mut reader = BufReader::new(file);

        // 读取并验证 Header
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| GraphError::ComputationError(format!("读取魔数失败: {e}")))?;
        if &magic != Self::PARAMS_MAGIC {
            return Err(GraphError::ComputationError(
                "无效的参数文件格式（魔数不匹配）".to_string(),
            ));
        }

        let mut version_bytes = [0u8; 4];
        reader
            .read_exact(&mut version_bytes)
            .map_err(|e| GraphError::ComputationError(format!("读取版本失败: {e}")))?;
        let version = u32::from_le_bytes(version_bytes);
        if version != Self::PARAMS_VERSION {
            return Err(GraphError::ComputationError(format!(
                "不支持的参数文件版本: {}（当前支持版本: {}）",
                version,
                Self::PARAMS_VERSION
            )));
        }

        let mut count_bytes = [0u8; 4];
        reader
            .read_exact(&mut count_bytes)
            .map_err(|e| GraphError::ComputationError(format!("读取参数数量失败: {e}")))?;
        let param_count = u32::from_le_bytes(count_bytes);

        // 构建名称到节点ID的映射
        let name_to_id: HashMap<String, NodeId> = self
            .nodes
            .iter()
            .filter_map(|(&id, node)| match node.node_type() {
                NodeType::Parameter(_) => Some((node.name().to_string(), id)),
                _ => None,
            })
            .collect();

        // 读取每个参数
        for _ in 0..param_count {
            // 读取名称
            let mut name_len_bytes = [0u8; 4];
            reader
                .read_exact(&mut name_len_bytes)
                .map_err(|e| GraphError::ComputationError(format!("读取名称长度失败: {e}")))?;
            let name_len = u32::from_le_bytes(name_len_bytes) as usize;

            let mut name_bytes = vec![0u8; name_len];
            reader
                .read_exact(&mut name_bytes)
                .map_err(|e| GraphError::ComputationError(format!("读取名称失败: {e}")))?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| GraphError::ComputationError(format!("名称编码无效: {e}")))?;

            // 读取形状
            let mut shape_dims_bytes = [0u8; 4];
            reader
                .read_exact(&mut shape_dims_bytes)
                .map_err(|e| GraphError::ComputationError(format!("读取形状维度失败: {e}")))?;
            let shape_dims = u32::from_le_bytes(shape_dims_bytes) as usize;

            let mut shape = Vec::with_capacity(shape_dims);
            for _ in 0..shape_dims {
                let mut dim_bytes = [0u8; 4];
                reader
                    .read_exact(&mut dim_bytes)
                    .map_err(|e| GraphError::ComputationError(format!("读取形状失败: {e}")))?;
                shape.push(u32::from_le_bytes(dim_bytes) as usize);
            }

            // 读取数据
            let data_len: usize = shape.iter().product();
            let mut data = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                let mut val_bytes = [0u8; 4];
                reader
                    .read_exact(&mut val_bytes)
                    .map_err(|e| GraphError::ComputationError(format!("读取数据失败: {e}")))?;
                data.push(f32::from_le_bytes(val_bytes));
            }

            // 查找并设置参数
            if let Some(&node_id) = name_to_id.get(&name) {
                let tensor = Tensor::new(&data, &shape);
                self.set_node_value(node_id, Some(&tensor))?;
            }
            // 注意：文件中存在但图中不存在的参数会被忽略（便于迁移学习）
        }

        Ok(())
    }

    // ========== 图描述（describe）==========

    /// 导出图的描述符（用于序列化、可视化、调试）
    ///
    /// 返回 `GraphDescriptor`，包含图的完整拓扑信息
    ///
    /// # 示例
    /// ```ignore
    /// let descriptor = graph.describe();
    /// println!("{}", descriptor.to_json().unwrap());
    /// ```
    pub fn describe(&self) -> GraphDescriptor {
        let mut descriptor = GraphDescriptor::new(&self.name);

        // 按 ID 排序节点，确保输出顺序一致
        let mut node_ids: Vec<_> = self.nodes.keys().copied().collect();
        node_ids.sort_by_key(|id| id.0);

        for node_id in node_ids {
            let node = self.nodes.get(&node_id).unwrap();
            let parents = self
                .backward_edges
                .get(&node_id)
                .map(|ids| ids.iter().map(|id| id.0).collect())
                .unwrap_or_default();

            let output_shape = node.value_expected_shape().to_vec();
            let node_type_desc = self.node_type_to_descriptor(node.node_type());

            // 获取动态形状信息
            let dyn_shape = node.dynamic_expected_shape();
            let dynamic_shape = if dyn_shape.has_dynamic_dims() {
                Some(dyn_shape.dims().to_vec())
            } else {
                None // 固定形状节点不需要额外存储
            };

            let node_desc = NodeDescriptor::new(
                node_id.0,
                node.name(),
                node_type_desc,
                output_shape,
                dynamic_shape,
                parents,
            );

            descriptor.add_node(node_desc);
        }

        descriptor
    }

    /// 保存完整模型（拓扑 JSON + 参数 bin）
    ///
    /// 自动生成两个文件：
    /// - `{path}.json`: 图的拓扑描述（可读）
    /// - `{path}.bin`: 参数数据（紧凑）
    ///
    /// # 示例
    /// ```ignore
    /// graph.save_model("models/mnist")?;
    /// // 生成：models/mnist.json + models/mnist.bin
    /// ```
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphError> {
        let path = path.as_ref();
        let json_path = path.with_extension("json");
        let bin_path = path.with_extension("bin");

        // 1. 保存参数到 bin 文件
        self.save_params(&bin_path)?;

        // 2. 生成描述符并设置 params_file
        let mut descriptor = self.describe();
        descriptor.params_file = Some(bin_path.file_name().map_or_else(
            || "params.bin".to_string(),
            |s| s.to_string_lossy().to_string(),
        ));

        // 3. 保存 JSON
        let json = descriptor
            .to_json()
            .map_err(|e| GraphError::ComputationError(format!("序列化图描述失败: {e}")))?;
        std::fs::write(&json_path, json)
            .map_err(|e| GraphError::ComputationError(format!("写入 JSON 文件失败: {e}")))?;

        Ok(())
    }

    /// 加载模型参数（需要先用代码构建相同结构的图）
    ///
    /// 注意：当前版本不会从 JSON 重建图结构，只加载参数。
    /// 用户需要先用代码构建与保存时相同结构的图，然后调用此方法加载参数。
    ///
    /// # 示例
    /// ```ignore
    /// // 1. 用代码构建图结构（与保存时相同）
    /// let mut graph = build_mnist_model();
    ///
    /// // 2. 加载参数
    /// graph.load_model("models/mnist")?;
    /// ```
    ///
    /// # TODO
    /// 未来版本将支持从 JSON 完整重建图结构，无需预先用代码构建。
    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Result<(), GraphError> {
        let path = path.as_ref();
        let json_path = path.with_extension("json");

        // 1. 读取并解析 JSON
        let json = std::fs::read_to_string(&json_path)
            .map_err(|e| GraphError::ComputationError(format!("读取 JSON 文件失败: {e}")))?;
        let descriptor = GraphDescriptor::from_json(&json)
            .map_err(|e| GraphError::ComputationError(format!("解析图描述失败: {e}")))?;

        // 2. 确定参数文件路径
        let bin_path = if let Some(ref params_file) = descriptor.params_file {
            path.parent().map_or_else(
                || Path::new(params_file).to_path_buf(),
                |p| p.join(params_file),
            )
        } else {
            path.with_extension("bin")
        };

        // 3. 加载参数
        self.load_params(&bin_path)?;

        Ok(())
    }

    // ========== 模型摘要（summary）==========

    /// 打印模型摘要（类似 Keras 的 `model.summary()`）
    ///
    /// 输出格式化的表格，显示所有节点的信息
    ///
    /// # 示例
    /// ```ignore
    /// graph.summary();
    /// // 输出：
    /// // ┌────────────────┬──────────────────┬─────────────┬──────────┬───────────────┐
    /// // │ 节点名称       │ 类型             │ 输出形状    │ 参数量   │ 父节点        │
    /// // ├────────────────┼──────────────────┼─────────────┼──────────┼───────────────┤
    /// // │ x              │ Input            │ [1, 784]    │ -        │ -             │
    /// // ...
    /// ```
    pub fn summary(&self) {
        println!("{}", self.summary_string());
    }

    /// 将模型摘要保存到文件
    ///
    /// 根据文件扩展名自动选择格式：
    /// - `.md` → Markdown 表格
    /// - 其他（`.txt` 等）→ Unicode 文本表格
    ///
    /// # 示例
    /// ```ignore
    /// graph.save_summary("model_summary.txt")?;  // 文本格式
    /// graph.save_summary("model_summary.md")?;   // Markdown 格式
    /// ```
    pub fn save_summary<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphError> {
        let path = path.as_ref();
        let summary = match path.extension().and_then(|e| e.to_str()) {
            Some("md") => self.summary_markdown(),
            _ => self.summary_string(),
        };
        std::fs::write(path, summary)
            .map_err(|e| GraphError::ComputationError(format!("保存摘要文件失败: {e}")))
    }

    /// 返回模型摘要的 Markdown 格式字符串
    pub fn summary_markdown(&self) -> String {
        let desc = self.describe();
        let mut output = String::new();

        // 标题
        output.push_str(&format!("# 模型摘要: {}\n\n", desc.name));

        // 表头
        output.push_str("| 节点名称 | 类型 | 输出形状 | 参数量 | 父节点 |\n");
        output.push_str("|----------|------|----------|--------|--------|\n");

        // 节点行
        for node in &desc.nodes {
            let type_name = Self::type_name(&node.node_type);
            let shape_str = format!("{:?}", node.output_shape);
            let param_str = node
                .param_count
                .map_or_else(|| "-".to_string(), Self::format_number);
            let parent_str = Self::format_parent_names(&desc, &node.parents);

            output.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                node.name, type_name, shape_str, param_str, parent_str
            ));
        }

        // 统计信息
        let total_params = desc.total_params();
        output.push_str(&format!(
            "\n**总参数量**: {}  \n**可训练参数**: {}\n",
            Self::format_number(total_params),
            Self::format_number(total_params)
        ));

        output
    }

    /// 返回模型摘要字符串（Unicode 文本表格，用于控制台输出）
    pub fn summary_string(&self) -> String {
        let desc = self.describe();

        // 计算各列宽度
        let name_width = desc
            .nodes
            .iter()
            .map(|n| Self::display_width(&n.name))
            .max()
            .unwrap_or(8)
            .max(8);
        let type_width = desc
            .nodes
            .iter()
            .map(|n| Self::type_name(&n.node_type).len())
            .max()
            .unwrap_or(8)
            .max(8);
        let shape_width = desc
            .nodes
            .iter()
            .map(|n| format!("{:?}", n.output_shape).len())
            .max()
            .unwrap_or(8)
            .max(8);
        let param_width = 10;
        let parent_width = desc
            .nodes
            .iter()
            .map(|n| Self::format_parent_names(&desc, &n.parents).len())
            .max()
            .unwrap_or(8)
            .max(6);

        let total_width = name_width + type_width + shape_width + param_width + parent_width + 16; // 边框和间距

        let mut output = String::new();

        // 表头
        output.push_str(&format!(
            "┌{}┬{}┬{}┬{}┬{}┐\n",
            "─".repeat(name_width + 2),
            "─".repeat(type_width + 2),
            "─".repeat(shape_width + 2),
            "─".repeat(param_width + 2),
            "─".repeat(parent_width + 2),
        ));
        output.push_str(&format!(
            "│ {:<name_w$} │ {:<type_w$} │ {:<shape_w$} │ {:<param_w$} │ {:<parent_w$} │\n",
            "节点名称",
            "类型",
            "输出形状",
            "参数量",
            "父节点",
            name_w = name_width,
            type_w = type_width,
            shape_w = shape_width,
            param_w = param_width,
            parent_w = parent_width,
        ));
        output.push_str(&format!(
            "├{}┼{}┼{}┼{}┼{}┤\n",
            "─".repeat(name_width + 2),
            "─".repeat(type_width + 2),
            "─".repeat(shape_width + 2),
            "─".repeat(param_width + 2),
            "─".repeat(parent_width + 2),
        ));

        // 节点行
        for node in &desc.nodes {
            let type_name = Self::type_name(&node.node_type);
            let shape_str = format!("{:?}", node.output_shape);
            let param_str = node
                .param_count
                .map_or_else(|| "-".to_string(), Self::format_number);
            let parent_str = Self::format_parent_names(&desc, &node.parents);

            output.push_str(&format!(
                "│ {:<name_w$} │ {:<type_w$} │ {:<shape_w$} │ {:>param_w$} │ {:<parent_w$} │\n",
                node.name,
                type_name,
                shape_str,
                param_str,
                parent_str,
                name_w = name_width,
                type_w = type_width,
                shape_w = shape_width,
                param_w = param_width,
                parent_w = parent_width,
            ));
        }

        // 分隔线
        output.push_str(&format!(
            "├{}┴{}┴{}┴{}┴{}┤\n",
            "─".repeat(name_width + 2),
            "─".repeat(type_width + 2),
            "─".repeat(shape_width + 2),
            "─".repeat(param_width + 2),
            "─".repeat(parent_width + 2),
        ));

        // 统计信息
        let total_params = desc.total_params();
        output.push_str(&format!(
            "│ {:<width$} │\n",
            format!("总参数量: {}", Self::format_number(total_params)),
            width = total_width - 4,
        ));
        output.push_str(&format!(
            "│ {:<width$} │\n",
            format!("可训练参数: {}", Self::format_number(total_params)),
            width = total_width - 4,
        ));

        // 底边
        output.push_str(&format!("└{}┘\n", "─".repeat(total_width - 2)));

        output
    }

    /// 格式化数字为千分位分隔形式
    fn format_number(n: usize) -> String {
        let s = n.to_string();
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result.chars().rev().collect()
    }

    /// 获取节点类型名称
    const fn type_name(node_type: &NodeTypeDescriptor) -> &'static str {
        match node_type {
            NodeTypeDescriptor::BasicInput => "BasicInput",
            NodeTypeDescriptor::TargetInput => "TargetInput",
            NodeTypeDescriptor::SmartInput => "SmartInput",
            NodeTypeDescriptor::RecurrentOutput => "RecurrentOutput",
            NodeTypeDescriptor::Parameter => "Parameter",
            NodeTypeDescriptor::State => "State",
            NodeTypeDescriptor::Identity => "Identity",
            NodeTypeDescriptor::Add => "Add",
            NodeTypeDescriptor::Divide => "Divide",
            NodeTypeDescriptor::Subtract => "Subtract",
            NodeTypeDescriptor::MatMul => "MatMul",
            NodeTypeDescriptor::Multiply => "Multiply",
            NodeTypeDescriptor::Sigmoid => "Sigmoid",
            NodeTypeDescriptor::Softmax => "Softmax",
            NodeTypeDescriptor::Tanh => "Tanh",
            NodeTypeDescriptor::LeakyReLU { .. } => "LeakyReLU",
            NodeTypeDescriptor::Sign => "Sign",
            NodeTypeDescriptor::SoftPlus => "SoftPlus",
            NodeTypeDescriptor::Step => "Step",
            NodeTypeDescriptor::Reshape { .. } => "Reshape",
            NodeTypeDescriptor::Flatten => "Flatten",
            NodeTypeDescriptor::Conv2d { .. } => "Conv2d",
            NodeTypeDescriptor::MaxPool2d { .. } => "MaxPool2d",
            NodeTypeDescriptor::AvgPool2d { .. } => "AvgPool2d",
            NodeTypeDescriptor::Select { .. } => "Select",
            NodeTypeDescriptor::MSELoss => "MSELoss",
            NodeTypeDescriptor::SoftmaxCrossEntropy => "SoftmaxCE",
            NodeTypeDescriptor::ZerosLike => "ZerosLike",
        }
    }

    /// 格式化父节点名称列表
    fn format_parent_names(desc: &GraphDescriptor, parent_ids: &[u64]) -> String {
        if parent_ids.is_empty() {
            "-".to_string()
        } else {
            parent_ids
                .iter()
                .filter_map(|id| desc.nodes.iter().find(|n| n.id == *id))
                .map(|n| n.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        }
    }

    /// 计算字符串显示宽度（考虑中文字符）
    fn display_width(s: &str) -> usize {
        s.chars().map(|c| if c.is_ascii() { 1 } else { 2 }).sum()
    }

    // ========== Graphviz DOT 可视化 ==========

    /// 生成 Graphviz DOT 格式的图描述字符串
    ///
    /// 返回的字符串可用于：
    /// - 在线预览：<https://dreampuf.github.io/GraphvizOnline/>
    /// - 嵌入到其他文档或工具中
    /// - 自定义保存逻辑
    ///
    /// # 推荐
    /// 如果只需保存可视化文件，推荐使用 [`save_visualization`] 方法，
    /// 它会自动生成 `.dot` 文件，并在 Graphviz 可用时生成图像。
    ///
    /// # 节点样式
    /// - **Input**: 椭圆形，浅蓝色
    /// - **Parameter**: 矩形，浅绿色
    /// - **运算节点**: 圆角矩形，浅黄色
    /// - **Loss**: 双椭圆，浅红色
    ///
    /// # 示例
    /// ```ignore
    /// // 获取 DOT 字符串用于自定义处理
    /// let dot = graph.to_dot();
    /// println!("{}", dot);  // 打印到控制台
    ///
    /// // 或者直接保存可视化（推荐）
    /// graph.save_visualization("outputs/model", None)?;
    /// ```
    pub fn to_dot(&self) -> String {
        self.to_dot_with_options(false)
    }

    /// 生成带层分组选项的 DOT 格式字符串
    ///
    /// # 参数
    /// - `group_layers`: 是否将同一层的节点用半透明框分组显示
    pub fn to_dot_with_options(&self, group_layers: bool) -> String {
        let desc = self.describe();
        let mut dot = String::new();

        // 图头部
        dot.push_str("digraph Model {\n");
        dot.push_str("    rankdir=TB;\n"); // 从上到下
        dot.push_str("    newrank=true;\n"); // 允许 rank=same 跨 cluster 正常工作
        // 使用折线（polyline）：边由直线段组成，可斜向转弯（非严格 90 度）
        // 相比 ortho，polyline 对边标签的位置支持更好
        dot.push_str("    splines=polyline;\n");
        dot.push_str("    node [fontname=\"Microsoft YaHei,SimHei,Arial\"];\n");
        dot.push_str("    edge [fontname=\"Microsoft YaHei,SimHei,Arial\"];\n");
        dot.push('\n');

        // 收集 SmartInput 节点及其所有下游节点（这些节点的 batch 维度应显示为 ?）
        let dynamic_batch_nodes = Self::find_dynamic_batch_nodes(&desc);

        // 收集曾经被 detach 过的 SmartInput/RecurrentOutput 节点 ID（用于显示虚线边框）
        use super::nodes::raw_node::InputVariant;
        let ever_detached_nodes: std::collections::HashSet<u64> = self
            .nodes
            .values()
            .filter_map(|node| {
                if let NodeType::Input(
                    InputVariant::Smart(smart) | InputVariant::RecurrentOutput(smart),
                ) = node.node_type()
                {
                    if smart.was_ever_detached() {
                        return Some(node.id().0);
                    }
                }
                None
            })
            .collect();

        // 获取变长场景的调用次数（用于标注多次调用的节点）
        // 如果有多个 RNN 层，取最大的调用次数
        let call_count: Option<usize> = self
            .recurrent_layer_metas
            .iter()
            .map(|m| m.unroll_infos.len())
            .max()
            .filter(|&count| count > 1); // 只有大于 1 次才显示

        // 分离 Model 分组和 Layer 分组
        // 注意：这些分组会在后面更新以包含代表性调用的节点
        let mut model_groups: Vec<LayerGroup> = self
            .layer_groups
            .iter()
            .filter(|g| g.kind == GroupKind::Model)
            .cloned()
            .collect();
        let mut layer_groups: Vec<LayerGroup> = self
            .layer_groups
            .iter()
            .filter(|g| g.kind == GroupKind::Layer)
            .cloned()
            .collect();

        // 收集所有已分组的节点 ID（初始版本，会在后面更新）
        let mut grouped_node_ids: std::collections::HashSet<u64> = if group_layers {
            self.layer_groups
                .iter()
                .flat_map(|g| g.node_ids.iter().map(|id| id.0))
                .collect()
        } else {
            std::collections::HashSet::new()
        };

        // 收集所有需要隐藏的节点 ID（如循环层的非代表性展开节点）
        let mut hidden_node_ids: std::collections::HashSet<u64> = if group_layers {
            self.layer_groups
                .iter()
                .flat_map(|g| g.hidden_node_ids.iter().map(|id| id.0))
                .collect()
        } else {
            std::collections::HashSet::new()
        };

        // 扩展隐藏节点的后代（从非代表性 RNN 输出开始的所有下游节点）
        if group_layers && !self.recurrent_layer_metas.is_empty() {
            // 构建父→子映射
            let mut children_map: std::collections::HashMap<u64, Vec<u64>> =
                std::collections::HashMap::new();
            for node in &desc.nodes {
                for &parent_id in &node.parents {
                    children_map.entry(parent_id).or_default().push(node.id);
                }
            }

            // 收集代表性调用的输入节点 ID（用于标记受保护路径）
            let repr_input_ids: std::collections::HashSet<u64> = self
                .recurrent_layer_metas
                .iter()
                .filter_map(|m| m.unroll_infos.last())
                .map(|info| info.input_node_id.0)
                .collect();

            // BFS 标记代表性路径上的所有节点为"受保护"
            let mut protected_nodes: std::collections::HashSet<u64> =
                repr_input_ids.iter().copied().collect();
            let mut queue: std::collections::VecDeque<u64> = repr_input_ids.iter().copied().collect();
            while let Some(node_id) = queue.pop_front() {
                if let Some(children) = children_map.get(&node_id) {
                    for &child_id in children {
                        if !protected_nodes.contains(&child_id) {
                            protected_nodes.insert(child_id);
                            queue.push_back(child_id);
                        }
                    }
                }
            }

            // 获取需要扩展后代的根节点（非代表性调用的 RNN 输出）
            let hidden_roots = self.hidden_output_nodes();

            // BFS 扩展后代（但跳过受保护节点）
            // 这会将非代表性调用的 FC、Loss 等下游节点添加到 hidden_node_ids
            let mut queue: std::collections::VecDeque<u64> = hidden_roots
                .iter()
                .flat_map(|root| children_map.get(&root.0).into_iter().flatten().copied())
                .filter(|id| !protected_nodes.contains(id))
                .collect();

            while let Some(node_id) = queue.pop_front() {
                if hidden_node_ids.contains(&node_id) || protected_nodes.contains(&node_id) {
                    continue;
                }
                hidden_node_ids.insert(node_id);
                if let Some(children) = children_map.get(&node_id) {
                    for &child_id in children {
                        if !hidden_node_ids.contains(&child_id) && !protected_nodes.contains(&child_id) {
                            queue.push_back(child_id);
                        }
                    }
                }
            }

            // 识别 Loss 节点（不应该包含在 Model 分组中）
            // Loss 节点包括：SoftmaxCrossEntropy、MSELoss 等
            let loss_node_ids: std::collections::HashSet<u64> = desc
                .nodes
                .iter()
                .filter(|n| {
                    matches!(
                        n.node_type,
                        NodeTypeDescriptor::SoftmaxCrossEntropy | NodeTypeDescriptor::MSELoss
                    )
                })
                .map(|n| n.id)
                .collect();

            // 从 protected_nodes 中排除 Loss 节点（Loss 不应该在 Model 内部）
            let model_nodes: std::collections::HashSet<u64> = protected_nodes
                .iter()
                .copied()
                .filter(|id| !loss_node_ids.contains(id))
                .collect();

            // 更新 Layer 分组（如 FC）以包含代表性调用的节点
            // 策略：找到该层的参数节点，然后找到代表性路径上使用这些参数的计算节点
            for layer in &mut layer_groups {
                // 移除被隐藏的节点
                layer.node_ids.retain(|id| !hidden_node_ids.contains(&id.0));

                // 收集该层的参数节点 ID
                let param_node_ids: std::collections::HashSet<u64> = layer
                    .node_ids
                    .iter()
                    .filter(|id| {
                        desc.nodes
                            .iter()
                            .find(|n| n.id == id.0)
                            .map(|n| matches!(n.node_type, NodeTypeDescriptor::Parameter { .. }))
                            .unwrap_or(false)
                    })
                    .map(|id| id.0)
                    .collect();

                if param_node_ids.is_empty() {
                    continue;
                }

                // 找到代表性路径上使用这些参数的计算节点
                // 注意：排除已隐藏的节点
                for &node_id in &model_nodes {
                    if hidden_node_ids.contains(&node_id) {
                        continue;
                    }
                    if let Some(node) = desc.nodes.iter().find(|n| n.id == node_id) {
                        // 检查该节点是否使用了该层的参数
                        let uses_layer_param = node.parents.iter().any(|p| param_node_ids.contains(p));
                        if uses_layer_param && !layer.node_ids.iter().any(|id| id.0 == node_id) {
                            layer.node_ids.push(NodeId(node_id));
                        }
                    }
                }

                // 继续查找使用这些计算节点的后续节点（如 Add 使用 MatMul）
                let mut current_nodes: Vec<u64> = layer
                    .node_ids
                    .iter()
                    .filter(|id| !param_node_ids.contains(&id.0))
                    .map(|id| id.0)
                    .collect();

                while !current_nodes.is_empty() {
                    let mut next_nodes = Vec::new();
                    for &node_id in &model_nodes {
                        if hidden_node_ids.contains(&node_id) {
                            continue;
                        }
                        if layer.node_ids.iter().any(|id| id.0 == node_id) {
                            continue;
                        }
                        if let Some(node) = desc.nodes.iter().find(|n| n.id == node_id) {
                            // 检查该节点是否使用了当前层的计算节点
                            let uses_layer_node = node.parents.iter().any(|p| current_nodes.contains(p));
                            // 排除使用了其他层参数的节点（属于其他层）
                            let uses_other_param = node.parents.iter().any(|p| {
                                desc.nodes
                                    .iter()
                                    .find(|n| n.id == *p)
                                    .map(|n| {
                                        matches!(n.node_type, NodeTypeDescriptor::Parameter { .. })
                                            && !param_node_ids.contains(p)
                                    })
                                    .unwrap_or(false)
                            });
                            if uses_layer_node && !uses_other_param {
                                layer.node_ids.push(NodeId(node_id));
                                next_nodes.push(node_id);
                            }
                        }
                    }
                    current_nodes = next_nodes;
                }
            }

            // 更新 Model 分组以包含代表性路径上的节点（排除 Loss）
            for group in &mut model_groups {
                // 移除非代表性调用的节点
                group.node_ids.retain(|id| !hidden_node_ids.contains(&id.0));
                // 添加代表性调用的节点
                for &node_id in &model_nodes {
                    if !group.node_ids.iter().any(|id| id.0 == node_id) {
                        group.node_ids.push(NodeId(node_id));
                    }
                }
                group.description = format!("{} nodes", group.node_ids.len());
            }

            // 更新 grouped_node_ids 以包含所有分组中的节点
            grouped_node_ids.clear();
            for group in &model_groups {
                for node_id in &group.node_ids {
                    grouped_node_ids.insert(node_id.0);
                }
            }
            for layer in &layer_groups {
                for node_id in &layer.node_ids {
                    grouped_node_ids.insert(node_id.0);
                }
            }

            // 隐藏"孤立"节点：所有子节点都被隐藏的非受保护节点
            loop {
                let mut newly_hidden = Vec::new();
                for node in &desc.nodes {
                    // 跳过已隐藏或受保护的节点
                    if hidden_node_ids.contains(&node.id) || protected_nodes.contains(&node.id) {
                        continue;
                    }
                    // 检查是否有可见的子节点
                    let children = children_map.get(&node.id);
                    let all_children_hidden = children
                        .map(|c| !c.is_empty() && c.iter().all(|id| hidden_node_ids.contains(id)))
                        .unwrap_or(false);
                    if all_children_hidden {
                        newly_hidden.push(node.id);
                    }
                }
                if newly_hidden.is_empty() {
                    break;
                }
                for id in newly_hidden {
                    hidden_node_ids.insert(id);
                }
            }
        }

        // 收集折叠节点信息（node_id -> (min_steps, max_steps)）
        // 注意：每个折叠节点有自己的步数，初始状态节点（ZerosLike）步数为 1，
        //       而时间步计算节点使用层级的 min_steps/max_steps
        let folded_node_info: std::collections::HashMap<u64, (usize, usize)> = if group_layers {
            self.layer_groups
                .iter()
                .flat_map(|g| {
                    let layer_min_s = g.min_steps.unwrap_or(0);
                    let layer_max_s = g.max_steps.unwrap_or(0);
                    g.folded_nodes.iter().map(move |(id, node_steps)| {
                        // 如果节点步数与层级最大步数相同，使用层级范围
                        // 否则使用节点自己的步数（如初始状态节点步数为 1）
                        if *node_steps == layer_max_s {
                            (id.0, (layer_min_s, layer_max_s))
                        } else {
                            (id.0, (*node_steps, *node_steps))
                        }
                    })
                })
                .collect()
        } else {
            std::collections::HashMap::new()
        };

        // 收集 SmartInput 节点的序列长度范围信息（用于变长 RNN 的输入节点显示）
        // 格式: node_id -> (min_seq_len, max_seq_len)
        let input_seq_range_info: std::collections::HashMap<u64, (usize, usize)> = self
            .recurrent_layer_metas
            .iter()
            .filter(|m| m.unroll_infos.len() > 1) // 只处理变长（多次调用）
            .filter_map(|m| {
                let repr_info = m.unroll_infos.last()?;
                let min_steps = m.unroll_infos.iter().map(|i| i.steps).min()?;
                let max_steps = m.unroll_infos.iter().map(|i| i.steps).max()?;
                Some((repr_info.input_node_id.0, (min_steps, max_steps)))
            })
            .collect();

        // 辅助函数：生成单个节点的 DOT 定义
        let generate_node_def = |node: &crate::nn::descriptor::NodeDescriptor,
                                 dynamic_batch_nodes: &std::collections::HashSet<u64>,
                                 ever_detached_nodes: &std::collections::HashSet<u64>,
                                 folded_info: &std::collections::HashMap<u64, (usize, usize)>,
                                 input_seq_range: &std::collections::HashMap<u64, (usize, usize)>,
                                 call_count: Option<usize>|
         -> String {
            let (shape, mut style, mut fillcolor) = Self::dot_node_style(&node.node_type);
            let style_owned: String;
            let fillcolor_owned: String;
            // 检查是否为折叠节点（时间步展开，橙色 ×N 标注）
            if let Some(&(min_steps, max_steps)) = folded_info.get(&node.id) {
                // 折叠节点：使用深色背景，保留原始形状（通过颜色区分）
                fillcolor_owned = Self::darken_color(fillcolor);
                fillcolor = &fillcolor_owned;
                // 修改节点名称，添加 ×N 或 ×min-max 标注（橙色）
                // 如果有 call_count > 1，同时显示蓝色 (×N) 标注
                let use_dynamic_batch = dynamic_batch_nodes.contains(&node.id);
                let label = Self::dot_node_label_html_folded_range(
                    node,
                    use_dynamic_batch,
                    min_steps,
                    max_steps,
                    call_count,
                );
                // 如果有 call_count，使用双边框（表示多次创建）
                if call_count.is_some() {
                    return format!(
                        "\"{}\" [label=<{}> shape={} style={} fillcolor=\"{}\" peripheries=2 fontsize=10];\n",
                        node.id, label, shape, style, fillcolor
                    );
                } else {
                    return format!(
                        "\"{}\" [label=<{}> shape={} style={} fillcolor=\"{}\" fontsize=10];\n",
                        node.id, label, shape, style, fillcolor
                    );
                }
            }

            if matches!(node.node_type, NodeTypeDescriptor::SmartInput)
                && ever_detached_nodes.contains(&node.id)
            {
                style_owned = "\"filled,dashed\"".to_string();
                style = &style_owned;
            }
            let use_dynamic_batch = dynamic_batch_nodes.contains(&node.id);

            // 检查是否为变长 RNN 的输入节点，需要显示序列长度范围
            let is_multi_call_input = input_seq_range.contains_key(&node.id);

            // 检查是否为多次调用的节点
            // - 计算节点（非参数、非输入）
            // - TargetInput 节点（如果 call_count > 1）
            let is_compute_node = !matches!(
                node.node_type,
                NodeTypeDescriptor::Parameter { .. }
                    | NodeTypeDescriptor::SmartInput
                    | NodeTypeDescriptor::TargetInput
                    | NodeTypeDescriptor::BasicInput
            );
            let is_target_input = matches!(node.node_type, NodeTypeDescriptor::TargetInput);

            let label = if let Some(&(min_seq, max_seq)) = input_seq_range.get(&node.id) {
                // 变长输入节点：显示序列范围 + 蓝色调用次数标注
                let count = call_count.unwrap_or(1);
                Self::dot_node_label_html_with_seq_range(node, use_dynamic_batch, min_seq, max_seq, count)
            } else if let Some(count) = call_count {
                // 多次调用的计算节点或 TargetInput 节点
                if is_compute_node || is_target_input {
                    Self::dot_node_label_html_with_call_count(node, use_dynamic_batch, count)
                } else {
                    Self::dot_node_label_html_with_dynamic_batch(node, use_dynamic_batch)
                }
            } else {
                Self::dot_node_label_html_with_dynamic_batch(node, use_dynamic_batch)
            };

            // 多次调用的节点使用双边框：
            // 1. 计算节点（蓝色 (×N) 标注）
            // 2. 变长 RNN 的输入节点（SmartInput，多次创建）
            // 3. TargetInput 节点（如果 call_count > 1）
            let is_multi_call_compute = call_count.is_some() && is_compute_node;
            let is_multi_call_target = call_count.is_some() && is_target_input;
            let use_double_border = is_multi_call_compute || is_multi_call_input || is_multi_call_target;

            if use_double_border {
                format!(
                    "\"{}\" [label=<{}> shape={} style={} fillcolor=\"{}\" peripheries=2 fontsize=10];\n",
                    node.id, label, shape, style, fillcolor
                )
            } else {
                format!(
                    "\"{}\" [label=<{}> shape={} style={} fillcolor=\"{}\" fontsize=10];\n",
                    node.id, label, shape, style, fillcolor
                )
            }
        };

        // 判断一个 Layer 分组是否完全包含在某个 Model 分组中
        let is_layer_in_model = |layer: &LayerGroup, model: &LayerGroup| -> bool {
            layer
                .node_ids
                .iter()
                .all(|nid| model.node_ids.iter().any(|mid| mid.0 == nid.0))
        };

        // 收集每个 Model 分组包含的 Layer 分组
        let model_to_layers: Vec<Vec<usize>> = model_groups
            .iter()
            .map(|model| {
                layer_groups
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, layer)| {
                        if is_layer_in_model(layer, model) {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        // 收集已被嵌套到 Model 中的 Layer 索引
        let nested_layer_indices: std::collections::HashSet<usize> =
            model_to_layers.iter().flatten().copied().collect();

        // 收集已在嵌套 Layer 中定义的节点 ID
        let mut nodes_in_nested_layers: std::collections::HashSet<u64> =
            std::collections::HashSet::new();

        // 如果启用分组，先输出 Model 分组（包含嵌套的 Layer 分组）
        if group_layers && !model_groups.is_empty() {
            for (model_idx, group) in model_groups.iter().enumerate() {
                let cluster_color = Self::model_group_color(model_idx);
                dot.push_str(&format!(
                    "    subgraph cluster_{} {{\n",
                    group.name.replace(['-', '.'], "_")
                ));
                dot.push_str(&format!(
                    "        label=<<B>{}</B><BR/><FONT POINT-SIZE=\"9\">{}: {}</FONT>>;\n",
                    group.name, group.layer_type, group.description
                ));
                dot.push_str("        style=filled;\n");
                dot.push_str(&format!("        fillcolor=\"{cluster_color}\";\n"));
                dot.push_str("        fontname=\"Microsoft YaHei,SimHei,Arial\";\n");
                dot.push_str("        fontsize=11;\n");
                dot.push_str("        margin=12;\n");

                // 在 Model 内部嵌套 Layer 分组
                for &layer_idx in &model_to_layers[model_idx] {
                    let layer = &layer_groups[layer_idx];
                    let layer_color = Self::layer_group_color(layer_idx);
                    let is_recurrent = layer.min_steps.is_some() && layer.max_steps.is_some();

                    dot.push_str(&format!(
                        "        subgraph cluster_{}_{} {{\n",
                        group.name.replace(['-', '.'], "_"),
                        layer.name.replace(['-', '.'], "_")
                    ));

                    // 层标签：只显示层类型和形状信息（时间步数已在节点上标注，无需重复）
                    dot.push_str(&format!(
                        "            label=<<B>{}</B><BR/><FONT POINT-SIZE=\"8\">{}: {}</FONT>>;\n",
                        layer.name, layer.layer_type, layer.description
                    ));

                    // 循环层：使用多重边框表示重叠效果
                    if is_recurrent {
                        dot.push_str("            style=\"filled,bold\";\n");
                        dot.push_str("            peripheries=3;\n");
                        dot.push_str("            penwidth=2;\n");
                    } else {
                        dot.push_str("            style=filled;\n");
                    }
                    dot.push_str(&format!("            fillcolor=\"{layer_color}\";\n"));
                    dot.push_str("            fontname=\"Microsoft YaHei,SimHei,Arial\";\n");
                    dot.push_str("            fontsize=10;\n");
                    dot.push_str("            margin=8;\n");

                    // 在嵌套 Layer 内定义节点
                    for node in &desc.nodes {
                        if layer.node_ids.iter().any(|nid| nid.0 == node.id) {
                            nodes_in_nested_layers.insert(node.id);
                            dot.push_str("            ");
                            dot.push_str(&generate_node_def(
                                node,
                                &dynamic_batch_nodes,
                                &ever_detached_nodes,
                                &folded_node_info,
                                &input_seq_range_info,
                                call_count,
                            ));
                        }
                    }

                    dot.push_str("        }\n");
                }

                // 在 Model 内定义不属于任何嵌套 Layer 的节点
                for node in &desc.nodes {
                    // 跳过隐藏节点（如循环层的非代表性展开节点）
                    if hidden_node_ids.contains(&node.id) {
                        continue;
                    }
                    if group.node_ids.iter().any(|nid| nid.0 == node.id)
                        && !nodes_in_nested_layers.contains(&node.id)
                    {
                        dot.push_str("        ");
                        dot.push_str(&generate_node_def(
                            node,
                            &dynamic_batch_nodes,
                            &ever_detached_nodes,
                            &folded_node_info,
                            &input_seq_range_info,
                            call_count,
                        ));
                    }
                }

                dot.push_str("    }\n\n");
            }
        }

        // 然后输出不属于任何 Model 的独立 Layer 分组
        if group_layers && !layer_groups.is_empty() {
            for (idx, group) in layer_groups.iter().enumerate() {
                // 跳过已嵌套到 Model 中的 Layer
                if nested_layer_indices.contains(&idx) {
                    continue;
                }

                let cluster_color = Self::layer_group_color(idx);
                let is_recurrent = group.min_steps.is_some() && group.max_steps.is_some();

                dot.push_str(&format!(
                    "    subgraph cluster_{} {{\n",
                    group.name.replace(['-', '.'], "_")
                ));

                // 层标签：只显示层类型和形状信息（时间步数已在节点上标注，无需重复）
                dot.push_str(&format!(
                    "        label=<<B>{}</B><BR/><FONT POINT-SIZE=\"9\">{}: {}</FONT>>;\n",
                    group.name, group.layer_type, group.description
                ));

                // 循环层：使用多重边框表示重叠效果
                if is_recurrent {
                    dot.push_str("        style=\"filled,bold\";\n");
                    dot.push_str("        peripheries=3;\n");
                    dot.push_str("        penwidth=2;\n");
                } else {
                    dot.push_str("        style=filled;\n");
                }
                dot.push_str(&format!("        fillcolor=\"{cluster_color}\";\n"));
                dot.push_str("        fontname=\"Microsoft YaHei,SimHei,Arial\";\n");
                dot.push_str("        fontsize=11;\n");
                dot.push_str("        margin=12;\n");

                for node in &desc.nodes {
                    if group.node_ids.iter().any(|nid| nid.0 == node.id) {
                        dot.push_str("        ");
                        dot.push_str(&generate_node_def(
                            node,
                            &dynamic_batch_nodes,
                            &ever_detached_nodes,
                            &folded_node_info,
                            &input_seq_range_info,
                            call_count,
                        ));
                    }
                }

                dot.push_str("    }\n\n");
            }
        }

        // 节点定义（未分组的节点，或不启用分组时的所有节点）
        for node in &desc.nodes {
            // 跳过隐藏节点（如循环层的非代表性展开节点）
            if hidden_node_ids.contains(&node.id) {
                continue;
            }
            if group_layers && grouped_node_ids.contains(&node.id) {
                continue; // 已在 cluster 中定义
            }
            // 使用统一的 generate_node_def 函数
            dot.push_str("    ");
            dot.push_str(&generate_node_def(
                node,
                &dynamic_batch_nodes,
                &ever_detached_nodes,
                &folded_node_info,
                &input_seq_range_info,
                call_count,
            ));
        }

        dot.push('\n');

        // 边定义（从父节点指向子节点）
        // 收集输出代理信息：(真实输出节点 -> 代表性输出节点)
        let output_proxies: std::collections::HashMap<u64, u64> = if group_layers {
            self.layer_groups
                .iter()
                .filter_map(|g| g.output_proxy.map(|(real, repr)| (real.0, repr.0)))
                .collect()
        } else {
            std::collections::HashMap::new()
        };

        // 收集循环层初始状态节点 ID（这些边会用橙色虚线单独绘制）
        // 使用最后一次调用的初始状态节点（与 infer_recurrent_layer_groups 一致）
        let init_state_node_ids: std::collections::HashSet<u64> = if group_layers {
            self.recurrent_layer_metas
                .iter()
                .filter_map(|m| m.unroll_infos.last())
                .flat_map(|i| i.init_state_node_ids.iter().map(|id| id.0))
                .collect()
        } else {
            std::collections::HashSet::new()
        };

        // 收集循环层的输出代理信息及时间步标签
        // 映射：real_output_node_id -> (repr_output_node_id, final_step_label)
        let recurrent_output_labels: std::collections::HashMap<u64, (u64, String)> = if group_layers
        {
            self.recurrent_layer_metas
                .iter()
                .filter_map(|m| {
                    let repr_info = m.unroll_infos.last()?;
                    let min_steps = m.unroll_infos.iter().map(|i| i.steps).min().unwrap();
                    let max_steps = m.unroll_infos.iter().map(|i| i.steps).max().unwrap();
                    let is_var_len = min_steps != max_steps;
                    let last_step = max_steps.saturating_sub(1);
                    let label = if is_var_len {
                        // 变长序列：显示范围 t=min~max
                        let min_last = min_steps.saturating_sub(1);
                        format!("t={}~{}", min_last, last_step)
                    } else {
                        // 固定长度：显示具体时间步
                        format!("t={}", last_step)
                    };
                    // 使用第一个 repr_output_node_id 作为代理节点（RNN/GRU 只有一个，LSTM 用 h）
                    let repr_id = repr_info.repr_output_node_ids.first()?.0;
                    Some((repr_info.real_output_node_id.0, (repr_id, label)))
                })
                .collect()
        } else {
            std::collections::HashMap::new()
        };

        for node in &desc.nodes {
            // 跳过隐藏节点相关的边
            if hidden_node_ids.contains(&node.id) {
                continue;
            }
            // 跳过指向初始状态节点的边（会用橙色虚线单独绘制）
            if init_state_node_ids.contains(&node.id) {
                continue;
            }
            for parent_id in &node.parents {
                // 如果父节点是隐藏的真实输出节点，检查是否有代理
                if hidden_node_ids.contains(parent_id) {
                    // 检查是否是循环层输出（需要添加时间步标签）
                    if let Some((repr_id, label)) = recurrent_output_labels.get(parent_id) {
                        // 添加从代表性节点到当前节点的边（带时间步标签）
                        dot.push_str(&format!(
                            "    \"{}\" -> \"{}\" [color=\"#E67E22\" label=<{}> fontcolor=\"#E67E22\" fontsize=9];\n",
                            repr_id, node.id, label
                        ));
                    } else if let Some(&repr_id) = output_proxies.get(parent_id) {
                        // 添加从代表性节点到当前节点的实线边（普通折叠层的输出）
                        dot.push_str(&format!("    \"{}\" -> \"{}\";\n", repr_id, node.id));
                    }
                    continue;
                }
                dot.push_str(&format!("    \"{}\" -> \"{}\";\n", parent_id, node.id));
            }
        }

        // 循环层的序列内记忆箭头（橙色虚线）
        // 这种表示方式适用于所有循环层：RNN、LSTM、GRU（固定长度和变长）
        if group_layers {
            for meta in &self.recurrent_layer_metas {
                if meta.unroll_infos.is_empty() {
                    continue;
                }

                // 使用最后一次调用作为代表（与 infer_recurrent_layer_groups 一致）
                let repr_info = meta.unroll_infos.last().unwrap();

                // 计算步数范围
                let min_steps = meta.unroll_infos.iter().map(|i| i.steps).min().unwrap();
                let max_steps = meta.unroll_infos.iter().map(|i| i.steps).max().unwrap();
                let is_var_len = min_steps != max_steps;

                // 计算回流边时间步标签
                // - 回流边：t=0~(max_steps-2)，即这些时刻产生的输出会回流给下一时刻
                // - 最终输出边标签在 recurrent_output_labels 中生成
                let recycle_last = max_steps.saturating_sub(2); // 最后一个回流的时间步
                let recycle_label = if is_var_len {
                    // 变长序列：t=0~(min_last~max_last)，表示回流时间步的范围
                    let min_recycle_last = min_steps.saturating_sub(2);
                    format!("t=0~({}~{})", min_recycle_last, recycle_last)
                } else {
                    // 固定长度：t=0~(N-2)
                    format!("t=0~{}", recycle_last)
                };

                // 为每对 (初始状态, 输出) 绘制边
                // - RNN/GRU：1 对 (h0, h_1)
                // - LSTM：2 对 (h0, h_1) 和 (c0, c_1)
                for (idx, init_id) in repr_info.init_state_node_ids.iter().enumerate() {
                    // 1. t=0 边：SmartInput → ZerosLike（初始化）
                    if let Some(init_node_desc) =
                        desc.nodes.iter().find(|n| n.id == init_id.0)
                    {
                        for parent_id in &init_node_desc.parents {
                            dot.push_str(&format!(
                                "    \"{}\" -> \"{}\" [style=dashed color=\"#E67E22\" label=<t=0> fontcolor=\"#E67E22\" fontsize=9];\n",
                                parent_id, init_id.0
                            ));
                        }
                    }

                    // 2. 回流边：output → ZerosLike（t=1~N-1 时使用上一时刻的输出）
                    if let Some(&output_id) = repr_info.repr_output_node_ids.get(idx) {
                        dot.push_str(&format!(
                            "    \"{}\" -> \"{}\" [style=dashed color=\"#E67E22\" label=<{}> fontcolor=\"#E67E22\" fontsize=9 constraint=false];\n",
                            output_id.0, init_id.0, recycle_label
                        ));
                    }
                }

                // 3. 强制所有初始状态节点在同一行（用于 LSTM 的 h0 和 c0）
                // 使用 Graphviz 的 rank=same 约束（配合 newrank=true 选项）
                if repr_info.init_state_node_ids.len() > 1 {
                    let node_ids: Vec<String> = repr_info
                        .init_state_node_ids
                        .iter()
                        .map(|id| format!("\"{}\"", id.0))
                        .collect();
                    dot.push_str(&format!(
                        "    {{ rank=same; {} }}\n",
                        node_ids.join("; ")
                    ));
                }
            }
        }

        // 数据流连接（紫色虚线箭头）
        // SmartInput/RecurrentOutput 节点可能有 gradient_target，表示数据从某个节点"流入"
        for node in self.nodes.values() {
            if let NodeType::Input(
                InputVariant::Smart(smart) | InputVariant::RecurrentOutput(smart),
            ) = node.node_type()
            {
                if let Some(target_id) = smart.gradient_target() {
                    // 绘制虚线箭头：从源节点到 SmartInput/RecurrentOutput（数据流方向）
                    dot.push_str(&format!(
                        "    \"{}\" -> \"{}\" [style=dashed color=\"#1565C0\" label=\"data flow\" fontcolor=\"#1565C0\" fontsize=9];\n",
                        target_id.0, node.id().0
                    ));
                }
            }
        }

        dot.push_str("}\n");

        dot
    }

    /// 获取层分组的背景颜色（半透明）
    fn layer_group_color(index: usize) -> &'static str {
        // 使用柔和的半透明颜色，不同层使用不同颜色以便区分
        const COLORS: &[&str] = &[
            "#E3F2FD80", // 浅蓝
            "#E8F5E980", // 浅绿
            "#FFF3E080", // 浅橙
            "#F3E5F580", // 浅紫
            "#E0F7FA80", // 浅青
            "#FFFDE780", // 浅黄
            "#FCE4EC80", // 浅粉
            "#EFEBE980", // 浅棕
        ];
        COLORS[index % COLORS.len()]
    }

    /// 获取模型分组的背景颜色（半透明，使用更鲜明的暖色调）
    fn model_group_color(index: usize) -> &'static str {
        // Model 分组使用更鲜明的暖色调，与 Layer 分组区分
        const COLORS: &[&str] = &[
            "#FFECB340", // 浅琥珀（更透明）
            "#FFCCBC40", // 浅深橙（更透明）
            "#D1C4E940", // 浅深紫（更透明）
            "#B2DFDB40", // 浅青绿（更透明）
            "#F8BBD040", // 浅粉红（更透明）
            "#DCEDC840", // 浅黄绿（更透明）
        ];
        COLORS[index % COLORS.len()]
    }

    /// 将 DOT 保存到文件（内部方法）
    fn save_dot<P: AsRef<Path>>(&self, path: P, group_layers: bool) -> Result<(), GraphError> {
        let dot = self.to_dot_with_options(group_layers);
        std::fs::write(path.as_ref(), dot)
            .map_err(|e| GraphError::ComputationError(format!("保存 DOT 文件失败: {e}")))
    }

    /// 保存计算图可视化
    ///
    /// 自动生成 `.dot` 文件，若系统安装了 Graphviz 则额外生成图像文件。
    ///
    /// # 参数
    /// - `base_path`: 基础路径（**不含后缀**），如 `"outputs/model"`
    /// - `format`: 可选的图像格式，默认为 PNG
    ///
    /// # 行为
    /// - 始终生成 `{base_path}.dot`
    /// - 若 Graphviz 可用，额外生成 `{base_path}.{format}`（如 `.png`）
    /// - 若 Graphviz 不可用，返回结果中包含安装提示
    ///
    /// # 错误
    /// - 若路径包含后缀（如 `.dot`、`.png`），返回错误并提示正确用法
    ///
    /// # 示例
    /// ```ignore
    /// // 基础用法（生成 model.dot + model.png）
    /// let result = graph.save_visualization("outputs/model", None)?;
    ///
    /// // 指定 SVG 格式（生成 model.dot + model.svg）
    /// let result = graph.save_visualization("outputs/model", Some(ImageFormat::Svg))?;
    ///
    /// // 启用层分组可视化（将 Linear、Conv2d 等层用半透明框分组显示）
    /// let result = graph.save_visualization_grouped("outputs/model", None)?;
    /// ```
    pub fn save_visualization<P: AsRef<Path>>(
        &self,
        base_path: P,
        format: Option<ImageFormat>,
    ) -> Result<VisualizationOutput, GraphError> {
        self.save_visualization_impl(base_path, format, false)
    }

    /// 保存计算图可视化（启用层分组）
    ///
    /// 与 `save_visualization` 相同，但会将同一层的节点用半透明框分组显示。
    /// 层分组信息来自 `linear()`、`conv2d()` 等 Layer API 的注册。
    ///
    /// # 示例
    /// ```ignore
    /// // 启用层分组的可视化
    /// graph.save_visualization_grouped("outputs/model", None)?;
    /// ```
    pub fn save_visualization_grouped<P: AsRef<Path>>(
        &self,
        base_path: P,
        format: Option<ImageFormat>,
    ) -> Result<VisualizationOutput, GraphError> {
        self.save_visualization_impl(base_path, format, true)
    }

    /// 保存计算图可视化的内部实现
    fn save_visualization_impl<P: AsRef<Path>>(
        &self,
        base_path: P,
        format: Option<ImageFormat>,
        group_layers: bool,
    ) -> Result<VisualizationOutput, GraphError> {
        let path = base_path.as_ref();

        // 1. 检查是否包含后缀（不应该有）
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy();
            let hint = if ImageFormat::from_extension(&ext_str).is_some() || ext_str == "dot" {
                format!(
                    "请提供不含后缀的基础路径。\n\
                     例如: \"outputs/model\" 而不是 \"outputs/model.{ext_str}\"\n\
                     库会自动生成 .dot 和图像文件。"
                )
            } else {
                format!(
                    "检测到未知后缀 '.{ext_str}'，请提供不含后缀的基础路径。\n\
                     例如: \"outputs/model\"\n\
                     支持的图像格式: png, svg, pdf"
                )
            };
            return Err(GraphError::InvalidOperation(hint));
        }

        // 2. 生成 .dot 文件
        let dot_path = path.with_extension("dot");
        self.save_dot(&dot_path, group_layers)?;

        // 3. 尝试生成图像（如果 Graphviz 可用）
        let format = format.unwrap_or_default();
        let image_path = path.with_extension(format.extension());

        let (graphviz_available, graphviz_hint, final_image_path) =
            match Self::render_with_graphviz(&dot_path, &image_path, format) {
                Ok(()) => (true, None, Some(image_path)),
                Err(hint) => (false, Some(hint), None),
            };

        Ok(VisualizationOutput {
            dot_path,
            image_path: final_image_path,
            graphviz_available,
            graphviz_hint,
        })
    }

    /// 检测 Graphviz 是否可用
    fn is_graphviz_available() -> bool {
        std::process::Command::new("dot")
            .arg("-V")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// 使用 Graphviz 渲染 DOT 文件为图像
    fn render_with_graphviz(
        dot_path: &Path,
        output_path: &Path,
        format: ImageFormat,
    ) -> Result<(), String> {
        if !Self::is_graphviz_available() {
            return Err("Graphviz 未安装或不在 PATH 中。\n\
                 安装方式:\n\
                 - Windows: winget install graphviz 或 choco install graphviz\n\
                 - macOS: brew install graphviz\n\
                 - Linux: sudo apt install graphviz\n\
                 安装后可用在线预览: https://dreampuf.github.io/GraphvizOnline/"
                .to_string());
        }

        // 执行 dot 命令: dot -Tpng input.dot -o output.png
        let output = std::process::Command::new("dot")
            .arg(format!("-T{}", format.extension()))
            .arg(dot_path)
            .arg("-o")
            .arg(output_path)
            .output();

        match output {
            Ok(result) if result.status.success() => Ok(()),
            Ok(result) => {
                let stderr = String::from_utf8_lossy(&result.stderr);
                Err(format!("Graphviz 渲染失败: {stderr}"))
            }
            Err(e) => Err(format!("执行 Graphviz 命令失败: {e}")),
        }
    }

    /// 找出所有应该显示动态 batch 维度的节点（SmartInput 及其所有下游节点）
    ///
    /// 这些节点在可视化时 batch 维度显示为 `?`，表示 batch 是动态的。
    fn find_dynamic_batch_nodes(desc: &GraphDescriptor) -> std::collections::HashSet<u64> {
        use std::collections::{HashSet, VecDeque};

        let mut dynamic_nodes = HashSet::new();

        // 1. 找出所有 SmartInput 节点
        let router_ids: Vec<u64> = desc
            .nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeTypeDescriptor::SmartInput))
            .map(|n| n.id)
            .collect();

        // 2. 构建邻接表（parent -> children）
        let mut children_map: std::collections::HashMap<u64, Vec<u64>> =
            std::collections::HashMap::new();
        for node in &desc.nodes {
            for &parent_id in &node.parents {
                children_map.entry(parent_id).or_default().push(node.id);
            }
        }

        // 3. BFS 找出所有下游节点
        let mut queue: VecDeque<u64> = router_ids.iter().copied().collect();
        while let Some(node_id) = queue.pop_front() {
            if dynamic_nodes.insert(node_id) {
                // 将所有子节点加入队列
                if let Some(children) = children_map.get(&node_id) {
                    for &child_id in children {
                        queue.push_back(child_id);
                    }
                }
            }
        }

        dynamic_nodes
    }

    /// 获取节点的 DOT 样式 (shape, style, fillcolor)
    const fn dot_node_style(
        node_type: &NodeTypeDescriptor,
    ) -> (&'static str, &'static str, &'static str) {
        match node_type {
            // 基本输入节点：椭圆形，浅蓝色
            NodeTypeDescriptor::BasicInput => ("ellipse", "filled", "#E3F2FD"),
            // 状态节点：圆柱体，浅橙色（与循环边颜色呼应，表示"记忆/存储"）
            NodeTypeDescriptor::State => ("cylinder", "filled", "#FFE0B2"),
            // Identity 节点：椭圆形，虚线边框，浅紫色（用户创建的 detach 边界）
            NodeTypeDescriptor::Identity => ("ellipse", "\"filled,dashed\"", "#E1BEE7"),
            // TargetInput 节点：椭圆形，浅橙色（Loss 的目标值）
            NodeTypeDescriptor::TargetInput => ("ellipse", "filled", "#FFE0B2"),
            // SmartInput 节点：椭圆形，浅灰色（模型入口，曾被 detach 时显示虚线边框）
            NodeTypeDescriptor::SmartInput => ("ellipse", "filled", "#E0E0E0"),
            // 参数节点：矩形，浅绿色
            NodeTypeDescriptor::Parameter => ("box", "filled", "#E8F5E9"),
            // 损失节点：八边形，浅红色（多次调用时通过 peripheries=2 显示双边框）
            NodeTypeDescriptor::MSELoss | NodeTypeDescriptor::SoftmaxCrossEntropy => {
                ("octagon", "filled", "#FFEBEE")
            }
            // 激活函数：菱形，浅橙色
            NodeTypeDescriptor::Sigmoid
            | NodeTypeDescriptor::Tanh
            | NodeTypeDescriptor::LeakyReLU { .. }
            | NodeTypeDescriptor::Sign
            | NodeTypeDescriptor::SoftPlus
            | NodeTypeDescriptor::Step => ("diamond", "filled", "#FFF3E0"),
            // ZerosLike：虚线圆角矩形，浅黄色（占位符：只在 t=0 时使用，之后被隐藏状态替代）
            NodeTypeDescriptor::ZerosLike => ("box", "\"filled,rounded,dashed\"", "#FFFDE7"),
            // 其他运算节点：圆角矩形，浅黄色
            _ => ("box", "\"filled,rounded\"", "#FFFDE7"),
        }
    }

    /// 生成节点的标签（名称 + 类型 + 形状 + 特殊参数）
    /// 生成节点的 HTML 格式标签（类型加粗）
    #[allow(dead_code)]
    fn dot_node_label_html(node: &NodeDescriptor) -> String {
        Self::dot_node_label_html_with_dynamic_batch(node, false)
    }

    /// 生成节点的 HTML 格式标签，支持动态 batch 显示
    ///
    /// # 参数
    /// - `use_dynamic_batch`: 如果为 true，对于没有 `dynamic_shape` 信息的旧节点，
    ///   回退到旧的逻辑（第一维显示为 `?`）
    ///
    /// 优先级：
    /// 1. 如果节点有 `dynamic_shape` 信息，直接使用它（最准确）
    /// 2. 如果 `use_dynamic_batch` 为 true 且没有 `dynamic_shape，使用旧的回退逻辑`
    /// 3. 否则使用固定形状
    fn dot_node_label_html_with_dynamic_batch(
        node: &NodeDescriptor,
        use_dynamic_batch: bool,
    ) -> String {
        let type_name = Self::type_name(&node.node_type);

        // 使用 NodeDescriptor 中存储的动态形状信息
        // 如果有 dynamic_shape，直接使用（不依赖 use_dynamic_batch 参数）
        let shape_str = if node.dynamic_shape.is_some() {
            node.display_shape()
        } else if use_dynamic_batch && node.output_shape.len() > 1 {
            // 回退：旧的逻辑（兼容没有 dynamic_shape 的旧 JSON）
            let shape_parts: Vec<String> = node
                .output_shape
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    if i == 0 {
                        "?".to_string()
                    } else {
                        dim.to_string()
                    }
                })
                .collect();
            format!("[{}]", shape_parts.join(", "))
        } else {
            format!("{:?}", node.output_shape)
        };

        // 根据节点类型添加特殊参数
        let extra_info = match &node.node_type {
            NodeTypeDescriptor::LeakyReLU { alpha } => Some(format!("α={alpha}")),
            _ => None,
        };

        // 使用 HTML 格式：类型加粗
        let mut parts = vec![
            node.name.clone(),
            format!("<B>{}</B>", type_name), // 类型加粗
            shape_str,
        ];

        if let Some(params) = node.param_count {
            parts.push(format!("({} params)", Self::format_number(params)));
        }

        if let Some(info) = extra_info {
            parts.push(info);
        }

        parts.join("<BR/>")
    }

    /// 生成带多次调用标注的节点标签
    /// 名称去掉索引后缀，添加蓝色 (×N) 标注表示调用次数
    fn dot_node_label_html_with_call_count(
        node: &NodeDescriptor,
        use_dynamic_batch: bool,
        call_count: usize,
    ) -> String {
        let type_name = Self::type_name(&node.node_type);

        // 使用 NodeDescriptor 中存储的动态形状信息
        let shape_str = if node.dynamic_shape.is_some() {
            node.display_shape()
        } else if use_dynamic_batch && node.output_shape.len() > 1 {
            let shape_parts: Vec<String> = node
                .output_shape
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    if i == 0 {
                        "?".to_string()
                    } else {
                        dim.to_string()
                    }
                })
                .collect();
            format!("[{}]", shape_parts.join(", "))
        } else {
            format!("{:?}", node.output_shape)
        };

        // 去掉名称中的索引后缀（如 mat_mul_153 -> mat_mul）
        let base_name = Self::strip_index_suffix(&node.name);

        // 使用 HTML 格式：名称后添加蓝色 (×N) 标注
        let mut parts = vec![
            format!("{} <FONT COLOR=\"#1565C0\">(×{})</FONT>", base_name, call_count),
            format!("<B>{}</B>", type_name),
            shape_str,
        ];

        if let Some(params) = node.param_count {
            parts.push(format!("({} params)", Self::format_number(params)));
        }

        parts.join("<BR/>")
    }

    /// 去掉节点名称中的索引后缀（如 mat_mul_153 -> mat_mul）
    fn strip_index_suffix(name: &str) -> String {
        // 找到最后一个下划线，检查后面是否全是数字
        if let Some(last_underscore) = name.rfind('_') {
            let suffix = &name[last_underscore + 1..];
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                return name[..last_underscore].to_string();
            }
        }
        name.to_string()
    }

    /// 生成带序列长度范围的节点标签（用于变长 RNN 的 SmartInput 节点）
    /// 形状显示为 `[?, min-max, ...]` 格式，并添加蓝色调用次数标注
    fn dot_node_label_html_with_seq_range(
        node: &NodeDescriptor,
        use_dynamic_batch: bool,
        min_seq: usize,
        max_seq: usize,
        call_count: usize,
    ) -> String {
        let type_name = Self::type_name(&node.node_type);

        // 使用序列长度范围替换形状中的序列维度
        // 假设形状为 [batch, seq_len, ...]，需要将 seq_len 维度显示为 min-max
        let shape_str = if use_dynamic_batch && node.output_shape.len() >= 2 {
            let shape_parts: Vec<String> = node
                .output_shape
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    if i == 0 {
                        "?".to_string() // batch 维度
                    } else if i == 1 && min_seq != max_seq {
                        format!("{}-{}", min_seq, max_seq) // 序列维度，显示范围
                    } else if i == 1 {
                        dim.to_string() // 序列维度，固定长度
                    } else {
                        dim.to_string() // 其他维度
                    }
                })
                .collect();
            format!("[{}]", shape_parts.join(", "))
        } else if node.output_shape.len() >= 2 && min_seq != max_seq {
            // 不使用动态 batch，但仍需显示序列范围
            let mut shape_parts: Vec<String> = node.output_shape.iter().map(|d| d.to_string()).collect();
            if shape_parts.len() >= 2 {
                shape_parts[1] = format!("{}-{}", min_seq, max_seq);
            }
            format!("[{}]", shape_parts.join(", "))
        } else {
            format!("{:?}", node.output_shape)
        };

        // 移除名称后缀，添加蓝色调用次数标注（与其他多次调用节点一致）
        let base_name = Self::strip_index_suffix(&node.name);
        let name_with_count = format!(
            "{} <FONT COLOR=\"#1565C0\">(×{})</FONT>",
            base_name, call_count
        );

        // 使用 HTML 格式：类型加粗
        let mut parts = vec![
            name_with_count,
            format!("<B>{}</B>", type_name),
            shape_str,
        ];

        if let Some(params) = node.param_count {
            parts.push(format!("({} params)", Self::format_number(params)));
        }

        parts.join("<BR/>")
    }

    /// 生成折叠节点的 HTML 格式标签（名称后添加 ×N 标注，可选蓝色 (×N) 标注）
    ///
    /// # 参数
    /// - `call_count`: 多次创建的次数（如果 > 1，则同时显示蓝色 (×N)）
    fn dot_node_label_html_folded_range(
        node: &NodeDescriptor,
        use_dynamic_batch: bool,
        min_steps: usize,
        max_steps: usize,
        call_count: Option<usize>,
    ) -> String {
        let type_name = Self::type_name(&node.node_type);

        // 使用 NodeDescriptor 中存储的动态形状信息
        let shape_str = if node.dynamic_shape.is_some() {
            node.display_shape()
        } else if use_dynamic_batch && node.output_shape.len() > 1 {
            let shape_parts: Vec<String> = node
                .output_shape
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    if i == 0 {
                        "?".to_string()
                    } else {
                        dim.to_string()
                    }
                })
                .collect();
            format!("[{}]", shape_parts.join(", "))
        } else {
            format!("{:?}", node.output_shape)
        };

        // 从节点名称中提取基础名称（去掉 _1 后缀）
        let base_name = if let Some(pos) = node.name.rfind('_') {
            if node.name[pos + 1..].chars().all(|c| c.is_ascii_digit()) {
                &node.name[..pos]
            } else {
                &node.name
            }
        } else {
            &node.name
        };

        // 使用 HTML 格式：名称后添加 ×N 或 ×min-max 标注
        let steps_str = if min_steps == max_steps {
            format!("×{}", max_steps)
        } else {
            format!("×{}-{}", min_steps, max_steps)
        };

        // 构建名称行：基础名称 + 橙色时间步标注 + 可选蓝色创建次数标注
        let name_line = if let Some(count) = call_count {
            // 双重语义：橙色 ×N（时间步）+ 蓝色 (×N)（创建次数）
            format!(
                "{} <FONT COLOR=\"#E67E22\">{}</FONT> <FONT COLOR=\"#1565C0\">(×{})</FONT>",
                base_name, steps_str, count
            )
        } else {
            // 单一语义：仅橙色 ×N（时间步）
            format!("{} <FONT COLOR=\"#E67E22\">{}</FONT>", base_name, steps_str)
        };

        let mut parts = vec![name_line, format!("<B>{}</B>", type_name), shape_str];

        if let Some(params) = node.param_count {
            parts.push(format!("({} params)", Self::format_number(params)));
        }

        parts.join("<BR/>")
    }

    /// 将颜色值加深（用于折叠节点的背景）
    fn darken_color(hex_color: &str) -> String {
        // 简单实现：对于已知颜色返回预定义的深色版本
        match hex_color {
            "#FFFDE7" => "#FFF9C4".to_string(), // 浅黄 -> 深黄
            "#FFF3E0" => "#FFE0B2".to_string(), // 浅橙 -> 深橙
            "#E8F5E9" => "#C8E6C9".to_string(), // 浅绿 -> 深绿
            "#E3F2FD" => "#BBDEFB".to_string(), // 浅蓝 -> 深蓝
            _ => hex_color.to_string(),
        }
    }

    #[allow(dead_code)]
    fn dot_node_label(node: &NodeDescriptor) -> String {
        let type_name = Self::type_name(&node.node_type);

        // 使用 NodeDescriptor 中的动态形状信息
        let shape_str = if node.dynamic_shape.is_some() {
            node.display_shape()
        } else if matches!(node.node_type, NodeTypeDescriptor::SmartInput)
            && node.output_shape.len() > 1
        {
            // 回退：兼容旧的 JSON（没有 dynamic_shape）
            let shape_parts: Vec<String> = node
                .output_shape
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    if i == 0 {
                        "?".to_string()
                    } else {
                        dim.to_string()
                    }
                })
                .collect();
            format!("[{}]", shape_parts.join(", "))
        } else {
            format!("{:?}", node.output_shape)
        };

        // 根据节点类型添加特殊参数
        let extra_info = match &node.node_type {
            NodeTypeDescriptor::LeakyReLU { alpha } => Some(format!("α={alpha}")),
            _ => None,
        };

        let mut label = if let Some(params) = node.param_count {
            format!(
                "{}\\n{}\\n{}\\n({} params)",
                node.name,
                type_name,
                shape_str,
                Self::format_number(params)
            )
        } else {
            format!("{}\\n{}\\n{}", node.name, type_name, shape_str)
        };

        // 追加特殊参数信息
        if let Some(info) = extra_info {
            label.push_str(&format!("\\n{info}"));
        }

        label
    }

    /// 将 `NodeType` 转换为 `NodeTypeDescriptor`
    fn node_type_to_descriptor(&self, node_type: &NodeType) -> NodeTypeDescriptor {
        use super::nodes::raw_node::InputVariant;
        match node_type {
            NodeType::Input(variant) => {
                // 根据 InputVariant 类型返回不同的描述
                match variant {
                    InputVariant::Data(_) => NodeTypeDescriptor::BasicInput,
                    InputVariant::Target(_) => NodeTypeDescriptor::TargetInput,
                    InputVariant::Smart(_) => NodeTypeDescriptor::SmartInput,
                    InputVariant::RecurrentOutput(_) => NodeTypeDescriptor::RecurrentOutput,
                }
            }
            NodeType::Parameter(_) => NodeTypeDescriptor::Parameter,
            NodeType::State(_) => NodeTypeDescriptor::State,
            NodeType::Identity(_) => NodeTypeDescriptor::Identity,
            NodeType::Add(_) => NodeTypeDescriptor::Add,
            NodeType::Divide(_) => NodeTypeDescriptor::Divide,
            NodeType::Subtract(_) => NodeTypeDescriptor::Subtract,
            NodeType::MatMul(_) => NodeTypeDescriptor::MatMul,
            NodeType::Multiply(_) => NodeTypeDescriptor::Multiply,
            NodeType::Sigmoid(_) => NodeTypeDescriptor::Sigmoid,
            NodeType::Softmax(_) => NodeTypeDescriptor::Softmax,
            NodeType::Tanh(_) => NodeTypeDescriptor::Tanh,
            NodeType::LeakyReLU(node) => NodeTypeDescriptor::LeakyReLU {
                alpha: node.alpha() as f32,
            },
            NodeType::Sign(_) => NodeTypeDescriptor::Sign,
            NodeType::SoftPlus(_) => NodeTypeDescriptor::SoftPlus,
            NodeType::Step(_) => NodeTypeDescriptor::Step,
            NodeType::Reshape(_) => NodeTypeDescriptor::Reshape {
                target_shape: vec![],
            }, // TODO: 获取实际值
            NodeType::Flatten(_) => NodeTypeDescriptor::Flatten,
            NodeType::Conv2d(_) => NodeTypeDescriptor::Conv2d {
                stride: (1, 1),
                padding: (0, 0),
            }, // TODO: 获取实际值
            NodeType::MaxPool2d(_) => NodeTypeDescriptor::MaxPool2d {
                kernel_size: (2, 2),
                stride: (2, 2),
            }, // TODO: 获取实际值
            NodeType::AvgPool2d(_) => NodeTypeDescriptor::AvgPool2d {
                kernel_size: (2, 2),
                stride: (2, 2),
            }, // TODO: 获取实际值
            NodeType::Select(_) => NodeTypeDescriptor::Select { axis: 0, index: 0 }, // TODO: 获取实际值
            NodeType::MSELoss(_) => NodeTypeDescriptor::MSELoss,
            NodeType::SoftmaxCrossEntropy(_) => NodeTypeDescriptor::SoftmaxCrossEntropy,
            NodeType::ZerosLike(_) => NodeTypeDescriptor::ZerosLike,
        }
    }

    const fn generate_valid_node_id(&mut self) -> NodeId {
        // 生成唯一的节点ID
        self.next_id += 1;
        NodeId(self.next_id)
    }

    // 用于调试
    pub fn nodes_count(&self) -> usize {
        self.nodes.len()
    }

    /// 获取所有可训练的参数节点ID
    pub fn get_trainable_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter_map(|(&id, node)| match node.node_type() {
                NodeType::Parameter(_) => Some(id),
                _ => None,
            })
            .collect()
    }

    /// 根据ID获取节点的引用
    pub(in crate::nn) fn get_node(&self, id: NodeId) -> Result<&NodeHandle, GraphError> {
        self.nodes.get(&id).ok_or(GraphError::NodeNotFound(id))
    }

    /// 根据ID获取节点的变引用
    fn get_node_mut(&mut self, id: NodeId) -> Result<&mut NodeHandle, GraphError> {
        self.nodes.get_mut(&id).ok_or(GraphError::NodeNotFound(id))
    }

    /// 获取多个节点的引用
    fn get_nodes<'a>(&'a self, ids: &[NodeId]) -> Result<Vec<&'a NodeHandle>, GraphError> {
        ids.iter().map(|id| self.get_node(*id)).collect()
    }

    /// 根据ID获取节点的值，处理节点查找和值提取
    pub fn get_node_value(&self, id: NodeId) -> Result<Option<&Tensor>, GraphError> {
        Ok(self.get_node(id)?.value())
    }

    /// 根据ID设置节点的值
    pub fn set_node_value(&mut self, id: NodeId, value: Option<&Tensor>) -> Result<(), GraphError> {
        self.get_node_mut(id)?.set_value(value)
    }

    /// 清除指定节点的梯度
    ///
    /// # 用途
    /// - V2 Optimizer 的 `zero_grad()` 只清除它管理的参数的梯度
    pub fn clear_node_grad(&mut self, id: NodeId) -> Result<(), GraphError> {
        self.get_node_mut(id)?.clear_grad()
    }

    /// 获取节点的梯度（VJP 模式）
    ///
    /// 在 VJP 模式下，梯度形状与节点值形状一致，无需转置或 reshape
    pub fn get_node_grad(&self, node_id: NodeId) -> Result<Option<Tensor>, GraphError> {
        let node = self.get_node(node_id)?;

        // 输入节点不应该有梯度
        if let NodeType::Input(_) = node.node_type() {
            return Err(GraphError::InvalidOperation(format!(
                "输入{node}不应该有梯度"
            )));
        }

        // VJP 模式下 grad 已经是正确的形状
        Ok(node.grad().cloned())
    }

    // 节点信息访问的公共方法
    pub fn get_node_name(&self, id: NodeId) -> Result<&str, GraphError> {
        self.get_node(id)
            .map(super::nodes::node_handle::NodeHandle::name)
    }

    pub fn get_node_parents(&self, id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        self.get_node(id)?; // 确保节点存在
        Ok(self.backward_edges.get(&id).cloned().unwrap_or_default())
    }

    pub fn get_node_children(&self, id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        self.get_node(id)?; // 确保节点存在
        Ok(self.forward_edges.get(&id).cloned().unwrap_or_default())
    }

    /// 收集从 output 到 router 之间的所有节点（用于模型分组）
    ///
    /// 使用 BFS 从 output 向上遍历，收集所有可达的节点，直到遇到 router。
    /// 返回的节点列表包含 router 和 output 本身。
    pub fn collect_nodes_between(
        &self,
        router_id: NodeId,
        output_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphError> {
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        queue.push_back(output_id);
        visited.insert(output_id);

        while let Some(current) = queue.pop_front() {
            result.push(current);

            // 如果到达 router，不再继续向上遍历
            if current == router_id {
                continue;
            }

            // 获取父节点并加入队列
            for parent_id in self.get_node_parents(current)? {
                if !visited.contains(&parent_id) {
                    visited.insert(parent_id);
                    queue.push_back(parent_id);
                }
            }
        }

        Ok(result)
    }

    /// 注册一个模型分组（用于可视化时将整个模型的节点框在一起）
    ///
    /// # 参数
    /// - `name`: 模型名称（如 "Generator", "Discriminator"）
    /// - `router_id`: 模型入口节点（SmartInput）
    /// - `output_id`: 模型输出节点
    pub fn register_model_group(
        &mut self,
        name: &str,
        router_id: NodeId,
        output_id: NodeId,
    ) -> Result<(), GraphError> {
        // 检查是否已存在同名分组
        if self.layer_groups.iter().any(|g| g.name == name) {
            return Ok(());
        }

        // 收集模型包含的所有节点
        let node_ids = self.collect_nodes_between(router_id, output_id)?;

        self.layer_groups.push(LayerGroup {
            name: name.to_string(),
            layer_type: "Model".to_string(),
            description: format!("{} nodes", node_ids.len()),
            node_ids,
            kind: GroupKind::Model,
            recurrent_steps: None,
            min_steps: None,
            max_steps: None,
            hidden_node_ids: vec![],
            folded_nodes: vec![],
            output_proxy: None,
        });

        Ok(())
    }

    pub fn is_node_inited(&self, id: NodeId) -> Result<bool, GraphError> {
        self.get_node(id)
            .map(super::nodes::node_handle::NodeHandle::is_inited)
    }

    pub fn get_node_value_shape(&self, id: NodeId) -> Result<Option<&[usize]>, GraphError> {
        Ok(self
            .get_node(id)?
            .value()
            .map(super::super::tensor::Tensor::shape))
    }

    pub fn get_node_value_expected_shape(&self, id: NodeId) -> Result<&[usize], GraphError> {
        Ok(self.get_node(id)?.value_expected_shape())
    }

    pub fn get_node_dynamic_expected_shape(
        &self,
        id: NodeId,
    ) -> Result<crate::nn::shape::DynamicShape, GraphError> {
        Ok(self.get_node(id)?.dynamic_expected_shape())
    }

    pub fn get_node_value_size(&self, id: NodeId) -> Result<Option<usize>, GraphError> {
        Ok(self
            .get_node(id)?
            .value()
            .map(super::super::tensor::Tensor::size))
    }
}

// 图模式相关
impl GraphInner {
    pub const fn set_train_mode(&mut self) {
        self.is_eval_mode = false;
    }

    pub const fn set_eval_mode(&mut self) {
        self.is_eval_mode = true;
    }

    pub const fn is_train_mode(&self) -> bool {
        !self.is_eval_mode
    }

    /// 设置 BPTT 调试模式（仅用于测试）
    #[cfg(test)]
    pub fn set_bptt_debug(&mut self, debug: bool) {
        self.bptt_debug = debug;
    }

    /// 检查是否启用梯度计算（等价于 `is_train_mode()`）
    ///
    /// 在训练模式下返回 `true`，`在评估模式（no_grad` 上下文）中返回 `false`。
    ///
    /// # 示例
    /// ```ignore
    /// assert!(graph.is_grad_enabled());  // 默认训练模式
    /// graph.no_grad_scope(|g| {
    ///     assert!(!g.is_grad_enabled());  // no_grad 上下文
    ///     Ok(())
    /// })?;
    /// ```
    pub const fn is_grad_enabled(&self) -> bool {
        self.is_train_mode()
    }

    // ========== detach 机制 ==========

    /// 将节点标记为 detached，阻止梯度回流到其父节点
    ///
    /// 被 detach 的节点在反向传播时会被视为叶子节点，梯度不会继续向上传播。
    /// 这在 GAN、Actor-Critic 等需要精细控制梯度流向的场景中非常有用。
    ///
    /// # 用途
    /// - **GAN 训练**：训练判别器时 detach 生成器输出，防止 D 的 loss 更新 G
    /// - **Actor-Critic**：Critic 的 value 估计传给 Actor 时需要 detach
    /// - **Target Network**：目标网络的输出需要 detach
    ///
    /// # 示例
    /// ```ignore
    /// // GAN 训练 - 训练判别器
    /// graph.forward(generator_output)?;
    /// graph.detach_node(fake)?;  // 防止 D 的 loss 更新 G
    /// graph.forward(discriminator_on_fake)?;
    /// graph.backward(d_loss)?;
    ///
    /// // 训练生成器时恢复梯度流
    /// graph.attach_node(fake)?;
    /// graph.backward(g_loss)?;
    /// ```
    pub fn detach_node(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        self.get_node_mut(node_id)?.set_detached(true);
        Ok(())
    }

    /// 取消节点的 detach 状态，恢复梯度流
    ///
    /// # 示例
    /// ```ignore
    /// graph.attach_node(fake)?;  // 恢复梯度流
    /// graph.backward(g_loss)?;
    /// ```
    pub fn attach_node(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        self.get_node_mut(node_id)?.set_detached(false);
        Ok(())
    }

    /// 检查节点是否被 detach
    ///
    /// # 返回
    /// - `true`: 节点被 detach，反向传播时不向父节点传播梯度
    /// - `false`: 正常状态，梯度会向父节点传播
    pub fn is_node_detached(&self, node_id: NodeId) -> Result<bool, GraphError> {
        Ok(self.get_node(node_id)?.is_detached())
    }

    // ========== 循环/记忆机制 API（Phase 1） ==========
    //
    // TODO(RNN 重构): 以下 BPTT 相关方法（connect_recurrent, step, backward_through_time 等）
    // 是旧的"显式时间步"设计，目前仅被 LSTM/GRU 层使用。
    //
    // 新的 Rnn 层已改用"展开式设计"（见 src/nn/layer/rnn.rs），不依赖这些方法。
    // 待 LSTM/GRU 也完成展开式重构后，可考虑删除这些底层 API。
    //
    // 相关测试文件（同样待清理）：
    // - src/nn/tests/bptt_pytorch_comparison.rs
    // - src/nn/tests/recurrent_bptt.rs
    // - src/nn/tests/recurrent_basic.rs
    // - src/nn/tests/node_state.rs
    //
    // 保留理由：可能对 NEAT 或其他高级动态网络场景有用。

    /// `声明循环连接：to_node` 在每次 `step()` 时从 `from_node` 的上一时间步值读取
    ///
    /// 这是实现 RNN 等循环网络的基础。循环连接不是普通的边，
    /// 而是声明一种"延迟读取"关系。
    ///
    /// # 参数
    /// - `from_node`: 提供值的节点（源节点）
    /// - `to_node`: 接收上一步值的节点（目标节点，通常是 Input 类型）
    ///
    /// # 示例
    /// ```ignore
    /// // 创建一个简单的循环：hidden 的值在下一步传给 hidden_prev
    /// let hidden_prev = graph.new_basic_input_node(&[hidden_size, 1], "hidden_prev")?;
    /// let hidden = graph.new_add_node(&[...], "hidden")?;
    /// graph.connect_recurrent(hidden, hidden_prev)?;
    /// ```
    ///
    /// # 注意
    /// - `to_node` 通常应该是 Input 节点（用于接收延迟值）
    /// - 一个 `to_node` 只能有一个循环连接源
    pub fn connect_recurrent(
        &mut self,
        from_node: NodeId,
        to_node: NodeId,
    ) -> Result<(), GraphError> {
        // 验证节点存在
        self.get_node(from_node)?;
        self.get_node(to_node)?;

        // 检查 to_node 是否已有循环连接
        if self.recurrent_edges.contains_key(&to_node) {
            return Err(GraphError::InvalidOperation(format!(
                "节点 {} 已经有循环连接源，不能重复声明",
                self.get_node_name(to_node)?
            )));
        }

        // 注册循环连接
        self.recurrent_edges.insert(to_node, from_node);
        Ok(())
    }

    /// 获取节点的循环连接源（如果有）
    pub fn get_recurrent_source(&self, to_node: NodeId) -> Option<NodeId> {
        self.recurrent_edges.get(&to_node).copied()
    }

    /// 检查图中是否有循环连接
    pub fn has_recurrent_edges(&self) -> bool {
        !self.recurrent_edges.is_empty()
    }

    /// 获取当前时间步
    pub const fn current_time_step(&self) -> u64 {
        self.time_step
    }

    /// 执行一个时间步的前向传播
    ///
    /// 与普通的 `forward` 不同，`step` 专门用于循环网络：
    /// 1. 将循环连接的上一步值传递给目标节点
    /// 2. 执行前向传播
    /// 3. 保存当前值用于下一步（双缓冲交换）
    /// 4. 递增时间步计数
    ///
    /// # 参数
    /// - `output_node`: 要计算的输出节点
    ///
    /// # 示例
    /// ```ignore
    /// graph.reset();  // 新序列开始
    /// for input in sequence {
    ///     graph.set_node_value(input_node, Some(&input))?;
    ///     graph.step(output_node)?;
    ///     let output = graph.get_node_value(output_node)?;
    /// }
    /// ```
    pub fn step(&mut self, output_node: NodeId) -> Result<(), GraphError> {
        // 1. 将上一步的值传递给循环连接的目标节点
        for (&to_node, &from_node) in &self.recurrent_edges.clone() {
            let prev_value = self.prev_values.get(&from_node).cloned();
            if let Some(value) = prev_value {
                self.set_node_value(to_node, Some(&value))?;
            }
            // 如果没有上一步值（第一步），to_node 保持其初始值（通常是 0）
        }

        // 2. 执行前向传播
        self.forward(output_node)?;

        // 3. 保存循环连接源节点的当前值，用于下一步
        for &from_node in self.recurrent_edges.values() {
            if let Some(value) = self.get_node_value(from_node)? {
                self.prev_values.insert(from_node, value.clone());
            }
        }

        // 4. 在训练模式下，保存快照用于 BPTT
        if self.is_train_mode() {
            let snapshot = self.capture_snapshot();
            self.step_history.push(snapshot);
        }

        // 5. 递增时间步
        self.time_step += 1;

        Ok(())
    }

    /// 捕获当前所有节点的快照（用于 BPTT）
    fn capture_snapshot(&self) -> HashMap<NodeId, StepSnapshot> {
        self.nodes
            .iter()
            .map(|(&id, node)| {
                (
                    id,
                    StepSnapshot {
                        value: node.value().cloned(),
                    },
                )
            })
            .collect()
    }

    /// 恢复节点值到指定快照（用于 BPTT）
    fn restore_snapshot(&mut self, snapshot: &HashMap<NodeId, StepSnapshot>) {
        for (&node_id, snap) in snapshot {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.set_value_unchecked(snap.value.as_ref());
            }
        }
    }

    /// 重置循环状态，用于新序列开始
    ///
    /// 清除所有双缓冲中的上一步值，将循环目标节点重置为零，并将时间步归零。
    /// 在处理新的序列之前必须调用此方法。
    ///
    /// # 示例
    /// ```ignore
    /// for sequence in sequences {
    ///     graph.reset();  // 每个新序列开始前重置
    ///     for input in sequence {
    ///         graph.step(output)?;
    ///     }
    /// }
    /// ```
    pub fn reset(&mut self) {
        // 清除上一步值缓存
        self.prev_values.clear();

        // 清除 BPTT 历史
        self.step_history.clear();

        // 收集循环目标节点的 ID 和形状
        let to_reset: Vec<(NodeId, Vec<usize>)> = self
            .recurrent_edges
            .keys()
            .filter_map(|&to_node| {
                self.get_node(to_node)
                    .ok()
                    .map(|node| (to_node, node.value_expected_shape().to_vec()))
            })
            .collect();

        // 将循环目标节点重置为零
        for (to_node, shape) in to_reset {
            let zeros = Tensor::zeros(&shape);
            // 忽略错误（边缘情况）
            let _ = self.set_node_value(to_node, Some(&zeros));
        }

        // 重置时间步
        self.time_step = 0;
    }

    // ========== BPTT API（Phase 2） ==========

    /// 获取当前存储的时间步历史长度
    pub const fn history_len(&self) -> usize {
        self.step_history.len()
    }

    /// 清除 BPTT 历史（不重置循环状态）
    ///
    /// 与 `reset()` 不同，此方法只清除历史记录，不影响当前的循环状态和时间步。
    /// 用于 TBPTT 截断后开始新的截断窗口。
    pub fn clear_history(&mut self) {
        self.step_history.clear();
    }

    /// 通过时间反向传播（BPTT）
    ///
    /// 遍历所有存储的时间步，从最后一步向前反向传播，累加梯度到参数节点。
    ///
    /// # 参数
    /// - `target_nodes`: 需要计算梯度的目标节点（通常是参数节点）
    /// - `loss_node`: 损失节点（每个时间步都会从这个节点开始反向传播）
    ///
    /// # 工作原理
    /// ```text
    /// 对于序列 [t=0, t=1, t=2]：
    ///   1. 恢复 t=2 的快照，backward(loss) → 梯度累加到参数
    ///   2. 恢复 t=1 的快照，backward(loss) → 梯度继续累加
    ///   3. 恢复 t=0 的快照，backward(loss) → 梯度继续累加
    /// 最终参数的梯度 = Σ(各时间步的梯度贡献)
    /// ```
    ///
    /// # 示例
    /// ```ignore
    /// // 前向传播整个序列
    /// for input in sequence {
    ///     graph.set_node_value(x, Some(&input))?;
    ///     graph.step(output)?;
    /// }
    ///
    /// // 反向传播整个序列
    /// graph.backward_through_time(&[w, b], loss)?;
    ///
    /// // 更新参数
    /// optimizer.step(&mut graph)?;
    /// graph.zero_grad();
    /// graph.reset();
    /// ```
    pub fn backward_through_time(
        &mut self,
        target_nodes: &[NodeId],
        loss_node: NodeId,
    ) -> Result<(), GraphError> {
        self.backward_through_time_truncated(target_nodes, loss_node, None)
    }

    /// 截断的通过时间反向传播（TBPTT）
    ///
    /// 与 `backward_through_time` 相同，但只反向传播最近的 `truncation_steps` 个时间步。
    /// 用于处理长序列时限制内存使用和梯度消失/爆炸问题。
    ///
    /// # 参数
    /// - `target_nodes`: 需要计算梯度的目标节点
    /// - `loss_node`: 损失节点
    /// - `truncation_steps`: 截断长度，None 表示不截断（等同于 `backward_through_time`）
    ///
    /// # TBPTT 策略
    /// ```text
    /// 序列长度 = 10，truncation = 3
    ///
    /// 方式 1（本实现）：只反向传播最近 3 步
    ///   [t=7, t=8, t=9] → backward
    ///
    /// 方式 2（高级）：分段处理（需要用户自己实现）
    ///   [t=0,1,2] → backward → step
    ///   [t=3,4,5] → backward → step
    ///   [t=6,7,8,9] → backward → step
    /// ```
    pub fn backward_through_time_truncated(
        &mut self,
        target_nodes: &[NodeId],
        loss_node: NodeId,
        truncation_steps: Option<usize>,
    ) -> Result<(), GraphError> {
        if self.step_history.is_empty() {
            return Err(GraphError::InvalidOperation(
                "BPTT 失败：没有时间步历史。请确保在训练模式下调用 step()。".to_string(),
            ));
        }

        // 确定要反向传播的时间步范围
        let total_steps = self.step_history.len();
        let steps_to_backprop = truncation_steps.unwrap_or(total_steps).min(total_steps);
        let start_idx = total_steps - steps_to_backprop;

        // 收集循环边信息: to_node (State, 如 h_prev) -> from_node (如 hidden)
        let recurrent_edges_vec: Vec<(NodeId, NodeId)> = self
            .recurrent_edges
            .iter()
            .map(|(&to, &from)| (to, from))
            .collect();

        // 收集所有 State 节点（循环边的目标节点）
        let state_nodes: Vec<NodeId> = recurrent_edges_vec.iter().map(|(to, _)| *to).collect();

        // 合并目标节点：params + State 节点
        let mut all_targets: Vec<NodeId> = target_nodes.to_vec();
        for state_id in &state_nodes {
            if !all_targets.contains(state_id) {
                all_targets.push(*state_id);
            }
        }

        // 存储来自"未来"时间步的梯度
        // key: source_node (如 hidden), value: dL/d(source[t]) 从 t+1 传来
        let mut incoming_grads: std::collections::HashMap<NodeId, Tensor> =
            std::collections::HashMap::new();

        // 从最后一个时间步向前反向传播
        let is_first_step = |t: usize| t == total_steps - 1;

        #[cfg(test)]
        let debug = self.bptt_debug;
        #[cfg(not(test))]
        let debug = false;

        // 清除参数的 grad（确保干净的累加起点）
        for &param in target_nodes {
            self.get_node_mut(param)?.clear_grad()?;
        }

        for t in (start_idx..total_steps).rev() {
            // 恢复该时间步的快照
            let snapshot = self.step_history[t].clone();
            self.restore_snapshot(&snapshot);

            if debug {
                println!("\n=== BPTT t={t} ===");
                // 打印当前 incoming_grads
                for (node_id, grad) in &incoming_grads {
                    let name = self
                        .get_node(*node_id)
                        .map(|n| n.name().to_string())
                        .unwrap_or_default();
                    println!(
                        "  incoming_grads[{}({})]: {:?}",
                        name,
                        node_id.0,
                        grad.data_as_slice()
                    );
                }
            }

            // 收集传递到上一时间步的梯度（VJP 模式）
            let mut next_incoming_grads: std::collections::HashMap<NodeId, Tensor> =
                std::collections::HashMap::new();

            if is_first_step(t) {
                // === 最后一个时间步：从 loss 反向传播（纯 VJP）===
                // 使用 backward_from_loss_vjp 进行反向传播
                // 这会：1) 累加参数的 grad，2) 返回 State 节点收到的 grad
                let state_grads =
                    self.backward_from_loss_vjp(target_nodes, &state_nodes, loss_node)?;

                if debug {
                    println!("  [t={t}] After backward_from_loss_vjp:");
                    for &param in target_nodes {
                        let name = self
                            .get_node(param)
                            .map(|n| n.name().to_string())
                            .unwrap_or_default();
                        let grad = self.get_node_grad_ref(param).ok().flatten();
                        println!(
                            "    {} grad: {:?}",
                            name,
                            grad.map(|g| g.data_as_slice().to_vec())
                        );
                    }
                }

                // 收集 State grad 用于跨时间传递
                for &(to_node, from_node) in &recurrent_edges_vec {
                    if let Some(state_grad) = state_grads.get(&to_node) {
                        if debug {
                            let to_name = self
                                .get_node(to_node)
                                .map(|n| n.name().to_string())
                                .unwrap_or_default();
                            let from_name = self
                                .get_node(from_node)
                                .map(|n| n.name().to_string())
                                .unwrap_or_default();
                            println!(
                                "  [t={}] State grad {}({}) -> {}({}): {:?}",
                                t,
                                to_name,
                                to_node.0,
                                from_name,
                                from_node.0,
                                state_grad.data_as_slice()
                            );
                        }
                        next_incoming_grads.insert(from_node, state_grad.clone());
                    }
                }
            } else {
                // === 中间时间步：只从 incoming_grad 传播（纯 VJP）===
                if !incoming_grads.is_empty() {
                    for &(to_node, from_node) in &recurrent_edges_vec {
                        if let Some(incoming_grad) = incoming_grads.get(&from_node) {
                            if debug {
                                let from_name = self
                                    .get_node(from_node)
                                    .map(|n| n.name().to_string())
                                    .unwrap_or_default();
                                println!(
                                    "  [t={}] Processing from {} with incoming {:?}",
                                    t,
                                    from_name,
                                    incoming_grad.data_as_slice()
                                );
                            }

                            // 1) 传播参数梯度
                            self.bptt_backward_from_node_vjp(
                                from_node,
                                incoming_grad,
                                target_nodes,
                            )?;

                            if debug {
                                println!("  [t={t}] After param grad propagation:");
                                for &param in target_nodes {
                                    let name = self
                                        .get_node(param)
                                        .map(|n| n.name().to_string())
                                        .unwrap_or_default();
                                    let grad = self.get_node_grad_ref(param).ok().flatten();
                                    println!(
                                        "    {} grad: {:?}",
                                        name,
                                        grad.map(|g| g.data_as_slice().to_vec())
                                    );
                                }
                            }

                            // 2) 传播 State 梯度（用于跨时间传递）
                            let state_grads = self.bptt_propagate_to_state_vjp(
                                from_node,
                                incoming_grad,
                                target_nodes,
                                false, // 参数梯度已由上面处理，不要重复累加
                            )?;

                            if let Some(state_grad) = state_grads.get(&to_node) {
                                if debug {
                                    let to_name = self
                                        .get_node(to_node)
                                        .map(|n| n.name().to_string())
                                        .unwrap_or_default();
                                    println!(
                                        "  [t={}] State {} received grad: {:?}",
                                        t,
                                        to_name,
                                        state_grad.data_as_slice()
                                    );
                                }
                                next_incoming_grads.insert(from_node, state_grad.clone());
                            }
                        }
                    }
                }
            }

            if debug {
                println!("  [t={t}] next_incoming_grads:");
                for (node_id, grad) in &next_incoming_grads {
                    let name = self
                        .get_node(*node_id)
                        .map(|n| n.name().to_string())
                        .unwrap_or_default();
                    println!("    {}({}): {:?}", name, node_id.0, grad.data_as_slice());
                }
            }

            incoming_grads = next_incoming_grads;
        }

        Ok(())
    }

    /// BPTT 辅助方法：从源节点传播梯度到 State 节点（VJP/grad 模式）
    ///
    /// 使用 VJP (Vector-Jacobian Product) 而非完整 Jacobian 矩阵，
    /// 避免 O(N²) 内存开销，支持大 batch/hidden 尺寸的 RNN 训练。
    ///
    /// # 与 Jacobian 版本 (`bptt_propagate_to_state`) 的区别
    /// - Jacobian 模式：构造 N×N 对角矩阵，然后矩阵乘法
    /// - VJP 模式：直接调用 `calc_grad_to_parent，做` O(N) 元素乘法
    ///
    /// # 参数
    /// - `source_node`: 开始传播的节点（如 hidden）
    /// - `initial_grad`: 该节点的上游梯度，形状与节点值相同
    /// - `target_params`: 需要累加 grad 的参数节点
    /// - `accumulate_params`: 是否累加 grad 到参数
    ///
    /// # 返回
    /// `HashMap`<`NodeId`, Tensor>: State 节点 ID -> 该节点收到的 grad
    fn bptt_propagate_to_state_vjp(
        &mut self,
        source_node: NodeId,
        initial_grad: &Tensor,
        target_params: &[NodeId],
        accumulate_params: bool,
    ) -> Result<std::collections::HashMap<NodeId, Tensor>, GraphError> {
        use std::collections::{HashMap, HashSet, VecDeque};

        let mut state_grads: HashMap<NodeId, Tensor> = HashMap::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // 起点：source_node，其 grad = initial_grad
        queue.push_back((source_node, initial_grad.clone()));
        visited.insert(source_node);

        while let Some((node_id, upstream_grad)) = queue.pop_front() {
            // 获取该节点的父节点
            let parent_ids = self.get_node_parents(node_id)?;
            if parent_ids.is_empty() {
                // 叶子节点（可能是 Parameter）
                if target_params.contains(&node_id) && accumulate_params {
                    // 累加到 Parameter 的 grad
                    // 注意：使用 get_node_grad_ref 读取 grad 字段（VJP 模式）
                    let existing_grad = self.get_node_grad_ref(node_id)?;
                    let new_grad = match existing_grad {
                        Some(existing) if existing.shape() == upstream_grad.shape() => {
                            existing + &upstream_grad
                        }
                        _ => upstream_grad.clone(),
                    };
                    self.get_node_mut(node_id)?.set_grad(Some(&new_grad))?;
                }
                continue;
            }

            // 计算所有父节点的 grad（只读阶段）
            let mut contributions: Vec<(NodeId, Tensor, bool, bool)> = Vec::new();

            {
                let node = self.get_node(node_id)?;
                for &parent_id in &parent_ids {
                    let parent = self.get_node(parent_id)?;

                    // 检查父节点类型
                    let is_input = matches!(parent.node_type(), NodeType::Input(_));
                    let is_state = matches!(parent.node_type(), NodeType::State(_));
                    let is_param = target_params.contains(&parent_id);

                    // 跳过 Input 节点
                    if is_input {
                        continue;
                    }

                    // 使用 VJP 模式计算梯度（关键：使用 calc_grad_to_parent 而非 calc_jacobi_to_a_parent）
                    let assistant_parent = parent_ids.iter().find(|&&id| id != parent_id).copied();
                    let assistant = assistant_parent.map(|id| self.get_node(id)).transpose()?;

                    let local_grad = node.calc_grad_to_parent(parent, &upstream_grad, assistant)?;

                    contributions.push((parent_id, local_grad, is_param, is_state));
                }
            }

            // 处理各类贡献（可变阶段）
            for (parent_id, local_grad, is_param, is_state) in contributions {
                if is_state {
                    // State 节点：收集 grad
                    state_grads
                        .entry(parent_id)
                        .and_modify(|existing| {
                            if existing.shape() == local_grad.shape() {
                                *existing = &*existing + &local_grad;
                            }
                        })
                        .or_insert_with(|| local_grad.clone());
                } else if is_param {
                    // Parameter 节点：根据参数决定是否累加 grad
                    if accumulate_params {
                        // 注意：使用 get_node_grad_ref 读取 grad 字段（VJP 模式）
                        let existing_grad = self.get_node_grad_ref(parent_id)?;
                        let new_grad = match existing_grad {
                            Some(existing) if existing.shape() == local_grad.shape() => {
                                existing + &local_grad
                            }
                            _ => local_grad.clone(),
                        };
                        self.get_node_mut(parent_id)?.set_grad(Some(&new_grad))?;
                    }
                    visited.insert(parent_id);
                } else {
                    // 中间节点：继续向上传播
                    if !visited.contains(&parent_id) {
                        visited.insert(parent_id);
                        queue.push_back((parent_id, local_grad));
                    }
                }
            }
        }

        Ok(state_grads)
    }

    /// BPTT 辅助方法：将 incoming grad 传播到参数（VJP 模式）
    ///
    /// 使用 VJP 而非 Jacobian 模式。与 `bptt_backward_from_node` 类似，但：
    /// - 使用 `calc_grad_to_parent` 而非 `calc_jacobi_to_a_parent`
    /// - 输入是值格式（[batch, hidden]）而非 jacobi 格式（[1, N]）
    /// - 累加到 `grad` 而非 `jacobi`
    fn bptt_backward_from_node_vjp(
        &mut self,
        source_node: NodeId,
        initial_grad: &Tensor,
        target_nodes: &[NodeId],
    ) -> Result<(), GraphError> {
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((source_node, initial_grad.clone()));
        visited.insert(source_node);

        while let Some((node_id, upstream_grad)) = queue.pop_front() {
            let parent_ids = self.get_node_parents(node_id)?;
            if parent_ids.is_empty() {
                continue;
            }

            // 第一阶段：计算所有父节点的贡献（只读）
            let mut contributions: Vec<(NodeId, Tensor, bool)> = Vec::new();

            {
                let node = self.get_node(node_id)?;
                for &parent_id in &parent_ids {
                    // 跳过 Input 和 State 节点
                    let parent = self.get_node(parent_id)?;
                    match parent.node_type() {
                        NodeType::Input(_) => continue,
                        NodeType::State(_) => continue,
                        _ => {}
                    }

                    // 使用 VJP 计算梯度
                    let assistant_parent = parent_ids.iter().find(|&&id| id != parent_id).copied();
                    let assistant = assistant_parent.map(|id| self.get_node(id)).transpose()?;

                    let local_grad = node.calc_grad_to_parent(parent, &upstream_grad, assistant)?;

                    let should_update = target_nodes.contains(&parent_id);
                    contributions.push((parent_id, local_grad, should_update));
                }
            }

            // 第二阶段：更新 grad 和队列（可变）
            for (parent_id, local_grad, should_update) in contributions {
                if should_update {
                    // 目标节点（Parameter）：累加到 grad
                    // 注意：使用 get_node_grad_ref 读取 grad 字段（VJP 模式）
                    let existing_grad = self.get_node_grad_ref(parent_id)?;
                    let new_grad = match existing_grad {
                        Some(existing) if existing.shape() == local_grad.shape() => {
                            existing + &local_grad
                        }
                        _ => local_grad.clone(),
                    };
                    self.get_node_mut(parent_id)?.set_grad(Some(&new_grad))?;
                    visited.insert(parent_id);
                } else {
                    // 非目标节点：继续向上传播
                    if !visited.contains(&parent_id) {
                        visited.insert(parent_id);
                        queue.push_back((parent_id, local_grad));
                    }
                }
            }
        }

        Ok(())
    }

    /// 从 loss 反向传播到目标节点（VJP 模式）
    ///
    /// 使用 VJP 模式计算梯度：
    /// - 梯度存储在 `grad` 字段
    /// - 使用 `calc_grad_to_parent` 计算梯度
    /// - 支持 batch 形状
    ///
    /// # 参数
    /// - `target_params`: 参数节点（累加 grad）
    /// - `state_nodes`: State 节点（收集 grad 用于跨时间传递）
    /// - `loss_node`: loss 节点（反向传播起点）
    ///
    /// # 返回
    /// `HashMap`<`NodeId`, Tensor>: State 节点收到的 grad
    fn backward_from_loss_vjp(
        &mut self,
        target_params: &[NodeId],
        state_nodes: &[NodeId],
        loss_node: NodeId,
    ) -> Result<std::collections::HashMap<NodeId, Tensor>, GraphError> {
        use std::collections::{HashMap, HashSet, VecDeque};

        #[cfg(test)]
        let debug = self.bptt_debug;
        #[cfg(not(test))]
        let debug = false;

        let mut state_grads: HashMap<NodeId, Tensor> = HashMap::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // 起点：loss 节点，其 grad = 1（标量 loss 的初始梯度）
        let loss_value = self
            .get_node_value(loss_node)?
            .ok_or_else(|| GraphError::ComputationError("Loss node has no value".to_string()))?;
        let initial_grad = Tensor::ones(loss_value.shape());

        if debug {
            println!(
                "  backward_from_loss_vjp: loss={}, target_params={:?}, state_nodes={:?}",
                loss_node.0,
                target_params.iter().map(|n| n.0).collect::<Vec<_>>(),
                state_nodes.iter().map(|n| n.0).collect::<Vec<_>>()
            );
        }

        queue.push_back((loss_node, initial_grad));
        visited.insert(loss_node);

        while let Some((node_id, upstream_grad)) = queue.pop_front() {
            let parent_ids = self.get_node_parents(node_id)?;
            if parent_ids.is_empty() {
                continue;
            }

            if debug {
                let node_name = self
                    .get_node(node_id)
                    .map(|n| n.name().to_string())
                    .unwrap_or_default();
                println!(
                    "    Processing node {}({}), upstream_grad={:?}, parents={:?}",
                    node_name,
                    node_id.0,
                    upstream_grad.data_as_slice(),
                    parent_ids.iter().map(|n| n.0).collect::<Vec<_>>()
                );
            }

            // 计算所有父节点的 grad
            let mut contributions: Vec<(NodeId, Tensor, bool, bool)> = Vec::new();

            {
                let node = self.get_node(node_id)?;
                for &parent_id in &parent_ids {
                    let parent = self.get_node(parent_id)?;

                    // 检查父节点类型
                    let is_input = matches!(parent.node_type(), NodeType::Input(_));
                    let is_state = state_nodes.contains(&parent_id);
                    let is_param = target_params.contains(&parent_id);

                    if debug {
                        println!(
                            "      parent {}({}): is_input={}, is_state={}, is_param={}",
                            parent.name(),
                            parent_id.0,
                            is_input,
                            is_state,
                            is_param
                        );
                    }

                    // 跳过 Input 节点
                    if is_input {
                        continue;
                    }

                    // 使用 VJP 计算梯度
                    let assistant_parent = parent_ids.iter().find(|&&id| id != parent_id).copied();
                    let assistant = assistant_parent.map(|id| self.get_node(id)).transpose()?;

                    let local_grad = node.calc_grad_to_parent(parent, &upstream_grad, assistant)?;

                    if debug {
                        println!("        -> local_grad={:?}", local_grad.data_as_slice());
                    }

                    contributions.push((parent_id, local_grad, is_param, is_state));
                }
            }

            // 处理各类贡献
            for (parent_id, local_grad, is_param, is_state) in contributions {
                if is_state {
                    // State 节点：收集 grad（不继续向上，State 是当前时间步的叶子）
                    if debug {
                        println!(
                            "      -> State {}({}): collecting grad",
                            self.get_node(parent_id)
                                .map(|n| n.name().to_string())
                                .unwrap_or_default(),
                            parent_id.0
                        );
                    }
                    state_grads
                        .entry(parent_id)
                        .and_modify(|existing| {
                            if existing.shape() == local_grad.shape() {
                                *existing = &*existing + &local_grad;
                            }
                        })
                        .or_insert_with(|| local_grad.clone());
                } else if is_param {
                    // Parameter 节点：累加到 grad
                    if debug {
                        println!(
                            "      -> Param {}({}): setting grad",
                            self.get_node(parent_id)
                                .map(|n| n.name().to_string())
                                .unwrap_or_default(),
                            parent_id.0
                        );
                    }
                    // 注意：使用 get_node_grad_ref 读取 grad 字段（VJP 模式）
                    let existing_grad = self.get_node_grad_ref(parent_id)?;
                    let new_grad = match existing_grad {
                        Some(existing) if existing.shape() == local_grad.shape() => {
                            existing + &local_grad
                        }
                        _ => local_grad.clone(),
                    };
                    self.get_node_mut(parent_id)?.set_grad(Some(&new_grad))?;
                    visited.insert(parent_id);
                } else {
                    // 中间节点：继续向上传播
                    if debug {
                        println!(
                            "      -> Intermediate {}({}): {}",
                            self.get_node(parent_id)
                                .map(|n| n.name().to_string())
                                .unwrap_or_default(),
                            parent_id.0,
                            if visited.contains(&parent_id) {
                                "already visited"
                            } else {
                                "adding to queue"
                            }
                        );
                    }
                    if !visited.contains(&parent_id) {
                        visited.insert(parent_id);
                        queue.push_back((parent_id, local_grad));
                    }
                }
            }
        }

        Ok(state_grads)
    }

    /// 在 `no_grad` 上下文中执行闭包
    ///
    /// 在此上下文中，图处于评估模式，前向传播不会为反向传播缓存中间值。
    /// 闭包执行完毕后，图会自动恢复到之前的模式。
    ///
    /// # 用途
    /// - **推理/评估**：模型评估时不需要计算梯度
    /// - **性能优化**：跳过梯度追踪相关的开销
    /// - **内存节省**：不存储用于反向传播的中间值
    ///
    /// # 参数
    /// - `f`: 在 `no_grad` 上下文中执行的闭包
    ///
    /// # 返回
    /// 闭包的返回值
    ///
    /// # 示例
    /// ```ignore
    /// // 训练阶段
    /// graph.set_train_mode();
    /// graph.forward(loss)?;
    /// graph.backward(loss)?;
    ///
    /// // 验证阶段（no_grad）
    /// let val_loss = graph.no_grad_scope(|g| {
    ///     g.set_node_value(x, Some(&val_data))?;
    ///     g.forward(output)?;
    ///     let loss_val = g.get_node_value(loss)?.unwrap().data()[0];
    ///     Ok(loss_val)
    /// })?;
    /// ```
    ///
    /// # 嵌套调用
    /// 支持嵌套调用，每次调用都会正确恢复到调用前的状态：
    /// ```ignore
    /// graph.set_train_mode();
    /// graph.no_grad_scope(|g| {
    ///     assert!(!g.is_grad_enabled());
    ///     g.no_grad_scope(|g2| {
    ///         assert!(!g2.is_grad_enabled());
    ///         Ok(())
    ///     })
    /// })?;
    /// assert!(graph.is_grad_enabled());  // 恢复到训练模式
    /// ```
    pub fn no_grad_scope<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        // 保存当前模式
        let was_train = self.is_train_mode();

        // 切换到评估模式（禁用梯度）
        self.set_eval_mode();

        // 执行闭包
        let result = f(self);

        // 恢复之前的模式
        if was_train {
            self.set_train_mode();
        }

        result
    }
}

// 便捷的节点构建方法
impl GraphInner {
    /// 添加节点到列表
    ///
    /// 注意：目前不启用自动节点复用。RNN 层使用自己的内部缓存机制（unroll_cache + output_bridge）。
    /// 全局节点复用机制已准备好，但需要更细粒度的控制才能正确工作（如区分不同 ModelState feature_shape）。
    fn add_node_to_list(
        &mut self,
        mut node_handle: NodeHandle,
        name: Option<&str>,
        node_type: &str,
        parents: &[NodeId],
    ) -> Result<NodeId, GraphError> {
        // 1. 生成节点ID和名称
        let node_id = self.generate_valid_node_id();
        let node_name = self.generate_valid_new_node_name(name.unwrap_or(""), node_type)?;

        // 2. 更新父子关系
        // 2.1 更新正向边：父节点 -> 子节点
        for &parent_id in parents {
            self.forward_edges
                .entry(parent_id)
                .or_default()
                .push(node_id);
        }
        // 2.2 更新反向边：子节点 -> 父节点
        self.backward_edges
            .entry(node_id)
            .or_default()
            .extend(parents);

        // 3. 绑定ID和名称
        node_handle.bind_id_and_name(node_id, &node_name);

        // 4. 将节点句柄插入到节点列表中，并返回ID
        self.nodes.insert(node_id, node_handle);
        Ok(node_id)
    }

    /// 创建基本输入节点（Data 变体）
    pub fn new_basic_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_basic_input(shape)?;
        self.add_node_to_list(node, name, "input", &[])
    }

    /// 创建目标输入节点（Target 变体，用于 Loss 的目标值，支持动态 batch）
    pub fn new_target_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_target_input(shape);
        self.add_node_to_list(node, name, "target", &[])
    }

    /// 创建 SmartInput 节点（智能输入）
    ///
    /// SmartInput 是 ModelState 内部使用的特殊节点，用于实现智能缓存：
    /// - 像 BasicInput 一样存储值（通过 `set_value` 设置）
    /// - 支持动态设置 detached 状态
    /// - 支持梯度路由到外部目标节点
    /// - 支持动态 batch
    ///
    /// # 参数
    /// - `shape`: 节点输出形状
    /// - `name`: 节点名称（可选）
    pub fn new_smart_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_smart_input(shape)?;
        self.add_node_to_list(node, name, "input", &[])
    }

    /// 创建 RecurrentOutput 节点（循环层输出桥接）
    ///
    /// RecurrentOutput 用于 RNN/LSTM/GRU 层的输出桥接：
    /// - 固定的 node_id，使下游层（如 FC）可以复用
    /// - 支持梯度路由到实际的 RNN 输出节点
    /// - 支持动态 batch
    ///
    /// # 参数
    /// - `shape`: 节点输出形状（通常是 `[batch, hidden_size]`）
    /// - `name`: 节点名称（可选）
    pub fn new_recurrent_output_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_recurrent_output(shape)?;
        self.add_node_to_list(node, name, "recurrent_output", &[])
    }

    /// 设置 SmartInput/RecurrentOutput 节点的 detached 状态
    ///
    /// # 参数
    /// - `node_id`: SmartInput/RecurrentOutput 节点 ID
    /// - `detached`: 是否阻止梯度传播
    /// - `mark_ever_detached`: 是否标记 was_ever_detached（用于可视化显示虚线边框）
    pub fn set_router_detached(
        &mut self,
        node_id: NodeId,
        detached: bool,
        mark_ever_detached: bool,
    ) -> Result<(), GraphError> {
        let node = self.get_node_mut(node_id)?;
        node.set_router_detached(detached, mark_ever_detached)
    }

    /// 设置 SmartInput/RecurrentOutput 节点的梯度路由目标
    pub fn set_gradient_target(
        &mut self,
        node_id: NodeId,
        target: Option<NodeId>,
    ) -> Result<(), GraphError> {
        let node = self.get_node_mut(node_id)?;
        node.set_gradient_target(target)
    }

    /// 获取 SmartInput 节点的梯度路由目标
    pub fn get_gradient_target(&self, node_id: NodeId) -> Result<Option<NodeId>, GraphError> {
        let node = self.get_node(node_id)?;
        Ok(node.gradient_target())
    }

    /// 创建参数节点
    ///
    /// 如果 Graph 有种子（通过 `new_with_seed` 或 `set_seed` 设置），
    /// 则使用 Graph 的 RNG 进行参数初始化（确定性）。
    /// 否则使用默认的随机初始化（非确定性）。
    pub fn new_parameter_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        // 如果 Graph 有 RNG，从中生成种子用于参数初始化
        let node = if let Some(ref mut rng) = self.rng {
            use rand::Rng;
            let seed: u64 = rng.r#gen();
            NodeHandle::new_parameter_seeded(shape, seed)?
        } else {
            NodeHandle::new_parameter(shape)?
        };
        self.add_node_to_list(node, name, "parameter", &[])
    }

    /// 使用固定种子创建参数节点（确保可重复性）
    ///
    /// 注意：此方法会覆盖 Graph 的 RNG 设置，直接使用指定的种子。
    /// 适用于需要精确控制单个参数初始化的场景（如单元测试）。
    pub fn new_parameter_node_seeded(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
        seed: u64,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_parameter_seeded(shape, seed)?;
        self.add_node_to_list(node, name, "parameter", &[])
    }

    /// 创建 State 节点（用于 RNN 时间状态）
    ///
    /// State 节点用于存储 RNN/LSTM 的隐藏状态（h, c 等）。
    /// 与 Input 节点不同，State 节点：
    /// - 可以接收并存储 jacobi（用于 BPTT 梯度传递）
    /// - 值由执行引擎（step/reset）管理，而非用户直接输入
    ///
    /// 与 Parameter 节点不同，State 节点：
    /// - 不被优化器更新（不在 `get_trainable_nodes()` 中返回）
    /// - 值在每个时间步都会变化
    ///
    /// # 参数
    /// - `shape`: 状态张量形状，如 `[batch, hidden_size]`
    /// - `name`: 可选的节点名称
    ///
    /// # 示例
    /// ```ignore
    /// let h_prev = graph.new_state_node(&[1, 64], Some("hidden"))?;
    /// graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 64])))?; // 初始化
    /// graph.connect_recurrent(hidden, h_prev)?; // 建立循环连接
    /// ```
    pub fn new_state_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_state(shape)?;
        self.add_node_to_list(node, name, "state", &[])
    }

    /// 创建 `ZerosLike` 节点（动态零张量）
    ///
    /// `ZerosLike` 节点在每次 forward 时根据参考节点的 `batch_size` 生成零张量。
    /// 用于 RNN/LSTM/GRU 的初始隐藏状态。
    ///
    /// # 参数
    /// - `reference`: 参考节点 ID（用于获取 `batch_size`）
    /// - `feature_shape`: 输出的特征维度（不包括 batch）
    /// - `name`: 可选的节点名称
    ///
    /// # 示例
    /// ```ignore
    /// // 创建一个输出 [?, hidden_size] 形状的零张量节点
    /// let h0 = graph.new_zeros_like_node(input_id, &[hidden_size], Some("h0"))?;
    /// ```
    pub fn new_zeros_like_node(
        &mut self,
        reference: NodeId,
        feature_shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_zeros_like(feature_shape);
        self.add_node_to_list(node, name, "zeros_like", &[reference])
    }

    pub fn new_add_node(
        &mut self,
        parents: &[NodeId],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_add(&self.get_nodes(parents)?)?;
        self.add_node_to_list(handle, name, "add", parents)
    }

    /// 创建 Conv2d（2D 卷积）节点
    ///
    /// # 设计
    /// - **`PyTorch` 风格**：单节点处理多通道，而非 `MatrixSlow` 的每通道独立节点
    /// - 支持 Jacobi 模式（单样本）和 Batch 模式
    ///
    /// # 参数
    /// - `input_id`: 输入节点 ID，形状 `[C_in, H, W]` 或 `[batch, C_in, H, W]`
    /// - `kernel_id`: 卷积核参数节点 ID，形状 `[C_out, C_in, kH, kW]`
    /// - `stride`: 步长 `(sH, sW)`
    /// - `padding`: 零填充 `(pH, pW)`
    /// - `name`: 可选的节点名称
    ///
    /// # 输出形状
    /// - 单样本: `[C_out, H', W']`
    /// - Batch: `[batch, C_out, H', W']`
    /// - 其中 `H' = (H + 2*pH - kH) / sH + 1`
    ///
    /// # 示例
    /// ```ignore
    /// // 创建卷积核参数: 32 个 3x3 卷积核，输入 1 通道
    /// let kernel = graph.new_parameter_node(&[32, 1, 3, 3], Some("conv1_kernel"))?;
    ///
    /// // 输入: [batch, 1, 28, 28]（如 MNIST 图像）
    /// let input = graph.new_basic_input_node(&[batch_size, 1, 28, 28], Some("input"))?;
    ///
    /// // 创建卷积层: stride=1, padding=1（保持尺寸）
    /// let conv_out = graph.new_conv2d_node(input, kernel, (1, 1), (1, 1), Some("conv1"))?;
    /// // 输出形状: [batch, 32, 28, 28]
    /// ```
    pub fn new_conv2d_node(
        &mut self,
        input_id: NodeId,
        kernel_id: NodeId,
        stride: (usize, usize),
        padding: (usize, usize),
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle =
            NodeHandle::new_conv2d(&self.get_nodes(&[input_id, kernel_id])?, stride, padding)?;
        self.add_node_to_list(handle, name, "conv2d", &[input_id, kernel_id])
    }

    /// 创建 MaxPool2d（2D 最大池化）节点
    ///
    /// # 设计
    /// - 在每个池化窗口中取最大值
    /// - 记录最大值位置用于反向传播（稀疏梯度）
    ///
    /// # 参数
    /// - `input_id`: 输入节点 ID，形状 `[C, H, W]` 或 `[batch, C, H, W]`
    /// - `kernel_size`: 池化窗口大小 `(kH, kW)`
    /// - `stride`: 步长 `(sH, sW)`，`None` 时默认等于 `kernel_size`
    /// - `name`: 可选的节点名称
    ///
    /// # 输出形状
    /// - 单样本: `[C, H', W']`
    /// - Batch: `[batch, C, H', W']`
    /// - 其中 `H' = (H - kH) / sH + 1`
    ///
    /// # 示例
    /// ```ignore
    /// // 输入: [batch, 32, 28, 28]
    /// let pool = graph.new_max_pool2d_node(conv_out, (2, 2), None, Some("pool1"))?;
    /// // 输出: [batch, 32, 14, 14]（默认 stride = kernel_size）
    ///
    /// // 自定义 stride
    /// let pool2 = graph.new_max_pool2d_node(input, (3, 3), Some((2, 2)), Some("pool2"))?;
    /// ```
    pub fn new_max_pool2d_node(
        &mut self,
        input_id: NodeId,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle =
            NodeHandle::new_max_pool2d(&self.get_nodes(&[input_id])?, kernel_size, stride)?;
        self.add_node_to_list(handle, name, "max_pool2d", &[input_id])
    }

    /// 创建 AvgPool2d（2D 平均池化）节点
    ///
    /// # 设计
    /// - 计算每个池化窗口内所有值的平均
    /// - 反向传播时梯度均匀分配到窗口内所有位置
    ///
    /// # 参数
    /// - `input_id`: 输入节点 ID，形状 `[C, H, W]` 或 `[batch, C, H, W]`
    /// - `kernel_size`: 池化窗口大小 `(kH, kW)`
    /// - `stride`: 步长 `(sH, sW)`，`None` 时默认等于 `kernel_size`
    /// - `name`: 可选的节点名称
    ///
    /// # 输出形状
    /// - 单样本: `[C, H', W']`
    /// - Batch: `[batch, C, H', W']`
    /// - 其中 `H' = (H - kH) / sH + 1`
    ///
    /// # 示例
    /// ```ignore
    /// // 输入: [batch, 32, 28, 28]
    /// let pool = graph.new_avg_pool2d_node(conv_out, (2, 2), None, Some("pool1"))?;
    /// // 输出: [batch, 32, 14, 14]（默认 stride = kernel_size）
    /// ```
    pub fn new_avg_pool2d_node(
        &mut self,
        input_id: NodeId,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle =
            NodeHandle::new_avg_pool2d(&self.get_nodes(&[input_id])?, kernel_size, stride)?;
        self.add_node_to_list(handle, name, "avg_pool2d", &[input_id])
    }

    pub fn new_mat_mul_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_mat_mul(&self.get_nodes(&[left_node_id, right_node_id])?)?;
        self.add_node_to_list(handle, name, "mat_mul", &[left_node_id, right_node_id])
    }

    /// 创建逐元素乘法节点
    /// 两个父节点必须形状相同
    pub fn new_multiply_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_multiply(&self.get_nodes(&[left_node_id, right_node_id])?)?;
        self.add_node_to_list(handle, name, "multiply", &[left_node_id, right_node_id])
    }

    /// 创建逐元素除法节点
    /// 支持广播：两个父节点形状需广播兼容
    ///
    /// # 参数
    /// - `left_node_id`: 被除数节点 ID
    /// - `right_node_id`: 除数节点 ID
    /// - `name`: 节点名称（可选）
    pub fn new_divide_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_divide(&self.get_nodes(&[left_node_id, right_node_id])?)?;
        self.add_node_to_list(handle, name, "divide", &[left_node_id, right_node_id])
    }

    /// 创建逐元素减法节点
    /// 支持广播：两个父节点形状需广播兼容
    ///
    /// # 参数
    /// - `left_node_id`: 被减数节点 ID
    /// - `right_node_id`: 减数节点 ID
    /// - `name`: 节点名称（可选）
    pub fn new_subtract_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_subtract(&self.get_nodes(&[left_node_id, right_node_id])?)?;
        self.add_node_to_list(handle, name, "subtract", &[left_node_id, right_node_id])
    }

    /// 创建 Flatten 节点
    ///
    /// 将张量展平为 2D，常用于 CNN 与全连接层之间的转换。
    ///
    /// # 参数
    /// - `parent_id`: 父节点 ID
    /// - `keep_first_dim`: 是否保留首维度
    ///   - `true`: 保留首维度（batch），展平其余维度
    ///   - `false`: 完全展平为行向量 `[1, n]`
    /// - `name`: 可选的节点名称
    ///
    /// # 示例
    /// ```ignore
    /// // CNN 输出 [batch, features] 展平（2D 保持不变）
    /// let flat = graph.new_flatten_node(conv_out, true, Some("flatten"))?;
    ///
    /// // 完全展平为行向量
    /// let row_vec = graph.new_flatten_node(input, false, Some("row_vec"))?;
    /// ```
    pub fn new_flatten_node(
        &mut self,
        parent_id: NodeId,
        keep_first_dim: bool,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_flatten(&self.get_nodes(&[parent_id])?, keep_first_dim)?;
        self.add_node_to_list(handle, name, "flatten", &[parent_id])
    }

    /// 创建 Reshape 节点
    ///
    /// 改变张量形状而不改变数据，常用于 CNN 与全连接层之间的转换。
    ///
    /// # 参数
    /// - `parent_id`: 父节点 ID
    /// - `target_shape`: 目标形状（元素总数必须与输入相同）
    /// - `name`: 可选的节点名称
    ///
    /// # 示例
    /// ```ignore
    /// // 将 [batch, 32, 7, 7] reshape 为 [batch, 1568]
    /// let flat = graph.new_reshape_node(conv_out, &[batch_size, 1568], Some("flatten"))?;
    /// ```
    pub fn new_reshape_node(
        &mut self,
        parent_id: NodeId,
        target_shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_reshape(&self.get_nodes(&[parent_id])?, target_shape)?;
        self.add_node_to_list(handle, name, "reshape", &[parent_id])
    }

    /// 创建 Sign 节点
    ///
    /// Sign 函数返回输入的符号：正数→1, 负数→-1, 零→0
    /// 导数在所有点都是0（不可微，与 Step 类似）
    ///
    /// 适用场景：
    /// - 二分类预测输出（使用 {-1, +1} 标签时）
    /// - 信号处理中的符号提取
    pub fn new_sign_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_sign(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "sign", &[parent_id])
    }

    pub fn new_step_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_step(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "step", &[parent_id])
    }

    pub fn new_tanh_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_tanh(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "tanh", &[parent_id])
    }

    /// 创建 Select 节点（从张量中选择指定轴和索引的切片）
    ///
    /// 用于 RNN 展开式设计：从 `[batch, seq_len, input_size]` 中提取单个时间步。
    ///
    /// # 参数
    /// - `parent_id`: 输入节点 ID
    /// - `axis`: 选择的轴
    /// - `index`: 选择的索引
    /// - `name`: 节点名称（可选）
    ///
    /// # 示例
    /// ```ignore
    /// // 从 [batch, seq_len, input_size] 中提取第 t 个时间步
    /// let x_t = graph.new_select_node(x_seq_id, 1, t, Some("x_t"))?;
    /// // 输出形状: [batch, input_size]
    /// ```
    pub fn new_select_node(
        &mut self,
        parent_id: NodeId,
        axis: usize,
        index: usize,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_select(&self.get_nodes(&[parent_id])?, axis, index)?;
        self.add_node_to_list(handle, name, "select", &[parent_id])
    }

    pub fn new_sigmoid_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_sigmoid(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "sigmoid", &[parent_id])
    }

    /// 创建 Identity 节点（恒等映射）
    ///
    /// 直接传递父节点的值，不做任何变换。
    /// 主要用于 `detach()` 操作：创建一个 detached 的 Identity 节点来阻断梯度流。
    ///
    /// # 参数
    /// - `parent_id`: 父节点 ID
    /// - `name`: 节点名称（可选）
    /// - `detached`: 是否阻断梯度流
    pub fn new_identity_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
        detached: bool,
    ) -> Result<NodeId, GraphError> {
        let mut handle = NodeHandle::new_identity(&self.get_nodes(&[parent_id])?)?;
        if detached {
            handle.set_detached(true);
        }
        self.add_node_to_list(handle, name, "identity", &[parent_id])
    }

    /// 创建 Softmax 激活节点
    ///
    /// 对输入张量沿最后一维计算 softmax。
    ///
    /// # 参数
    /// - `parent_id`: 输入节点 ID，形状 [batch, `num_classes`]
    /// - `name`: 节点名称（可选）
    pub fn new_softmax_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_softmax(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "softmax", &[parent_id])
    }

    /// 创建 `SoftPlus` 激活节点
    ///
    /// `SoftPlus` 是 `ReLU` 的平滑近似: f(x) = ln(1 + e^x)
    /// 导数为 sigmoid: f'(x) = 1 / (1 + e^(-x))
    ///
    /// 适用场景：
    /// - 需要正值输出（如方差/标准差预测）
    /// - 需要平滑梯度的优化
    /// - 概率模型（VAE）、连续动作空间 RL（SAC/PPO）
    pub fn new_softplus_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_softplus(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "softplus", &[parent_id])
    }

    /// 创建 `LeakyReLU` 激活节点
    ///
    /// # 参数
    /// - `parent_id`: 父节点 ID
    /// - `negative_slope`: 负半轴斜率（0.0 时等价于标准 ReLU，MatrixSlow 使用 0.1）
    /// - `name`: 节点名称（可选）
    pub fn new_leaky_relu_node(
        &mut self,
        parent_id: NodeId,
        negative_slope: f64,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_leaky_relu(&self.get_nodes(&[parent_id])?, negative_slope)?;
        self.add_node_to_list(handle, name, "leaky_relu", &[parent_id])
    }

    /// 创建标准 `ReLU` 激活节点（等价于 `negative_slope=0` 的 `LeakyReLU`）
    ///
    /// # 参数
    /// - `parent_id`: 父节点 ID
    /// - `name`: 节点名称（可选）
    pub fn new_relu_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        self.new_leaky_relu_node(parent_id, 0.0, name)
    }

    /// 创建 `SoftmaxCrossEntropy` 损失节点
    ///
    /// # 参数
    /// - `logits_id`: 预测值节点 ID（未经 softmax 的原始分数）
    /// - `labels_id`: 标签节点 ID（one-hot 编码）
    /// - `name`: 可选的节点名称
    ///
    /// # 返回
    /// 新创建的损失节点 ID，输出为标量 [1, 1]
    pub fn new_softmax_cross_entropy_node(
        &mut self,
        logits_id: NodeId,
        labels_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[logits_id, labels_id])?;
        let handle = NodeHandle::new_softmax_cross_entropy(&parents)?;
        self.add_node_to_list(handle, name, "softmax_ce", &[logits_id, labels_id])
    }

    /// 创建 MSELoss（均方误差损失）节点
    ///
    /// 使用默认的 Mean reduction 模式。
    ///
    /// # 参数
    /// - `input_id`: 预测值节点 ID
    /// - `target_id`: 目标值节点 ID
    /// - `name`: 可选的节点名称
    ///
    /// # 返回
    /// 新创建的损失节点 ID，输出为标量 [1, 1]
    ///
    /// # 公式
    /// `MSE = mean((input - target)^2)`
    pub fn new_mse_loss_node(
        &mut self,
        input_id: NodeId,
        target_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[input_id, target_id])?;
        let handle = NodeHandle::new_mse_loss(&parents)?;
        self.add_node_to_list(handle, name, "mse_loss", &[input_id, target_id])
    }

    /// 创建 `MSELoss` 节点（指定 reduction 模式）
    ///
    /// # 参数
    /// - `input_id`: 预测值节点 ID
    /// - `target_id`: 目标值节点 ID
    /// - `reduction`: Reduction 模式（Mean 或 Sum）
    /// - `name`: 可选的节点名称
    ///
    /// # 返回
    /// 新创建的损失节点 ID，输出为标量 [1, 1]
    pub fn new_mse_loss_node_with_reduction(
        &mut self,
        input_id: NodeId,
        target_id: NodeId,
        reduction: Reduction,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[input_id, target_id])?;
        let handle = NodeHandle::new_mse_loss_with_reduction(&parents, reduction)?;
        self.add_node_to_list(handle, name, "mse_loss", &[input_id, target_id])
    }
}

/// 图错误类型
#[derive(Debug, PartialEq, Eq)]
pub enum GraphError {
    GraphNotFound(String),
    NodeNotFound(NodeId),
    InvalidOperation(String),
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
        message: String,
    },
    DimensionMismatch {
        expected: usize,
        got: usize,
        message: String,
    },
    ComputationError(String),
    DuplicateName(String),
    DuplicateNodeName(String),
}

// ========== 可视化相关类型 ==========

/// 图像输出格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFormat {
    /// PNG 格式（默认）
    #[default]
    Png,
    /// SVG 矢量格式
    Svg,
    /// PDF 格式
    Pdf,
}

impl ImageFormat {
    /// 获取文件扩展名（不含点号）
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Svg => "svg",
            Self::Pdf => "pdf",
        }
    }

    /// 从扩展名解析格式（用于错误提示）
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "png" => Some(Self::Png),
            "svg" => Some(Self::Svg),
            "pdf" => Some(Self::Pdf),
            _ => None,
        }
    }
}

/// 可视化输出结果
#[derive(Debug)]
pub struct VisualizationOutput {
    /// DOT 文件路径（始终生成）
    pub dot_path: std::path::PathBuf,
    /// 图像文件路径（仅当 Graphviz 可用时生成）
    pub image_path: Option<std::path::PathBuf>,
    /// Graphviz 是否可用
    pub graphviz_available: bool,
    /// 如果 Graphviz 不可用，提供安装提示
    pub graphviz_hint: Option<String>,
}
