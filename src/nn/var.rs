/*
 * @Author       : 老董
 * @Date         : 2026-01-08
 * @Description  : Smart Var - 智能变量句柄，支持算子重载和链式调用
 *
 * 这是架构的核心组件，提供 PyTorch 级用户体验。
 */

use super::graph::{Graph, GraphInner};
use super::nodes::NodeInner;
use super::{GraphError, NodeId};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::{Rc, Weak};

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
/// # 设计原则（方案 C 正式版本）
/// - 持有 `Rc<NodeInner>` 直接控制节点生命周期
/// - 持有 `Weak<RefCell<GraphInner>>` 引用用于全局配置
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
    /// 节点内部结构 - 直接持有节点，控制生命周期
    node: Rc<NodeInner>,
    /// 图引用（Weak 引用，不阻止 Graph 释放）
    graph: Weak<RefCell<GraphInner>>,
}

impl std::fmt::Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Var")
            .field("id", &self.node.id())
            .field("name", &self.node.name())
            .finish()
    }
}

impl Var {
    /// 创建新的 Var（方案 C 正式版本）
    ///
    /// 直接持有 NodeInner，graph 为 Weak 引用
    pub(crate) fn new(node: Rc<NodeInner>, graph: Weak<RefCell<GraphInner>>) -> Self {
        Self { node, graph }
    }

    /// 从 Rc<RefCell<GraphInner>> 创建 Var（便捷方法）
    pub(crate) fn new_with_rc_graph(node: Rc<NodeInner>, graph: &Rc<RefCell<GraphInner>>) -> Self {
        Self {
            node,
            graph: Rc::downgrade(graph),
        }
    }

    /// 获取节点 ID
    pub fn node_id(&self) -> NodeId {
        self.node.id()
    }

    /// 获取 NodeInner 的引用
    pub(crate) fn node(&self) -> &Rc<NodeInner> {
        &self.node
    }

    /// 获取内部图引用（升级 Weak 为 Rc）
    ///
    /// # Panics
    /// 如果 Graph 已被释放，则 panic
    pub(crate) fn graph(&self) -> Rc<RefCell<GraphInner>> {
        self.graph.upgrade().expect("Graph 已被释放，Var 不再有效")
    }

    /// 尝试获取内部图引用（不 panic）
    pub(crate) fn try_graph(&self) -> Option<Rc<RefCell<GraphInner>>> {
        self.graph.upgrade()
    }

    /// 检查两个 Var 是否来自同一个 Graph
    pub fn same_graph(&self, other: &Self) -> bool {
        match (self.graph.upgrade(), other.graph.upgrade()) {
            (Some(a), Some(b)) => Rc::ptr_eq(&a, &b),
            _ => false, // 任一 Graph 已释放，认为不同
        }
    }

    /// 获取 Var 所属的 Graph handle
    ///
    /// # Panics
    /// 如果 Graph 已被释放，则 panic
    pub fn get_graph(&self) -> super::graph::Graph {
        super::graph::Graph::from_rc(self.graph())
    }

    /// 获取节点的预期输出形状
    pub fn value_expected_shape(&self) -> Vec<usize> {
        self.node.shape()
    }

    /// 获取节点的动态形状
    ///
    /// 返回支持动态维度的形状表示（如 `[?, 128]`）
    pub fn dynamic_expected_shape(&self) -> crate::nn::shape::DynamicShape {
        self.node().dynamic_expected_shape()
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
    /// 创建一个 detached 的 Var（与 PyTorch `tensor.detach()` 语义一致）
    ///
    /// 在图中创建一个新的 Identity 节点，标记为 detached。
    /// 返回的 Var：
    /// - 与原 Var 共享前向计算的值
    /// - 反向传播时，梯度不会通过此节点传回原 Var
    /// - 可以继续参与后续的图计算
    ///
    /// # 示例
    /// ```ignore
    /// // GAN 训练：训练 D 时阻止梯度流向 G
    /// let fake_images = G.forward(&noise)?;
    /// let d_fake = D.forward(&fake_images.detach())?;  // 梯度阻断
    /// d_loss.backward()?;  // 梯度不会流向 G
    /// ```
    pub fn detach(&self) -> Self {
        let new_node = self
            .graph()
            .borrow_mut()
            .create_identity_node(Rc::clone(&self.node), None, true) // detached=true
            .expect("内部错误：detach 创建 Identity 节点失败");
        Self {
            node: new_node,
            graph: self.graph.clone(),
        }
    }

    /// 检查此 Var 对应的节点是否处于 detached 状态
    ///
    /// detached 节点在反向传播时不会传递梯度给其父节点。
    ///
    /// # 用途
    /// - `ModelState` 使用此方法判断 Var 输入是否可以缓存
    /// - detached Var 只需要值，不需要梯度流，因此可以像 Tensor 一样缓存
    pub fn is_detached(&self) -> bool {
        self.node.is_detached()
    }

    // ==================== 执行 ====================

    /// 前向传播
    ///
    /// 递归执行从当前节点到所有父节点的前向计算
    pub fn forward(&self) -> Result<(), GraphError> {
        self.graph().borrow_mut().forward_via_node_inner(&self.node)
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
    pub fn backward_ex(&self, retain_graph: bool) -> Result<f32, GraphError> {
        let graph_rc = self.graph();
        let mut g = graph_rc.borrow_mut();
        // ensure-forward：先执行前向传播
        g.forward_via_node_inner(&self.node)?;
        // 然后执行反向传播
        g.backward_via_node_inner(&self.node, retain_graph)
    }

    // ==================== 值访问和设置 ====================

    /// 获取节点的值（克隆的 Tensor）
    ///
    /// # 自动 forward 语义
    /// 如果节点尚未计算（value 为 None），会自动触发前向传播。
    /// 这让用户无需手动调用 `forward()` 即可直接获取值。
    pub fn value(&self) -> Result<Option<Tensor>, GraphError> {
        // 如果值还没计算，自动触发 forward
        if self.node.value().is_none() {
            self.forward()?;
        }
        Ok(self.node.value())
    }

    /// 设置节点的值
    pub fn set_value(&self, value: &Tensor) -> Result<(), GraphError> {
        self.node.set_value(Some(value))
    }

    /// 获取标量值（假设是 1x1 Tensor）
    pub fn item(&self) -> Result<f32, GraphError> {
        let val = self
            .value()?
            .ok_or(GraphError::NodeNotFound(self.node_id()))?;
        val.get_data_number()
            .ok_or_else(|| GraphError::InvalidOperation("Tensor 不是标量".to_string()))
    }

    /// 获取节点的梯度
    pub fn grad(&self) -> Result<Option<Tensor>, GraphError> {
        Ok(self.node.grad())
    }

    // ==================== 安全版本（返回 Result）====================

    /// 安全的加法（返回 Result）
    pub fn try_add(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行加法".to_string(),
            ));
        }
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_add_node(vec![Rc::clone(&self.node), Rc::clone(&other.node)], None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
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
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_subtract_node(vec![Rc::clone(&self.node), Rc::clone(&other.node)], None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 安全的元素级乘法（返回 Result）
    pub fn try_mul(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行乘法".to_string(),
            ));
        }
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_multiply_node(vec![Rc::clone(&self.node), Rc::clone(&other.node)], None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
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
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_divide_node(vec![Rc::clone(&self.node), Rc::clone(&other.node)], None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    // ==================== 可视化 ====================

    /// 生成以该 Var 为输出的计算图的 DOT 格式字符串
    ///
    /// 从当前 Var 开始，沿 parents 链遍历整个上游计算图，生成 Graphviz DOT 格式。
    ///
    /// # 示例
    /// ```ignore
    /// let loss = model.forward(&x)?.cross_entropy(&target)?;
    /// println!("{}", loss.to_dot());
    /// ```
    pub fn to_dot(&self) -> String {
        Self::vars_to_dot(&[self])
    }

    /// 保存以该 Var 为输出的计算图可视化
    ///
    /// # 参数
    /// - `base_path`: 输出文件路径（不含后缀，自动生成 .dot 和 .png）
    ///
    /// # 示例
    /// ```ignore
    /// let loss = model.forward(&x)?.cross_entropy(&target)?;
    /// loss.save_visualization("model")?;  // 生成 model.dot 和 model.png
    /// ```
    pub fn save_visualization<P: AsRef<std::path::Path>>(
        &self,
        base_path: P,
    ) -> Result<super::VisualizationOutput, GraphError> {
        Self::save_visualization_for_vars(&[self], base_path)
    }

    /// 合并多个 Var 的计算图并保存可视化（不引入额外节点）
    ///
    /// 适用于多输出/多 loss 场景，将所有分支合并到一张图中。
    ///
    /// # 示例
    /// ```ignore
    /// // 多任务学习：两个 loss 分支合并显示
    /// Var::visualize_all(&[&cls_loss, &reg_loss], "multi_task")?;
    /// ```
    pub fn visualize_all<P: AsRef<std::path::Path>>(
        vars: &[&Self],
        base_path: P,
    ) -> Result<super::VisualizationOutput, GraphError> {
        Self::save_visualization_for_vars(vars, base_path)
    }

    /// 内部方法：从多个 Var 生成 DOT 格式
    fn vars_to_dot(vars: &[&Self]) -> String {
        use std::collections::{HashMap, HashSet, VecDeque};

        if vars.is_empty() {
            return "digraph ComputeGraph {}\n".to_string();
        }

        // 1. 收集所有节点（BFS 遍历 parents）
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut nodes: Vec<Rc<NodeInner>> = Vec::new();
        let mut queue: VecDeque<Rc<NodeInner>> = VecDeque::new();

        for var in vars {
            queue.push_back(Rc::clone(&var.node));
        }

        while let Some(node) = queue.pop_front() {
            let id = node.id();
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id);
            nodes.push(Rc::clone(&node));

            // 将父节点加入队列
            for parent in node.parents() {
                queue.push_back(Rc::clone(parent));
            }
        }

        // 2. 获取层分组信息（从图中）
        let layer_groups = vars[0]
            .graph
            .upgrade()
            .map(|g| g.borrow().layer_groups().to_vec())
            .unwrap_or_default();

        // 收集已分组的节点 ID -> 所属层索引
        let mut node_to_group: HashMap<u64, usize> = HashMap::new();
        for (group_idx, group) in layer_groups.iter().enumerate() {
            for node_id in &group.node_ids {
                // 只记录在当前可视化范围内的节点
                if visited.contains(node_id) {
                    node_to_group.insert(node_id.0, group_idx);
                }
            }
        }

        // === RNN 时间步折叠 ===
        // 读取循环层元信息，计算需要隐藏的节点和输出重定向
        let recurrent_metas = vars[0]
            .graph
            .upgrade()
            .map(|g| g.borrow().recurrent_layer_metas().to_vec())
            .unwrap_or_default();

        // 需要隐藏的节点 ID（步骤 1..N-1 的计算节点）
        let mut rnn_hidden_ids: HashSet<u64> = HashSet::new();
        // 输出重定向：real_output → repr_output（最后一步 → 第一步）
        let mut rnn_output_redirects: HashMap<u64, u64> = HashMap::new();
        // RNN cluster 信息：(name, layer_type, description, visible_node_ids)
        let mut rnn_clusters: Vec<(String, String, String, Vec<u64>)> = Vec::new();
        // RNN 节点标注信息（步数范围：跨所有 forward 调用统计 min~max）
        let mut rnn_step_ranges: HashMap<u64, (usize, usize)> = HashMap::new(); // 计算节点 → (min, max)
        let mut rnn_init_state_ids: HashSet<u64> = HashSet::new(); // 初始状态节点(×1)
        let mut rnn_input_step_ranges: HashMap<u64, (usize, usize)> = HashMap::new(); // input 节点 → 步数范围
        // RNN 特殊边信息
        let mut rnn_feedback_edges: Vec<(u64, u64, usize, usize)> = Vec::new(); // (repr_output, init_state, min, max)
        let mut rnn_init_edges: HashSet<(u64, u64)> = HashSet::new(); // (input, init_state) 橙色虚线
        let mut rnn_output_repr_step_ranges: HashMap<u64, (usize, usize)> = HashMap::new(); // repr_output → (min, max)

        for meta in &recurrent_metas {
            // 查找与当前可视化图匹配的展开信息
            // （visited 中包含该 unroll 的 first_step_start_id）
            let matching_info = meta
                .unroll_infos
                .iter()
                .rev()
                .find(|info| visited.contains(&info.first_step_start_id));
            if let Some(info) = matching_info {
                if info.steps <= 1 {
                    continue; // 单步不需要折叠
                }

                let base = info.first_step_start_id.0;
                let nps = meta.nodes_per_step;

                // 步骤 1..N-1 的节点全部隐藏
                for step in 1..info.steps {
                    for offset in 0..nps {
                        let hidden_id = base + (step * nps) as u64 + offset as u64;
                        rnn_hidden_ids.insert(hidden_id);
                    }
                }

                // 重定向：最后一步的输出 → 第一步的代表输出
                if let Some(&repr_id) = info.repr_output_node_ids.first() {
                    rnn_output_redirects.insert(info.real_output_node_id.0, repr_id.0);
                }

                // 收集可见的 RNN 节点（参数 + 初始状态 + 第一步计算节点）
                let mut visible_ids: Vec<u64> = Vec::new();
                // 参数节点
                for pid in &meta.param_node_ids {
                    if visited.contains(pid) {
                        visible_ids.push(pid.0);
                    }
                }
                // 初始状态节点
                for sid in &info.init_state_node_ids {
                    if visited.contains(sid) {
                        visible_ids.push(sid.0);
                    }
                }
                // 第一步计算节点
                for offset in 0..nps {
                    let step0_id = base + offset as u64;
                    if visited.contains(&NodeId(step0_id)) {
                        visible_ids.push(step0_id);
                    }
                }

                // 计算所有 forward 调用的步数范围（用于标注）
                let min_steps = meta.unroll_infos.iter().map(|i| i.steps).min().unwrap_or(info.steps);
                let max_steps = meta.unroll_infos.iter().map(|i| i.steps).max().unwrap_or(info.steps);
                let step_range_str = if min_steps == max_steps {
                    format!("{}", min_steps)
                } else {
                    format!("{}-{}", min_steps, max_steps)
                };

                let desc = format!(
                    "{}: {} (×{} steps)",
                    meta.layer_type, meta.description, step_range_str
                );
                rnn_clusters.push((meta.name.clone(), meta.layer_type.clone(), desc, visible_ids));

                // 记录 RNN 节点标注和特殊边信息（使用步数范围）
                for offset in 0..nps {
                    let step0_id = base + offset as u64;
                    rnn_step_ranges.insert(step0_id, (min_steps, max_steps));
                }
                for sid in &info.init_state_node_ids {
                    rnn_init_state_ids.insert(sid.0);
                    rnn_init_edges.insert((info.input_node_id.0, sid.0));
                }
                for &repr_id in &info.repr_output_node_ids {
                    rnn_output_repr_step_ranges.insert(repr_id.0, (min_steps, max_steps));
                    for sid in &info.init_state_node_ids {
                        rnn_feedback_edges.push((repr_id.0, sid.0, min_steps, max_steps));
                    }
                }
                rnn_input_step_ranges.insert(info.input_node_id.0, (min_steps, max_steps));

                // 把 RNN cluster 的节点也加入 node_to_group（防止被当作"未分组"节点）
                let rnn_group_idx = layer_groups.len() + rnn_clusters.len() - 1;
                for &nid in rnn_clusters.last().unwrap().3.iter() {
                    node_to_group.insert(nid, rnn_group_idx);
                }
            }
        }

        // 层分组颜色（交替使用不同颜色）
        let group_colors = ["#E3F2FD80", "#E8F5E980", "#FFF3E080", "#F3E5F580"];

        // 节点定义生成闭包
        let generate_node_def = |node: &Rc<NodeInner>| -> String {
            let id = node.id().0;
            let node_type = node.type_name();
            let raw_name = node
                .name()
                .map(|s| s.to_string())
                .unwrap_or_else(|| node_type.to_lowercase());
            // 可视化时去掉模型前缀（"Generator/fc1_W" → "fc1_W"）
            let name = match raw_name.split_once('/') {
                Some((_, after)) => after.to_string(),
                None => raw_name,
            };
            let shape = node.value_expected_shape();

            // 形状字符串：Parameter 显示固定形状，其他节点第一维显示为 ? (batch 维度)
            // 对 RNN 输入节点，序列维度（dim 1）显示步数范围
            let shape_str = if node_type == "Parameter" || shape.is_empty() {
                format!("{:?}", shape)
            } else {
                let input_range = rnn_input_step_ranges.get(&id);
                let dims: Vec<String> = shape
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| {
                        if i == 0 {
                            "?".to_string()
                        } else if i == 1 {
                            if let Some(&(min_s, max_s)) = input_range {
                                if min_s != max_s {
                                    return format!("{}-{}", min_s, max_s);
                                }
                            }
                            d.to_string()
                        } else {
                            d.to_string()
                        }
                    })
                    .collect();
                format!("[{}]", dims.join(", "))
            };

            // Parameter 节点显示参数数量
            let param_count_str = if node_type == "Parameter" && !shape.is_empty() {
                let count: usize = shape.iter().product();
                // 千分位格式化
                let s = count.to_string();
                let bytes = s.as_bytes();
                let mut formatted = String::new();
                for (i, &b) in bytes.iter().enumerate() {
                    if i > 0 && (bytes.len() - i) % 3 == 0 {
                        formatted.push(',');
                    }
                    formatted.push(b as char);
                }
                format!("<BR/>({formatted} params)")
            } else {
                String::new()
            };

            // 超参数信息（Dropout p, LeakyReLU alpha 等）
            let hyperparam_str = node.with_raw_node(|raw| {
                use super::nodes::raw_node::NodeType;
                match raw {
                    NodeType::Dropout(d) => format!("<BR/>(p={:.1})", d.p()),
                    NodeType::LeakyReLU(lr) => {
                        let a = lr.alpha();
                        // alpha=0 就是标准 ReLU，不显示
                        if a != 0.0 {
                            format!("<BR/>(α={a})")
                        } else {
                            String::new()
                        }
                    }
                    _ => String::new(),
                }
            });

            // 根据节点类型选择样式
            let (node_shape, style, fill_color) = match node_type.as_str() {
                "Input" | "BasicInput" => ("ellipse", "filled", "#E3F2FD"),
                "SmartInput" | "RecurrentOutput" => ("ellipse", "filled", "#E0E0E0"),
                "TargetInput" => ("ellipse", "filled", "#FFE0B2"),
                "State" => ("cylinder", "filled", "#FFE0B2"),
                "Identity" => ("ellipse", "\"filled,dashed\"", "#E1BEE7"),
                "Parameter" => ("box", "filled", "#E8F5E9"),
                "ZerosLike" => ("box", "\"filled,rounded,dashed\"", "#FFFDE7"),
                t if t.contains("Loss")
                    || t.contains("BCE")
                    || t.contains("MSE")
                    || t.contains("MAE")
                    || t.contains("Huber")
                    || t.contains("CrossEntropy") =>
                {
                    ("octagon", "filled", "#FFEBEE")
                }
                "Sigmoid" | "Tanh" | "ReLU" | "LeakyReLU" | "Sign" | "SoftPlus" | "Step"
                | "Softmax" | "LogSoftmax" | "Abs" | "Ln" => ("diamond", "filled", "#FFF3E0"),
                // 正则化节点：菱形 + 浅紫色，区别于激活函数（训练/推理行为不同）
                "Dropout" | "BatchNorm" => ("diamond", "filled", "#E1BEE7"),
                _ => ("box", "\"filled,rounded\"", "#FFFDE7"),
            };

            // RNN 步数标注：计算节点显示 ×N（或 ×min-max），初始状态节点显示 ×1
            let rnn_step_str = if let Some(&(min_s, max_s)) = rnn_step_ranges.get(&id) {
                if min_s == max_s {
                    format!(" <FONT COLOR=\"#E67E22\">×{}</FONT>", min_s)
                } else {
                    format!(" <FONT COLOR=\"#E67E22\">×{}-{}</FONT>", min_s, max_s)
                }
            } else if rnn_init_state_ids.contains(&id) {
                " <FONT COLOR=\"#E67E22\">×1</FONT>".to_string()
            } else {
                String::new()
            };
            // 双层边框：仅用于实际存在多个副本的节点（橙色 ×N, N>1）
            let peripheries_str = if let Some(&(min_s, _)) = rnn_step_ranges.get(&id) {
                if min_s > 1 { " peripheries=2" } else { "" }
            } else {
                ""
            };

            format!(
                "\"{}\" [label=<{}{}<BR/><B>{}</B><BR/>{}{}{}> shape={} style={} fillcolor=\"{}\" fontsize=10{}];\n",
                id, name, rnn_step_str, node_type, shape_str, param_count_str, hyperparam_str, node_shape, style, fill_color, peripheries_str
            )
        };

        // 3. 生成 DOT 格式
        let mut dot = String::new();
        dot.push_str("digraph ComputeGraph {\n");
        dot.push_str("    rankdir=TB;\n");
        dot.push_str("    newrank=true;\n");
        dot.push_str("    splines=polyline;\n");
        dot.push_str("    node [fontname=\"Microsoft YaHei,SimHei,Arial\"];\n");
        dot.push_str("    edge [fontname=\"Microsoft YaHei,SimHei,Arial\"];\n\n");

        // 4. 输出层分组（cluster），支持嵌套（Model / Layer）
        //
        // 层名含 "/" 时（如 "Generator/fc1"）：
        //   外层 cluster = "Generator"（模型），内层 cluster = "fc1"（层）
        // 层名不含 "/" 时（如 "fc1"）：
        //   单层 cluster（现有行为）

        // 将分组按模型前缀归类
        use super::graph::LayerGroup;
        let mut model_groups: std::collections::BTreeMap<
            String,
            Vec<(usize, String, &LayerGroup)>,
        > = std::collections::BTreeMap::new();
        let mut standalone_groups: Vec<(usize, &LayerGroup)> = Vec::new();

        for (group_idx, group) in layer_groups.iter().enumerate() {
            if let Some((model, layer)) = group.name.split_once('/') {
                model_groups
                    .entry(model.to_string())
                    .or_default()
                    .push((group_idx, layer.to_string(), group));
            } else {
                standalone_groups.push((group_idx, group));
            }
        }

        // === 拓扑推断：将未分组节点归入正确的模型 ===
        // 初始映射：从 layer group 名称中提取模型归属
        let mut node_to_model: HashMap<u64, String> = HashMap::new();
        for group in &layer_groups {
            if let Some((model, _)) = group.name.split_once('/') {
                for node_id in &group.node_ids {
                    if visited.contains(node_id) {
                        node_to_model.insert(node_id.0, model.to_string());
                    }
                }
            }
        }
        // RNN 节点加入 node_to_model（RNN 不调用 register_layer_group，
        // 需要从 RecurrentLayerMeta + rnn_clusters 中提取模型归属）
        // 包含：参数节点 + 可见的计算/状态节点，确保拓扑推断能追溯到 input
        for (name, _, _, visible_ids) in &rnn_clusters {
            if let Some((model, _)) = name.split_once('/') {
                for &nid in visible_ids {
                    node_to_model.insert(nid, model.to_string());
                }
            }
        }

        // 仅在存在模型分组时执行推断
        if !model_groups.is_empty() {
            // 建立 node_id -> children 映射（用于叶子节点上推）
            let mut node_children: HashMap<u64, Vec<u64>> = HashMap::new();
            for node in &nodes {
                for parent in node.parents() {
                    node_children
                        .entry(parent.id().0)
                        .or_default()
                        .push(node.id().0);
                }
            }

            // 迭代推断，直到稳定
            loop {
                let mut changed = false;

                for node in &nodes {
                    let nid = node.id().0;
                    if node_to_model.contains_key(&nid) {
                        continue;
                    }

                    let parents = node.parents();

                    if !parents.is_empty() {
                        // 向下传播：所有父节点都在同一模型 → 归入该模型
                        let parent_models: Vec<Option<&str>> = parents
                            .iter()
                            .map(|p| node_to_model.get(&p.id().0).map(|s| s.as_str()))
                            .collect();

                        if parent_models.iter().all(|m| m.is_some()) {
                            let unique: HashSet<&str> =
                                parent_models.iter().filter_map(|m| *m).collect();
                            if unique.len() == 1 {
                                let model = unique.into_iter().next().unwrap().to_string();
                                node_to_model.insert(nid, model);
                                changed = true;
                            }
                        }
                    } else {
                        // 叶子节点上推：所有（非隐藏的）子节点都在同一模型 → 归入该模型
                        if let Some(children) = node_children.get(&nid) {
                            let child_models: Vec<Option<&str>> = children
                                .iter()
                                .filter(|c| !rnn_hidden_ids.contains(c)) // 排除 RNN 隐藏节点
                                .map(|c| node_to_model.get(c).map(|s| s.as_str()))
                                .collect();

                            if !child_models.is_empty()
                                && child_models.iter().all(|m| m.is_some())
                            {
                                let unique: HashSet<&str> =
                                    child_models.iter().filter_map(|m| *m).collect();
                                if unique.len() == 1 {
                                    let model = unique.into_iter().next().unwrap().to_string();
                                    node_to_model.insert(nid, model);
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                if !changed {
                    break;
                }
            }
        }

        // === 预计算跨模型边和虚拟 Input 节点 ===
        // key: (parent_id, child_model) → virtual_node_id
        let mut virtual_input_map: HashMap<(u64, String), String> = HashMap::new();
        // model_name → [(virt_id, shape_str)]
        let mut virtual_inputs_by_model: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for node in &nodes {
            let child_id = node.id().0;
            let child_model = node_to_model.get(&child_id);
            for parent in node.parents() {
                let parent_id = parent.id().0;
                let parent_model = node_to_model.get(&parent_id);

                if let (Some(pm), Some(cm)) = (parent_model, child_model) {
                    if pm != cm {
                        let key = (parent_id, cm.clone());
                        if !virtual_input_map.contains_key(&key) {
                            let virt_id = format!("virt_{}_{}", parent_id, cm.replace(' ', "_"));
                            // 形状信息
                            let shape = parent.value_expected_shape();
                            let shape_str = if shape.is_empty() {
                                String::new()
                            } else {
                                let dims: Vec<String> = shape
                                    .iter()
                                    .enumerate()
                                    .map(|(i, &d)| {
                                        if i == 0 {
                                            "?".to_string()
                                        } else {
                                            d.to_string()
                                        }
                                    })
                                    .collect();
                                format!("[{}]", dims.join(", "))
                            };
                            virtual_inputs_by_model
                                .entry(cm.clone())
                                .or_default()
                                .push((virt_id.clone(), shape_str));
                            virtual_input_map.insert(key, virt_id);
                        }
                    }
                }
            }
        }

        // 模型级 cluster 颜色（柔和半透明）
        let model_colors = ["#FFEBEE40", "#E8EAF640", "#E0F2F140", "#FFF8E140"];

        // 辅助闭包：渲染单个层 cluster 的内容
        let render_layer_cluster =
            |dot: &mut String,
             indent: &str,
             cluster_id: &str,
             display_name: &str,
             group: &LayerGroup,
             color: &str| {
                let group_node_ids: Vec<u64> = group
                    .node_ids
                    .iter()
                    .filter(|id| visited.contains(id))
                    .map(|id| id.0)
                    .collect();

                if group_node_ids.is_empty() {
                    return;
                }

                dot.push_str(&format!(
                    "{indent}subgraph cluster_{cluster_id} {{\n"
                ));
                dot.push_str(&format!(
                    "{indent}    label=<<B>{display_name}</B><BR/><FONT POINT-SIZE=\"9\">{}: {}</FONT>>;\n",
                    group.layer_type, group.description
                ));
                dot.push_str(&format!("{indent}    style=filled;\n"));
                dot.push_str(&format!("{indent}    fillcolor=\"{color}\";\n"));
                dot.push_str(&format!(
                    "{indent}    fontname=\"Microsoft YaHei,SimHei,Arial\";\n"
                ));
                dot.push_str(&format!("{indent}    fontsize=11;\n"));
                dot.push_str(&format!("{indent}    margin=12;\n"));

                for node in &nodes {
                    if group_node_ids.contains(&node.id().0) {
                        dot.push_str(&format!("{indent}        "));
                        dot.push_str(&generate_node_def(node));
                    }
                }

                dot.push_str(&format!("{indent}}}\n\n"));
            };

        // 建立 RNN cluster 的模型归属索引
        let mut rnn_clusters_for_model: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, (name, _, _, _)) in rnn_clusters.iter().enumerate() {
            if let Some((model, _)) = name.split_once('/') {
                rnn_clusters_for_model
                    .entry(model.to_string())
                    .or_default()
                    .push(i);
            }
        }
        let mut rendered_rnn_clusters: HashSet<usize> = HashSet::new();

        // 渲染模型级嵌套 cluster
        for (model_idx, (model_name, layers)) in model_groups.iter().enumerate() {
            let model_id = model_name.replace(['-', '.', ' ', '/'], "_");
            let model_color = model_colors[model_idx % model_colors.len()];

            // 统计该模型的总参数量
            let model_param_count: usize = nodes
                .iter()
                .filter(|n| {
                    node_to_model.get(&n.id().0).map(|m| m.as_str()) == Some(model_name)
                        && n.type_name() == "Parameter"
                })
                .map(|n| n.value_expected_shape().iter().product::<usize>())
                .sum();

            dot.push_str(&format!("    subgraph cluster_model_{model_id} {{\n"));
            // 千分位格式化参数量
            let param_str = {
                let s = model_param_count.to_string();
                let bytes = s.as_bytes();
                let mut result = String::new();
                for (i, &b) in bytes.iter().enumerate() {
                    if i > 0 && (bytes.len() - i) % 3 == 0 {
                        result.push(',');
                    }
                    result.push(b as char);
                }
                result
            };
            dot.push_str(&format!(
                "        label=<<B>{model_name}</B><BR/><FONT POINT-SIZE=\"9\">({param_str} params)</FONT>>;\n"
            ));
            dot.push_str("        style=\"filled,bold\";\n");
            dot.push_str(&format!("        fillcolor=\"{model_color}\";\n"));
            dot.push_str("        fontname=\"Microsoft YaHei,SimHei,Arial\";\n");
            dot.push_str("        fontsize=13;\n");
            dot.push_str("        margin=16;\n\n");

            // 内层 layer cluster
            for (group_idx, layer_name, group) in layers {
                let cluster_id =
                    format!("{model_id}_{}", layer_name.replace(['-', '.', ' ', '/'], "_"));
                let color = group_colors[*group_idx % group_colors.len()];
                render_layer_cluster(
                    &mut dot,
                    "        ",
                    &cluster_id,
                    layer_name,
                    group,
                    color,
                );
            }

            // RNN 折叠 cluster（属于该模型的，嵌套在模型内部）
            if let Some(rnn_indices) = rnn_clusters_for_model.get(model_name.as_str()) {
                for &idx in rnn_indices {
                    let (rnn_name, _, desc, visible_ids) = &rnn_clusters[idx];
                    let layer_name =
                        rnn_name.split_once('/').map(|(_, l)| l).unwrap_or(rnn_name);
                    let cluster_id = format!(
                        "{model_id}_{}",
                        layer_name.replace(['-', '.', ' ', '/'], "_")
                    );
                    let color =
                        group_colors[(layer_groups.len() + idx) % group_colors.len()];

                    dot.push_str(&format!(
                        "        subgraph cluster_{cluster_id} {{\n"
                    ));
                    dot.push_str(&format!(
                        "            label=<<B>{layer_name}</B><BR/><FONT POINT-SIZE=\"9\">{desc}</FONT>>;\n"
                    ));
                    dot.push_str("            style=\"filled,bold\";\n");
                    dot.push_str("            peripheries=3;\n");
                    dot.push_str("            penwidth=2;\n");
                    dot.push_str(&format!("            fillcolor=\"{color}\";\n"));
                    dot.push_str(
                        "            fontname=\"Microsoft YaHei,SimHei,Arial\";\n",
                    );
                    dot.push_str("            fontsize=11;\n");
                    dot.push_str("            margin=12;\n");

                    for node in &nodes {
                        if visible_ids.contains(&node.id().0) {
                            dot.push_str("            ");
                            dot.push_str(&generate_node_def(node));
                        }
                    }

                    dot.push_str("        }\n\n");
                    rendered_rnn_clusters.insert(idx);
                }
            }

            // 推断归属的节点（不在任何 layer group 中，但拓扑推断属于此模型）
            for node in &nodes {
                let nid = node.id().0;
                if !node_to_group.contains_key(&nid) && !rnn_hidden_ids.contains(&nid) {
                    if let Some(m) = node_to_model.get(&nid) {
                        if m == model_name {
                            dot.push_str("        ");
                            dot.push_str(&generate_node_def(node));
                        }
                    }
                }
            }

            // 跨模型虚拟 Input 节点（纯可视化，虚线椭圆）
            if let Some(virt_inputs) = virtual_inputs_by_model.get(model_name.as_str()) {
                for (virt_id, shape_str) in virt_inputs {
                    dot.push_str(&format!(
                        "        \"{}\" [label=<Input<BR/>{}> shape=ellipse style=\"filled,dashed\" fillcolor=\"#E0E0E0\" fontsize=10];\n",
                        virt_id, shape_str
                    ));
                }
            }

            dot.push_str("    }\n\n");
        }

        // 渲染独立分组（无模型前缀，保持现有行为）
        for (group_idx, group) in &standalone_groups {
            let cluster_id = group.name.replace(['-', '.', ' ', '/'], "_");
            let color = group_colors[*group_idx % group_colors.len()];
            render_layer_cluster(&mut dot, "    ", &cluster_id, &group.name, group, color);
        }

        // 5. 输出 RNN 折叠 cluster（加粗边框，三层轮廓；跳过已嵌套在模型内的）
        for (i, (name, _layer_type, desc, visible_ids)) in rnn_clusters.iter().enumerate() {
            if rendered_rnn_clusters.contains(&i) {
                continue; // 已在模型 cluster 内渲染
            }
            // 独立渲染时，显示完整名称（去掉可能的模型前缀）
            let display_name = name.split_once('/').map(|(_, l)| l).unwrap_or(name);
            let cluster_id = format!("rnn_{}", name.replace(['-', '.', ' ', '/'], "_"));
            let color = group_colors[(layer_groups.len() + i) % group_colors.len()];

            dot.push_str(&format!("    subgraph cluster_{cluster_id} {{\n"));
            dot.push_str(&format!(
                "        label=<<B>{display_name}</B><BR/><FONT POINT-SIZE=\"9\">{desc}</FONT>>;\n"
            ));
            dot.push_str("        style=\"filled,bold\";\n");
            dot.push_str("        peripheries=3;\n");
            dot.push_str("        penwidth=2;\n");
            dot.push_str(&format!("        fillcolor=\"{color}\";\n"));
            dot.push_str("        fontname=\"Microsoft YaHei,SimHei,Arial\";\n");
            dot.push_str("        fontsize=11;\n");
            dot.push_str("        margin=12;\n");

            for node in &nodes {
                if visible_ids.contains(&node.id().0) {
                    dot.push_str("        ");
                    dot.push_str(&generate_node_def(node));
                }
            }

            dot.push_str("    }\n\n");
        }

        // 6. 输出未分组的节点（排除已推断归入模型的节点 + RNN 隐藏节点）
        for node in &nodes {
            let nid = node.id().0;
            if !node_to_group.contains_key(&nid)
                && !node_to_model.contains_key(&nid)
                && !rnn_hidden_ids.contains(&nid)
            {
                dot.push_str("    ");
                dot.push_str(&generate_node_def(node));
            }
        }

        dot.push('\n');

        // 7. 生成边（父节点 -> 子节点），跨模型边通过虚拟 Input 中转
        for node in &nodes {
            let child_id = node.id().0;

            // 跳过 RNN 隐藏节点的边
            if rnn_hidden_ids.contains(&child_id) {
                continue;
            }

            let child_model = node_to_model.get(&child_id);

            for parent in node.parents() {
                let original_parent_id = parent.id().0;

                // RNN 输出重定向：如果父节点是被折叠的 real_output，
                // 重定向到 repr_output（第一步的输出）
                // 注意：必须在隐藏检查之前执行，否则 real_output 会被当作隐藏节点跳过
                let parent_id =
                    *rnn_output_redirects
                        .get(&original_parent_id)
                        .unwrap_or(&original_parent_id);
                let was_redirected = original_parent_id != parent_id;

                // 跳过隐藏父节点的边（重定向后再检查）
                if rnn_hidden_ids.contains(&parent_id) {
                    continue;
                }

                // RNN 特殊边样式
                let edge_attrs = if rnn_init_edges.contains(&(parent_id, child_id)) {
                    // A2: input → init_state 橙色虚线，标注 t=0
                    " [style=dashed color=\"#E67E22\" label=<t=0> fontcolor=\"#E67E22\" fontsize=9]".to_string()
                } else if was_redirected {
                    // A3: repr_output → 下游层，橙色实线标注最后一步
                    if let Some(&(min_s, max_s)) = rnn_output_repr_step_ranges.get(&parent_id) {
                        let t_label = if min_s == max_s {
                            format!("t={}", min_s - 1)
                        } else {
                            format!("t={}~{}", min_s - 1, max_s - 1)
                        };
                        format!(
                            " [color=\"#E67E22\" label=<{}> fontcolor=\"#E67E22\" fontsize=9]",
                            t_label
                        )
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                let parent_model = node_to_model.get(&parent_id);

                // 检查是否为跨模型边
                let is_cross_model = match (parent_model, child_model) {
                    (Some(pm), Some(cm)) => pm != cm,
                    _ => false,
                };

                if is_cross_model {
                    // 通过虚拟 Input 节点中转
                    let key = (parent_id, child_model.unwrap().clone());
                    if let Some(virt_id) = virtual_input_map.get(&key) {
                        dot.push_str(&format!(
                            "    \"{}\" -> \"{}\" [style=dashed color=\"#999999\"];\n",
                            parent_id, virt_id
                        ));
                        dot.push_str(&format!(
                            "    \"{}\" -> \"{}\"{};\n",
                            virt_id, child_id, edge_attrs
                        ));
                    }
                } else {
                    dot.push_str(&format!(
                        "    \"{}\" -> \"{}\"{};\n",
                        parent_id, child_id, edge_attrs
                    ));
                }
            }
        }

        // A1: RNN 回流反馈边（概念性虚线，表示隐藏状态循环）
        for (repr_id, init_state_id, min_steps, max_steps) in &rnn_feedback_edges {
            let label = if min_steps == max_steps {
                if *min_steps <= 2 {
                    "t=0".to_string()
                } else {
                    format!("t=0~{}", min_steps - 2)
                }
            } else {
                // 变长：显示范围 t=0~(min_last~max_last)
                format!("t=0~({}~{})", min_steps - 2, max_steps - 2)
            };
            dot.push_str(&format!(
                "    \"{}\" -> \"{}\" [style=dashed color=\"#E67E22\" label=<{}> fontcolor=\"#E67E22\" fontsize=9 constraint=false];\n",
                repr_id, init_state_id, label
            ));
        }

        dot.push_str("}\n");
        dot
    }

    /// 内部方法：保存多个 Var 的可视化
    fn save_visualization_for_vars<P: AsRef<std::path::Path>>(
        vars: &[&Self],
        base_path: P,
    ) -> Result<super::VisualizationOutput, GraphError> {
        use std::fs::File;
        use std::io::Write;
        use std::process::Command;

        let base = base_path.as_ref();

        // 检查路径不应包含后缀
        if let Some(ext) = base.extension() {
            return Err(GraphError::InvalidOperation(format!(
                "base_path 不应包含文件后缀（如 .{}），请使用不带后缀的路径",
                ext.to_string_lossy()
            )));
        }

        let dot_path = base.with_extension("dot");
        let png_path = base.with_extension("png");

        // 生成 DOT 内容
        let dot_content = Self::vars_to_dot(vars);

        // 保存 .dot 文件
        {
            let mut file = File::create(&dot_path)
                .map_err(|e| GraphError::ComputationError(format!("无法创建 DOT 文件: {}", e)))?;
            file.write_all(dot_content.as_bytes())
                .map_err(|e| GraphError::ComputationError(format!("写入 DOT 文件失败: {}", e)))?;
            // 确保文件内容刷新到磁盘
            file.sync_all()
                .map_err(|e| GraphError::ComputationError(format!("同步 DOT 文件失败: {}", e)))?;
            // file 在此作用域结束时会被自动关闭
        }

        // 尝试用 Graphviz 生成 PNG
        let graphviz_available = Command::new("dot")
            .arg("-V")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        if graphviz_available {
            let output = Command::new("dot")
                .arg("-Tpng")
                .arg(&dot_path)
                .arg("-o")
                .arg(&png_path)
                .output();

            match output {
                Ok(o) if o.status.success() => Ok(super::VisualizationOutput {
                    dot_path,
                    image_path: Some(png_path),
                    graphviz_available: true,
                    graphviz_hint: None,
                }),
                Ok(o) => {
                    // 执行失败，提取错误信息
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    let hint = if stderr.is_empty() {
                        format!("Graphviz 执行失败 (exit: {:?})", o.status.code())
                    } else {
                        format!(
                            "Graphviz 执行失败 (exit: {:?}): {}",
                            o.status.code(),
                            stderr.trim()
                        )
                    };
                    Ok(super::VisualizationOutput {
                        dot_path,
                        image_path: None,
                        graphviz_available: true,
                        graphviz_hint: Some(hint),
                    })
                }
                Err(e) => Ok(super::VisualizationOutput {
                    dot_path,
                    image_path: None,
                    graphviz_available: true,
                    graphviz_hint: Some(format!("无法执行 Graphviz: {}", e)),
                }),
            }
        } else {
            Ok(super::VisualizationOutput {
                dot_path,
                image_path: None,
                graphviz_available: false,
                graphviz_hint: Some("请安装 Graphviz: https://graphviz.org/download/".to_string()),
            })
        }
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
    /// 用于 Var-Tensor 混合运算（如加减乘除）。
    pub(crate) fn tensor_to_var(&self, tensor: &Tensor) -> Self {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_basic_input_node(tensor.shape(), None)
            .expect("创建 Tensor->Var 转换节点失败");
        node.set_value(Some(tensor)).expect("设置 Tensor 值失败");
        Self::new_with_rc_graph(node, &graph)
    }

    /// 将 Tensor 转换为 TargetInput 类型的 Var（用于损失函数的目标值）
    ///
    /// 与 `tensor_to_var` 的区别：创建 TargetInput 节点而非普通 Input，
    /// 在可视化中显示为橙色椭圆，便于区分模型输入和损失目标。
    pub(crate) fn tensor_to_target_var(&self, tensor: &Tensor) -> Self {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_target_input_node(tensor.shape(), None)
            .expect("创建 TargetInput 节点失败");
        node.set_value(Some(tensor)).expect("设置 Tensor 值失败");
        Self::new_with_rc_graph(node, &graph)
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
        let graph = self.graph();
        let mut g = graph.borrow_mut();
        // 创建 -1 常量
        let neg_one_node = g
            .create_basic_input_node(&[1, 1], None)
            .expect("创建 -1 节点失败");
        neg_one_node
            .set_value(Some(&Tensor::new(&[-1.0], &[1, 1])))
            .expect("设置 -1 值失败");
        // -self = -1 * self（Multiply 支持广播）
        let node = g
            .create_multiply_node(vec![neg_one_node, Rc::clone(&self.node)], None)
            .expect("创建取反节点失败");
        drop(g); // 释放借用
        Var::new_with_rc_graph(node, &graph)
    }
}

// Neg for Var
impl Neg for Var {
    type Output = Self;

    fn neg(self) -> Self {
        -&self
    }
}

// ==================== IntoVar ====================

/// 前向传播输入类型转换 trait
///
/// 允许模型的 forward 方法同时接受 `&Tensor` 和 `&Var`，
/// 与 PyTorch 中统一使用 Tensor 的体验类似。
///
/// # 示例
/// ```ignore
/// // 模型定义
/// impl MyModel {
///     pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
///         let input = x.into_var(&self.graph)?;
///         let h = self.fc1.forward(&input).relu();
///         Ok(self.fc2.forward(&h))
///     }
/// }
///
/// // 使用时：Tensor 和 Var 都可以
/// let out1 = model.forward(&tensor)?;  // &Tensor
/// let out2 = model.forward(&var)?;     // &Var
/// let out3 = model.forward(var)?;      // Var
/// ```
pub trait IntoVar {
    fn into_var(self, graph: &Graph) -> Result<Var, GraphError>;
}

impl IntoVar for &Tensor {
    fn into_var(self, graph: &Graph) -> Result<Var, GraphError> {
        graph.input(self)
    }
}

impl IntoVar for Tensor {
    fn into_var(self, graph: &Graph) -> Result<Var, GraphError> {
        graph.input(&self)
    }
}

impl IntoVar for &Var {
    fn into_var(self, _graph: &Graph) -> Result<Var, GraphError> {
        Ok(self.clone())
    }
}

impl IntoVar for Var {
    fn into_var(self, _graph: &Graph) -> Result<Var, GraphError> {
        Ok(self)
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
    fn test_detach_creates_node() {
        // 验证 detach() 创建 Identity 节点（与 PyTorch 行为一致）
        use crate::nn::Graph;
        let graph = Graph::new();
        let x = graph.input(&crate::tensor::Tensor::ones(&[1, 2])).unwrap();
        let initial_count = graph.inner().nodes_count();

        // detach() 创建新的 Identity 节点
        let x_detached = x.detach();
        let after_count = graph.inner().nodes_count();
        assert_eq!(initial_count + 1, after_count, "detach() 应创建一个新节点");
        assert!(x_detached.is_detached(), "detach 返回的 Var 应标记为 detached");
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
