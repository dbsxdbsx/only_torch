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
/// # 设计原则
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
    /// 创建新的 Var
    ///
    /// 直接持有 NodeInner，graph 为 Weak 引用
    #[allow(dead_code)]
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

    /// 获取节点名称
    pub fn name(&self) -> Option<&str> {
        self.node.name()
    }

    /// 获取节点分组标签（用于可视化 cluster）
    pub fn node_group_tag(&self) -> Option<super::graph::NodeGroupTag> {
        self.node.node_group_tag()
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
    #[allow(dead_code)]
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

    /// 创建一个 detached 的 Var（与 PyTorch `tensor.detach()` 语义一致）
    ///
    /// 在图中创建一个新的 Detach 节点。
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
            .create_detach_node(Rc::clone(&self.node), None)
            .expect("内部错误：detach 创建 Detach 节点失败");
        Self {
            node: new_node,
            graph: self.graph.clone(),
        }
    }

    /// 检查此 Var 对应的节点是否处于 detached 状态
    ///
    /// detached 节点在反向传播时不会传递梯度给其父节点。
    /// 判断依据：节点类型为 `Detach`，或底层标志位 `is_detached` 为 true。
    ///
    /// # 用途
    /// - `ModelState` 使用此方法判断 Var 输入是否可以缓存
    /// - detached Var 只需要值，不需要梯度流，因此可以像 Tensor 一样缓存
    pub fn is_detached(&self) -> bool {
        use crate::nn::nodes::NodeType;
        self.node
            .with_raw_node(|raw| matches!(raw, NodeType::Detach(_)))
            || self.node.is_detached()
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
    /// # 语义
    /// - 自动先执行 forward()，确保 loss 值已计算
    /// - 然后执行反向传播
    /// - 动态图架构下，中间结果由 Rc 引用计数管理，天然支持多次 backward
    ///
    /// # 多任务学习示例
    /// ```ignore
    /// optimizer.zero_grad()?;
    /// let v1 = loss1.backward()?;  // 第一个 loss
    /// let v2 = loss2.backward()?;  // 第二个 loss，梯度自动累积到共享参数
    /// optimizer.step()?;
    /// ```
    ///
    /// # 返回值
    /// 返回 loss 的标量值
    pub fn backward(&self) -> Result<f32, GraphError> {
        let graph_rc = self.graph();
        let mut g = graph_rc.borrow_mut();
        // ensure-forward：先执行前向传播
        g.forward_via_node_inner(&self.node)?;
        // 然后执行反向传播
        g.backward_via_node_inner(&self.node)
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

    /// 逐元素取最小值：`min(self, other)`
    ///
    /// 用于 TD3/SAC 的 Twin Q 网络：`min(Q1, Q2)` 减少 Q 值过估计。
    ///
    /// # 示例
    /// ```ignore
    /// let q_min = q1_var.minimum(&q2_var)?;
    /// ```
    pub fn minimum(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行 minimum".to_string(),
            ));
        }
        let graph = self.graph();
        let node = graph.borrow_mut().create_minimum_node(
            Rc::clone(&self.node),
            Rc::clone(&other.node),
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 逐元素取最大值：`max(self, other)`
    ///
    /// # 示例
    /// ```ignore
    /// let q_max = q1_var.maximum(&q2_var)?;
    /// ```
    pub fn maximum(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行 maximum".to_string(),
            ));
        }
        let graph = self.graph();
        let node = graph.borrow_mut().create_maximum_node(
            Rc::clone(&self.node),
            Rc::clone(&other.node),
            None,
        )?;
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

    /// 从多个命名 Var 构建可视化快照（BFS 遍历 + 提取轻量级节点信息）
    ///
    /// 快照与 `Rc<NodeInner>` 完全解耦，创建后不依赖节点生命周期。
    ///
    /// **节点排序策略**：按路径逐个 BFS，第一个 loss 路径的全部节点排在前面，
    /// 第二个 loss 路径的新增节点排在后面。这样同一模型的节点编号连续、直观。
    pub(crate) fn build_snapshot(named_vars: &[(&str, &Self)]) -> super::VisualizationSnapshot {
        use std::collections::{HashSet, VecDeque};

        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut snapshot_nodes: Vec<super::SnapshotNode> = Vec::new();

        // 提取单个节点信息的闭包
        let extract_node = |node: &Rc<NodeInner>| -> super::SnapshotNode {
            let hyperparam_html = node.with_raw_node(|raw| {
                use super::nodes::raw_node::NodeType;
                match raw {
                    NodeType::Dropout(d) => Some(format!("<BR/>(p={:.1})", d.p())),
                    NodeType::LeakyReLU(lr) => {
                        let a = lr.alpha();
                        if a != 0.0 {
                            Some(format!("<BR/>(α={a})"))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            });

            // 提取 Input/TargetInput 节点的数据源 ID（用于同源数据追踪）
            let data_source_id = node.with_raw_node(|raw| {
                use super::nodes::raw_node::NodeType;
                match raw {
                    NodeType::Input(variant) => variant.value_source_id(),
                    _ => None,
                }
            });

            super::SnapshotNode {
                id: node.id(),
                name: node.name().map(|s| s.to_string()),
                type_name: node.type_name(),
                shape: node.value_expected_shape(),
                parent_ids: node.parents().iter().map(|p| p.id()).collect(),
                is_detached: node.is_detached() || node.type_name() == "Detach",
                hyperparam_html,
                node_group_tag: node.node_group_tag(),
                data_source_id,
            }
        };

        // 按路径逐个 BFS：先收集第一个 loss 的全部节点，再收集第二个 loss 的新增节点...
        // 这样同一模型的节点在 snapshot_nodes 中连续排列，编号更直观
        for (_, var) in named_vars {
            let mut queue: VecDeque<Rc<NodeInner>> = VecDeque::new();
            queue.push_back(Rc::clone(&var.node));

            while let Some(node) = queue.pop_front() {
                let id = node.id();
                if visited.contains(&id) {
                    continue;
                }
                visited.insert(id);
                snapshot_nodes.push(extract_node(&node));

                for parent in node.parents() {
                    queue.push_back(Rc::clone(parent));
                }
            }
        }

        let named_outputs = named_vars
            .iter()
            .map(|(name, var)| (name.to_string(), var.node.id()))
            .collect();

        super::VisualizationSnapshot {
            nodes: snapshot_nodes,
            named_outputs,
        }
    }

    /// 从可视化快照 + 图元数据生成 DOT 格式字符串（含多 Loss 路径边着色）
    ///
    /// 这是新的 DOT 生成入口，从 `VisualizationSnapshot` 驱动，
    /// 不依赖 `Rc<NodeInner>` 生命周期。
    pub(crate) fn snapshot_to_dot(
        snapshot: &super::VisualizationSnapshot,
        folding_metas: &[super::graph::RecurrentFoldingMeta],
    ) -> String {
        use std::collections::{HashMap, HashSet};

        if snapshot.nodes.is_empty() {
            return "digraph ComputeGraph {}\n".to_string();
        }

        // 快照节点 ID 查找表
        let node_map: HashMap<u64, &super::SnapshotNode> =
            snapshot.nodes.iter().map(|n| (n.id.0, n)).collect();
        let visited: HashSet<NodeId> = snapshot.nodes.iter().map(|n| n.id).collect();

        // 输出节点 ID 集合
        let output_node_ids: HashSet<u64> =
            snapshot.named_outputs.iter().map(|(_, id)| id.0).collect();

        // === 多 Loss 路径着色：per-root BFS ===
        // 路径颜色表
        const PATH_COLORS: &[&str] = &[
            "#1976D2", // 蓝
            "#D32F2F", // 红
            "#388E3C", // 绿
            "#F57C00", // 橙
            "#7B1FA2", // 紫
            "#00796B", // 青
        ];
        let multi_path = snapshot.named_outputs.len() > 1;

        // node_id → 所属路径索引集合
        let mut node_to_paths: HashMap<u64, HashSet<usize>> = HashMap::new();
        if multi_path {
            for (path_idx, (_, root_id)) in snapshot.named_outputs.iter().enumerate() {
                // BFS from this root
                let mut bfs_visited: HashSet<u64> = HashSet::new();
                let mut bfs_queue = std::collections::VecDeque::new();
                bfs_queue.push_back(root_id.0);
                while let Some(nid) = bfs_queue.pop_front() {
                    if !bfs_visited.insert(nid) {
                        continue;
                    }
                    node_to_paths.entry(nid).or_default().insert(path_idx);
                    if let Some(snode) = node_map.get(&nid) {
                        for pid in &snode.parent_ids {
                            bfs_queue.push_back(pid.0);
                        }
                    }
                }
            }
        }

        // 获取节点的路径颜色（用于边和输出标记）
        let get_edge_color = |parent_id: u64, child_id: u64| -> &str {
            if !multi_path {
                return "#333333"; // 单路径默认深灰
            }
            let parent_paths = node_to_paths.get(&parent_id);
            let child_paths = node_to_paths.get(&child_id);
            match (parent_paths, child_paths) {
                (Some(pp), Some(cp)) => {
                    // 取交集
                    let common: HashSet<_> = pp.intersection(cp).collect();
                    if common.len() == 1 {
                        let idx = **common.iter().next().unwrap();
                        PATH_COLORS[idx % PATH_COLORS.len()]
                    } else {
                        "#888888" // 共享边：灰色
                    }
                }
                _ => "#333333",
            }
        };

        // 输出节点对应的路径颜色
        let output_path_color = |node_id: u64| -> &str {
            if !multi_path {
                return "#C62828"; // 单路径默认红色
            }
            for (idx, (_, id)) in snapshot.named_outputs.iter().enumerate() {
                if id.0 == node_id {
                    return PATH_COLORS[idx % PATH_COLORS.len()];
                }
            }
            "#C62828"
        };

        // 输出节点对应的路径名称
        let output_path_name = |node_id: u64| -> Option<&str> {
            for (name, id) in &snapshot.named_outputs {
                if id.0 == node_id {
                    return Some(name.as_str());
                }
            }
            None
        };

        // === 统一 cluster 收集（从 node_group_tag）===
        use super::graph::GroupStyle;

        struct ClusterInfo {
            group_type: String,
            #[allow(dead_code)]
            instance_id: usize,
            display_name: Option<String>,
            description: Option<String>,
            style: GroupStyle,
            visible_node_ids: Vec<u64>,
        }

        let mut clusters: Vec<ClusterInfo> = Vec::new();
        let mut cluster_key_to_idx: HashMap<(String, usize), usize> = HashMap::new();
        let mut rnn_hidden_ids: HashSet<u64> = HashSet::new();
        let mut node_to_cluster: HashMap<u64, usize> = HashMap::new();

        for snode in &snapshot.nodes {
            if let Some(tag) = &snode.node_group_tag {
                let key = (tag.group_type.clone(), tag.instance_id);
                let idx = *cluster_key_to_idx.entry(key).or_insert_with(|| {
                    let i = clusters.len();
                    clusters.push(ClusterInfo {
                        group_type: tag.group_type.clone(),
                        instance_id: tag.instance_id,
                        display_name: tag.display_name.clone(),
                        description: tag.description.clone(),
                        style: tag.style,
                        visible_node_ids: Vec::new(),
                    });
                    i
                });
                if tag.hidden {
                    rnn_hidden_ids.insert(snode.id.0);
                } else {
                    clusters[idx].visible_node_ids.push(snode.id.0);
                    node_to_cluster.insert(snode.id.0, idx);
                }
            }
        }

        // === RNN 折叠信息（从 RecurrentFoldingMeta）===
        let mut rnn_output_redirects: HashMap<u64, u64> = HashMap::new();
        let mut rnn_step_ranges: HashMap<u64, (usize, usize)> = HashMap::new();
        let mut rnn_init_state_ids: HashSet<u64> = HashSet::new();
        let mut rnn_input_step_ranges: HashMap<u64, (usize, usize)> = HashMap::new();
        let mut rnn_feedback_edges: Vec<(u64, u64, usize, usize)> = Vec::new();
        let mut rnn_init_edges: HashSet<(u64, u64)> = HashSet::new();
        let mut rnn_output_repr_step_ranges: HashMap<u64, (usize, usize)> = HashMap::new();
        let mut folding_step_ranges: HashMap<String, (usize, usize)> = HashMap::new();

        for meta in folding_metas {
            let matching_info = meta
                .unroll_infos
                .iter()
                .rev()
                .find(|info| visited.contains(&info.first_step_start_id));
            if let Some(info) = matching_info {
                if info.steps <= 1 {
                    continue;
                }
                let base = info.first_step_start_id.0;
                let nps = meta.nodes_per_step;

                if let Some(&repr_id) = info.repr_output_node_ids.first() {
                    rnn_output_redirects.insert(info.real_output_node_id.0, repr_id.0);
                }

                let min_steps = meta
                    .unroll_infos
                    .iter()
                    .map(|i| i.steps)
                    .min()
                    .unwrap_or(info.steps);
                let max_steps = meta
                    .unroll_infos
                    .iter()
                    .map(|i| i.steps)
                    .max()
                    .unwrap_or(info.steps);

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
                }

                for (repr_id, sid) in info
                    .repr_output_node_ids
                    .iter()
                    .zip(&info.init_state_node_ids)
                {
                    rnn_feedback_edges.push((repr_id.0, sid.0, min_steps, max_steps));
                }

                rnn_input_step_ranges.insert(info.input_node_id.0, (min_steps, max_steps));
                folding_step_ranges.insert(meta.name.clone(), (min_steps, max_steps));
            }
        }

        // cluster 颜色
        let group_colors = ["#E3F2FD80", "#E8F5E980", "#FFF3E080", "#F3E5F580"];

        // === DOT 输出归一化 ===
        let id_remap: HashMap<u64, u64> = snapshot
            .nodes
            .iter()
            .filter(|n| !rnn_hidden_ids.contains(&n.id.0))
            .enumerate()
            .map(|(i, n)| (n.id.0, (i + 1) as u64))
            .collect();
        let name_remap: HashMap<u64, String> = {
            let mut type_counters: HashMap<String, usize> = HashMap::new();
            snapshot
                .nodes
                .iter()
                .filter(|n| !rnn_hidden_ids.contains(&n.id.0))
                .map(|n| {
                    let id = n.id.0;
                    let node_type = &n.type_name;
                    let type_label = node_type.to_lowercase();
                    let raw_name = n.name.as_deref().unwrap_or(&type_label).to_string();
                    let display = match raw_name.split_once('/') {
                        Some((_, after)) => after.to_string(),
                        None => raw_name,
                    };
                    if node_type == "Parameter" {
                        (id, display)
                    } else {
                        let prefix = if let Some(pos) = display.rfind('_') {
                            if display[pos + 1..].chars().all(|c| c.is_ascii_digit())
                                && pos + 1 < display.len()
                            {
                                &display[..pos]
                            } else {
                                &display
                            }
                        } else {
                            &display
                        };
                        let counter = type_counters.entry(prefix.to_string()).or_insert(0);
                        *counter += 1;
                        (id, format!("{}_{}", prefix, counter))
                    }
                })
                .collect()
        };
        let rid = |id: u64| -> u64 { *id_remap.get(&id).unwrap_or(&id) };

        // 节点定义生成闭包
        let generate_node_def = |snode: &super::SnapshotNode| -> String {
            let id = snode.id.0;
            let stable_id = rid(id);
            let node_type = &snode.type_name;
            let name = name_remap.get(&id).cloned().unwrap_or_else(|| {
                let raw = snode
                    .name
                    .as_deref()
                    .unwrap_or(&node_type.to_lowercase())
                    .to_string();
                match raw.split_once('/') {
                    Some((_, after)) => after.to_string(),
                    None => raw,
                }
            });
            let shape = &snode.shape;

            // 形状字符串
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

            // Parameter 参数数量
            let param_count_str = if node_type == "Parameter" && !shape.is_empty() {
                let count: usize = shape.iter().product();
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

            // 超参数
            let hyperparam_str = snode.hyperparam_html.as_deref().unwrap_or("").to_string();

            // 节点样式
            let (node_shape, style, fill_color) = match node_type.as_str() {
                "Input" | "BasicInput" => ("ellipse", "filled", "#E3F2FD"),
                "TargetInput" => ("ellipse", "filled", "#FFE0B2"),
                "State" => ("cylinder", "filled", "#FFE0B2"),
                "Identity" => ("ellipse", "\"filled,dashed\"", "#E0E0E0"),
                "Detach" => ("ellipse", "\"filled,dashed\"", "#E1BEE7"),
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
                // 多输入合并节点：梯形（倒三角感），浅青色，突出数据流汇聚点
                "Stack" | "Concat" => ("invtrapezium", "filled", "#E0F2F1"),
                "Sigmoid" | "Tanh" | "ReLU" | "LeakyReLU" | "Sign" | "SoftPlus" | "Step"
                | "Softmax" | "LogSoftmax" | "Abs" | "Ln" => ("diamond", "filled", "#FFF3E0"),
                "Dropout" | "BatchNorm" => ("diamond", "filled", "#E1BEE7"),
                _ => ("box", "\"filled,rounded\"", "#FFFDE7"),
            };

            // RNN 步数标注
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

            // 输出节点标记：彩色粗边框 + 路径名称（替代旧版 peripheries=2 + ★ Output）
            let is_output = output_node_ids.contains(&id);
            let output_border_str = if is_output {
                let color = output_path_color(id);
                format!(" penwidth=2.5 color=\"{}\"", color)
            } else if let Some(&(min_s, _)) = rnn_step_ranges.get(&id) {
                if min_s > 1 {
                    " peripheries=2".to_string()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let output_suffix = if is_output {
                if let Some(path_name) = output_path_name(id) {
                    let color = output_path_color(id);
                    format!(
                        "<BR/><FONT COLOR=\"{}\" POINT-SIZE=\"9\">▸ {}</FONT>",
                        color, path_name
                    )
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            format!(
                "\"{}\" [label=<{}{}<BR/><B>{}</B><BR/>{}{}{}{}> shape={} style={} fillcolor=\"{}\" fontsize=10{}];\n",
                stable_id,
                name,
                rnn_step_str,
                node_type,
                shape_str,
                param_count_str,
                hyperparam_str,
                output_suffix,
                node_shape,
                style,
                fill_color,
                output_border_str
            )
        };

        // === DOT 输出 ===
        let mut dot = String::new();
        dot.push_str("digraph ComputeGraph {\n");
        dot.push_str("    rankdir=TB;\n");
        dot.push_str("    newrank=true;\n");
        dot.push_str("    splines=polyline;\n");
        dot.push_str("    node [fontname=\"Microsoft YaHei,SimHei,Arial\"];\n");
        dot.push_str("    edge [fontname=\"Microsoft YaHei,SimHei,Arial\"];\n\n");

        // === 统一 cluster 分组渲染 ===

        // 按模型前缀分类 cluster
        let mut model_cluster_indices: std::collections::BTreeMap<String, Vec<usize>> =
            std::collections::BTreeMap::new();
        let mut standalone_cluster_indices: Vec<usize> = Vec::new();
        let mut node_to_model: HashMap<u64, String> = HashMap::new();

        for (idx, cluster) in clusters.iter().enumerate() {
            if cluster.visible_node_ids.is_empty() {
                continue;
            }
            if let Some(display_name) = &cluster.display_name {
                if let Some((model, _)) = display_name.split_once('/') {
                    model_cluster_indices
                        .entry(model.to_string())
                        .or_default()
                        .push(idx);
                    for &nid in &cluster.visible_node_ids {
                        node_to_model.insert(nid, model.to_string());
                    }
                } else {
                    standalone_cluster_indices.push(idx);
                }
            } else {
                // Distribution cluster（无 display_name）→ 延迟处理
                standalone_cluster_indices.push(idx);
            }
        }

        // 迭代推断（将未分组节点归入正确的模型）
        if !model_cluster_indices.is_empty() {
            let mut node_children: HashMap<u64, Vec<u64>> = HashMap::new();
            for snode in &snapshot.nodes {
                // 被 RNN 折叠隐藏的节点不参与推断
                if rnn_hidden_ids.contains(&snode.id.0) {
                    continue;
                }
                for pid in &snode.parent_ids {
                    node_children.entry(pid.0).or_default().push(snode.id.0);
                }
            }

            loop {
                let mut changed = false;
                for snode in &snapshot.nodes {
                    let nid = snode.id.0;
                    if node_to_model.contains_key(&nid) || rnn_hidden_ids.contains(&nid) {
                        continue;
                    }

                    if !snode.parent_ids.is_empty() {
                        let parent_models: Vec<Option<&str>> = snode
                            .parent_ids
                            .iter()
                            .filter(|p| !rnn_hidden_ids.contains(&p.0))
                            .map(|p| node_to_model.get(&p.0).map(|s| s.as_str()))
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
                        // 叶子节点：所有子节点归同一模型 → 也归入
                        if let Some(children) = node_children.get(&nid) {
                            let child_models: Vec<Option<&str>> = children
                                .iter()
                                .map(|c| node_to_model.get(c).map(|s| s.as_str()))
                                .collect();
                            if child_models.iter().all(|m| m.is_some()) {
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

        // 虚拟 Input 节点（跨模型边中转）
        let mut virtual_input_map: HashMap<(u64, String), String> = HashMap::new();
        let mut virtual_inputs_by_model: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for snode in &snapshot.nodes {
            let child_id = snode.id.0;
            let child_model = node_to_model.get(&child_id);
            for pid in &snode.parent_ids {
                let parent_id = pid.0;
                let parent_model = node_to_model.get(&parent_id);
                if let (Some(pm), Some(cm)) = (parent_model, child_model) {
                    if pm != cm {
                        let key = (parent_id, cm.clone());
                        if !virtual_input_map.contains_key(&key) {
                            let virt_id =
                                format!("virt_{}_{}", rid(parent_id), cm.replace(' ', "_"));
                            let shape_str = if let Some(parent_node) = node_map.get(&parent_id) {
                                let shape = &parent_node.shape;
                                if shape.is_empty() {
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
                                }
                            } else {
                                String::new()
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

        let model_colors = ["#FFEBEE40", "#E8EAF640", "#E0F2F140", "#FFF8E140"];

        // Cluster 渲染闭包（按 GroupStyle 区分视觉样式）
        let render_cluster = |dot: &mut String,
                              indent: &str,
                              cluster: &ClusterInfo,
                              cluster_id: &str,
                              color: &str| {
            if cluster.visible_node_ids.is_empty() {
                return;
            }
            let display_name = cluster
                .display_name
                .as_ref()
                .map(|n| n.split('/').last().unwrap_or(n).to_string())
                .unwrap_or_else(|| cluster.group_type.clone());

            dot.push_str(&format!("{indent}subgraph cluster_{cluster_id} {{\n"));
            match cluster.style {
                GroupStyle::Layer => {
                    let desc = cluster.description.as_deref().unwrap_or("");
                    dot.push_str(&format!(
                            "{indent}    label=<<B>{display_name}</B><BR/><FONT POINT-SIZE=\"9\">{}: {desc}</FONT>>;\n",
                            cluster.group_type
                        ));
                    dot.push_str(&format!("{indent}    style=filled;\n"));
                    dot.push_str(&format!("{indent}    fillcolor=\"{color}\";\n"));
                }
                GroupStyle::Distribution => {
                    dot.push_str(&format!(
                        "{indent}    label=<<B>{}</B>>;\n",
                        cluster.group_type
                    ));
                    dot.push_str(&format!("{indent}    style=\"filled,dashed\";\n"));
                    dot.push_str(&format!("{indent}    fillcolor=\"#F3E5F540\";\n"));
                }
                GroupStyle::Recurrent => {
                    // 用折叠步数范围丰富描述
                    let base_desc = cluster.description.as_deref().unwrap_or("");
                    let enriched = if let Some(name) = &cluster.display_name {
                        if let Some(&(min_s, max_s)) = folding_step_ranges.get(name) {
                            if min_s != max_s {
                                if let Some(pos) = base_desc.rfind("(×") {
                                    format!("{}(×{}-{} steps)", &base_desc[..pos], min_s, max_s)
                                } else {
                                    base_desc.to_string()
                                }
                            } else {
                                base_desc.to_string()
                            }
                        } else {
                            base_desc.to_string()
                        }
                    } else {
                        base_desc.to_string()
                    };
                    dot.push_str(&format!(
                            "{indent}    label=<<B>{display_name}</B><BR/><FONT POINT-SIZE=\"9\">{enriched}</FONT>>;\n"
                        ));
                    dot.push_str(&format!("{indent}    style=\"filled,bold\";\n"));
                    dot.push_str(&format!("{indent}    peripheries=3;\n"));
                    dot.push_str(&format!("{indent}    penwidth=2;\n"));
                    dot.push_str(&format!("{indent}    fillcolor=\"{color}\";\n"));
                }
            }
            dot.push_str(&format!(
                "{indent}    fontname=\"Microsoft YaHei,SimHei,Arial\";\n"
            ));
            dot.push_str(&format!("{indent}    fontsize=11;\n"));
            dot.push_str(&format!("{indent}    margin=12;\n"));
            for snode in &snapshot.nodes {
                if cluster.visible_node_ids.contains(&snode.id.0) {
                    dot.push_str(&format!("{indent}        "));
                    dot.push_str(&generate_node_def(snode));
                }
            }
            dot.push_str(&format!("{indent}}}\n\n"));
        };

        // Distribution cluster 按模型归属分类
        let mut dist_for_model: HashMap<String, Vec<usize>> = HashMap::new();
        let mut toplevel_standalone: Vec<usize> = Vec::new();
        for &idx in &standalone_cluster_indices {
            let cluster = &clusters[idx];
            let model_hits: HashMap<&str, usize> = cluster
                .visible_node_ids
                .iter()
                .filter_map(|nid| node_to_model.get(nid).map(|m| m.as_str()))
                .fold(HashMap::new(), |mut acc, m| {
                    *acc.entry(m).or_insert(0) += 1;
                    acc
                });
            if model_hits.len() == 1 {
                let model = model_hits.keys().next().unwrap().to_string();
                dist_for_model.entry(model).or_default().push(idx);
            } else {
                toplevel_standalone.push(idx);
            }
        }

        // 渲染模型级嵌套 cluster
        for (model_idx, (model_name, cluster_indices)) in model_cluster_indices.iter().enumerate() {
            let has_visible = snapshot
                .nodes
                .iter()
                .any(|n| node_to_model.get(&n.id.0).map(|m| m.as_str()) == Some(model_name));
            if !has_visible {
                continue;
            }

            let model_id = model_name.replace(['-', '.', ' ', '/'], "_");
            let model_color = model_colors[model_idx % model_colors.len()];

            let model_param_count: usize = snapshot
                .nodes
                .iter()
                .filter(|n| {
                    node_to_model.get(&n.id.0).map(|m| m.as_str()) == Some(model_name)
                        && n.type_name == "Parameter"
                })
                .map(|n| n.shape.iter().product::<usize>())
                .sum();

            dot.push_str(&format!("    subgraph cluster_model_{model_id} {{\n"));
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

            // 模型内的 Layer/Recurrent cluster
            for &idx in cluster_indices {
                let cluster = &clusters[idx];
                let layer_name = cluster
                    .display_name
                    .as_ref()
                    .and_then(|n| n.split_once('/'))
                    .map(|(_, l)| l)
                    .unwrap_or(&cluster.group_type);
                let cluster_id = format!(
                    "{model_id}_{}",
                    layer_name.replace(['-', '.', ' ', '/'], "_")
                );
                let color = group_colors[idx % group_colors.len()];
                render_cluster(&mut dot, "        ", cluster, &cluster_id, color);
            }

            // 模型内的 Distribution cluster
            if let Some(dist_indices) = dist_for_model.get(model_name.as_str()) {
                for &idx in dist_indices {
                    let cluster = &clusters[idx];
                    let cluster_id = format!(
                        "{model_id}_group_{}_{}",
                        cluster.group_type.to_lowercase(),
                        cluster.instance_id
                    );
                    render_cluster(&mut dot, "        ", cluster, &cluster_id, "#F3E5F540");
                }
            }

            // 推断归属的散装节点
            for snode in &snapshot.nodes {
                let nid = snode.id.0;
                if !node_to_cluster.contains_key(&nid) && !rnn_hidden_ids.contains(&nid) {
                    if let Some(m) = node_to_model.get(&nid) {
                        if m == model_name {
                            dot.push_str("        ");
                            dot.push_str(&generate_node_def(snode));
                        }
                    }
                }
            }

            // 跨模型虚拟 Input 节点
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

        // 顶层独立 cluster
        for &idx in &toplevel_standalone {
            let cluster = &clusters[idx];
            let cluster_id = format!(
                "group_{}_{}",
                cluster.group_type.to_lowercase(),
                cluster.instance_id
            );
            let color = group_colors[idx % group_colors.len()];
            render_cluster(&mut dot, "    ", cluster, &cluster_id, color);
        }

        // 未分组节点
        for snode in &snapshot.nodes {
            let nid = snode.id.0;
            if !node_to_cluster.contains_key(&nid)
                && !node_to_model.contains_key(&nid)
                && !rnn_hidden_ids.contains(&nid)
            {
                dot.push_str("    ");
                dot.push_str(&generate_node_def(snode));
            }
        }
        dot.push('\n');

        // === 边生成（含路径着色）===
        for snode in &snapshot.nodes {
            let child_id = snode.id.0;
            if rnn_hidden_ids.contains(&child_id) {
                continue;
            }
            let child_model = node_to_model.get(&child_id);

            for pid in &snode.parent_ids {
                let original_parent_id = pid.0;
                let parent_id = *rnn_output_redirects
                    .get(&original_parent_id)
                    .unwrap_or(&original_parent_id);
                let was_redirected = original_parent_id != parent_id;

                if rnn_hidden_ids.contains(&parent_id) {
                    continue;
                }

                // RNN 特殊边样式
                let rnn_edge_attrs = if rnn_init_edges.contains(&(parent_id, child_id)) {
                    " style=dashed color=\"#E67E22\" label=<t=0> fontcolor=\"#E67E22\" fontsize=9"
                        .to_string()
                } else if was_redirected {
                    if let Some(&(min_s, max_s)) = rnn_output_repr_step_ranges.get(&parent_id) {
                        let t_label = if min_s == max_s {
                            format!("t={}", min_s - 1)
                        } else {
                            format!("t={}~{}", min_s - 1, max_s - 1)
                        };
                        format!(
                            " color=\"#E67E22\" label=<{}> fontcolor=\"#E67E22\" fontsize=9",
                            t_label
                        )
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                // 路径着色（仅在非 RNN 特殊边时应用）
                let path_color_attr = if rnn_edge_attrs.is_empty() {
                    let color = get_edge_color(parent_id, child_id);
                    if color != "#333333" {
                        format!(" color=\"{}\"", color)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                let edge_attrs = if !rnn_edge_attrs.is_empty() {
                    format!(" [{}]", rnn_edge_attrs)
                } else if !path_color_attr.is_empty() {
                    format!(" [{}]", path_color_attr.trim())
                } else {
                    String::new()
                };

                let parent_model = node_to_model.get(&parent_id);
                let is_cross_model = match (parent_model, child_model) {
                    (Some(pm), Some(cm)) => pm != cm,
                    _ => false,
                };

                if is_cross_model {
                    let key = (parent_id, child_model.unwrap().clone());
                    if let Some(virt_id) = virtual_input_map.get(&key) {
                        // 跨模型虚线（保持灰色虚线）
                        dot.push_str(&format!(
                            "    \"{}\" -> \"{}\" [style=dashed color=\"#999999\"];\n",
                            rid(parent_id),
                            virt_id
                        ));
                        dot.push_str(&format!(
                            "    \"{}\" -> \"{}\"{};\n",
                            virt_id,
                            rid(child_id),
                            edge_attrs
                        ));
                    }
                } else {
                    dot.push_str(&format!(
                        "    \"{}\" -> \"{}\"{};\n",
                        rid(parent_id),
                        rid(child_id),
                        edge_attrs
                    ));
                }
            }
        }

        // RNN 回流反馈边
        for (repr_id, init_state_id, min_steps, max_steps) in &rnn_feedback_edges {
            let label = if min_steps == max_steps {
                if *min_steps <= 2 {
                    "t=0".to_string()
                } else {
                    format!("t=0~{}", min_steps - 2)
                }
            } else {
                format!("t=0~({}~{})", min_steps - 2, max_steps - 2)
            };
            dot.push_str(&format!(
                "    \"{}\" -> \"{}\" [style=dashed color=\"#E67E22\" label=<{}> fontcolor=\"#E67E22\" fontsize=9 constraint=false];\n",
                rid(*repr_id), rid(*init_state_id), label
            ));
        }

        // === 同源数据节点链式虚线标注 ===
        // 按 data_source_id 分组，相同 source_id 的 Input/TargetInput 节点表示使用了同一份数据
        let mut source_groups: HashMap<u64, Vec<u64>> = HashMap::new();
        for snode in &snapshot.nodes {
            if let Some(sid) = snode.data_source_id {
                if !rnn_hidden_ids.contains(&snode.id.0) {
                    source_groups.entry(sid).or_default().push(snode.id.0);
                }
            }
        }
        for (_sid, ids) in &source_groups {
            if ids.len() < 2 {
                continue;
            }
            // 按模型名排序，确保链的顺序稳定
            let mut sorted = ids.clone();
            sorted.sort_by_key(|id| node_to_model.get(id).cloned().unwrap_or_default());
            // 取共享数据的节点名作为标签
            let shared_name = sorted
                .first()
                .and_then(|id| {
                    let original_id = *id;
                    snapshot
                        .nodes
                        .iter()
                        .find(|n| n.id.0 == original_id)
                        .and_then(|n| n.name.clone())
                })
                .unwrap_or_default();
            // 去除名字中的模型前缀（如 "Actor/obs" → "obs"）
            let label = match shared_name.split_once('/') {
                Some((_, after)) => after,
                None => &shared_name,
            };
            // 链式连接：A -- B -- C（n-1 条边）
            for pair in sorted.windows(2) {
                dot.push_str(&format!(
                    "    \"{}\" -> \"{}\" [style=dashed dir=none color=\"#4FC3F7\" constraint=false label=<= {}> fontcolor=\"#4FC3F7\" fontsize=8];\n",
                    rid(pair[0]), rid(pair[1]), label
                ));
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// 内部方法：从多个 Var 生成 DOT 格式
    ///
    /// 内部构建 snapshot 后委托给 `snapshot_to_dot`，确保行为一致。
    fn vars_to_dot(vars: &[&Self]) -> String {
        if vars.is_empty() {
            return "digraph ComputeGraph {}\n".to_string();
        }

        // 构建命名输出列表（自动命名）
        let named: Vec<(String, &Self)> = vars
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let name = if vars.len() == 1 {
                    "Output".to_string()
                } else {
                    format!("Output {}", i + 1)
                };
                (name, *v)
            })
            .collect();
        let named_refs: Vec<(&str, &Self)> = named.iter().map(|(n, v)| (n.as_str(), *v)).collect();

        // 构建 snapshot
        let snapshot = Self::build_snapshot(&named_refs);

        // 从图中读取折叠渲染元信息
        let folding_metas = vars[0]
            .graph
            .upgrade()
            .map(|g| g.borrow().recurrent_folding_metas().to_vec())
            .unwrap_or_default();

        Self::snapshot_to_dot(&snapshot, &folding_metas)
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

// Sub for &Var（使用原生 Subtract 节点）
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

// Neg for &Var（原生 Negate 节点）
impl Neg for &Var {
    type Output = Var;

    fn neg(self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_negate_node(Rc::clone(&self.node), None)
            .expect("创建 Negate 节点失败");
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
