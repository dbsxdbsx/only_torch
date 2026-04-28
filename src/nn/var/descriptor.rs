/*
 * @Author       : 老董
 * @Description  : 图描述符提取 — 将计算图导出为 GraphDescriptor IR
 *
 * 核心功能：
 * - `Var::to_graph_descriptor()`: BFS 遍历计算图，提取所有节点到 GraphDescriptor
 * - `node_type_to_descriptor()`: NodeType → NodeTypeDescriptor 映射
 *
 * 用于统一的 .otm 模型格式保存。
 */

use super::Var;
use crate::nn::NodeId;
use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use crate::nn::nodes::NodeInner;
use crate::nn::nodes::raw_node::NodeType;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

/// 将 NodeType 转换为 NodeTypeDescriptor（提取节点类型 + 配置参数）
fn node_type_to_descriptor(raw: &NodeType) -> NodeTypeDescriptor {
    match raw {
        // === 输入/参数/状态 ===
        NodeType::Input(variant) => {
            use crate::nn::nodes::raw_node::InputVariant;
            match variant {
                InputVariant::Data(_) => NodeTypeDescriptor::BasicInput,
                InputVariant::Target(_) => NodeTypeDescriptor::TargetInput,
            }
        }
        NodeType::Parameter(_) => NodeTypeDescriptor::Parameter,
        NodeType::State(_) => NodeTypeDescriptor::State,

        // === 算术 ===
        NodeType::Add(_) => NodeTypeDescriptor::Add,
        NodeType::Subtract(_) => NodeTypeDescriptor::Subtract,
        NodeType::Multiply(_) => NodeTypeDescriptor::Multiply,
        NodeType::Divide(_) => NodeTypeDescriptor::Divide,
        NodeType::Negate(_) => NodeTypeDescriptor::Negate,
        NodeType::MatMul(_) => NodeTypeDescriptor::MatMul,

        // === 卷积/池化 ===
        NodeType::Conv2d(c) => NodeTypeDescriptor::Conv2d {
            stride: c.stride(),
            padding: c.padding(),
            dilation: c.dilation(),
        },
        NodeType::ConvTranspose2d(c) => NodeTypeDescriptor::ConvTranspose2d {
            stride: c.stride(),
            padding: c.padding(),
            output_padding: c.output_padding(),
        },
        NodeType::DeformableConv2d(c) => NodeTypeDescriptor::DeformableConv2d {
            stride: c.stride(),
            padding: c.padding(),
            dilation: c.dilation(),
            deformable_groups: c.deformable_groups(),
        },
        NodeType::MaxPool2d(p) => {
            // raw_node 内部用 4 维 (top, bottom, left, right),IR 只承诺对称语义。
            // 全部入口都通过 create_max_pool2d_node 走对称展开,这里读到的应当
            // 满足 top == bottom && left == right。debug 模式下断言,release 时取
            // (top, left) 作为对称代表(若真出现非对称即数据丢失)。
            let (top, bottom, left, right) = p.padding();
            debug_assert_eq!(
                (top, left),
                (bottom, right),
                "MaxPool2d 内部 padding 应保持对称(top==bottom && left==right),\
                 IR 层只暴露对称形式;实际得到 (t,b,l,r)=({top},{bottom},{left},{right})"
            );
            NodeTypeDescriptor::MaxPool2d {
                kernel_size: p.kernel_size(),
                stride: p.stride(),
                padding: (top, left),
                ceil_mode: p.ceil_mode(),
            }
        }
        NodeType::AvgPool2d(p) => NodeTypeDescriptor::AvgPool2d {
            kernel_size: p.kernel_size(),
            stride: p.stride(),
        },
        NodeType::Upsample2d(u) => NodeTypeDescriptor::Upsample2d {
            scale_h: u.scale_h(),
            scale_w: u.scale_w(),
        },

        // === 形状变换 ===
        NodeType::Reshape(r) => NodeTypeDescriptor::Reshape {
            target_shape: r.target_shape().to_vec(),
        },
        NodeType::Flatten(f) => NodeTypeDescriptor::Flatten {
            keep_first_dim: f.keep_first_dim(),
        },
        NodeType::Select(s) => NodeTypeDescriptor::Select {
            axis: s.axis(),
            index: s.index(),
        },
        NodeType::Gather(g) => NodeTypeDescriptor::Gather { dim: g.dim() },
        NodeType::Narrow(n) => NodeTypeDescriptor::Narrow {
            axis: n.axis(),
            start: n.start(),
            length: n.length(),
        },
        NodeType::Permute(p) => NodeTypeDescriptor::Permute {
            dims: p.dims().to_vec(),
        },
        NodeType::Stack(s) => NodeTypeDescriptor::Stack { axis: s.axis() },
        NodeType::Concat(c) => NodeTypeDescriptor::Concat { axis: c.axis() },
        NodeType::Pad(p) => NodeTypeDescriptor::Pad {
            paddings: p.paddings().to_vec(),
            pad_value: p.pad_value(),
        },
        NodeType::Repeat(r) => NodeTypeDescriptor::Repeat {
            repeats: r.repeats().to_vec(),
        },

        // === 选择 ===
        NodeType::TopK(t) => NodeTypeDescriptor::TopK {
            k: t.k(),
            axis: t.axis(),
            sorted: t.sorted(),
        },
        NodeType::SortNode(s) => NodeTypeDescriptor::SortNode {
            axis: s.axis(),
            descending: s.descending(),
        },

        // === 归约 ===
        NodeType::Maximum(_) => NodeTypeDescriptor::Maximum,
        NodeType::Minimum(_) => NodeTypeDescriptor::Minimum,
        NodeType::Amax(a) => NodeTypeDescriptor::Amax { axis: a.axis() },
        NodeType::Amin(a) => NodeTypeDescriptor::Amin { axis: a.axis() },
        NodeType::Sum(s) => NodeTypeDescriptor::Sum { axis: s.axis() },
        NodeType::Mean(m) => NodeTypeDescriptor::Mean { axis: m.axis() },

        // === 激活函数 ===
        NodeType::Sigmoid(_) => NodeTypeDescriptor::Sigmoid,
        NodeType::Tanh(_) => NodeTypeDescriptor::Tanh,
        NodeType::ReLU(_) => NodeTypeDescriptor::ReLU,
        NodeType::LeakyReLU(l) => NodeTypeDescriptor::LeakyReLU { alpha: l.alpha() },
        NodeType::Softmax(_) => NodeTypeDescriptor::Softmax,
        NodeType::LogSoftmax(_) => NodeTypeDescriptor::LogSoftmax,
        NodeType::SoftPlus(_) => NodeTypeDescriptor::SoftPlus,
        NodeType::Gelu(_) => NodeTypeDescriptor::Gelu,
        NodeType::Swish(_) => NodeTypeDescriptor::Swish,
        NodeType::Elu(e) => NodeTypeDescriptor::Elu { alpha: e.alpha() },
        NodeType::Selu(_) => NodeTypeDescriptor::Selu,
        NodeType::Mish(_) => NodeTypeDescriptor::Mish,
        NodeType::HardSwish(_) => NodeTypeDescriptor::HardSwish,
        NodeType::HardSigmoid(_) => NodeTypeDescriptor::HardSigmoid,
        NodeType::ReLU6(_) => NodeTypeDescriptor::ReLU6,
        NodeType::HardTanh(h) => NodeTypeDescriptor::HardTanh {
            min_val: h.min_val(),
            max_val: h.max_val(),
        },
        NodeType::Step(_) => NodeTypeDescriptor::Step,
        NodeType::Sign(_) => NodeTypeDescriptor::Sign,
        NodeType::Abs(_) => NodeTypeDescriptor::Abs,
        NodeType::Ln(_) => NodeTypeDescriptor::Ln,
        NodeType::Log10(_) => NodeTypeDescriptor::Log10,
        NodeType::Log2(_) => NodeTypeDescriptor::Log2,
        NodeType::Exp(_) => NodeTypeDescriptor::Exp,
        NodeType::Sqrt(_) => NodeTypeDescriptor::Sqrt,
        NodeType::Pow(p) => NodeTypeDescriptor::Pow {
            exponent: p.exponent(),
        },
        NodeType::Square(_) => NodeTypeDescriptor::Square,
        NodeType::Reciprocal(_) => NodeTypeDescriptor::Reciprocal,

        // === 裁剪/条件 ===
        NodeType::Clip(c) => NodeTypeDescriptor::Clip {
            min: c.min(),
            max: c.max(),
        },
        NodeType::WhereCond(w) => {
            let cond = w.condition();
            NodeTypeDescriptor::WhereCond {
                condition_data: cond.data_as_slice().to_vec(),
                condition_shape: cond.shape().to_vec(),
            }
        }

        // === 损失函数 ===
        NodeType::MSE(m) => NodeTypeDescriptor::MSE {
            reduction: m.reduction(),
        },
        NodeType::MAE(m) => NodeTypeDescriptor::MAE {
            reduction: m.reduction(),
        },
        NodeType::BCE(b) => NodeTypeDescriptor::BCE {
            reduction: b.reduction(),
        },
        NodeType::Huber(h) => NodeTypeDescriptor::Huber {
            delta: h.delta(),
            reduction: h.reduction(),
        },
        NodeType::SoftmaxCrossEntropy(_) => NodeTypeDescriptor::SoftmaxCrossEntropy,

        // === 辅助/归一化 ===
        NodeType::Identity(_) => NodeTypeDescriptor::Identity,
        NodeType::Detach(_) => NodeTypeDescriptor::Detach,
        NodeType::Dropout(d) => NodeTypeDescriptor::Dropout { p: d.p() },
        NodeType::BatchNormOp(b) => NodeTypeDescriptor::BatchNormOp {
            eps: b.eps(),
            momentum: b.momentum(),
            num_features: b.num_features(),
        },
        NodeType::LayerNormOp(l) => NodeTypeDescriptor::LayerNormOp {
            normalized_dims: l.normalized_dims(),
            eps: l.eps(),
        },
        NodeType::RMSNormOp(r) => NodeTypeDescriptor::RMSNormOp {
            normalized_dims: r.normalized_dims(),
            eps: r.eps(),
        },
        NodeType::ZerosLike(_) => NodeTypeDescriptor::ZerosLike,
    }
}

/// 从 NodeInner 提取 NodeDescriptor
fn extract_node_descriptor(node: &Rc<NodeInner>) -> NodeDescriptor {
    let node_type = node.with_raw_node(node_type_to_descriptor);
    let output_shape = node.value_expected_shape();
    let dyn_shape = node.dynamic_expected_shape();

    // 转换动态形状
    let dynamic_shape = if dyn_shape.has_dynamic_dims() {
        Some(dyn_shape.dims().to_vec())
    } else {
        None
    };

    let parent_ids: Vec<u64> = node.parents().iter().map(|p| p.id().0).collect();

    NodeDescriptor::new(
        node.id().0,
        node.name().unwrap_or(""),
        node_type,
        output_shape,
        dynamic_shape,
        parent_ids,
    )
}

impl Var {
    /// 将以当前 Var 为输出的计算图导出为 GraphDescriptor
    ///
    /// BFS 遍历所有祖先节点，按拓扑顺序（输入在前、输出在后）提取节点描述符。
    ///
    /// # 返回
    /// `GraphDescriptor` 包含完整的图拓扑和节点配置参数
    pub fn to_graph_descriptor(&self) -> GraphDescriptor {
        Self::vars_to_graph_descriptor(&[self], "model")
    }

    /// 从多个输出 Var 构建 GraphDescriptor（支持多输出模型）
    ///
    /// # 参数
    /// - `vars`: 输出 Var 列表
    /// - `name`: 图名称
    pub fn vars_to_graph_descriptor(vars: &[&Self], name: &str) -> GraphDescriptor {
        // 第 1 步：BFS 收集所有可达节点
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut all_nodes: Vec<Rc<NodeInner>> = Vec::new();

        let mut queue: VecDeque<Rc<NodeInner>> = VecDeque::new();
        for var in vars {
            queue.push_back(Rc::clone(var.node()));
        }
        while let Some(node) = queue.pop_front() {
            let id = node.id();
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id);
            all_nodes.push(Rc::clone(&node));
            for parent in node.parents() {
                queue.push_back(Rc::clone(parent));
            }
        }

        // 第 2 步：Kahn 拓扑排序（parents 在前，children 在后）
        //
        // 在这个图中，"边" 从 child 指向 parent（child 依赖 parent），
        // 所以 parent 是 child 的前驱。
        // Kahn 的 in-degree 指的是每个节点有多少前驱（即 parents）*在子图内*。

        // 构建节点描述和 in-degree 映射
        let node_descs: HashMap<NodeId, NodeDescriptor> = all_nodes
            .iter()
            .map(|n| (n.id(), extract_node_descriptor(n)))
            .collect();

        // children_of[parent_id] = [child_ids...]
        let mut children_of: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();

        for node in &all_nodes {
            let nid = node.id();
            in_degree.entry(nid).or_insert(0);
            children_of.entry(nid).or_default();
            for parent in node.parents() {
                let pid = parent.id();
                if visited.contains(&pid) {
                    *in_degree.entry(nid).or_insert(0) += 1;
                    children_of.entry(pid).or_default().push(nid);
                }
            }
        }

        // BFS：从 in_degree=0 的节点开始
        let mut topo_queue: VecDeque<NodeId> = VecDeque::new();
        for (&nid, &deg) in &in_degree {
            if deg == 0 {
                topo_queue.push_back(nid);
            }
        }

        let mut sorted: Vec<NodeDescriptor> = Vec::with_capacity(all_nodes.len());
        while let Some(nid) = topo_queue.pop_front() {
            if let Some(desc) = node_descs.get(&nid) {
                sorted.push(desc.clone());
            }
            if let Some(children) = children_of.get(&nid) {
                for &child_id in children {
                    let deg = in_degree.get_mut(&child_id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        topo_queue.push_back(child_id);
                    }
                }
            }
        }

        let mut result = GraphDescriptor::new(name);
        for node in sorted {
            result.add_node(node);
        }
        result
    }
}
