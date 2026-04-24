/*
 * @Author       : 老董
 * @Description  : GraphDescriptor → Graph 重建
 *
 * 核心功能：
 * - `Graph::from_descriptor()`: 从 GraphDescriptor 重建计算图
 * - 按拓扑序逐节点调用 create_*_node，维护 old_id → new_Var 映射
 *
 * 用于统一的 .otm 模型加载（通用路径）。
 */

use super::error::GraphError;
use super::handle::Graph;
use super::onnx_import::ImportReport;
use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use crate::nn::layer::{Gru, Lstm, Rnn};
use crate::nn::var::Var;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// 从 GraphDescriptor 重建图的结果
pub struct RebuildResult {
    /// 重建后的图
    pub graph: Graph,
    /// 数据输入节点（名称, Var）—— 按 descriptor 中出现顺序
    pub inputs: Vec<(String, Var)>,
    /// 目标输入节点（名称, Var）—— 按 descriptor 中出现顺序
    pub targets: Vec<(String, Var)>,
    /// 输出节点（descriptor 中没有被其他节点引用为 parent 的节点）
    pub outputs: Vec<Var>,
    /// 可训练参数 Var 列表 —— 直接用于优化器
    ///
    /// 与 `inputs` / `outputs` 对称：加载完模型即可拿到全套训练所需的 Var。
    pub parameters: Vec<Var>,
    /// 旧 ID → 新 Var 的完整映射
    pub node_map: HashMap<u64, Var>,
    /// ONNX 导入路径的可观测报告（仅 `from_onnx*` 路径填充，其他重建路径为 `None`）
    pub import_report: Option<ImportReport>,
}

impl Graph {
    /// 从 GraphDescriptor 重建计算图
    ///
    /// descriptor 中节点必须按拓扑序排列（叶子在前，输出在后），
    /// 即 `Var::vars_to_graph_descriptor()` 生成的顺序。
    ///
    /// # 返回
    /// `RebuildResult` 包含新图、输入/输出 Var 和完整的节点映射。
    ///
    /// # 注意
    /// - Parameter 节点使用默认初始化，权重由后续 load 步骤填充
    /// - Dropout 使用固定 seed=42，加载后建议设为 eval 模式
    /// - BatchNormOp 的 running_mean/running_var 初始化为零
    /// 从 GraphDescriptor 重建计算图（使用指定种子，每代 build 可复现）
    pub fn from_descriptor_seeded(
        desc: &GraphDescriptor,
        seed: u64,
    ) -> Result<RebuildResult, GraphError> {
        let graph = Graph::new_with_seed(seed).with_model_name("EvolutionNet");
        Self::rebuild_into(graph, desc)
    }

    pub fn from_descriptor(desc: &GraphDescriptor) -> Result<RebuildResult, GraphError> {
        let graph = Graph::new();
        Self::rebuild_into(graph, desc)
    }

    fn rebuild_into(graph: Graph, desc: &GraphDescriptor) -> Result<RebuildResult, GraphError> {
        let mut node_map: HashMap<u64, Var> = HashMap::new();
        let mut inputs: Vec<(String, Var)> = Vec::new();
        let mut targets: Vec<(String, Var)> = Vec::new();

        // 收集所有被引用为 parent 的 ID，用于确定输出节点
        let all_parent_ids: HashSet<u64> = desc
            .nodes
            .iter()
            .flat_map(|n| n.parents.iter().copied())
            .collect();

        for node_desc in &desc.nodes {
            let var = rebuild_node(&graph, node_desc, &node_map)?;

            // 归类输入/目标节点
            match &node_desc.node_type {
                NodeTypeDescriptor::BasicInput => {
                    inputs.push((node_desc.name.clone(), var.clone()));
                }
                NodeTypeDescriptor::TargetInput => {
                    targets.push((node_desc.name.clone(), var.clone()));
                }
                _ => {}
            }

            node_map.insert(node_desc.id, var);
        }

        // 输出节点：ID 不被任何其他节点引用为 parent
        let outputs: Vec<Var> = desc
            .nodes
            .iter()
            .filter(|n| !all_parent_ids.contains(&n.id))
            .filter_map(|n| node_map.get(&n.id))
            .cloned()
            .collect();

        // 参数 Var 列表：从图的参数注册表收集
        let inner_rc = graph.inner_rc();
        let parameters: Vec<Var> = graph
            .inner()
            .get_all_parameters()
            .into_iter()
            .map(|(_, node)| Var::new_with_rc_graph(node, &inner_rc))
            .collect();

        Ok(RebuildResult {
            graph,
            inputs,
            targets,
            outputs,
            parameters,
            node_map,
            import_report: None,
        })
    }
}

// ==================== 内部实现 ====================

/// 获取单个父节点的 Rc<NodeInner>
fn get_parent(
    node_desc: &NodeDescriptor,
    node_map: &HashMap<u64, Var>,
    index: usize,
) -> Result<Rc<crate::nn::nodes::NodeInner>, GraphError> {
    let parent_id = node_desc.parents.get(index).ok_or_else(|| {
        GraphError::InvalidOperation(format!(
            "节点 '{}' (id={}) 缺少第 {} 个父节点",
            node_desc.name, node_desc.id, index
        ))
    })?;
    let parent_var = node_map.get(parent_id).ok_or_else(|| {
        GraphError::InvalidOperation(format!(
            "节点 '{}' (id={}) 的父节点 id={} 未找到（可能 descriptor 未按拓扑序排列）",
            node_desc.name, node_desc.id, parent_id
        ))
    })?;
    Ok(Rc::clone(parent_var.node()))
}

/// 获取所有父节点的 Rc<NodeInner>
fn get_all_parents(
    node_desc: &NodeDescriptor,
    node_map: &HashMap<u64, Var>,
) -> Result<Vec<Rc<crate::nn::nodes::NodeInner>>, GraphError> {
    node_desc
        .parents
        .iter()
        .map(|pid| {
            let parent_var = node_map.get(pid).ok_or_else(|| {
                GraphError::InvalidOperation(format!(
                    "节点 '{}' (id={}) 的父节点 id={} 未找到",
                    node_desc.name, node_desc.id, pid
                ))
            })?;
            Ok(Rc::clone(parent_var.node()))
        })
        .collect()
}

/// 获取指定父节点的 Var（用于循环单元重建）
fn get_parent_var(
    node_desc: &NodeDescriptor,
    node_map: &HashMap<u64, Var>,
    index: usize,
    param_name: &str,
) -> Result<Var, GraphError> {
    let parent_id = node_desc.parents.get(index).ok_or_else(|| {
        GraphError::InvalidOperation(format!(
            "节点 '{}' (id={}) 缺少第 {} 个父节点 ({})",
            node_desc.name, node_desc.id, index, param_name
        ))
    })?;
    node_map.get(parent_id).cloned().ok_or_else(|| {
        GraphError::InvalidOperation(format!(
            "节点 '{}' (id={}) 的父节点 {} (id={}) 未在 node_map 中找到",
            node_desc.name, node_desc.id, param_name, parent_id
        ))
    })
}

/// 确保循环单元输入 Var 有占位值，供 `validate_input` 推断 seq_len。
///
/// 处理三种情况：
/// 1. 若输入 Var 已有值（如前一层已 set_value），直接返回。
/// 2. 否则尝试调用 `forward()`，让前驱节点链（如 Stack）计算出值。
/// 3. 若 forward 失败（典型为纯 Input 节点），回退到 `set_value(zeros)`。
fn ensure_recurrent_input_value(input_var: &Var, shape: &[usize]) -> Result<(), GraphError> {
    if let Ok(Some(_)) = input_var.value() {
        return Ok(());
    }
    if input_var.forward().is_ok() {
        if let Ok(Some(_)) = input_var.value() {
            return Ok(());
        }
    }
    input_var.set_value(&Tensor::zeros(shape))
}

/// 根据节点描述重建单个节点
fn rebuild_node(
    graph: &Graph,
    node_desc: &NodeDescriptor,
    node_map: &HashMap<u64, Var>,
) -> Result<Var, GraphError> {
    let name = Some(node_desc.name.as_str());
    let inner_rc = graph.inner_rc();

    match &node_desc.node_type {
        // ==================== 输入/参数/状态 ====================
        NodeTypeDescriptor::BasicInput => {
            let node = graph
                .inner_mut()
                .create_basic_input_node(&node_desc.output_shape, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::TargetInput => {
            let node = graph
                .inner_mut()
                .create_target_input_node(&node_desc.output_shape, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Parameter => {
            let node = graph
                .inner_mut()
                .create_parameter_node(&node_desc.output_shape, name)?;
            // 注册参数（使权重 save/load 正常工作）
            graph
                .inner_mut()
                .register_parameter(node_desc.name.clone(), Rc::downgrade(&node))?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::State => {
            let node = graph
                .inner_mut()
                .create_state_node(&node_desc.output_shape, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 算术 ====================
        NodeTypeDescriptor::Add => {
            let parents = get_all_parents(node_desc, node_map)?;
            let node = graph.inner_mut().create_add_node(parents, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Subtract => {
            let parents = get_all_parents(node_desc, node_map)?;
            let node = graph.inner_mut().create_subtract_node(parents, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Multiply => {
            let parents = get_all_parents(node_desc, node_map)?;
            let node = graph.inner_mut().create_multiply_node(parents, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Divide => {
            let parents = get_all_parents(node_desc, node_map)?;
            let node = graph.inner_mut().create_divide_node(parents, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Negate => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_negate_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::MatMul => {
            let parents = get_all_parents(node_desc, node_map)?;
            let node = graph.inner_mut().create_mat_mul_node(parents, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 卷积/池化 ====================
        NodeTypeDescriptor::Conv2d {
            stride,
            padding,
            dilation,
        } => {
            let parents = get_all_parents(node_desc, node_map)?;
            let node = graph
                .inner_mut()
                .create_conv2d_node(parents, *stride, *padding, *dilation, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::ConvTranspose2d {
            stride,
            padding,
            output_padding,
        } => {
            let parents = get_all_parents(node_desc, node_map)?;
            let node = graph.inner_mut().create_conv_transpose2d_node(
                parents,
                *stride,
                *padding,
                *output_padding,
                name,
            )?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::MaxPool2d {
            kernel_size,
            stride,
            padding,
            ceil_mode,
        } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_max_pool2d_node(
                parent,
                *kernel_size,
                Some(*stride),
                *padding,
                *ceil_mode,
                name,
            )?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::AvgPool2d {
            kernel_size,
            stride,
        } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_avg_pool2d_node(
                parent,
                *kernel_size,
                Some(*stride),
                name,
            )?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Upsample2d { scale_h, scale_w } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph
                .inner_mut()
                .create_upsample2d_node(parent, *scale_h, *scale_w, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 形状变换 ====================
        NodeTypeDescriptor::Flatten { keep_first_dim } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph
                .inner_mut()
                .create_flatten_node(parent, *keep_first_dim, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Reshape { target_shape } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph
                .inner_mut()
                .create_reshape_node(parent, target_shape, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Select { axis, index } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph
                .inner_mut()
                .create_select_node(parent, *axis, *index, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Gather { dim } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let index = get_parent(node_desc, node_map, 1)?;
            let node = graph
                .inner_mut()
                .create_gather_node(input, index, *dim, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Narrow {
            axis,
            start,
            length,
        } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph
                .inner_mut()
                .create_narrow_node(parent, *axis, *start, *length, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Permute { dims } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_permute_node(parent, dims, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Stack { axis } => {
            let parents = get_all_parents(node_desc, node_map)?;
            let node = graph.inner_mut().create_stack_node(parents, *axis, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Concat { axis } => {
            let parents = get_all_parents(node_desc, node_map)?;
            let node = graph.inner_mut().create_concat_node(parents, *axis, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Pad {
            paddings,
            pad_value,
        } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node =
                graph
                    .inner_mut()
                    .create_pad_node(parent, paddings.clone(), *pad_value, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Repeat { repeats } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph
                .inner_mut()
                .create_repeat_node(parent, repeats.clone(), name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 选择 ====================
        NodeTypeDescriptor::TopK { k, axis, sorted } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph
                .inner_mut()
                .create_topk_node(parent, *k, *axis, *sorted, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::SortNode { axis, descending } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph
                .inner_mut()
                .create_sort_node(parent, *axis, *descending, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 归约 ====================
        NodeTypeDescriptor::Maximum => {
            let a = get_parent(node_desc, node_map, 0)?;
            let b = get_parent(node_desc, node_map, 1)?;
            let node = graph.inner_mut().create_maximum_node(a, b, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Minimum => {
            let a = get_parent(node_desc, node_map, 0)?;
            let b = get_parent(node_desc, node_map, 1)?;
            let node = graph.inner_mut().create_minimum_node(a, b, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Amax { axis } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_amax_node(input, *axis, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Amin { axis } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_amin_node(input, *axis, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Sum { axis } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_sum_node(input, *axis, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Mean { axis } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_mean_node(input, *axis, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 激活函数 ====================
        NodeTypeDescriptor::Sigmoid => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_sigmoid_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Tanh => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_tanh_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::ReLU => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_relu_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::LeakyReLU { alpha } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph
                .inner_mut()
                .create_leaky_relu_node(parent, *alpha, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Softmax => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_softmax_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::LogSoftmax => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_log_softmax_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::SoftPlus => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_softplus_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Gelu => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_gelu_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Swish => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_swish_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Elu { alpha } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_elu_node(parent, *alpha, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Selu => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_selu_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Mish => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_mish_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::HardSwish => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_hard_swish_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::HardSigmoid => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_hard_sigmoid_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::ReLU6 => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_relu6_node(parent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::HardTanh { min_val, max_val } => {
            let parent = get_parent(node_desc, node_map, 0)?;
            let node = graph
                .inner_mut()
                .create_hard_tanh_node(parent, *min_val, *max_val, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 逐元素数学运算 ====================
        NodeTypeDescriptor::Exp => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_exp_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Sqrt => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_sqrt_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Ln => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_ln_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Log10 => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_log10_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Log2 => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_log2_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Pow { exponent } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_pow_node(input, *exponent, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Square => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_square_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Reciprocal => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_reciprocal_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Abs => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_abs_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Sign => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_sign_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Step => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_step_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 裁剪/条件 ====================
        NodeTypeDescriptor::Clip { min, max } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph
                .inner_mut()
                .create_clip_node(input, *min, *max, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::WhereCond {
            condition_data,
            condition_shape,
        } => {
            let x = get_parent(node_desc, node_map, 0)?;
            let y = get_parent(node_desc, node_map, 1)?;
            let condition = Tensor::new(condition_data, condition_shape);
            let node = graph
                .inner_mut()
                .create_where_cond_node(x, y, condition, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 辅助 ====================
        NodeTypeDescriptor::Identity => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_identity_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Detach => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node = graph.inner_mut().create_detach_node(input, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Dropout { p } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let seed = graph.inner_mut().next_seed();
            let node = graph
                .inner_mut()
                .create_dropout_node(input, *p, seed, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::ZerosLike => {
            let reference = get_parent(node_desc, node_map, 0)?;
            // feature_shape = output_shape[1..] （output_shape 是 [1, feature_dims...]）
            let feature_shape = if node_desc.output_shape.len() > 1 {
                &node_desc.output_shape[1..]
            } else {
                &node_desc.output_shape[..]
            };
            let node = graph
                .inner_mut()
                .create_zeros_like_node(reference, feature_shape, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 归一化 ====================
        NodeTypeDescriptor::BatchNormOp {
            eps,
            momentum,
            num_features,
        } => {
            let input = get_parent(node_desc, node_map, 0)?;
            // 初始化 running stats 为零（后续由 load 恢复）
            let running_mean = Rc::new(std::cell::RefCell::new(Tensor::zeros(&[1, *num_features])));
            let running_var = Rc::new(std::cell::RefCell::new(Tensor::ones(&[1, *num_features])));
            let node = graph.inner_mut().create_batch_norm_op_node(
                input,
                *eps,
                *momentum,
                running_mean,
                running_var,
                name,
            )?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::LayerNormOp {
            normalized_dims,
            eps,
        } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node =
                graph
                    .inner_mut()
                    .create_layer_norm_op_node(input, *normalized_dims, *eps, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::RMSNormOp {
            normalized_dims,
            eps,
        } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let node =
                graph
                    .inner_mut()
                    .create_rms_norm_op_node(input, *normalized_dims, *eps, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 损失函数 ====================
        NodeTypeDescriptor::MSE { reduction } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let target = get_parent(node_desc, node_map, 1)?;
            let node = graph
                .inner_mut()
                .create_mse_node(input, target, *reduction, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::MAE { reduction } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let target = get_parent(node_desc, node_map, 1)?;
            let node = graph
                .inner_mut()
                .create_mae_node(input, target, *reduction, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::BCE { reduction } => {
            let logits = get_parent(node_desc, node_map, 0)?;
            let target = get_parent(node_desc, node_map, 1)?;
            let node = graph
                .inner_mut()
                .create_bce_node(logits, target, *reduction, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::Huber { delta, reduction } => {
            let input = get_parent(node_desc, node_map, 0)?;
            let target = get_parent(node_desc, node_map, 1)?;
            let node = graph
                .inner_mut()
                .create_huber_node(input, target, *reduction, *delta, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        NodeTypeDescriptor::SoftmaxCrossEntropy => {
            let logits = get_parent(node_desc, node_map, 0)?;
            let labels = get_parent(node_desc, node_map, 1)?;
            let node = graph
                .inner_mut()
                .create_softmax_cross_entropy_node(logits, labels, name)?;
            Ok(Var::new_with_rc_graph(node, &inner_rc))
        }

        // ==================== 循环单元（复合模板节点）====================
        NodeTypeDescriptor::CellRnn {
            input_size,
            hidden_size,
            return_sequences,
            seq_len,
        } => {
            let effective_seq = (*seq_len).max(1);
            let input_var = get_parent_var(node_desc, node_map, 0, "input")?;
            // validate_input 需要读取值来确定 seq_len；对 Input 节点直接 set_value 零值占位，
            // 对派生节点（如前一层 RNN 的 Stack 输出）走 forward 以得到真实形状。
            ensure_recurrent_input_value(&input_var, &[1, effective_seq, *input_size])?;
            let w_ih = get_parent_var(node_desc, node_map, 1, "w_ih")?;
            let w_hh = get_parent_var(node_desc, node_map, 2, "w_hh")?;
            let b_h = get_parent_var(node_desc, node_map, 3, "b_h")?;
            let rnn = Rnn::from_vars(w_ih, w_hh, b_h, *input_size, *hidden_size);
            if *return_sequences {
                rnn.forward_seq(input_var)
            } else {
                rnn.forward(input_var)
            }
        }

        NodeTypeDescriptor::CellLstm {
            input_size,
            hidden_size,
            return_sequences,
            seq_len,
        } => {
            let effective_seq = (*seq_len).max(1);
            let input_var = get_parent_var(node_desc, node_map, 0, "input")?;
            ensure_recurrent_input_value(&input_var, &[1, effective_seq, *input_size])?;
            let w_ii = get_parent_var(node_desc, node_map, 1, "w_ii")?;
            let w_hi = get_parent_var(node_desc, node_map, 2, "w_hi")?;
            let b_i = get_parent_var(node_desc, node_map, 3, "b_i")?;
            let w_if = get_parent_var(node_desc, node_map, 4, "w_if")?;
            let w_hf = get_parent_var(node_desc, node_map, 5, "w_hf")?;
            let b_f = get_parent_var(node_desc, node_map, 6, "b_f")?;
            let w_ig = get_parent_var(node_desc, node_map, 7, "w_ig")?;
            let w_hg = get_parent_var(node_desc, node_map, 8, "w_hg")?;
            let b_g = get_parent_var(node_desc, node_map, 9, "b_g")?;
            let w_io = get_parent_var(node_desc, node_map, 10, "w_io")?;
            let w_ho = get_parent_var(node_desc, node_map, 11, "w_ho")?;
            let b_o = get_parent_var(node_desc, node_map, 12, "b_o")?;
            let lstm = Lstm::from_vars(
                w_ii,
                w_hi,
                b_i,
                w_if,
                w_hf,
                b_f,
                w_ig,
                w_hg,
                b_g,
                w_io,
                w_ho,
                b_o,
                *input_size,
                *hidden_size,
            );
            if *return_sequences {
                lstm.forward_seq(input_var)
            } else {
                lstm.forward(input_var)
            }
        }

        NodeTypeDescriptor::CellGru {
            input_size,
            hidden_size,
            return_sequences,
            seq_len,
        } => {
            let effective_seq = (*seq_len).max(1);
            let input_var = get_parent_var(node_desc, node_map, 0, "input")?;
            ensure_recurrent_input_value(&input_var, &[1, effective_seq, *input_size])?;
            let w_ir = get_parent_var(node_desc, node_map, 1, "w_ir")?;
            let w_hr = get_parent_var(node_desc, node_map, 2, "w_hr")?;
            let b_r = get_parent_var(node_desc, node_map, 3, "b_r")?;
            let w_iz = get_parent_var(node_desc, node_map, 4, "w_iz")?;
            let w_hz = get_parent_var(node_desc, node_map, 5, "w_hz")?;
            let b_z = get_parent_var(node_desc, node_map, 6, "b_z")?;
            let w_in = get_parent_var(node_desc, node_map, 7, "w_in")?;
            let w_hn = get_parent_var(node_desc, node_map, 8, "w_hn")?;
            let b_n = get_parent_var(node_desc, node_map, 9, "b_n")?;
            let gru = Gru::from_vars(
                w_ir,
                w_hr,
                b_r,
                w_iz,
                w_hz,
                b_z,
                w_in,
                w_hn,
                b_n,
                *input_size,
                *hidden_size,
            );
            if *return_sequences {
                gru.forward_seq(input_var)
            } else {
                gru.forward(input_var)
            }
        }
    }
}
