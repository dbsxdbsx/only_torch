/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner Graphviz DOT 可视化
 */

use super::super::error::{GraphError, ImageFormat, VisualizationOutput};
use super::super::types::{GroupKind, LayerGroup};
use super::GraphInner;
use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use crate::nn::nodes::raw_node::InputVariant;
use crate::nn::nodes::NodeType;
use crate::nn::NodeId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::process::Command;

impl GraphInner {
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
    pub fn to_dot(&self) -> String {
        self.to_dot_with_options(true)
    }

    /// 生成带层分组选项的 DOT 格式字符串（内部方法）
    ///
    /// # 参数
    /// - `group_layers`: 是否将同一层的节点用半透明框分组显示
    pub(in crate::nn::graph) fn to_dot_with_options(&self, group_layers: bool) -> String {
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
        let ever_detached_nodes: HashSet<u64> = self
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
        let model_groups: Vec<LayerGroup> = self
            .layer_groups
            .iter()
            .filter(|g| g.kind == GroupKind::Model)
            .cloned()
            .collect();
        let layer_groups: Vec<LayerGroup> = self
            .layer_groups
            .iter()
            .filter(|g| g.kind == GroupKind::Layer)
            .cloned()
            .collect();

        // 收集所有已分组的节点 ID
        let grouped_node_ids: HashSet<u64> = if group_layers {
            self.layer_groups
                .iter()
                .flat_map(|g| g.node_ids.iter().map(|id| id.0))
                .collect()
        } else {
            HashSet::new()
        };

        // 收集所有需要隐藏的节点 ID（如循环层的非代表性展开节点）
        let mut hidden_node_ids: HashSet<u64> = if group_layers {
            self.layer_groups
                .iter()
                .flat_map(|g| g.hidden_node_ids.iter().map(|id| id.0))
                .collect()
        } else {
            HashSet::new()
        };

        // 分离 Model 分组和 Layer 分组（可变版本，用于后续更新）
        let mut model_groups: Vec<LayerGroup> = model_groups;
        let mut layer_groups: Vec<LayerGroup> = layer_groups;
        let mut grouped_node_ids = grouped_node_ids;

        // 扩展隐藏节点的后代（从非代表性 RNN 输出开始的所有下游节点）
        if group_layers && !self.recurrent_layer_metas.is_empty() {
            // 构建父→子映射
            let mut children_map: HashMap<u64, Vec<u64>> = HashMap::new();
            for node in &desc.nodes {
                for &parent_id in &node.parents {
                    children_map.entry(parent_id).or_default().push(node.id);
                }
            }

            // 收集代表性调用的输入节点 ID（用于标记受保护路径）
            let repr_input_ids: HashSet<u64> = self
                .recurrent_layer_metas
                .iter()
                .filter_map(|m| m.unroll_infos.last())
                .map(|info| info.input_node_id.0)
                .collect();

            // BFS 标记代表性路径上的所有节点为"受保护"
            let mut protected_nodes: HashSet<u64> = repr_input_ids.iter().copied().collect();
            let mut queue: VecDeque<u64> = repr_input_ids.iter().copied().collect();
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
            let mut queue: VecDeque<u64> = hidden_roots
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
                        if !hidden_node_ids.contains(&child_id)
                            && !protected_nodes.contains(&child_id)
                        {
                            queue.push_back(child_id);
                        }
                    }
                }
            }

            // 识别 Loss 节点（不应该包含在 Model 分组中）
            let loss_node_ids: HashSet<u64> = desc
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

            // 从 protected_nodes 中排除 Loss 节点
            let model_nodes: HashSet<u64> = protected_nodes
                .iter()
                .copied()
                .filter(|id| !loss_node_ids.contains(id))
                .collect();

            // 更新 Layer 分组以包含代表性调用的节点
            for layer in &mut layer_groups {
                // 移除被隐藏的节点
                layer.node_ids.retain(|id| !hidden_node_ids.contains(&id.0));

                // 收集该层的参数节点 ID
                let param_node_ids: HashSet<u64> = layer
                    .node_ids
                    .iter()
                    .filter(|id| {
                        desc.nodes
                            .iter()
                            .find(|n| n.id == id.0)
                            .is_some_and(|n| matches!(n.node_type, NodeTypeDescriptor::Parameter))
                    })
                    .map(|id| id.0)
                    .collect();

                if param_node_ids.is_empty() {
                    continue;
                }

                // 找到代表性路径上使用这些参数的计算节点
                for &node_id in &model_nodes {
                    if hidden_node_ids.contains(&node_id) {
                        continue;
                    }
                    if let Some(node) = desc.nodes.iter().find(|n| n.id == node_id) {
                        let uses_layer_param =
                            node.parents.iter().any(|p| param_node_ids.contains(p));
                        if uses_layer_param && !layer.node_ids.iter().any(|id| id.0 == node_id) {
                            layer.node_ids.push(NodeId(node_id));
                        }
                    }
                }

                // 继续查找使用这些计算节点的后续节点
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
                            let uses_layer_node =
                                node.parents.iter().any(|p| current_nodes.contains(p));
                            let uses_other_param = node.parents.iter().any(|p| {
                                desc.nodes.iter().find(|n| n.id == *p).is_some_and(|n| {
                                    matches!(n.node_type, NodeTypeDescriptor::Parameter)
                                        && !param_node_ids.contains(p)
                                })
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
                group.node_ids.retain(|id| !hidden_node_ids.contains(&id.0));
                for &node_id in &model_nodes {
                    if !group.node_ids.iter().any(|id| id.0 == node_id) {
                        group.node_ids.push(NodeId(node_id));
                    }
                }
                group.description = format!("{} nodes", group.node_ids.len());
            }

            // 更新 grouped_node_ids
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
                    if hidden_node_ids.contains(&node.id) || protected_nodes.contains(&node.id) {
                        continue;
                    }
                    let children = children_map.get(&node.id);
                    let all_children_hidden = children.is_some_and(|c| {
                        !c.is_empty() && c.iter().all(|id| hidden_node_ids.contains(id))
                    });
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
        let folded_node_info: HashMap<u64, (usize, usize)> = if group_layers {
            self.layer_groups
                .iter()
                .flat_map(|g| {
                    let layer_min_s = g.min_steps.unwrap_or(0);
                    let layer_max_s = g.max_steps.unwrap_or(0);
                    g.folded_nodes.iter().map(move |(id, node_steps)| {
                        if *node_steps == layer_max_s {
                            (id.0, (layer_min_s, layer_max_s))
                        } else {
                            (id.0, (*node_steps, *node_steps))
                        }
                    })
                })
                .collect()
        } else {
            HashMap::new()
        };

        // 收集 SmartInput 节点的序列长度范围信息（用于变长 RNN 的输入节点显示）
        let input_seq_range_info: HashMap<u64, (usize, usize)> = self
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

        // 辅助闭包：生成单个节点的 DOT 定义
        let generate_node_def = |node: &NodeDescriptor,
                                 dynamic_batch_nodes: &HashSet<u64>,
                                 ever_detached_nodes: &HashSet<u64>,
                                 folded_info: &HashMap<u64, (usize, usize)>,
                                 input_seq_range: &HashMap<u64, (usize, usize)>,
                                 call_count: Option<usize>|
         -> String {
            let (shape, mut style, fillcolor) = Self::dot_node_style(&node.node_type);
            let style_owned: String;

            // 检查是否为折叠节点
            if let Some(&(min_steps, max_steps)) = folded_info.get(&node.id) {
                let fillcolor_owned = Self::darken_color(fillcolor);
                let use_dynamic_batch = dynamic_batch_nodes.contains(&node.id);
                let label = Self::dot_node_label_html_folded_range(
                    node,
                    use_dynamic_batch,
                    min_steps,
                    max_steps,
                    call_count,
                );
                if call_count.is_some() {
                    return format!(
                        "    \"{}\" [label=<{}> shape={} style={} fillcolor=\"{}\" peripheries=2 fontsize=10];\n",
                        node.id, label, shape, style, fillcolor_owned
                    );
                }
                return format!(
                    "    \"{}\" [label=<{}> shape={} style={} fillcolor=\"{}\" fontsize=10];\n",
                    node.id, label, shape, style, fillcolor_owned
                );
            }

            if matches!(node.node_type, NodeTypeDescriptor::SmartInput)
                && ever_detached_nodes.contains(&node.id)
            {
                style_owned = "\"filled,dashed\"".to_string();
                style = &style_owned;
            }

            let use_dynamic_batch = dynamic_batch_nodes.contains(&node.id);
            let is_multi_call_input = input_seq_range.contains_key(&node.id);
            let is_compute_node = !matches!(
                node.node_type,
                NodeTypeDescriptor::Parameter
                    | NodeTypeDescriptor::SmartInput
                    | NodeTypeDescriptor::TargetInput
                    | NodeTypeDescriptor::BasicInput
            );
            let is_target_input = matches!(node.node_type, NodeTypeDescriptor::TargetInput);

            let label = if let Some(&(min_seq, max_seq)) = input_seq_range.get(&node.id) {
                let count = call_count.unwrap_or(1);
                Self::dot_node_label_html_with_seq_range(
                    node,
                    use_dynamic_batch,
                    min_seq,
                    max_seq,
                    count,
                )
            } else if let Some(count) = call_count {
                if is_compute_node || is_target_input {
                    Self::dot_node_label_html_with_call_count(node, use_dynamic_batch, count)
                } else {
                    Self::dot_node_label_html_with_dynamic_batch(node, use_dynamic_batch)
                }
            } else {
                Self::dot_node_label_html_with_dynamic_batch(node, use_dynamic_batch)
            };

            let is_multi_call_compute = call_count.is_some() && is_compute_node;
            let is_multi_call_target = call_count.is_some() && is_target_input;
            let use_double_border =
                is_multi_call_compute || is_multi_call_input || is_multi_call_target;

            if use_double_border {
                format!(
                    "    \"{}\" [label=<{}> shape={} style={} fillcolor=\"{}\" peripheries=2 fontsize=10];\n",
                    node.id, label, shape, style, fillcolor
                )
            } else {
                format!(
                    "    \"{}\" [label=<{}> shape={} style={} fillcolor=\"{}\" fontsize=10];\n",
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
        let nested_layer_indices: HashSet<usize> =
            model_to_layers.iter().flatten().copied().collect();

        // 收集已在嵌套 Layer 中定义的节点 ID
        let mut nodes_in_nested_layers: HashSet<u64> = HashSet::new();

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

                    dot.push_str(&format!(
                        "            label=<<B>{}</B><BR/><FONT POINT-SIZE=\"8\">{}: {}</FONT>>;\n",
                        layer.name, layer.layer_type, layer.description
                    ));

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

                    for node in &desc.nodes {
                        if layer.node_ids.iter().any(|nid| nid.0 == node.id) {
                            nodes_in_nested_layers.insert(node.id);
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

                    dot.push_str("        }\n");
                }

                for node in &desc.nodes {
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

        // 输出不属于任何 Model 的独立 Layer 分组
        if group_layers && !layer_groups.is_empty() {
            for (idx, group) in layer_groups.iter().enumerate() {
                if nested_layer_indices.contains(&idx) {
                    continue;
                }

                let cluster_color = Self::layer_group_color(idx);
                let is_recurrent = group.min_steps.is_some() && group.max_steps.is_some();

                dot.push_str(&format!(
                    "    subgraph cluster_{} {{\n",
                    group.name.replace(['-', '.'], "_")
                ));

                dot.push_str(&format!(
                    "        label=<<B>{}</B><BR/><FONT POINT-SIZE=\"9\">{}: {}</FONT>>;\n",
                    group.name, group.layer_type, group.description
                ));

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
            if hidden_node_ids.contains(&node.id) {
                continue;
            }
            if group_layers && grouped_node_ids.contains(&node.id) {
                continue;
            }
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
        let output_proxies: HashMap<u64, u64> = if group_layers {
            self.layer_groups
                .iter()
                .filter_map(|g| g.output_proxy.map(|(real, repr)| (real.0, repr.0)))
                .collect()
        } else {
            HashMap::new()
        };

        // 收集循环层初始状态节点 ID（这些边会用橙色虚线单独绘制）
        let init_state_node_ids: HashSet<u64> = if group_layers {
            self.recurrent_layer_metas
                .iter()
                .filter_map(|m| m.unroll_infos.last())
                .flat_map(|i| i.init_state_node_ids.iter().map(|id| id.0))
                .collect()
        } else {
            HashSet::new()
        };

        // 收集循环层的输出代理信息及时间步标签
        // 映射：real_output_node_id -> (repr_output_node_id, final_step_label)
        let recurrent_output_labels: HashMap<u64, (u64, String)> = if group_layers {
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
                        format!("t={min_last}~{last_step}")
                    } else {
                        // 固定长度：显示具体时间步
                        format!("t={last_step}")
                    };
                    // 使用第一个 repr_output_node_id 作为代理节点
                    let repr_id = repr_info.repr_output_node_ids.first()?.0;
                    Some((repr_info.real_output_node_id.0, (repr_id, label)))
                })
                .collect()
        } else {
            HashMap::new()
        };

        for node in &desc.nodes {
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
                        dot.push_str(&format!(
                            "    \"{}\" -> \"{}\" [color=\"#E67E22\" label=<{}> fontcolor=\"#E67E22\" fontsize=9];\n",
                            repr_id, node.id, label
                        ));
                    } else if let Some(&repr_id) = output_proxies.get(parent_id) {
                        dot.push_str(&format!("    \"{}\" -> \"{}\";\n", repr_id, node.id));
                    }
                    continue;
                }
                dot.push_str(&format!("    \"{}\" -> \"{}\";\n", parent_id, node.id));
            }
        }

        // 循环层的序列内记忆箭头（橙色虚线）
        if group_layers {
            for meta in &self.recurrent_layer_metas {
                if meta.unroll_infos.is_empty() {
                    continue;
                }

                // 使用最后一次调用作为代表
                let repr_info = meta.unroll_infos.last().unwrap();

                // 计算步数范围
                let min_steps = meta.unroll_infos.iter().map(|i| i.steps).min().unwrap();
                let max_steps = meta.unroll_infos.iter().map(|i| i.steps).max().unwrap();
                let is_var_len = min_steps != max_steps;

                // 计算回流边时间步标签
                // - 回流边：t=0~(max_steps-2)，即这些时刻产生的输出会回流给下一时刻
                let recycle_last = max_steps.saturating_sub(2);
                let recycle_label = if is_var_len {
                    // 变长序列：t=0~(min_last~max_last)
                    let min_recycle_last = min_steps.saturating_sub(2);
                    format!("t=0~({min_recycle_last}~{recycle_last})")
                } else {
                    // 固定长度：t=0~(N-2)
                    format!("t=0~{recycle_last}")
                };

                // 为每对 (初始状态, 输出) 绘制边
                // - RNN/GRU：1 对 (h0, h_1)
                // - LSTM：2 对 (h0, h_1) 和 (c0, c_1)
                for (idx, init_id) in repr_info.init_state_node_ids.iter().enumerate() {
                    // 1. t=0 边：SmartInput → ZerosLike（初始化）
                    if let Some(init_node_desc) = desc.nodes.iter().find(|n| n.id == init_id.0) {
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
                if repr_info.init_state_node_ids.len() > 1 {
                    let node_ids: Vec<String> = repr_info
                        .init_state_node_ids
                        .iter()
                        .map(|id| format!("\"{}\"", id.0))
                        .collect();
                    dot.push_str(&format!("    {{ rank=same; {} }}\n", node_ids.join("; ")));
                }
            }
        }

        // 数据流连接（紫色虚线箭头）
        for node in self.nodes.values() {
            if let NodeType::Input(
                InputVariant::Smart(smart) | InputVariant::RecurrentOutput(smart),
            ) = node.node_type()
            {
                if let Some(target_id) = smart.gradient_target() {
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

    /// 获取模型分组的背景颜色
    fn model_group_color(index: usize) -> &'static str {
        const COLORS: &[&str] = &[
            "#FFECB340", // 浅琥珀
            "#FFCCBC40", // 浅深橙
            "#D1C4E940", // 浅深紫
            "#B2DFDB40", // 浅青绿
            "#F8BBD040", // 浅粉红
            "#DCEDC840", // 浅黄绿
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
    pub fn save_visualization<P: AsRef<Path>>(
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
        Command::new("dot")
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

        let output = Command::new("dot")
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
    fn find_dynamic_batch_nodes(desc: &GraphDescriptor) -> HashSet<u64> {
        let mut dynamic_nodes = HashSet::new();

        // 1. 找出所有 SmartInput 节点
        let router_ids: Vec<u64> = desc
            .nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeTypeDescriptor::SmartInput))
            .map(|n| n.id)
            .collect();

        // 2. 构建邻接表（parent -> children）
        let mut children_map: HashMap<u64, Vec<u64>> = HashMap::new();
        for node in &desc.nodes {
            for &parent_id in &node.parents {
                children_map.entry(parent_id).or_default().push(node.id);
            }
        }

        // 3. BFS 找出所有下游节点
        let mut queue: VecDeque<u64> = router_ids.iter().copied().collect();
        while let Some(node_id) = queue.pop_front() {
            if dynamic_nodes.insert(node_id) {
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
            // 状态节点：圆柱体，浅橙色
            NodeTypeDescriptor::State => ("cylinder", "filled", "#FFE0B2"),
            // Identity 节点：椭圆形，虚线边框，浅紫色
            NodeTypeDescriptor::Identity => ("ellipse", "\"filled,dashed\"", "#E1BEE7"),
            // TargetInput 节点：椭圆形，浅橙色
            NodeTypeDescriptor::TargetInput => ("ellipse", "filled", "#FFE0B2"),
            // SmartInput 节点：椭圆形，浅灰色
            NodeTypeDescriptor::SmartInput => ("ellipse", "filled", "#E0E0E0"),
            // 参数节点：矩形，浅绿色
            NodeTypeDescriptor::Parameter => ("box", "filled", "#E8F5E9"),
            // 损失节点：八边形，浅红色
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
            // ZerosLike：虚线圆角矩形，浅黄色
            NodeTypeDescriptor::ZerosLike => ("box", "\"filled,rounded,dashed\"", "#FFFDE7"),
            // 其他运算节点：圆角矩形，浅黄色
            _ => ("box", "\"filled,rounded\"", "#FFFDE7"),
        }
    }

    /// 生成节点的 HTML 格式标签，支持动态 batch 显示
    fn dot_node_label_html_with_dynamic_batch(
        node: &NodeDescriptor,
        use_dynamic_batch: bool,
    ) -> String {
        let type_name = Self::type_name_for_vis(&node.node_type);

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

        let extra_info = match &node.node_type {
            NodeTypeDescriptor::LeakyReLU { alpha } => Some(format!("α={alpha}")),
            _ => None,
        };

        let mut parts = vec![
            node.name.clone(),
            format!("<B>{}</B>", type_name),
            shape_str,
        ];

        if let Some(params) = node.param_count {
            parts.push(format!("({} params)", Self::format_number_for_vis(params)));
        }

        if let Some(info) = extra_info {
            parts.push(info);
        }

        parts.join("<BR/>")
    }

    /// 生成带多次调用标注的节点标签
    fn dot_node_label_html_with_call_count(
        node: &NodeDescriptor,
        use_dynamic_batch: bool,
        call_count: usize,
    ) -> String {
        let type_name = Self::type_name_for_vis(&node.node_type);

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

        let base_name = Self::strip_index_suffix(&node.name);

        let mut parts = vec![
            format!(
                "{} <FONT COLOR=\"#1565C0\">(×{})</FONT>",
                base_name, call_count
            ),
            format!("<B>{}</B>", type_name),
            shape_str,
        ];

        if let Some(params) = node.param_count {
            parts.push(format!("({} params)", Self::format_number_for_vis(params)));
        }

        parts.join("<BR/>")
    }

    /// 去掉节点名称中的索引后缀
    fn strip_index_suffix(name: &str) -> String {
        if let Some(last_underscore) = name.rfind('_') {
            let suffix = &name[last_underscore + 1..];
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                return name[..last_underscore].to_string();
            }
        }
        name.to_string()
    }

    /// 生成带序列长度范围的节点标签
    fn dot_node_label_html_with_seq_range(
        node: &NodeDescriptor,
        use_dynamic_batch: bool,
        min_seq: usize,
        max_seq: usize,
        call_count: usize,
    ) -> String {
        let type_name = Self::type_name_for_vis(&node.node_type);

        let shape_str = if use_dynamic_batch && node.output_shape.len() >= 2 {
            let shape_parts: Vec<String> = node
                .output_shape
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    if i == 0 {
                        "?".to_string()
                    } else if i == 1 && min_seq != max_seq {
                        format!("{min_seq}-{max_seq}")
                    } else {
                        dim.to_string()
                    }
                })
                .collect();
            format!("[{}]", shape_parts.join(", "))
        } else {
            format!("{:?}", node.output_shape)
        };

        let base_name = Self::strip_index_suffix(&node.name);
        let name_with_count = format!("{base_name} <FONT COLOR=\"#1565C0\">(×{call_count})</FONT>");

        let mut parts = vec![name_with_count, format!("<B>{}</B>", type_name), shape_str];

        if let Some(params) = node.param_count {
            parts.push(format!("({} params)", Self::format_number_for_vis(params)));
        }

        parts.join("<BR/>")
    }

    /// 生成折叠节点的 HTML 格式标签
    fn dot_node_label_html_folded_range(
        node: &NodeDescriptor,
        use_dynamic_batch: bool,
        min_steps: usize,
        max_steps: usize,
        call_count: Option<usize>,
    ) -> String {
        let type_name = Self::type_name_for_vis(&node.node_type);

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

        let base_name = if let Some(pos) = node.name.rfind('_') {
            if node.name[pos + 1..].chars().all(|c| c.is_ascii_digit()) {
                &node.name[..pos]
            } else {
                &node.name
            }
        } else {
            &node.name
        };

        let steps_str = if min_steps == max_steps {
            format!("×{max_steps}")
        } else {
            format!("×{min_steps}-{max_steps}")
        };

        let name_line = if let Some(count) = call_count {
            format!(
                "{base_name} <FONT COLOR=\"#E67E22\">{steps_str}</FONT> <FONT COLOR=\"#1565C0\">(×{count})</FONT>"
            )
        } else {
            format!("{base_name} <FONT COLOR=\"#E67E22\">{steps_str}</FONT>")
        };

        let mut parts = vec![name_line, format!("<B>{}</B>", type_name), shape_str];

        if let Some(params) = node.param_count {
            parts.push(format!("({} params)", Self::format_number_for_vis(params)));
        }

        parts.join("<BR/>")
    }

    /// 将颜色值加深
    fn darken_color(hex_color: &str) -> String {
        match hex_color {
            "#FFFDE7" => "#FFF9C4".to_string(),
            "#FFF3E0" => "#FFE0B2".to_string(),
            "#E8F5E9" => "#C8E6C9".to_string(),
            "#E3F2FD" => "#BBDEFB".to_string(),
            _ => hex_color.to_string(),
        }
    }

    /// 获取节点类型名称（用于可视化）
    fn type_name_for_vis(node_type: &NodeTypeDescriptor) -> &'static str {
        match node_type {
            NodeTypeDescriptor::BasicInput => "BasicInput",
            NodeTypeDescriptor::TargetInput => "TargetInput",
            NodeTypeDescriptor::SmartInput => "SmartInput",
            NodeTypeDescriptor::RecurrentOutput => "RecurrentOut",
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

    /// 格式化数字（用于可视化）
    fn format_number_for_vis(n: usize) -> String {
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

    // ========== 惰性推断循环层分组 ==========

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
            let output_proxy = repr_info
                .repr_output_node_ids
                .first()
                .map(|&repr_id| (repr_info.real_output_node_id, repr_id));

            // 固定长度时 recurrent_steps = Some(steps)，变长时为 None
            let recurrent_steps = if min_steps == max_steps {
                Some(max_steps)
            } else {
                None
            };

            self.layer_groups.push(LayerGroup {
                name: meta.name.clone(),
                layer_type: meta.layer_type.clone(),
                description: meta.description.clone(),
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
    }

    /// 收集需要隐藏的"根节点"
    #[allow(dead_code)]
    pub(in crate::nn::graph) fn hidden_output_nodes(&self) -> Vec<NodeId> {
        self.recurrent_layer_metas
            .iter()
            .flat_map(|meta| {
                meta.unroll_infos
                    .iter()
                    .take(meta.unroll_infos.len().saturating_sub(1))
                    .map(|info| info.real_output_node_id)
            })
            .collect()
    }

    /// 注册模型分组
    pub fn register_model_group(
        &mut self,
        name: &str,
        router_id: NodeId,
        output_id: NodeId,
    ) -> Result<(), GraphError> {
        if self.layer_groups.iter().any(|g| g.name == name) {
            return Ok(());
        }

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

    /// 收集两个节点之间的所有节点
    fn collect_nodes_between(
        &self,
        start_id: NodeId,
        end_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphError> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(end_id);

        while let Some(current) = queue.pop_front() {
            if result.contains(&current) {
                continue;
            }
            result.insert(current);

            if current == start_id {
                continue;
            }

            if let Ok(parents) = self.get_node_parents(current) {
                for parent in parents {
                    queue.push_back(parent);
                }
            }
        }

        Ok(result.into_iter().collect())
    }

    // ========== 节点信息查询 ==========

    pub fn is_node_inited(&self, id: NodeId) -> Result<bool, GraphError> {
        self.get_node(id).map(|n| n.is_inited())
    }

    pub fn get_node_value_shape(&self, id: NodeId) -> Result<Option<&[usize]>, GraphError> {
        Ok(self.get_node(id)?.value().map(|v| v.shape()))
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
        Ok(self.get_node(id)?.value().map(|v| v.size()))
    }
}
