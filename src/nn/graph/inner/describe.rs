/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner describe/summary 相关方法
 */

use super::super::error::GraphError;
use super::GraphInner;
use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use crate::nn::nodes::raw_node::InputVariant;
use crate::nn::nodes::NodeType;
use std::path::Path;

impl GraphInner {
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

    /// 将 `NodeType` 转换为 `NodeTypeDescriptor`
    pub(in crate::nn::graph) fn node_type_to_descriptor(
        &self,
        node_type: &NodeType,
    ) -> NodeTypeDescriptor {
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
            NodeType::Reshape(node) => NodeTypeDescriptor::Reshape {
                target_shape: node.target_shape().to_vec(),
            },
            NodeType::Flatten(_) => NodeTypeDescriptor::Flatten,
            NodeType::Conv2d(node) => NodeTypeDescriptor::Conv2d {
                stride: node.stride(),
                padding: node.padding(),
            },
            NodeType::MaxPool2d(node) => NodeTypeDescriptor::MaxPool2d {
                kernel_size: node.kernel_size(),
                stride: node.stride(),
            },
            NodeType::AvgPool2d(node) => NodeTypeDescriptor::AvgPool2d {
                kernel_size: node.kernel_size(),
                stride: node.stride(),
            },
            NodeType::Select(node) => NodeTypeDescriptor::Select {
                axis: node.axis(),
                index: node.index(),
            },
            NodeType::MSELoss(_) => NodeTypeDescriptor::MSELoss,
            NodeType::SoftmaxCrossEntropy(_) => NodeTypeDescriptor::SoftmaxCrossEntropy,
            NodeType::ZerosLike(_) => NodeTypeDescriptor::ZerosLike,
        }
    }
}
