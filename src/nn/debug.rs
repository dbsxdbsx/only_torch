/*
 * @Author       : 老董
 * @Date         : 2026-02-01
 * @Description  : 调试工具模块
 *
 * 提供节点类型枚举、Graph API 检查等调试功能。
 * 使用 strum 自动从 NodeType 枚举获取变体信息。
 */

use super::nodes::raw_node::NodeType;
use std::fmt;
use strum::{EnumCount, VariantNames};

/// 节点类型信息
#[derive(Debug, Clone)]
pub struct NodeTypeInfo {
    /// 节点类型名称（从枚举自动获取）
    pub name: &'static str,
    /// 分类（输入/参数/运算/激活/损失等）
    pub category: &'static str,
    /// 简要描述
    pub description: &'static str,
    /// 对应的 Var 方法（如果有）
    pub var_method: Option<&'static str>,
}

impl fmt::Display for NodeTypeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<20} [{:<8}] {}",
            self.name, self.category, self.description
        )?;
        if let Some(method) = self.var_method {
            write!(f, " (Var::{method})")?;
        }
        Ok(())
    }
}

/// 节点类别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeCategory {
    Input,
    Parameter,
    State,
    Arithmetic,
    MatrixConv,
    Shape,
    Reduce,
    Activation,
    Loss,
    Utility,
}

impl NodeCategory {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Input => "输入",
            Self::Parameter => "参数",
            Self::State => "状态",
            Self::Arithmetic => "算术",
            Self::MatrixConv => "矩阵/卷积",
            Self::Shape => "形状",
            Self::Reduce => "归约",
            Self::Activation => "激活",
            Self::Loss => "损失",
            Self::Utility => "辅助",
        }
    }
}

/// 获取节点的元数据（类别、描述、Var 方法）
///
/// 当添加新的 NodeType 变体时，必须在这里添加对应的元数据，
/// 否则 `describe_registered_node_types()` 会返回 "未知" 信息，
/// 测试也会失败提醒你。
fn get_node_metadata(name: &str) -> (NodeCategory, &'static str, Option<&'static str>) {
    match name {
        // ==================== 输入/参数/状态 ====================
        "Input" => (NodeCategory::Input, "外部数据输入（Data/Target/Smart/RecurrentOutput）", None),
        "Parameter" => (NodeCategory::Parameter, "可学习参数（weight/bias）", None),
        "State" => (NodeCategory::State, "时间状态节点（RNN 隐藏状态）", None),

        // ==================== 算术运算 ====================
        "Add" => (NodeCategory::Arithmetic, "逐元素加法（支持广播）", Some("+ 运算符")),
        "Subtract" => (NodeCategory::Arithmetic, "逐元素减法（支持广播）", Some("- 运算符")),
        "Multiply" => (NodeCategory::Arithmetic, "逐元素乘法（支持广播）", Some("* 运算符")),
        "Divide" => (NodeCategory::Arithmetic, "逐元素除法（支持广播）", Some("/ 运算符")),

        // ==================== 矩阵/卷积运算 ====================
        "MatMul" => (NodeCategory::MatrixConv, "矩阵乘法", Some("matmul()")),
        "Conv2d" => (NodeCategory::MatrixConv, "2D 卷积", None),
        "MaxPool2d" => (NodeCategory::MatrixConv, "2D 最大池化", None),
        "AvgPool2d" => (NodeCategory::MatrixConv, "2D 平均池化", None),

        // ==================== 形状变换 ====================
        "Reshape" => (NodeCategory::Shape, "张量变形", Some("reshape()")),
        "Flatten" => (NodeCategory::Shape, "展平（保留 batch 维）", Some("flatten()")),
        "Select" => (NodeCategory::Shape, "固定索引选择（RNN 时间步）", Some("select()")),
        "Gather" => (NodeCategory::Shape, "动态索引收集（强化学习）", Some("gather()")),
        "Stack" => (NodeCategory::Shape, "张量堆叠/拼接", Some("Var::stack()")),

        // ==================== 比较/归约 ====================
        "Maximum" => (NodeCategory::Reduce, "逐元素取最大值", None),
        "Minimum" => (NodeCategory::Reduce, "逐元素取最小值", None),
        "Amax" => (NodeCategory::Reduce, "沿轴取最大值", None),
        "Amin" => (NodeCategory::Reduce, "沿轴取最小值", None),
        "Sum" => (NodeCategory::Reduce, "归约求和", Some("sum() / sum_axis()")),
        "Mean" => (NodeCategory::Reduce, "归约求均值", Some("mean() / mean_axis()")),

        // ==================== 激活函数 ====================
        "Sigmoid" => (NodeCategory::Activation, "Sigmoid 激活", Some("sigmoid()")),
        "Tanh" => (NodeCategory::Activation, "Tanh 激活", Some("tanh()")),
        "LeakyReLU" => (NodeCategory::Activation, "LeakyReLU 激活（含 ReLU）", Some("leaky_relu() / relu()")),
        "Softmax" => (NodeCategory::Activation, "Softmax 归一化", Some("softmax()")),
        "LogSoftmax" => (NodeCategory::Activation, "数值稳定的 log(softmax)", Some("log_softmax()")),
        "SoftPlus" => (NodeCategory::Activation, "SoftPlus 激活（平滑 ReLU）", Some("softplus()")),
        "Step" => (NodeCategory::Activation, "阶跃函数", Some("step()")),
        "Sign" => (NodeCategory::Activation, "符号函数", Some("sign()")),
        "Abs" => (NodeCategory::Activation, "绝对值", Some("abs()")),
        "Ln" => (NodeCategory::Activation, "自然对数", Some("ln()")),

        // ==================== 损失函数 ====================
        "MSE" => (NodeCategory::Loss, "均方误差损失", Some("mse_loss()")),
        "MAE" => (NodeCategory::Loss, "平均绝对误差损失", Some("mae_loss()")),
        "BCE" => (NodeCategory::Loss, "二元交叉熵损失", Some("bce_loss()")),
        "Huber" => (NodeCategory::Loss, "Huber 损失（强化学习）", Some("huber_loss()")),
        "SoftmaxCrossEntropy" => (NodeCategory::Loss, "Softmax + 交叉熵损失", Some("cross_entropy()")),

        // ==================== 辅助节点 ====================
        "Identity" => (NodeCategory::Utility, "恒等映射（用于 detach）", Some("detach_node()")),
        "Dropout" => (NodeCategory::Utility, "随机丢弃（正则化）", Some("dropout()")),
        "ZerosLike" => (NodeCategory::Utility, "动态零张量（RNN 初始状态）", None),

        // ==================== 未知节点（新增节点时会触发）====================
        _ => (NodeCategory::Utility, "⚠️ 未添加描述，请更新 debug.rs", None),
    }
}

/// 获取 NodeType 枚举的变体数量（编译时常量）
pub const fn node_type_count() -> usize {
    NodeType::COUNT
}

/// 获取 NodeType 枚举的所有变体名称（编译时常量）
pub fn node_type_variant_names() -> &'static [&'static str] {
    NodeType::VARIANTS
}

/// 获取所有已注册的节点类型
///
/// 自动从 `NodeType` 枚举获取变体列表，无需手动维护。
/// 当添加新节点时，只需在 `get_node_metadata()` 中添加描述即可。
///
/// # 示例
/// ```ignore
/// use only_torch::nn::debug::describe_registered_node_types;
///
/// let nodes = describe_registered_node_types();
/// println!("已注册节点类型（共 {} 个）：", nodes.len());
/// for node in &nodes {
///     println!("  {}", node);
/// }
/// ```
pub fn describe_registered_node_types() -> Vec<NodeTypeInfo> {
    NodeType::VARIANTS
        .iter()
        .map(|&name| {
            let (category, description, var_method) = get_node_metadata(name);
            NodeTypeInfo {
                name,
                category: category.as_str(),
                description,
                var_method,
            }
        })
        .collect()
}

/// 打印所有已注册的节点类型（调试用）
///
/// 按类别分组显示所有节点类型，便于检查和对比。
pub fn print_registered_node_types() {
    let nodes = describe_registered_node_types();

    println!();
    println!(
        "========== 已注册节点类型（共 {} 个，来自 NodeType 枚举）==========",
        nodes.len()
    );
    println!();

    // 按类别分组
    let categories = [
        ("输入", NodeCategory::Input.as_str()),
        ("参数", NodeCategory::Parameter.as_str()),
        ("状态", NodeCategory::State.as_str()),
        ("算术", NodeCategory::Arithmetic.as_str()),
        ("矩阵/卷积", NodeCategory::MatrixConv.as_str()),
        ("形状", NodeCategory::Shape.as_str()),
        ("归约", NodeCategory::Reduce.as_str()),
        ("激活", NodeCategory::Activation.as_str()),
        ("损失", NodeCategory::Loss.as_str()),
        ("辅助", NodeCategory::Utility.as_str()),
    ];

    for (cat_name, cat_str) in categories {
        let cat_nodes: Vec<_> = nodes.iter().filter(|n| n.category == cat_str).collect();
        if cat_nodes.is_empty() {
            continue;
        }

        println!("[{cat_name}]（{} 个）", cat_nodes.len());
        for node in cat_nodes {
            let var_info = node
                .var_method
                .map_or(String::new(), |m| format!(" → Var::{m}"));
            println!(
                "  • {:<24} {:<28}{var_info}",
                node.name, node.description
            );
        }
        println!();
    }

    println!("==============================================");
}

/// 获取按类别分组的节点统计
pub fn get_node_type_summary() -> Vec<(&'static str, usize)> {
    let nodes = describe_registered_node_types();
    let categories = [
        "输入", "参数", "状态", "算术", "矩阵/卷积", "形状", "归约", "激活", "损失", "辅助",
    ];

    categories
        .iter()
        .map(|&cat| {
            let count = nodes.iter().filter(|n| n.category == cat).count();
            (cat, count)
        })
        .filter(|(_, count)| *count > 0)
        .collect()
}

/// 检查是否所有节点都有元数据描述
///
/// 返回未添加描述的节点列表。如果返回空列表，说明所有节点都已正确配置。
pub fn check_missing_metadata() -> Vec<&'static str> {
    NodeType::VARIANTS
        .iter()
        .filter(|&&name| {
            let (_, desc, _) = get_node_metadata(name);
            desc.contains("未添加描述")
        })
        .copied()
        .collect()
}
