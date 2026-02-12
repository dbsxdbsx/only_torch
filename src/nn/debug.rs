/*
 * @Author       : 老董
 * @Date         : 2026-02-01
 * @Description  : 调试工具模块
 *
 * 提供节点类型枚举、Graph API 检查等调试功能。
 *
 * ## 元数据来源
 *
 * 节点元数据（类别、描述、Var 方法）由 `raw_node/mod.rs` 中的
 * `define_node_types!` 宏统一定义，本模块直接导入使用。
 *
 * ## 添加新节点时
 *
 * 只需在 `src/nn/nodes/raw_node/mod.rs` 的 `define_node_types!` 宏调用中
 * 添加一项，枚举和元数据自动同步，本文件无需修改。
 */

use super::nodes::raw_node::{NODE_METADATA, NodeType};
use std::fmt;
use strum::{EnumCount, VariantNames};

/// 节点类型信息
#[derive(Debug, Clone)]
pub struct NodeTypeInfo {
    /// 节点类型名称（从枚举自动获取）
    pub name: &'static str,
    /// 分类
    pub category: &'static str,
    /// 简要描述
    pub description: &'static str,
    /// 对应的 Var 方法
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
/// 自动从 `NodeType` 枚举获取变体名称，从 `NODE_METADATA` 获取描述信息。
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
        .enumerate()
        .map(|(i, &name)| {
            let (category, description, var_method) = NODE_METADATA.get(i).copied().unwrap_or((
                "未分类",
                "（索引越界，请检查 NODE_METADATA）",
                None,
            ));
            NodeTypeInfo {
                name,
                category,
                description,
                var_method,
            }
        })
        .collect()
}

/// 获取所有唯一分类（按首次出现顺序）
///
/// 动态从 `NODE_METADATA` 提取，无需手动维护分类列表。
fn get_unique_categories() -> Vec<&'static str> {
    let mut categories = Vec::new();
    for &(cat, _, _) in NODE_METADATA {
        if !categories.contains(&cat) {
            categories.push(cat);
        }
    }
    categories
}

/// 打印所有已注册的节点类型（调试用）
///
/// 按类别分组显示所有节点类型，便于检查和对比。
/// 分类顺序由 `define_node_types!` 宏中的定义顺序决定。
pub fn print_registered_node_types() {
    let nodes = describe_registered_node_types();

    println!();
    println!(
        "========== 已注册节点类型（共 {} 个，来自 NodeType 枚举）==========",
        nodes.len()
    );
    println!();

    // 动态获取分类（按定义顺序）
    for cat_name in get_unique_categories() {
        let cat_nodes: Vec<_> = nodes.iter().filter(|n| n.category == cat_name).collect();
        if cat_nodes.is_empty() {
            continue;
        }

        println!("[{cat_name}]（{} 个）", cat_nodes.len());
        for node in cat_nodes {
            let var_info = node
                .var_method
                .map_or(String::new(), |m| format!(" → Var::{m}"));
            println!("  • {:<24} {:<28}{var_info}", node.name, node.description);
        }
        println!();
    }

    println!("==============================================");
}

/// 获取按类别分组的节点统计
pub fn get_node_type_summary() -> Vec<(&'static str, usize)> {
    let nodes = describe_registered_node_types();

    // 动态获取分类（按定义顺序）
    get_unique_categories()
        .into_iter()
        .map(|cat| {
            let count = nodes.iter().filter(|n| n.category == cat).count();
            (cat, count)
        })
        .filter(|(_, count)| *count > 0)
        .collect()
}
