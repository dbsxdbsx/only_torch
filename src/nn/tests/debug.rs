/*
 * Debug 模块测试
 *
 * 测试节点类型枚举和调试工具功能。
 * 使用 strum 自动从 NodeType 获取变体信息。
 */

use crate::nn::debug::{
    check_missing_metadata, describe_registered_node_types, get_node_type_summary,
    node_type_count, node_type_variant_names, print_registered_node_types,
};

/// 关键测试：确保所有 NodeType 变体都有元数据描述
///
/// 当添加新节点时，如果忘记在 `get_node_metadata()` 中添加描述，
/// 这个测试会失败并列出缺失的节点。
#[test]
fn test_all_nodes_have_metadata() {
    let missing = check_missing_metadata();

    assert!(
        missing.is_empty(),
        "\n\n⚠️  以下 NodeType 变体缺少元数据描述，请更新 src/nn/debug.rs 的 get_node_metadata()：\n  - {}\n",
        missing.join("\n  - ")
    );
}

/// 验证 strum 自动获取的变体数量与预期一致
#[test]
fn test_node_type_count_matches() {
    let count = node_type_count();
    let names = node_type_variant_names();
    let nodes = describe_registered_node_types();

    assert_eq!(count, names.len(), "COUNT 与 VARIANTS 长度应一致");
    assert_eq!(count, nodes.len(), "节点信息数量应与枚举变体数一致");

    // 当前应该是 40 个变体
    assert!(
        count >= 40,
        "NodeType 变体数量应该 >= 40，实际: {}",
        count
    );
}

#[test]
fn test_describe_registered_node_types() {
    let nodes = describe_registered_node_types();

    // 验证每个节点都有必要字段
    for node in &nodes {
        assert!(!node.name.is_empty(), "节点名称不能为空");
        assert!(!node.category.is_empty(), "节点类别不能为空");
        assert!(!node.description.is_empty(), "节点描述不能为空");
    }
}

#[test]
fn test_print_registered_node_types() {
    // 简单测试不会 panic
    print_registered_node_types();
}

#[test]
fn test_get_node_type_summary() {
    let summary = get_node_type_summary();

    // 验证有多个类别
    assert!(summary.len() >= 5, "应该有多个类别");

    // 验证总数与枚举变体数一致
    let total: usize = summary.iter().map(|(_, count)| count).sum();
    assert_eq!(
        total,
        node_type_count(),
        "分类统计总数应与枚举变体数一致"
    );
}

#[test]
fn test_node_type_info_display() {
    let nodes = describe_registered_node_types();

    // 验证 Display trait 实现正常
    for node in &nodes {
        let display = format!("{}", node);
        assert!(!display.is_empty(), "Display 输出不应为空");
        assert!(display.contains(node.name), "Display 应包含节点名称");
    }
}

#[test]
fn test_all_categories_covered() {
    let summary = get_node_type_summary();

    // 验证关键类别都有节点
    let category_names: Vec<&str> = summary.iter().map(|(name, _)| *name).collect();

    assert!(category_names.contains(&"输入"), "应该有输入类别");
    assert!(category_names.contains(&"激活"), "应该有激活类别");
    assert!(category_names.contains(&"损失"), "应该有损失类别");
}
