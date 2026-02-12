/*
 * Debug 模块单元测试
 *
 * 测试 nn::debug 模块的功能，包括：
 * - 节点类型枚举和元数据
 * - 调试工具函数
 *
 * 注意：元数据由 `define_node_types!` 宏统一生成，编译时保证同步，
 * 无需运行时检查。
 */

use crate::nn::debug::*;

/// 测试 describe_registered_node_types 返回完整列表
#[test]
fn test_describe_registered_node_types() {
    let nodes = describe_registered_node_types();

    // 验证返回非空列表
    assert!(!nodes.is_empty(), "节点列表不应为空");

    // 验证数量与 strum 报告一致
    assert_eq!(
        nodes.len(),
        node_type_count(),
        "节点数量应与 NodeType::COUNT 一致"
    );

    // 验证每个节点都有名称和分类
    for node in &nodes {
        assert!(!node.name.is_empty(), "节点名称不应为空");
        assert!(!node.category.is_empty(), "节点分类不应为空");
    }
}

/// 测试 print_registered_node_types 不会 panic
#[test]
fn test_print_registered_node_types() {
    // 只验证函数能正常执行，不 panic
    print_registered_node_types();
}

/// 测试 get_node_type_summary 返回正确的分类统计
#[test]
fn test_get_node_type_summary() {
    let summary = get_node_type_summary();

    // 验证返回非空
    assert!(!summary.is_empty(), "统计信息不应为空");

    // 验证总数等于节点数
    let total: usize = summary.iter().map(|(_, count)| count).sum();
    assert_eq!(total, node_type_count(), "分类统计总数应等于节点总数");
}

/// 测试 NodeTypeInfo 的 Display 实现
#[test]
fn test_node_type_info_display() {
    let nodes = describe_registered_node_types();
    let first = &nodes[0];

    let display = format!("{}", first);
    assert!(display.contains(first.name), "Display 应包含节点名称");
    assert!(display.contains(first.category), "Display 应包含分类");
}

/// 测试变体名称与元数据一一对应
#[test]
fn test_variant_names_match_metadata() {
    let variants = node_type_variant_names();
    let nodes = describe_registered_node_types();

    for (i, (&variant_name, node_info)) in variants.iter().zip(nodes.iter()).enumerate() {
        assert_eq!(
            variant_name, node_info.name,
            "索引 {} 处的变体名 '{}' 与节点信息名 '{}' 不匹配",
            i, variant_name, node_info.name
        );
    }
}
