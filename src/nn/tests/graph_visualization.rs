/*
 * @Author       : 老董
 * @Date         : 2026-02-13
 * @Description  : 计算图可视化测试
 *
 * 测试 snapshot_to_dot 生成的 DOT 输出中 cluster 的正确性，
 * 特别是分布 cluster 的生成和模型内嵌套。
 */

use crate::nn::distributions::Categorical;
use crate::nn::graph::Graph;
use crate::nn::var::Var;
use crate::nn::{VarActivationOps, VarReduceOps};
use crate::tensor::Tensor;

/// 辅助：从 snapshot 生成 DOT 字符串
fn build_dot_from_named_outputs(named_outputs: &[(&str, &Var)]) -> String {
    let snapshot = Var::build_snapshot(named_outputs);
    Var::snapshot_to_dot(&snapshot, &[])
}

// ==================== DOT 输出 cluster 存在 ====================

/// snapshot_to_dot 输出应包含分布 cluster 子图
#[test]
fn test_dot_contains_distribution_cluster() {
    let graph = Graph::new();
    let logits = graph
        .input(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))
        .unwrap();

    let dist = Categorical::new(logits);
    let entropy = dist.entropy();

    let dot = build_dot_from_named_outputs(&[("Loss", &entropy)]);

    // DOT 应包含分布 cluster
    assert!(
        dot.contains("subgraph cluster_"),
        "DOT 应包含 cluster 子图"
    );
    assert!(
        dot.contains("Categorical"),
        "DOT 应包含 Categorical 标签"
    );
    // 应使用虚线边框
    assert!(
        dot.contains("dashed"),
        "分布 cluster 应使用虚线边框"
    );
}

/// 分布 cluster 内应包含 softmax 和 log_softmax 节点
#[test]
fn test_dot_cluster_contains_expected_nodes() {
    let graph = Graph::new();
    let logits = graph
        .input(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))
        .unwrap();

    let dist = Categorical::new(logits);
    let entropy = dist.entropy();

    let dot = build_dot_from_named_outputs(&[("Loss", &entropy)]);

    // 在 cluster 部分应能找到 Softmax 和 LogSoftmax 节点
    // 查找 cluster 子图块
    let cluster_start = dot.find("subgraph cluster_").expect("应有 cluster");
    let cluster_section = &dot[cluster_start..];

    assert!(
        cluster_section.contains("Softmax"),
        "cluster 内应含 Softmax 节点"
    );
    assert!(
        cluster_section.contains("LogSoftmax"),
        "cluster 内应含 LogSoftmax 节点"
    );
}

// ==================== 无分布时不产生 cluster ====================

/// 不使用分布时，DOT 输出不应包含分组 cluster
#[test]
fn test_dot_no_cluster_without_distribution() {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))
        .unwrap();

    // 纯手写操作，不使用分布 API
    let softmax = input.softmax();
    let loss = softmax.sum_axis(1).mean();

    let dot = build_dot_from_named_outputs(&[("Loss", &loss)]);

    // 不应包含分组 cluster（只有 group_ 前缀的 cluster 是分布的）
    assert!(
        !dot.contains("cluster_group_"),
        "无分布时不应有分组 cluster"
    );
}

// ==================== ONNX provenance 渲染（origin tooltip）====================

/// 非 ONNX 路径下的节点 origin 为空 Vec → DOT 输出不应含 "origin:" 标签
#[test]
fn test_dot_no_origin_tooltip_for_non_onnx_path() {
    let graph = Graph::new();
    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))
        .unwrap();
    let softmax = input.softmax();
    let loss = softmax.sum_axis(1).mean();

    let dot = build_dot_from_named_outputs(&[("Loss", &loss)]);

    // 演化 / Layer / 直接构图 路径下,所有节点的 origin_onnx_nodes 都是空 Vec,
    // DOT 输出不应出现 origin tooltip 或小字标签 → 与改造前行为完全一致
    assert!(
        !dot.contains("origin:"),
        "非 ONNX 路径不应渲染 origin: 标签,实际 DOT 含 origin"
    );
    assert!(
        !dot.contains("tooltip=\"origin:"),
        "非 ONNX 路径不应有 tooltip 属性"
    );
}

/// ONNX 路径节点的 origin 应在 DOT 中渲染（label 小字 + tooltip 属性）
#[test]
fn test_dot_renders_origin_tooltip_for_onnx_node() {
    use crate::nn::graph::Graph;

    // 手工构造一个简单 GraphDescriptor:1 个 BasicInput + 1 个 Sigmoid,
    // 给 Sigmoid 显式注入 origin_onnx_nodes=["MyOrigConv_42"]
    use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};

    let mut desc = GraphDescriptor::new("provenance_test");
    desc.add_node(
        NodeDescriptor::new(
            1,
            "input",
            NodeTypeDescriptor::BasicInput,
            vec![1, 3],
            None,
            vec![],
        )
        .with_origin_onnx_nodes(vec!["<input:input>".to_string()]),
    );
    desc.add_node(
        NodeDescriptor::new(
            2,
            "sigmoid_out",
            NodeTypeDescriptor::Sigmoid,
            vec![1, 3],
            None,
            vec![1],
        )
        .with_origin_onnx_nodes(vec!["MyOrigConv_42".to_string()]),
    );

    let result = Graph::from_descriptor(&desc).expect("rebuild");
    // 设置输入值以便 build_snapshot 可正常运作
    result.inputs[0]
        .1
        .set_value(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))
        .ok();

    // outputs[0] 是 sigmoid_out
    let sig = &result.outputs[0];
    let dot = build_dot_from_named_outputs(&[("Out", sig)]);

    assert!(
        dot.contains("origin: MyOrigConv_42"),
        "ONNX 路径节点应在 DOT label 中渲染 origin: 小字,实际 DOT:\n{dot}"
    );
    assert!(
        dot.contains("tooltip=\"origin: MyOrigConv_42\""),
        "ONNX 路径节点应有 tooltip=\"origin: ...\" 属性"
    );
}

/// origin 项数 > 3 时应显示 "+N more" 摘要,tooltip 仍含完整列表
#[test]
fn test_dot_origin_summarizes_when_many() {
    use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
    use crate::nn::graph::Graph;

    let mut desc = GraphDescriptor::new("many_origins_test");
    desc.add_node(
        NodeDescriptor::new(
            1,
            "input",
            NodeTypeDescriptor::BasicInput,
            vec![1, 3],
            None,
            vec![],
        )
        .with_origin_onnx_nodes(vec!["<input:input>".to_string()]),
    );
    let many = vec![
        "Shape_1".to_string(),
        "Gather_2".to_string(),
        "Concat_3".to_string(),
        "Reshape_4".to_string(),
        "Cast_5".to_string(),
    ];
    desc.add_node(
        NodeDescriptor::new(2, "out", NodeTypeDescriptor::Sigmoid, vec![1, 3], None, vec![1])
            .with_origin_onnx_nodes(many.clone()),
    );

    let result = Graph::from_descriptor(&desc).expect("rebuild");
    result.inputs[0]
        .1
        .set_value(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))
        .ok();

    let dot = build_dot_from_named_outputs(&[("Out", &result.outputs[0])]);

    // label 应该是 "Shape_1, Gather_2, Concat_3, +2 more"
    assert!(
        dot.contains("origin: Shape_1, Gather_2, Concat_3, +2 more"),
        "label 应显示前 3 项 + N more, 实际 DOT 段:\n{dot}"
    );
    // tooltip 应含完整 5 项
    assert!(
        dot.contains("tooltip=\"origin: Shape_1, Gather_2, Concat_3, Reshape_4, Cast_5\""),
        "tooltip 应含完整 origin 列表"
    );
}
