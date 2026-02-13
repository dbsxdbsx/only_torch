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
    Var::snapshot_to_dot(&snapshot, &[], &[])
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
