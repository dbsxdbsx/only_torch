//! `consistency.rs` SimSiam 负余弦测试：同向 / 正交 / 反向三个锚点值

use super::super::consistency::negative_cosine_similarity;
use crate::nn::{Graph, Var};
use crate::tensor::Tensor;

fn input(graph: &Graph, data: &[f32]) -> Var {
    graph
        .input(&Tensor::new(data, &[1, data.len()]))
        .expect("建 input 失败")
}

#[test]
fn identical_vectors_give_neg_one() {
    let graph = Graph::new_with_seed(0);
    let a = input(&graph, &[1.0, 2.0, 3.0, 4.0]);
    let b = input(&graph, &[1.0, 2.0, 3.0, 4.0]);
    let loss = negative_cosine_similarity(&a, &b).unwrap();
    let s = loss.value().unwrap().unwrap().data_as_slice()[0];
    assert!((s + 1.0).abs() < 1e-4, "相同向量负余弦应 ≈ -1，实际 {s}");
}

#[test]
fn orthogonal_vectors_give_zero() {
    let graph = Graph::new_with_seed(0);
    let a = input(&graph, &[1.0, 0.0]);
    let b = input(&graph, &[0.0, 1.0]);
    let loss = negative_cosine_similarity(&a, &b).unwrap();
    let s = loss.value().unwrap().unwrap().data_as_slice()[0];
    assert!(s.abs() < 1e-4, "正交向量负余弦应 ≈ 0，实际 {s}");
}

#[test]
fn opposite_vectors_give_pos_one() {
    let graph = Graph::new_with_seed(0);
    let a = input(&graph, &[1.0, 2.0, 3.0]);
    let b = input(&graph, &[-1.0, -2.0, -3.0]);
    let loss = negative_cosine_similarity(&a, &b).unwrap();
    let s = loss.value().unwrap().unwrap().data_as_slice()[0];
    assert!((s - 1.0).abs() < 1e-4, "反向向量负余弦应 ≈ +1，实际 {s}");
}
