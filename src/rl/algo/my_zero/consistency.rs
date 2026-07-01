//! MyZero 自监督 consistency loss（SimSiam 负余弦相似度，Chen & He 2021）。
//!
//! 让 dynamics 预测的 `next_latent`（经 projector + predictor 的 online 分支）与
//! `repr(next_obs)`（经 projector 的 target 分支，**stop-gradient**）对齐，给 dynamics 一个
//! 稠密自监督信号（样本效率的关键之一）。
//!
//! 本模块只提供「两个向量的负余弦」纯 `Var` 运算；projector / predictor 头属网络结构，
//! 在 [`super::network`] 实现。

use crate::nn::{GraphError, Var, VarActivationOps, VarReduceOps};

/// SimSiam 负余弦相似度：`-cos(p, stop_grad(z))`（**逐样本**，再对 batch 取均值）。
///
/// - `p`：online 分支输出（dynamics 预测 next_latent 经 projector + predictor）。
/// - `z`：target 分支输出（repr(next_obs) 经 projector），内部对其 `detach()` 做 stop-gradient。
///
/// 两者形状须一致（`[B, dim]`）。沿特征维（axis=1）逐样本算余弦，再对 batch 取均值，
/// 返回**标量** loss `Var`（值域 [-1, 1]，越小越对齐）。
/// B=1 时 `sum_axis(1)` 等价旧的全局 `sum()`、`mean()` 恒等，故逐 bit 一致。
pub fn negative_cosine_similarity(p: &Var, z: &Var) -> Result<Var, GraphError> {
    // stop-gradient：target 分支不回传梯度（SimSiam 防坍缩的关键）
    let z_sg = z.detach();

    let dot = (p * &z_sg).sum_axis(1); // [B,1] 逐样本 Σ_dim p_i z_i
    let p_norm = p.square().sum_axis(1).sqrt(); // [B,1] ‖p‖
    let z_norm = z_sg.square().sum_axis(1).sqrt(); // [B,1] ‖z‖
    let denom = &(&p_norm * &z_norm) + 1e-8_f32; // [B,1] 防除零
    let cos = &dot / &denom; // [B,1]
    Ok((cos * -1.0_f32).mean()) // 标量：对 batch 取均值
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Graph;
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
}
