//! 自监督 reconstruction loss（Scholz et al. 2021，
//! *Improving Model-Based Reinforcement Learning with Internal State Representations through Self-Supervision*，
//! arXiv:2102.05599，\(l_g\)）。
//!
//! 从 unroll 各步 latent 解码重建真实观测（MSE），梯度回 dynamics / representation；
//! **不参与 MCTS**，训完可丢弃解码器。与 [`super::consistency`] 互补：
//! consistency 管步间 latent 格式一致，reconstruction 管 latent 保留观测信息。

#[cfg(test)]
mod tests {
    use super::super::loss::RECONSTRUCTION_LOSS_COEF;
    use crate::nn::Graph;
    use crate::nn::VarLossOps;
    use crate::tensor::Tensor;

    #[test]
    fn reconstruction_coef_matches_paper_default() {
        assert!(
            (RECONSTRUCTION_LOSS_COEF - 1.0).abs() < 1e-6,
            "论文 lg 默认权重 1.0"
        );
    }

    #[test]
    fn mse_reconstruction_loss_finite() {
        let graph = Graph::new_with_seed(0);
        let pred = graph
            .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]))
            .unwrap();
        let target = Tensor::new(&[1.1, 1.9, 3.2, 3.8], &[1, 4]);
        let loss = pred.mse_loss(&target).unwrap();
        let v = loss.value().unwrap().unwrap().data_as_slice()[0];
        assert!(v.is_finite() && v > 0.0);
    }
}
