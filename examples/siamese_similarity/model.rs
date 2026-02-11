/*
 * @Author       : 老董
 * @Date         : 2026-01-28
 * @Description  : Siamese 网络模型（共享编码器验证）
 *
 * 核心特点：两个输入共享同一个编码器，验证：
 * - 共享参数的正确复用
 * - 梯度正确累积到共享参数
 */

use only_torch::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// Siamese 相似度网络
///
/// 结构：
/// ```text
/// 输入1 ─> Encoder ─> 特征1 ─┐
///            ↑                ├─> Concat ─> Classifier ─> 相似度
///          共享参数           │
///            ↓                │
/// 输入2 ─> Encoder ─> 特征2 ─┘
/// ```
pub struct SiameseSimilarity {
    /// 共享编码器（两个输入复用同一个）
    encoder: Linear,
    /// 分类器（判断是否相似）
    classifier: Linear,
}

impl SiameseSimilarity {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        // 编码器：1 -> 8（共享参数）
        let encoder = Linear::new(graph, 1, 8, true, "shared_encoder")?;
        // 分类器：16 -> 1（两个 8 维特征拼接后输出相似度）
        let classifier = Linear::new(graph, 16, 1, true, "classifier")?;

        Ok(Self {
            encoder,
            classifier,
        })
    }

    /// 双输入 forward（验证共享编码器）
    pub fn forward(&self, x1: &Tensor, x2: &Tensor) -> Result<Var, GraphError> {
        // 两个输入经过**同一个**编码器（共享参数）
        let feat1 = self.encoder.forward(x1).relu();
        let feat2 = self.encoder.forward(x2).relu();

        // 拼接两个特征向量
        let combined = Var::stack(&[&feat1, &feat2], 1, false)?;

        // 分类器输出相似度（sigmoid 归一化到 0~1）
        Ok(self.classifier.forward(&combined).sigmoid())
    }
}

impl Module for SiameseSimilarity {
    fn parameters(&self) -> Vec<Var> {
        let mut params = self.encoder.parameters();
        params.extend(self.classifier.parameters());
        params
    }
}
