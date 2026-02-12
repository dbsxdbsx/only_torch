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
/// 输入1 ─> Enc1 ─> ReLU ─> Enc2 ─> ReLU ─> 特征1 ─┐
///            ↑                ↑                     ├─> Concat ─> Cls1(ReLU) ─> Cls2(Sigmoid)
///          共享参数         共享参数                │
///            ↓                ↓                     │
/// 输入2 ─> Enc1 ─> ReLU ─> Enc2 ─> ReLU ─> 特征2 ─┘
/// ```
pub struct SiameseSimilarity {
    /// 共享编码器第 1 层
    encoder1: Linear,
    /// 共享编码器第 2 层
    encoder2: Linear,
    /// 分类器第 1 层
    classifier1: Linear,
    /// 分类器第 2 层
    classifier2: Linear,
}

impl SiameseSimilarity {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        // 共享编码器：1 -> 16 -> 8（两层，容量更大）
        let encoder1 = Linear::new(graph, 1, 16, true, "shared_enc1")?;
        let encoder2 = Linear::new(graph, 16, 8, true, "shared_enc2")?;
        // 分类器：16 -> 8 -> 1（两层，更强的非线性拟合）
        let classifier1 = Linear::new(graph, 16, 8, true, "classifier1")?;
        let classifier2 = Linear::new(graph, 8, 1, true, "classifier2")?;

        Ok(Self {
            encoder1,
            encoder2,
            classifier1,
            classifier2,
        })
    }

    /// 双输入 forward（验证共享编码器）
    pub fn forward(&self, x1: &Tensor, x2: &Tensor) -> Result<Var, GraphError> {
        // 两个输入经过**同一个**双层编码器（共享参数）
        let feat1 = self
            .encoder2
            .forward(&self.encoder1.forward(x1).relu())
            .relu();
        let feat2 = self
            .encoder2
            .forward(&self.encoder1.forward(x2).relu())
            .relu();

        // 拼接两个特征向量
        let combined = Var::stack(&[&feat1, &feat2], 1, false)?;

        // 双层分类器输出相似度
        Ok(self
            .classifier2
            .forward(&self.classifier1.forward(&combined).relu())
            .sigmoid())
    }
}

impl Module for SiameseSimilarity {
    fn parameters(&self) -> Vec<Var> {
        let mut params = self.encoder1.parameters();
        params.extend(self.encoder2.parameters());
        params.extend(self.classifier1.parameters());
        params.extend(self.classifier2.parameters());
        params
    }
}
