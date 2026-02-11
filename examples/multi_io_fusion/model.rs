/*
 * @Author       : 老董
 * @Date         : 2026-01-28
 * @Description  : 多输入多输出融合模型
 *
 * 展示双输入 + 多输出的完整用法：
 * - 两个不同形状的输入
 * - 两个不同类型的输出（分类 + 回归）
 */

use only_torch::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// 多输入多输出融合模型
///
/// 结构：
/// ```text
/// 输入A [batch, 4] ─> 编码器A(8) ─┐
///                                 ├─> 融合层(16) ─┬─> 分类头(2) → CrossEntropy
/// 输入B [batch, 8] ─> 编码器B(8) ─┘               │
///                                                └─> 回归头(1) → MSE
/// ```
pub struct MultiIOFusion {
    /// 编码器 A（处理输入 A）
    encoder_a: Linear,
    /// 编码器 B（处理输入 B）
    encoder_b: Linear,
    /// 融合层
    fusion: Linear,
    /// 分类头
    cls_head: Linear,
    /// 回归头
    reg_head: Linear,
    /// 计算图
    graph: Graph,
}

impl MultiIOFusion {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        // 编码器 A：4 -> 8
        let encoder_a = Linear::new(graph, 4, 8, true, "encoder_a")?;
        // 编码器 B：8 -> 8
        let encoder_b = Linear::new(graph, 8, 8, true, "encoder_b")?;
        // 融合层：16 -> 16（拼接后的特征）
        let fusion = Linear::new(graph, 16, 16, true, "fusion")?;
        // 分类头：16 -> 2
        let cls_head = Linear::new(graph, 16, 2, true, "cls_head")?;
        // 回归头：16 -> 1
        let reg_head = Linear::new(graph, 16, 1, true, "reg_head")?;

        Ok(Self {
            encoder_a,
            encoder_b,
            fusion,
            cls_head,
            reg_head,
            graph: graph.clone(),
        })
    }

    /// 前向传播（多输入 + 多输出）
    ///
    /// # 参数
    /// - `input_a`: 输入 A `[batch, 4]`
    /// - `input_b`: 输入 B `[batch, 8]`
    ///
    /// # 返回
    /// - `cls_logits`: 分类 logits `[batch, 2]`
    /// - `reg_pred`: 回归预测 `[batch, 1]`
    pub fn forward(&self, input_a: &Tensor, input_b: &Tensor) -> Result<(Var, Var), GraphError> {
        let a = self.graph.input(input_a)?;
        let b = self.graph.input(input_b)?;
        // 分别编码两个输入
        let feat_a = self.encoder_a.forward(&a).relu();
        let feat_b = self.encoder_b.forward(&b).relu();

        // 拼接特征 [batch, 8] + [batch, 8] -> [batch, 16]
        let combined = Var::stack(&[&feat_a, &feat_b], 1, false)?;

        // 融合层
        let fused = self.fusion.forward(&combined).relu();

        // 分类头
        let cls_logits = self.cls_head.forward(&fused);

        // 回归头
        let reg_pred = self.reg_head.forward(&fused);

        Ok((cls_logits, reg_pred))
    }
}

impl Module for MultiIOFusion {
    fn parameters(&self) -> Vec<Var> {
        let mut params = self.encoder_a.parameters();
        params.extend(self.encoder_b.parameters());
        params.extend(self.fusion.parameters());
        params.extend(self.cls_head.parameters());
        params.extend(self.reg_head.parameters());
        params
    }
}
