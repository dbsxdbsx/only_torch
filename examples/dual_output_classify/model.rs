/*
 * @Author       : 老董
 * @Date         : 2026-01-28
 * @Description  : 双输出模型定义（分类头 + 回归头）
 *
 * 核心特点：展示 Graph 的多输出功能
 * - 共享特征层：一次编码，两个任务共用
 * - 分类头：判断正/负
 * - 回归头：预测绝对值
 */

use only_torch::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// 双输出分类模型
///
/// 结构：
/// ```text
/// 输入(1) ─> 共享层(8, ReLU) ─┬─> 分类头(2) ─> 正/负（softmax + cross_entropy）
///                             │
///                             └─> 回归头(1) ─> |x|（MSE）
/// ```
pub struct DualOutputClassifier {
    /// 共享特征层
    shared: Linear,
    /// 分类头（二分类：负=0, 正=1）
    cls_head: Linear,
    /// 回归头（预测绝对值）
    reg_head: Linear,
    /// 计算图
    graph: Graph,
}

impl DualOutputClassifier {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        // 共享层：1 -> 16（增加容量）
        let shared = Linear::new(graph, 1, 16, true, "shared")?;
        // 分类头：16 -> 2（二分类 logits）
        let cls_head = Linear::new(graph, 16, 2, true, "cls_head")?;
        // 回归头：16 -> 1（绝对值预测）
        let reg_head = Linear::new(graph, 16, 1, true, "reg_head")?;

        Ok(Self {
            shared,
            cls_head,
            reg_head,
            graph: graph.clone(),
        })
    }

    /// 前向传播（返回双输出）
    ///
    /// # 返回
    /// - `cls_logits`: 分类 logits `[batch, 2]`
    /// - `reg_pred`: 回归预测 `[batch, 1]`
    pub fn forward(&self, x: &Tensor) -> Result<(Var, Var), GraphError> {
        let input = self.graph.input(x)?;
        // 共享特征提取（使用 tanh 保留正负信息）
        let feat = self.shared.forward(&input).tanh();

        // 分类头（二分类 logits）
        let cls_logits = self.cls_head.forward(&feat);

        // 回归头（绝对值预测，使用 abs 确保非负）
        let reg_pred = self.reg_head.forward(&feat).abs();

        Ok((cls_logits, reg_pred))
    }
}

impl Module for DualOutputClassifier {
    fn parameters(&self) -> Vec<Var> {
        let mut params = self.shared.parameters();
        params.extend(self.cls_head.parameters());
        params.extend(self.reg_head.parameters());
        params
    }
}
