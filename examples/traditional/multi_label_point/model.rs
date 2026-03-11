/*
 * @Author       : 老董
 * @Date         : 2026-01-29
 * @Description  : 多标签点分类模型
 *
 * 展示 BCE Loss 的独特价值：多标签分类（一个样本可同时属于多个类别）
 *
 * 输入：二维点 (x, y)
 * 输出：4 个独立的二值属性（每个属性使用独立的 Sigmoid）
 */

use only_torch::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// 多标签点分类模型
///
/// ## 网络结构
/// ```text
/// Input(2) -> Linear(32, Tanh) -> Linear(32, Tanh) -> Linear(4) -> [logits]
/// ```
///
/// ## 输出标签
/// 4 个独立的二值属性：
/// - `is_right`: x > 0.5（点在右半边）
/// - `is_top`: y > 0.5（点在上半边）
/// - `is_diagonal_above`: x + y > 1（点在对角线上方）
/// - `is_center`: (x-0.5)² + (y-0.5)² < 0.15（点在中心圆内）
pub struct MultiLabelPointClassifier {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl MultiLabelPointClassifier {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("MultiLabelPointClassifier");
        Ok(Self {
            fc1: Linear::new(&graph, 2, 32, true, "fc1")?,
            fc2: Linear::new(&graph, 32, 32, true, "fc2")?,
            fc3: Linear::new(&graph, 32, 4, true, "fc3")?, // 4 个独立的 logits
        })
    }

    /// 前向传播
    ///
    /// # 返回
    /// - logits: `[batch, 4]`，未经激活的原始输出
    ///
    /// 注意：BCE Loss 内置 Sigmoid，所以这里输出 logits 而非概率
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h1 = self.fc1.forward(x).tanh();
        let h2 = self.fc2.forward(&h1).tanh();
        Ok(self.fc3.forward(&h2))
    }

    /// 预测概率（用于推理）
    ///
    /// 对 logits 应用 Sigmoid，返回各属性的概率
    pub fn predict_probs(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h1 = self.fc1.forward(x).tanh();
        let h2 = self.fc2.forward(&h1).tanh();
        Ok(self.fc3.forward(&h2).sigmoid())
    }
}

impl Module for MultiLabelPointClassifier {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc2.parameters(),
            self.fc3.parameters(),
        ]
        .concat()
    }
}
