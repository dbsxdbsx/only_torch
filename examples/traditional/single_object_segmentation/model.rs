/*
 * @Author       : 老董
 * @Date         : 2026-04-27
 * @Description  : 单目标语义分割小型 CNN 模型
 */

use only_torch::nn::{Conv2d, Graph, GraphError, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// 单目标二值语义分割网络。
///
/// 输入 / 输出均保持空间结构：
/// ```text
/// [N, 1, 16, 16]
///   -> Conv(1->4, 3x3, padding=1) -> ReLU
///   -> Conv(4->4, 3x3, padding=1) -> ReLU
///   -> Conv(4->1, 1x1) -> logits [N, 1, 16, 16]
/// ```
pub struct SingleObjectSegmentationNet {
    conv1: Conv2d,
    conv2: Conv2d,
    head: Conv2d,
}

impl SingleObjectSegmentationNet {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("SingleObjectSegmentationNet");
        Ok(Self {
            conv1: Conv2d::new(&graph, 1, 4, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv1")?,
            conv2: Conv2d::new(&graph, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv2")?,
            head: Conv2d::new(&graph, 4, 1, (1, 1), (1, 1), (0, 0), (1, 1), true, "head")?,
        })
    }

    /// 返回 logits，训练时直接接 BCEWithLogits 形式的 `bce_loss`。
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h = self.conv1.forward(x).relu();
        let h = self.conv2.forward(&h).relu();
        Ok(self.head.forward(&h))
    }

    /// 推理时转成概率图，方便按阈值计算 mask 指标。
    pub fn predict_probs(&self, x: &Tensor) -> Result<Var, GraphError> {
        Ok(self.forward(x)?.sigmoid())
    }
}

impl Module for SingleObjectSegmentationNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.head.parameters(),
        ]
        .concat()
    }
}
