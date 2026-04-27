/*
 * @Author       : 老董
 * @Date         : 2026-04-27
 * @Description  : 固定两实例分割小型 CNN 模型
 */

use only_torch::nn::{Conv2d, Graph, GraphError, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// 固定两实例 mask 预测网络。
///
/// 这是教学用 toy instance segmentation 模型：输入是一张单通道小图，
/// 输出两个固定 slot 的 logits 通道，而不是可变数量实例列表。
///
/// ```text
/// [N, 1, 16, 16]
///   -> Conv(1->8, 3x3, padding=1) -> ReLU
///   -> Conv(8->8, 3x3, padding=1) -> ReLU
///   -> Conv(8->2, 1x1) -> logits [N, 2, 16, 16]
/// ```
pub struct MultiInstanceSegmentationNet {
    conv1: Conv2d,
    conv2: Conv2d,
    head: Conv2d,
}

impl MultiInstanceSegmentationNet {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("MultiInstanceSegmentationNet");
        Ok(Self {
            conv1: Conv2d::new(&graph, 1, 8, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv1")?,
            conv2: Conv2d::new(&graph, 8, 8, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv2")?,
            head: Conv2d::new(&graph, 8, 2, (1, 1), (1, 1), (0, 0), (1, 1), true, "head")?,
        })
    }

    /// 返回两个实例 slot 的 logits，训练时直接接 BCEWithLogits 形式的 `bce_loss`。
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h = self.conv1.forward(x).relu();
        let h = self.conv2.forward(&h).relu();
        Ok(self.head.forward(&h))
    }

    /// 推理时转成概率图，方便按阈值计算每个实例 slot 的 mask。
    pub fn predict_probs(&self, x: &Tensor) -> Result<Var, GraphError> {
        Ok(self.forward(x)?.sigmoid())
    }
}

impl Module for MultiInstanceSegmentationNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.head.parameters(),
        ]
        .concat()
    }
}
