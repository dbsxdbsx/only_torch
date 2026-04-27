/*
 * @Author       : 老董
 * @Date         : 2026-04-27
 * @Description  : 重叠形状语义分割 benchmark 的小型全卷积模型
 */

use only_torch::nn::{Conv2d, Graph, GraphError, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// 多形状语义分割网络。
///
/// ```text
/// [N, 1, 64, 64]
///   -> Conv(1->12, 3x3, padding=1) -> ReLU
///   -> Conv(12->16, 3x3, padding=1) -> ReLU
///   -> Conv(16->4, 1x1) -> logits [N, 4, 64, 64]
/// ```
pub struct OverlappingShapesSemanticSegmentationNet {
    conv1: Conv2d,
    conv2: Conv2d,
    head: Conv2d,
}

impl OverlappingShapesSemanticSegmentationNet {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("OverlappingShapesSemanticSegmentationNet");
        Ok(Self {
            conv1: Conv2d::new(&graph, 1, 12, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv1")?,
            conv2: Conv2d::new(
                &graph,
                12,
                16,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                true,
                "conv2",
            )?,
            head: Conv2d::new(&graph, 16, 4, (1, 1), (1, 1), (0, 0), (1, 1), true, "head")?,
        })
    }

    /// 返回每类语义 mask 的 logits，训练时使用 one-hot target + BCEWithLogits。
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h = self.conv1.forward(x).relu();
        let h = self.conv2.forward(&h).relu();
        Ok(self.head.forward(&h))
    }

    pub fn predict_probs(&self, x: &Tensor) -> Result<Var, GraphError> {
        Ok(self.forward(x)?.sigmoid())
    }
}

impl Module for OverlappingShapesSemanticSegmentationNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.head.parameters(),
        ]
        .concat()
    }
}
