/*
 * @Author       : 老董
 * @Date         : 2026-04-28
 * @Description  : 重叠形状语义分割 benchmark 的 U-Net-lite 强基线
 */

use only_torch::nn::layer::ConvTranspose2d;
use only_torch::nn::{Conv2d, Graph, GraphError, MaxPool2d, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// U-Net-lite 多形状语义分割网络。
///
/// ```text
/// [N, 1, 64, 64]
///   -> Encoder: Conv(1->8) -> ReLU -> Conv(8->8) -> ReLU
///   -> MaxPool(2x2)
///   -> Bottleneck: Conv(8->16) -> ReLU
///   -> ConvTranspose(16->8, stride=2)
///   -> Concat(skip, up)
///   -> Decoder: Conv(16->12) -> ReLU
///   -> Conv(12->4, 1x1) -> logits [N, 4, 64, 64]
/// ```
pub struct OverlappingShapesUnetLiteSegmentationNet {
    enc1: Conv2d,
    enc2: Conv2d,
    pool: MaxPool2d,
    bottleneck: Conv2d,
    up1: ConvTranspose2d,
    dec1: Conv2d,
    head: Conv2d,
}

impl OverlappingShapesUnetLiteSegmentationNet {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("OverlappingShapesUnetLiteSegmentationNet");
        Ok(Self {
            enc1: Conv2d::new(&graph, 1, 8, (3, 3), (1, 1), (1, 1), (1, 1), true, "enc1")?,
            enc2: Conv2d::new(&graph, 8, 8, (3, 3), (1, 1), (1, 1), (1, 1), true, "enc2")?,
            pool: MaxPool2d::new(&graph, (2, 2), None, "pool1"),
            bottleneck: Conv2d::new(
                &graph,
                8,
                16,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                true,
                "bottleneck",
            )?,
            up1: ConvTranspose2d::new(&graph, 16, 8, (2, 2), (2, 2), (0, 0), (0, 0), true, "up1")?,
            dec1: Conv2d::new(&graph, 16, 12, (3, 3), (1, 1), (1, 1), (1, 1), true, "dec1")?,
            head: Conv2d::new(&graph, 12, 4, (1, 1), (1, 1), (0, 0), (1, 1), true, "head")?,
        })
    }

    /// 返回每类语义 mask 的 logits，训练时使用 one-hot target + BCEWithLogits。
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h = self.enc1.forward(x).relu();
        let skip = self.enc2.forward(&h).relu();
        let pooled = self.pool.forward(&skip);
        let h = self.bottleneck.forward(&pooled).relu();
        let up = self.up1.forward(&h).relu();
        let merged = Var::concat(&[&up, &skip], 1)?;
        let h = self.dec1.forward(&merged).relu();
        Ok(self.head.forward(&h))
    }

    pub fn predict_probs(&self, x: &Tensor) -> Result<Var, GraphError> {
        Ok(self.forward(x)?.sigmoid())
    }
}

impl Module for OverlappingShapesUnetLiteSegmentationNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.enc1.parameters(),
            self.enc2.parameters(),
            self.bottleneck.parameters(),
            self.up1.parameters(),
            self.dec1.parameters(),
            self.head.parameters(),
        ]
        .concat()
    }
}
