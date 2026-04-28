/*
 * @Author       : 老董
 * @Date         : 2026-04-28
 * @Description  : DeformableConv2d 语义分割手写网络示例模型
 */

use only_torch::nn::{Conv2d, DeformableConv2d, Graph, GraphError, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

pub struct DeformableSegmentationNet {
    stem: Conv2d,
    deform: DeformableConv2d,
    refine: Conv2d,
    head: Conv2d,
}

impl DeformableSegmentationNet {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("DeformableSegmentationNet");
        Ok(Self {
            stem: Conv2d::new(&graph, 1, 8, (3, 3), (1, 1), (1, 1), (1, 1), true, "stem")?,
            deform: DeformableConv2d::new(
                &graph,
                8,
                8,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                1,
                true,
                "deform",
            )?,
            refine: Conv2d::new(&graph, 8, 8, (3, 3), (1, 1), (1, 1), (1, 1), true, "refine")?,
            head: Conv2d::new(&graph, 8, 1, (1, 1), (1, 1), (0, 0), (1, 1), true, "head")?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h = self.stem.forward(x).relu();
        let h = self.deform.forward(&h).relu();
        let h = self.refine.forward(&h).relu();
        Ok(self.head.forward(&h))
    }

    pub fn predict_probs(&self, x: &Tensor) -> Result<Var, GraphError> {
        Ok(self.forward(x)?.sigmoid())
    }
}

impl Module for DeformableSegmentationNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.stem.parameters(),
            self.deform.parameters(),
            self.refine.parameters(),
            self.head.parameters(),
        ]
        .concat()
    }
}
