/*
 * @Author       : 老董
 * @Date         : 2026-04-27
 * @Description  : 单目标检测小型 CNN 模型
 */

use only_torch::nn::{
    Conv2d, Graph, GraphError, Linear, MaxPool2d, Module, Var, VarActivationOps, VarShapeOps,
};
use only_torch::tensor::Tensor;

/// 单目标检测网络。
///
/// 输入为 `[N, 1, 16, 16]` 灰度图，输出为 `[N, 4]` 的归一化
/// `cx, cy, w, h` bbox。
pub struct SingleObjectDetectionNet {
    conv1: Conv2d,
    pool1: MaxPool2d,
    conv2: Conv2d,
    pool2: MaxPool2d,
    fc1: Linear,
    fc2: Linear,
}

impl SingleObjectDetectionNet {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("SingleObjectDetectionNet");
        Ok(Self {
            conv1: Conv2d::new(&graph, 1, 8, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv1")?,
            pool1: MaxPool2d::new(&graph, (2, 2), None, "pool1"),
            conv2: Conv2d::new(&graph, 8, 16, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv2")?,
            pool2: MaxPool2d::new(&graph, (2, 2), None, "pool2"),
            fc1: Linear::new(&graph, 16 * 4 * 4, 64, true, "fc1")?,
            fc2: Linear::new(&graph, 64, 4, true, "bbox_head")?,
        })
    }

    /// 返回归一化 bbox，训练时直接接 Huber / MSE 回归损失。
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h = self.conv1.forward(x).relu();
        let h = self.pool1.forward(&h);
        let h = self.conv2.forward(&h).relu();
        let h = self.pool2.forward(&h);
        let h = h.flatten()?;
        let h = self.fc1.forward(&h).relu();
        Ok(self.fc2.forward(&h).sigmoid())
    }

    pub fn predict_boxes(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.forward(x)
    }
}

impl Module for SingleObjectDetectionNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.fc1.parameters(),
            self.fc2.parameters(),
        ]
        .concat()
    }
}
