//! MNIST CNN 模型
//!
//! LeNet 风格卷积网络用于手写数字识别（PyTorch 风格）
//!
//! 相比 MLP 版本：参数更少、泛化更好（平移不变性）

use only_torch::nn::{
    Conv2d, Graph, GraphError, Linear, MaxPool2d, Module, Var, VarActivationOps,
    VarRegularizationOps, VarShapeOps,
};
use only_torch::tensor::Tensor;

/// MNIST CNN
///
/// 网络结构:
/// ```text
/// Input [batch, 1, 28, 28]
///   → Conv1 (1→4, 3x3, pad=1) → ReLU → MaxPool(2x2)   [batch, 4, 14, 14]
///   → Conv2 (4→8, 3x3, pad=1) → ReLU → MaxPool(2x2)   [batch, 8, 7, 7]
///   → Flatten                                            [batch, 392]
///   → FC1 (392→32) → ReLU → Dropout(0.2)
///   → FC2 (32→10)
/// ```
///
/// - 参数量: ~13K（精简设计，快速训练）
/// - 平移不变性: 裁切偏移几像素也能正确识别
pub struct MnistCNN {
    conv1: Conv2d,
    pool1: MaxPool2d,
    conv2: Conv2d,
    pool2: MaxPool2d,
    fc1: Linear,
    fc2: Linear,
}

impl MnistCNN {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("MnistCNN");
        Ok(Self {
            // 卷积层 1: 1→4 通道, 3x3 核, padding=1 (same padding)
            conv1: Conv2d::new(&graph, 1, 4, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv1")?,
            pool1: MaxPool2d::new(&graph, (2, 2), None, "pool1"),
            // 卷积层 2: 4→8 通道, 3x3 核, padding=1
            conv2: Conv2d::new(&graph, 4, 8, (3, 3), (1, 1), (1, 1), (1, 1), true, "conv2")?,
            pool2: MaxPool2d::new(&graph, (2, 2), None, "pool2"),
            // 全连接层: 8*7*7=392 → 32 → 10
            fc1: Linear::new(&graph, 392, 32, true, "fc1")?,
            fc2: Linear::new(&graph, 32, 10, true, "fc2")?,
        })
    }

    /// PyTorch 风格 forward：直接接收 Tensor
    ///
    /// 输入形状: [batch, 1, 28, 28]
    /// 输出形状: [batch, 10]（logits，未经 softmax）
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        // Conv1 → ReLU → Pool1
        let h = self.conv1.forward(x).relu();
        let h = self.pool1.forward(&h);
        // Conv2 → ReLU → Pool2
        let h = self.conv2.forward(&h).relu();
        let h = self.pool2.forward(&h);
        // Flatten → FC1 → ReLU → Dropout → FC2
        let h = h.flatten()?;
        let h = self.fc1.forward(&h).relu();
        let h = h.dropout(0.2)?;
        Ok(self.fc2.forward(&h))
    }
}

impl Module for MnistCNN {
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
