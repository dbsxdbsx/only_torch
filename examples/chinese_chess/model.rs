//! 中国象棋棋子 CNN 分类器模型
//!
//! 输入 28x28 RGB patch，输出 15 类概率

use only_torch::nn::{
    Conv2d, Graph, GraphError, Linear, MaxPool2d, Module, Var, VarActivationOps,
    VarRegularizationOps, VarShapeOps,
};
use only_torch::tensor::Tensor;

/// 中国象棋棋子 CNN 分类器
///
/// 网络结构:
/// ```text
/// Input [batch, 3, 28, 28]
///   → Conv1 (3→16, 3x3, pad=1) → ReLU → MaxPool(2x2)   [batch, 16, 14, 14]
///   → Conv2 (16→32, 3x3, pad=1) → ReLU → MaxPool(2x2)  [batch, 32, 7, 7]
///   → Flatten                                              [batch, 1568]
///   → FC1 (1568→128) → ReLU → Dropout(0.2)
///   → FC2 (128→15)
/// ```
///
/// - 参数量: ~205K
/// - 15 类: 1 空位 + 7 红子 + 7 黑子
pub struct ChessPieceCNN {
    conv1: Conv2d,
    pool1: MaxPool2d,
    conv2: Conv2d,
    pool2: MaxPool2d,
    fc1: Linear,
    fc2: Linear,
}

impl ChessPieceCNN {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("ChessPieceCNN");
        Ok(Self {
            conv1: Conv2d::new(&graph, 3, 16, (3, 3), (1, 1), (1, 1), true, "conv1")?,
            pool1: MaxPool2d::new(&graph, (2, 2), None, "pool1"),
            conv2: Conv2d::new(&graph, 16, 32, (3, 3), (1, 1), (1, 1), true, "conv2")?,
            pool2: MaxPool2d::new(&graph, (2, 2), None, "pool2"),
            // 32 * 7 * 7 = 1568
            fc1: Linear::new(&graph, 1568, 128, true, "fc1")?,
            fc2: Linear::new(&graph, 128, 15, true, "fc2")?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h = self.conv1.forward(x).relu();
        let h = self.pool1.forward(&h);
        let h = self.conv2.forward(&h).relu();
        let h = self.pool2.forward(&h);
        let h = h.flatten()?;
        let h = self.fc1.forward(&h).relu();
        let h = h.dropout(0.2)?;
        Ok(self.fc2.forward(&h))
    }
}

impl Module for ChessPieceCNN {
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
