//! MNIST MLP 模型
//!
//! 两层全连接网络用于手写数字识别（PyTorch 风格）
//!
//! 包含 Dropout 正则化，演示 train/eval 模式切换

use only_torch::nn::{
    Graph, GraphError, Linear, Module, Var, VarActivationOps, VarRegularizationOps,
};
use only_torch::tensor::Tensor;

/// MNIST MLP
///
/// 网络结构: Input(784) -> Linear(128, Softplus) -> Dropout(0.3) -> Linear(10)
///
/// # 注意
/// 使用了 Dropout，训练/测试时需要切换模式：
/// - 训练前：`graph.train()`（默认）
/// - 测试前：`graph.eval()`
pub struct MnistMLP {
    fc1: Linear,
    fc2: Linear,
}

impl MnistMLP {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            // 784 = 28x28 (MNIST 图片展平后的维度)
            fc1: Linear::new(graph, 784, 128, true, "fc1")?,
            fc2: Linear::new(graph, 128, 10, true, "fc2")?,
        })
    }

    /// `PyTorch` 风格 forward：直接接收 Tensor
    ///
    /// 注意：包含 Dropout，行为取决于当前模式（train/eval）
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h1 = self.fc1.forward(x).softplus();
        let h1 = h1.dropout(0.3)?; // Dropout: 训练时丢弃 30%，评估时直接通过
        Ok(self.fc2.forward(&h1))
    }
}

impl Module for MnistMLP {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}
