/*
 * MNIST GAN 模型定义（Phase 3 新版本）
 *
 * 使用 PyTorch 风格的 API。
 *
 * # 架构
 * - Generator: 噪声 -> 隐藏层 -> 图像
 * - Discriminator: 图像 -> 隐藏层 -> 真/假判别
 *
 * # 关键特性
 * - 使用 `IntoVar` trait：forward 同时接受 Tensor 和 Var
 * - 使用 `with_model_name` 实现可视化模型分组
 * - 使用 `detach()` 实现 GAN 训练的梯度控制
 */

use only_torch::nn::{Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps};

/// 噪声维度（latent space）
pub const LATENT_DIM: usize = 64;
/// 图像维度（28x28 = 784）
pub const IMAGE_DIM: usize = 784;
/// 隐藏层维度
const HIDDEN_DIM: usize = 128;

/// 生成器
///
/// 将随机噪声转换为 28x28 的图像
/// 结构：z(64) -> FC(128, LeakyReLU) -> FC(784, Sigmoid)
pub struct Generator {
    graph: Graph,
    fc1: Linear,
    fc2: Linear,
}

impl Generator {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        // with_model_name: 可视化时自动将层分组到 "Generator" cluster 内
        let graph = graph.with_model_name("Generator");
        Ok(Self {
            graph: graph.clone(),
            fc1: Linear::new(&graph, LATENT_DIM, HIDDEN_DIM, true, "fc1")?,
            fc2: Linear::new(&graph, HIDDEN_DIM, IMAGE_DIM, true, "fc2")?,
        })
    }

    /// 前向传播
    ///
    /// 输入: [batch, LATENT_DIM] 的噪声（Tensor 或 Var）
    /// 输出: [batch, IMAGE_DIM] 的生成图像 Var（值域 [0, 1]）
    pub fn forward(&self, z: impl IntoVar) -> Result<Var, GraphError> {
        let input = z.into_var(&self.graph)?;
        let h1 = self.fc1.forward(&input).leaky_relu(0.2);
        let out = self.fc2.forward(&h1).sigmoid();
        Ok(out)
    }
}

impl Module for Generator {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

/// 判别器
///
/// 判断输入图像是真实的还是生成的
/// 结构：image(784) -> FC(128, LeakyReLU) -> FC(1, Sigmoid)
pub struct Discriminator {
    graph: Graph,
    fc1: Linear,
    fc2: Linear,
}

impl Discriminator {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Discriminator");
        Ok(Self {
            graph: graph.clone(),
            fc1: Linear::new(&graph, IMAGE_DIM, HIDDEN_DIM, true, "fc1")?,
            fc2: Linear::new(&graph, HIDDEN_DIM, 1, true, "fc2")?,
        })
    }

    /// 前向传播
    ///
    /// 输入: [batch, IMAGE_DIM] 的图像（Tensor 或 Var 均可）
    /// 输出: [batch, 1] 的判别概率（经过 sigmoid，值域 [0, 1]）
    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let input = x.into_var(&self.graph)?;
        let h1 = self.fc1.forward(&input).leaky_relu(0.2);
        let out = self.fc2.forward(&h1).sigmoid();
        Ok(out)
    }
}

impl Module for Discriminator {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}
