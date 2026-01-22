/*
 * MNIST GAN 模型定义
 *
 * 使用 PyTorch 风格的 API，展示 Only-Torch 的 GAN 训练能力。
 *
 * # 架构
 * - Generator: 噪声 -> 隐藏层 -> 图像
 * - Discriminator: 图像 -> 隐藏层 -> 真/假判别
 *
 * # 关键特性
 * - 使用 `ForwardInput` trait 实现 Tensor/Var 统一输入
 * - 使用函数式 `detach()` 实现 GAN 训练的梯度控制
 */

use only_torch::nn::{
    ForwardInput, Graph, GraphError, Linear, ModelState, Module, Var, VarActivationOps,
};

/// 噪声维度（latent space）
pub const LATENT_DIM: usize = 64;
/// 图像维度（28x28 = 784）
pub const IMAGE_DIM: usize = 784;
/// 隐藏层维度（与 archive 版本一致）
const HIDDEN_DIM: usize = 128;

/// 生成器
///
/// 将随机噪声转换为 28x28 的图像
/// 结构与 archive 版本一致：z(64) -> FC(128, `LeakyReLU`) -> FC(784, Sigmoid)
pub struct Generator {
    fc1: Linear,
    fc2: Linear,
    state: ModelState,
}

impl Generator {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, LATENT_DIM, HIDDEN_DIM, true, "g_fc1")?,
            fc2: Linear::new(graph, HIDDEN_DIM, IMAGE_DIM, true, "g_fc2")?,
            state: ModelState::new(graph),
        })
    }

    /// 前向传播
    ///
    /// 输入: [batch, `LATENT_DIM`] 的噪声
    /// 输出: [batch, `IMAGE_DIM`] 的生成图像（值域 [0, 1]）
    pub fn forward(&self, z: impl ForwardInput) -> Result<Var, GraphError> {
        self.state.forward(z, |input| {
            let h1 = self.fc1.forward(input).leaky_relu(0.2);
            let out = self.fc2.forward(&h1).sigmoid();
            Ok(out)
        })
    }
}

impl Module for Generator {
    fn parameters(&self) -> Vec<Var> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}

/// 判别器
///
/// 判断输入图像是真实的还是生成的
/// 结构与 archive 版本一致：image(784) -> FC(128, `LeakyReLU`) -> FC(1, Sigmoid)
pub struct Discriminator {
    fc1: Linear,
    fc2: Linear,
    state: ModelState,
}

impl Discriminator {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, IMAGE_DIM, HIDDEN_DIM, true, "d_fc1")?,
            fc2: Linear::new(graph, HIDDEN_DIM, 1, true, "d_fc2")?,
            state: ModelState::new(graph),
        })
    }

    /// 前向传播
    ///
    /// 输入: [batch, `IMAGE_DIM`] 的图像
    /// 输出: [batch, 1] 的判别概率（经过 sigmoid，值域 [0, 1]）
    ///
    /// # 输入类型
    /// - `&Tensor`: 真实图像（使用缓存）
    /// - `&Var`: 生成图像（不缓存，支持 GAN 训练）
    pub fn forward(&self, x: impl ForwardInput) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            let h1 = self.fc1.forward(input).leaky_relu(0.2);
            let out = self.fc2.forward(&h1).sigmoid();
            Ok(out)
        })
    }
}

impl Module for Discriminator {
    fn parameters(&self) -> Vec<Var> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}
