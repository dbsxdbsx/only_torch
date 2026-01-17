/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Conv2d (2D 卷积) 层 - PyTorch 风格 API
 *
 * 输入/输出形状：
 * - 输入：[batch_size, in_channels, H, W]
 * - 输出：[batch_size, out_channels, H', W']
 *
 * 输出尺寸计算：
 * H' = (H + 2*padding_h - kernel_h) / stride_h + 1
 * W' = (W + 2*padding_w - kernel_w) / stride_w + 1
 *
 * 计算：output = conv2d(x, K) + b
 */

use crate::nn::{Graph, GraphError, Init, Module, Var};

// ==================== 新版 Conv2d 结构体（推荐）====================

/// Conv2d (2D 卷积) 层
///
/// PyTorch 风格的卷积层：`output = conv2d(x, K) + b`
///
/// # 输入/输出形状
/// - 输入：[batch_size, in_channels, H, W]
/// - 输出：[batch_size, out_channels, H', W']
///
/// # 输出尺寸计算
/// ```text
/// H' = (H + 2*padding_h - kernel_h) / stride_h + 1
/// W' = (W + 2*padding_w - kernel_w) / stride_w + 1
/// ```
///
/// # 使用示例
/// ```ignore
/// let conv = Conv2d::new(&graph, 1, 32, (3, 3), (1, 1), (1, 1), true, "conv1")?;
/// let h = conv.forward(&x).relu();  // 链式调用
/// ```
pub struct Conv2d {
    /// 卷积核参数 [out_channels, in_channels, kernel_h, kernel_w]
    kernel: Var,
    /// 偏置参数 [1, out_channels]（可选）
    bias: Option<Var>,
    /// 输入通道数
    in_channels: usize,
    /// 输出通道数
    out_channels: usize,
    /// 卷积核大小 (kernel_h, kernel_w)
    kernel_size: (usize, usize),
    /// 步长 (stride_h, stride_w)
    stride: (usize, usize),
    /// 填充 (padding_h, padding_w)
    padding: (usize, usize),
    /// 层名称（用于可视化分组）
    name: String,
}

impl Conv2d {
    /// 创建新的 Conv2d 层
    ///
    /// # 参数
    /// - `graph`: 计算图句柄
    /// - `in_channels`: 输入通道数
    /// - `out_channels`: 输出通道数
    /// - `kernel_size`: 卷积核大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)
    /// - `padding`: 填充 (pH, pW)
    /// - `use_bias`: 是否使用偏置
    /// - `name`: 层名称前缀
    ///
    /// # 返回
    /// Conv2d 层实例
    pub fn new(
        graph: &Graph,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        use_bias: bool,
        name: &str,
    ) -> Result<Self, GraphError> {
        let (k_h, k_w) = kernel_size;

        // 创建卷积核参数：Kaiming 初始化
        let kernel = graph.parameter(
            &[out_channels, in_channels, k_h, k_w],
            Init::Kaiming,
            &format!("{name}_K"),
        )?;

        // 创建偏置参数（可选）：零初始化
        let bias = if use_bias {
            Some(graph.parameter(&[1, out_channels], Init::Zeros, &format!("{name}_b"))?)
        } else {
            None
        };

        Ok(Self {
            kernel,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            name: name.to_string(),
        })
    }

    /// 创建新的 Conv2d 层（带种子，确保可重复性）
    ///
    /// # 参数
    /// - `graph`: 计算图句柄
    /// - `in_channels`: 输入通道数
    /// - `out_channels`: 输出通道数
    /// - `kernel_size`: 卷积核大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)
    /// - `padding`: 填充 (pH, pW)
    /// - `use_bias`: 是否使用偏置
    /// - `name`: 层名称前缀
    /// - `seed`: 随机种子
    ///
    /// # 返回
    /// Conv2d 层实例
    pub fn new_seeded(
        graph: &Graph,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        use_bias: bool,
        name: &str,
        seed: u64,
    ) -> Result<Self, GraphError> {
        let (k_h, k_w) = kernel_size;

        // 创建卷积核参数：使用固定种子初始化
        let kernel = graph.parameter_seeded(
            &[out_channels, in_channels, k_h, k_w],
            &format!("{name}_K"),
            seed,
        )?;

        // 创建偏置参数（可选）：零初始化（无需种子）
        let bias = if use_bias {
            Some(graph.parameter(&[1, out_channels], Init::Zeros, &format!("{name}_b"))?)
        } else {
            None
        };

        Ok(Self {
            kernel,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            name: name.to_string(),
        })
    }

    /// 前向传播
    ///
    /// 计算 `conv2d(x, K) + b`
    ///
    /// # 参数
    /// - `x`: 输入 Var，形状 [batch_size, in_channels, H, W]
    ///
    /// # 返回
    /// 输出 Var，形状 [batch_size, out_channels, H', W']
    pub fn forward(&self, x: &Var) -> Var {
        let graph = x.get_graph();

        // Conv2d: [batch, C_in, H, W] * [C_out, C_in, kH, kW] -> [batch, C_out, H', W']
        let conv_out = {
            let mut g = graph.inner_mut();
            let conv_id = g
                .new_conv2d_node(
                    x.node_id(),
                    self.kernel.node_id(),
                    self.stride,
                    self.padding,
                    Some(&format!("{}_conv", self.name)),
                )
                .expect("Conv2d conv 失败");
            Var::new(conv_id, graph.inner_rc())
        };

        // 如果有 bias，通过 ChannelBiasAdd 添加
        if let Some(ref bias) = self.bias {
            let output = {
                let mut g = graph.inner_mut();
                let out_id = g
                    .new_channel_bias_add_node(
                        conv_out.node_id(),
                        bias.node_id(),
                        Some(&format!("{}_out", self.name)),
                    )
                    .expect("Conv2d bias add 失败");
                Var::new(out_id, graph.inner_rc())
            };

            // 注册层分组（用于可视化）
            graph.inner_mut().register_layer_group(
                &self.name,
                "Conv2d",
                &format!(
                    "{}→{}, {}×{}",
                    self.in_channels, self.out_channels, self.kernel_size.0, self.kernel_size.1
                ),
                vec![
                    self.kernel.node_id(),
                    bias.node_id(),
                    conv_out.node_id(),
                    output.node_id(),
                ],
            );

            output
        } else {
            // 无 bias 时，直接返回 conv_out
            graph.inner_mut().register_layer_group(
                &self.name,
                "Conv2d",
                &format!(
                    "{}→{}, {}×{} (no bias)",
                    self.in_channels, self.out_channels, self.kernel_size.0, self.kernel_size.1
                ),
                vec![self.kernel.node_id(), conv_out.node_id()],
            );
            conv_out
        }
    }

    /// 获取输入通道数
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// 获取输出通道数
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// 获取卷积核大小
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// 获取步长
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// 获取填充
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// 获取卷积核 Var
    pub fn kernel(&self) -> &Var {
        &self.kernel
    }

    /// 获取偏置 Var（如果有）
    pub fn bias(&self) -> Option<&Var> {
        self.bias.as_ref()
    }
}

impl Module for Conv2d {
    fn parameters(&self) -> Vec<Var> {
        let mut params = vec![self.kernel.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}
