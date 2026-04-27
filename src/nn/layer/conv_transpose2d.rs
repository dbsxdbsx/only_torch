/*
 * @Author       : 老董
 * @Date         : 2026-04-19
 * @Description  : ConvTranspose2d (2D 转置卷积) 层 - PyTorch 风格 API
 *
 * 输入/输出形状：
 * - 输入：[batch_size, in_channels, H, W]
 * - 输出：[batch_size, out_channels, H', W']
 *
 * 输出尺寸计算：
 * H' = (H - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h
 * W' = (W - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w
 *
 * 卷积核布局：[in_channels, out_channels, kH, kW]
 * 计算：output = conv_transpose2d(x, K) + b
 */

use crate::nn::graph::NodeGroupContext;
use crate::nn::{Graph, GraphError, Init, IntoVar, Module, Var};

/// ConvTranspose2d (2D 转置卷积 / 反卷积) 层
///
/// `PyTorch` 风格：`output = conv_transpose2d(x, K) + b`
///
/// # 输入/输出形状
/// - 输入：[`batch_size`, `in_channels`, H, W]
/// - 输出：[`batch_size`, `out_channels`, H', W']
///
/// # 使用示例
/// ```ignore
/// let deconv = ConvTranspose2d::new(&graph, 32, 16, (3, 3), (2, 2), (1, 1), (1, 1), true, "deconv1")?;
/// let h = deconv.forward(&x).relu();
/// ```
pub struct ConvTranspose2d {
    kernel: Var,
    bias: Option<Var>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    name: String,
    instance_id: usize,
}

impl ConvTranspose2d {
    /// 创建新的 ConvTranspose2d 层
    ///
    /// # 参数
    /// - `graph`: 计算图句柄
    /// - `in_channels`: 输入通道数
    /// - `out_channels`: 输出通道数
    /// - `kernel_size`: 卷积核大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)
    /// - `padding`: 填充 (pH, pW)
    /// - `output_padding`: 输出侧额外尺寸 (opH, opW)
    /// - `use_bias`: 是否使用偏置
    /// - `name`: 层名称前缀
    pub fn new(
        graph: &Graph,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        use_bias: bool,
        name: &str,
    ) -> Result<Self, GraphError> {
        let (k_h, k_w) = kernel_size;

        // 转置卷积核: [C_in, C_out, kH, kW]
        let kernel = graph.parameter(
            &[in_channels, out_channels, k_h, k_w],
            Init::Kaiming,
            &format!("{name}_K"),
        )?;

        let bias = if use_bias {
            Some(graph.parameter(&[1, out_channels, 1, 1], Init::Zeros, &format!("{name}_b"))?)
        } else {
            None
        };

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Ok(Self {
            kernel,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            name: name.to_string(),
            instance_id,
        })
    }

    /// 前向传播
    pub fn forward(&self, x: impl IntoVar) -> Var {
        use std::rc::Rc;
        let x = x
            .into_var(&self.kernel.get_graph())
            .expect("ConvTranspose2d 输入转换失败");
        let graph = x.get_graph();

        let desc = if self.bias.is_some() {
            format!(
                "[?, {}, ?, ?] → [?, {}, ?, ?], kernel {}×{}",
                self.in_channels, self.out_channels, self.kernel_size.0, self.kernel_size.1
            )
        } else {
            format!(
                "[?, {}, ?, ?] → [?, {}, ?, ?], kernel {}×{} (no bias)",
                self.in_channels, self.out_channels, self.kernel_size.0, self.kernel_size.1
            )
        };
        let _guard =
            NodeGroupContext::for_layer(&x, "ConvTranspose2d", self.instance_id, &self.name, &desc);
        _guard.tag_existing(&self.kernel);
        if let Some(ref bias) = self.bias {
            _guard.tag_existing(bias);
        }

        let conv_out = {
            let conv_node = graph
                .inner_mut()
                .create_conv_transpose2d_node(
                    vec![Rc::clone(x.node()), Rc::clone(self.kernel.node())],
                    self.stride,
                    self.padding,
                    self.output_padding,
                    Some(&format!("{}_deconv", self.name)),
                )
                .expect("ConvTranspose2d 前向计算失败");
            Var::new_with_rc_graph(conv_node, &graph.inner_rc())
        };

        if let Some(ref bias) = self.bias {
            let out_node = graph
                .inner_mut()
                .create_add_node(
                    vec![Rc::clone(conv_out.node()), Rc::clone(bias.node())],
                    Some(&format!("{}_out", self.name)),
                )
                .expect("ConvTranspose2d bias add 失败");
            Var::new_with_rc_graph(out_node, &graph.inner_rc())
        } else {
            conv_out
        }
    }

    pub const fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub const fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub const fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    pub const fn stride(&self) -> (usize, usize) {
        self.stride
    }

    pub const fn padding(&self) -> (usize, usize) {
        self.padding
    }

    pub const fn output_padding(&self) -> (usize, usize) {
        self.output_padding
    }

    /// 获取卷积核 Var
    pub const fn kernel(&self) -> &Var {
        &self.kernel
    }

    /// 获取偏置 Var（如果有）
    pub const fn bias(&self) -> Option<&Var> {
        self.bias.as_ref()
    }
}

impl Module for ConvTranspose2d {
    fn parameters(&self) -> Vec<Var> {
        let mut params = vec![self.kernel.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}
