/*
 * @Author       : 老董
 * @Date         : 2026-04-28
 * @Description  : DeformableConv2d 层 - offset-only v1
 */

use crate::nn::graph::NodeGroupContext;
use crate::nn::{Graph, GraphError, Init, IntoVar, Module, Var};

/// DeformableConv2d 层。
///
/// v1 版本采用 offset-only Deformable Conv2d：
/// - 主卷积核负责采样后的卷积；
/// - offset 由一个同尺寸卷积预测，初始为 0，因此初始行为等价普通 Conv2d。
pub struct DeformableConv2d {
    kernel: Var,
    bias: Option<Var>,
    offset_kernel: Var,
    offset_bias: Var,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    deformable_groups: usize,
    name: String,
    instance_id: usize,
}

impl DeformableConv2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        graph: &Graph,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        deformable_groups: usize,
        use_bias: bool,
        name: &str,
    ) -> Result<Self, GraphError> {
        if deformable_groups == 0 || in_channels % deformable_groups != 0 {
            return Err(GraphError::InvalidOperation(format!(
                "in_channels={in_channels} 必须能被 deformable_groups={deformable_groups} 整除"
            )));
        }
        let (k_h, k_w) = kernel_size;
        let kernel = graph.parameter(
            &[out_channels, in_channels, k_h, k_w],
            Init::Kaiming,
            &format!("{name}_K"),
        )?;
        let bias = if use_bias {
            Some(graph.parameter(&[1, out_channels, 1, 1], Init::Zeros, &format!("{name}_b"))?)
        } else {
            None
        };
        let offset_channels = 2 * deformable_groups * k_h * k_w;
        let offset_kernel = graph.parameter(
            &[offset_channels, in_channels, k_h, k_w],
            Init::Zeros,
            &format!("{name}_offset_K"),
        )?;
        let offset_bias = graph.parameter(
            &[1, offset_channels, 1, 1],
            Init::Zeros,
            &format!("{name}_offset_b"),
        )?;
        let instance_id = graph.inner_mut().next_node_group_instance_id();
        Ok(Self {
            kernel,
            bias,
            offset_kernel,
            offset_bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            deformable_groups,
            name: name.to_string(),
            instance_id,
        })
    }

    pub fn forward(&self, x: impl IntoVar) -> Var {
        use std::rc::Rc;
        let x = x
            .into_var(&self.kernel.get_graph())
            .expect("DeformableConv2d 输入转换失败");
        let graph = x.get_graph();
        let desc = format!(
            "[?, {}, ?, ?] → [?, {}, ?, ?], kernel {}×{}, deformable_groups={}",
            self.in_channels,
            self.out_channels,
            self.kernel_size.0,
            self.kernel_size.1,
            self.deformable_groups
        );
        let guard = NodeGroupContext::for_layer(
            &x,
            "DeformableConv2d",
            self.instance_id,
            &self.name,
            &desc,
        );
        guard.tag_existing(&self.kernel);
        guard.tag_existing(&self.offset_kernel);
        guard.tag_existing(&self.offset_bias);
        if let Some(bias) = &self.bias {
            guard.tag_existing(bias);
        }

        let offset_conv = graph
            .inner_mut()
            .create_conv2d_node(
                vec![Rc::clone(x.node()), Rc::clone(self.offset_kernel.node())],
                self.stride,
                self.padding,
                self.dilation,
                Some(&format!("{}_offset_conv", self.name)),
            )
            .expect("DeformableConv2d offset conv 失败");
        let offset_conv = Var::new_with_rc_graph(offset_conv, &graph.inner_rc());
        let offset = graph
            .inner_mut()
            .create_add_node(
                vec![
                    Rc::clone(offset_conv.node()),
                    Rc::clone(self.offset_bias.node()),
                ],
                Some(&format!("{}_offset", self.name)),
            )
            .expect("DeformableConv2d offset bias add 失败");
        let offset = Var::new_with_rc_graph(offset, &graph.inner_rc());

        let deform = graph
            .inner_mut()
            .create_deformable_conv2d_node(
                vec![
                    Rc::clone(x.node()),
                    Rc::clone(self.kernel.node()),
                    Rc::clone(offset.node()),
                ],
                self.stride,
                self.padding,
                self.dilation,
                self.deformable_groups,
                Some(&format!("{}_deform", self.name)),
            )
            .expect("DeformableConv2d deform conv 失败");
        let deform = Var::new_with_rc_graph(deform, &graph.inner_rc());

        if let Some(bias) = &self.bias {
            let out = graph
                .inner_mut()
                .create_add_node(
                    vec![Rc::clone(deform.node()), Rc::clone(bias.node())],
                    Some(&format!("{}_out", self.name)),
                )
                .expect("DeformableConv2d bias add 失败");
            Var::new_with_rc_graph(out, &graph.inner_rc())
        } else {
            deform
        }
    }

    pub const fn kernel(&self) -> &Var {
        &self.kernel
    }

    pub const fn bias(&self) -> Option<&Var> {
        self.bias.as_ref()
    }

    pub const fn offset_kernel(&self) -> &Var {
        &self.offset_kernel
    }

    pub const fn offset_bias(&self) -> &Var {
        &self.offset_bias
    }
}

impl Module for DeformableConv2d {
    fn parameters(&self) -> Vec<Var> {
        let mut params = vec![
            self.kernel.clone(),
            self.offset_kernel.clone(),
            self.offset_bias.clone(),
        ];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }
}
