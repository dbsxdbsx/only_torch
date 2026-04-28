/*
 * @Author       : 老董
 * @Date         : 2026-04-28
 * @Description  : Deformable Conv2d 节点（v1：offset-only）
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::{GradResult, TraitNode, hash_dedup_params};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// 2D 可变形卷积节点。
///
/// 父节点：
/// - parents[0]: 输入 `[N, C_in, H, W]`
/// - parents[1]: 卷积核 `[C_out, C_in, kH, kW]`
/// - parents[2]: offset `[N, 2 * deformable_groups * kH * kW, H_out, W_out]`
#[derive(Clone)]
pub(crate) struct DeformableConv2d {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    fixed_shape: Vec<usize>,
    dynamic_shape: DynamicShape,
    supports_dynamic: bool,
    #[allow(dead_code)]
    parents_ids: Vec<NodeId>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    deformable_groups: usize,
}

#[derive(Clone, Copy)]
struct BilinearSample {
    value: f32,
    y_low: isize,
    x_low: isize,
    y_high: isize,
    x_high: isize,
    wy_low: f32,
    wy_high: f32,
    wx_low: f32,
    wx_high: f32,
}

impl DeformableConv2d {
    #[allow(dead_code)]
    pub(in crate::nn) const fn stride(&self) -> (usize, usize) {
        self.stride
    }

    #[allow(dead_code)]
    pub(in crate::nn) const fn padding(&self) -> (usize, usize) {
        self.padding
    }

    #[allow(dead_code)]
    pub(in crate::nn) const fn dilation(&self) -> (usize, usize) {
        self.dilation
    }

    #[allow(dead_code)]
    pub(in crate::nn) const fn deformable_groups(&self) -> usize {
        self.deformable_groups
    }

    pub(in crate::nn) fn new(
        parent_shapes: &[&[usize]],
        parent_dynamic_shapes: &[DynamicShape],
        parent_ids: Vec<NodeId>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        deformable_groups: usize,
    ) -> Result<Self, GraphError> {
        if parent_shapes.len() != 3 {
            return Err(GraphError::InvalidOperation(
                "DeformableConv2d 节点需要 3 个父节点：[输入, 卷积核, offset]".to_string(),
            ));
        }
        if deformable_groups == 0 {
            return Err(GraphError::InvalidOperation(
                "deformable_groups 必须大于 0".to_string(),
            ));
        }

        let input_shape = parent_shapes[0];
        let kernel_shape = parent_shapes[1];
        let offset_shape = parent_shapes[2];
        if input_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: input_shape.to_vec(),
                message: format!(
                    "DeformableConv2d 输入必须是 4D [batch, C_in, H, W]，得到 {input_shape:?}"
                ),
            });
        }
        if kernel_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: kernel_shape.to_vec(),
                message: format!(
                    "DeformableConv2d 卷积核必须是 4D [C_out, C_in, kH, kW]，得到 {kernel_shape:?}"
                ),
            });
        }
        if offset_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: offset_shape.to_vec(),
                message: format!(
                    "DeformableConv2d offset 必须是 4D [batch, 2*g*kH*kW, H_out, W_out]，得到 {offset_shape:?}"
                ),
            });
        }

        let (batch_size, input_c, input_h, input_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_channels, kernel_in_c, kernel_h, kernel_w) = (
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            kernel_shape[3],
        );
        if input_c != kernel_in_c {
            return Err(GraphError::ShapeMismatch {
                expected: vec![kernel_in_c],
                got: vec![input_c],
                message: format!("输入通道数 {input_c} 与卷积核输入通道数 {kernel_in_c} 不匹配"),
            });
        }
        if input_c % deformable_groups != 0 {
            return Err(GraphError::InvalidOperation(format!(
                "输入通道数 {input_c} 必须能被 deformable_groups={deformable_groups} 整除"
            )));
        }

        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let (dil_h, dil_w) = dilation;
        if stride_h == 0 || stride_w == 0 || dil_h == 0 || dil_w == 0 {
            return Err(GraphError::InvalidOperation(
                "stride 和 dilation 必须大于 0".to_string(),
            ));
        }
        let effective_kh = dil_h * (kernel_h - 1) + 1;
        let effective_kw = dil_w * (kernel_w - 1) + 1;
        let output_h = (input_h + 2 * pad_h - effective_kh) / stride_h + 1;
        let output_w = (input_w + 2 * pad_w - effective_kw) / stride_w + 1;
        if output_h == 0 || output_w == 0 {
            return Err(GraphError::InvalidOperation(format!(
                "DeformableConv2d 输出尺寸无效：输入 {input_h}x{input_w}，核 {kernel_h}x{kernel_w}，步长 {stride:?}，填充 {padding:?}，空洞 {dilation:?}"
            )));
        }

        let expected_offset = vec![
            batch_size,
            2 * deformable_groups * kernel_h * kernel_w,
            output_h,
            output_w,
        ];
        if offset_shape != expected_offset.as_slice() {
            return Err(GraphError::ShapeMismatch {
                expected: expected_offset,
                got: offset_shape.to_vec(),
                message: "DeformableConv2d offset 形状必须匹配 [batch, 2*g*kH*kW, H_out, W_out]"
                    .to_string(),
            });
        }

        let fixed_shape = vec![batch_size, out_channels, output_h, output_w];
        let parent_dyn = &parent_dynamic_shapes[0];
        let supports_dynamic = parent_dyn.has_dynamic_dims();
        let dynamic_shape = if supports_dynamic && parent_dyn.is_dynamic(0) {
            let mut dims: Vec<Option<usize>> = fixed_shape.iter().map(|&d| Some(d)).collect();
            dims[0] = None;
            DynamicShape::new(&dims)
        } else {
            DynamicShape::fixed(&fixed_shape)
        };

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
            parents_ids: parent_ids,
            in_channels: input_c,
            out_channels,
            kernel_size: (kernel_h, kernel_w),
            stride,
            padding,
            dilation,
            deformable_groups,
        })
    }

    fn sample(input: &Tensor, n: usize, c: usize, y: f32, x: f32) -> BilinearSample {
        let h = input.shape()[2] as isize;
        let w = input.shape()[3] as isize;
        let y_low = y.floor() as isize;
        let x_low = x.floor() as isize;
        let y_high = y_low + 1;
        let x_high = x_low + 1;
        let ly = y - y_low as f32;
        let lx = x - x_low as f32;
        let wy_low = 1.0 - ly;
        let wy_high = ly;
        let wx_low = 1.0 - lx;
        let wx_high = lx;
        let at = |yy: isize, xx: isize| -> f32 {
            if yy >= 0 && yy < h && xx >= 0 && xx < w {
                input[[n, c, yy as usize, xx as usize]]
            } else {
                0.0
            }
        };
        let value = at(y_low, x_low) * wy_low * wx_low
            + at(y_low, x_high) * wy_low * wx_high
            + at(y_high, x_low) * wy_high * wx_low
            + at(y_high, x_high) * wy_high * wx_high;
        BilinearSample {
            value,
            y_low,
            x_low,
            y_high,
            x_high,
            wy_low,
            wy_high,
            wx_low,
            wx_high,
        }
    }

    fn input_value(input: &Tensor, n: usize, c: usize, y: isize, x: isize) -> f32 {
        let h = input.shape()[2] as isize;
        let w = input.shape()[3] as isize;
        if y >= 0 && y < h && x >= 0 && x < w {
            input[[n, c, y as usize, x as usize]]
        } else {
            0.0
        }
    }

    fn add_input_grad(
        data: &mut [f32],
        shape: &[usize],
        n: usize,
        c: usize,
        y: isize,
        x: isize,
        v: f32,
    ) {
        let h = shape[2] as isize;
        let w = shape[3] as isize;
        if y >= 0 && y < h && x >= 0 && x < w {
            let idx = ((n * shape[1] + c) * shape[2] + y as usize) * shape[3] + x as usize;
            data[idx] += v;
        }
    }

    fn offset_index(
        &self,
        n: usize,
        c: usize,
        kh: usize,
        kw: usize,
        oh: usize,
        ow: usize,
        is_y: bool,
    ) -> usize {
        let (_, k_w) = self.kernel_size;
        let channels_per_group = self.in_channels / self.deformable_groups;
        let group = c / channels_per_group;
        let pair_index = group * self.kernel_size.0 * k_w + kh * k_w + kw;
        let offset_c = 2 * pair_index + usize::from(!is_y);
        let out_h = self.fixed_shape[2];
        let out_w = self.fixed_shape[3];
        ((n * (2 * self.deformable_groups * self.kernel_size.0 * k_w) + offset_c) * out_h + oh)
            * out_w
            + ow
    }

    fn data_index(shape: &[usize], n: usize, c: usize, h: usize, w: usize) -> usize {
        ((n * shape[1] + c) * shape[2] + h) * shape[3] + w
    }
}

impl TraitNode for DeformableConv2d {
    fn id(&self) -> NodeId {
        self.id.unwrap()
    }

    fn set_id(&mut self, id: NodeId) {
        self.id = Some(id);
    }

    fn name(&self) -> &str {
        self.name.as_ref().unwrap()
    }

    fn set_name(&mut self, name: &str) {
        self.name = Some(name.to_string());
    }

    fn value_expected_shape(&self) -> &[usize] {
        &self.fixed_shape
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        self.dynamic_shape.clone()
    }

    fn supports_dynamic_batch(&self) -> bool {
        self.supports_dynamic
    }

    fn dedup_fingerprint(&self) -> Option<u64> {
        Some(hash_dedup_params(&[
            self.in_channels as u64,
            self.out_channels as u64,
            self.kernel_size.0 as u64,
            self.kernel_size.1 as u64,
            self.stride.0 as u64,
            self.stride.1 as u64,
            self.padding.0 as u64,
            self.padding.1 as u64,
            self.dilation.0 as u64,
            self.dilation.1 as u64,
            self.deformable_groups as u64,
        ]))
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let input = parent_values[0];
        let kernel = parent_values[1];
        let offset = parent_values[2];
        let (batch, out_c, out_h, out_w) = (
            self.fixed_shape[0],
            self.fixed_shape[1],
            self.fixed_shape[2],
            self.fixed_shape[3],
        );
        let (k_h, k_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;
        let (dil_h, dil_w) = self.dilation;
        let mut out = vec![0.0; batch * out_c * out_h * out_w];

        for n in 0..batch {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;
                        for ic in 0..self.in_channels {
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let off_y_idx = self.offset_index(n, ic, kh, kw, oh, ow, true);
                                    let off_x_idx = self.offset_index(n, ic, kh, kw, oh, ow, false);
                                    let y = (oh * stride_h + kh * dil_h) as f32 - pad_h as f32
                                        + offset.data_as_slice()[off_y_idx];
                                    let x = (ow * stride_w + kw * dil_w) as f32 - pad_w as f32
                                        + offset.data_as_slice()[off_x_idx];
                                    let sample = Self::sample(input, n, ic, y, x).value;
                                    sum += kernel[[oc, ic, kh, kw]] * sample;
                                }
                            }
                        }
                        let idx = Self::data_index(&self.fixed_shape, n, oc, oh, ow);
                        out[idx] = sum;
                    }
                }
            }
        }
        self.value = Some(Tensor::new(&out, &self.fixed_shape));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let input = parent_values[0];
        let kernel = parent_values[1];
        let offset = parent_values[2];
        let (batch, out_c, out_h, out_w) = (
            self.fixed_shape[0],
            self.fixed_shape[1],
            self.fixed_shape[2],
            self.fixed_shape[3],
        );
        let (k_h, k_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;
        let (dil_h, dil_w) = self.dilation;

        match target_parent_index {
            0 => {
                let mut grad = vec![0.0; input.size()];
                for n in 0..batch {
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let go = upstream_grad[[n, oc, oh, ow]];
                                for ic in 0..self.in_channels {
                                    for kh in 0..k_h {
                                        for kw in 0..k_w {
                                            let off_y_idx =
                                                self.offset_index(n, ic, kh, kw, oh, ow, true);
                                            let off_x_idx =
                                                self.offset_index(n, ic, kh, kw, oh, ow, false);
                                            let y = (oh * stride_h + kh * dil_h) as f32
                                                - pad_h as f32
                                                + offset.data_as_slice()[off_y_idx];
                                            let x = (ow * stride_w + kw * dil_w) as f32
                                                - pad_w as f32
                                                + offset.data_as_slice()[off_x_idx];
                                            let sample = Self::sample(input, n, ic, y, x);
                                            let scale = go * kernel[[oc, ic, kh, kw]];
                                            Self::add_input_grad(
                                                &mut grad,
                                                input.shape(),
                                                n,
                                                ic,
                                                sample.y_low,
                                                sample.x_low,
                                                scale * sample.wy_low * sample.wx_low,
                                            );
                                            Self::add_input_grad(
                                                &mut grad,
                                                input.shape(),
                                                n,
                                                ic,
                                                sample.y_low,
                                                sample.x_high,
                                                scale * sample.wy_low * sample.wx_high,
                                            );
                                            Self::add_input_grad(
                                                &mut grad,
                                                input.shape(),
                                                n,
                                                ic,
                                                sample.y_high,
                                                sample.x_low,
                                                scale * sample.wy_high * sample.wx_low,
                                            );
                                            Self::add_input_grad(
                                                &mut grad,
                                                input.shape(),
                                                n,
                                                ic,
                                                sample.y_high,
                                                sample.x_high,
                                                scale * sample.wy_high * sample.wx_high,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(GradResult::Computed(Tensor::new(&grad, input.shape())))
            }
            1 => {
                let mut grad = vec![0.0; kernel.size()];
                for n in 0..batch {
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let go = upstream_grad[[n, oc, oh, ow]];
                                for ic in 0..self.in_channels {
                                    for kh in 0..k_h {
                                        for kw in 0..k_w {
                                            let off_y_idx =
                                                self.offset_index(n, ic, kh, kw, oh, ow, true);
                                            let off_x_idx =
                                                self.offset_index(n, ic, kh, kw, oh, ow, false);
                                            let y = (oh * stride_h + kh * dil_h) as f32
                                                - pad_h as f32
                                                + offset.data_as_slice()[off_y_idx];
                                            let x = (ow * stride_w + kw * dil_w) as f32
                                                - pad_w as f32
                                                + offset.data_as_slice()[off_x_idx];
                                            let idx = ((oc * self.in_channels + ic) * k_h + kh)
                                                * k_w
                                                + kw;
                                            grad[idx] +=
                                                go * Self::sample(input, n, ic, y, x).value;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(GradResult::Computed(Tensor::new(&grad, kernel.shape())))
            }
            2 => {
                let mut grad = vec![0.0; offset.size()];
                for n in 0..batch {
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let go = upstream_grad[[n, oc, oh, ow]];
                                for ic in 0..self.in_channels {
                                    for kh in 0..k_h {
                                        for kw in 0..k_w {
                                            let off_y_idx =
                                                self.offset_index(n, ic, kh, kw, oh, ow, true);
                                            let off_x_idx =
                                                self.offset_index(n, ic, kh, kw, oh, ow, false);
                                            let y = (oh * stride_h + kh * dil_h) as f32
                                                - pad_h as f32
                                                + offset.data_as_slice()[off_y_idx];
                                            let x = (ow * stride_w + kw * dil_w) as f32
                                                - pad_w as f32
                                                + offset.data_as_slice()[off_x_idx];
                                            let sample = Self::sample(input, n, ic, y, x);
                                            let x00 = Self::input_value(
                                                input,
                                                n,
                                                ic,
                                                sample.y_low,
                                                sample.x_low,
                                            );
                                            let x01 = Self::input_value(
                                                input,
                                                n,
                                                ic,
                                                sample.y_low,
                                                sample.x_high,
                                            );
                                            let x10 = Self::input_value(
                                                input,
                                                n,
                                                ic,
                                                sample.y_high,
                                                sample.x_low,
                                            );
                                            let x11 = Self::input_value(
                                                input,
                                                n,
                                                ic,
                                                sample.y_high,
                                                sample.x_high,
                                            );
                                            let dv_dy = (x10 - x00) * sample.wx_low
                                                + (x11 - x01) * sample.wx_high;
                                            let dv_dx = (x01 - x00) * sample.wy_low
                                                + (x11 - x10) * sample.wy_high;
                                            let scale = go * kernel[[oc, ic, kh, kw]];
                                            grad[off_y_idx] += scale * dv_dy;
                                            grad[off_x_idx] += scale * dv_dx;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(GradResult::Computed(Tensor::new(&grad, offset.shape())))
            }
            _ => Err(GraphError::InvalidOperation(format!(
                "DeformableConv2d 不存在父节点索引 {target_parent_index}"
            ))),
        }
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    fn grad_mut(&mut self) -> Option<&mut Tensor> {
        self.grad.as_mut()
    }

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }

    fn clear_value(&mut self) -> Result<(), GraphError> {
        self.value = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
