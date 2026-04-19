/*
 * @Author       : 老董
 * @Date         : 2026-04-19
 * @Description  : 2D 转置卷积节点（ConvTranspose2d / 反卷积，PyTorch 风格）
 *
 * 设计决策：
 * - 单节点处理多通道（PyTorch 风格）
 * - Batch-First：输入 [batch, C_in, H, W]，输出 [batch, C_out, H_out, W_out]
 * - 卷积核布局 [C_in, C_out, kH, kW]（与 Conv2d 的 [C_out, C_in, kH, kW] 在通道维上转置）
 * - 前向：GEMM + col2im（scatter-add）；反向：col2im 的伴随 + 与 Conv2d 对称的 GEMM
 * - 使用 Rayon 在 batch 维度并行
 *
 * 父节点：
 * - parents[0]: 输入
 * - parents[1]: 卷积核（Parameter）
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use ndarray::Array2;
use rayon::prelude::*;

/// 2D 转置卷积节点
#[derive(Clone)]
pub(crate) struct ConvTranspose2d {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    fixed_shape: Vec<usize>,
    dynamic_shape: DynamicShape,
    #[allow(dead_code)]
    supports_dynamic: bool,
    #[allow(dead_code)]
    parents_ids: Vec<NodeId>,
    #[allow(dead_code)]
    in_channels: usize,
    #[allow(dead_code)]
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    /// 反向传播用：最近一次前向的输入形状 [N, C_in, H_in, W_in]
    cached_input_shape: Vec<usize>,
}

impl ConvTranspose2d {
    /// 获取步长
    #[allow(dead_code)]
    pub(in crate::nn) const fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// 获取填充
    #[allow(dead_code)]
    pub(in crate::nn) const fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// 获取输出侧额外填充（output_padding）
    #[allow(dead_code)]
    pub(in crate::nn) const fn output_padding(&self) -> (usize, usize) {
        self.output_padding
    }

    /// 从父节点形状创建 ConvTranspose2d 节点
    ///
    /// # 参数
    /// - `parent_shapes`: [输入形状, 卷积核形状]
    /// - `parent_dynamic_shapes`: 父节点动态形状
    /// - `parent_ids`: 父节点 ID
    /// - `stride`: 步长 (sH, sW)
    /// - `padding`: 填充 (pH, pW)
    /// - `output_padding`: 输出侧额外尺寸 (opH, opW)
    pub(in crate::nn) fn new(
        parent_shapes: &[&[usize]],
        parent_dynamic_shapes: &[DynamicShape],
        parent_ids: Vec<NodeId>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    ) -> Result<Self, GraphError> {
        if parent_shapes.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "ConvTranspose2d 节点需要 2 个父节点：[输入, 卷积核]".to_string(),
            ));
        }

        let input_shape = parent_shapes[0];
        let kernel_shape = parent_shapes[1];

        // 卷积核 [C_in, C_out, kH, kW]
        if kernel_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: kernel_shape.to_vec(),
                message: format!(
                    "转置卷积核必须是 4D [C_in, C_out, kH, kW]，得到 {kernel_shape:?}"
                ),
            });
        }

        let in_channels = kernel_shape[0];
        let out_channels = kernel_shape[1];
        let kernel_h = kernel_shape[2];
        let kernel_w = kernel_shape[3];

        if input_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: input_shape.to_vec(),
                message: format!(
                    "ConvTranspose2d 输入必须是 4D [batch, C_in, H, W]，得到 {input_shape:?}。单样本请使用 [1, C_in, H, W]"
                ),
            });
        }

        let (batch_size, input_c, input_h, input_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        if input_c != in_channels {
            return Err(GraphError::ShapeMismatch {
                expected: vec![in_channels],
                got: vec![input_c],
                message: format!("输入通道数 {input_c} 与卷积核 C_in={in_channels} 不匹配"),
            });
        }

        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let (op_h, op_w) = output_padding;

        if stride_h == 0 || stride_w == 0 {
            return Err(GraphError::InvalidOperation(
                "转置卷积步长必须大于 0".to_string(),
            ));
        }

        // H_out = (H_in - 1) * s - 2p + k + output_padding（与 PyTorch dilation=1 一致）
        let h_sum = (input_h - 1) * stride_h + kernel_h + op_h;
        let w_sum = (input_w - 1) * stride_w + kernel_w + op_w;
        let output_h = h_sum.saturating_sub(2 * pad_h);
        let output_w = w_sum.saturating_sub(2 * pad_w);

        if output_h == 0 || output_w == 0 {
            return Err(GraphError::InvalidOperation(format!(
                "转置卷积输出尺寸无效：输入 {input_h}x{input_w}，核 {kernel_h}x{kernel_w}，步长 {stride:?}，填充 {padding:?}，output_padding {output_padding:?}"
            )));
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
            in_channels,
            out_channels,
            kernel_size: (kernel_h, kernel_w),
            stride,
            padding,
            output_padding,
            cached_input_shape: input_shape.to_vec(),
        })
    }

    /// 将单样本输入 [C_in, H, W] 展平为矩阵 [C_in, H*W]（列主序：列 = 空间展平）
    fn input_batch_matrix(input: &Tensor, b: usize, in_c: usize, h_in: usize, w_in: usize) -> Array2<f32> {
        let hw = h_in * w_in;
        let mut mat = Array2::<f32>::zeros((in_c, hw));
        for ic in 0..in_c {
            for hi in 0..h_in {
                for wi in 0..w_in {
                    let col = hi * w_in + wi;
                    mat[[ic, col]] = input[[b, ic, hi, wi]];
                }
            }
        }
        mat
    }

    /// 前向 col2im：将 [H_in*W_in, C_out*kH*kW] 按转置卷积几何 scatter-add 到 [C_out, H_out, W_out]
    fn col2im_transpose_forward(
        col_t: &Array2<f32>,
        c_out: usize,
        h_out: usize,
        w_out: usize,
        k_h: usize,
        k_w: usize,
        h_in: usize,
        w_in: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; c_out * h_out * w_out];
        for ih in 0..h_in {
            for iw in 0..w_in {
                let row = ih * w_in + iw;
                let mut col_idx = 0usize;
                for oc in 0..c_out {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let oh = ih as isize * stride_h as isize - pad_h as isize + kh as isize;
                            let ow = iw as isize * stride_w as isize - pad_w as isize + kw as isize;
                            if oh >= 0
                                && ow >= 0
                                && oh < h_out as isize
                                && ow < w_out as isize
                            {
                                let idx = oc * h_out * w_out
                                    + oh as usize * w_out
                                    + ow as usize;
                                out[idx] += col_t[[row, col_idx]];
                            }
                            col_idx += 1;
                        }
                    }
                }
            }
        }
        out
    }

    /// col2im 的伴随：将上游梯度 [C_out, H_out, W_out] 分配回 [H_in*W_in, C_out*kH*kW]
    fn col2im_transpose_adjoint(
        d_y: &Tensor,
        b: usize,
        c_out: usize,
        h_out: usize,
        w_out: usize,
        k_h: usize,
        k_w: usize,
        h_in: usize,
        w_in: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Array2<f32> {
        let mut d_col = Array2::<f32>::zeros((h_in * w_in, c_out * k_h * k_w));
        for ih in 0..h_in {
            for iw in 0..w_in {
                let row = ih * w_in + iw;
                let mut col_idx = 0usize;
                for oc in 0..c_out {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let oh =
                                ih as isize * stride_h as isize - pad_h as isize + kh as isize;
                            let ow =
                                iw as isize * stride_w as isize - pad_w as isize + kw as isize;
                            if oh >= 0
                                && ow >= 0
                                && oh < h_out as isize
                                && ow < w_out as isize
                            {
                                d_col[[row, col_idx]] += d_y[[b, oc, oh as usize, ow as usize]];
                            }
                            col_idx += 1;
                        }
                    }
                }
            }
        }
        d_col
    }

    /// 转置卷积前向（Rayon 批并行）
    fn conv_transpose(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        h_out: usize,
        w_out: usize,
    ) -> Tensor {
        let input_shape = input.shape();
        let (batch_size, in_c, h_in, w_in) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        let (k_h, k_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;
        let out_c = self.out_channels;

        let m = out_c * k_h * k_w;
        let k_flat = kernel.flatten_view();
        let kernel_mat = Array2::from_shape_vec((in_c, m), k_flat.to_vec()).unwrap();

        let output_shape = vec![batch_size, out_c, h_out, w_out];

        let batch_results: Vec<Vec<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let input_b = Self::input_batch_matrix(input, b, in_c, h_in, w_in);
                let col = kernel_mat.t().dot(&input_b);
                let col_t = col.t().to_owned();
                Self::col2im_transpose_forward(
                    &col_t,
                    out_c,
                    h_out,
                    w_out,
                    k_h,
                    k_w,
                    h_in,
                    w_in,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                )
            })
            .collect();

        let all_data: Vec<f32> = batch_results.into_iter().flatten().collect();
        Tensor::new(&all_data, &output_shape)
    }
}

impl TraitNode for ConvTranspose2d {
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
        use crate::nn::nodes::raw_node::hash_dedup_params;
        Some(hash_dedup_params(&[
            self.in_channels as u64,
            self.out_channels as u64,
            self.kernel_size.0 as u64,
            self.kernel_size.1 as u64,
            self.stride.0 as u64,
            self.stride.1 as u64,
            self.padding.0 as u64,
            self.padding.1 as u64,
            self.output_padding.0 as u64,
            self.output_padding.1 as u64,
        ]))
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let input = parent_values[0];
        let kernel = parent_values[1];
        self.cached_input_shape = input.shape().to_vec();
        let h_out = self.fixed_shape[2];
        let w_out = self.fixed_shape[3];
        let result = self.conv_transpose(input, kernel, h_out, w_out);
        self.value = Some(result);
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// 转置卷积 VJP（batch 维 Rayon 并行）
    ///
    /// - dL/dX：`kernel_mat @ col2im^*(dL/dY)`，等价于对上游梯度做伴随 scatter 再与核矩阵相乘
    /// - dL/dK：`sum_b input_b @ d_col_b^T`，与线性段 `COL = K^T X` 的链式法则一致
    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let input = parent_values
            .first()
            .ok_or_else(|| GraphError::ComputationError("转置卷积梯度计算需要输入".to_string()))?;
        let kernel = parent_values
            .get(1)
            .ok_or_else(|| GraphError::ComputationError("转置卷积梯度计算需要卷积核".to_string()))?;

        let orig_input_shape = &self.cached_input_shape;
        let (batch_size, in_c, h_in, w_in) = (
            orig_input_shape[0],
            orig_input_shape[1],
            orig_input_shape[2],
            orig_input_shape[3],
        );

        let grad_shape = upstream_grad.shape();
        let (g_batch, out_c, out_h, out_w) =
            (grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]);
        if g_batch != batch_size || out_c != self.out_channels {
            return Err(GraphError::ShapeMismatch {
                expected: vec![batch_size, self.out_channels, out_h, out_w],
                got: grad_shape.to_vec(),
                message: "上游梯度形状与转置卷积输出不一致".to_string(),
            });
        }

        let (k_h, k_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;
        let m = out_c * k_h * k_w;

        let k_flat = kernel.flatten_view();
        let kernel_mat = Array2::from_shape_vec((in_c, m), k_flat.to_vec()).unwrap();

        if target_parent_index == 0 {
            let batch_results: Vec<Vec<f32>> = (0..batch_size)
                .into_par_iter()
                .map(|b| {
                    let d_col = Self::col2im_transpose_adjoint(
                        upstream_grad,
                        b,
                        out_c,
                        out_h,
                        out_w,
                        k_h,
                        k_w,
                        h_in,
                        w_in,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                    );
                    let d_in = kernel_mat.dot(&d_col.t());
                    d_in.iter().copied().collect::<Vec<f32>>()
                })
                .collect();

            let all_data: Vec<f32> = batch_results.into_iter().flatten().collect();
            Ok(GradResult::Computed(Tensor::new(&all_data, orig_input_shape)))
        } else {
            let kernel_shape = kernel.shape();

            let kernel_grad_data: Vec<f32> = {
                let kernel_grad = (0..batch_size)
                    .into_par_iter()
                    .map(|b| {
                        let input_b = Self::input_batch_matrix(input, b, in_c, h_in, w_in);
                        let d_col = Self::col2im_transpose_adjoint(
                            upstream_grad,
                            b,
                            out_c,
                            out_h,
                            out_w,
                            k_h,
                            k_w,
                            h_in,
                            w_in,
                            stride_h,
                            stride_w,
                            pad_h,
                            pad_w,
                        );
                        input_b.dot(&d_col)
                    })
                    .reduce(
                        || Array2::<f32>::zeros((in_c, m)),
                        |mut acc, g| {
                            acc += &g;
                            acc
                        },
                    );

                kernel_grad.as_slice().unwrap().to_vec()
            };

            Ok(GradResult::Computed(Tensor::new(
                &kernel_grad_data,
                kernel_shape,
            )))
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

#[cfg(test)]
mod tests {
    use super::ConvTranspose2d;
    use crate::nn::nodes::raw_node::TraitNode;
    use crate::nn::shape::DynamicShape;
    use crate::nn::nodes::NodeId;
    use crate::nn::nodes::raw_node::GradResult;
    use crate::tensor::Tensor;

    /// 与 PyTorch 对照：全 1 输入、全 1 核、stride=1、无 padding
    #[test]
    fn forward_matches_pytorch_all_ones() {
        let in_sh = [1usize, 1, 2, 2];
        let ker_sh = [1usize, 1, 2, 2];
        let parent_shapes: Vec<&[usize]> = vec![&in_sh, &ker_sh];
        let parent_ds = vec![
            DynamicShape::fixed(&in_sh),
            DynamicShape::fixed(&ker_sh),
        ];
        let mut op = ConvTranspose2d::new(
            &parent_shapes,
            &parent_ds,
            vec![NodeId(0), NodeId(1)],
            (1, 1),
            (0, 0),
            (0, 0),
        )
        .unwrap();
        let input = Tensor::ones(&[1, 1, 2, 2]);
        let kernel = Tensor::ones(&[1, 1, 2, 2]);
        op.calc_value_by_parents(&[&input, &kernel]).unwrap();
        let y = op.value().unwrap();
        let expected = [[1f32, 2., 1.], [2., 4., 2.], [1., 2., 1.]];
        for i in 0..3 {
            for j in 0..3 {
                assert!((y[[0, 0, i, j]] - expected[i][j]).abs() < 1e-5);
            }
        }
    }

    /// 与 PyTorch 对照：sum(y) 反向，d_input / d_weight 全为 4
    #[test]
    fn backward_matches_pytorch_sum_upstream() {
        let in_sh = [1usize, 1, 2, 2];
        let ker_sh = [1usize, 1, 2, 2];
        let parent_shapes: Vec<&[usize]> = vec![&in_sh, &ker_sh];
        let parent_ds = vec![
            DynamicShape::fixed(&in_sh),
            DynamicShape::fixed(&ker_sh),
        ];
        let mut op = ConvTranspose2d::new(
            &parent_shapes,
            &parent_ds,
            vec![NodeId(0), NodeId(1)],
            (1, 1),
            (0, 0),
            (0, 0),
        )
        .unwrap();
        let input = Tensor::ones(&[1, 1, 2, 2]);
        let kernel = Tensor::ones(&[1, 1, 2, 2]);
        op.calc_value_by_parents(&[&input, &kernel]).unwrap();
        let g = Tensor::ones(&[1, 1, 3, 3]);
        let gi = op
            .calc_grad_to_parent(0, &[&input, &kernel], &g)
            .unwrap();
        let gk = op
            .calc_grad_to_parent(1, &[&input, &kernel], &g)
            .unwrap();
        let gi = match gi {
            GradResult::Computed(t) => t,
            _ => panic!("expected computed grad"),
        };
        let gk = match gk {
            GradResult::Computed(t) => t,
            _ => panic!("expected computed grad"),
        };
        for &v in gi.flatten_view().iter() {
            assert!((v - 4.0).abs() < 1e-4);
        }
        for &v in gk.flatten_view().iter() {
            assert!((v - 4.0).abs() < 1e-4);
        }
    }
}
