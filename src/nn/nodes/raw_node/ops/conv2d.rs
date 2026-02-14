/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : 2D 卷积节点（PyTorch 风格）
 *
 * 设计决策：
 * - 单节点处理多通道（PyTorch 风格），而非每通道独立节点（MatrixSlow 风格）
 * - Batch-First 格式：输入必须是 4D [batch, C_in, H, W]
 * - 输出格式：[batch, C_out, H', W']
 * - 单样本使用 batch=1，如 [1, C_in, H, W]
 * - 使用 Rayon 在 batch 维度并行加速
 *
 * 父节点：
 * - parents[0]: 输入数据
 * - parents[1]: 卷积核参数（Parameter 节点）
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use ndarray::Array2;
use rayon::prelude::*;

/// 2D 卷积节点
#[derive(Clone)]
pub(crate) struct Conv2d {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状（用于 `value_expected_shape`）
    fixed_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
    #[allow(dead_code)]
    parents_ids: Vec<NodeId>, // [input_id, kernel_id]

    // 卷积参数（保留供后续 NEAT 进化时使用）
    #[allow(dead_code)]
    in_channels: usize,
    #[allow(dead_code)]
    out_channels: usize,
    kernel_size: (usize, usize), // (kH, kW)
    stride: (usize, usize),      // (sH, sW)
    padding: (usize, usize),     // (pH, pW)

    // 缓存（用于反向传播）
    padded_input: Option<Tensor>, // 填充后的输入
    input_shape: Vec<usize>,      // 原始输入形状（用于梯度计算）
}

impl Conv2d {
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

    /// 从父节点形状信息创建 Conv2d 节点（核心实现）
    ///
    /// # 参数
    /// - `parent_shapes`: [输入形状, 卷积核形状]
    /// - `parent_dynamic_shapes`: 父节点的动态形状
    /// - `parent_ids`: 父节点 ID
    /// - `stride`: 步长 (sH, sW)
    /// - `padding`: 填充 (pH, pW)
    pub(in crate::nn) fn new(
        parent_shapes: &[&[usize]],
        parent_dynamic_shapes: &[DynamicShape],
        parent_ids: Vec<NodeId>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parent_shapes.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "Conv2d 节点需要 2 个父节点：[输入, 卷积核]".to_string(),
            ));
        }

        let input_shape = parent_shapes[0];
        let kernel_shape = parent_shapes[1];

        // 2. 验证卷积核形状：必须是 4D [C_out, C_in, kH, kW]
        if kernel_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: kernel_shape.to_vec(),
                message: format!("卷积核必须是 4D [C_out, C_in, kH, kW]，得到 {kernel_shape:?}"),
            });
        }

        let out_channels = kernel_shape[0];
        let in_channels = kernel_shape[1];
        let kernel_h = kernel_shape[2];
        let kernel_w = kernel_shape[3];

        // 3. 验证输入形状：必须是 4D [batch, C_in, H, W]
        if input_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: input_shape.to_vec(),
                message: format!(
                    "Conv2d 输入必须是 4D [batch, C_in, H, W]，得到 {input_shape:?}。单样本请使用 [1, C_in, H, W]"
                ),
            });
        }
        let (batch_size, input_c, input_h, input_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        // 4. 验证通道数匹配
        if input_c != in_channels {
            return Err(GraphError::ShapeMismatch {
                expected: vec![in_channels],
                got: vec![input_c],
                message: format!("输入通道数 {input_c} 与卷积核输入通道数 {in_channels} 不匹配"),
            });
        }

        // 5. 计算输出尺寸
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;

        let output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
        let output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

        if output_h == 0 || output_w == 0 {
            return Err(GraphError::InvalidOperation(format!(
                "卷积输出尺寸无效：输入 {input_h}x{input_w}，核 {kernel_h}x{kernel_w}，步长 {stride:?}，填充 {padding:?}"
            )));
        }

        // 6. 确定输出形状
        let fixed_shape = vec![batch_size, out_channels, output_h, output_w];

        // 7. 计算动态形状
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
            padded_input: None,
            input_shape: input_shape.to_vec(),
        })
    }

    /// 对输入进行零填充
    /// 输入必须是 4D [batch, C, H, W]
    fn pad_input(&self, input: &Tensor) -> Tensor {
        let (pad_h, pad_w) = self.padding;
        if pad_h == 0 && pad_w == 0 {
            return input.clone();
        }

        let input_shape = input.shape();
        let (batch_size, c, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let new_h = h + 2 * pad_h;
        let new_w = w + 2 * pad_w;
        let new_shape = vec![batch_size, c, new_h, new_w];
        let single_sample_size = c * new_h * new_w;

        // Rayon 并行处理每个 batch 样本
        let batch_results: Vec<Vec<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|bi| {
                let mut sample_data = vec![0.0f32; single_sample_size];
                for ci in 0..c {
                    for hi in 0..h {
                        for wi in 0..w {
                            let idx = ci * new_h * new_w + (hi + pad_h) * new_w + (wi + pad_w);
                            sample_data[idx] = input[[bi, ci, hi, wi]];
                        }
                    }
                }
                sample_data
            })
            .collect();

        let all_data: Vec<f32> = batch_results.into_iter().flatten().collect();
        Tensor::new(&all_data, &new_shape)
    }

    /// im2col：将单样本输入 [C_in, H, W] 展开为列矩阵 [out_h*out_w, C_in*k_h*k_w]
    ///
    /// 每个输出位置对应的感受野窗口被展开为一行，使卷积变为矩阵乘法：
    /// output = kernel_mat [out_c, C_in*k_h*k_w] × col^T [C_in*k_h*k_w, out_h*out_w]
    fn im2col(
        input: &Tensor,       // 单样本 [C_in, H, W]（已填充）
        b: usize,             // batch 索引（input 实际是 4D 但只取第 b 个）
        in_c: usize,
        k_h: usize,
        k_w: usize,
        out_h: usize,
        out_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Array2<f32> {
        let col_h = out_h * out_w;
        let col_w = in_c * k_h * k_w;
        let mut col = Array2::<f32>::zeros((col_h, col_w));

        for oh in 0..out_h {
            for ow in 0..out_w {
                let row = oh * out_w + ow;
                let h_start = oh * stride_h;
                let w_start = ow * stride_w;
                let mut col_idx = 0;
                for ic in 0..in_c {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            col[[row, col_idx]] = input[[b, ic, h_start + kh, w_start + kw]];
                            col_idx += 1;
                        }
                    }
                }
            }
        }
        col
    }

    /// 批量 im2col：对所有 batch 样本并行执行 im2col，垂直拼接为 [batch*spatial, col_w]
    ///
    /// 合并 N 次小 im2col 为一个大矩阵，使后续 GEMM 可一次完成。
    /// Rayon 并行处理各样本的 im2col，然后顺序拼接。
    fn batch_im2col(
        input: &Tensor,
        batch_size: usize,
        in_c: usize,
        k_h: usize,
        k_w: usize,
        out_h: usize,
        out_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Array2<f32> {
        let spatial = out_h * out_w;
        let col_w = in_c * k_h * k_w;

        // Rayon 并行 im2col
        let cols: Vec<Array2<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| Self::im2col(input, b, in_c, k_h, k_w, out_h, out_w, stride_h, stride_w))
            .collect();

        // 垂直拼接为 [batch*spatial, col_w]
        let mut all_data = Vec::with_capacity(batch_size * spatial * col_w);
        for col in &cols {
            all_data.extend_from_slice(col.as_slice().unwrap());
        }
        Array2::from_shape_vec((batch_size * spatial, col_w), all_data).unwrap()
    }

    /// 构建批量梯度矩阵：upstream_grad [batch, out_c, H', W'] → [out_c, batch*spatial]
    ///
    /// 将 [batch, out_c, spatial] 转置为 [out_c, batch*spatial]，
    /// 使用 copy_from_slice 按通道连续复制，高效利用内存带宽。
    fn build_batch_grad_matrix(
        upstream_grad: &Tensor,
        batch_size: usize,
        out_c: usize,
        spatial: usize,
    ) -> Array2<f32> {
        let flat = upstream_grad.flatten_view();
        let flat = flat.as_slice().unwrap();
        let sample_size = out_c * spatial;
        let total_spatial = batch_size * spatial;
        let mut data = vec![0.0f32; out_c * total_spatial];
        for oc in 0..out_c {
            for b in 0..batch_size {
                let src = b * sample_size + oc * spatial;
                let dst = oc * total_spatial + b * spatial;
                data[dst..dst + spatial].copy_from_slice(&flat[src..src + spatial]);
            }
        }
        Array2::from_shape_vec((out_c, total_spatial), data).unwrap()
    }

    /// col2im：im2col 的逆操作，将列矩阵累加回 [C_in, H, W] 形状
    ///
    /// 注意：有重叠区域时需要累加（而非覆盖），用于反向传播 dL/dX
    fn col2im(
        col: &Array2<f32>,    // [out_h*out_w, C_in*k_h*k_w]
        in_c: usize,
        in_h: usize,
        in_w: usize,
        k_h: usize,
        k_w: usize,
        out_h: usize,
        out_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Vec<f32> {
        let mut result = vec![0.0f32; in_c * in_h * in_w];

        for oh in 0..out_h {
            for ow in 0..out_w {
                let row = oh * out_w + ow;
                let h_start = oh * stride_h;
                let w_start = ow * stride_w;
                let mut col_idx = 0;
                for ic in 0..in_c {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let idx = ic * in_h * in_w + (h_start + kh) * in_w + (w_start + kw);
                            result[idx] += col[[row, col_idx]];
                            col_idx += 1;
                        }
                    }
                }
            }
        }
        result
    }

    /// im2col + GEMM 卷积（Rayon 批并行）
    ///
    /// 核心优化：将嵌套循环卷积转化为矩阵乘法
    /// kernel_mat [out_c, in_c*k_h*k_w] × col^T [in_c*k_h*k_w, out_h*out_w] → [out_c, out_h*out_w]
    fn convolve(&self, input: &Tensor, kernel: &Tensor) -> Tensor {
        let input_shape = input.shape();
        let (batch_size, in_c, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        let (out_c, _, k_h, k_w) = (
            kernel.shape()[0],
            kernel.shape()[1],
            kernel.shape()[2],
            kernel.shape()[3],
        );

        let (stride_h, stride_w) = self.stride;
        let out_h = (in_h - k_h) / stride_h + 1;
        let out_w = (in_w - k_w) / stride_w + 1;

        let output_shape = vec![batch_size, out_c, out_h, out_w];

        // 将 kernel [out_c, in_c, k_h, k_w] reshape 为 [out_c, in_c*k_h*k_w]
        let k_flat = kernel.flatten_view();
        let col_w = in_c * k_h * k_w;
        let kernel_mat = Array2::from_shape_vec(
            (out_c, col_w),
            k_flat.to_vec(),
        ).unwrap();

        // Rayon 并行计算每个 batch 样本
        let batch_results: Vec<Vec<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                // im2col: [out_h*out_w, col_w]
                let col = Self::im2col(input, b, in_c, k_h, k_w, out_h, out_w, stride_h, stride_w);
                // GEMM: kernel_mat [out_c, col_w] × col^T [col_w, out_h*out_w] → [out_c, out_h*out_w]
                let result = kernel_mat.dot(&col.t());
                // result 是 [out_c, out_h*out_w]，展平为行优先顺序（与 NCHW 一致）
                result.iter().copied().collect()
            })
            .collect();

        let all_data: Vec<f32> = batch_results.into_iter().flatten().collect();
        Tensor::new(&all_data, &output_shape)
    }
}

impl TraitNode for Conv2d {
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
        ]))
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let input = parent_values[0];
        let kernel = parent_values[1];
        // 填充输入
        let padded = self.pad_input(input);
        self.padded_input = Some(padded.clone());
        self.input_shape = input.shape().to_vec();
        // 执行卷积
        self.value = Some(self.convolve(&padded, kernel));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    // ========== VJP 模式 ==========

    /// 计算 Batch 梯度（批量 GEMM 优化版本）
    ///
    /// 对于 Y = conv(X, K):
    /// - dL/dX: 一次大 GEMM 替代 N 次小 GEMM，再并行 col2im
    /// - dL/dK: 批量 im2col 拼接后一次大 GEMM，自然包含 batch 求和
    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let padded_input = self
            .padded_input
            .as_ref()
            .ok_or_else(|| GraphError::ComputationError("缺少填充后的输入缓存".to_string()))?;

        let kernel = parent_values
            .get(1)
            .ok_or_else(|| GraphError::ComputationError("Conv2D 梯度计算需要卷积核".to_string()))?;

        let grad_shape = upstream_grad.shape();
        let (batch_size, out_c, out_h, out_w) =
            (grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]);

        let (k_h, k_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;

        let padded_shape = padded_input.shape();
        let in_c = padded_shape[1];
        let spatial = out_h * out_w;
        let col_w = in_c * k_h * k_w;

        // kernel 展平为 [out_c, col_w]
        let k_flat = kernel.flatten_view();
        let kernel_mat =
            Array2::from_shape_vec((out_c, col_w), k_flat.to_vec()).unwrap();

        if target_parent_index == 0 {
            // ========== dL/dX（对输入的梯度）==========
            // 批量化：一次大 GEMM 替代 N 次小 GEMM
            //
            // all_grads^T [batch*spatial, out_c] × kernel [out_c, col_w]
            //   → all_dx [batch*spatial, col_w]
            //
            // 结果中每个 batch 样本对应连续的 [spatial, col_w] 块，
            // 直接用于 col2im，无需转置。

            let orig_input_shape = &self.input_shape;
            let (orig_in_h, orig_in_w) = (orig_input_shape[2], orig_input_shape[3]);
            let padded_h = orig_in_h + 2 * pad_h;
            let padded_w = orig_in_w + 2 * pad_w;

            // 构建 [out_c, batch*spatial] 然后转置参与 GEMM
            let all_grads = Self::build_batch_grad_matrix(
                upstream_grad, batch_size, out_c, spatial,
            );

            // 一次大 GEMM: [batch*spatial, out_c] × [out_c, col_w] → [batch*spatial, col_w]
            let all_dx = all_grads.t().dot(&kernel_mat);
            let all_dx_slice = all_dx.as_slice().unwrap();
            let row_size = col_w;

            // 分割为每个 batch 样本并 col2im（Rayon 并行）
            let batch_results: Vec<Vec<f32>> = (0..batch_size)
                .into_par_iter()
                .map(|b| {
                    // 连续切片: [spatial, col_w]
                    let start = b * spatial * row_size;
                    let end = start + spatial * row_size;
                    let dx_col_t = Array2::from_shape_vec(
                        (spatial, col_w),
                        all_dx_slice[start..end].to_vec(),
                    )
                    .unwrap();

                    // col2im 映射到填充后的输入空间
                    let padded_grad = Self::col2im(
                        &dx_col_t, in_c, padded_h, padded_w, k_h, k_w, out_h, out_w,
                        stride_h, stride_w,
                    );

                    // 裁剪 padding 区域
                    if pad_h == 0 && pad_w == 0 {
                        padded_grad
                    } else {
                        let mut result = Vec::with_capacity(in_c * orig_in_h * orig_in_w);
                        for ic in 0..in_c {
                            for ih in 0..orig_in_h {
                                for iw in 0..orig_in_w {
                                    let idx = ic * padded_h * padded_w
                                        + (ih + pad_h) * padded_w
                                        + (iw + pad_w);
                                    result.push(padded_grad[idx]);
                                }
                            }
                        }
                        result
                    }
                })
                .collect();

            let all_data: Vec<f32> = batch_results.into_iter().flatten().collect();
            Ok(GradResult::Computed(Tensor::new(&all_data, orig_input_shape)))
        } else {
            // ========== dL/dK（对卷积核的梯度）==========
            // 批量化：将 N 个 im2col 矩阵垂直拼接，一次大 GEMM 自然包含 batch 求和
            //
            // all_grads [out_c, batch*spatial] × all_cols [batch*spatial, col_w]
            //   → kernel_grad [out_c, col_w]

            let kernel_shape = kernel.shape();

            // 批量 im2col: [batch*spatial, col_w]
            let all_cols = Self::batch_im2col(
                padded_input, batch_size, in_c, k_h, k_w, out_h, out_w, stride_h, stride_w,
            );

            // 批量梯度矩阵: [out_c, batch*spatial]
            let all_grads = Self::build_batch_grad_matrix(
                upstream_grad, batch_size, out_c, spatial,
            );

            // 一次大 GEMM（自然包含 batch 维度求和）
            let kernel_grad = all_grads.dot(&all_cols);

            Ok(GradResult::Computed(Tensor::new(
                kernel_grad.as_slice().unwrap(),
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
