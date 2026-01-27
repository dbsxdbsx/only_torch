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
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
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
    supports_dynamic: bool,
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
    pub(in crate::nn) const fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// 获取填充
    pub(in crate::nn) const fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// 创建 Conv2d 节点
    ///
    /// # 参数
    /// - `parents`: [输入节点, 卷积核节点]
    /// - `stride`: 步长 (sH, sW)
    /// - `padding`: 填充 (pH, pW)
    ///
    /// # 输入形状约定
    /// - 输入: [`C_in`, H, W] 或 [batch, `C_in`, H, W]
    /// - 卷积核: [`C_out`, `C_in`, kH, kW]
    pub(crate) fn new(
        parents: &[&NodeHandle],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "Conv2d 节点需要 2 个父节点：[输入, 卷积核]".to_string(),
            ));
        }

        let input_shape = parents[0].value_expected_shape();
        let kernel_shape = parents[1].value_expected_shape();

        // 2. 验证卷积核形状：必须是 4D [C_out, C_in, kH, kW]
        if kernel_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0], // 占位
                got: kernel_shape.to_vec(),
                message: format!("卷积核必须是 4D [C_out, C_in, kH, kW]，得到 {kernel_shape:?}"),
            });
        }

        let out_channels = kernel_shape[0];
        let in_channels = kernel_shape[1];
        let kernel_h = kernel_shape[2];
        let kernel_w = kernel_shape[3];

        // 3. 验证输入形状：必须是 4D [batch, C_in, H, W]（Batch-First）
        if input_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0], // 占位
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

        // 6. 确定输出形状：始终是 4D [batch, C_out, H', W']
        let fixed_shape = vec![batch_size, out_channels, output_h, output_w];

        // 7. 计算动态形状
        let parent = &parents[0];
        let parent_dyn = parent.dynamic_expected_shape();
        let supports_dynamic = parent.supports_dynamic_batch();

        // 如果父节点第一维是动态的，输出也保持第一维动态
        let dynamic_shape = if supports_dynamic && parent_dyn.is_dynamic(0) {
            let mut dims: Vec<Option<usize>> = fixed_shape.iter().map(|&d| Some(d)).collect();
            dims[0] = None; // 第一维动态
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
            parents_ids: vec![parents[0].id(), parents[1].id()],
            in_channels,
            out_channels,
            kernel_size: (kernel_h, kernel_w),
            stride,
            padding,
            padded_input: None,
            input_shape: input_shape.to_vec(),
        })
    }

    /// 对输入进行零填充（Rayon 并行版本）
    /// 输入必须是 4D [batch, C, H, W]
    fn pad_input(&self, input: &Tensor) -> Tensor {
        let (pad_h, pad_w) = self.padding;
        if pad_h == 0 && pad_w == 0 {
            return input.clone();
        }

        let input_shape = input.shape();
        let (batch_size, c, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
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

        // 合并结果
        let all_data: Vec<f32> = batch_results.into_iter().flatten().collect();
        Tensor::new(&all_data, &new_shape)
    }

    /// 执行卷积运算（Rayon 并行版本）
    /// 输入必须是 4D [batch, C_in, H, W]
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

        // 输出形状：始终是 4D [batch, C_out, H', W']
        let output_shape = vec![batch_size, out_c, out_h, out_w];
        let single_sample_size = out_c * out_h * out_w;

        // Rayon 并行计算每个 batch 样本
        let batch_results: Vec<Vec<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut sample_data = vec![0.0f32; single_sample_size];
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0f32;
                            let h_start = oh * stride_h;
                            let w_start = ow * stride_w;

                            for ic in 0..in_c {
                                for kh in 0..k_h {
                                    for kw in 0..k_w {
                                        let input_val =
                                            input[[b, ic, h_start + kh, w_start + kw]];
                                        sum += input_val * kernel[[oc, ic, kh, kw]];
                                    }
                                }
                            }
                            let idx = oc * out_h * out_w + oh * out_w + ow;
                            sample_data[idx] = sum;
                        }
                    }
                }
                sample_data
            })
            .collect();

        // 合并结果
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

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 获取输入和卷积核
        let input = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的输入父{}没有值",
                self.display_node(),
                parents[0]
            ))
        })?;

        let kernel = parents[1].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的卷积核父{}没有值",
                self.display_node(),
                parents[1]
            ))
        })?;

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

    /// 计算 Batch 梯度（Rayon 并行版本）
    ///
    /// 对于 Y = conv(X, K):
    /// - dL/dX: 使用转置卷积（反卷积）
    /// - dL/dK: 使用输入和上游梯度的相关运算
    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let padded_input = self
            .padded_input
            .as_ref()
            .ok_or_else(|| GraphError::ComputationError("缺少填充后的输入缓存".to_string()))?;

        let kernel = if target_parent.id() == self.parents_ids[0] {
            assistant_parent
                .ok_or_else(|| GraphError::ComputationError("计算输入梯度需要卷积核".to_string()))?
                .value()
                .ok_or_else(|| GraphError::ComputationError("卷积核没有值".to_string()))?
        } else {
            target_parent
                .value()
                .ok_or_else(|| GraphError::ComputationError("卷积核没有值".to_string()))?
        };

        // 输入必须是 4D [batch, C_out, H', W']
        let grad_shape = upstream_grad.shape();
        let (batch_size, out_c, out_h, out_w) = (
            grad_shape[0],
            grad_shape[1],
            grad_shape[2],
            grad_shape[3],
        );

        let (k_h, k_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;

        let padded_shape = padded_input.shape();
        let in_c = padded_shape[1];

        if target_parent.id() == self.parents_ids[0] {
            // ========== 计算 dL/dX（对输入的梯度）==========
            let orig_input_shape = &self.input_shape;
            let (orig_in_h, orig_in_w) = (orig_input_shape[2], orig_input_shape[3]);
            let single_sample_size = in_c * orig_in_h * orig_in_w;

            // Rayon 并行处理每个 batch 样本
            let batch_results: Vec<Vec<f32>> = (0..batch_size)
                .into_par_iter()
                .map(|b| {
                    let mut sample_grad = vec![0.0f32; single_sample_size];

                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let grad_val = upstream_grad[[b, oc, oh, ow]];
                                let h_start = oh * stride_h;
                                let w_start = ow * stride_w;

                                for ic in 0..in_c {
                                    for kh in 0..k_h {
                                        for kw in 0..k_w {
                                            let orig_h = (h_start + kh) as isize - pad_h as isize;
                                            let orig_w = (w_start + kw) as isize - pad_w as isize;

                                            if orig_h >= 0
                                                && orig_h < orig_in_h as isize
                                                && orig_w >= 0
                                                && orig_w < orig_in_w as isize
                                            {
                                                let orig_h = orig_h as usize;
                                                let orig_w = orig_w as usize;
                                                let idx = ic * orig_in_h * orig_in_w
                                                    + orig_h * orig_in_w
                                                    + orig_w;
                                                sample_grad[idx] +=
                                                    grad_val * kernel[[oc, ic, kh, kw]];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    sample_grad
                })
                .collect();

            let all_data: Vec<f32> = batch_results.into_iter().flatten().collect();
            Ok(Tensor::new(&all_data, orig_input_shape))
        } else {
            // ========== 计算 dL/dK（对卷积核的梯度）==========
            // dK 需要跨 batch 累加，使用 map-reduce 模式
            let kernel_shape = kernel.shape();
            let kernel_size = out_c * in_c * k_h * k_w;

            // Rayon 并行 + reduce 累加
            let batch_kernel_grads: Vec<Vec<f32>> = (0..batch_size)
                .into_par_iter()
                .map(|b| {
                    let mut sample_kernel_grad = vec![0.0f32; kernel_size];

                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let grad_val = upstream_grad[[b, oc, oh, ow]];
                                let h_start = oh * stride_h;
                                let w_start = ow * stride_w;

                                for ic in 0..in_c {
                                    for kh in 0..k_h {
                                        for kw in 0..k_w {
                                            let input_val =
                                                padded_input[[b, ic, h_start + kh, w_start + kw]];
                                            let idx = oc * in_c * k_h * k_w
                                                + ic * k_h * k_w
                                                + kh * k_w
                                                + kw;
                                            sample_kernel_grad[idx] += grad_val * input_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    sample_kernel_grad
                })
                .collect();

            // Reduce: 累加所有 batch 样本的梯度
            let mut total_kernel_grad = vec![0.0f32; kernel_size];
            for sample_grad in batch_kernel_grads {
                for (i, g) in sample_grad.into_iter().enumerate() {
                    total_kernel_grad[i] += g;
                }
            }

            Ok(Tensor::new(&total_kernel_grad, kernel_shape))
        }
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
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
