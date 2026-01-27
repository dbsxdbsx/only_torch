/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : 2D 最大池化节点（PyTorch 风格）
 *
 * 设计决策：
 * - 记录最大值位置用于反向传播（稀疏梯度）
 * - Batch-First 格式：输入必须是 4D [batch, C, H, W]
 * - 输出格式：[batch, C, H', W']
 * - 单样本使用 batch=1，如 [1, C, H, W]
 * - 使用 Rayon 在 batch 维度并行加速
 *
 * 父节点：
 * - parents[0]: 输入数据
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use rayon::prelude::*;

/// 2D 最大池化节点
#[derive(Clone)]
pub(crate) struct MaxPool2d {
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

    // 池化参数
    kernel_size: (usize, usize), // (kH, kW)
    stride: (usize, usize),      // (sH, sW)

    // 缓存（用于反向传播）
    // 存储每个输出位置对应的最大值在输入中的索引
    // 形状与输出相同，值为展平后的输入索引
    max_indices: Option<Tensor>,
    input_shape: Vec<usize>, // 原始输入形状
}

impl MaxPool2d {
    /// 创建 `MaxPool2d` 节点
    ///
    /// # 参数
    /// - `parents`: [输入节点]
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)，默认等于 `kernel_size`
    ///
    /// # 输入形状约定
    /// - 输入必须是 4D: [batch, C, H, W]（Batch-First）
    /// - 单样本使用 [1, C, H, W]
    pub(crate) fn new(
        parents: &[&NodeHandle],
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "MaxPool2d 节点需要 1 个父节点".to_string(),
            ));
        }

        let input_shape = parents[0].value_expected_shape();

        // 2. 验证输入形状：必须是 4D [batch, C, H, W]（Batch-First）
        if input_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: input_shape.to_vec(),
                message: format!(
                    "MaxPool2d 输入必须是 4D [batch, C, H, W]，得到 {input_shape:?}。单样本请使用 [1, C, H, W]"
                ),
            });
        }
        let (batch_size, channels, input_h, input_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride.unwrap_or(kernel_size); // 默认 stride = kernel_size

        // 3. 验证池化窗口不超过输入尺寸
        if k_h > input_h || k_w > input_w {
            return Err(GraphError::InvalidOperation(format!(
                "MaxPool2d 池化窗口 {k_h}x{k_w} 超出输入尺寸 {input_h}x{input_w}"
            )));
        }

        // 4. 计算输出尺寸
        let output_h = (input_h - k_h) / s_h + 1;
        let output_w = (input_w - k_w) / s_w + 1;

        if output_h == 0 || output_w == 0 {
            return Err(GraphError::InvalidOperation(format!(
                "MaxPool2d 输出尺寸无效：输入 {}x{}，核 {}x{}，步长 {:?}",
                input_h,
                input_w,
                k_h,
                k_w,
                (s_h, s_w)
            )));
        }

        // 5. 确定输出形状：始终是 4D [batch, C, H', W']
        let fixed_shape = vec![batch_size, channels, output_h, output_w];

        // 计算动态形状
        let parent = &parents[0];
        let parent_dyn = parent.dynamic_expected_shape();
        let supports_dynamic = parent.supports_dynamic_batch();

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
            kernel_size,
            stride: (s_h, s_w),
            max_indices: None,
            input_shape: input_shape.to_vec(),
        })
    }
}

impl TraitNode for MaxPool2d {
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
        let input = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的输入父节点{}没有值",
                self.display_node(),
                parents[0]
            ))
        })?;

        // 输入必须是 4D [batch, C, H, W]
        let input_shape = input.shape();
        let (batch_size, channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        let (k_h, k_w) = self.kernel_size;
        let (s_h, s_w) = self.stride;
        let out_h = (in_h - k_h) / s_h + 1;
        let out_w = (in_w - k_w) / s_w + 1;

        // 输出形状：始终是 4D [batch, C, H', W']
        let output_shape = vec![batch_size, channels, out_h, out_w];
        let single_sample_size = channels * out_h * out_w;

        // Rayon 并行处理每个 batch 样本
        let batch_results: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut sample_output = vec![0.0f32; single_sample_size];
                let mut sample_indices = vec![0.0f32; single_sample_size];

                for c in 0..channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let h_start = oh * s_h;
                            let w_start = ow * s_w;

                            let mut max_val = f32::NEG_INFINITY;
                            let mut max_idx: usize = 0;

                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = h_start + kh;
                                    let iw = w_start + kw;
                                    let val = input[[b, c, ih, iw]];

                                    if val > max_val {
                                        max_val = val;
                                        max_idx = ih * in_w + iw;
                                    }
                                }
                            }

                            let idx = c * out_h * out_w + oh * out_w + ow;
                            sample_output[idx] = max_val;
                            sample_indices[idx] = max_idx as f32;
                        }
                    }
                }
                (sample_output, sample_indices)
            })
            .collect();

        // 合并结果
        let mut all_output = Vec::with_capacity(batch_size * single_sample_size);
        let mut all_indices = Vec::with_capacity(batch_size * single_sample_size);

        for (output, indices) in batch_results {
            all_output.extend(output);
            all_indices.extend(indices);
        }

        self.value = Some(Tensor::new(&all_output, &output_shape));
        self.max_indices = Some(Tensor::new(&all_indices, &output_shape));
        self.input_shape = input_shape.to_vec();
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    // ========== VJP 模式 ==========

    /// 计算梯度（Rayon 并行版本）
    ///
    /// `MaxPool` 的梯度非常简单：
    /// - 最大值位置：梯度 = `upstream_grad`
    /// - 其他位置：梯度 = 0
    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let max_indices = self
            .max_indices
            .as_ref()
            .ok_or_else(|| GraphError::ComputationError("缺少最大值索引缓存".to_string()))?;

        // 输入必须是 4D [batch, C, H', W']
        let input_shape = &self.input_shape;
        let grad_shape = upstream_grad.shape();
        let (batch_size, channels, out_h, out_w) = (
            grad_shape[0],
            grad_shape[1],
            grad_shape[2],
            grad_shape[3],
        );
        let (in_h, in_w) = (input_shape[2], input_shape[3]);
        let single_sample_size = channels * in_h * in_w;

        // Rayon 并行处理每个 batch 样本
        let batch_results: Vec<Vec<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut sample_grad = vec![0.0f32; single_sample_size];

                for c in 0..channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let grad_val = upstream_grad[[b, c, oh, ow]];
                            let max_pos = max_indices[[b, c, oh, ow]] as usize;
                            let ih = max_pos / in_w;
                            let iw = max_pos % in_w;
                            let idx = c * in_h * in_w + ih * in_w + iw;
                            sample_grad[idx] += grad_val;
                        }
                    }
                }
                sample_grad
            })
            .collect();

        let all_data: Vec<f32> = batch_results.into_iter().flatten().collect();
        Ok(Tensor::new(&all_data, input_shape))
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
