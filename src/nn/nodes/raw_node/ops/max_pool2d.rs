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
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
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
    #[allow(dead_code)]
    supports_dynamic: bool,

    // 池化参数
    kernel_size: (usize, usize), // (kH, kW)
    stride: (usize, usize),      // (sH, sW)
    /// 填充 (top, bottom, left, right)
    ///
    /// MaxPool 的 padding 用 `f32::NEG_INFINITY` 填充（避免污染 max 结果）。
    /// 对称 padding 用 (p, p, p, p)，单边非对称用 (1, 0, 0, 0) 等。
    padding: (usize, usize, usize, usize),
    /// ONNX 风格 ceil_mode：true 用 ceil 计算输出尺寸，false 用 floor
    ceil_mode: bool,

    // 缓存（用于反向传播）
    // 存储每个输出位置对应的最大值在 padded 输入空间中的索引
    // （即包含 padding 的索引，反向时按 in_h_padded 反推 (ih_pad, iw_pad)，
    //   再减去 padding 得到原始输入坐标）
    // 形状与输出相同，值为展平后的 padded 输入索引
    max_indices: Option<Tensor>,
    input_shape: Vec<usize>, // 原始输入形状（不含 padding）
}

impl MaxPool2d {
    /// 获取核大小
    #[allow(dead_code)]
    pub(in crate::nn) const fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// 获取步长
    #[allow(dead_code)]
    pub(in crate::nn) const fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// 获取 padding (top, bottom, left, right)
    #[allow(dead_code)]
    pub(in crate::nn) const fn padding(&self) -> (usize, usize, usize, usize) {
        self.padding
    }

    /// 获取 ceil_mode
    #[allow(dead_code)]
    pub(in crate::nn) const fn ceil_mode(&self) -> bool {
        self.ceil_mode
    }

    /// 从父节点形状信息创建 MaxPool2d 节点（核心实现）
    ///
    /// # 参数
    /// - `parent_shape`: 输入形状 [batch, C, H, W]
    /// - `parent_dynamic_shape`: 父节点的动态形状
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)，None 则默认等于 kernel_size
    /// - `padding`: 填充 (top, bottom, left, right)，对称 padding 用 (p,p,p,p)
    /// - `ceil_mode`: true 用 ceil 计算输出尺寸，false 用 floor
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize, usize, usize),
        ceil_mode: bool,
    ) -> Result<Self, GraphError> {
        // 1. 验证输入形状：必须是 4D [batch, C, H, W]
        if parent_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: parent_shape.to_vec(),
                message: format!(
                    "MaxPool2d 输入必须是 4D [batch, C, H, W]，得到 {parent_shape:?}。单样本请使用 [1, C, H, W]"
                ),
            });
        }
        let (batch_size, channels, input_h, input_w) = (
            parent_shape[0],
            parent_shape[1],
            parent_shape[2],
            parent_shape[3],
        );

        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride.unwrap_or(kernel_size);
        let (pad_t, pad_b, pad_l, pad_r) = padding;

        // 2. 验证池化窗口不超过 padded 输入尺寸
        let padded_h = input_h + pad_t + pad_b;
        let padded_w = input_w + pad_l + pad_r;
        if k_h > padded_h || k_w > padded_w {
            return Err(GraphError::InvalidOperation(format!(
                "MaxPool2d 池化窗口 {k_h}x{k_w} 超出 padded 输入尺寸 {padded_h}x{padded_w}（原 {input_h}x{input_w}, padding {padding:?}）"
            )));
        }

        // 3. 计算输出尺寸（按 ONNX MaxPool 公式）
        //    floor: (padded_h - k_h) / s_h + 1
        //    ceil:  ceil((padded_h - k_h) / s_h) + 1
        let output_h = if ceil_mode {
            div_ceil(padded_h - k_h, s_h) + 1
        } else {
            (padded_h - k_h) / s_h + 1
        };
        let output_w = if ceil_mode {
            div_ceil(padded_w - k_w, s_w) + 1
        } else {
            (padded_w - k_w) / s_w + 1
        };

        if output_h == 0 || output_w == 0 {
            return Err(GraphError::InvalidOperation(format!(
                "MaxPool2d 输出尺寸无效：输入 {}x{}，核 {}x{}，步长 {:?}, padding {:?}",
                input_h,
                input_w,
                k_h,
                k_w,
                (s_h, s_w),
                padding
            )));
        }

        // 4. 确定输出形状
        let fixed_shape = vec![batch_size, channels, output_h, output_w];

        // 5. 计算动态形状
        let supports_dynamic = parent_dynamic_shape.has_dynamic_dims();
        let dynamic_shape = if supports_dynamic && parent_dynamic_shape.is_dynamic(0) {
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
            padding,
            ceil_mode,
            max_indices: None,
            input_shape: parent_shape.to_vec(),
        })
    }
}

/// 整数向上取整除法
#[inline]
const fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
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

    fn dedup_fingerprint(&self) -> Option<u64> {
        use crate::nn::nodes::raw_node::hash_dedup_params;
        Some(hash_dedup_params(&[
            self.kernel_size.0 as u64,
            self.kernel_size.1 as u64,
            self.stride.0 as u64,
            self.stride.1 as u64,
            self.padding.0 as u64,
            self.padding.1 as u64,
            self.padding.2 as u64,
            self.padding.3 as u64,
            self.ceil_mode as u64,
        ]))
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let input = parent_values[0];
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
        let (pad_t, pad_b, pad_l, pad_r) = self.padding;
        let padded_h = in_h + pad_t + pad_b;
        let padded_w = in_w + pad_l + pad_r;
        let out_h = if self.ceil_mode {
            div_ceil(padded_h - k_h, s_h) + 1
        } else {
            (padded_h - k_h) / s_h + 1
        };
        let out_w = if self.ceil_mode {
            div_ceil(padded_w - k_w, s_w) + 1
        } else {
            (padded_w - k_w) / s_w + 1
        };

        // 输出形状：始终是 4D [batch, C, H', W']
        let output_shape = vec![batch_size, channels, out_h, out_w];
        let single_sample_size = channels * out_h * out_w;

        // 池化窗口在 padded 空间的访问；padding 区域虚拟为 -inf，
        // 实际通过越界检测跳过（避免实际分配 padded tensor，省内存）。
        // max_indices 存 padded 空间的展平索引（行 * padded_w + 列），
        // 反向时减去 (pad_t, pad_l) 还原到原始输入坐标。
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
                            let mut max_idx_padded: usize = 0;

                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih_padded = h_start + kh;
                                    let iw_padded = w_start + kw;
                                    // 检查是否在原始输入 (非 padding) 区域
                                    if ih_padded < pad_t || ih_padded >= pad_t + in_h
                                        || iw_padded < pad_l || iw_padded >= pad_l + in_w
                                    {
                                        // padding 区域 = -inf，不更新 max
                                        continue;
                                    }
                                    let ih = ih_padded - pad_t;
                                    let iw = iw_padded - pad_l;
                                    let val = input[[b, c, ih, iw]];

                                    if val > max_val {
                                        max_val = val;
                                        max_idx_padded = ih_padded * padded_w + iw_padded;
                                    }
                                }
                            }

                            // ceil_mode 边界情形：池化窗口完全落在 padding 区域
                            // → max 仍为 -inf。ONNX 标准要求 ceil_mode 不产生这种情形
                            // （padded 后切出的位置必须至少含 1 个真实输入），
                            // 否则视为模型 bug。这里用 0.0 兜底防 NaN 传播。
                            if max_val == f32::NEG_INFINITY {
                                max_val = 0.0;
                            }

                            let idx = c * out_h * out_w + oh * out_w + ow;
                            sample_output[idx] = max_val;
                            sample_indices[idx] = max_idx_padded as f32;
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
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let max_indices = self
            .max_indices
            .as_ref()
            .ok_or_else(|| GraphError::ComputationError("缺少最大值索引缓存".to_string()))?;

        // 输入必须是 4D [batch, C, H', W']
        let input_shape = &self.input_shape;
        let grad_shape = upstream_grad.shape();
        let (batch_size, channels, out_h, out_w) =
            (grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]);
        let (in_h, in_w) = (input_shape[2], input_shape[3]);
        let (pad_t, pad_b, pad_l, pad_r) = self.padding;
        let padded_w = in_w + pad_l + pad_r;
        let padded_h_check = in_h + pad_t + pad_b;
        let single_sample_size = channels * in_h * in_w;

        // 预分配单一连续 buffer（避免 Vec<Vec> + flatten 的双重分配）
        let total_size = batch_size * single_sample_size;
        let mut all_data = vec![0.0f32; total_size];

        // max_indices 存的是 padded 空间索引：max_pos = ih_padded * padded_w + iw_padded
        // 反向时减去 (pad_t, pad_l) 还原到原始输入坐标。
        // padding 区域因前向被设为 -inf 永不被选为 max，所以反算的 (ih, iw) 必在原图范围内。
        all_data
            .par_chunks_mut(single_sample_size)
            .enumerate()
            .for_each(|(b, sample_grad)| {
                for c in 0..channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let grad_val = upstream_grad[[b, c, oh, ow]];
                            let max_pos = max_indices[[b, c, oh, ow]] as usize;
                            let ih_padded = max_pos / padded_w;
                            let iw_padded = max_pos % padded_w;
                            // 防御：max_pos == 0 且 grad_val == 0 时会被解析为 (0, 0)，
                            // 但此时不写入梯度（grad_val=0 加 0 无副作用）；
                            // 真实有效 max_pos 必满足 (pad_t, pad_l) ≤ (ih_padded, iw_padded) < (pad_t+in_h, pad_l+in_w)
                            if ih_padded >= pad_t
                                && ih_padded < pad_t + in_h
                                && iw_padded >= pad_l
                                && iw_padded < pad_l + in_w
                            {
                                let ih = ih_padded - pad_t;
                                let iw = iw_padded - pad_l;
                                let idx = c * in_h * in_w + ih * in_w + iw;
                                sample_grad[idx] += grad_val;
                            }
                            // else: ceil_mode 兜底窗口（max_val 被前向重置为 0）→ 跳过
                        }
                    }
                }
            });
        let _ = padded_h_check; // 仅用于命名意图，实际上限校验在前向已做

        Ok(GradResult::Computed(Tensor::new(&all_data, input_shape)))
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
