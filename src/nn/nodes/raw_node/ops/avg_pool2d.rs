/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : 2D 平均池化节点（PyTorch 风格）
 *
 * 设计决策：
 * - 计算窗口内所有值的平均
 * - 反向传播时梯度均匀分配到窗口内所有位置
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
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use rayon::prelude::*;

/// 2D 平均池化节点
#[derive(Clone)]
pub(crate) struct AvgPool2d {
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

    // 缓存
    input_shape: Vec<usize>, // 原始输入形状
}

impl AvgPool2d {
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

    /// 从父节点形状信息创建 AvgPool2d 节点（核心实现）
    ///
    /// # 参数
    /// - `parent_shape`: 输入形状 [batch, C, H, W]
    /// - `parent_dynamic_shape`: 父节点的动态形状
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)，None 则默认等于 kernel_size
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
    ) -> Result<Self, GraphError> {
        // 1. 验证输入形状：必须是 4D [batch, C, H, W]
        if parent_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: parent_shape.to_vec(),
                message: format!(
                    "AvgPool2d 输入必须是 4D [batch, C, H, W]，得到 {parent_shape:?}。单样本请使用 [1, C, H, W]"
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

        // 2. 验证池化窗口不超过输入尺寸
        if k_h > input_h || k_w > input_w {
            return Err(GraphError::InvalidOperation(format!(
                "AvgPool2d 池化窗口 {k_h}x{k_w} 超出输入尺寸 {input_h}x{input_w}"
            )));
        }

        // 3. 计算输出尺寸
        let output_h = (input_h - k_h) / s_h + 1;
        let output_w = (input_w - k_w) / s_w + 1;

        if output_h == 0 || output_w == 0 {
            return Err(GraphError::InvalidOperation(format!(
                "AvgPool2d 输出尺寸无效：输入 {}x{}，核 {}x{}，步长 {:?}",
                input_h,
                input_w,
                k_h,
                k_w,
                (s_h, s_w)
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
            input_shape: parent_shape.to_vec(),
        })
    }
}

impl TraitNode for AvgPool2d {
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
        ]))
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        // 手写平铺偏移索引要求行主序连续；父节点可能传入 permute 等非连续视图
        // （连续时零拷贝借用）。逐窗口 IxDyn 多维索引是此前 bench 实测的主开销。
        let input_c = parent_values[0].contiguous();
        // 输入必须是 4D [batch, C, H, W]
        let input_shape = input_c.shape().to_vec();
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
        let pool_size = (k_h * k_w) as f32;
        let single_sample_size = channels * out_h * out_w;
        let sample_in_size = channels * in_h * in_w;
        let in_s = input_c.data_as_slice();

        // 预分配单一连续 buffer + 按样本直写（消除 Vec<Vec> + flatten 双重分配）；
        // 窗口内按 (kh, kw) 行序累加，与旧实现逐 bit 一致。
        let mut all_data = vec![0.0f32; batch_size * single_sample_size];
        all_data
            .par_chunks_mut(single_sample_size)
            .enumerate()
            .for_each(|(b, sample_output)| {
                let in_base = b * sample_in_size;
                for c in 0..channels {
                    let in_c_base = in_base + c * in_h * in_w;
                    let out_c_base = c * out_h * out_w;
                    for oh in 0..out_h {
                        let h_start = oh * s_h;
                        for ow in 0..out_w {
                            let w_start = ow * s_w;

                            let mut sum = 0.0f32;
                            for kh in 0..k_h {
                                let row_start = in_c_base + (h_start + kh) * in_w + w_start;
                                for &v in &in_s[row_start..row_start + k_w] {
                                    sum += v;
                                }
                            }

                            sample_output[out_c_base + oh * out_w + ow] = sum / pool_size;
                        }
                    }
                }
            });

        self.value = Some(Tensor::from_vec(all_data, &output_shape));
        self.input_shape = input_shape;
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    // ========== VJP 模式 ==========

    /// 计算梯度（Rayon 并行版本）
    ///
    /// `AvgPool` 的梯度：
    /// - 每个输入位置的梯度 = 所有包含该位置的输出的 `upstream_grad` / `pool_size` 之和
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // 输入必须是 4D [batch, C, H', W']
        let input_shape = &self.input_shape;
        let grad_shape = upstream_grad.shape();
        let (batch_size, channels, out_h, out_w) =
            (grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]);

        let (k_h, k_w) = self.kernel_size;
        let (s_h, s_w) = self.stride;
        let pool_size = (k_h * k_w) as f32;
        let grad_val = 1.0 / pool_size;
        let (in_h, in_w) = (input_shape[2], input_shape[3]);
        let single_sample_size = channels * in_h * in_w;

        // 平铺读要求连续；upstream 可能来自非连续视图（连续时零拷贝借用）
        let up_c = upstream_grad.contiguous();
        let up_s = up_c.data_as_slice();
        let sample_up_size = channels * out_h * out_w;

        // 预分配单一连续 buffer + 按样本直写；`upstream * grad_val` 外提
        // （同操作数乘积恒定，逐 bit 等价），窗口内写入序与旧实现一致。
        let mut all_data = vec![0.0f32; batch_size * single_sample_size];
        all_data
            .par_chunks_mut(single_sample_size)
            .enumerate()
            .for_each(|(b, sample_grad)| {
                let up_base = b * sample_up_size;
                for c in 0..channels {
                    let up_c_base = up_base + c * out_h * out_w;
                    let in_c_base = c * in_h * in_w;
                    for oh in 0..out_h {
                        let h_start = oh * s_h;
                        for ow in 0..out_w {
                            let g = up_s[up_c_base + oh * out_w + ow] * grad_val;
                            let w_start = ow * s_w;

                            for kh in 0..k_h {
                                let row_start = in_c_base + (h_start + kh) * in_w + w_start;
                                for slot in &mut sample_grad[row_start..row_start + k_w] {
                                    *slot += g;
                                }
                            }
                        }
                    }
                }
            });

        Ok(GradResult::Computed(Tensor::from_vec(
            all_data,
            input_shape,
        )))
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
