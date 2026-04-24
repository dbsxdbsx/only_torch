/*
 * @Author       : 老董
 * @Date         : 2026-04-24
 * @Description  : 2D 最近邻上采样节点（PyTorch 风格）
 *
 * 用途说明：
 * - YOLOv5 等目标检测网络的 PAN/FPN 颈部需要把深层小特征图上采样后跟浅层特征图拼接
 * - 仅实现 nearest mode（YOLOv5 用），bilinear/bicubic 暂不支持
 * - 仅支持整数倍 scale（最常见场景，避免引入小数索引的复杂性）
 *
 * 数学定义（以 scale_h=scale_w=s 为例）：
 *
 *   前向：y[n, c, i, j] = x[n, c, i / s, j / s]
 *         （即输入每个像素被复制到输出的 s×s 块）
 *
 *   反向：dL/dx[n, c, i, j] = ∑(di in 0..s) ∑(dj in 0..s) dL/dy[n, c, i*s+di, j*s+dj]
 *         （即对上游梯度按 s×s 块"求和"，注意是 sum 不是 mean）
 *
 *   ⚠️ 易错点：反向是 sum_pool 不是 avg_pool。avg_pool 会多除一个 s×s。
 *
 * 设计决策：
 * - Batch-First 格式：输入必须是 4D [batch, C, H, W]，输出 [batch, C, H*scale_h, W*scale_w]
 * - 单样本使用 batch=1，如 [1, C, H, W]
 * - 使用 Rayon 在 batch 维度并行加速
 * - 仅缓存输入形状（反向不需要前向值，跟 AvgPool 一样无需缓存 max_indices）
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

/// 2D 最近邻上采样节点
#[derive(Clone)]
pub(crate) struct Upsample2d {
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

    // 上采样参数
    scale_h: usize, // H 方向放大倍数
    scale_w: usize, // W 方向放大倍数

    // 缓存
    input_shape: Vec<usize>, // 原始输入形状（反向用）
}

impl Upsample2d {
    /// 获取 H 方向放大倍数
    #[allow(dead_code)]
    pub(in crate::nn) const fn scale_h(&self) -> usize {
        self.scale_h
    }

    /// 获取 W 方向放大倍数
    #[allow(dead_code)]
    pub(in crate::nn) const fn scale_w(&self) -> usize {
        self.scale_w
    }

    /// 从父节点形状信息创建 Upsample2d 节点
    ///
    /// # 参数
    /// - `parent_shape`: 输入形状 [batch, C, H, W]
    /// - `parent_dynamic_shape`: 父节点的动态形状
    /// - `scale_h`: H 方向放大倍数（必须 ≥ 1）
    /// - `scale_w`: W 方向放大倍数（必须 ≥ 1）
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        scale_h: usize,
        scale_w: usize,
    ) -> Result<Self, GraphError> {
        // 1. 验证输入形状：必须是 4D [batch, C, H, W]
        if parent_shape.len() != 4 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: parent_shape.to_vec(),
                message: format!(
                    "Upsample2d 输入必须是 4D [batch, C, H, W]，得到 {parent_shape:?}。单样本请使用 [1, C, H, W]"
                ),
            });
        }

        // 2. 验证 scale 参数有效
        if scale_h == 0 || scale_w == 0 {
            return Err(GraphError::InvalidOperation(format!(
                "Upsample2d scale 必须 ≥ 1，得到 scale_h={scale_h}, scale_w={scale_w}"
            )));
        }

        let (batch_size, channels, input_h, input_w) = (
            parent_shape[0],
            parent_shape[1],
            parent_shape[2],
            parent_shape[3],
        );

        // 3. 计算输出形状：[batch, C, H*scale_h, W*scale_w]
        let output_h = input_h * scale_h;
        let output_w = input_w * scale_w;
        let fixed_shape = vec![batch_size, channels, output_h, output_w];

        // 4. 计算动态形状（与池化算子一致，仅 batch 维支持动态）
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
            scale_h,
            scale_w,
            input_shape: parent_shape.to_vec(),
        })
    }
}

impl TraitNode for Upsample2d {
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
            self.scale_h as u64,
            self.scale_w as u64,
        ]))
    }

    /// 前向传播：nearest 像素复制
    ///
    /// y[n, c, i, j] = x[n, c, i / scale_h, j / scale_w]
    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let input = parent_values[0];
        let input_shape = input.shape();
        let (batch_size, channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        let s_h = self.scale_h;
        let s_w = self.scale_w;
        let out_h = in_h * s_h;
        let out_w = in_w * s_w;

        let output_shape = vec![batch_size, channels, out_h, out_w];
        let single_sample_size = channels * out_h * out_w;

        // Rayon 并行处理每个 batch 样本
        let batch_results: Vec<Vec<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut sample_output = vec![0.0f32; single_sample_size];
                for c in 0..channels {
                    for oh in 0..out_h {
                        let ih = oh / s_h;
                        for ow in 0..out_w {
                            let iw = ow / s_w;
                            let idx = c * out_h * out_w + oh * out_w + ow;
                            sample_output[idx] = input[[b, c, ih, iw]];
                        }
                    }
                }
                sample_output
            })
            .collect();

        let all_data: Vec<f32> = batch_results.into_iter().flatten().collect();
        self.value = Some(Tensor::new(&all_data, &output_shape));
        self.input_shape = input_shape.to_vec();
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    // ========== VJP 模式 ==========

    /// 反向传播：sum_pool 累加（**注意是 sum 不是 avg**）
    ///
    /// 输入 x[n,c,i,j] 唯一影响输出的 (s_h × s_w) 块：
    ///   y[n, c, i*s_h : (i+1)*s_h, j*s_w : (j+1)*s_w]
    /// 因此：
    ///   dL/dx[n,c,i,j] = ∑(di in 0..s_h) ∑(dj in 0..s_w) dL/dy[n,c,i*s_h+di, j*s_w+dj]
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let input_shape = &self.input_shape;
        let grad_shape = upstream_grad.shape();
        let (batch_size, channels, out_h, out_w) =
            (grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]);
        let (in_h, in_w) = (input_shape[2], input_shape[3]);

        let s_h = self.scale_h;
        let s_w = self.scale_w;
        let single_sample_size = channels * in_h * in_w;

        // 校验上游梯度形状与前向输出形状匹配
        if out_h != in_h * s_h || out_w != in_w * s_w {
            return Err(GraphError::ShapeMismatch {
                expected: vec![batch_size, channels, in_h * s_h, in_w * s_w],
                got: grad_shape.to_vec(),
                message: format!(
                    "Upsample2d 上游梯度形状不匹配：期望 [{},{},{},{}]，得到 {grad_shape:?}",
                    batch_size,
                    channels,
                    in_h * s_h,
                    in_w * s_w
                ),
            });
        }

        let total_size = batch_size * single_sample_size;
        let mut all_data = vec![0.0f32; total_size];

        // Rayon 并行处理每个 batch 样本，写入预分配 buffer
        all_data
            .par_chunks_mut(single_sample_size)
            .enumerate()
            .for_each(|(b, sample_grad)| {
                for c in 0..channels {
                    for ih in 0..in_h {
                        for iw in 0..in_w {
                            // 累加该输入位置对应的 (s_h × s_w) 输出块的梯度
                            let mut acc = 0.0f32;
                            for di in 0..s_h {
                                for dj in 0..s_w {
                                    let oh = ih * s_h + di;
                                    let ow = iw * s_w + dj;
                                    acc += upstream_grad[[b, c, oh, ow]];
                                }
                            }
                            let idx = c * in_h * in_w + ih * in_w + iw;
                            sample_grad[idx] = acc;
                        }
                    }
                }
            });

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
