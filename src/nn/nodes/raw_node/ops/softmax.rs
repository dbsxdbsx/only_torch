/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : Softmax 激活节点
 *                 实现沿最后一维的 softmax: softmax(x)_i = exp(x_i) / Σ exp(x_j)
 */

use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::nn::{GraphError, Mode};
use crate::tensor::Tensor;
/// Softmax 激活节点
///
/// 对输入张量沿最后一维计算 softmax，输出与输入形状相同。
/// 使用数值稳定的 log-sum-exp 技巧避免溢出。
///
/// ## 输入
/// - 父节点: [batch, `num_classes`] 或 [1, `num_classes`]
///
/// ## 输出
/// - 与输入形状相同，每行归一化为概率分布
#[derive(Clone)]
pub(crate) struct Softmax {
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
    /// 缓存输出结果，用于反向传播
    output_cache: Option<Tensor>,
    /// Inference 模式下跳过 backward 缓存
    should_cache_for_backward: bool,
}

impl Softmax {
    /// 从父节点形状信息创建 Softmax 节点（核心实现）
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
    ) -> Result<Self, GraphError> {
        // 验证形状：Softmax 需要 2D 输入
        if parent_shape.len() != 2 {
            return Err(GraphError::InvalidOperation(format!(
                "Softmax 节点需要 2D 输入 [batch, num_classes]，但得到 {parent_shape:?}"
            )));
        }

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: parent_shape.to_vec(),
            dynamic_shape: parent_dynamic_shape.clone(),
            supports_dynamic: parent_dynamic_shape.has_dynamic_dims(),
            output_cache: None,
            should_cache_for_backward: true,
        })
    }
}

impl TraitNode for Softmax {
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

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        // 复用 Tensor 层的数值稳定 softmax 实现
        let output = parent_values[0].softmax_last_dim();
        if self.should_cache_for_backward {
            self.output_cache = Some(output.clone());
        } else {
            self.output_cache = None;
        }
        self.value = Some(output);
        Ok(())
    }

    fn set_mode(&mut self, mode: Mode) {
        self.should_cache_for_backward = mode.caches_for_backward();
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Softmax 反向传播的 VJP 计算
    ///
    /// 对于 y = softmax(x)，Jacobian 矩阵为：
    /// `dL/dx_i` = `Σ_j` (`dL/dy_j` * `dy_j/dx_i`)
    ///         = `Σ_j` (`dL/dy_j` * (`y_i` * (`δ_ij` - `y_j`)))
    ///         = `y_i` * (`dL/dy_i` - `Σ_j` (`dL/dy_j` * `y_j`))
    ///         = `y_i` * (`dL/dy_i` - <dL/dy, y>)
    ///
    /// 其中 <dL/dy, y> 是上游梯度与 softmax 输出的内积。
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let softmax_output = self
            .output_cache
            .as_ref()
            .ok_or_else(|| GraphError::backward_cache_missing(self.display_node(), "output"))?;

        let shape = softmax_output.shape();
        let batch_size = shape[0];
        let num_classes = shape[1];

        // 行 slice 直取（免逐元素 IxDyn 索引/边界检查）；框架约定运算产物为标准布局，
        // 极少数非连续来源（转置视图等）付一次物化兜底。
        let out_owned;
        let out_slice = if softmax_output.is_contiguous() {
            softmax_output.data_as_slice()
        } else {
            out_owned = softmax_output.clone().into_contiguous();
            out_owned.data_as_slice()
        };
        let up_owned;
        let up_slice = if upstream_grad.is_contiguous() {
            upstream_grad.data_as_slice()
        } else {
            up_owned = upstream_grad.clone().into_contiguous();
            up_owned.data_as_slice()
        };

        // 单块缓冲按行直写（与旧逐元素索引版运算顺序一致，逐 bit 等价；
        // 消除 Vec<Vec> 每行分配 + flatten 二次拷贝）。
        // map-only 行独立写入，经阈值分流：小任务（如 [1,N] 推理）串行免调度开销。
        let mut grad_data = vec![0.0f32; batch_size * num_classes];
        let total_work = batch_size * num_classes * 4;
        crate::utils::parallel::for_each_chunk_mut(
            &mut grad_data,
            num_classes,
            total_work,
            |b, row| {
                let y = &out_slice[b * num_classes..(b + 1) * num_classes];
                let g = &up_slice[b * num_classes..(b + 1) * num_classes];

                // 计算 <dL/dy, y> = Σ_j (dL/dy_j * y_j)
                let mut dot_product = 0.0f32;
                for c in 0..num_classes {
                    dot_product += g[c] * y[c];
                }

                // dL/dx_i = y_i * (dL/dy_i - dot_product)
                for c in 0..num_classes {
                    row[c] = y[c] * (g[c] - dot_product);
                }
            },
        );

        // owned Vec 传入 new 零拷贝接管
        Ok(GradResult::Computed(Tensor::new(grad_data, shape)))
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
        self.output_cache = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
