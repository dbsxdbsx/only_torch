/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : Pad（填充）节点
 *                 实现常量值填充: y = pad(x, paddings, value)
 *
 * 主要用于 CNN same-padding 和序列对齐。
 * 反向传播时，梯度通过裁切（slice）回原始形状来传递。
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// 常量值填充节点
///
/// forward: y = pad(x, paddings, value)
/// backward: grad_to_parent = slice(upstream_grad, 原始区域)
///
/// ## 输入
/// - 父节点：任意形状的张量
/// - paddings: 每个维度的 (before, after) 填充量
/// - value: 填充值
///
/// ## 输出
/// - 形状为原形状 + 各维度填充量
#[derive(Clone)]
pub(crate) struct Pad {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 输出形状（填充后）
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
    /// 每个维度的填充量
    paddings: Vec<(usize, usize)>,
    /// 填充值
    pad_value: f32,
    /// 原始输入形状（用于反向传播裁切）
    input_shape: Vec<usize>,
}

impl Pad {
    pub(crate) fn paddings(&self) -> &[(usize, usize)] {
        &self.paddings
    }
    pub(crate) const fn pad_value(&self) -> f32 {
        self.pad_value
    }

    /// 从父节点形状信息创建 Pad 节点
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        paddings: Vec<(usize, usize)>,
        pad_value: f32,
    ) -> Result<Self, GraphError> {
        if paddings.len() != parent_shape.len() {
            return Err(GraphError::InvalidOperation(format!(
                "Pad: paddings 长度 {} 与输入维度数 {} 不一致",
                paddings.len(),
                parent_shape.len()
            )));
        }

        // 计算输出形状
        let output_shape: Vec<usize> = parent_shape
            .iter()
            .zip(paddings.iter())
            .map(|(&dim, &(before, after))| dim + before + after)
            .collect();

        // 动态形状处理：如果第一维是动态的，填充后仍然是动态的
        let supports_dynamic = parent_dynamic_shape.dims().first() == Some(&None);
        let mut output_dynamic = parent_dynamic_shape.clone();
        if supports_dynamic {
            // 动态 batch 维度：填充量加到固定部分（这里简化处理，不支持 batch 维度填充）
            // 实际上 CNN padding 通常只填充 H/W，不填充 batch 和 channel
        }
        // 更新非动态维度的大小
        let output_dims: Vec<Option<usize>> = output_dynamic
            .dims()
            .iter()
            .zip(output_shape.iter())
            .map(|(dyn_dim, &out_size)| {
                if dyn_dim.is_none() {
                    None // 保持动态
                } else {
                    Some(out_size)
                }
            })
            .collect();
        output_dynamic = DynamicShape::new(&output_dims);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: output_shape,
            dynamic_shape: output_dynamic,
            supports_dynamic,
            paddings,
            pad_value,
            input_shape: parent_shape.to_vec(),
        })
    }
}

impl TraitNode for Pad {
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
        self.value = Some(parent_values[0].pad(&self.paddings, self.pad_value));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Pad 反向传播
    ///
    /// 填充区域对原始输入没有贡献，梯度为 0。
    /// 因此只需将 upstream_grad 中对应原始区域的部分提取出来。
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // 从 upstream_grad 中裁切出原始区域
        let ranges: Vec<(usize, usize)> = self
            .paddings
            .iter()
            .zip(self.input_shape.iter())
            .map(|(&(before, _), &dim)| (before, before + dim))
            .collect();

        Ok(GradResult::Computed(upstream_grad.slice_ranges(&ranges)))
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
