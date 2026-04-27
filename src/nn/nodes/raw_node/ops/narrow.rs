use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Narrow 节点 — 沿单轴取连续范围（不降维）
///
/// forward: output = input.narrow(axis, start, length)
/// backward: 创建 parent_shape 大小零张量，在 [start..start+length] 位置放入梯度
///
/// 这是 `select` 的范围版，也是 `split` 的基础原语。
/// 命名遵循 PyTorch 的 `tensor.narrow(dim, start, length)`。
#[derive(Clone)]
pub(crate) struct Narrow {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状（narrowed 后的形状）
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    #[allow(dead_code)]
    supports_dynamic: bool,
    /// 操作的轴
    axis: usize,
    /// 起始索引
    start: usize,
    /// 取的长度
    length: usize,
}

impl Narrow {
    pub(crate) const fn axis(&self) -> usize {
        self.axis
    }
    pub(crate) const fn start(&self) -> usize {
        self.start
    }
    pub(crate) const fn length(&self) -> usize {
        self.length
    }

    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        axis: usize,
        start: usize,
        length: usize,
    ) -> Result<Self, GraphError> {
        // 验证参数
        if axis >= parent_shape.len() {
            return Err(GraphError::InvalidOperation(format!(
                "Narrow: axis {axis} 超出维度 {}",
                parent_shape.len()
            )));
        }
        if start + length > parent_shape[axis] {
            return Err(GraphError::InvalidOperation(format!(
                "Narrow: start({start}) + length({length}) 超出轴 {axis} 的大小 {}",
                parent_shape[axis]
            )));
        }

        // 计算输出形状
        let mut output_shape = parent_shape.to_vec();
        output_shape[axis] = length;

        // 计算动态形状：复制 parent 的动态维度信息，固定维度用 narrow 后的大小
        let output_dims: Vec<Option<usize>> = (0..parent_shape.len())
            .map(|i| {
                if parent_dynamic_shape.is_dynamic(i) {
                    None // 动态维度保持动态
                } else if i == axis {
                    Some(length) // narrow 轴用新长度
                } else {
                    Some(parent_shape[i]) // 其他固定维度不变
                }
            })
            .collect();
        let output_dynamic = DynamicShape::new(&output_dims);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: output_shape,
            dynamic_shape: output_dynamic,
            supports_dynamic: parent_dynamic_shape.has_dynamic_dims(),
            axis,
            start,
            length,
        })
    }
}

impl TraitNode for Narrow {
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
        // 用 axis + start + length 组合生成指纹
        Some((self.axis as u64) << 32 | (self.start as u64) << 16 | self.length as u64)
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        self.value = Some(parent_values[0].narrow(self.axis, self.start, self.length));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // 创建与 parent 同形状的零张量，在 [start..start+length] 位置放入 upstream_grad
        let parent_shape = parent_values[0].shape();
        let mut grad = Tensor::zeros(parent_shape);
        grad.scatter_range(self.axis, self.start, upstream_grad);
        Ok(GradResult::Computed(grad))
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }
    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }

    fn grad_mut(&mut self) -> Option<&mut Tensor> {
        self.grad.as_mut()
    }
    fn clear_value(&mut self) -> Result<(), GraphError> {
        self.value = None;
        Ok(())
    }
    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
