/*
 * @Author       : 老董
 * @Date         : 2026-01-21
 * @Description  : Select 节点 - 从张量中选择指定轴和索引的切片
 *
 * 用于 RNN 展开式设计：从 [batch, seq_len, input_size] 中提取 [batch, input_size]
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Select 节点：从张量中选择指定轴和索引的切片
///
/// 前向：output = input.select(axis, index)
/// 反向：将梯度放回对应位置，其他位置为 0
///
/// # 主要用途
/// 用于 RNN 展开式设计，从序列 tensor 中提取单个时间步。
#[derive(Clone)]
pub(crate) struct Select {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 输出形状（去掉被 select 的维度）
    fixed_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    supports_dynamic: bool,
    /// 选择的轴
    axis: usize,
    /// 选择的索引
    index: usize,
    /// 父节点的形状（用于反向传播时创建 scatter tensor）
    parent_shape: Vec<usize>,
}

impl Select {
    /// 获取选择的轴
    pub(in crate::nn) const fn axis(&self) -> usize {
        self.axis
    }

    /// 获取选择的索引
    pub(in crate::nn) const fn index(&self) -> usize {
        self.index
    }

    pub(crate) fn new(
        parents: &[&NodeHandle],
        axis: usize,
        index: usize,
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Select 节点只需要 1 个父节点".to_string(),
            ));
        }

        let parent = &parents[0];
        let parent_shape = parent.value_expected_shape();

        // 2. 验证轴
        if axis >= parent_shape.len() {
            return Err(GraphError::InvalidOperation(format!(
                "Select axis {} 超出张量维度 {}",
                axis,
                parent_shape.len()
            )));
        }

        // 3. 验证索引
        if index >= parent_shape[axis] {
            return Err(GraphError::InvalidOperation(format!(
                "Select index {} 超出 axis {} 的大小 {}",
                index, axis, parent_shape[axis]
            )));
        }

        // 4. 计算输出形状（去掉被 select 的维度）
        let mut fixed_shape: Vec<usize> = parent_shape.to_vec();
        fixed_shape.remove(axis);

        // 5. 计算动态形状（去掉被 select 的维度）
        let parent_dyn = parent.dynamic_expected_shape();
        let parent_dims = parent_dyn.dims();
        let mut output_dims: Vec<Option<usize>> = parent_dims.to_vec();
        output_dims.remove(axis);
        let dynamic_shape = DynamicShape::new(&output_dims);

        // 是否支持动态 batch 取决于父节点
        let supports_dynamic = parent.supports_dynamic_batch();

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
            axis,
            index,
            parent_shape: parent_shape.to_vec(),
        })
    }
}

impl TraitNode for Select {
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
        let parent_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{} 的父节点 {} 没有值",
                self.display_node(),
                parents[0]
            ))
        })?;

        // 使用 Tensor::select 提取切片
        self.value = Some(parent_value.select(self.axis, self.index));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // Select 的反向传播：将梯度放回对应位置，其他位置为 0
        //
        // 前向：output = input[:, index, :] （假设 axis=1）
        // 反向：grad_input 是一个全零张量，只在 [:, index, :] 处填入 upstream_grad
        //
        // 使用 scatter 操作：在 parent_shape 大小的零张量中，将 upstream_grad 放入 (axis, index) 处

        let mut grad_input = Tensor::zeros(&self.parent_shape);

        // 将 upstream_grad 扩展一个维度后放入对应位置
        // upstream_grad: [batch, input_size] → expanded: [batch, 1, input_size]
        // 然后用 scatter 放入 grad_input[:, index, :] 位置
        let expanded_shape: Vec<usize> = {
            let mut s = upstream_grad.shape().to_vec();
            s.insert(self.axis, 1);
            s
        };
        let expanded_grad = upstream_grad.reshape(&expanded_shape);

        // 使用 slice 赋值（scatter）
        grad_input.scatter_at(self.axis, self.index, &expanded_grad);

        Ok(grad_input)
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
