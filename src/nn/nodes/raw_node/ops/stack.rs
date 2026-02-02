use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::NodeId;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Stack 节点：将多个张量沿指定轴拼接/堆叠
///
/// - `new_dim=true`：在指定位置插入新维度后堆叠（类似 `torch.stack`）
/// - `new_dim=false`：沿现有维度拼接（类似 `torch.cat`）
#[derive(Clone)]
pub(crate) struct Stack {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 沿哪个轴进行堆叠/拼接
    axis: usize,
    /// 是否插入新维度
    new_dim: bool,
    /// 父节点 ID 列表（用于 backward 时识别目标父节点）
    parent_ids: Vec<NodeId>,
    /// 各父节点在 axis 维度的大小（用于 backward 时 split）
    parent_sizes: Vec<usize>,
    /// 固定形状
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    supports_dynamic: bool,
}

impl Stack {
    /// 获取操作的轴
    pub(crate) const fn axis(&self) -> usize {
        self.axis
    }

    /// 是否插入新维度
    pub(crate) const fn new_dim(&self) -> bool {
        self.new_dim
    }

    /// 从父节点形状信息创建 Stack 节点（核心实现）
    pub(in crate::nn) fn new(
        parent_shapes: &[&[usize]],
        parent_dynamic_shapes: &[DynamicShape],
        parent_ids: Vec<NodeId>,
        axis: usize,
        new_dim: bool,
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parent_shapes.is_empty() {
            return Err(GraphError::InvalidOperation(
                "Stack 节点至少需要 1 个父节点".to_string(),
            ));
        }

        let first_shape = parent_shapes[0];
        let ndim = first_shape.len();

        // 2. 验证 axis
        let max_axis = if new_dim { ndim } else { ndim.saturating_sub(1) };
        if axis > max_axis {
            return Err(GraphError::InvalidOperation(format!(
                "Stack: axis {axis} 超出有效范围 [0, {max_axis}]"
            )));
        }

        // 3. 验证形状并收集 parent_sizes
        let mut parent_sizes = Vec::with_capacity(parent_shapes.len());

        if new_dim {
            for (i, shape) in parent_shapes.iter().enumerate().skip(1) {
                if *shape != first_shape {
                    return Err(GraphError::ShapeMismatch {
                        expected: first_shape.to_vec(),
                        got: shape.to_vec(),
                        message: format!("Stack (new_dim=true): 父节点 {i} 形状不一致"),
                    });
                }
            }
            parent_sizes = vec![1; parent_shapes.len()];
        } else {
            for (i, shape) in parent_shapes.iter().enumerate() {
                if shape.len() != ndim {
                    return Err(GraphError::ShapeMismatch {
                        expected: first_shape.to_vec(),
                        got: shape.to_vec(),
                        message: format!("Stack (new_dim=false): 父节点 {i} 维度数不一致"),
                    });
                }
                for (d, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                    if d != axis && s1 != s2 {
                        return Err(GraphError::ShapeMismatch {
                            expected: first_shape.to_vec(),
                            got: shape.to_vec(),
                            message: format!(
                                "Stack (new_dim=false): 父节点 {i} 在维度 {d} 大小不一致"
                            ),
                        });
                    }
                }
                parent_sizes.push(shape[axis]);
            }
        }

        // 4. 计算输出形状
        let fixed_shape = if new_dim {
            let mut shape = first_shape.to_vec();
            shape.insert(axis, parent_shapes.len());
            shape
        } else {
            let mut shape = first_shape.to_vec();
            shape[axis] = parent_sizes.iter().sum();
            shape
        };

        // 5. 计算动态形状
        let supports_dynamic = parent_dynamic_shapes.iter().any(|d| d.has_dynamic_dims());
        let first_dyn = &parent_dynamic_shapes[0];

        let dynamic_shape = if new_dim {
            let mut dims: Vec<Option<usize>> = Vec::with_capacity(fixed_shape.len());
            for (i, &size) in fixed_shape.iter().enumerate() {
                if i == axis {
                    dims.push(Some(size));
                } else {
                    let orig_idx = if i < axis { i } else { i - 1 };
                    dims.push(first_dyn.dim(orig_idx));
                }
            }
            DynamicShape::new(&dims)
        } else {
            let mut dims: Vec<Option<usize>> = Vec::with_capacity(fixed_shape.len());
            for (i, &size) in fixed_shape.iter().enumerate() {
                if i == axis {
                    dims.push(Some(size));
                } else {
                    dims.push(first_dyn.dim(i));
                }
            }
            DynamicShape::new(&dims)
        };

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            axis,
            new_dim,
            parent_ids,
            parent_sizes,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
        })
    }

}

impl TraitNode for Stack {
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
        // 调用 Tensor::stack
        let result = Tensor::stack(parent_values, self.axis, self.new_dim);
        self.value = Some(result);
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        // 使用传入的索引（新签名直接提供索引）
        let idx = target_parent_index;

        // 计算该父节点在 axis 维度的起始偏移
        let start_offset: usize = self.parent_sizes[..idx].iter().sum();

        // 从 upstream_grad 中提取对应部分
        let grad = if self.new_dim {
            // stack 模式：使用 select 取出第 idx 个切片
            upstream_grad.select(self.axis, idx)
        } else {
            // concat 模式：使用 split 逻辑，取 [start_offset, start_offset + size) 范围
            // 使用切片操作
            let size = self.parent_sizes[idx];
            Self::slice_along_axis(upstream_grad, self.axis, start_offset, size)
        };

        Ok(grad)
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

impl Stack {
    /// 沿指定轴切片（辅助方法）
    fn slice_along_axis(tensor: &Tensor, axis: usize, start: usize, len: usize) -> Tensor {
        // 使用 split 然后取对应部分
        // 构造 sizes：[start, len, rest]
        let axis_size = tensor.shape()[axis];
        let rest = axis_size - start - len;

        let mut sizes = Vec::new();
        if start > 0 {
            sizes.push(start);
        }
        sizes.push(len);
        if rest > 0 {
            sizes.push(rest);
        }

        let parts = tensor.split(axis, &sizes);

        // 返回目标部分的索引
        let target_idx = usize::from(start > 0);
        parts.into_iter().nth(target_idx).unwrap()
    }
}
