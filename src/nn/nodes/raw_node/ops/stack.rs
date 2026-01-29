use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
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

    pub(crate) fn new(
        parents: &[&NodeHandle],
        axis: usize,
        new_dim: bool,
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.is_empty() {
            return Err(GraphError::InvalidOperation(
                "Stack 节点至少需要 1 个父节点".to_string(),
            ));
        }

        let first_shape = parents[0].value_expected_shape();
        let ndim = first_shape.len();

        // 2. 验证 axis
        let max_axis = if new_dim {
            ndim
        } else {
            ndim.saturating_sub(1)
        };
        if axis > max_axis {
            return Err(GraphError::InvalidOperation(format!(
                "Stack: axis {axis} 超出有效范围 [0, {max_axis}]"
            )));
        }

        // 3. 验证形状并收集 parent_sizes
        let mut parent_sizes = Vec::with_capacity(parents.len());

        if new_dim {
            // stack 模式：所有父节点形状必须完全相同
            for (i, parent) in parents.iter().enumerate().skip(1) {
                let shape = parent.value_expected_shape();
                if shape != first_shape {
                    return Err(GraphError::ShapeMismatch {
                        expected: first_shape.to_vec(),
                        got: shape.to_vec(),
                        message: format!("Stack (new_dim=true): 父节点 {i} 形状不一致"),
                    });
                }
            }
            // stack 模式下每个父节点贡献 1 个切片
            parent_sizes = vec![1; parents.len()];
        } else {
            // concat 模式：除 axis 外其他维度必须相同
            for (i, parent) in parents.iter().enumerate() {
                let shape = parent.value_expected_shape();
                if shape.len() != ndim {
                    return Err(GraphError::ShapeMismatch {
                        expected: first_shape.to_vec(),
                        got: shape.to_vec(),
                        message: format!("Stack (new_dim=false): 父节点 {i} 维度数不一致"),
                    });
                }
                // 检查除 axis 外的维度
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
            // 在 axis 位置插入新维度
            let mut shape = first_shape.to_vec();
            shape.insert(axis, parents.len());
            shape
        } else {
            // axis 维度是所有父节点在该维度的大小之和
            let mut shape = first_shape.to_vec();
            shape[axis] = parent_sizes.iter().sum();
            shape
        };

        // 5. 计算动态形状
        // Stack/Concat 不涉及广播，直接基于 fixed_shape 构建
        // 但需要保留父节点的动态维度信息
        let supports_dynamic = parents.iter().any(|p| p.supports_dynamic_batch());
        let first_dyn = parents[0].dynamic_expected_shape();

        let dynamic_shape = if new_dim {
            // Stack 模式：在 axis 位置插入固定维度（父节点数量），其他维度继承
            let mut dims: Vec<Option<usize>> = Vec::with_capacity(fixed_shape.len());
            for (i, &size) in fixed_shape.iter().enumerate() {
                if i == axis {
                    // 新插入的维度是固定的（父节点数量）
                    dims.push(Some(size));
                } else {
                    // 其他维度继承第一个父节点的动态性
                    let orig_idx = if i < axis { i } else { i - 1 };
                    dims.push(first_dyn.dim(orig_idx));
                }
            }
            DynamicShape::new(&dims)
        } else {
            // Concat 模式：axis 维度是固定的，其他维度继承
            let mut dims: Vec<Option<usize>> = Vec::with_capacity(fixed_shape.len());
            for (i, &size) in fixed_shape.iter().enumerate() {
                if i == axis {
                    // 拼接轴是固定的（各父节点 axis 维度之和）
                    dims.push(Some(size));
                } else {
                    // 其他维度继承第一个父节点的动态性
                    dims.push(first_dyn.dim(i));
                }
            }
            DynamicShape::new(&dims)
        };

        // 6. 记录父节点 ID（用于 backward 时识别目标父节点）
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

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

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 收集所有父节点的值
        let parent_values: Vec<&Tensor> = parents
            .iter()
            .map(|p| {
                p.value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{} 的父节点 {} 没有值",
                        self.display_node(),
                        p
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // 调用 Tensor::stack
        let result = Tensor::stack(&parent_values, self.axis, self.new_dim);
        self.value = Some(result);

        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 通过父节点 ID 查找索引
        let target_id = target_parent.id();
        let idx = self
            .parent_ids
            .iter()
            .position(|&id| id == target_id)
            .ok_or_else(|| {
                GraphError::ComputationError(format!(
                    "Stack 无法找到父节点 {target_parent} (id={target_id}) 的索引"
                ))
            })?;

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
