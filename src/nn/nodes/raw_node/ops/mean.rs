/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Mean 节点（归约求均值）
 *
 * 支持两种模式：
 * - 全局均值：axis = None，将所有元素求均值为 [1, 1]
 * - 按轴均值：axis = Some(i)，沿指定轴求均值（keepdims=true）
 *
 * 反向传播：
 * - 对于 y = mean(x)，有 ∂y/∂x_i = 1/n
 * - grad_to_parent = broadcast(upstream_grad, input_shape) / n
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Mean 节点（归约求均值）
///
/// ## 模式
/// - `axis = None`：全局均值，输出 [1, 1]
/// - `axis = Some(i)`：沿轴 i 求均值，保持维度（keepdims=true）
///
/// ## 输入
/// - 父节点：任意形状的张量
///
/// ## 输出
/// - 全局模式：[1, 1]
/// - 按轴模式：原 shape 中第 axis 维变为 1
///
/// ## 梯度计算
/// 对于 y = mean(x)，∂y/∂x_i = 1/n，反向传播时将上游梯度除以 n 后广播回输入形状
#[derive(Clone)]
pub(crate) struct Mean {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 求均值轴：None 表示全局均值，Some(i) 表示沿轴 i 求均值
    axis: Option<usize>,
    /// 固定形状（输出 shape）
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
    /// 缓存输入形状，用于反向传播
    input_shape_cache: Option<Vec<usize>>,
    /// 缓存归约元素数量，用于反向传播
    reduction_count_cache: Option<usize>,
}

impl Mean {
    /// 从父节点形状信息创建 Mean 节点（核心实现）
    pub(in crate::nn) fn new(
        input_shape: &[usize],
        input_dynamic_shape: &DynamicShape,
        axis: Option<usize>,
    ) -> Result<Self, GraphError> {
        // 验证 axis 有效性
        if let Some(ax) = axis {
            if ax >= input_shape.len() {
                return Err(GraphError::InvalidOperation(format!(
                    "Mean: axis {} 超出输入维度范围 {}",
                    ax,
                    input_shape.len()
                )));
            }
        }

        // 计算输出形状
        let fixed_shape = match axis {
            None => vec![1, 1],
            Some(ax) => {
                let mut shape = input_shape.to_vec();
                shape[ax] = 1;
                shape
            }
        };

        // 动态形状
        let dynamic_shape = match axis {
            None => DynamicShape::new(&[Some(1), Some(1)]),
            Some(ax) => {
                let mut dims: Vec<_> = input_dynamic_shape.dims().to_vec();
                if ax < dims.len() {
                    dims[ax] = Some(1);
                }
                DynamicShape::new(&dims)
            }
        };

        // 是否支持动态 batch
        let supports_dynamic = input_dynamic_shape.dims().first() == Some(&None);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            axis,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
            input_shape_cache: None,
            reduction_count_cache: None,
        })
    }

    /// 获取 axis 配置
    #[allow(dead_code)]
    pub fn axis(&self) -> Option<usize> {
        self.axis
    }
}

impl TraitNode for Mean {
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
        let input = parent_values[0];
        // 缓存输入形状用于反向传播
        self.input_shape_cache = Some(input.shape().to_vec());
        // 计算归约元素数量
        let reduction_count = match self.axis {
            None => input.shape().iter().product(), // 全局均值：所有元素
            Some(ax) => input.shape()[ax],          // 按轴均值：该轴的大小
        };
        self.reduction_count_cache = Some(reduction_count);
        // 根据 axis 计算
        let output = match self.axis {
            None => input.mean(),                     // 全局均值 -> [1, 1]
            Some(ax) => input.mean_axis_keepdims(ax), // 按轴均值
        };
        self.value = Some(output);
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Mean 反向传播的 VJP 计算
    ///
    /// 对于 y = mean(x) = sum(x) / n，有：
    /// ∂y/∂x_i = 1/n
    ///
    /// VJP: grad_to_parent = broadcast(upstream_grad, input_shape) / n
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        let input_shape = self.input_shape_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("Mean 输入形状缓存为空，需先执行前向传播".to_string())
        })?;

        let reduction_count = self.reduction_count_cache.ok_or_else(|| {
            GraphError::ComputationError("Mean 归约计数缓存为空，需先执行前向传播".to_string())
        })?;

        // 将上游梯度广播回输入形状，然后除以归约元素数量
        let grad = upstream_grad.broadcast_to(input_shape) / (reduction_count as f32);
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
        self.input_shape_cache = None;
        self.reduction_count_cache = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
