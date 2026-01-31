/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Sum 节点（归约求和）
 *
 * 支持两种模式：
 * - 全局求和：axis = None，将所有元素求和为 [1, 1]
 * - 按轴求和：axis = Some(i)，沿指定轴求和（keepdims=true）
 *
 * 反向传播：
 * - 全局求和：grad broadcast 回原 shape
 * - 按轴求和：grad broadcast 回原 shape（沿求和轴广播）
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Sum 节点（归约求和）
///
/// ## 模式
/// - `axis = None`：全局求和，输出 [1, 1]
/// - `axis = Some(i)`：沿轴 i 求和，保持维度（keepdims=true）
///
/// ## 输入
/// - 父节点：任意形状的张量
///
/// ## 输出
/// - 全局模式：[1, 1]
/// - 按轴模式：原 shape 中第 axis 维变为 1
///
/// ## 梯度计算
/// 对于 y = sum(x)，反向传播时将上游梯度广播回输入形状
#[derive(Clone)]
pub(crate) struct Sum {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 求和轴：None 表示全局求和，Some(i) 表示沿轴 i 求和
    axis: Option<usize>,
    /// 固定形状（输出 shape）
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    supports_dynamic: bool,
    /// 缓存输入形状，用于反向传播
    input_shape_cache: Option<Vec<usize>>,
}

impl Sum {
    pub(crate) fn new(parents: &[&NodeHandle], axis: Option<usize>) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Sum 节点需要正好 1 个父节点".to_string(),
            ));
        }

        // 2. 获取输入形状
        let parent = &parents[0];
        let input_shape = parent.value_expected_shape();

        // 3. 验证 axis 有效性
        if let Some(ax) = axis {
            if ax >= input_shape.len() {
                return Err(GraphError::InvalidOperation(format!(
                    "Sum: axis {} 超出输入维度范围 {}",
                    ax,
                    input_shape.len()
                )));
            }
        }

        // 4. 计算输出形状
        let fixed_shape = match axis {
            None => vec![1, 1], // 全局求和
            Some(ax) => {
                // 按轴求和，keepdims=true
                let mut shape = input_shape.to_vec();
                shape[ax] = 1;
                shape
            }
        };

        // 5. 处理动态形状
        let parent_dynamic = parent.dynamic_expected_shape();
        let supports_dynamic = parent.supports_dynamic_batch();

        let dynamic_shape = match axis {
            None => {
                // 全局求和输出固定 [1, 1]
                DynamicShape::fixed(&[1, 1])
            }
            Some(ax) => {
                // 按轴求和，保持除 axis 外的动态性
                // 第 axis 维变为固定 1
                let dims: Vec<Option<usize>> = (0..input_shape.len())
                    .map(|i| {
                        if i == ax {
                            Some(1) // 被求和的维度变为固定 1
                        } else {
                            parent_dynamic.dim(i) // 保持原有动态性
                        }
                    })
                    .collect();
                DynamicShape::new(&dims)
            }
        };

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
        })
    }

    /// 获取 axis 配置
    pub fn axis(&self) -> Option<usize> {
        self.axis
    }
}

impl TraitNode for Sum {
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
        let input = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!("{}的父{}没有值", self.display_node(), parents[0]))
        })?;

        // 缓存输入形状用于反向传播
        self.input_shape_cache = Some(input.shape().to_vec());

        // 根据 axis 计算
        let output = match self.axis {
            None => input.sum(),                     // 全局求和 -> [1, 1]
            Some(ax) => input.sum_axis_keepdims(ax), // 按轴求和
        };

        self.value = Some(output);
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Sum 反向传播的 VJP 计算
    ///
    /// 对于 y = sum(x)，有：
    /// ∂y/∂x_i = 1（每个输入元素对输出的贡献都是 1）
    ///
    /// VJP: grad_to_parent = broadcast(upstream_grad, input_shape)
    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let input_shape = self.input_shape_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("Sum 输入形状缓存为空，需先执行前向传播".to_string())
        })?;

        // 将上游梯度广播回输入形状
        let grad = upstream_grad.broadcast_to(input_shape);
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
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
