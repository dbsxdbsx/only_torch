use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

/// Leaky ReLU 激活函数节点
///
/// forward: f(x) = x if x > 0, else negative_slope * x
/// backward: d(f)/dx = 1 if x > 0, else negative_slope
///
/// 当 negative_slope = 0 时，等价于标准 ReLU
/// MatrixSlow 使用 negative_slope = 0.1
#[derive(Clone)]
pub(crate) struct LeakyReLU {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    grad: Option<Tensor>, // Batch 模式的梯度
    shape: Vec<usize>,
    /// 负半轴斜率，默认 0.0（标准 ReLU）
    negative_slope: f64,
    /// 缓存父节点的值（用于反向传播时判断梯度）
    parent_value: Option<Tensor>,
}

impl LeakyReLU {
    pub(crate) fn new(parents: &[&NodeHandle], negative_slope: f64) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "LeakyReLU节点只需要1个父节点".to_string(),
            ));
        }

        // 1.2 negative_slope 验证（通常应该是非负小数）
        if negative_slope < 0.0 {
            return Err(GraphError::InvalidOperation(format!(
                "LeakyReLU的negative_slope应为非负数，但得到: {}",
                negative_slope
            )));
        }

        // 2. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            jacobi: None,
            grad: None,
            shape: parents[0].value_expected_shape().to_vec(),
            negative_slope,
            parent_value: None,
        })
    }
}

impl TraitNode for LeakyReLU {
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
        &self.shape
    }

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 1. 获取父节点的值
        let parent_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的父节点{}没有值。不该触及本错误，否则说明crate代码有问题",
                self.display_node(),
                parents[0]
            ))
        })?;

        // 2. 缓存父节点的值（用于反向传播）
        self.parent_value = Some(parent_value.clone());

        // 3. 计算 LeakyReLU: f(x) = x if x > 0, else negative_slope * x
        let slope = self.negative_slope as f32;
        let result = parent_value.where_with_f32(
            |x| x > 0.0,
            |x| x,         // x > 0 时保持原值
            |x| slope * x, // x <= 0 时乘以 slope
        );
        self.value = Some(result);

        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_jacobi_to_a_parent(
        &self,
        _target_parent: &NodeHandle,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // LeakyReLU 的导数: d(f(x))/dx = 1 if x > 0, else negative_slope
        // 由于是逐元素操作，雅可比矩阵是对角矩阵
        let parent_value = self.parent_value.as_ref().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}没有缓存的父节点值。不该触及本错误，否则说明crate代码有问题",
                self.display_node()
            ))
        })?;

        // 计算导数：x > 0 时为 1，否则为 negative_slope
        let slope = self.negative_slope as f32;
        let derivative = parent_value.where_with_f32(
            |x| x > 0.0,
            |_| 1.0,   // x > 0 时导数为 1
            |_| slope, // x <= 0 时导数为 slope
        );

        // 转换为 Jacobian 对角矩阵
        Ok(derivative.jacobi_diag())
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.cloned();
        Ok(())
    }

    // ========== Batch 模式 ==========

    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // LeakyReLU 的梯度: upstream_grad * (1 if x > 0 else negative_slope)
        let parent_value = self.parent_value.as_ref().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}没有缓存的父节点值，无法计算梯度",
                self.display_node()
            ))
        })?;

        // 计算局部梯度（逐元素）
        let slope = self.negative_slope as f32;
        let local_grad = parent_value.where_with_f32(
            |x| x > 0.0,
            |_| 1.0,   // x > 0 时导数为 1
            |_| slope, // x <= 0 时导数为 slope
        );

        // 逐元素乘以上游梯度
        Ok(upstream_grad * &local_grad)
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }
}
