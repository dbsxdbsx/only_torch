use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

/// Tanh激活函数节点
///
/// forward: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
/// backward: d(tanh)/dx = 1 - tanh²(x)
#[derive(Clone)]
pub(crate) struct Tanh {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    grad: Option<Tensor>, // Batch 模式的梯度
    shape: Vec<usize>,
}

impl Tanh {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Tanh节点只需要1个父节点".to_string(),
            ));
        }

        // 2. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            jacobi: None,
            grad: None,
            shape: parents[0].value_expected_shape().to_vec(),
        })
    }
}

impl TraitNode for Tanh {
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

        // 2. 计算tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        // 等价于 tanh(x) = 2 / (1 + e^(-2x)) - 1，这种形式数值更稳定
        self.value = Some(parent_value.tanh());
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
        // tanh 的导数: d(tanh(x))/dx = 1 - tanh²(x)
        // 由于是逐元素操作，雅可比矩阵是对角矩阵
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}没有值。不该触及本错误，否则说明crate代码有问题",
                self.display_node()
            ))
        })?;

        // 计算 1 - tanh²(x)，并转换为 Jacobian 对角矩阵
        let tanh_squared = value * value;
        let derivative = Tensor::ones(value.shape()) - tanh_squared;
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
        // Tanh 的梯度: upstream_grad * (1 - tanh²(x))
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}没有值，无法计算梯度",
                self.display_node()
            ))
        })?;

        // 计算 1 - tanh²(x)（逐元素）
        let tanh_squared = value * value;
        let local_grad = Tensor::ones(value.shape()) - tanh_squared;

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

    fn clear_value(&mut self) -> Result<(), GraphError> {
        self.value = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
