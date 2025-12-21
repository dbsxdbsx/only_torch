use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

/// Sigmoid 激活函数节点
///
/// forward: sigmoid(x) = 1 / (1 + e^(-x))
/// backward: d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
#[derive(Clone)]
pub(crate) struct Sigmoid {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    grad: Option<Tensor>, // Batch 模式的梯度
    shape: Vec<usize>,
}

impl Sigmoid {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Sigmoid节点只需要1个父节点".to_string(),
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

impl TraitNode for Sigmoid {
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

        // 2. 计算 sigmoid(x) = 1 / (1 + e^(-x))
        self.value = Some(parent_value.sigmoid());
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
        // sigmoid 的导数: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        // 由于是逐元素操作，雅可比矩阵是对角矩阵
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}没有值。不该触及本错误，否则说明crate代码有问题",
                self.display_node()
            ))
        })?;

        // 计算 sigmoid(x) * (1 - sigmoid(x))，并转换为 Jacobian 对角矩阵
        let one_minus_sigmoid = Tensor::ones(value.shape()) - value;
        let derivative = value * &one_minus_sigmoid;
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
        // Sigmoid 的梯度: upstream_grad * sigmoid(x) * (1 - sigmoid(x))
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}没有值，无法计算梯度",
                self.display_node()
            ))
        })?;

        // 计算 sigmoid(x) * (1 - sigmoid(x))（逐元素）
        let one_minus_sigmoid = Tensor::ones(value.shape()) - value;
        let local_grad = value * &one_minus_sigmoid;

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
