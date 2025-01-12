use crate::nn::{GraphError, NodeId};
use crate::tensor::Tensor;

use super::{NodeHandle, TraitNode};

#[derive(Clone)]
pub(crate) struct Parameter {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    shape: Vec<usize>,
}

impl Parameter {
    pub(crate) fn new(shape: &[usize]) -> Result<Self, GraphError> {
        // 1. 必要的验证
        if shape.len() != 2 {
            return Err(GraphError::DimensionMismatch {
                expected: 2,
                got: shape.len(),
                message: format!(
                    "神经网络中的节点张量必须是2维的（矩阵），但收到的维度是{}维。",
                    shape.len(),
                ),
            });
        }

        // 2. 返回
        Ok(Self {
            id: None,
            name: None,
            value: Some(Tensor::normal(0.0, 0.001, shape)),
            jacobi: None,
            shape: shape.to_vec(),
        })
    }
}

impl TraitNode for Parameter {
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

    fn calc_value_by_parents(&mut self, _parents: &[NodeHandle]) -> Result<(), GraphError> {
        Err(GraphError::InvalidOperation(format!(
            "{}被执行了前向传播。不该触及本错误，否则说明crate代码有问题",
            self.display_node()
        )))
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        self.value = value.map(|v| v.clone());
        Ok(())
    }

    fn calc_jacobi_to_a_parent(
        &self,
        _target_parent: &NodeHandle,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        Err(GraphError::InvalidOperation(format!(
            "{}没有父节点。不该触及本错误，否则说明crate代码有问题",
            self.display_node()
        )))
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.map(|j| j.clone());
        Ok(())
    }

    fn value_expected_shape(&self) -> &[usize] {
        &self.shape
    }
}
