use crate::nn::GraphError;
use crate::tensor::Tensor;

use super::{NodeHandle, TraitNode};

#[derive(Clone)]
pub(crate) struct Variable {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    trainable: bool,
    shape: Vec<usize>,
}

impl Variable {
    pub(crate) fn new(
        shape: &[usize],
        init: bool,
        trainable: bool,
        name: &str,
    ) -> Result<Self, GraphError> {
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

        let mut var = Self {
            name: name.to_string(),
            value: None,
            jacobi: None,
            trainable,
            shape: shape.to_vec(),
        };
        if init {
            // 若需要初始化，则使用正态分布初始化
            var.value = Some(Tensor::normal(0.0, 0.001, shape));
        }

        // 2. 返回
        Ok(var)
    }
}

impl TraitNode for Variable {
    fn name(&self) -> &str {
        &self.name
    }

    fn calc_value_by_parents(&mut self, _parents: &[NodeHandle]) -> Result<(), GraphError> {
        Err(GraphError::InvalidOperation(
            "即使是用户不小心将Variable节点执行了图的前向传播，也不该触及本错误（否则说明crate代码有问题）".to_string(),
        ))
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
        _another_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        Err(GraphError::InvalidOperation(
            "Variable节点没有父节点".to_string(),
        ))
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.map(|j| j.clone());
        Ok(())
    }

    fn is_trainable(&self) -> bool {
        self.trainable
    }

    fn set_trainable(&mut self, trainable: bool) -> Result<(), GraphError> {
        self.trainable = trainable;
        Ok(())
    }

    fn value_expected_shape(&self) -> &[usize] {
        &self.shape
    }
}
