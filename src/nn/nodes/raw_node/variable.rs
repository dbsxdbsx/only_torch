use crate::nn::{GraphError, NodeHandle, NodeId};
use crate::tensor::Tensor;

use super::TraitNode;

pub(crate) struct Variable {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    trainable: bool,
}

impl Variable {
    pub(crate) fn new(shape: &[usize], init: bool, trainable: bool, name: &str) -> Self {
        // 1. 构造前必要的校验
        assert!(
            shape.len() == 2,
            "Variable节点必须是2阶张量, 但得到的形状却是`{:?}`",
            shape.len()
        );

        // 2. 根据条件设置value
        let value = if init {
            // 如果需要初始化,则使用正态分布初始化
            Some(Tensor::normal(0.0, 0.001, shape))
        } else {
            None
        };

        // 3. 返回
        Self {
            name: name.to_string(),
            value,
            jacobi: None,
            trainable,
        }
    }
}

impl TraitNode for Variable {
    fn name(&self) -> &str {
        &self.name
    }

    fn calc_value_by_parents(&mut self, _parents: &[&NodeHandle]) -> Result<(), GraphError> {
        // Variable节点不需要计算值
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        match value {
            Some(value) => {
                // 1. 校验形状
                if let Some(current_value) = &self.value {
                    if value.shape() != current_value.shape() {
                        return Err(GraphError::ShapeMismatch {
                            expected: current_value.shape().to_vec(),
                            got: value.shape().to_vec(),
                        });
                    }
                }

                // 2. 设置value
                self.value = Some(value.clone());
            }
            None => {
                // 清除值
                self.value = None;
            }
        }
        Ok(())
    }

    fn calc_jacobi_to_a_parent(&self, _parent: &NodeHandle) -> Result<Tensor, GraphError> {
        Err(GraphError::InvalidOperation("Variable节点没有父节点"))
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
}
