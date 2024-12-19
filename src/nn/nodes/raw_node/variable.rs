use crate::nn::{Graph, GraphError, NodeHandle, NodeId, TraitNode};
use crate::tensor::Tensor;

pub struct Variable {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    children: Vec<NodeId>,
    trainable: bool,
}

impl Variable {
    pub fn new(shape: &[usize], init: bool, trainable: bool, name: Option<&str>) -> Self {
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
            name: name.unwrap_or_default().to_string(),
            value,
            jacobi: None,
            children: Vec::new(),
            trainable,
        }
    }
}

impl TraitNode for Variable {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_value(&mut self, _parents: &[NodeHandle]) -> Result<(), GraphError> {
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

    fn parents_ids(&self) -> &[NodeId] {
        &[]
    }

    fn children_ids(&self) -> &[NodeId] {
        &self.children
    }

    fn is_trainable(&self) -> bool {
        self.trainable
    }
}
