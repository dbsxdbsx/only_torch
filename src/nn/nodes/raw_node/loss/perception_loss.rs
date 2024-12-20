use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::{GraphError, NodeHandle};
use crate::tensor::Tensor;
use crate::tensor_where;

pub struct PerceptionLoss {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
}

impl PerceptionLoss {
    pub fn new(name: &str) -> Self {
        // 1. 基本验证：PerceptionLoss只能有1个父节点
        // assert!(parents.len() == 1, "PerceptionLoss节点只能有1个父节点");

        // 2. 返回实例

        Self {
            name: name.to_string(),
            value: None,
            jacobi: None,
        }
    }
}

impl TraitNode for PerceptionLoss {
    fn name(&self) -> &str {
        &self.name
    }

    fn calc_value_by_parents(&mut self, parents: &[&NodeHandle]) -> Result<(), GraphError> {
        // 1. 获取父节点的值
        let parent_value = parents[0]
            .value()
            .ok_or_else(|| GraphError::ComputationError("父节点没有值".to_string()))?;

        // 2. 计算感知损失：x >= 0 时为0，否则为-x
        self.value = Some(tensor_where!(parent_value >= 0.0, 0.0, -parent_value));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_jacobi_to_a_parent(&self, parent: &NodeHandle) -> Result<Tensor, GraphError> {
        // 1. 计算对角线元素：x >= 0 时为0，否则为-1
        let parent_value = parent
            .value()
            .ok_or_else(|| GraphError::ComputationError("父节点没有值".to_string()))?;
        let diag = tensor_where!(parent_value >= 0.0, 0.0, -1.0);

        // 2. 构造对角矩阵作为雅可比矩阵
        let flatten = diag.flatten();
        Ok(Tensor::diag(&flatten))
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.map(|j| j.clone());
        Ok(())
    }

    fn is_trainable(&self) -> bool {
        true
    }
}
