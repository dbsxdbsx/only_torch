use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::NodeHandle;
use crate::nn::GraphError;
use crate::tensor::Tensor;
use crate::tensor_where;

#[derive(Clone)]
pub struct PerceptionLoss {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    trainable: bool,
    shape: Vec<usize>,
}

impl PerceptionLoss {
    pub fn new(parents: &[&NodeHandle], trainable: bool, name: &str) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "PerceptionLoss节点只需要1个父节点".to_string(),
            ));
        }

        // 2. 返回
        Ok(Self {
            name: name.to_string(),
            value: None,
            jacobi: None,
            trainable,
            shape: parents[0].value_expected_shape().to_vec(),
        })
    }
}

impl TraitNode for PerceptionLoss {
    fn name(&self) -> &str {
        &self.name
    }

    fn value_expected_shape(&self) -> &[usize] {
        &self.shape
    }

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
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

    fn calc_jacobi_to_a_parent(
        &self,
        target_parent: &NodeHandle,
        _another_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 1. 计算对角线元素：x >= 0 时为0，否则为-1
        let parent_value = target_parent
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
        self.trainable
    }

    fn set_trainable(&mut self, trainable: bool) -> Result<(), GraphError> {
        self.trainable = trainable;
        Ok(())
    }
}
