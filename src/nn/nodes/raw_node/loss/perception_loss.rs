use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::GraphError;
use crate::tensor::Tensor;
use crate::tensor_where;

#[derive(Clone)]
pub(crate) struct PerceptionLoss {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    shape: Vec<usize>,
}

impl PerceptionLoss {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "PerceptionLoss节点只需要1个父节点".to_string(),
            ));
        }

        // 2. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            jacobi: None,
            shape: parents[0].value_expected_shape().to_vec(),
        })
    }
}

impl TraitNode for PerceptionLoss {
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
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 1. 计算对角线元素：x >= 0 时为0，否则为-1
        let parent_value = target_parent.value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的父节点{}没有值。不该触及本错误，否则说明crate代码有问题",
                self.display_node(),
                target_parent
            ))
        })?;
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
}
