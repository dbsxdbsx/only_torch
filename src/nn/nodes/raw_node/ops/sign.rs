use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

#[derive(Clone)]
pub(crate) struct Sign {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    shape: Vec<usize>,
}

impl Sign {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Sign节点只需要1个父节点".to_string(),
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

impl TraitNode for Sign {
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

        // 2. 计算Sign函数值：正数→1, 负数→-1, 零→0
        self.value = Some(parent_value.sign());
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
        // Sign函数的导数在所有点都是0（与Step一样不可微）
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}没有值。不该触及本错误，否则说明crate代码有问题",
                self.display_node()
            ))
        })?;
        let size = value.size();

        // 创建一个size x size的方阵，所有元素都是0
        Ok(Tensor::zeros(&[size, size]))
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.cloned();
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

