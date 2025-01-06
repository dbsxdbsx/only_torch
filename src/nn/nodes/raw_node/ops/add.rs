use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::GraphError;
use crate::tensor::Tensor;

#[derive(Clone)]
pub(crate) struct Add {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    trainable: bool,
    shape: Vec<usize>,
}

impl Add {
    pub(crate) fn new(parents: &[&NodeHandle], trainable: bool) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() < 2 {
            return Err(GraphError::InvalidOperation(
                "Add节点至少需要2个父节点".to_string(),
            ));
        }

        // 1.2 验证所有父节点形状相同
        let shape = parents[0].value_expected_shape().to_vec();
        for parent in parents.iter().skip(1) {
            if parent.value_expected_shape() != shape {
                return Err(GraphError::ShapeMismatch {
                    expected: shape.clone(),
                    got: parent.value_expected_shape().to_vec(),
                    message: format!("Add节点的所有父节点形状必须相同"),
                });
            }
        }

        // 2. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            jacobi: None,
            trainable,
            shape,
        })
    }
}

impl TraitNode for Add {
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
        // 1. 从图中获取父节点的值
        let mut result = None;
        for parent in parents {
            let parent_value = parent.value().ok_or_else(|| {
                GraphError::ComputationError(format!(
                    "{}的父节点{}没有值。不该触及本错误，否则说明crate代码有问题",
                    self.display_node(),
                    parent
                ))
            })?;

            // 1.2 添加到结果中
            match &mut result {
                None => result = Some(parent_value.clone()),
                Some(sum) => {
                    *sum += parent_value;
                }
            }
        }

        // 2. 将结果赋值给当前节点
        self.value = result;

        // 3. 返回
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
        // Add节点的雅可比矩阵是单位矩阵
        let size = self
            .value()
            .ok_or_else(|| {
                GraphError::ComputationError(format!(
                    "{}没有值。不该触及本错误，否则说明crate代码有问题",
                    self.display_node()
                ))
            })?
            .size();
        Ok(Tensor::eyes(size))
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
