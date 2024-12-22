use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::NodeHandle;
use crate::nn::GraphError;
use crate::tensor::Tensor;

#[derive(Clone)]
pub(crate) struct Add {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    trainable: bool,
}

impl Add {
    pub(crate) fn new(name: &str, trainable: bool) -> Self {
        Self {
            name: name.to_string(),
            value: None,
            jacobi: None,
            trainable,
        }
    }
}

impl TraitNode for Add {
    fn name(&self) -> &str {
        &self.name
    }

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 1. 从图中获取父节点的值
        let mut result = None;
        for parent in parents {
            let parent_value = parent
                .value()
                .ok_or_else(|| GraphError::ComputationError("父节点没有值".to_string()))?;

            // 1.2 添加到结果中
            match &mut result {
                None => result = Some(parent_value.clone()),
                Some(sum) => {
                    // 1.3 验证形状兼容性
                    if sum.shape() != parent_value.shape() {
                        return Err(GraphError::ShapeMismatch {
                            expected: sum.shape().to_vec(),
                            got: parent_value.shape().to_vec(),
                            message: format!(
                                "Add节点 '{}' 的父节点 '{}' 的值与当前值形状不匹配。",
                                self.name(),
                                parent.name(),
                            ),
                        });
                    }
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
        _another_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // Add节点的雅可比矩阵是单位矩阵
        let size = self
            .value()
            .ok_or_else(|| GraphError::ComputationError("节点没有值".to_string()))?
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
