use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::{GraphError, NodeHandle};
use crate::tensor::Tensor;

pub(crate) struct Add {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
}

impl Add {
    pub(crate) fn new(name: &str) -> Self {
        // 1. 基本验证：加法至少需要2个父节点
        // assert!(parents.len() >= 2, "Add节点至少需2个父节点");
        Self {
            name: name.to_string(),
            value: None,
            jacobi: None,
        }
    }
}

impl TraitNode for Add {
    fn name(&self) -> &str {
        &self.name
    }

    fn calc_value_by_parents(&mut self, parents: &[&NodeHandle]) -> Result<(), GraphError> {
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

    fn calc_jacobi_to_a_parent(&self, _parent: &NodeHandle) -> Result<Tensor, GraphError> {
        // 对于加法运算，雅可比矩阵是单位矩阵
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
        true
    }
}
