use crate::nn::{Graph, GraphError, NodeHandle, NodeId, TraitNode};
use crate::tensor::Tensor;

pub struct Add {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    parents: Vec<NodeId>,
    children: Vec<NodeId>,
}

impl Add {
    pub fn new(parents: &[NodeId], name: Option<&str>) -> Self {
        // 1. 基本验证：加法至少需要2个父节点
        assert!(parents.len() >= 2, "Add节点至少需2个父节点");

        // 2. 返回实例
        Self {
            name: name.unwrap_or_default().to_string(),
            value: None,
            jacobi: None,
            parents: parents.to_vec(),
            children: Vec::new(),
        }
    }
}

impl TraitNode for Add {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_value(&mut self, parents: &[&NodeHandle]) -> Result<(), GraphError> {
        // 1. 从图中获取父节点的值
        let mut result = None;
        for parent_id in &self.parents {
            let parent_value = graph.get_node_value(*parent_id)?;

            // 1.1 验证父节点值的维度
            if parent_value.shape().len() != 2 {
                return Err(GraphError::ComputationError(format!(
                    "Add节点的输入必须是2阶张量, 但得到的是{}阶张量",
                    parent_value.shape().len()
                )));
            }

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
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_jacobi_to_a_parent(&self, parent: &NodeHandle) -> Result<Tensor, GraphError> {
        // 1. 首先验证输入的节点是否为父节点
        if !self.parents.contains(&parent_id) {
            return Err(GraphError::InvalidOperation(
                "输入的节点不是该Add节点的父节点",
            ));
        }

        // 2. 对于加法运算，雅可比矩阵是单位矩阵
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

    fn parents_ids(&self) -> &[NodeId] {
        &self.parents
    }

    fn children_ids(&self) -> &[NodeId] {
        &self.children
    }

    fn is_trainable(&self) -> bool {
        true
    }
}
