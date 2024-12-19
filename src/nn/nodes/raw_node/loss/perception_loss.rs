use crate::nn::{Graph, GraphError, NodeId, TraitNode};
use crate::tensor::Tensor;
use crate::tensor_where;

pub struct PerceptionLoss {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    parents: Vec<NodeId>,
    children: Vec<NodeId>,
}

impl PerceptionLoss {
    pub fn new(parents: &[NodeId], name: Option<&str>) -> Self {
        // 1. 基本验证：PerceptionLoss只能有1个父节点
        assert!(parents.len() == 1, "PerceptionLoss节点只能有1个父节点");

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

impl TraitNode for PerceptionLoss {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_value(&mut self, parents: &[&NodeHandle]) -> Result<(), GraphError> {
        // 1. 从图中获取父节点的值
        let parent_value = graph.get_node_value(self.parents[0])?;

        // 2. 验证输入维度
        if parent_value.shape().len() != 2 {
            return Err(GraphError::ComputationError(format!(
                "PerceptionLoss节点的输入必须是2阶张量, 但得到的是{}阶张量",
                parent_value.shape().len()
            )));
        }

        // 3. 计算感知损失：x >= 0 时为0，否则为-x
        self.value = Some(tensor_where!(parent_value >= 0.0, 0.0, -parent_value));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_jacobi_to_a_parent(&self, parent: &NodeHandle) -> Result<Tensor, GraphError> {
        // 1. 计算对角线元素：x >= 0 时为0，否则为-1
        let parent_value = graph.get_node_value(parent_id)?;
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
