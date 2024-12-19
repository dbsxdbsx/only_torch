use crate::nn::{Graph, GraphError, NodeHandle, NodeId, TraitNode};
use crate::tensor::Tensor;
use crate::tensor_where;

pub struct Step {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    parents: Vec<NodeId>,
    children: Vec<NodeId>,
}

impl Step {
    pub fn new(parents: &[NodeId], name: Option<&str>) -> Self {
        // 1. 基本验证：Step算子只能有1个父节点
        assert!(parents.len() == 1, "Step节点只能有1个父节点");

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

impl TraitNode for Step {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_value(&mut self, parents: &[&NodeHandle]) -> Result<(), GraphError> {
        // 1. 从图中获取父节点的值
        let parent_value = graph.get_node_value(self.parents[0])?;

        // 2. 验证输入维度
        if parent_value.shape().len() != 2 {
            return Err(GraphError::ComputationError(format!(
                "Step节点的输入必须是2阶张量, 但得到的是{}阶张量",
                parent_value.shape().len()
            )));
        }

        // 3. 计算Step函数值
        self.value = Some(tensor_where!(parent_value >= 0.0, 1.0, 0.0));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_jacobi_to_a_parent(&self, parent: &NodeHandle) -> Result<Tensor, GraphError> {
        // Step函数的导数在所有点都是0
        // NOTE: 这里的实现的形状和MatrixSlow有些不同：https://github.com/zc911/MatrixSlow/blob/a6db0d38802004449941e6644e609a2455b26327/matrixslow/ops/ops.py#L351
        // 这里没有改变形状，而MatrixSlow会改变形状成向量
        let parent_value = graph.get_node_value(parent_id)?;
        Ok(Tensor::zeros(parent_value.shape()))
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
