use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::{GraphError, NodeHandle};
use crate::tensor::Tensor;
use crate::tensor_where;

pub(crate) struct Step {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
}

impl Step {
    pub(crate) fn new(name: &str) -> Self {
        // 1. 基本验证：阶跃函数需要1个父节点
        // assert!(parents.len() == 1, "Step节点需要1个父节点");

        // 2. 返回实例
        Self {
            name: name.to_string(),
            value: None,
            jacobi: None,
        }
    }
}

impl TraitNode for Step {
    fn name(&self) -> &str {
        &self.name
    }

    fn calc_value_by_parents(&mut self, parents: &[&NodeHandle]) -> Result<(), GraphError> {
        // 1. 获取父节点的值
        let parent_value = parents[0]
            .value()
            .ok_or_else(|| GraphError::ComputationError("父节点没有值".to_string()))?;

        // 2. 计算Step函数值
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
        let parent_value = parent
            .value()
            .ok_or_else(|| GraphError::ComputationError("父节点没有值".to_string()))?;
        Ok(Tensor::zeros(parent_value.shape()))
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
