use crate::nn::{GraphError, NodeId};
use crate::tensor::Tensor;

use super::{NodeHandle, TraitNode};

#[derive(Clone)]
pub(crate) struct Input {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    // 注意：Input 节点没有 jacobi 字段，因为输入数据不参与梯度更新
    shape: Vec<usize>,
}

impl Input {
    pub(crate) fn new(shape: &[usize]) -> Result<Self, GraphError> {
        // 1. 必要的验证：支持 2D-4D 张量
        // - 2D: 标准全连接层 [batch, features] 或 [rows, cols]
        // - 3D: 单样本 CNN [C, H, W]
        // - 4D: Batch CNN [batch, C, H, W]
        if shape.len() < 2 || shape.len() > 4 {
            return Err(GraphError::DimensionMismatch {
                expected: 2, // 表示 2-4 维
                got: shape.len(),
                message: format!(
                    "节点张量必须是 2-4 维（支持 FC 和 CNN），但收到的维度是 {} 维。",
                    shape.len(),
                ),
            });
        }

        // 2. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            shape: shape.to_vec(),
        })
    }
}

impl TraitNode for Input {
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

    fn calc_value_by_parents(&mut self, _parents: &[NodeHandle]) -> Result<(), GraphError> {
        Err(GraphError::InvalidOperation(format!(
            "{}被执行了前向传播。不该触及本错误，否则说明crate代码有问题",
            self.display_node()
        )))
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        self.value = value.cloned();
        Ok(())
    }

    fn clear_value(&mut self) -> Result<(), GraphError> {
        self.value = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }

    fn value_expected_shape(&self) -> &[usize] {
        &self.shape
    }
}
