use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;
use crate::tensor_where;

#[derive(Clone)]
pub(crate) struct PerceptionLoss {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
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

        // 2. 返回：输出为标量 [1, 1]（与 MSELoss/SoftmaxCE 一致）
        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            shape: vec![1, 1], // 标量输出
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

        // 2. 计算感知损失：loss = mean(max(0, -x))
        //    - x >= 0 时对应的元素损失为 0
        //    - x < 0 时对应的元素损失为 -x
        //    - 最终输出为标量 [1, 1]
        let element_loss = tensor_where!(parent_value >= 0.0, 0.0, -parent_value);
        let mean_loss = element_loss.mean();
        self.value = Some(Tensor::new(&[mean_loss], &[1, 1]));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    // ========== VJP 模式 ==========

    /// 计算 VJP 梯度
    ///
    /// `PerceptionLoss`（mean reduction）的梯度：
    /// - x >= 0 时为 0
    /// - x < 0 时为 -1/n（其中 n 为元素数）
    ///
    /// 上游梯度为标量 [1, 1]，输出梯度形状与父节点相同。
    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let parent_value = target_parent.value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的父节点{}没有值",
                self.display_node(),
                target_parent
            ))
        })?;

        // 元素数量
        let n = parent_value.size() as f32;

        // 局部梯度：x >= 0 时为 0，否则为 -1/n（mean reduction）
        let local_grad = tensor_where!(parent_value >= 0.0, 0.0, -1.0 / n);

        // 上游梯度为标量 [1, 1]，需要广播到父节点形状
        let upstream_scalar = upstream_grad.get_data_number().unwrap_or(1.0);

        // VJP: upstream_scalar * local_grad
        Ok(&local_grad * upstream_scalar)
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
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
