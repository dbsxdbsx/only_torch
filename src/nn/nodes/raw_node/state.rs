/*
 * State 节点：用于 RNN 中的时间状态（如隐藏状态 h、LSTM 的 c）
 *
 * 与 Input 节点的区别：
 *   - Input：用户数据输入，不接收梯度（叶子节点）
 *   - State：执行引擎管理的时间状态，接收并传递梯度（用于 BPTT）
 *
 * 与 Parameter 节点的区别：
 *   - Parameter：可训练参数，被优化器更新
 *   - State：时间状态，不被优化器更新，值由 step()/reset() 管理
 *
 * 语义：State 是"要记的东西"，不是"要学的东西"
 *
 * # 动态 Batch 支持
 * State 节点支持动态 batch：第一维可以是任意值。
 * 这使得同一个 RNN 结构可以处理不同 batch_size 的输入。
 */

use crate::nn::shape::DynamicShape;
use crate::nn::{GraphError, NodeId};
use crate::tensor::Tensor;

use super::{NodeHandle, TraitNode};

#[derive(Clone)]
pub(crate) struct State {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 动态形状：第一维是 None（动态 batch）
    dynamic_shape: DynamicShape,
    /// 固定形状缓存（首次创建时的形状）
    fixed_shape: Vec<usize>,
}

impl State {
    pub(crate) fn new(shape: &[usize]) -> Result<Self, GraphError> {
        // 支持 2D-4D 张量
        // - 2D: 标准 RNN 隐藏状态 [batch, hidden_size]
        // - 3D: 序列隐藏状态 [batch, seq_len, hidden_size]
        // - 4D: ConvLSTM 状态 [batch, C, H, W]
        if shape.len() < 2 || shape.len() > 4 {
            return Err(GraphError::DimensionMismatch {
                expected: 2,
                got: shape.len(),
                message: format!(
                    "State 张量必须是 2-4 维（支持 RNN/LSTM/ConvLSTM），但收到的维度是 {} 维。",
                    shape.len(),
                ),
            });
        }

        // 创建动态形状：第一维是 None（动态 batch）
        let dynamic_shape = DynamicShape::with_dynamic_batch(&shape[1..]);

        Ok(Self {
            id: None,
            name: None,
            value: None, // 初始值为 None，由 reset() 或用户设置
            grad: None,
            dynamic_shape,
            fixed_shape: shape.to_vec(),
        })
    }

    /// 从 `DynamicShape` 创建
    #[allow(dead_code)]
    pub(crate) fn with_dynamic_shape(
        dynamic_shape: DynamicShape,
        initial_fixed: &[usize],
    ) -> Result<Self, GraphError> {
        if initial_fixed.len() < 2 || initial_fixed.len() > 4 {
            return Err(GraphError::DimensionMismatch {
                expected: 2,
                got: initial_fixed.len(),
                message: format!(
                    "State 张量必须是 2-4 维，但收到的维度是 {} 维。",
                    initial_fixed.len(),
                ),
            });
        }

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            dynamic_shape,
            fixed_shape: initial_fixed.to_vec(),
        })
    }
}

impl TraitNode for State {
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
        // State 节点的值由执行引擎（step/reset）管理，不通过父节点计算
        Err(GraphError::InvalidOperation(format!(
            "{}的值由执行引擎管理，不通过前向传播计算。不该触及本错误，否则说明crate代码有问题",
            self.display_node()
        )))
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        // State 节点允许外部设置值（由执行引擎或用户初始化）
        self.value = value.cloned();
        Ok(())
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }

    fn clear_grad(&mut self) -> Result<(), GraphError> {
        self.grad = None;
        Ok(())
    }

    /// State 节点不计算 grad（它是叶子节点），但会接收来自 BPTT 的 grad
    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        _upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        Err(GraphError::InvalidOperation(format!(
            "{}是叶子节点，没有父节点来计算 grad",
            self.display_node()
        )))
    }

    fn clear_value(&mut self) -> Result<(), GraphError> {
        self.value = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }

    fn value_expected_shape(&self) -> &[usize] {
        &self.fixed_shape
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        self.dynamic_shape.clone()
    }

    fn supports_dynamic_batch(&self) -> bool {
        true
    }
}
