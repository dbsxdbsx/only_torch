/*
 * ZerosLike 节点：根据父节点的 batch_size 动态生成零张量
 *
 * 用于 RNN/LSTM/GRU 的初始隐藏状态：
 * - 形状 [?, hidden_size]，batch 维度与参考节点相同
 * - 每次 forward 时自动根据输入的 batch_size 生成正确形状的零张量
 *
 * 这解决了 ModelState 缓存命中时无法更新初始状态的问题。
 */

use crate::nn::shape::DynamicShape;
use crate::nn::{GraphError, NodeId};
use crate::tensor::Tensor;

use crate::nn::nodes::node_handle::NodeHandle;
use crate::nn::nodes::raw_node::TraitNode;

/// `ZerosLike` 节点：根据参考节点的 `batch_size` 生成零张量
#[derive(Clone)]
pub(crate) struct ZerosLike {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    /// 输出特征维度（不包括 batch）
    feature_shape: Vec<usize>,
    /// 动态形状 [?, `feature_dims`...]
    dynamic_shape: DynamicShape,
    /// 固定形状（首次创建时使用 batch=1）
    fixed_shape: Vec<usize>,
}

impl ZerosLike {
    /// 创建 `ZerosLike` 节点
    ///
    /// # 参数
    /// - `feature_shape`: 输出的特征维度（不包括 batch）
    ///
    /// # 示例
    /// ```ignore
    /// // 创建一个输出 [?, hidden_size] 形状的零张量节点
    /// let h0 = ZerosLike::new(&[hidden_size]);
    /// ```
    pub(crate) fn new(feature_shape: &[usize]) -> Self {
        let dynamic_shape = DynamicShape::with_dynamic_batch(feature_shape);
        let mut fixed_shape = vec![1]; // 使用 batch=1 作为占位符
        fixed_shape.extend_from_slice(feature_shape);

        Self {
            id: None,
            name: None,
            value: None,
            feature_shape: feature_shape.to_vec(),
            dynamic_shape,
            fixed_shape,
        }
    }
}

impl TraitNode for ZerosLike {
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

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 从第一个父节点获取 batch_size
        let ref_value = parents.first().and_then(|p| p.value()).ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{} 需要参考节点有值以确定 batch_size",
                self.display_node()
            ))
        })?;

        let batch_size = ref_value.shape()[0];

        // 生成零张量 [batch_size, feature_dims...]
        let mut shape = vec![batch_size];
        shape.extend_from_slice(&self.feature_shape);
        self.value = Some(Tensor::zeros(&shape));

        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        self.value = value.cloned();
        Ok(())
    }

    fn grad(&self) -> Option<&Tensor> {
        // ZerosLike 是常量，不参与梯度计算
        None
    }

    fn set_grad(&mut self, _grad: Option<&Tensor>) -> Result<(), GraphError> {
        // 忽略梯度设置
        Ok(())
    }

    fn clear_grad(&mut self) -> Result<(), GraphError> {
        Ok(())
    }

    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        _upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // ZerosLike 的梯度对父节点是零（因为它是常量）
        // 但实际上这个方法不应该被调用，因为 ZerosLike 的输出不依赖于父节点的值
        Err(GraphError::InvalidOperation(format!(
            "{} 不支持梯度计算（它是常量节点）",
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
