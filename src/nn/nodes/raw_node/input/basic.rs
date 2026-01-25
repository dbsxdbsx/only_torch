/*
 * BasicInput 节点：基础输入节点
 *
 * 用于 Data 和 Target 两种变体，共用同一个结构体。
 * - Data: 用户手动创建的通用输入
 * - Target: Loss 的目标值（真实标签）
 *
 * # 动态 Batch 支持
 * BasicInput 支持动态 batch：第一维可以是任意值。
 * 这使得同一个计算图可以处理不同 batch_size 的输入。
 */

use crate::nn::nodes::NodeHandle;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::nn::{GraphError, NodeId};
use crate::tensor::Tensor;

/// 基础输入节点（Data 和 Target 共用）
///
/// # 动态 Batch 支持
/// `BasicInput` 支持动态 batch：第一维可以是任意值。
/// 这使得同一个计算图可以处理不同 `batch_size` 的输入。
#[derive(Clone)]
pub(crate) struct BasicInput {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    // 注意：Input 节点没有 grad 字段，因为输入数据不参与梯度更新
    /// 动态形状：第一维是 None（动态 batch）
    dynamic_shape: DynamicShape,
    /// 固定形状缓存（首次创建时的形状）
    fixed_shape: Vec<usize>,
}

impl BasicInput {
    pub(crate) fn new(shape: &[usize]) -> Result<Self, GraphError> {
        // 1. 必要的验证：支持 2D-4D 张量
        // - 2D: 标准全连接层 [batch, features] 或 [rows, cols]
        // - 3D: RNN 输入 [batch, seq_len, features]
        // - 4D: Batch CNN [batch, C, H, W]
        if shape.len() < 2 || shape.len() > 4 {
            return Err(GraphError::DimensionMismatch {
                expected: 2, // 表示 2-4 维
                got: shape.len(),
                message: format!(
                    "节点张量必须是 2-4 维（支持 FC、RNN 和 CNN），但收到的维度是 {} 维。",
                    shape.len(),
                ),
            });
        }

        // 创建动态形状：第一维是 None（动态 batch）
        let dynamic_shape = DynamicShape::with_dynamic_batch(&shape[1..]);

        // 2. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            dynamic_shape,
            fixed_shape: shape.to_vec(),
        })
    }

    /// 创建支持动态 batch 的 `BasicInput` 节点（指定特征形状）
    ///
    /// # 参数
    /// - `feature_shape`: 特征维度形状（不包括 batch）
    /// - `initial_batch`: 初始 `batch_size（用于固定形状`）
    #[allow(dead_code)]
    pub(crate) fn with_dynamic_batch(
        feature_shape: &[usize],
        initial_batch: usize,
    ) -> Result<Self, GraphError> {
        let ndim = feature_shape.len() + 1;
        if !(2..=4).contains(&ndim) {
            return Err(GraphError::DimensionMismatch {
                expected: 2,
                got: ndim,
                message: format!("节点张量必须是 2-4 维，但收到的维度是 {ndim} 维。",),
            });
        }

        let dynamic_shape = DynamicShape::with_dynamic_batch(feature_shape);
        let mut fixed_shape = vec![initial_batch];
        fixed_shape.extend_from_slice(feature_shape);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            dynamic_shape,
            fixed_shape,
        })
    }
}

impl TraitNode for BasicInput {
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
        &self.fixed_shape
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        self.dynamic_shape.clone()
    }

    fn supports_dynamic_batch(&self) -> bool {
        true
    }
}
